# Oyta-AI 底层系统 · 完整技术说明（中文版）

---

# 目录

- [一、核心技术架构](#一核心技术架构)
- [二、Tensor 数据结构与布局](#二tensor-数据结构与布局)
- [三、Backend trait 完整定义与 82 算子签名](#三backend-trait-完整定义与-82-算子签名)
- [四、四设备后端实现原理](#四四设备后端实现原理)
- [五、算子引擎与动态 dispatch](#五算子引擎与动态-dispatch)
- [六、FlashAttention 在线 Softmax + 算子融合](#六flashattention-在线-softmax--算子融合)
- [七、DeviceContext 运行时](#七devicecontext-运行时)
- [八、完整 API 参考](#八完整-api-参考)
- [九、使用示例](#九使用示例)
- [十、数值稳定性与安全保证](#十数值稳定性与安全保证)

---

# 一、核心技术架构

## 1.1 分层与依赖方向

```
┌─────────────────────────────────────────────────────┐
│  oyta-tensor  ·  公共 API 入口（re-export 所有模块）  │
└────────────────────────┬────────────────────────────┘
                         │ 依赖
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌──────────────────────┐   ┌──────────────────────┐
│  oyta-core            │   │  oyta-ops             │
│  Tensor + Shape +     │   │  dispatch + fuse +    │
│  Strides + Index      │   │  matmul + flash +     │
│  safetensors + dtype  │   │  reduce + elwise      │
└──────────┬───────────┘   └──────────┬───────────┘
           │                           │
           └──────────┬────────────────┘
                      ▼
            ┌──────────────────────┐
            │  oyta-device          │
            │  Backend trait +      │
            │  Cpu/Cuda/Metal/Wgpu  │
            └──────────────────────┘
```

- **oyta-device**（最底层）：Backend trait + 四设备实现。依赖 rayon + thiserror + 可选 wgpu/blas-sys/metal。
- **oyta-core**（数据核心）：Tensor\<T,D\> + DynamicShape + Strides + IndexEngine + safetensors + f16/bf16。依赖 oyta-device 的 Device trait。
- **oyta-ops**（算子引擎）：动态 dispatch + 融合决策器 + matmul 策略 + FlashAttention + reduce/elwise 实现。同时依赖 oyta-core 和 oyta-device。
- **oyta-tensor**（公共入口）：显式 re-export 所有模块，`use oyta_tensor::prelude::*` 一行导入。

## 1.2 Feature 自动传递链

每个上游 crate 的 feature 自动传递到下游：

```
oyta-tensor --features cuda
    → oyta-ops/cuda 激活
        → oyta-core/cuda 激活
            → oyta-device/cuda 激活 + nvcc 编译 5 个 .cu kernels
```

## 1.3 Release 编译优化

```toml
[profile.release]
lto = "fat"          # 跨 crate 全量内联
codegen-units = 1    # 单编译单元 → 编译器跨模块最大优化
strip = "symbols"    # 剥除符号表
opt-level = 3        # 最高优化等级
panic = "abort"      # 直接 abort → 零 unwind 开销
```

## 1.4 Cargo 命令

```bash
# 基础
cargo check --workspace
cargo test --workspace                                    # 9 个单元测试
cargo build --release --features cuda,wgpu,blas

# CUDA（需要 PATH=/usr/local/cuda/bin）
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
cargo build -p oyta-device --features cuda               # nvcc 13.5 真实编译 5 个 .cu

# 全 feature 矩阵
cargo check --all-features                                # cpu + cuda + wgpu + blas + metal + memory_pool + debug_assertions 一次性
cargo check --no-default-features --features wgpu
cargo check --no-default-features --features metal

# 示例
cargo run --example example_11_wgpu_verify --features wgpu
```

---

# 二、Tensor 数据结构与布局

## 2.1 核心结构定义

```rust
// oyta-core/src/tensor/base.rs

pub struct Tensor<T, D: Backend> {
    /// 共享底层数据（Arc 原子操作 5-10ns < L1 miss ~100ns）
    pub data: Arc<Vec<T>>,
    /// 逻辑形状
    pub shape: DynamicShape,
    /// 步长向量（支持非 contiguous view）
    pub strides: Strides,
    /// flat 数组起始偏移（切片 view 用）
    pub offset: usize,
    /// 设备标记（零开销泛型 PhantomData）
    pub _device: PhantomData<D>,
}
```

64-bit 系统结构体大小约 72 字节。

## 2.2 零拷贝 view 机制

```
原 Tensor: data=[a,b,c,d,e,f,g,h]  shape=[2,4] strides=[4,1]

transpose(0,1) → 零拷贝 view：
  data 不变 → Arc 多一次引用
  strides 从 [4,1] 变成 [1,4]
  offset 不变

逻辑访问路径变化：
  t[0,0] → data_ptr + 0×1 + 0×4 = data_ptr + 0
  t[0,1] → data_ptr + 0×1 + 1×4 = data_ptr + 4
  t[1,0] → data_ptr + 1×1 + 0×4 = data_ptr + 1
  t[1,1] → data_ptr + 1×1 + 1×4 = data_ptr + 5

Clone 只多一次 Arc 原子加，不拷贝数据
```

## 2.3 DynamicShape

```rust
pub struct DynamicShape { pub dims: Vec<usize> }

impl DynamicShape {
    pub fn new(dims: impl IntoIterator<Item = usize>) -> Self
    pub fn numel(&self) -> usize
    pub fn rank(&self) -> usize
    pub fn dim(&self, i: isize) -> usize     // 负索引：-1 = 最右维
    pub fn reshape(&self, new_dims: &[isize]) -> Result<DynamicShape>
    pub fn broadcast_compatible(&self, other: &DynamicShape) -> bool
    pub fn broadcast_with(&self, other: &DynamicShape) -> DynamicShape
}
```

`numel` 用 `checked_product` 防溢出。`reshape` 支持 `[2,3,4]` 重写成 `[2,-1]` → 自动推断 `-1 = 12`。

## 2.4 Strides

```rust
pub struct Strides { dims: Vec<usize> }

impl Strides {
    pub fn contiguous(shape: &DynamicShape) -> Self    // C-contiguous strides
    pub fn is_contiguous(&self, shape: &DynamicShape) -> bool  // 最后一维步长 = 1
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Self
    pub fn permute(&self, order: &[usize]) -> Self
}
```

## 2.5 IndexEngine

```rust
// oyta-core/src/index.rs
pub struct IndexEngine { shape: DynamicShape, strides: Strides }

impl IndexEngine {
    pub fn offset(&self, multi_idx: &[usize]) -> usize   // 多维→线性偏移
    pub fn reverse_offset(&self, flat: usize) -> Vec<usize>  // 反向转换
}
```

非 contiguous + 广播通用逐元素算子（elwise_generic）的核心引擎。

## 2.6 TensorError（thiserror 10 变体）

```rust
pub enum TensorError {
    ShapeMismatch { expected, actual },
    MatMulDimMismatch { m, k1, k2, n },
    BroadcastFailed { shape_a, shape_b },
    IndexOutOfBounds { shape, index },
    DataSizeMismatch { need, have },
    ReshapeImpossible { old, new },
    DeviceError { msg: String },
    InvalidDtype { dtype: String },
    IoError { path: String, source: std::io::Error },
    SerdeError { msg: String },
}
```

---

# 三、Backend trait 完整定义与 82 算子签名

## 3.1 Backend 聚合 trait

```rust
// oyta-device/src/backend/aggregate.rs
pub trait Backend:
    BackendBase + ElwiseOpsF32 + ElwiseOpsF16 + ElwiseOpsBF16
    + ReduceOpsF32 + MatmulOps + FlashAttentionOps
    + FusedOps + InPlaceOpsF32
    + Copy + Clone + Send + Sync + 'static
{
    // 编译期设备属性（关联常量，零运行时开销）
    const IS_CPU: bool = false;
    const MAX_ACCUMULATOR_PRECISION: Precision = Precision::F32;
    const HAS_FP16_TENSOR_CORE: bool = false;
    const HAS_BF16: bool = false;
    const HAS_BATCHED_GEMM: bool = false;
    const HAS_FLASH_ATTENTION_HW: bool = false;
    const DEVICE_NAME: &'static str = "Unknown";

    fn should_fuse(op1: OpKind, op2: OpKind, intermediate_elements: usize) -> bool
        { !Self::IS_CPU || intermediate_elements >= 1024 }
}
```

## 3.2 BackendBase（3 方法）

```rust
pub trait BackendBase {
    /// 用值填充 C-contiguous 数组
    fn fill(ptr: *mut f32, n: usize, value: f32);
    /// 拷贝 C-contiguous 数组
    fn copy(src: *const f32, dst: *mut f32, n: usize);
    /// 按布尔掩码条件拷贝（where_bool）
    fn where_bool(cond: *const bool, a: *const f32, b: *const f32, out: *mut f32, n: usize);
}
```

## 3.3 ElwiseOpsF32（23 方法）

| 方法 | 签名 | 类型 |
|---|---|---|
| `elwise_add` | `(a, b, c, n)` | 二元 |
| `elwise_sub` | `(a, b, c, n)` | 二元 |
| `elwise_mul` | `(a, b, c, n)` | 二元 |
| `elwise_div` | `(a, b, c, n)` | 二元 |
| `elwise_pow` | `(a, exp: f32, c, n)` | 二元带标量 |
| `elwise_add_broadcast` | `(a, b, c, n_a, n_b, n_out)` | 广播二元 |
| `elwise_neg` | `(a, c, n)` | 一元 |
| `elwise_abs` | `(a, c, n)` | 一元 |
| `elwise_exp` | `(a, c, n)` | 一元 |
| `elwise_log` | `(a, c, n)` | 一元 |
| `elwise_sqrt` | `(a, c, n)` | 一元 |
| `elwise_sin` | `(a, c, n)` | 一元 |
| `elwise_cos` | `(a, c, n)` | 一元 |
| `elwise_relu` | `(a, c, n)` | 激活 |
| `elwise_gelu` | `(a, c, n)` | 激活 |
| `elwise_gelu_fast` | `(a, c, n)` | 激活 |
| `elwise_sigmoid` | `(a, c, n)` | 激活 |
| `elwise_tanh` | `(a, c, n)` | 激活 |
| `elwise_silu` | `(a, c, n)` | 激活 |

ElwiseOpsF16 / ElwiseOpsBF16 同样 20 个方法（不含广播版本）。

## 3.4 ReduceOpsF32（7 方法）

```rust
pub trait ReduceOpsF32 {
    /// 对每一行做规约（outer 行 × inner 元素）
    fn reduce_sum(a: *const f32, out: *mut f32, outer: usize, inner: usize);
    fn reduce_mean(a: *const f32, out: *mut f32, outer: usize, inner: usize);
    fn reduce_max(a: *const f32, out: *mut f32, outer: usize, inner: usize);
    fn reduce_min(a: *const f32, out: *mut f32, outer: usize, inner: usize);
    /// 数值稳定 softmax：先减 max + f64 累加器
    fn softmax(a: *const f32, out: *mut f32, outer: usize, inner: usize);
    /// LayerNorm：f64 统计 + f32 归一化 + eps
    fn layernorm(a: *const f32, out: *mut f32, gamma: *const f32, beta: *const f32,
                 outer: usize, inner: usize, eps: f32);
    /// RMSNorm：f64 RMS + eps
    fn rmsnorm(a: *const f32, out: *mut f32, gamma: *const f32,
               outer: usize, inner: usize, eps: f32);
    /// 原地 LayerNorm（节省内存分配）
    fn layernorm_inplace(a: *mut f32, gamma: *const f32, beta: *const f32,
                         outer: usize, inner: usize, eps: f32);
}
```

## 3.5 MatmulOps（8 方法）

```rust
pub trait MatmulOps {
    /// C[m,n] = α·A[m,k]·B[k,n] + β·C[m,n]（行主序）
    fn matmul(a: *const f32, lda: usize, b: *const f32, ldb: usize,
              c: *mut f32, ldc: usize, m: usize, n: usize, k: usize,
              alpha: f32, beta: f32);

    /// 带转置的 matmul
    fn matmul_t(a: *const f32, lda: usize, transpose_a: bool,
                b: *const f32, ldb: usize, transpose_b: bool,
                c: *mut f32, ldc: usize, m: usize, n: usize, k: usize,
                alpha: f32, beta: f32);

    /// batched matmul（stride_a/stride_b/stride_c 控制每个 batch 间跳距）
    fn matmul_batched(a: *const f32, lda: usize, stride_a: usize,
                      b: *const f32, ldb: usize, stride_b: usize,
                      c: *mut f32, ldc: usize, stride_c: usize,
                      m: usize, n: usize, k: usize, batch: usize,
                      alpha: f32, beta: f32);

    /// f16 matmul：内部 cast f16→f32，算完 cast 回 f16
    fn matmul_f16(a: *const F16, lda: usize, b: *const F16, ldb: usize,
                  c: *mut F16, ldc: usize, m: usize, n: usize, k: usize,
                  alpha: f32, beta: f32);

    /// bf16 matmul
    fn matmul_bf16(a: *const BF16, lda: usize, b: *const BF16, ldb: usize,
                   c: *mut BF16, ldc: usize, m: usize, n: usize, k: usize,
                   alpha: f32, beta: f32);
}
```

**两套独立的 matmul 实现必须区分**：

### 体系 A：Tensor::matmul 真实调用链（被实际调用）

```
Tensor::<f32, Cpu>::matmul()                          // oyta-core/src/tensor/ops.rs L65
  → D::matmul()                                       // MatmulOps::matmul trait 方法
    → <Self as MatmulOps>::default_matmul()            // 默认实现（backend/matmul.rs）
      → matmul_dispatch()                             // 内部私有 dispatch 函数
        → m,n,k 都 ≤ 32 (NAIVE_THRESHOLD) → matmul_naive
        → m,n,k 都 ≥ 512 (STRASSEN_CUTOFF, 公开常量) → matmul_strassen_recursive
        → 其他 → matmul_tiled_fma
```

### 体系 B：strategy.rs 完整实现（独立存在，未被调用，预留 API）

`oyta-ops/src/matmul/strategy.rs` 里有一套更完整的替代实现：
- **点积特化** `m==1 && n==1 → dot_product_flat`（最 FMA 友好）
- 阈值 32 / 1024 / 128（STRASSEN_ENTRY=1024 进入，STRASSEN_CROSSOVER=128 递归回退）
- IS_CPU 条件分支 + GPU backend 特化 dispatch
- 通过 `pub use strategy::*` 导出，但**没有代码调用**

**Cuda::IS_CPU 的影响**：`Cuda::IS_CPU = true`（保守值，cuBLAS 需 runtime probe），所以 strategy.rs 里 CUDA 大矩阵会走 `matmul_strassen` 而不是 `D::matmul`。但这不影响体系 A——体系 A 里 Cuda 有自己的 MatmulOps::matmul 实现（先 runtime detect，有硬件调 cuBLAS）。

**BLAS feature 的位置**：BLAS 在体系 A 的 MatmulOps 默认实现内部判断（BLAS 激活优先 cblas_sgemm），strategy.rs 里没有 BLAS 相关逻辑。

## 3.6 FlashAttentionOps（2 方法）

```rust
pub trait FlashAttentionOps {
    /// FlashAttention: out = softmax(QKᵀ / √d_k) · V，不物化 QKᵀ
    /// - 在线 f64 softmax（m/l/o 三个累积器分块更新）
    /// - is_causal=true 时 q_i 只看 K[0..=i]，不需要显式创建 mask 矩阵
    fn flash_attention(q: *const f32, k: *const f32, v: *const f32, out: *mut f32,
                       seq_q: usize, seq_k: usize, d_k: usize, d_v: usize,
                       scale: f32, is_causal: bool);

    /// batched multi-head FlashAttention
    fn flash_attention_batched(q: *const f32, k: *const f32, v: *const f32, out: *mut f32,
                               batch: usize, heads: usize,
                               seq_q: usize, seq_k: usize, d_k: usize, d_v: usize,
                               scale: f32, is_causal: bool);
}
```

## 3.7 FusedOps（4 方法）

```rust
pub enum FusedOp { NoFusion, LnGelu, LinearBiasGelu, MatmulActivation, RmsnormSwiglu }

pub trait FusedOps {
    /// LayerNorm + GELU 融合
    /// 典型场景 Transformer encoder：LayerNorm(h) → GELU(LayerNorm(h))
    fn layernorm_gelu(a: *const f32, out: *mut f32, gamma: *const f32, beta: *const f32,
                      outer: usize, inner: usize, eps: f32);

    /// Linear + Bias + GELU 融合
    /// 典型场景 MLP block：Linear(x) + Bias → GELU
    fn linear_bias_gelu(x: *const f32, w: *const f32, bias: *const f32, out: *mut f32,
                        m: usize, n: usize, k: usize);

    /// Matmul + Activation 融合（参数化激活类型）
    fn matmul_activation(a: *const f32, lda: usize, b: *const f32, ldb: usize,
                         c: *mut f32, ldc: usize,
                         m: usize, n: usize, k: usize, alpha: f32, beta: f32,
                         act: ActivationType);

    /// RMSNorm + SwiGLU 融合（LLaMA MLP 全路径）
    /// 原始公式: h = (silu(XW1) ⊙ XW2) · W3
    /// 融合: 先 RMSNorm X → 再 SwiGLU → 再线性 W3
    fn rmsnorm_swiglu(x: *const f32, w1: *const f32, w2: *const f32, w3: *const f32,
                      out: *mut f32, gamma: *const f32,
                      m: usize, n: usize, k: usize, eps: f32);
}

pub enum ActivationType { Relu, Gelu, GeluFast, Silu, Sigmoid, Tanh }
```

## 3.8 InPlaceOpsF32（5 方法）

```rust
pub trait InPlaceOpsF32 {
    fn elwise_add_inplace(a: *mut f32, b: *const f32, n: usize);
    fn elwise_mul_inplace(a: *mut f32, b: *const f32, n: usize);
    fn elwise_relu_inplace(a: *mut f32, n: usize);
    fn elwise_scalar_add_inplace(a: *mut f32, n: usize, value: f32);
    fn elwise_scalar_mul_inplace(a: *mut f32, n: usize, value: f32);
}
```

## 3.9 完整 Backend 算子计数

| 子 trait | f32 | f16 | bf16 | inplace | 合计 |
|---|---|---|---|---|---|
| BackendBase | 3 | - | - | - | 3 |
| ElwiseOps（f32 21 + f16/bf16 各 4） | 21 | 4 | 4 | 0 | 29 |
| ReduceOps | 8 | - | - | 1 | 9 |
| MatmulOps | 8 | - | - | - | 8 |
| FlashAttentionOps | 2 | - | - | - | 2 |
| FusedOps | 4 | - | - | - | 4 |
| InPlaceOps | - | - | - | 5 | 5 |
| **合计** | **44** | **16** | **16** | **6** | **82** |

---

# 四、四设备后端实现原理

## 4.1 Cpu（默认后端）

### matmul dispatch 决策树（体系 A · 真实被调用的路径）

```
Tensor::<f32, Cpu>::matmul()                        // oyta-core/src/tensor/ops.rs
  → D::matmul()                                     // MatmulOps::matmul trait 方法
    → <Cpu as MatmulOps>::default_matmul()          // backend/matmul.rs 默认 impl
      → matmul_dispatch()                           // 内部私有函数
         │
    ┌────┴────┐
    ▼         ▼
 m,n,k 都 ≤ 32?   m,n,k 都 ≥ 512 (STRASSEN_CUTOFF)?
    │ 是            │ 是
    ▼               ▼
 matmul_naive     matmul_strassen_recursive
 朴素三重循环        7 公式递归
                    （内部 <128 回退 tiled_fma）
    │ 否            │ 否
    └────┬───────────┘
         ▼
 matmul_tiled_fma(BM=BN=BK=128)
 rayon 外层并行 + L2 分块
```

**阈值常量**（backend/matmul.rs 公开导出）：

```rust
// oyta-device/src/backend/matmul.rs
const NAIVE_THRESHOLD: usize = 32;
pub const STRASSEN_CUTOFF: usize = 512;   // 公开常量，oyta-device::STRASSEN_CUTOFF
```

**strassen_recursive 内部阈值**：递归分块时如果子块任一维度 < 128 则回退到 `matmul_tiled_fma`。

### tiled_fma 详细参数

```rust
const MATMUL_BM: usize = 128;
const MATMUL_BN: usize = 128;
const MATMUL_BK: usize = 128;

// L2 分块设计：
// A_block = 128 × 128 × 4B = 64 KB
// B_block = 128 × 128 × 4B = 64 KB
// 合计 128 KB ≤ L2 256 KB

// rayon 把 (num_m_blocks × num_n_blocks) 全部分块到多核
// 每块内部串行：编译器自动 FMA + SIMD 向量化
rayon::prelude::IntoParallelRefIterator::par_iter()
    .for_each(|&(mi, ni)| { /* tiled_fma_serial */ });
```

### Rayon 并行 elwise

```rust
// 每个 Backend 方法内部统一模式
let chunk_size = n.div_ceil(rayon::current_num_threads());
(0..rayon::current_num_threads()).into_par_iter().for_each(|tid| {
    let start = tid * chunk_size;
    let end = (start + chunk_size).min(n);
    for i in start..end { unsafe { *out.add(i) = ... } }
});
```

### Reduce f64 累加

```rust
// 所有 ReduceOps 实现都显式用 f64
let mut acc: f64 = 0.0;
for i in 0..inner { acc += unsafe { *a.add(i) as f64 }; }
unsafe { *out.add(row) = (acc / inner as f64) as f32; }
```

## 4.2 Cuda（feature = cuda）

### runtime probe

```rust
// oyta-device/src/device/cuda.rs
pub struct Cuda;

// ======== 设计关键点：IS_CPU = true ========
// cuBLAS 必须 runtime probe（cudaGetDeviceCount）才能确认硬件存在。
// 但 Backend 关联常量是编译期确定的，无法 runtime 改变。
// 所以保守设 IS_CPU = true → strategy.rs 大矩阵走 Strassen（CPU 算法），
// 但 MatmulOps::matmul 的 CUDA 实现内部**自己还有一层 runtime dispatch**：
//   has_cuda_hardware() → cublasSgemm（有硬件）
//   !has_cuda_hardware() → <Self as MatmulOps>::default_matmul()（CPU fallback）
// 这是设计冗余但正确的路径——最终 cuBLAS 还是会被调用。
impl Backend for Cuda {
    const IS_CPU: bool = true;           // ← 关键！保守值
    const MAX_ACCUMULATOR_PRECISION: Precision = Precision::F32;
    const HAS_FP16_TENSOR_CORE: bool = false;  // 需 runtime detect
    const HAS_BF16: bool = false;
    const HAS_BATCHED_GEMM: bool = true;
    const HAS_FLASH_ATTENTION_HW: bool = false;
    const DEVICE_NAME: &'static str = "NVIDIA CUDA";
}

impl DeviceContext for Cuda {
    fn init() -> Result<DeviceHandle<Cuda>, DeviceError> {
        // 1. cudaGetDeviceCount → 有硬件才走 GPU 路径
        let mut count: i32 = 0;
        unsafe { cudaGetDeviceCount(&mut count as *mut i32) };
        // 2. cudaSetDevice(0) + cublasCreate + cublasSetStream + cudaGetDeviceProperties
        // 3. 无硬件时：device_count=0 → handle=0 → matmul runtime dispatch fallback CPU
    }
}

// ======== Backend 子 trait 默认继承 ========
// 除 MatmulOps 外，其他子 trait 都空 impl → 继承 Cpu 的 rayon 默认实现
impl BackendBase for Cuda {}
impl ElwiseOpsF32 for Cuda {}
impl ReduceOpsF32 for Cuda {}
// MatmulOps 有真实 impl（见下方 cuBLAS 详情）
```

### cuBLAS matmul 内部逻辑（cuda.rs 源码）

```rust
impl MatmulOps for Cuda {
    fn matmul(a_ptr, lda, b_ptr, ldb, c_ptr, ldc, m, n, k, alpha, beta) {
        // 1. runtime detect：无硬件 → 直接 fallback CPU 默认实现
        if !gpu_ffi::has_cuda_hardware() {
            return <Self as MatmulOps>::default_matmul(...);  // 继承 rayon
        }
        let handle = get_cublas_handle();  // 从 DeviceHandle::Cuda.inner 取
        if handle == 0 { return <Self as MatmulOps>::default_matmul(...); }

        // 2. 检查 contiguous（lda==k, ldb==n, ldc==n）
        if lda == k && ldb == n && ldc == n {
            // 完美 contiguous：直接调 cublasSgemm
            cublasSgemm(handle, N, N, n, m, k, b_ptr, n, a_ptr, k, c_ptr, n);
        } else {
            // 有 stride：先把 a/b/c 各 copy 成 contiguous row-major Vec
            // 再调 cublasSgemm，最后写回（处理 ldc != n 的情况）
            let a_contig = copy_contiguous(a_ptr, m, k, lda);
            let b_contig = copy_contiguous(b_ptr, k, n, ldb);
            let mut c_local = if beta != 0.0 { copy_contiguous(c_ptr, m, n, ldc) } else { zeros(m*n) };
            cublasSgemm(handle, N, N, n, m, k, b_contig, n, a_contig, k, c_local, n);
            write_back(c_ptr, m, n, ldc, &c_local);
        }
        cudaDeviceSynchronize();
    }
}
```

### cuBLAS row-major trick

```rust
// cuBLAS 原生列主序。行主序 C = A·B 等价于列主序 Cᵀ = Bᵀ · Aᵀ
// 因为转置不改变矩阵乘法结果：(AB)ᵀ = BᵀAᵀ

unsafe { cublasSgemm(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,   // 不转置任何矩阵（因为已经把 A,B 交换了）
    n, m, k,                     // 注意：m/n 交换！（列主序 Cᵀ 的 m 是 n，n 是 m）
    &alpha,
    b_ptr, ldb,                  // ！A、B 位置交换
    a_ptr, lda,
    &beta,
    c_ptr, ldc,
); }
```

### 5 个 .cu kernels（nvcc 13.5 + GCC 13 真实编译通过）

| 文件 | 核心参数 | 说明 |
|---|---|---|
| `cuda_kernels/elwise_add.cu` | BM=256 | 标准 1D grid/block |
| `cuda_kernels/softmax.cu` | BM=256 | 数值稳定（先减 max） |
| `cuda_kernels/layernorm.cu` | BM=256 | 融合版 |
| `cuda_kernels/matmul.cu` | BM=BN=BK=64, shared_memory=32KB | 64³ tile，刚好 fit 48KB 默认上限 |
| `cuda_kernels/flash_attn.cu` | d_k=64 默认 | online softmax + causal mask |

### build.rs 自动探测（真实源码）

```rust
fn main() {
    #[cfg(feature = "cuda")] {
        // 强制 nvcc 用 GCC 13（Ubuntu 26.04 默认 GCC 15 与 CUDA toolkit 兼容性差）
        unsafe {
            std::env::set_var("CXX", "g++-13");
            std::env::set_var("CC", "gcc-13");
        }

        // 探测 nvcc 是否存在
        let nvcc = Command::new("nvcc").arg("--version").output()
            .map(|o| o.status.success()).unwrap_or(false);

        if nvcc {
            cc::Build::new()
                .cuda(true)
                .include("cuda_kernels")
                .file("cuda_kernels/elwise_add.cu")
                .file("cuda_kernels/softmax.cu")
                .file("cuda_kernels/layernorm.cu")
                .file("cuda_kernels/matmul.cu")
                .file("cuda_kernels/flash_attn.cu")
                .flag("-O3")
                .flag("--use_fast_math")
                .flag("-std=c++17")
                .compile("oyta_device_cuda_kernels");

            println!("cargo:rerun-if-changed=cuda_kernels");
            println!("cargo:warning=CUDA kernels compiled with nvcc");
        } else {
            println!("cargo:warning=nvcc not found, CUDA kernels skipped (will use CPU fallback)");
        }
    }
}
```

关键行为：
- **CXX/CC 环境变量强制 g++-13 / gcc-13**（通过 cc crate 传给 nvcc `-ccbin`）
- `cuda_kernels/` 目录变化时自动重新编译
- nvcc 不在时静默跳过，不报错（运行时走 CPU fallback）

## 4.3 Wgpu（feature = wgpu）

### 初始化链

```rust
let instance = wgpu::Instance::default();
let adapter = instance.request_adapter(None).await
    .expect("WGPU adapter not found");
let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await
    .expect("WGPU device request failed");

// WSL2 DirectX GPU passthrough: /dev/dxg
// Adapter::get_info().name == "Microsoft ..."  → DirectX
```

### compute shader matmul

```wgsl
// oyta-device/src/wgsl/matmul.wgsl
// wgpu 30 推荐维度用 STORAGE buffer 而非 UNIFORM（跨平台更稳定）

@group(0) @binding(0) var<storage, read> a: array<f32>;     // [m, k] 行主序
@group(0) @binding(1) var<storage, read> b: array<f32>;     // [k, n]
@group(0) @binding(2) var<storage, read_write> c: array<f32>; // [m, n]
@group(0) @binding(3) var<storage, read> dims: vec4<u32>;   // x=m, y=n, z=k

const BLOCK_SIZE: u32 = 16;

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;        // WGS 约定：x=col, y=row
    let col = gid.x;
    let m = dims.x;
    let n = dims.y;
    let k = dims.z;

    if row >= m || col >= n { return; }

    var acc: f32 = 0.0;
    for (var p: u32 = 0u; p < k; p++) {
        acc += a[row * k + p] * b[p * n + col];
    }
    c[row * n + col] = acc;
}
```

**workgroup 16×16** = 256 invocations，WSL2 DX GPU 限制 ≤ 256，所以不超。

### dispatch 计算

```rust
let block = 16u32;
let work_x = (n + block - 1) / block;  // ceil_div
let work_y = (m + block - 1) / block;
queue.submit(Some(command_encoder.finish()));
```

### 数值验证

```
CPU vs WGPU matmul (16×16 / 32×32 / 64×64 / 128×128):
  全部 4/4 PASS, max_diff = 0.00e0（完全精确）
```

## 4.4 Metal（feature = metal）

- Apple 平台 MPS 接口预留
- 非 Apple 平台 feature 空激活（无依赖编译）
- DeviceContext OnceLock 单例实现完整

---

# 五、算子引擎与动态 dispatch

## 5.1 dispatch.rs（策略选择器 · 预留 API）

```rust
// oyta-ops/src/dispatch.rs
// 注意：select_matmul_strategy 只是一个策略枚举选择器，**从未被实际调用**
// 真实的 matmul 执行路径是 strategy.rs::matmul()（见 4.1 和 5.2）

pub enum MatmulStrategy { Naive, TiledFma, Strassen, GpuTensorCore }

pub fn select_matmul_strategy<D: Backend>(m: usize, k: usize, n: usize) -> MatmulStrategy {
    let max_dim = m.max(k).max(n);
    if max_dim < 32 { return MatmulStrategy::Naive; }
    if max_dim < 128 { return MatmulStrategy::TiledFma; }
    if D::HAS_FP16_TENSOR_CORE || D::HAS_BF16 || !D::IS_CPU {
        MatmulStrategy::GpuTensorCore
    } else {
        MatmulStrategy::Strassen
    }
}

pub enum ReduceStrategy { Rayon, Sequential }
pub fn select_reduce_strategy(numel: usize) -> ReduceStrategy {
    if numel < 1024 { ReduceStrategy::Sequential } else { ReduceStrategy::Rayon }
}
```

**关键事实**：`dispatch.rs::select_matmul_strategy` 是一个**简化策略选择器**，只导出不调用。真实执行路径是 `strategy.rs::matmul()`（§4.1 已详述），那个函数有完整的 5 条分支（含点积特化 dot_product_flat 和 1024/128 双阈值）。

## 5.2 Strassen 7 公式递归

```
M1 = (A11+A22)(B11+B22)
M2 = (A21+A22)B11
M3 = A11(B12-B22)
M4 = A22(B21-B11)
M5 = (A11+A12)B22
M6 = (A21-A11)(B11+B12)
M7 = (A12-A22)(B21+B22)

C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6
```

**两个阈值常量**（strategy.rs 源码定义）：
- `STRASSEN_ENTRY = 1024`：进入 Strassen 模式的门槛（m,n,k 都 ≥ 1024）
- `STRASSEN_CROSSOVER = 128`：递归内部的回退点（任何维度 < 128 时回退到 tiled_fma）

## 5.3 broadcast.rs 预处理

```rust
// 逐元素算子的广播逻辑：
// 若两个 Tensor shape 不相等，先 broadcast 成共同 shape
// 规则：从最右维开始检查 → 相等 / 一方为 1 / 缺失 → 取 max
// 例：[1, 256, 1024] + [1024] → broadcast 到 [1, 256, 1024]
```

## 5.4 profiling.rs

```rust
// 256 槽 AtomicU64 计数器（无锁）
pub struct ProfilingEngine { counters: [AtomicU64; 256] }

impl ProfilingEngine {
    pub fn profile<F: FnOnce()>(label: &str, f: F) -> Duration  // 自动记录耗时
    pub fn print_report()                                       // 打印 Top-K 热点算子
}
```

## 5.5 dropout.rs

```rust
// 训练/推理态切换
pub fn set_training_mode(enabled: bool)  // AtomicBool 全局
pub fn is_training() -> bool

// Dropout 掩码生成（bernoulli 采样 + binomial）
pub fn generate_dropout_mask(n: usize, rate: f32) -> Vec<f32>  // 保持率 = 1 - rate
```

---

# 六、FlashAttention 在线 Softmax + 算子融合

## 6.1 FlashAttention 推导

### 标准 Attention 瓶颈

```
Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V
QKᵀ = [seq_q, seq_k]    ← O(seq²) 中间张量
seq=4096 → 64 MB HBM 读写
```

### FlashAttention 核心思想

```
对 Q 的每一行 q_i，分块遍历 K/V：

初始化在线 softmax 三个累积器：
  m = -∞      ← 已见过最大值
  l = 0       ← 已见过 exp 和
  o = 0       ← 已见过加权和

for K_chunk, V_chunk in 分块:
    s = q_i · K_chunkᵀ                    ← 只算一小块点积（在 SRAM/L1）
    m_new = max(m, max(s))                 ← 新的全局 max
    correction = exp(m - m_new)            ← 旧值需要乘的修正因子
    l = l × correction + sum(exp(s - m_new))
    o = o × correction + sum(exp(s - m_new) · V_chunk)
    m = m_new

o /= l                                    ← 最终归一化
```

### f64 累加保证精度

```rust
let mut m_val: f64 = f64::NEG_INFINITY;
let mut l_val: f64 = 0.0;
let mut o_acc: Vec<f64> = vec![0.0f64; d_v];
// ... 在线更新 ...
o_row[d] = (o_acc[d] * inv_l) as f32;
```

### Causal Mask 无矩阵技巧

```rust
// 每一行 q_i 只看 K[0..=qi] 范围
// 不需要显式创建 seq_q × seq_k 的 mask 矩阵
// 又省了一个 O(seq²) 中间张量
let k_end = if is_causal { (qi + 1).min(seq_k) } else { seq_k };
for ki in 0..k_end { /* 正常处理 */ }
```

## 6.2 算子融合决策器

```rust
// oyta-ops/src/fuse.rs

pub fn should_fuse<D: Backend>(first: OpKind, second: OpKind, intermediate_numel: usize) -> FusedOp {
    // GPU 后端：总是融合（HBM 读写是瓶颈）
    if !D::IS_CPU { return decide_fused_op(first, second); }
    // CPU 后端：中间 Tensor ≥ 1024 元素才融合（小 Tensor 融合开销 > 收益）
    if intermediate_numel < 1024 { return FusedOp::NoFusion; }
    decide_fused_op(first, second)
}

fn decide_fused_op(first: OpKind, second: OpKind) -> FusedOp {
    match (first, second) {
        (OpKind::LayerNorm, OpKind::Activation(_)) => FusedOp::LnGelu,
        (OpKind::MatmulBias, OpKind::Activation(_)) => FusedOp::LinearBiasGelu,
        (OpKind::Matmul, OpKind::Activation(_))     => FusedOp::MatmulActivation,
        (OpKind::RMSNorm, OpKind::Swiglu)            => FusedOp::RmsnormSwiglu,
        _                                             => FusedOp::NoFusion,
    }
}
```

### OpKind 枚举

```rust
pub enum OpKind {
    // Elwise 二元
    ElwiseAdd, ElwiseSub, ElwiseMul, ElwiseDiv, ElwisePow,
    // Elwise 一元
    ElwiseRelu, ElwiseGelu, ElwiseGeluFast, ElwiseSilu, ElwiseSigmoid, ElwiseTanh,
    ElwiseNeg, ElwiseAbs, ElwiseExp, ElwiseLog, ElwiseSqrt, ElwiseSin, ElwiseCos,
    // Norm
    LayerNorm, RMSNorm, Softmax,
    // Matmul / Attention
    Matmul, MatmulBias, FlashAttention,
    // SwiGLU / 激活参数化
    Swiglu, Activation(ActivationType),
    // 形状变换
    Reshape, Contiguous, Transpose, Permute,
}
```

---

# 七、DeviceContext 运行时

## 7.1 设计

```rust
// oyta-device/src/backend/aggregate.rs

pub trait DeviceContext: Backend {
    fn init() -> Result<DeviceHandle<Self>, DeviceError>;
    fn get() -> DeviceHandle<Self>;       // OnceLock 自动 init
    fn shutdown() {}                       // 可选显式清理（GPU 资源）
    fn init_with_id(device_id: usize) -> Result<DeviceHandle<Self>>;
}

pub struct DeviceHandle<D: Backend> {
    pub properties: DeviceProperties,
    pub _device: PhantomData<D>,
    pub(crate) inner: DeviceInner,
}

pub struct DeviceProperties {
    pub name: String,                       // "CPU (3 cores)" / "NVIDIA H100"
    pub device_id: usize,
    pub memory_bytes: usize,
    pub compute_capability: String,          // "Host" / "SM_90" / "Apple M3"
    pub num_cores: usize,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_fp8: bool,
    pub supports_tensor_core: bool,
}

pub enum DeviceInner {
    Cpu,
    Cuda { cublas_handle: *mut cublasContext, stream: cudaStream_t },
    Metal { device_id: usize },
    Wgpu { device_id: usize },
}
```

## 7.2 OnceLock 全局单例

```rust
// 每个 DeviceContext 实现内部都有 OnceLock：
static HANDLE: OnceLock<DeviceHandle<Cpu>> = OnceLock::new();

impl DeviceContext for Cpu {
    fn get() -> DeviceHandle<Cpu> {
        HANDLE.get_or_init(|| {
            let num_cores = rayon::current_num_threads();
            DeviceHandle {
                properties: DeviceProperties {
                    name: format!("CPU ({} cores)", num_cores),
                    num_cores,
                    ..Default::default()
                },
                inner: DeviceInner::Cpu,
                _device: PhantomData,
            }
        }).clone()
    }
}
```

## 7.3 四设备 init() 内容

| 设备 | init() 操作 | OnceLock |
|---|---|---|
| Cpu | rayon::current_num_threads() 探测 + 填充 DeviceProperties | ✅ |
| Cuda | cudaGetDeviceCount → cudaSetDevice(0) → cublasCreate → cublasSetStream → cudaGetDeviceProperties | ✅（无硬件返回 DeviceError） |
| Metal | DeviceInner::Metal(device_id=0) | ✅ |
| Wgpu | Instance::default() → Adapter::request_device() | ✅ |

## 7.4 AnyBackend 运行时切换

```rust
// oyta-device/src/backend/any_backend.rs
static CURRENT_RUNTIME: AtomicU8 = AtomicU8::new(0);

pub enum DeviceRuntimeId { Cpu = 0, Cuda = 1, Metal = 2, Wgpu = 3 }
pub fn set_runtime_device(id: DeviceRuntimeId) { CURRENT_RUNTIME.store(id as u8, Relaxed); }
pub fn get_runtime_device() -> DeviceRuntimeId { ... }
```

---

# 八、完整 API 参考

## 8.1 一行导入

```rust
use oyta_tensor::prelude::*;
```

### prelude 导出清单

```rust
// 从 oyta-core re-export
pub use oyta_core::{
    Tensor, DynamicShape, Strides, IndexEngine,
    Device, TensorError, Result as TensorResult,
    save_safetensors, load_safetensors, SafetensorsHeader, TensorHeader,
    f32_to_bytes, bytes_to_f32, f64_to_bytes, bytes_to_f64,
    f16_to_bytes, bytes_to_f16, bf16_to_bytes, bytes_to_bf16,
    tensor_byte_size, write_bytes, read_bytes, AlignedAlloc,
};

// 从 oyta-device re-export
pub use oyta_device::{
    Backend, BackendBase,
    ElwiseOpsF32, ElwiseOpsF16, ElwiseOpsBF16,
    ReduceOpsF32, MatmulOps, FlashAttentionOps,
    FusedOps, InPlaceOpsF32,
    Cpu, DeviceContext, DeviceHandle, DeviceProperties, DeviceInner,
    AnyBackend, DeviceRuntimeId, set_runtime_device, get_runtime_device,
    OpKind, ActivationType, DeviceError,
    F16, BF16, DType, Float, Accumulate, Precision,
    LAYERNORM_EPS, RMSNORM_EPS, STRASSEN_CUTOFF,
};

// feature-gated
#[cfg(feature = "cuda")] pub use oyta_device::Cuda;
#[cfg(feature = "metal")] pub use oyta_device::Metal;
#[cfg(feature = "wgpu")]  pub use oyta_device::Wgpu;

// 所有算子
pub use oyta_ops::*;
```

## 8.2 Tensor\<T, D\> 方法全集

### 构造

| 方法 | 签名 | 说明 |
|---|---|---|
| `from_data` | `(Vec<T>, shape) -> Result<Self>` | 最常用构造 |
| `from_shared` | `(Arc<Vec<T>>, shape) -> Self` | 已有 Arc 数据，零拷贝创建 view |
| `zeros` | `(shape) -> Self` | 全零 |
| `ones` | `(shape) -> Self` | 全一 |
| `full` | `(shape, value) -> Self` | 填充值 |
| `randn` | `(shape) -> Self` | N(0,1) 标准正态 |
| `rand` | `(shape) -> Self` | Uniform[0,1) |
| `arange` | `(start, end, step) -> Self` | 等差数列 |
| `scalar` | `(value) -> Self` | 标量（rank=0） |

### 属性查询

| 方法 | 签名 | 说明 |
|---|---|---|
| `shape()` | `-> &DynamicShape` | 逻辑形状引用 |
| `strides()` | `-> &Strides` | 步长引用 |
| `numel()` | `-> usize` | 元素总数 |
| `rank()` | `-> usize` | 维度数 |
| `data_ptr()` | `-> *const T` | 原始数据指针（已加 offset） |
| `data_slice()` | `-> &[T]` | 切片引用 |
| `is_contiguous()` | `-> bool` | C-contiguous 检查 |
| `is_scalar()` | `-> bool` | rank=0 或 numel=1 |
| `to_vec()` | `-> Vec<T>` | 拷贝出 Vec |
| `item()` | `-> T` | 取标量值 |

### 形状变换（全部零拷贝）

| 方法 | 签名 | 说明 |
|---|---|---|
| `reshape` | `(&[isize]) -> Result<Self>` | 支持 -1 自动推断 |
| `transpose` | `(dim1: isize, dim2: isize) -> Self` | 交换两维，支持负索引 |
| `permute` | `(&[usize]) -> Self` | 重排所有维度 |
| `unsqueeze` | `(dim: isize) -> Self` | 插入 size=1 维度 |
| `squeeze` | `() -> Self` | 移除所有 size=1 维度 |
| `slice` | `(dim: isize, range: Range<usize>) -> Self` | 单维切片，零拷贝 |
| `multi_slice` | `(&[Range<usize>]) -> Self` | 多维切片 |
| `select` | `(dim: usize, indices: &[usize]) -> Self` | 按索引选择行/列 |

### 数据布局

| 方法 | 签名 | 说明 |
|---|---|---|
| `contiguous()` | `-> Self` | 非 contiguous 时真实拷贝为 C-contiguous |

### 跨设备

| 方法 | 签名 | 条件激活 |
|---|---|---|
| `to_device<D2>()` | `-> Tensor<T, D2>` | 始终可用 |
| `to_cpu()` | `-> Tensor<T, Cpu>` | 始终可用 |
| `to_cuda()` | `-> Tensor<T, Cuda>` | `#[cfg(feature = "cuda")]` |
| `to_metal()` | `-> Tensor<T, Metal>` | `#[cfg(feature = "metal")]` |
| `to_wgpu()` | `-> Tensor<T, Wgpu>` | `#[cfg(feature = "wgpu")]` |

### 精度转换

| 源类型 | 目标 | 方法 |
|---|---|---|
| `Tensor<f32, D>` | F16 | `.to_f16()` |
| `Tensor<f32, D>` | BF16 | `.to_bf16()` |
| `Tensor<f32, D>` | f64 | `.to_f64()` |
| `Tensor<F16, D>` | f32 | `.to_f32()` |
| `Tensor<BF16, D>` | f32 | `.to_f32()` |
| `Tensor<f64, D>` | f32 | `.to_f32()` |

### 逐元素运算（自动广播）

```rust
// 二元
add(&self, other: &Tensor<f32, D>) -> Result<Tensor<f32, D>>
sub(&self, other: &Tensor<f32, D>) -> Result<Tensor<f32, D>>
mul(&self, other: &Tensor<f32, D>) -> Result<Tensor<f32, D>>
div(&self, other: &Tensor<f32, D>) -> Result<Tensor<f32, D>>
pow(&self, exp: f32) -> Result<Tensor<f32, D>>

// 激活函数
relu(&self)    -> Result<Tensor<f32, D>>
gelu(&self)    -> Result<Tensor<f32, D>>
gelu_fast(&self) -> Result<Tensor<f32, D>>
sigmoid(&self) -> Result<Tensor<f32, D>>
tanh(&self)    -> Result<Tensor<f32, D>>
silu(&self)    -> Result<Tensor<f32, D>>

// 数学一元
neg(&self)     -> Result<Tensor<f32, D>>
abs(&self)     -> Result<Tensor<f32, D>>
exp(&self)     -> Result<Tensor<f32, D>>
log(&self)     -> Result<Tensor<f32, D>>
sqrt(&self)    -> Result<Tensor<f32, D>>
sin(&self)     -> Result<Tensor<f32, D>>
cos(&self)     -> Result<Tensor<f32, D>>
```

### 规约运算

```rust
sum(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor<f32, D>>
mean(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor<f32, D>>
max(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor<f32, D>>
min(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor<f32, D>>
softmax(&self, dim: usize) -> Result<Tensor<f32, D>>
layernorm(&self, gamma: &Tensor<f32, D>, beta: &Tensor<f32, D>, eps: f32) -> Result<Tensor<f32, D>>
rmsnorm(&self, gamma: &Tensor<f32, D>, eps: f32) -> Result<Tensor<f32, D>>
```

### 矩阵乘法与 Attention

```rust
matmul(&self, other: &Tensor<f32, D>) -> Result<Tensor<f32, D>>
matmul_t(&self, other: &Tensor<f32, D>, transpose_a: bool, transpose_b: bool) -> Result<Tensor<f32, D>>
flash_attention(&self, k: &Tensor<f32, D>, v: &Tensor<f32, D>, is_causal: bool) -> Result<Tensor<f32, D>>
flash_attention_batched(&self, k: &Tensor<f32, D>, v: &Tensor<f32, D>,
                        batch: usize, heads: usize, is_causal: bool) -> Result<Tensor<f32, D>>
```

## 8.3 safetensors API

```rust
// oyta-core/src/safetensors.rs

pub struct TensorHeader {
    pub dtype: String,                  // "F32" / "F16" / "BF16" / "F64"
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],         // [start, end] 字节偏移
}

pub struct SafetensorsHeader {
    pub tensors: BTreeMap<String, TensorHeader>,
    pub metadata: Option<BTreeMap<String, String>>,
}

/// 保存多个 Tensor 到文件（HuggingFace safetensors 格式）
pub fn save_safetensors(
    tensors: &BTreeMap<String, (&[f32], DynamicShape, DType)>,
    path: &Path,
) -> Result<()>;

/// 从文件加载所有 Tensor（返回 header + (name, bytes, dtype, shape) 列表）
pub fn load_safetensors(
    path: &Path,
) -> Result<(SafetensorsHeader, Vec<(String, Vec<u8>, DType, DynamicShape)>)>;
```

### 文件格式规范

```
小端序：
  8 字节:  JSON header 长度 (u64)
  N 字节:  JSON header（TensorHeader 字典）
  M 字节:  所有 Tensor 原始数据（连续排列）
```

## 8.4 字节序列化辅助

```rust
// oyta-core/src/bytes.rs
pub fn f32_to_bytes(v: f32) -> [u8; 4]
pub fn bytes_to_f32(b: &[u8]) -> f32
pub fn f64_to_bytes(v: f64) -> [u8; 8]
pub fn bytes_to_f64(b: &[u8]) -> f64
pub fn f16_to_bytes(v: F16) -> [u8; 2]
pub fn bytes_to_f16(b: &[u8]) -> F16
pub fn bf16_to_bytes(v: BF16) -> [u8; 2]
pub fn bytes_to_bf16(b: &[u8]) -> BF16
pub fn tensor_byte_size(shape: &DynamicShape, dtype: DType) -> usize
pub fn write_bytes<T: Copy>(dst: &mut Vec<u8>, v: &[T])
pub fn read_bytes<T: Copy>(src: &[u8], n: usize) -> Vec<T>
```

---

# 九、使用示例

## 示例 1：CPU matmul + LayerNorm + FlashAttention

```rust
use oyta_tensor::prelude::*;

fn main() -> Result<(), TensorError> {
    // 1. 创建张量（默认 CPU）
    let a = Tensor::<f32, Cpu>::randn([256, 512]);
    let b = Tensor::<f32, Cpu>::randn([512, 1024]);

    // 2. 矩阵乘法（自动选择 naive / tiled / BLAS / strassen）
    let c = a.matmul(&b)?;  // 256×1024

    // 3. LayerNorm + GELU（融合执行，减少中间 Tensor）
    let gamma = Tensor::<f32, Cpu>::ones([1024]);
    let beta  = Tensor::<f32, Cpu>::zeros([1024]);
    let ln = c.layer_norm(&gamma, &beta, 1e-5)?;
    let out = ln.gelu()?;

    // 4. FlashAttention（消灭 O(seq²) 中间张量）
    let q = Tensor::<f32, Cpu>::randn([512, 64]);
    let k = Tensor::<f32, Cpu>::randn([512, 64]);
    let v = Tensor::<f32, Cpu>::randn([512, 64]);
    let attn = q.flash_attention(&k, &v, true)?;  // causal mask

    // 5. 设备上下文
    let handle = Cpu::get();
    println!("CPU cores: {}", handle.properties.num_cores);

    Ok(())
}
```

## 示例 2：WGPU GPU matmul + 回读验证

```rust
use oyta_tensor::prelude::*;

fn main() {
    // 创建 WGPU Tensor（需要 feature=wgpu）
    let a = Tensor::<f32, Wgpu>::randn([64, 64]);
    let b = Tensor::<f32, Wgpu>::randn([64, 64]);
    let c_gpu = a.matmul(&b).unwrap();

    // 回读 CPU 对比（WSL2 实测 max_diff=0.00e0）
    let c_cpu = c_gpu.to_cpu();

    // 设备信息
    let handle = Wgpu::get();
    println!("WGPU device: {}", handle.properties.name);
}
```

## 示例 3：跨设备传输

```rust
use oyta_tensor::prelude::*;

fn main() {
    // CPU 创建 → 传到 GPU 算 → 回 CPU
    let a = Tensor::<f32, Cpu>::randn([128, 256]);
    let b = Tensor::<f32, Cpu>::randn([256, 512]);

    #[cfg(feature = "wgpu")]
    let result = {
        let a_gpu = a.to_wgpu();
        let b_gpu = b.to_wgpu();
        let c_gpu = a_gpu.matmul(&b_gpu).unwrap();
        c_gpu.to_cpu()
    };

    #[cfg(not(feature = "wgpu"))]
    let result = a.matmul(&b).unwrap();  // fallback CPU

    println!("{}", result.shape());  // [128, 512]
}
```

## 示例 4：safetensors 加载预训练权重

```rust
use oyta_tensor::prelude::*;
use std::collections::BTreeMap;

fn main() -> Result<(), TensorError> {
    // 从 HuggingFace safetensors 加载
    let (header, tensors) = load_safetensors("bert-base.safetensors")?;

    for (name, bytes, dtype, shape) in &tensors {
        let f32_data: Vec<f32> = match dtype {
            DType::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
            DType::F16 => bytes.chunks_exact(2).map(|c| {
                F16::from_le_bytes([c[0], c[1]]).to_f32()
            }).collect(),
            _ => continue,
        };
        let t = Tensor::<f32, Cpu>::from_data(f32_data, shape.clone())?;
        println!("{} -> {:?}", name, t.shape());
    }

    Ok(())
}
```

## 示例 5：f16 / bf16 精度转换

```rust
use oyta_tensor::prelude::*;

fn main() {
    let f32_t = Tensor::<f32, Cpu>::randn([1024, 1024]);

    // 降精度（F16 内存减一半）
    let f16_t = f32_t.to_f16();

    // BF16（范围和 f32 一样，但低 16 位被截断）
    let bf16_t = f32_t.to_bf16();

    // 计算（内部自动 f32 累加）
    let result_f32 = f16_t.matmul(&f16_t).unwrap().to_f32();
    // Accumulate trait 自动提升 → F16×F16 matmul 内部用 f32 累加器
}
```

## 示例 6：动态设备切换（AnyBackend）

```rust
use oyta_tensor::prelude::*;

fn main() {
    // 运行时全局设备切换
    set_runtime_device(DeviceRuntimeId::Wgpu);  // 所有未指定设备的操作走 WGPU

    let a = Tensor::<f32, Cpu>::randn([64, 64]);
    let b = Tensor::<f32, Cpu>::randn([64, 64]);
    let c = a.matmul(&b).unwrap();  // 内部走 WGPU dispatch
}
```

---

# 十、数值稳定性与安全保证

## 10.1 Softmax 稳定版公式

```rust
// oyta-device/src/backend/reduce.rs
pub fn softmax(a: *const f32, out: *mut f32, outer: usize, inner: usize) {
    // 1. 先找最大值 → exp 输入永远 ≤ 0，不会溢出
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..inner { max_val = max_val.max(unsafe { *a.add(i) }); }

    // 2. f64 累加器（避免百万元素时精度损失）
    let mut sum_exp = 0.0f64;
    for i in 0..inner {
        let v = ((unsafe { *a.add(i) } - max_val) as f64).exp();
        unsafe { *out.add(i) = v as f32; }
        sum_exp += v;
    }

    // 3. 最终归一化时再 cast 回 f32
    let inv = 1.0f64 / sum_exp;
    for i in 0..inner {
        unsafe { *out.add(i) = (*out.add(i) as f64 * inv) as f32; }
    }
}
```

## 10.2 LayerNorm / RMSNorm eps

```rust
pub const LAYERNORM_EPS: f32 = 1e-5;     // PyTorch 默认
pub const RMSNORM_EPS: f32 = 1e-6;       // 更小（RMS 不需要减 mean）
```

## 10.3 pairwise_sum（大数精度）

```rust
// 百万元素 sum：两两配对避免"大数淹没小数"
pub fn pairwise_sum_f64(a: &[f64]) -> f64 {
    if a.len() <= 256 { a.iter().sum() }
    else {
        let mid = a.len() / 2;
        pairwise_sum_f64(&a[..mid]) + pairwise_sum_f64(&a[mid..])
    }
}
```

## 10.4 Accumulate trait 自动提升

```rust
pub trait Accumulate {
    type Accumulator: TensorElement;
    fn to_accumulator(self) -> Self::Accumulator;
}

// F16 × F16 matmul 内部自动用 f32 累加器
// BF16 × BF16 matmul 内部自动用 f32 累加器
// f16 / bf16 reduce 自动用 f64 统计
```

## 10.5 debug_assert! 边界检查

所有 Backend 算子和 Tensor 方法都在 `#[cfg(debug_assertions)]` 下加了 debug_assert!：

| 位置 | 检查项 |
|---|---|
| matmul | `m > 0, n > 0, k > 0, lda >= k, ldb >= n, ldc >= n` |
| elwise_* | `n > 0` |
| layernorm / rmsnorm | `outer > 0, inner > 0, eps > 0.0` |
| flash_attention | `seq_q > 0, seq_k > 0, d_k >= 16, d_v >= 16` |
| reshape | 负索引有效位置 |
| slice | 范围在 shape 内 |
| DynamicShape::dim | 索引不越界 |

## 10.6 安全审计清单

| 检查项 | 结果 |
|---|---|
| `panic!/unimplemented!/todo!` | **0** |
| 非 `debug_assert!` 的 `assert!` | **0**（生产代码全用 debug_assert!） |
| `#[allow]` lint 抑制 | **1**（cuda.rs gpu_ffi 集中抑制，合法） |
| `unsafe {}` 块 | 仅 Backend 裸指针 + cuda.rs FFI + cuBLAS + wgpu 内部调用 |
| `checked_mul / checked_product` | 所有维度乘积都防溢出 |

## 10.7 数值精度保证

| 算子 | f32 相对误差阈值 | 内部精度 |
|---|---|---|
| matmul | ≤ 1e-4 | f32 累加器 |
| softmax | 行和 ≤ 1e-6 | f64 max + f64 sum + f64 inv_sum |
| layernorm | mean/var 精确 + 归一化误差 ≤ 1e-5 | f64 stat + f32 norm |
| elwise add | ≤ 1e-6 | f32 直接 |
| f16 roundtrip | ≤ 2e-3 | f32 中间 |
| bf16 roundtrip | ≤ 8e-3 | f32 中间 |

## 10.8 完整测试矩阵

| 层级 | 数量 | 内容 |
|---|---|---|
| Backend trait 签名（9 trait × 82 方法） | 编译器类型检查 | feature 组合全过 = 签名正确 |
| 单元测试 | **9** | matmul_correctness / dtype_cast_roundtrip / device_context_cpu / profiler / op_index / matmul_non_contiguous_ldc |
| 集成示例 | **10** | example_01 ~ example_10 全跑通 |
| GPU 数值验证 | **1** | example_11 → CPU vs WGPU matmul 4 尺寸 max_diff=0.00e0 |
| CUDA .cu 真实编译 | **5** | nvcc 13.5 + GCC 13 → 5 .cu kernels 全部出 .o |
| Feature 矩阵 | **10** | default / cuda / metal / wgpu / blas / memory_pool / debug_assertions / no-default+metal / no-default+wgpu / all-features |

---

**文档版本**：v0.1.0 · 2026-11-20
