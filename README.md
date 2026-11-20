# oyta-ai — Rust AI 底层系统 · Generic Backend Tensor Library

[![crates.io](https://img.shields.io/crates/v/oyta-tensor.svg)](https://crates.io/crates/oyta-tensor)
[![license](https://img.shields.io/crates/l/oyta-tensor.svg)](https://crates.io/crates/oyta-tensor)
[![downloads](https://img.shields.io/crates/d/oyta-tensor.svg)](https://crates.io/crates/oyta-tensor)
[![rust](https://img.shields.io/badge/rust-1.85+-orange.svg)](https://www.rust-lang.org/)

一套 **纯 Rust 实现的 AI 底层算子库**，核心设计决策：**泛型 Backend trait（路线 B blanket impl）**——一套 trait 签名覆盖 f32 / f16 / bf16 / f64 全部 dtype，一套 blanket impl 覆盖 CPU / CUDA / WGPU / Metal 全部设备。

## 🚀 快速开始

### 一行引入

```toml
[dependencies]
oyta-tensor = "0.1.1"
```

### Hello World

```rust
use oyta_tensor::prelude::*;

fn main() {
    // 1. 创建张量
    let a = Tensor::<f32, Cpu>::randn([32, 512]);
    let b = Tensor::<f32, Cpu>::randn([512, 768]);

    // 2. 矩阵乘法（内部自动选择 naive / tiled / BLAS / Strassen）
    let c = a.matmul(&b).unwrap();
    println!("matmul shape: {:?}", c.shape());

    // 3. LayerNorm + GELU（融合算子，一次 kernel 完成）
    let gamma = Tensor::<f32, Cpu>::ones([768]);
    let beta  = Tensor::<f32, Cpu>::zeros([768]);
    let ln    = c.layer_norm(&gamma, &beta, 1e-5).unwrap();
    let act   = ln.gelu().unwrap();

    println!("output shape: {:?}", act.shape());
}
```

### 跨设备无缝切换

```rust
// 从 CPU 切到 WGPU（编译期泛型切换，零开销）
let a_gpu = a.to_wgpu();
let c_gpu = a_gpu.matmul(&b.to_wgpu()).unwrap();

// FlashAttention（消灭 O(seq²) 中间张量）
let q = Tensor::<f32, Cpu>::randn([4096, 64]);
let k = Tensor::<f32, Cpu>::randn([4096, 64]);
let v = Tensor::<f32, Cpu>::randn([4096, 64]);
let out = q.flash_attention(&k, &v, true).unwrap(); // causal mask
```

## ✨ 核心特性

| 特性 | 说明 |
|---|---|
| **泛型 Backend** | `ElwiseOps<T>` / `ReduceOps<T>` / `FlashAttentionOps<T>` / `FusedOps<T>` / `InPlaceOps<T>` — 一套 trait 覆盖所有 dtype |
| **Blanket Impl** | `impl<T: Float> ElwiseOps<T> for Cpu {}` — 每个 Device 5 行 blanket impl 覆盖全部 dtype |
| **自动精度提升** | `Accumulate` trait：f16/bf16→f32、f32→f64，softmax/layernorm/rmsnorm 全用 f64 累加 |
| **四设备后端** | CPU (rayon) / CUDA (cuBLAS + nvcc) / WGPU (WGSL compute shader) / Metal (stub) |
| **FlashAttention v2** | 在线 softmax + causal mask，内存 O(seq·d) 而非 O(seq²) |
| **4 融合算子** | LayerNormGELU / LinearBiasGELU / RMSNormSwiglu / MatmulActivation |
| **Strassen 递归** | matmul dispatch：32→naive，32-512→tiled+BLAS，≥512→Strassen crossover=512 |
| **数值稳定规约** | pairwise_sum 递归二分归约（比 naive sum 精度高 2-3 个数量级） |
| **safetensors** | HuggingFace 格式完整读写 |
| **安全审计** | 零 `panic!/unimplemented!/todo!`、33 处 `debug_assert!` 边界检查、1 处合法 `#[allow]` |

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户代码                              │
│  Tensor<f32, Cpu> / Tensor<F16, Wgpu> / AnyBackend      │
└─────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  oyta-tensor  (公共 API 入口 · prelude re-export)       │
└─────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  oyta-ops     (算子引擎 · dispatch / FlashAttn / fuse)  │
└─────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  oyta-core    (Tensor 核心 · DynamicShape / Strides)     │
└─────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  oyta-device  (Backend trait + 4 设备 blanket impl)     │
│  ┌───────────┬───────────┬───────────┬───────────┐    │
│  │   Cpu     │   Cuda    │   Wgpu     │   Metal   │    │
│  │  rayon    │  cuBLAS   │  WGSL      │   stub    │    │
│  └───────────┴───────────┴───────────┴───────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Crate 依赖链

```
oyta-tensor → oyta-ops → oyta-core → oyta-device
     └──────────────┬───────────────┘
                    ▼
              oyta-device（Backend trait 定义处）
```

### Feature Flags

```toml
[dependencies]
oyta-tensor = { version = "0.1.1", features = ["cpu", "cuda", "wgpu", "blas"] }
```

| Feature | 默认 | 说明 |
|---|---|---|
| `cpu` | ✅ | CPU rayon 并行后端 |
| `cuda` | ❌ | NVIDIA CUDA + cuBLAS（需要 nvcc 13.x） |
| `wgpu` | ❌ | WebGPU（跨平台 DX/Vulkan/Metal） |
| `metal` | ❌ | Apple Metal（macOS/iOS） |
| `blas` | ❌ | OpenBLAS/MKL 加速 |
| `memory_pool` | ❌ | 设备内存池 |
| `debug_assertions` | ❌ | 33 处边界检查（release 关闭） |

## 📖 API 速查

### Tensor 构造

```rust
Tensor::<f32, Cpu>::zeros([3, 4, 5]);
Tensor::<f32, Cpu>::ones([64, 64]);
Tensor::<f32, Cpu>::randn([32, 512]);
Tensor::<f32, Cpu>::arange(0.0, 100.0, 1.0);
Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0], [3]);
```

### 形状变换

```rust
t.reshape([-1, 10]);      // 自动推断 -1
t.transpose(0, 1);
t.permute([2, 0, 1]);
t.unsqueeze(0);           // 插入维度
t.squeeze();               // 压缩大小为 1 的维度
t.slice(0..10, 0..5);     // 切片
t.contiguous();            // 保证内存连续
```

### 逐元素算子

```rust
t.add(&s) / t.sub(&s) / t.mul(&s) / t.div(&s) / t.pow(2.0)
t.neg() / t.abs() / t.exp() / t.log() / t.sqrt() / t.sin() / t.cos()
t.relu() / t.gelu() / t.sigmoid() / t.tanh() / t.silu()
```

### 规约 + 归一化

```rust
t.sum(-1) / t.mean(-1) / t.max(-1) / t.min(-1) / t.softmax(-1)
t.layer_norm(&gamma, &beta, 1e-5)
t.rms_norm(&gamma, 1e-6)
```

### matmul + attention

```rust
a.matmul(&b)                      // 自动 dispatch：naive→tiled→BLAS→Strassen
a.batch_matmul(&b)                // batched GEMM
q.flash_attention(&k, &v, true)   // causal mask FlashAttention v2
```

### 融合算子

```rust
t.layernorm_gelu(&gamma, &beta, 1e-5)
x.linear_bias_gelu(&w, &bias)
x.rmsnorm_swiglu(&w1, &w2, &w3, &gamma, 1e-6)
a.matmul_activation(&b, &c, ActivationType::ReLU)
```

### 跨设备传输

```rust
t.to_cpu();
t.to_cuda();
t.to_wgpu();
t.to_device::<Metal>();
```

### 精度转换

```rust
t.to_f16() / t.to_bf16() / t.to_f32() / t.to_f64()
```

### safetensors

```rust
let mut map = std::collections::HashMap::new();
map.insert("model.layers.0.weight".into(), &tensor);
save_safetensors(&map, "model.safetensors").unwrap();

let loaded = load_safetensors("model.safetensors").unwrap();
```

## 🔧 从源码编译

```bash
# 完整 feature（需要 CUDA toolkit 13.x + nvcc）
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
cargo build --workspace --all-features

# 仅 CPU + WGPU（推荐，大多数环境可直接编译）
cargo build --workspace --features wgpu

# 运行测试
cargo test --workspace --tests

# 运行示例
cargo run --example example_01_tensor_basics
cargo run --example example_11_wgpu_verify --features wgpu

# criterion benchmark
cargo bench -p oyta-ops --bench matmul_bench
```

### CUDA 要求

| 组件 | 版本 | 下载 |
|---|---|---|
| CUDA Toolkit | ≥ 13.0（推荐 13.5） | [developer.download.nvidia.com](https://developer.download.nvidia.com/compute/cuda/repos/) |
| GCC | ≥ 13 | `apt-get install gcc-13 g++-13` |
| glibc | ≥ 2.40 | Ubuntu 24.04+ / Debian 13+ |

## 📊 性能目标

| 算子 | CPU (512×512) | GPU (A100) | 说明 |
|---|---|---|---|
| matmul naive | 基准 | — | 参考 |
| matmul tiled | ~1.5× | — | 128×128 tile |
| matmul BLAS/Strassen | ~3× | — | 递归 crossover=512 |
| cuBLAS Sgemm | — | ~300× | row-major trick |
| FlashAttention | 基准 | ~50× | vs naive O(seq²) |

## 📦 4 Crate 说明

| Crate | 职责 | 发布地址 |
|---|---|---|
| `oyta-device` | Backend trait 定义 + 4 设备 blanket impl | [crates.io/oyta-device](https://crates.io/crates/oyta-device) |
| `oyta-core` | Tensor 核心数据结构 | [crates.io/oyta-core](https://crates.io/crates/oyta-core) |
| `oyta-ops` | 算子引擎 + dispatch + FlashAttention | [crates.io/oyta-ops](https://crates.io/crates/oyta-ops) |
| `oyta-tensor` | 公共 API 入口（用户只需依赖这个） | [crates.io/oyta-tensor](https://crates.io/crates/oyta-tensor) |

## 🛡️ 安全保证

- ✅ 零 `panic!` / `unimplemented!` / `todo!`（release 路径）
- ✅ 33 处 `debug_assert!` 边界检查（release 自动关闭）
- ✅ 1 处 `#[allow]`（cuda.rs extern "C" FFI 常量，合法）
- ✅ 所有 unsafe 块前置 `checked_mul` / `debug_assert!` 防御
- ✅ `saturating` checked 乘法 + `pairwise_sum` 数值稳定
- ✅ f64 累加器（无论 T 是什么类型）

## 📄 许可证

MIT + Apache-2.0 双许可证，自由使用于商业和非商业场景。

## 🤝 贡献

```bash
git clone https://github.com/oytaer/oyta_ai.git
cd oyta_ai
cargo build --workspace --all-features
cargo test --workspace
```

---

**仓库**：[github.com/oytaer/oyta_ai](https://github.com/oytaer/oyta_ai)  
**文档**：[docs/AI底层完整技术说明文档.md](docs/AI底层完整技术说明文档.md)（中文版）/ [docs/AI底层完整技术说明文档-English.md](docs/AI底层完整技术说明文档-English.md)
