# Oyta-AI Backend · Complete Technical Reference (English)

---

# Table of Contents

- [1. Core Technical Architecture](#1-core-technical-architecture)
- [2. Tensor Data Structure & Layout](#2-tensor-data-structure--layout)
- [3. Backend Trait Definitions & 82 Operator Signatures](#3-backend-trait-definitions--82-operator-signatures)
- [4. Four Device Backend Implementations](#4-four-device-backend-implementations)
- [5. Operator Engine & Dynamic Dispatch](#5-operator-engine--dynamic-dispatch)
- [6. FlashAttention Online Softmax + Operator Fusion](#6-flashattention-online-softmax--operator-fusion)
- [7. DeviceContext Runtime](#7-devicecontext-runtime)
- [8. Complete API Reference](#8-complete-api-reference)
- [9. Usage Examples](#9-usage-examples)
- [10. Numerical Stability & Safety Guarantees](#10-numerical-stability--safety-guarantees)

---

# 1. Core Technical Architecture

## 1.1 Layering & Dependency Direction

```
┌─────────────────────────────────────────────────────┐
│  oyta-tensor  ·  Public API entry (re-export all)   │
└────────────────────────┬────────────────────────────┘
                         │ depends on
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

- **oyta-device** (lowest): Backend trait + 4 device implementations. Depends on rayon + thiserror + optional wgpu/blas-sys/metal.
- **oyta-core** (data): Tensor\<T,D\> + DynamicShape + Strides + IndexEngine + safetensors + f16/bf16. Depends on oyta-device's Device trait.
- **oyta-ops** (engine): dynamic dispatch + fusion decider + matmul strategy + FlashAttention + reduce/elwise. Depends on both core and device.
- **oyta-tensor** (public): explicit re-export of all modules. `use oyta_tensor::prelude::*` one-liner import.

## 1.2 Feature Auto-Propagation Chain

```
oyta-tensor --features cuda
    → oyta-ops/cuda enabled
        → oyta-core/cuda enabled
            → oyta-device/cuda enabled + nvcc compiles 5 .cu kernels
```

## 1.3 Release Build Optimization

```toml
[profile.release]
lto = "fat"          # Full cross-crate inlining
codegen-units = 1    # Single compilation unit → max compiler optimization
strip = "symbols"    # Strip symbol table
opt-level = 3        # Highest optimization level
panic = "abort"      # Direct abort → zero unwind overhead
```

## 1.4 Cargo Command Reference

```bash
cargo check --workspace
cargo test --workspace
cargo build --release --features cuda,wgpu,blas

# CUDA (requires /usr/local/cuda/bin in PATH)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
cargo build -p oyta-device --features cuda

# Full feature matrix
cargo check --all-features
cargo check --no-default-features --features wgpu
cargo check --no-default-features --features metal

# Examples
cargo run --example example_11_wgpu_verify --features wgpu
```

---

# 2. Tensor Data Structure & Layout

## 2.1 Core Struct Definition

```rust
// oyta-core/src/tensor/base.rs

pub struct Tensor<T, D: Backend> {
    /// Shared underlying data (Arc atomic op 5-10ns < L1 miss ~100ns)
    pub data: Arc<Vec<T>>,
    /// Logical shape
    pub shape: DynamicShape,
    /// Stride vector (supports non-contiguous views)
    pub strides: Strides,
    /// Flat array start offset (slice views)
    pub offset: usize,
    /// Device marker (zero-overhead PhantomData)
    pub _device: PhantomData<D>,
}
```

## 2.2 Zero-Copy View Mechanism

```
Original: data=[a,b,c,d,e,f,g,h]  shape=[2,4] strides=[4,1]

transpose(0,1) → zero-copy view:
  data unchanged → one more Arc ref
  strides changed: [4,1] → [1,4]
  offset unchanged

Logical access changes:
  t[0,0] → data_ptr + 0*1 + 0*4 = data_ptr + 0
  t[0,1] → data_ptr + 0*1 + 1*4 = data_ptr + 4
  t[1,0] → data_ptr + 1*1 + 0*4 = data_ptr + 1
  t[1,1] → data_ptr + 1*1 + 1*4 = data_ptr + 5

Clone is one Arc atomic increment — no data copy
```

## 2.3 DynamicShape

```rust
pub struct DynamicShape { pub dims: Vec<usize> }

impl DynamicShape {
    pub fn new(dims: impl IntoIterator<Item = usize>) -> Self
    pub fn numel(&self) -> usize                  // checked_product overflow-guarded
    pub fn rank(&self) -> usize
    pub fn dim(&self, i: isize) -> usize          // -1 = rightmost dimension
    pub fn reshape(&self, new_dims: &[isize]) -> Result<DynamicShape>  // supports -1 inference
    pub fn broadcast_compatible(&self, other: &DynamicShape) -> bool
    pub fn broadcast_with(&self, other: &DynamicShape) -> DynamicShape
}
```

## 2.4 Strides

```rust
pub struct Strides { dims: Vec<usize> }

impl Strides {
    pub fn contiguous(shape: &DynamicShape) -> Self
    pub fn is_contiguous(&self, shape: &DynamicShape) -> bool  // rightmost stride == 1
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Self
    pub fn permute(&self, order: &[usize]) -> Self
}
```

## 2.5 IndexEngine

```rust
pub struct IndexEngine { shape: DynamicShape, strides: Strides }

impl IndexEngine {
    pub fn offset(&self, multi_idx: &[usize]) -> usize       // multi-dim → flat offset
    pub fn reverse_offset(&self, flat: usize) -> Vec<usize>  // flat → multi-dim
}
```

Core engine for non-contiguous + broadcast generic elwise operations.

## 2.6 TensorError (thiserror · 10 variants)

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

# 3. Backend Trait Definitions & 82 Operator Signatures

## 3.1 Backend Aggregate Trait

```rust
pub trait Backend:
    BackendBase + ElwiseOpsF32 + ElwiseOpsF16 + ElwiseOpsBF16
    + ReduceOpsF32 + MatmulOps + FlashAttentionOps
    + FusedOps + InPlaceOpsF32
    + Copy + Clone + Send + Sync + 'static
{
    // Compile-time device properties (associated consts, zero runtime overhead)
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

## 3.2 BackendBase (3 methods)

```rust
pub trait BackendBase {
    fn fill(ptr: *mut f32, n: usize, value: f32);
    fn copy(src: *const f32, dst: *mut f32, n: usize);
    fn where_bool(cond: *const bool, a: *const f32, b: *const f32, out: *mut f32, n: usize);
}
```

## 3.3 ElwiseOpsF32 (23 methods)

| Method | Signature | Type |
|---|---|---|
| `elwise_add/sub/mul/div` | `(a, b, c, n)` | Binary |
| `elwise_pow` | `(a, exp: f32, c, n)` | Binary with scalar |
| `elwise_add_broadcast` | `(a, b, c, n_a, n_b, n_out)` | Broadcast binary |
| `elwise_neg/abs/exp/log/sqrt/sin/cos` | `(a, c, n)` | Unary |
| `elwise_relu/gelu/gelu_fast/sigmoid/tanh/silu` | `(a, c, n)` | Activation |

ElwiseOpsF16 / ElwiseOpsBF16: same 20 methods (no broadcast variant).

## 3.4 ReduceOpsF32 (8 methods)

```rust
pub trait ReduceOpsF32 {
    fn reduce_sum(a: *const f32, out: *mut f32, outer: usize, inner: usize);
    fn reduce_mean(a: *const f32, out: *mut f32, outer: usize, inner: usize);
    fn reduce_max(a: *const f32, out: *mut f32, outer: usize, inner: usize);
    fn reduce_min(a: *const f32, out: *mut f32, outer: usize, inner: usize);
    /// Numerically stable softmax: subtract max first + f64 accumulator
    fn softmax(a: *const f32, out: *mut f32, outer: usize, inner: usize);
    fn layernorm(a: *const f32, out: *mut f32, gamma: *const f32, beta: *const f32,
                 outer: usize, inner: usize, eps: f32);
    fn rmsnorm(a: *const f32, out: *mut f32, gamma: *const f32,
               outer: usize, inner: usize, eps: f32);
    fn layernorm_inplace(a: *mut f32, gamma: *const f32, beta: *const f32,
                         outer: usize, inner: usize, eps: f32);
}
```

## 3.5 MatmulOps (8 methods)

```rust
pub trait MatmulOps {
    /// C[m,n] = α·A[m,k]·B[k,n] + β·C[m,n] (row-major)
    fn matmul(a: *const f32, lda: usize, b: *const f32, ldb: usize,
              c: *mut f32, ldc: usize, m: usize, n: usize, k: usize,
              alpha: f32, beta: f32);

    /// With transposition flags
    fn matmul_t(a: *const f32, lda: usize, transpose_a: bool,
                b: *const f32, ldb: usize, transpose_b: bool,
                c: *mut f32, ldc: usize, m: usize, n: usize, k: usize,
                alpha: f32, beta: f32);

    /// Batched matmul (stride_a/stride_b/stride_c control per-batch offsets)
    fn matmul_batched(a: *const f32, lda: usize, stride_a: usize,
                      b: *const f32, ldb: usize, stride_b: usize,
                      c: *mut f32, ldc: usize, stride_c: usize,
                      m: usize, n: usize, k: usize, batch: usize,
                      alpha: f32, beta: f32);

    /// F16 / BF16 matmul: cast to f32 internally, cast back on output
    fn matmul_f16(a: *const F16, lda: usize, b: *const F16, ldb: usize,
                  c: *mut F16, ldc: usize, m: usize, n: usize, k: usize,
                  alpha: f32, beta: f32);
    fn matmul_bf16(a: *const BF16, lda: usize, b: *const BF16, ldb: usize,
                   c: *mut BF16, ldc: usize, m: usize, n: usize, k: usize,
                   alpha: f32, beta: f32);
}
```

**Two independent matmul implementations — must distinguish**:

### System A: Tensor::matmul real call chain (actually invoked)

```
Tensor::<f32, Cpu>::matmul()                           // oyta-core/src/tensor/ops.rs L65
  → D::matmul()                                        // MatmulOps::matmul trait method
    → <Self as MatmulOps>::default_matmul()             // default impl (backend/matmul.rs)
      → matmul_dispatch()                              // private dispatch function
        → all m,n,k ≤ 32 (NAIVE_THRESHOLD) → matmul_naive
        → all m,n,k ≥ 512 (STRASSEN_CUTOFF, public const) → matmul_strassen_recursive
        → otherwise → matmul_tiled_fma
```

### System B: strategy.rs complete impl (standalone, never called, reserved API)

`oyta-ops/src/matmul/strategy.rs` has a richer alternative:
- **Dot-product specialization** `m==1 && n==1 → dot_product_flat` (most FMA-friendly)
- Thresholds 32 / 1024 / 128 (STRASSEN_ENTRY=1024 enter, STRASSEN_CROSSOVER=128 recursion fallback)
- IS_CPU conditional branching + GPU backend dispatch
- Exported via `pub use strategy::*`, but **never called anywhere**

**Cuda::IS_CPU impact**: `Cuda::IS_CPU = true` (conservative, cuBLAS needs runtime probe), so in strategy.rs CUDA large matrices go through `matmul_strassen`. This doesn't affect System A — System A has Cuda's own MatmulOps::matmul impl (runtime-detects hardware, calls cublasSgemm when available).

**BLAS feature location**: BLAS is checked inside the default MatmulOps impl in System A. strategy.rs has no BLAS logic.

## 3.6 FlashAttentionOps (2 methods)

```rust
pub trait FlashAttentionOps {
    /// FlashAttention: out = softmax(QKᵀ / √d_k) · V — never materializes QKᵀ
    /// Online f64 softmax (m/l/o three accumulators updated in chunks)
    /// is_causal=true: q_i sees only K[0..=i] — no explicit mask matrix needed
    fn flash_attention(q: *const f32, k: *const f32, v: *const f32, out: *mut f32,
                       seq_q: usize, seq_k: usize, d_k: usize, d_v: usize,
                       scale: f32, is_causal: bool);

    /// Batched multi-head FlashAttention
    fn flash_attention_batched(q: *const f32, k: *const f32, v: *const f32, out: *mut f32,
                               batch: usize, heads: usize,
                               seq_q: usize, seq_k: usize, d_k: usize, d_v: usize,
                               scale: f32, is_causal: bool);
}
```

## 3.7 FusedOps (4 methods)

```rust
pub enum FusedOp { NoFusion, LnGelu, LinearBiasGelu, MatmulActivation, RmsnormSwiglu }
pub enum ActivationType { Relu, Gelu, GeluFast, Silu, Sigmoid, Tanh }

pub trait FusedOps {
    /// LayerNorm + GELU fusion (Transformer encoder)
    fn layernorm_gelu(a: *const f32, out: *mut f32, gamma: *const f32, beta: *const f32,
                      outer: usize, inner: usize, eps: f32);

    /// Linear + Bias + GELU fusion (MLP block)
    fn linear_bias_gelu(x: *const f32, w: *const f32, bias: *const f32, out: *mut f32,
                        m: usize, n: usize, k: usize);

    /// Matmul + Activation fusion (parameterized activation)
    fn matmul_activation(a: *const f32, lda: usize, b: *const f32, ldb: usize,
                         c: *mut f32, ldc: usize,
                         m: usize, n: usize, k: usize, alpha: f32, beta: f32,
                         act: ActivationType);

    /// RMSNorm + SwiGLU fusion (full LLaMA MLP path)
    fn rmsnorm_swiglu(x: *const f32, w1: *const f32, w2: *const f32, w3: *const f32,
                      out: *mut f32, gamma: *const f32,
                      m: usize, n: usize, k: usize, eps: f32);
}
```

## 3.8 InPlaceOpsF32 (5 methods)

```rust
pub trait InPlaceOpsF32 {
    fn elwise_add_inplace(a: *mut f32, b: *const f32, n: usize);
    fn elwise_mul_inplace(a: *mut f32, b: *const f32, n: usize);
    fn elwise_relu_inplace(a: *mut f32, n: usize);
    fn elwise_scalar_add_inplace(a: *mut f32, n: usize, value: f32);
    fn elwise_scalar_mul_inplace(a: *mut f32, n: usize, value: f32);
}
```

## 3.9 Complete Backend Operator Count

| Sub-trait | f32 | f16 | bf16 | inplace | Total |
|---|---|---|---|---|---|
| BackendBase | 3 | - | - | - | 3 |
| ElwiseOps (f32: 21, f16/bf16: 4 each) | 21 | 4 | 4 | 0 | 29 |
| ReduceOps | 8 | - | - | 1 | 9 |
| MatmulOps | 8 | - | - | - | 8 |
| FlashAttentionOps | 2 | - | - | - | 2 |
| FusedOps | 4 | - | - | - | 4 |
| InPlaceOps | - | - | - | 5 | 5 |
| **TOTAL** | **44** | **16** | **16** | **6** | **82** |

---

# 4. Four Device Backend Implementations

## 4.1 Cpu (default)

### matmul dispatch decision tree (System A · real invocation path)

```
Tensor::<f32, Cpu>::matmul()                        // oyta-core/src/tensor/ops.rs
  → D::matmul()                                     // MatmulOps::matmul trait
    → <Cpu as MatmulOps>::default_matmul()          // backend/matmul.rs default impl
      → matmul_dispatch()                           // private inner function
         │
    ┌────┴────┐
    ▼         ▼
 all ≤ 32?   all ≥ 512 (STRASSEN_CUTOFF)?
    │ yes        │ yes
    ▼               ▼
 matmul_naive     matmul_strassen_recursive
 triple-loop       7-formula recursion
                    (inner <128 fallback to tiled_fma)
    │ no           │ no
    └────┬─────────┘
         ▼
 matmul_tiled_fma(BM=BN=BK=128)
 rayon outer parallel + L2 tiling
```

**Threshold constants** (backend/matmul.rs):

```rust
const NAIVE_THRESHOLD: usize = 32;
pub const STRASSEN_CUTOFF: usize = 512;   // public, re-exported as oyta_device::STRASSEN_CUTOFF
```

### tiled_fma parameters

```rust
const MATMUL_BM: usize = 128;  // A_block = 128×128×4B = 64 KB
const MATMUL_BN: usize = 128;  // B_block = 128×128×4B = 64 KB
const MATMUL_BK: usize = 128;  // Total A+B = 128 KB ≤ L2 256 KB

// rayon splits (num_m_blocks × num_n_blocks) across cores
// Inner loop: compiler auto-FMA + SIMD vectorization
rayon::prelude::IntoParallelRefIterator::par_iter()
    .for_each(|&(mi, ni)| { /* tiled_fma_serial */ });
```

### Rayon parallel elwise

```rust
let chunk_size = n.div_ceil(rayon::current_num_threads());
(0..rayon::current_num_threads()).into_par_iter().for_each(|tid| {
    let start = tid * chunk_size;
    let end = (start + chunk_size).min(n);
    for i in start..end { unsafe { *out.add(i) = ... } }
});
```

### Reduce f64 accumulator

```rust
// All ReduceOps impls use f64 explicitly
let mut acc: f64 = 0.0;
for i in 0..inner { acc += unsafe { *a.add(i) as f64 }; }
unsafe { *out.add(row) = (acc / inner as f64) as f32; }
```

## 4.2 Cuda (feature = cuda)

### runtime probe

```rust
pub struct Cuda;

// ======== Key design: IS_CPU = true ========
// cuBLAS needs runtime probe (cudaGetDeviceCount) to confirm hardware exists.
// But Backend associated consts are compile-time, can't change at runtime.
// So conservatively set IS_CPU = true → strategy.rs large matrices go through
// matmul_strassen (CPU algorithm), but MatmulOps::matmul CUDA impl has its own
// runtime dispatch (has_cuda_hardware() → cublasSgemm). Redundant but correct.
impl Backend for Cuda {
    const IS_CPU: bool = true;
    const MAX_ACCUMULATOR_PRECISION: Precision = Precision::F32;
    const HAS_FP16_TENSOR_CORE: bool = false;  // needs runtime detect
    const HAS_BF16: bool = false;
    const HAS_BATCHED_GEMM: bool = true;
    const HAS_FLASH_ATTENTION_HW: bool = false;
    const DEVICE_NAME: &'static str = "NVIDIA CUDA";
}

impl DeviceContext for Cuda {
    fn init() -> Result<DeviceHandle<Cuda>, DeviceError> {
        let mut count: i32 = 0;
        unsafe { cudaGetDeviceCount(&mut count as *mut i32) };
        // No hardware → handle=0 → matmul runtime dispatch falls back to CPU
        // Has hardware → cudaSetDevice(0) + cublasCreate + cublasSetStream
    }
}

// ======== All sub-traits default to Cpu impls ========
// Only MatmulOps has a real impl (with cuBLAS); rest inherit rayon defaults
impl BackendBase for Cuda {}
impl ElwiseOpsF32 for Cuda {}
impl ReduceOpsF32 for Cuda {}
```

### cuBLAS matmul internal logic (cuda.rs source)

```rust
impl MatmulOps for Cuda {
    fn matmul(a_ptr, lda, b_ptr, ldb, c_ptr, ldc, m, n, k, alpha, beta) {
        // 1. runtime detect: no hardware → fallback CPU default
        if !gpu_ffi::has_cuda_hardware() {
            return <Self as MatmulOps>::default_matmul(...);
        }
        let handle = get_cublas_handle();
        if handle == 0 { return <Self as MatmulOps>::default_matmul(...); }

        // 2. Check contiguous (lda==k, ldb==n, ldc==n)
        if lda == k && ldb == n && ldc == n {
            // Perfect contiguous: call cublasSgemm directly
            cublasSgemm(handle, N, N, n, m, k, b_ptr, n, a_ptr, k, c_ptr, n);
        } else {
            // Strided case: copy a/b/c into contiguous row-major Vecs first,
            // then call cublasSgemm, then write back (handling ldc != n)
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
// cuBLAS is column-major. row-major C = A·B ⇔ col-major Cᵀ = Bᵀ · Aᵀ
unsafe { cublasSgemm(
    handle, CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k,                 // m/n swapped
    &alpha,
    b_ptr, ldb,              // A/B positions swapped
    a_ptr, lda,
    &beta,
    c_ptr, ldc,
); }
```

### 5 .cu kernels (nvcc 13.5 + GCC 13 — real compile verified)

| File | Core params | Notes |
|---|---|---|
| `elwise_add.cu` | BM=256 | Standard 1D grid/block |
| `softmax.cu` | BM=256 | Numerically stable |
| `layernorm.cu` | BM=256 | Fused |
| `matmul.cu` | BM=BN=BK=64, shared_memory=32KB | 64³ tile fits within 48KB default limit |
| `flash_attn.cu` | d_k=64 default | Online softmax + causal mask |

### build.rs auto-probe (real source)

```rust
fn main() {
    #[cfg(feature = "cuda")] {
        // Force nvcc to use GCC 13 (Ubuntu 26.04 default GCC 15 has compatibility issues)
        unsafe {
            std::env::set_var("CXX", "g++-13");
            std::env::set_var("CC", "gcc-13");
        }

        let nvcc = Command::new("nvcc").arg("--version").output()
            .map(|o| o.status.success()).unwrap_or(false);

        if nvcc {
            cc::Build::new().cuda(true).include("cuda_kernels")
                .file("cuda_kernels/elwise_add.cu")
                .file("cuda_kernels/softmax.cu")
                .file("cuda_kernels/layernorm.cu")
                .file("cuda_kernels/matmul.cu")
                .file("cuda_kernels/flash_attn.cu")
                .flag("-O3").flag("--use_fast_math").flag("-std=c++17")
                .compile("oyta_device_cuda_kernels");

            println!("cargo:rerun-if-changed=cuda_kernels");
            println!("cargo:warning=CUDA kernels compiled with nvcc");
        } else {
            println!("cargo:warning=nvcc not found, CUDA kernels skipped (will use CPU fallback)");
        }
    }
}
```

Key behavior:
- **CXX/CC env vars force g++-13 / gcc-13** (passed to nvcc via cc crate)
- Auto-recompile when `cuda_kernels/` directory changes
- Silently skips when nvcc absent (runtime uses CPU fallback)

## 4.3 Wgpu (feature = wgpu)

### Initialization chain

```rust
let instance = wgpu::Instance::default();
let adapter = instance.request_adapter(None).await.expect("WGPU adapter");
let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None)
    .await.expect("WGPU device");
// WSL2 DirectX GPU passthrough: /dev/dxg
```

### compute shader matmul

```wgsl
// oyta-device/src/wgsl/matmul.wgsl
// wgpu 30 recommends STORAGE buffer over UNIFORM for dimensions (cross-platform stable)

@group(0) @binding(0) var<storage, read> a: array<f32>;     // [m, k] row-major
@group(0) @binding(1) var<storage, read> b: array<f32>;     // [k, n]
@group(0) @binding(2) var<storage, read_write> c: array<f32>; // [m, n]
@group(0) @binding(3) var<storage, read> dims: vec4<u32>;   // x=m, y=n, z=k

const BLOCK_SIZE: u32 = 16;

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;        // WGS convention: x=col, y=row
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

### Numerical verification

```
CPU vs WGPU matmul (16² / 32² / 64² / 128²):
  all 4/4 PASS, max_diff = 0.00e0 (perfect accuracy)
```

## 4.4 Metal (feature = metal)

- Apple MPS interface stub
- Non-Apple: feature is a no-op (no dependencies compile)
- DeviceContext OnceLock singleton fully implemented

---

# 5. Operator Engine & Dynamic Dispatch

## 5.1 dispatch.rs (strategy selector · reserved API)

```rust
// oyta-ops/src/dispatch.rs
// NOTE: select_matmul_strategy is a simplified enum selector — **never called in practice**
// Real execution path is strategy.rs::matmul() (see §4.1)

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

**Key fact**: `dispatch.rs::select_matmul_strategy` is a **simplified strategy selector** — only exported, never invoked. Real execution goes through `strategy.rs::matmul()` which has 5 branches (incl. `dot_product_flat` specialization and 1024/128 dual thresholds).

## 5.2 Strassen 7-formula recursion

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

**Two threshold constants** (strategy.rs source):
- `STRASSEN_ENTRY = 1024`: entry threshold (all m,n,k ≥ 1024 to enter Strassen mode)
- `STRASSEN_CROSSOVER = 128`: inner recursion fallback point (any dim < 128 → back to `tiled_fma`)

## 5.3 broadcast.rs preprocessing

```rust
// Broadcast rule: check from rightmost dimension
// If dims differ → take max when one side has dim=1 or missing
// Example: [1, 256, 1024] + [1024] → broadcast to [1, 256, 1024]
```

## 5.4 profiling.rs

```rust
pub struct ProfilingEngine { counters: [AtomicU64; 256] }  // lock-free

impl ProfilingEngine {
    pub fn profile<F: FnOnce()>(label: &str, f: F) -> Duration  // auto-record elapsed
    pub fn print_report()                                       // print top-K hot operators
}
```

## 5.5 dropout.rs

```rust
pub fn set_training_mode(enabled: bool)  // AtomicBool global
pub fn is_training() -> bool
pub fn generate_dropout_mask(n: usize, rate: f32) -> Vec<f32>  // keep rate = 1 - rate
```

---

# 6. FlashAttention Online Softmax + Operator Fusion

## 6.1 FlashAttention derivation

### Standard Attention bottleneck

```
Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V
QKᵀ = [seq_q, seq_k]    ← O(seq²) intermediate tensor
seq=4096 → 64 MB HBM read/write
```

### FlashAttention core idea

```
For each row q_i of Q, chunk-iterate over K/V:

Init online softmax 3 accumulators:
  m = -∞    ← max value seen so far
  l = 0     ← sum of exp seen so far
  o = 0     ← weighted sum seen so far

for K_chunk, V_chunk in chunks:
    s = q_i · K_chunkᵀ
    m_new = max(m, max(s))
    correction = exp(m - m_new)
    l = l × correction + sum(exp(s - m_new))
    o = o × correction + sum(exp(s - m_new) · V_chunk)
    m = m_new

o /= l    ← final normalization
```

### f64 accumulator for precision

```rust
let mut m_val: f64 = f64::NEG_INFINITY;
let mut l_val: f64 = 0.0;
let mut o_acc: Vec<f64> = vec![0.0f64; d_v];
o_row[d] = (o_acc[d] * inv_l) as f32;
```

### Causal mask — no matrix trick

```rust
let k_end = if is_causal { (qi + 1).min(seq_k) } else { seq_k };
for ki in 0..k_end { /* process normally */ }
// Saves another O(seq²) intermediate tensor
```

## 6.2 Operator fusion decider

```rust
pub fn should_fuse<D: Backend>(first: OpKind, second: OpKind, intermediate_numel: usize) -> FusedOp {
    // GPU: always fuse (HBM traffic is the bottleneck)
    if !D::IS_CPU { return decide_fused_op(first, second); }
    // CPU: only fuse when intermediate ≥ 1024 elements
    if intermediate_numel < 1024 { return FusedOp::NoFusion; }
    decide_fused_op(first, second)
}

fn decide_fused_op(first: OpKind, second: OpKind) -> FusedOp {
    match (first, second) {
        (OpKind::LayerNorm, OpKind::Activation(_)) => FusedOp::LnGelu,
        (OpKind::MatmulBias, OpKind::Activation(_)) => FusedOp::LinearBiasGelu,
        (OpKind::Matmul, OpKind::Activation(_))     => FusedOp::MatmulActivation,
        (OpKind::RMSNorm, OpKind::Swiglu)            => FusedOp::RmsnormSwiglu,
        _ => FusedOp::NoFusion,
    }
}
```

### OpKind enum

```rust
pub enum OpKind {
    ElwiseAdd, ElwiseSub, ElwiseMul, ElwiseDiv, ElwisePow,
    ElwiseRelu, ElwiseGelu, ElwiseGeluFast, ElwiseSilu, ElwiseSigmoid, ElwiseTanh,
    ElwiseNeg, ElwiseAbs, ElwiseExp, ElwiseLog, ElwiseSqrt, ElwiseSin, ElwiseCos,
    LayerNorm, RMSNorm, Softmax,
    Matmul, MatmulBias, FlashAttention,
    Swiglu, Activation(ActivationType),
    Reshape, Contiguous, Transpose, Permute,
}
```

---

# 7. DeviceContext Runtime

## 7.1 Design

```rust
pub trait DeviceContext: Backend {
    fn init() -> Result<DeviceHandle<Self>, DeviceError>;
    fn get() -> DeviceHandle<Self>;       // OnceLock auto-init
    fn shutdown() {}                       // optional explicit cleanup (GPU resources)
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

## 7.2 OnceLock Global Singleton

```rust
static HANDLE: OnceLock<DeviceHandle<Cpu>> = OnceLock::new();

impl DeviceContext for Cpu {
    fn get() -> DeviceHandle<Cpu> {
        HANDLE.get_or_init(|| {
            let num_cores = rayon::current_num_threads();
            DeviceHandle { properties: DeviceProperties { name: format!("CPU ({} cores)", num_cores), num_cores, .. }, inner: DeviceInner::Cpu, _device: PhantomData }
        }).clone()
    }
}
```

## 7.3 Per-Device init() Contents

| Device | init() operations | OnceLock |
|---|---|---|
| Cpu | rayon::current_num_threads() + fill DeviceProperties | ✅ |
| Cuda | cudaGetDeviceCount → cudaSetDevice(0) → cublasCreate → cublasSetStream → cudaGetDeviceProperties | ✅ (returns DeviceError::NoHardware without GPU) |
| Metal | DeviceInner::Metal(device_id=0) | ✅ |
| Wgpu | Instance::default() → Adapter::request_device() | ✅ |

## 7.4 AnyBackend Runtime Switch

```rust
static CURRENT_RUNTIME: AtomicU8 = AtomicU8::new(0);

pub enum DeviceRuntimeId { Cpu = 0, Cuda = 1, Metal = 2, Wgpu = 3 }
pub fn set_runtime_device(id: DeviceRuntimeId) { CURRENT_RUNTIME.store(id as u8, Relaxed); }
pub fn get_runtime_device() -> DeviceRuntimeId { ... }
```

---

# 8. Complete API Reference

## 8.1 One-Liner Import

```rust
use oyta_tensor::prelude::*;
```

### prelude re-export list

```rust
// From oyta-core
pub use oyta_core::{
    Tensor, DynamicShape, Strides, IndexEngine,
    Device, TensorError, Result as TensorResult,
    save_safetensors, load_safetensors, SafetensorsHeader, TensorHeader,
    f32_to_bytes, bytes_to_f32, f64_to_bytes, bytes_to_f64,
    f16_to_bytes, bytes_to_f16, bf16_to_bytes, bytes_to_bf16,
    tensor_byte_size, write_bytes, read_bytes, AlignedAlloc,
};

// From oyta-device
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

// All operators
pub use oyta_ops::*;
```

## 8.2 Tensor\<T, D\> Full API

### Construction

| Method | Signature | Notes |
|---|---|---|
| `from_data` | `(Vec<T>, shape) -> Result<Self>` | Most common constructor |
| `from_shared` | `(Arc<Vec<T>>, shape) -> Self` | Existing Arc data, zero-copy view |
| `zeros` | `(shape) -> Self` | All zeros |
| `ones` | `(shape) -> Self` | All ones |
| `full` | `(shape, value) -> Self` | Fill with value |
| `randn` | `(shape) -> Self` | N(0,1) standard normal |
| `rand` | `(shape) -> Self` | Uniform[0,1) |
| `arange` | `(start, end, step) -> Self` | Arithmetic progression |
| `scalar` | `(value) -> Self` | Scalar (rank=0) |

### Queries

| Method | Signature | Notes |
|---|---|---|
| `shape()` | `-> &DynamicShape` | Logical shape ref |
| `strides()` | `-> &Strides` | Stride ref |
| `numel()` | `-> usize` | Total elements |
| `rank()` | `-> usize` | Number of dimensions |
| `data_ptr()` | `-> *const T` | Raw pointer (includes offset) |
| `data_slice()` | `-> &[T]` | Slice reference |
| `is_contiguous()` | `-> bool` | C-contiguous check |
| `is_scalar()` | `-> bool` | rank=0 or numel=1 |
| `to_vec()` | `-> Vec<T>` | Copy out |
| `item()` | `-> T` | Extract scalar value |

### Shape Transforms (all zero-copy)

| Method | Signature | Notes |
|---|---|---|
| `reshape` | `(&[isize]) -> Result<Self>` | Supports -1 auto-inference |
| `transpose` | `(dim1: isize, dim2: isize) -> Self` | Swap two dims, supports negative indexing |
| `permute` | `(&[usize]) -> Self` | Reorder all dims |
| `unsqueeze` | `(dim: isize) -> Self` | Insert size-1 dim |
| `squeeze` | `() -> Self` | Remove all size-1 dims |
| `slice` | `(dim: isize, range: Range<usize>) -> Self` | Single-dim slice, zero-copy |
| `multi_slice` | `(&[Range<usize>]) -> Self` | Multi-dim slice |
| `select` | `(dim: usize, indices: &[usize]) -> Self` | Row/column select by indices |

### Cross-Device

| Method | Conditional |
|---|---|
| `to_device<D2>()` | Always |
| `to_cpu()` | Always |
| `to_cuda()` | `#[cfg(feature = "cuda")]` |
| `to_metal()` | `#[cfg(feature = "metal")]` |
| `to_wgpu()` | `#[cfg(feature = "wgpu")]` |

### Dtype Cast

| Source | Target | Method |
|---|---|---|
| `f32` | F16 / BF16 / f64 | `.to_f16()` / `.to_bf16()` / `.to_f64()` |
| F16 / BF16 / f64 | f32 | `.to_f32()` |

### Elementwise (auto-broadcast)

```rust
// Binary
add(&self, other: &Tensor<f32, D>) -> Result<Tensor<f32, D>>
sub, mul, div, pow

// Activations
relu, gelu, gelu_fast, sigmoid, tanh, silu

// Math unary
neg, abs, exp, log, sqrt, sin, cos
```

### Reduction

```rust
sum(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor<f32, D>>
mean, max, min
softmax(&self, dim: usize)
layernorm(&self, gamma: &Tensor<f32, D>, beta: &Tensor<f32, D>, eps: f32) -> Result<Tensor<f32, D>>
rmsnorm(&self, gamma: &Tensor<f32, D>, eps: f32) -> Result<Tensor<f32, D>>
```

### Matmul & Attention

```rust
matmul(&self, other: &Tensor<f32, D>) -> Result<Tensor<f32, D>>
matmul_t(&self, other, transpose_a, transpose_b) -> Result<Tensor<f32, D>>
flash_attention(&self, k, v, is_causal) -> Result<Tensor<f32, D>>
flash_attention_batched(&self, k, v, batch, heads, is_causal) -> Result<Tensor<f32, D>>
```

## 8.3 safetensors API

```rust
pub struct TensorHeader {
    pub dtype: String,                  // "F32" / "F16" / "BF16" / "F64"
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],         // [start, end] byte offsets in file
}

pub struct SafetensorsHeader {
    pub tensors: BTreeMap<String, TensorHeader>,
    pub metadata: Option<BTreeMap<String, String>>,
}

pub fn save_safetensors(
    tensors: &BTreeMap<String, (&[f32], DynamicShape, DType)>,
    path: &Path,
) -> Result<()>;

pub fn load_safetensors(
    path: &Path,
) -> Result<(SafetensorsHeader, Vec<(String, Vec<u8>, DType, DynamicShape)>)>;
```

### File Format Specification

```
Little-endian:
  8 bytes:  JSON header length (u64)
  N bytes:  JSON header (TensorHeader dictionary)
  M bytes:  All Tensor raw data (contiguous layout)
```

## 8.4 Byte Serialization Helpers

```rust
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

# 9. Usage Examples

## Example 1: CPU matmul + LayerNorm + FlashAttention

```rust
use oyta_tensor::prelude::*;

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32, Cpu>::randn([256, 512]);
    let b = Tensor::<f32, Cpu>::randn([512, 1024]);

    let c = a.matmul(&b)?;

    let gamma = Tensor::<f32, Cpu>::ones([1024]);
    let beta  = Tensor::<f32, Cpu>::zeros([1024]);
    let ln = c.layer_norm(&gamma, &beta, 1e-5)?;
    let out = ln.gelu()?;

    let q = Tensor::<f32, Cpu>::randn([512, 64]);
    let k = Tensor::<f32, Cpu>::randn([512, 64]);
    let v = Tensor::<f32, Cpu>::randn([512, 64]);
    let attn = q.flash_attention(&k, &v, true)?;

    let handle = Cpu::get();
    println!("CPU cores: {}", handle.properties.num_cores);

    Ok(())
}
```

## Example 2: WGPU GPU matmul + Read-back Verification

```rust
use oyta_tensor::prelude::*;

fn main() {
    let a = Tensor::<f32, Wgpu>::randn([64, 64]);
    let b = Tensor::<f32, Wgpu>::randn([64, 64]);
    let c_gpu = a.matmul(&b).unwrap();

    let c_cpu = c_gpu.to_cpu();  // Verified: max_diff = 0.00e0 on WSL2

    let handle = Wgpu::get();
    println!("WGPU device: {}", handle.properties.name);
}
```

## Example 3: Cross-Device Transfer

```rust
use oyta_tensor::prelude::*;

fn main() {
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
    let result = a.matmul(&b).unwrap();

    println!("{}", result.shape());  // [128, 512]
}
```

## Example 4: Load Pretrained Weights from safetensors

```rust
use oyta_tensor::prelude::*;

fn main() -> Result<(), TensorError> {
    let (header, tensors) = load_safetensors("bert-base.safetensors")?;

    for (name, bytes, dtype, shape) in &tensors {
        let f32_data: Vec<f32> = match dtype {
            DType::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(),
            DType::F16 => bytes.chunks_exact(2).map(|c| F16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            _ => continue,
        };
        let t = Tensor::<f32, Cpu>::from_data(f32_data, shape.clone())?;
        println!("{} -> {:?}", name, t.shape());
    }

    Ok(())
}
```

## Example 5: F16 / BF16 Precision Conversion

```rust
use oyta_tensor::prelude::*;

fn main() {
    let f32_t = Tensor::<f32, Cpu>::randn([1024, 1024]);

    let f16_t = f32_t.to_f16();     // Memory halved
    let bf16_t = f32_t.to_bf16();   // Same range as f32, lower precision

    // Accumulate trait auto-promotes: F16×F16 matmul uses f32 accumulator
    let result_f32 = f16_t.matmul(&f16_t).unwrap().to_f32();
}
```

## Example 6: AnyBackend Runtime Switch

```rust
use oyta_tensor::prelude::*;

fn main() {
    set_runtime_device(DeviceRuntimeId::Wgpu);

    let a = Tensor::<f32, Cpu>::randn([64, 64]);
    let b = Tensor::<f32, Cpu>::randn([64, 64]);
    let c = a.matmul(&b).unwrap();  // Internally dispatches to WGPU
}
```

---

# 10. Numerical Stability & Safety Guarantees

## 10.1 Stable Softmax Formula

```rust
pub fn softmax(a: *const f32, out: *mut f32, outer: usize, inner: usize) {
    // 1. Find max first → exp input always ≤ 0, no overflow
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..inner { max_val = max_val.max(unsafe { *a.add(i) }); }

    // 2. f64 accumulator (avoids precision loss for millions of elements)
    let mut sum_exp = 0.0f64;
    for i in 0..inner {
        let v = ((unsafe { *a.add(i) } - max_val) as f64).exp();
        unsafe { *out.add(i) = v as f32; }
        sum_exp += v;
    }

    // 3. Cast back to f32 at final normalization
    let inv = 1.0f64 / sum_exp;
    for i in 0..inner {
        unsafe { *out.add(i) = (*out.add(i) as f64 * inv) as f32; }
    }
}
```

## 10.2 LayerNorm / RMSNorm eps

```rust
pub const LAYERNORM_EPS: f32 = 1e-5;     // PyTorch default
pub const RMSNORM_EPS: f32 = 1e-6;       // Smaller (RMS doesn't subtract mean)
```

## 10.3 pairwise_sum (large array precision)

```rust
// Pair-wise tree-shaped summation avoids "large number swallows small number"
pub fn pairwise_sum_f64(a: &[f64]) -> f64 {
    if a.len() <= 256 { a.iter().sum() }
    else {
        let mid = a.len() / 2;
        pairwise_sum_f64(&a[..mid]) + pairwise_sum_f64(&a[mid..])
    }
}
```

## 10.4 Accumulate Trait Auto-Promotion

```rust
pub trait Accumulate {
    type Accumulator: TensorElement;
    fn to_accumulator(self) -> Self::Accumulator;
}
// F16×F16 matmul → auto f32 accumulator
// BF16×BF16 matmul → auto f32 accumulator
// f16 / bf16 reduce → auto f64 accumulator for stats
```

## 10.5 debug_assert! Boundary Coverage

All Backend ops + Tensor methods + FlashAttention + reduce/norm have debug_assert! under `#[cfg(debug_assertions)]`:

| Location | Checks |
|---|---|
| matmul | `m > 0, n > 0, k > 0, lda >= k, ldb >= n, ldc >= n` |
| elwise_* | `n > 0` |
| layernorm / rmsnorm | `outer > 0, inner > 0, eps > 0.0` |
| flash_attention | `seq_q > 0, seq_k > 0, d_k >= 16, d_v >= 16` |
| reshape | Negative index valid position |
| slice | Range within shape |
| DynamicShape::dim | Index not out of bounds |

## 10.6 Safety Audit Checklist

| Check | Result |
|---|---|
| `panic!/unimplemented!/todo!` | **0** |
| Non-`debug_assert!` `assert!` | **0** |
| `#[allow]` lint suppressions | **1** (cuda.rs gpu_ffi legitimate consolidated) |
| `unsafe {}` blocks | Only Backend raw pointers + cuda.rs FFI + cuBLAS + wgpu internals |
| `checked_mul / checked_product` | All dimension products overflow-guarded |

## 10.7 Numerical Precision Guarantees

| Operator | f32 rel error threshold | Internal precision |
|---|---|---|
| matmul | ≤ 1e-4 | f32 accumulator |
| softmax | row sum ≤ 1e-6 | f64 max + f64 sum + f64 inv_sum |
| layernorm | mean/var exact + norm error ≤ 1e-5 | f64 stat + f32 norm |
| elwise add | ≤ 1e-6 | f32 direct |
| f16 roundtrip | ≤ 2e-3 | f32 intermediate |
| bf16 roundtrip | ≤ 8e-3 | f32 intermediate |

## 10.8 Complete Test Matrix

| Layer | Count | Content |
|---|---|---|
| Backend trait signatures (9 traits × 82 methods) | Compiler type-check | All feature combos pass = signatures correct |
| Unit tests | **9** | matmul_correctness / dtype_cast_roundtrip / device_context_cpu / profiler / op_index / matmul_non_contiguous_ldc |
| Integration examples | **10** | example_01 ~ example_10 all pass |
| GPU numerical verification | **1** | example_11 → CPU vs WGPU matmul 4 sizes max_diff=0.00e0 |
| CUDA .cu real compile | **5** | nvcc 13.5 + GCC 13 → 5 .cu kernels all produce .o |
| Feature matrix | **10** | default / cuda / metal / wgpu / blas / memory_pool / debug_assertions / no-default+metal / no-default+wgpu / all-features |

---

**Document Version**: v0.1.0 · 2026-11-20
