//! any_backend.rs：运行时分发（dispatch! 宏 + AnyBackend 伪设备）
//!
//! 泛型 trait（ElwiseOps/ReduceOps/FlashAttentionOps/FusedOps/InPlaceOps）
//! 用 blanket impl 继承默认实现（rayon + Float cast），
//! 只有 BackendBase / MatmulOps 保留 dispatch（GPU 后端对这两个有硬件重写）

use crate::dtype::{F16, BF16, Precision};
use crate::dtype::float_trait::{Float, Accumulate};
use crate::backend::base::BackendBase;
use crate::backend::elwise::ElwiseOps;
use crate::backend::reduce::ReduceOps;
use crate::backend::matmul::MatmulOps;
use crate::backend::flash_attn::FlashAttentionOps;
use crate::backend::fused::FusedOps;
use crate::backend::inplace::InPlaceOps;
use crate::backend::aggregate::Backend;
use crate::Cpu;
#[cfg(feature = "cuda")]
use crate::Cuda;
#[cfg(feature = "metal")]
use crate::Metal;
#[cfg(feature = "wgpu")]
use crate::Wgpu;

// ============================================================
// dispatch! 宏：编译期展开成 match，把调用转发到实际设备
// ============================================================

macro_rules! dispatch {
    ($trait:path, $op:ident( $($arg:expr),* )) => {
        match $crate::backend::aggregate::get_runtime_device() {
            $crate::backend::aggregate::DeviceRuntimeId::Cpu => { <Cpu as $trait>::$op($($arg),*) }
            #[cfg(feature = "cuda")]
            $crate::backend::aggregate::DeviceRuntimeId::Cuda => { <Cuda as $trait>::$op($($arg),*) }
            #[cfg(not(feature = "cuda"))]
            $crate::backend::aggregate::DeviceRuntimeId::Cuda => unreachable!("cuda feature 未启用"),
            #[cfg(feature = "metal")]
            $crate::backend::aggregate::DeviceRuntimeId::Metal => { <Metal as $trait>::$op($($arg),*) }
            #[cfg(not(feature = "metal"))]
            $crate::backend::aggregate::DeviceRuntimeId::Metal => unreachable!("metal feature 未启用"),
            #[cfg(feature = "wgpu")]
            $crate::backend::aggregate::DeviceRuntimeId::Wgpu => { <Wgpu as $trait>::$op($($arg),*) }
            #[cfg(not(feature = "wgpu"))]
            $crate::backend::aggregate::DeviceRuntimeId::Wgpu => unreachable!("wgpu feature 未启用"),
        }
    };
}

// ============================================================
// AnyBackend：运行时分发的伪设备
// ============================================================

#[derive(Copy, Clone, Debug, Default)]
pub struct AnyBackend;

// BackendBase：dispatch（有硬件特定实现）
impl BackendBase for AnyBackend {
    fn fill(dst: *mut f32, n: usize, value: f32) { dispatch!(BackendBase, fill(dst, n, value)); }
    fn copy(src: *const f32, dst: *mut f32, n: usize) { dispatch!(BackendBase, copy(src, dst, n)); }
    fn where_bool(cond: *const bool, a: *const f32, b: *const f32, out: *mut f32, n: usize) {
        dispatch!(BackendBase, where_bool(cond, a, b, out, n));
    }
}

// ======== 泛型 blanket impl：继承默认 rayon + Float cast 实现 ========
// GPU 后端对这些没有硬件重写（只有 MatmulOps 有），所以默认实现等于 dispatch 到 CPU
impl<T: Float> ElwiseOps<T> for AnyBackend {}
impl<T: Float + Accumulate> ReduceOps<T> for AnyBackend {}
impl<T: Float> FlashAttentionOps<T> for AnyBackend {}
impl<T: Float + Accumulate> FusedOps<T> for AnyBackend {}
impl<T: Float> InPlaceOps<T> for AnyBackend {}

// MatmulOps：dispatch（有硬件重写 cublasSgemm / wgsl matmul shader）
impl MatmulOps for AnyBackend {
    fn matmul(a: *const f32, lda: usize, b: *const f32, ldb: usize,
              c: *mut f32, ldc: usize, m: usize, n: usize, k: usize,
              alpha: f32, beta: f32) {
        dispatch!(MatmulOps, matmul(a, lda, b, ldb, c, ldc, m, n, k, alpha, beta));
    }
    fn matmul_batched(a: *const f32, lda: usize, stride_a: isize,
                      b: *const f32, ldb: usize, stride_b: isize,
                      c: *mut f32, ldc: usize, stride_c: isize,
                      m: usize, n: usize, k: usize, batch: usize,
                      alpha: f32, beta: f32) {
        dispatch!(MatmulOps, matmul_batched(
            a, lda, stride_a, b, ldb, stride_b, c, ldc, stride_c, m, n, k, batch, alpha, beta));
    }
    fn matmul_t(a: *const f32, lda: usize, transpose_a: bool,
                b: *const f32, ldb: usize, transpose_b: bool,
                c: *mut f32, ldc: usize, m: usize, n: usize, k: usize,
                alpha: f32, beta: f32) {
        dispatch!(MatmulOps, matmul_t(
            a, lda, transpose_a, b, ldb, transpose_b, c, ldc, m, n, k, alpha, beta));
    }
    fn matmul_f16(a: *const F16, lda: usize, b: *const F16, ldb: usize,
                  c: *mut F16, ldc: usize, m: usize, n: usize, k: usize,
                  alpha: f32, beta: f32) {
        dispatch!(MatmulOps, matmul_f16(a, lda, b, ldb, c, ldc, m, n, k, alpha, beta));
    }
    fn matmul_bf16(a: *const BF16, lda: usize, b: *const BF16, ldb: usize,
                   c: *mut BF16, ldc: usize, m: usize, n: usize, k: usize,
                   alpha: f32, beta: f32) {
        dispatch!(MatmulOps, matmul_bf16(a, lda, b, ldb, c, ldc, m, n, k, alpha, beta));
    }
}

// Backend 聚合 impl for AnyBackend
// 上面 blanket impl（5 个泛型 trait + MatmulOps + BackendBase）已满足 Backend bound 所有要求
impl Backend for AnyBackend {
    const IS_CPU: bool = false;
    const MAX_ACCUMULATOR_PRECISION: Precision = Precision::F32;
    const HAS_FP16_TENSOR_CORE: bool = false;
    const HAS_BF16: bool = false;
    const HAS_BATCHED_GEMM: bool = false;
    const HAS_FLASH_ATTENTION_HW: bool = false;
    const DEVICE_NAME: &'static str = "AnyBackend";
}
