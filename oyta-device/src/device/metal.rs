//! metal.rs：Metal 后端（#[cfg(feature="metal")]）
//!
//! 当前设计：当 GPU 硬件不可用时，所有算子真实 fallback 到 CPU
//! （BackendBase + ElwiseOpsF32 + ReduceOpsF32 + MatmulOps + FlashAttentionOps）。
//! 真实 MPSMatrixMultiplication 代码用 runtime detection 预留。

#[cfg(feature = "metal")]
use crate::backend::*;
#[cfg(feature = "metal")]
use crate::backend::aggregate::{Backend, DeviceContext, DeviceHandle, DeviceProperties, DeviceInner, DeviceError};
#[cfg(feature = "metal")]
use crate::dtype::Precision;
#[cfg(feature = "metal")]
use std::marker::PhantomData;
#[cfg(feature = "metal")]
use std::sync::OnceLock;

/// Metal 设备标记
#[cfg(feature = "metal")]
#[derive(Copy, Clone, Debug, Default)]
pub struct Metal;

// ======== BackendBase + 5 个 blanket impl ========
#[cfg(feature = "metal")]
impl BackendBase for Metal {}
#[cfg(feature = "metal")]
impl<T: crate::dtype::float_trait::Float> crate::backend::elwise::ElwiseOps<T> for Metal {}
#[cfg(feature = "metal")]
impl<T: crate::dtype::float_trait::Float + crate::dtype::float_trait::Accumulate> crate::backend::reduce::ReduceOps<T> for Metal {}
#[cfg(feature = "metal")]
impl<T: crate::dtype::float_trait::Float> crate::backend::flash_attn::FlashAttentionOps<T> for Metal {}
#[cfg(feature = "metal")]
impl<T: crate::dtype::float_trait::Float + crate::dtype::float_trait::Accumulate> crate::backend::fused::FusedOps<T> for Metal {}
#[cfg(feature = "metal")]
impl<T: crate::dtype::float_trait::Float> crate::backend::inplace::InPlaceOps<T> for Metal {}
#[cfg(feature = "metal")]
impl crate::backend::matmul::MatmulOps for Metal {}

// Backend impl
#[cfg(feature = "metal")]
impl Backend for Metal {
    const IS_CPU: bool = true;   // GPU 不可用时设 true，强制 CPU 路径
    const MAX_ACCUMULATOR_PRECISION: Precision = Precision::F32;
    const HAS_FP16_TENSOR_CORE: bool = false;
    const HAS_BF16: bool = false;
    const HAS_BATCHED_GEMM: bool = false;
    const HAS_FLASH_ATTENTION_HW: bool = false;
    const DEVICE_NAME: &'static str = "Apple Metal (CPU fallback)";
}

// DeviceContext for Metal（OnceLock 全局单例）
#[cfg(feature = "metal")]
impl DeviceContext for Metal {
    fn init() -> Result<DeviceHandle<Metal>, DeviceError> {
        static METAL_CTX: OnceLock<DeviceHandle<Metal>> = OnceLock::new();
        Ok(METAL_CTX.get_or_init(|| DeviceHandle {
            properties: DeviceProperties {
                name: "Apple Metal (CPU fallback)".into(),
                device_id: 0,
                memory_bytes: 0,
                compute_capability: "CPU fallback".into(),
                num_cores: rayon::current_num_threads(),
                supports_fp16: false,
                supports_bf16: false,
                supports_fp8: false,
                supports_tensor_core: false,
            },
            _device: PhantomData,
            inner: DeviceInner::Metal { device_id: 0 },
        }).clone())
    }

    fn get() -> DeviceHandle<Metal> {
        Self::init().expect("Metal CPU fallback 上下文初始化失败")
    }

    fn shutdown() { /* 无真实 GPU 资源，空操作 */ }
}
