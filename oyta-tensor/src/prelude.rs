//! 便捷导入：use oyta_tensor::prelude::*;
//!
//! 和 lib.rs 一样用显式 re-export，避免 glob 冲突。
pub use oyta_core::{
    Tensor, DynamicShape, Strides, IndexEngine,
    Device, TensorError,
    Result as TensorResult,
    save_safetensors, load_safetensors, SafetensorsHeader, TensorHeader,
    f32_to_bytes, bytes_to_f32, f64_to_bytes, bytes_to_f64,
    f16_to_bytes, bytes_to_f16, bf16_to_bytes, bytes_to_bf16,
    tensor_byte_size, write_bytes, read_bytes,
    AlignedAlloc,
    device as core_device_module,
};

pub use oyta_device::{
    Backend, BackendBase,
    ElwiseOps,
    ReduceOps, MatmulOps, FlashAttentionOps,
    FusedOps, InPlaceOps,
    Cpu,
    DeviceContext, DeviceHandle, DeviceProperties, DeviceInner,
    AnyBackend,
    DeviceRuntimeId,
    set_runtime_device, get_runtime_device,
    OpKind, ActivationType, DeviceError,
    F16, BF16, DType, Float, Accumulate, Precision,
    LAYERNORM_EPS, RMSNORM_EPS,
    STRASSEN_CUTOFF, matmul_strassen,
    error_function,
    broadcast_strides,
};

#[cfg(feature = "cuda")]
pub use oyta_device::Cuda;
#[cfg(feature = "metal")]
pub use oyta_device::Metal;
#[cfg(feature = "wgpu")]
pub use oyta_device::Wgpu;

pub use oyta_ops::*;
