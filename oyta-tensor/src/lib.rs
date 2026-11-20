//! oyta-tensor：对外统一入口
//!
//! 显式 re-export，避免 oyta_core 和 oyta_device 都导出的 `dtype` / `device` 模块冲突。

// ============================================================
// 从 oyta-core re-export
// ============================================================
pub use oyta_core::{
    // Tensor 核心类型
    Tensor, DynamicShape, Strides, IndexEngine,
    Device, TensorError,
    Result as TensorResult,
    // 安全张量创建/加载
    save_safetensors, load_safetensors, SafetensorsHeader, TensorHeader,
    // bytes 序列化
    f32_to_bytes, bytes_to_f32, f64_to_bytes, bytes_to_f64,
    f16_to_bytes, bytes_to_f16, bf16_to_bytes, bytes_to_bf16,
    tensor_byte_size, write_bytes, read_bytes,
    // 对齐分配器
    AlignedAlloc,
    // 模块别名（与 oyta_device 冲突的 dtype/device 用别名）
    device as core_device_module,
};

// ============================================================
// 从 oyta-device re-export
// ============================================================
pub use oyta_device::{
    // Backend trait（聚合 + 子 trait）
    Backend, BackendBase,
    ElwiseOps,
    ReduceOps, MatmulOps, FlashAttentionOps,
    FusedOps, InPlaceOps,
    // CPU 设备（始终可用）
    Cpu,
    // 设备运行时上下文
    DeviceContext, DeviceHandle, DeviceProperties, DeviceInner,
    // AnyBackend 运行时分发
    AnyBackend,
    // 设备运行时标识 + 切换
    DeviceRuntimeId,
    set_runtime_device, get_runtime_device,
    // OpKind / ActivationType / DeviceError
    OpKind, ActivationType, DeviceError,
    // 数据类型（从 oyta_device 来）
    F16, BF16, DType, Float, Accumulate, Precision,
    // 常量
    LAYERNORM_EPS, RMSNORM_EPS,
    STRASSEN_CUTOFF, matmul_strassen,
    error_function,
    broadcast_strides,
};

// GPU 设备类型（feature-gated，独立 re-export）
#[cfg(feature = "cuda")]
pub use oyta_device::Cuda;
#[cfg(feature = "metal")]
pub use oyta_device::Metal;
#[cfg(feature = "wgpu")]
pub use oyta_device::Wgpu;

// ============================================================
// 从 oyta-ops glob re-export（oyta-ops 不和前两个 crate 冲突）
// ============================================================
pub use oyta_ops::*;

pub mod prelude;
