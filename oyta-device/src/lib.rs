//! oyta-device：设备抽象层
//!
//! 职责：
//!   1. Backend trait（9 子 trait 聚合 + 编译期设备属性）
//!   2. Cpu / Cuda / Metal / Wgpu 设备类型标记
//!   3. DeviceContext（设备运行时上下文 + DeviceHandle + DeviceProperties）
//!   4. AnyBackend（运行时分发 + dispatch! 宏）
//!
//! 已按业务域拆分为 dtype / backend / device 三个子目录。

// 子目录声明
pub mod dtype;
pub mod backend;
pub mod device;
pub mod checked;

// 内存池（feature gate）
#[cfg(feature = "memory_pool")]
pub mod memory_pool;
#[cfg(feature = "memory_pool")]
pub use memory_pool::DeviceMemoryPool;

// --- 数据类型 re-export ---
pub use dtype::{F16, BF16, Float, Accumulate, DType, Precision};

// --- 所有子 trait re-export（泛型版本一套覆盖所有 dtype）---
pub use backend::base::BackendBase;
pub use backend::elwise::ElwiseOps;
pub use backend::reduce::ReduceOps;
pub use backend::matmul::MatmulOps;
pub use backend::flash_attn::FlashAttentionOps;
pub use backend::fused::FusedOps;
pub use backend::inplace::InPlaceOps;

// --- Backend 聚合 + Device 相关 ---
pub use backend::aggregate::{
    Backend, DeviceContext, DeviceHandle, DeviceProperties, DeviceInner,
    DeviceRuntimeId, OpKind, ActivationType, DeviceError,
    broadcast_strides, set_runtime_device, get_runtime_device,
};

// --- 设备类型标记 ---
pub use device::Cpu;
#[cfg(feature = "cuda")]
pub use device::Cuda;
#[cfg(feature = "metal")]
pub use device::Metal;
#[cfg(feature = "wgpu")]
pub use device::Wgpu;

// --- 运行时分发 ---
pub use backend::any_backend::AnyBackend;

// --- 常量 + 辅助函数重导出 ---
pub use backend::reduce::{LAYERNORM_EPS, RMSNORM_EPS, pairwise_sum};
pub use backend::matmul::{STRASSEN_CUTOFF, matmul_strassen};
pub use backend::elwise::error_function;
