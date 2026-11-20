//! device.rs：Device trait（零成本类型标记）
//!
//! §3.1 目录结构 → device.rs
//! §6.1 Device trait 完整实现
//!
//! 设计：Device 是编译期类型标记，不携带运行时状态。
//! Tensor<T, D> 中的 D 只告诉编译器选哪个设备算子实现。
//! Backend trait 在 oyta-device 里定义（extends Device）。

/// 设备零成本类型标记 trait
///
/// 所有设备类型（Cpu/Cuda/Metal/Wgpu）都实现这个 trait。
/// 运行时 Tensor 本身不携带设备信息——data 始终在当前进程内存中。
/// GPU Backend 内部用 cudaMemcpy 做 H2D/D2H 搬运。
pub trait Device: Copy + Clone + Send + Sync + 'static {
    /// 设备名称（字符串常量）
    fn name() -> &'static str;
}

/// CPU 设备标记（默认实现）
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Cpu;

impl Device for Cpu {
    fn name() -> &'static str { "CPU" }
}

/// CUDA 设备标记（feature gate）
#[cfg(feature = "cuda")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Cuda;

#[cfg(feature = "cuda")]
impl Device for Cuda {
    fn name() -> &'static str { "CUDA" }
}

/// Metal 设备标记（feature gate）
#[cfg(feature = "metal")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Metal;

#[cfg(feature = "metal")]
impl Device for Metal {
    fn name() -> &'static str { "Metal" }
}

/// Wgpu 设备标记（feature gate）
#[cfg(feature = "wgpu")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Wgpu;

#[cfg(feature = "wgpu")]
impl Device for Wgpu {
    fn name() -> &'static str { "WGPU" }
}
