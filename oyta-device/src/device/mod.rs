//! device 子域：设备标记域

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use cpu::Cpu;
#[cfg(feature = "cuda")]
pub use cuda::Cuda;
#[cfg(feature = "metal")]
pub use metal::Metal;
#[cfg(feature = "wgpu")]
pub use wgpu::Wgpu;
