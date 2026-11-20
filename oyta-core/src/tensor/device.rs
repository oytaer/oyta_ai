//! tensor_device.rs：跨设备传输（to_device / to_cpu / to_cuda / to_metal / to_wgpu）
use super::Tensor;
use oyta_device::Backend;

impl<T: Clone + 'static, D1: Backend> Tensor<T, D1> {
    /// 跨设备传输：返回目标设备 D2 上的新 Tensor
    pub fn to_device<D2: Backend>(&self) -> Tensor<T, D2> {
        Tensor::<T, D2>::from_data(self.data_slice().to_vec(), self.shape.clone()).expect("跨设备传输失败")
    }
    /// 便捷：转到 Cpu
    pub fn to_cpu(&self) -> Tensor<T, oyta_device::Cpu> { self.to_device::<oyta_device::Cpu>() }
    /// 便捷：转到 Cuda（feature gate）
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self) -> Tensor<T, oyta_device::Cuda> { self.to_device::<oyta_device::Cuda>() }
    /// 便捷：转到 Metal（feature gate）
    #[cfg(feature = "metal")]
    pub fn to_metal(&self) -> Tensor<T, oyta_device::Metal> { self.to_device::<oyta_device::Metal>() }
    /// 便捷：转到 Wgpu（feature gate）
    #[cfg(feature = "wgpu")]
    pub fn to_wgpu(&self) -> Tensor<T, oyta_device::Wgpu> { self.to_device::<oyta_device::Wgpu>() }
}
