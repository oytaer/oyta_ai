//! tensor_dtype.rs：Dtype Cast（f32 ↔ f16/bf16/f64）
use super::Tensor;
use oyta_device::{Backend, F16, BF16};

impl<D: Backend> Tensor<f32, D> {
    /// f32 → f16（逐元素 cast）
    pub fn to_f16(&self) -> Tensor<F16, D> {
        let data: Vec<F16> = self.data_slice().iter().map(|&v| F16::from_f32(v)).collect();
        Tensor::from_data(data, self.shape.clone()).expect("f32→f16 cast 失败")
    }
    /// f32 → bf16
    pub fn to_bf16(&self) -> Tensor<BF16, D> {
        let data: Vec<BF16> = self.data_slice().iter().map(|&v| BF16::from_f32(v)).collect();
        Tensor::from_data(data, self.shape.clone()).expect("f32→bf16 cast 失败")
    }
    /// f32 → f64
    pub fn to_f64(&self) -> Tensor<f64, D> {
        let data: Vec<f64> = self.data_slice().iter().map(|&v| v as f64).collect();
        Tensor::from_data(data, self.shape.clone()).expect("f32→f64 cast 失败")
    }
}

impl<D: Backend> Tensor<F16, D> {
    /// f16 → f32（cast 后算子内部用 f32 计算）
    pub fn to_f32(&self) -> Tensor<f32, D> {
        let data: Vec<f32> = self.data_slice().iter().map(|&v| v.to_f32()).collect();
        Tensor::from_data(data, self.shape.clone()).expect("f16→f32 cast 失败")
    }
}

impl<D: Backend> Tensor<BF16, D> {
    /// bf16 → f32
    pub fn to_f32(&self) -> Tensor<f32, D> {
        let data: Vec<f32> = self.data_slice().iter().map(|&v| v.to_f32()).collect();
        Tensor::from_data(data, self.shape.clone()).expect("bf16→f32 cast 失败")
    }
}

impl<D: Backend> Tensor<f64, D> {
    /// f64 → f32
    pub fn to_f32(&self) -> Tensor<f32, D> {
        let data: Vec<f32> = self.data_slice().iter().map(|&v| v as f32).collect();
        Tensor::from_data(data, self.shape.clone()).expect("f64→f32 cast 失败")
    }
}
