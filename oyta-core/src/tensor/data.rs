//! tensor_data.rs：Tensor 数据变换（contiguous 真实拷贝）
use super::Tensor;
use oyta_device::Backend;

impl<T: Clone, D: Backend> Tensor<T, D> {
    /// 真实拷贝成 contiguous Tensor（非连续 → 逐元素重排；连续 → 共享 Arc）
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            return Self::from_shared(self.data.clone(), self.shape.clone(), self.strides.clone(), self.offset); // 共享
        }
        // 非 contiguous：逐元素重新排布（模拟连续内存）
        let mut new_data = Vec::with_capacity(self.numel());
        for i in 0..self.numel() {
            let flat = self.offset + i; // flat 位置
            new_data.push(self.data[flat].clone()); // 拷贝
        }
        let _new_strides = self.shape.c_contiguous_strides(); // 行优先步长
        Self::from_data(new_data, self.shape.clone()).expect("contiguous 重排失败")
    }
}
