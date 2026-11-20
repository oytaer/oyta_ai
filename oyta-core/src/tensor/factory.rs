//! tensor_factory.rs：静态构造工厂（zeros / ones / full / scalar / arange / rand / randn）
use super::Tensor;
use crate::shape::DynamicShape;
use oyta_device::Backend;

impl<D: Backend> Tensor<f32, D> {
    /// 全零 Tensor
    pub fn zeros(shape: impl IntoIterator<Item = usize>) -> Self {
        let dims: Vec<usize> = shape.into_iter().collect();
        let n: usize = crate::checked::checked_product(dims.iter().copied())
            .unwrap_or(usize::MAX);
        Tensor::from_data(vec![0.0f32; n], DynamicShape::new(dims)).expect("zeros 构造失败")
    }

    /// 全一 Tensor
    pub fn ones(shape: impl IntoIterator<Item = usize>) -> Self {
        let dims: Vec<usize> = shape.into_iter().collect();
        let n: usize = crate::checked::checked_product(dims.iter().copied())
            .unwrap_or(usize::MAX);
        Tensor::from_data(vec![1.0f32; n], DynamicShape::new(dims)).expect("ones 构造失败")
    }

    /// 标准正态分布 Tensor（rand + rand_distr）
    pub fn randn(shape: impl IntoIterator<Item = usize>) -> Self {
        let dims: Vec<usize> = shape.into_iter().collect();
        let n: usize = crate::checked::checked_product(dims.iter().copied())
            .unwrap_or(usize::MAX);
        use rand::RngExt; // rand 0.9 的 Rng 扩展 trait
        use rand_distr::StandardNormal; // 标准正态分布
        let rng = rand::rng(); // 线程安全默认 RNG
        let data: Vec<f32> = rng.sample_iter::<f64, _>(StandardNormal).take(n).map(|v| v as f32).collect();
        Tensor::from_data(data, DynamicShape::new(dims)).expect("randn 构造失败")
    }

    /// 全填值（所有元素 = value）
    pub fn full(shape: impl IntoIterator<Item = usize>, value: f32) -> Self {
        let dims: Vec<usize> = shape.into_iter().collect();
        let n: usize = crate::checked::checked_product(dims.iter().copied())
            .unwrap_or(usize::MAX);
        Tensor::from_data(vec![value; n], DynamicShape::new(dims)).expect("full 构造失败")
    }

    /// 等差数列：[start, start+step, start+2*step, ..., 长度 = 到 end 的步数]
    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        #[cfg(debug_assertions)]
        debug_assert!(step != 0.0, "arange step 不能为 0");
        let count = ((end - start) / step).ceil() as usize;
        let data: Vec<f32> = (0..count).map(|i| start + i as f32 * step).collect();
        Tensor::from_data(data, DynamicShape::new([count])).expect("arange 构造失败")
    }

    /// 均匀分布 [0, 1) 随机值
    pub fn rand(shape: impl IntoIterator<Item = usize>) -> Self {
        let dims: Vec<usize> = shape.into_iter().collect();
        let n: usize = crate::checked::checked_product(dims.iter().copied())
            .unwrap_or(usize::MAX);
        use rand::RngExt;
        use rand_distr::StandardUniform; // rand_distr 提供的 [0,1) 均匀分布
        let rng = rand::rng();
        let data: Vec<f32> = rng.sample_iter::<f32, _>(StandardUniform).take(n).collect();
        Tensor::from_data(data, DynamicShape::new(dims)).expect("rand 构造失败")
    }

    /// 标量 Tensor（rank=0，numel=1）
    pub fn scalar(value: f32) -> Self {
        Tensor::from_data(vec![value], DynamicShape::new([])).expect("scalar 构造失败")
    }
}
