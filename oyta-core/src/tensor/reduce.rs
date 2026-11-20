//! tensor_reduce.rs：Reduce 算子（sum / mean / max / min / prod）
use super::Tensor;
use crate::shape::DynamicShape;
use crate::error::TensorError;
use oyta_device::{Backend, Float};

impl<T: Float + Clone, D: Backend> Tensor<T, D> {
    /// 所有元素求和（reduce to scalar）
    pub fn sum(&self) -> Result<T, TensorError> {
        let a = self.contiguous();
        let mut sum_val: f32 = 0.0;
        for &v in a.data_slice() { sum_val += v.to_f32(); }
        Ok(T::from_f32(sum_val))
    }

    // -- reduce 算子（mean / max / min / prod 沿 axis=None 全局归约 或 axis=Some(i) 指定维归约） --

    /// 均值（axis=None → 全局均值；axis=Some(i) → 沿指定维）
    pub fn mean(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        let a = self.contiguous();
        let (outer, inner) = match axis {
            None => (1usize, self.numel()),
            Some(d) => { let s = self.shape.dims[d]; (self.numel() / s, s) }
        };
        let mut out_f = vec![0.0f32; outer];
        D::reduce_mean(a.data_slice().as_ptr() as *const f32, out_f.as_mut_ptr(), outer, inner);
        let out_data: Vec<T> = out_f.into_iter().map(T::from_f32).collect();
        let out_shape = match axis {
            None => DynamicShape::new([]),
            Some(d) => {
                let mut nd: Vec<usize> = self.shape.dims.iter().enumerate().filter(|(i, _)| *i != d).map(|(_, &s)| s).collect();
                if nd.is_empty() { nd.push(1); }
                DynamicShape::new(nd)
            }
        };
        Tensor::from_data(out_data, out_shape)
    }

    /// 最大值（axis=None → 全局 max；axis=Some(i) → 沿指定维）
    pub fn max(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        let a = self.contiguous();
        let (outer, inner) = match axis {
            None => (1usize, self.numel()),
            Some(d) => { let s = self.shape.dims[d]; (self.numel() / s, s) }
        };
        let mut out_f = vec![0.0f32; outer];
        D::reduce_max(a.data_slice().as_ptr() as *const f32, out_f.as_mut_ptr(), outer, inner);
        let out_data: Vec<T> = out_f.into_iter().map(T::from_f32).collect();
        let out_shape = match axis {
            None => DynamicShape::new([]),
            Some(d) => {
                let mut nd: Vec<usize> = self.shape.dims.iter().enumerate().filter(|(i, _)| *i != d).map(|(_, &s)| s).collect();
                if nd.is_empty() { nd.push(1); }
                DynamicShape::new(nd)
            }
        };
        Tensor::from_data(out_data, out_shape)
    }

    /// 最小值（axis=None → 全局 min；axis=Some(i) → 沿指定维）
    pub fn min(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        let a = self.contiguous();
        let (outer, inner) = match axis {
            None => (1usize, self.numel()),
            Some(d) => { let s = self.shape.dims[d]; (self.numel() / s, s) }
        };
        let mut out_f = vec![0.0f32; outer];
        D::reduce_min(a.data_slice().as_ptr() as *const f32, out_f.as_mut_ptr(), outer, inner);
        let out_data: Vec<T> = out_f.into_iter().map(T::from_f32).collect();
        let out_shape = match axis {
            None => DynamicShape::new([]),
            Some(d) => {
                let mut nd: Vec<usize> = self.shape.dims.iter().enumerate().filter(|(i, _)| *i != d).map(|(_, &s)| s).collect();
                if nd.is_empty() { nd.push(1); }
                DynamicShape::new(nd)
            }
        };
        Tensor::from_data(out_data, out_shape)
    }

    /// 乘积（axis=None → 全局 prod；axis=Some(i) → 沿指定维）
    pub fn prod(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        let a = self.contiguous();
        let (outer, inner) = match axis {
            None => (1usize, self.numel()),
            Some(d) => { let s = self.shape.dims[d]; (self.numel() / s, s) }
        };
        let a_s = a.data_slice();
        let mut out_f = vec![1.0f32; outer];
        for oi in 0..outer {
            let base = oi * inner;
            let mut p = 1.0f32;
            for i in 0..inner { p = p * a_s[base + i].to_f32(); }
            out_f[oi] = p;
        }
        let out_data: Vec<T> = out_f.into_iter().map(T::from_f32).collect();
        let out_shape = match axis {
            None => DynamicShape::new([]),
            Some(d) => {
                let mut nd: Vec<usize> = self.shape.dims.iter().enumerate().filter(|(i, _)| *i != d).map(|(_, &s)| s).collect();
                if nd.is_empty() { nd.push(1); }
                DynamicShape::new(nd)
            }
        };
        Tensor::from_data(out_data, out_shape)
    }
}
