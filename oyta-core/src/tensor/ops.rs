//! tensor_ops.rs：二元算子 + 一元激活 + Norm（Float trait → Backend 调用）
use super::Tensor;
use crate::shape::DynamicShape;
use crate::error::TensorError;
use oyta_device::{Backend, Float};

impl<T: Float + Clone, D: Backend> Tensor<T, D> {
    /// 逐元素加（a + b）
    pub fn add(&self, other: &Tensor<T, D>) -> Result<Tensor<T, D>, TensorError> {
        if self.shape != other.shape { return Err(Self::shape_err(&self.shape.dims, &other.shape.dims)); }
        let a = self.contiguous(); let b = other.contiguous();
        let mut out_data = vec![T::zero(); self.numel()];
        D::elwise_add(a.data_slice().as_ptr() as *const f32, b.data_slice().as_ptr() as *const f32,
                      out_data.as_mut_ptr() as *mut f32, self.numel());
        Tensor::from_data(out_data, self.shape.clone())
    }

    /// 逐元素乘（a * b）
    pub fn mul(&self, other: &Tensor<T, D>) -> Result<Tensor<T, D>, TensorError> {
        if self.shape != other.shape { return Err(Self::shape_err(&self.shape.dims, &other.shape.dims)); }
        let a = self.contiguous(); let b = other.contiguous();
        let mut out_data = vec![T::zero(); self.numel()];
        D::elwise_mul(a.data_slice().as_ptr() as *const f32, b.data_slice().as_ptr() as *const f32,
                      out_data.as_mut_ptr() as *mut f32, self.numel());
        Tensor::from_data(out_data, self.shape.clone())
    }

    /// 逐元素减（a - b）
    pub fn sub(&self, other: &Tensor<T, D>) -> Result<Tensor<T, D>, TensorError> {
        if self.shape != other.shape { return Err(Self::shape_err(&self.shape.dims, &other.shape.dims)); }
        let a = self.contiguous(); let b = other.contiguous();
        let mut out_data = vec![T::zero(); self.numel()];
        D::elwise_sub(a.data_slice().as_ptr() as *const f32, b.data_slice().as_ptr() as *const f32,
                      out_data.as_mut_ptr() as *mut f32, self.numel());
        Tensor::from_data(out_data, self.shape.clone())
    }

    /// 逐元素除（a / b）
    pub fn div(&self, other: &Tensor<T, D>) -> Result<Tensor<T, D>, TensorError> {
        if self.shape != other.shape { return Err(Self::shape_err(&self.shape.dims, &other.shape.dims)); }
        let a = self.contiguous(); let b = other.contiguous();
        let mut out_data = vec![T::zero(); self.numel()];
        D::elwise_div(a.data_slice().as_ptr() as *const f32, b.data_slice().as_ptr() as *const f32,
                      out_data.as_mut_ptr() as *mut f32, self.numel());
        Tensor::from_data(out_data, self.shape.clone())
    }

    /// 矩阵乘法（支持 batched matmul）
    pub fn matmul(&self, other: &Tensor<T, D>) -> Result<Tensor<T, D>, TensorError> {
        let rank_a = self.rank(); let rank_b = other.rank();
        #[cfg(debug_assertions)]
        debug_assert!(rank_a >= 2 && rank_b >= 2, "matmul 需要 rank >= 2：A.rank={}, B.rank={}", rank_a, rank_b);
        if rank_a < 2 || rank_b < 2 { return Err(TensorError::Unsupported("matmul 需要 rank >= 2".into())); }
        let m = self.shape.dims[rank_a - 2]; let k = self.shape.dims[rank_a - 1]; let n = other.shape.dims[rank_b - 1];
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            other.shape.dims[rank_b - 2], k,
            "matmul K 维度不匹配：A.cols={}, B.rows={}", k, other.shape.dims[rank_b - 2],
        );
        if other.shape.dims[rank_b - 2] != k {
            return Err(TensorError::ShapeMismatch { expected: vec![k], actual: vec![other.shape.dims[rank_b - 2]] });
        }
        let a = self.contiguous(); let b = other.contiguous();
        let mut out_data = vec![T::zero(); m * n];
        D::matmul(a.data_slice().as_ptr() as *const f32, k, b.data_slice().as_ptr() as *const f32, n,
                  out_data.as_mut_ptr() as *mut f32, n, m, n, k, 1.0, 0.0);
        let mut out_shape = if rank_a >= rank_b { self.shape.dims[..rank_a - 2].to_vec() } else { other.shape.dims[..rank_b - 2].to_vec() };
        out_shape.push(m); out_shape.push(n);
        Tensor::from_data(out_data, DynamicShape::new(out_shape))
    }

    /// Softmax（沿指定 axis）
    pub fn softmax(&self, axis: isize) -> Result<Tensor<T, D>, TensorError> {
        let rank = self.rank();
        let axis = if axis < 0 { (rank as isize + axis) as usize } else { axis as usize };
        let inner = self.shape.dims[axis]; let outer = self.numel() / inner;
        let tensor = self.contiguous();
        let mut out_data = vec![T::zero(); self.numel()];
        D::softmax(tensor.data_slice().as_ptr() as *const f32, out_data.as_mut_ptr() as *mut f32, outer, inner);
        Tensor::from_data(out_data, self.shape.clone())
    }

    /// LayerNorm（最后一维归一化 + gamma/beta 仿射）
    pub fn layernorm(&self, gamma: &Tensor<T, D>, beta: &Tensor<T, D>, eps: f32) -> Result<Tensor<T, D>, TensorError> {
        let inner = self.shape.dims[self.rank() - 1]; let outer = self.numel() / inner;
        let a = self.contiguous(); let g = gamma.contiguous(); let b = beta.contiguous();
        let mut out_data = vec![T::zero(); self.numel()];
        D::layernorm(a.data_slice().as_ptr() as *const f32, out_data.as_mut_ptr() as *mut f32,
                     g.data_slice().as_ptr() as *const f32, b.data_slice().as_ptr() as *const f32,
                     outer, inner, eps);
        Tensor::from_data(out_data, self.shape.clone())
    }

    /// GELU 激活（Gaussian Error Linear Unit）
    pub fn gelu(&self) -> Result<Tensor<T, D>, TensorError> {
        let a = self.contiguous();
        let mut out_data = vec![T::zero(); self.numel()];
        D::elwise_gelu(a.data_slice().as_ptr() as *const f32, out_data.as_mut_ptr() as *mut f32, self.numel());
        Tensor::from_data(out_data, self.shape.clone())
    }

    /// ReLU 激活（max(0, x)）
    pub fn relu(&self) -> Result<Tensor<T, D>, TensorError> {
        let a = self.contiguous();
        let mut out_data = vec![T::zero(); self.numel()];
        D::elwise_relu(a.data_slice().as_ptr() as *const f32, out_data.as_mut_ptr() as *mut f32, self.numel());
        Tensor::from_data(out_data, self.shape.clone())
    }

    // -- 一元算子通用辅助：contiguous → Backend kernel → from_data --
    fn unary(&self, backend_fn: unsafe fn(*const f32, *mut f32, usize)) -> Self {
        let a = self.contiguous();
        let mut out_data = vec![T::zero(); self.numel()];
        unsafe {
            backend_fn(
                a.data_slice().as_ptr() as *const f32,
                out_data.as_mut_ptr() as *mut f32,
                self.numel(),
            );
        }
        Tensor::from_data(out_data, self.shape.clone()).expect("unary 算子输出构造失败")
    }

    // -- 补充一元激活：neg / abs / exp / log / sqrt / sigmoid / tanh / silu / sin / cos --

    /// 逐元素取负
    pub fn neg(&self) -> Self { self.unary(D::elwise_neg) }
    /// 逐元素绝对值
    pub fn abs(&self) -> Self { self.unary(D::elwise_abs) }
    /// 逐元素 exp
    pub fn exp(&self) -> Self { self.unary(D::elwise_exp) }
    /// 逐元素 log
    pub fn log(&self) -> Self { self.unary(D::elwise_log) }
    /// 逐元素 sqrt
    pub fn sqrt(&self) -> Self { self.unary(D::elwise_sqrt) }
    /// 逐元素 sigmoid
    pub fn sigmoid(&self) -> Self { self.unary(D::elwise_sigmoid) }
    /// 逐元素 tanh
    pub fn tanh(&self) -> Self { self.unary(D::elwise_tanh) }
    /// 逐元素 silu
    pub fn silu(&self) -> Self { self.unary(D::elwise_silu) }
    /// 逐元素 sin
    pub fn sin(&self) -> Self { self.unary(D::elwise_sin) }
    /// 逐元素 cos
    pub fn cos(&self) -> Self { self.unary(D::elwise_cos) }

    // -- 内部辅助：构造 ShapeMismatch 错误 --
    fn shape_err(expected: &[usize], actual: &[usize]) -> TensorError {
        TensorError::ShapeMismatch { expected: expected.to_vec(), actual: actual.to_vec() }
    }
}
