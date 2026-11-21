//! tensor_ml.rs：Tensor 高层 ML 算子封装
//!
//! 规格文档第四部分 + 第五部分的所有缺失高层 API

use crate::{Tensor, TensorError};
use oyta_device::{ActivationType, Backend, Float, Accumulate};

/// 从 shape 计算 contiguous strides（右对齐广播友好）
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![0usize; rank];
    let mut s = 1usize;
    for i in (0..rank).rev() {
        strides[i] = s;
        s = s.checked_mul(shape[i]).unwrap_or(usize::MAX);
    }
    strides
}

// ============================================================
// 泛型 ML 算子 impl：Conv1d / BatchNorm / Pooling / cat / split / bias_add / sort / topk / scalar_inplace
// Backend trait 已聚合 Conv1dOps<f32/F16/BF16> / BatchNormOps / PoolOps / ElwiseOps / InPlaceOps
// 但 Tensor<T, D> 是泛型 T，需额外 bound Conv1dOps<T> 等才能调 D::conv1d_fwd
// ============================================================

impl<T: Float + Clone, D: Backend> Tensor<T, D>
where D: oyta_device::Conv1dOps<T> + oyta_device::BatchNormOps<T> + oyta_device::PoolOps<T>
       + oyta_device::ElwiseOps<T> + oyta_device::InPlaceOps<T>
{
    /// Conv1d forward（NCHW 格式，可带 bias）
    pub fn conv1d(
        &self, weight: &Tensor<T, D>, bias: Option<&Tensor<T, D>>,
        padding: usize, stride: usize, dilation: usize,
    ) -> Result<Tensor<T, D>, TensorError> {
        let s = self.shape().dims.clone();
        let w_s = weight.shape().dims.clone();
        if s.len() != 3 { return Err(TensorError::Unsupported("conv1d 输入必须 3D [N,C_in,L]".into())); }
        if w_s.len() != 3 { return Err(TensorError::Unsupported("conv1d 权重必须 3D [C_out,C_in,K]".into())); }
        let l_in = s[2]; let k = w_s[2];
        let l_out = (l_in + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
        let shape_out = vec![s[0], w_s[0], l_out];
        let total = shape_out[0] * shape_out[1] * shape_out[2];
        let mut out_data = vec![T::from_f32(0.0); total];
        let b_ptr = match bias {
            Some(b) => b.data_slice().as_ptr(),
            None => std::ptr::null(),
        };
        unsafe {
            D::conv1d_fwd(
                self.data_slice().as_ptr(), [s[0], s[1], s[2]],
                weight.data_slice().as_ptr(), [w_s[0], w_s[1], w_s[2]],
                b_ptr,
                out_data.as_mut_ptr(), [shape_out[0], shape_out[1], shape_out[2]],
                padding, stride, dilation,
            );
        }
        Tensor::<T, D>::from_data(out_data, shape_out)
    }

    /// BN1d forward
    pub fn batchnorm1d(
        &self, gamma: &Tensor<T, D>, beta: &Tensor<T, D>,
        running_mean: &mut Tensor<T, D>, running_var: &mut Tensor<T, D>,
        eps: f32, momentum: f32, training: bool,
    ) -> Result<Tensor<T, D>, TensorError> {
        let s = self.shape().dims.clone();
        let total: usize = s.iter().product();
        let mut out_data = vec![T::from_f32(0.0); total];
        unsafe {
            D::batchnorm1d_fwd(
                self.data_slice().as_ptr(), &s,
                gamma.data_slice().as_ptr(), beta.data_slice().as_ptr(),
                running_mean.data_slice().as_ptr() as *mut T,
                running_var.data_slice().as_ptr() as *mut T,
                out_data.as_mut_ptr(),
                eps, momentum, training,
            );
        }
        Tensor::<T, D>::from_data(out_data, s)
    }

    /// MaxPool1d forward
    pub fn maxpool1d(
        &self, kernel: usize, stride: usize, padding: usize,
    ) -> Result<Tensor<T, D>, TensorError> {
        let s = self.shape().dims.clone();
        if s.len() != 3 { return Err(TensorError::Unsupported("maxpool1d 输入必须 3D".into())); }
        let l_in = s[2];
        let l_out = (l_in + 2 * padding - kernel) / stride + 1;
        let shape_out = vec![s[0], s[1], l_out];
        let total = shape_out[0] * shape_out[1] * shape_out[2];
        let mut out_data = vec![T::from_f32(0.0); total];
        let mut argmax = vec![0i64; total];
        unsafe {
            D::maxpool1d_fwd(
                self.data_slice().as_ptr(), &s,
                out_data.as_mut_ptr(), &shape_out,
                argmax.as_mut_ptr(),
                kernel, stride, padding,
            );
        }
        Tensor::<T, D>::from_data(out_data, shape_out)
    }

    /// AvgPool1d forward
    pub fn avgpool1d(
        &self, kernel: usize, stride: usize, padding: usize,
    ) -> Result<Tensor<T, D>, TensorError> {
        let s = self.shape().dims.clone();
        if s.len() != 3 { return Err(TensorError::Unsupported("avgpool1d 输入必须 3D".into())); }
        let l_in = s[2];
        let l_out = (l_in + 2 * padding - kernel) / stride + 1;
        let shape_out = vec![s[0], s[1], l_out];
        let total = shape_out[0] * shape_out[1] * shape_out[2];
        let mut out_data = vec![T::from_f32(0.0); total];
        unsafe {
            D::avgpool1d_fwd(
                self.data_slice().as_ptr(), &s,
                out_data.as_mut_ptr(), &shape_out,
                kernel, stride, padding,
            );
        }
        Tensor::<T, D>::from_data(out_data, shape_out)
    }

    // ============================================================
    // cat / split（纯 Vec 操作）
    // ============================================================

    pub fn cat(tensors: &[&Tensor<T, D>], axis: isize) -> Result<Tensor<T, D>, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError::Unsupported("cat: 至少需要 1 个 tensor".into()));
        }
        let rank = tensors[0].shape().dims.len();
        let axis = if axis < 0 { (axis + rank as isize) as usize } else { axis as usize };
        let mut out_shape = tensors[0].shape().dims.clone();
        out_shape[axis] = tensors.iter().map(|t| t.shape().dims[axis]).sum();
        let total: usize = out_shape.iter().product();
        let mut result: Vec<T> = Vec::with_capacity(total);
        for t in tensors {
            for &v in t.data_slice() { result.push(v.clone()); }
        }
        Tensor::<T, D>::from_data(result, out_shape)
    }

    pub fn split(&self, sizes: &[usize], axis: isize) -> Result<Vec<Tensor<T, D>>, TensorError> {
        let rank = self.shape().dims.len();
        let axis = if axis < 0 { (axis + rank as isize) as usize } else { axis as usize };
        let outer_block: usize = self.shape().dims[..axis].iter().product();
        let inner_block: usize = self.shape().dims[axis + 1..].iter().product();
        let data = self.data_slice();
        let mut results = Vec::with_capacity(sizes.len());
        let mut outer_offset = 0;
        for &sz in sizes {
            let mut shape = self.shape().dims.clone();
            shape[axis] = sz;
            let start = outer_offset * inner_block;
            let end = (outer_offset + sz) * inner_block;
            let slice: Vec<T> = data[start..end].iter().map(|v| v.clone()).collect();
            results.push(Tensor::<T, D>::from_data(slice, shape)?);
            outer_offset += sz;
        }
        Ok(results)
    }

    // ============================================================
    // bias_add
    // ============================================================

    pub fn bias_add(&self, bias: &Tensor<T, D>) -> Result<Tensor<T, D>, TensorError> {
        let self_dims = self.shape().dims.clone();
        let rank = self_dims.len();
        let last_dim = self_dims[rank - 1];
        if bias.numel() != last_dim {
            return Err(TensorError::ShapeMismatch {
                expected: vec![last_dim], actual: vec![bias.numel()],
            });
        }
        let self_strides = compute_strides(&self_dims);
        let mut bias_shape = vec![1usize; rank - 1];
        bias_shape.push(last_dim);
        let bias_strides = compute_strides(&bias_shape);
        let out_strides = self_strides.clone();
        let total = self.numel();
        let mut out_data = vec![T::from_f32(0.0); total];
        unsafe {
            D::elwise_add_broadcast(
                self.data_slice().as_ptr(), self_strides.as_ptr(), self_dims.as_ptr(), rank,
                bias.data_slice().as_ptr(), bias_strides.as_ptr(), bias_shape.as_ptr(), rank,
                out_data.as_mut_ptr(), out_strides.as_ptr(), rank,
            );
        }
        Tensor::<T, D>::from_data(out_data, self_dims)
    }

    // ============================================================
    // sort / topk
    // ============================================================

    pub fn sort(
        &self, axis: isize, descending: bool,
    ) -> Result<(Tensor<T, D>, Tensor<i64, D>), TensorError>
    where T: PartialOrd
    {
        let s = self.shape().dims.clone();
        let rank = s.len();
        let axis = if axis < 0 { (axis + rank as isize) as usize } else { axis as usize };
        let outer: usize = s[..axis].iter().product();
        let middle = s[axis];
        let inner: usize = s[axis + 1..].iter().product();
        let total: usize = s.iter().product();
        let data = self.data_slice();
        let mut out_data = vec![T::from_f32(0.0); total];
        let mut idx_data = vec![0i64; total];
        for o in 0..outer {
            for i in 0..inner {
                let mut block: Vec<(usize, T)> = (0..middle)
                    .map(|m| { let idx = o * middle * inner + m * inner + i; (m, data[idx]) }).collect();
                if descending {
                    block.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    block.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }
                for (out_m, (orig_m, val)) in block.into_iter().enumerate() {
                    let dst = o * middle * inner + out_m * inner + i;
                    out_data[dst] = val; idx_data[dst] = orig_m as i64;
                }
            }
        }
        Ok((Tensor::from_data(out_data, s.clone())?, Tensor::from_data(idx_data, s)?))
    }

    pub fn topk(
        &self, k: usize, axis: isize,
    ) -> Result<(Tensor<T, D>, Tensor<i64, D>), TensorError>
    where T: PartialOrd
    {
        let s = self.shape().dims.clone();
        let rank = s.len();
        let axis = if axis < 0 { (axis + rank as isize) as usize } else { axis as usize };
        let orig_dim = s[axis];
        let k_actual = k.min(orig_dim);
        let mut out_shape = s.clone(); out_shape[axis] = k_actual;
        let outer: usize = s[..axis].iter().product();
        let inner: usize = s[axis + 1..].iter().product();
        let out_total: usize = out_shape.iter().product();
        let data = self.data_slice();
        let mut out_data = vec![T::from_f32(0.0); out_total];
        let mut idx_data = vec![0i64; out_total];
        for o in 0..outer {
            for i in 0..inner {
                let mut block: Vec<(usize, T)> = (0..orig_dim)
                    .map(|m| { let idx = o * orig_dim * inner + m * inner + i; (m, data[idx]) }).collect();
                block.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (out_m, (orig_m, val)) in block.into_iter().take(k_actual).enumerate() {
                    let dst = o * k_actual * inner + out_m * inner + i;
                    out_data[dst] = val; idx_data[dst] = orig_m as i64;
                }
            }
        }
        Ok((Tensor::from_data(out_data, out_shape.clone())?, Tensor::from_data(idx_data, out_shape)?))
    }

    // ============================================================
    // scalar in-place
    // ============================================================

    pub fn scalar_add(&mut self, val: f32) {
        let n = self.numel();
        unsafe {
            let ptr = self.data_slice().as_ptr() as *mut T;
            D::elwise_scalar_add_inplace(ptr, n, val);
        }
    }

    pub fn scalar_mul(&mut self, val: f32) {
        let n = self.numel();
        unsafe {
            let ptr = self.data_slice().as_ptr() as *mut T;
            D::elwise_scalar_mul_inplace(ptr, n, val);
        }
    }
}

// ============================================================
// Fused 高层封装（需要 Accumulate bound）
// ============================================================
impl<T: Float + Accumulate + Clone, D: Backend> Tensor<T, D>
where D: oyta_device::FusedOps<T>
{
    pub fn layernorm_gelu(
        &self, gamma: &Tensor<T, D>, beta: &Tensor<T, D>, eps: f32,
    ) -> Result<Tensor<T, D>, TensorError> {
        let s = self.shape().dims.clone();
        let total: usize = s.iter().product();
        let mut out_data = vec![T::from_f32(0.0); total];
        let inner = s[s.len() - 1];
        let outer: usize = s[..s.len() - 1].iter().product();
        unsafe {
            D::layernorm_gelu(
                self.data_slice().as_ptr(), out_data.as_mut_ptr(),
                gamma.data_slice().as_ptr(), beta.data_slice().as_ptr(),
                outer, inner, eps,
            );
        }
        Tensor::from_data(out_data, s)
    }

    pub fn linear_bias_gelu(
        &self, weight: &Tensor<T, D>, bias: &Tensor<T, D>,
    ) -> Result<Tensor<T, D>, TensorError> {
        let m = self.shape().dims[0];
        let k = self.shape().dims[1];
        let n = weight.shape().dims[0];
        let mut out_data = vec![T::from_f32(0.0); m * n];
        unsafe {
            D::linear_bias_gelu(
                self.data_slice().as_ptr(),
                weight.data_slice().as_ptr(),
                bias.data_slice().as_ptr(),
                out_data.as_mut_ptr(),
                m, n, k,
            );
        }
        Tensor::from_data(out_data, vec![m, n])
    }

    pub fn matmul_activation(
        &self, other: &Tensor<T, D>, act: ActivationType,
    ) -> Result<Tensor<T, D>, TensorError> {
        let m = self.shape().dims[0];
        let k = self.shape().dims[1];
        let n = other.shape().dims[1];
        let mut out_data = vec![T::from_f32(0.0); m * n];
        unsafe {
            D::matmul_activation(
                self.data_slice().as_ptr(), k,
                other.data_slice().as_ptr(), n,
                out_data.as_mut_ptr(), n,
                m, n, k,
                1.0, 0.0, act,
            );
        }
        Tensor::from_data(out_data, vec![m, n])
    }
}

// ============================================================
// Matmul batched（底层 f32* 签名，仅 f32 Tensor 可用）
// ============================================================
impl<D: Backend> Tensor<f32, D> where D: oyta_device::MatmulOps {
    pub fn matmul_batched(&self, other: &Tensor<f32, D>) -> Result<Tensor<f32, D>, TensorError> {
        let a_s = self.shape().dims.clone();
        let b_s = other.shape().dims.clone();
        if a_s.len() != 3 || b_s.len() != 3 {
            return Err(TensorError::Unsupported("matmul_batched 需要 3D".into()));
        }
        if a_s[0] != b_s[0] {
            return Err(TensorError::ShapeMismatch { expected: vec![a_s[0]], actual: vec![b_s[0]] });
        }
        if a_s[2] != b_s[1] {
            return Err(TensorError::MatMulDimMismatch { m: a_s[1], k1: a_s[2], k2: b_s[1], n: b_s[2] });
        }
        let batch = a_s[0]; let m = a_s[1]; let k = a_s[2]; let n = b_s[2];
        let mut out_data = vec![0.0f32; batch * m * n];
        unsafe {
            D::matmul_batched(
                self.data_slice().as_ptr(), k, (m * k) as isize,
                other.data_slice().as_ptr(), n, (k * n) as isize,
                out_data.as_mut_ptr(), n, (m * n) as isize,
                m, n, k, batch,
                1.0, 0.0,
            );
        }
        Tensor::from_data(out_data, vec![batch, m, n])
    }
}
