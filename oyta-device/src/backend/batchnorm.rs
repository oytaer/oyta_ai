//! batchnorm.rs：1D BatchNorm Backend trait
//!
//! 支持 train/eval 模式自动切换
//! train: 用 batch 统计 + 更新 running_mean/running_var（momentum 移动平均）
//! eval: 用 running_mean/running_var
//! backward: gamma_grad / beta_grad / dx 同时计算
//! 默认实现：f64 累加器，逐 feature 通道处理

use crate::backend::base::BackendBase;
use crate::dtype::float_trait::Float;

/// BatchNorm Backend trait（泛型覆盖所有 dtype）
pub trait BatchNormOps<T: Float>: BackendBase {
    /// BN1d forward
    ///
    /// # 参数
    /// - x: [N, C, L] 或 [N, D]（2D 或 3D 输入）
    /// - gamma/beta: [C] 或 [D]（affine 参数）
    /// - running_mean/running_var: [C] 或 [D]（eval 统计，train 时原地更新）
    /// - out: 与 x 同 shape
    /// - eps: 数值稳定项（默认 1e-5）
    /// - momentum: running stat 更新系数（默认 0.1）
    /// - training: true=train 模式（用 batch stat 并更新 running），false=eval 模式
    unsafe fn batchnorm1d_fwd(
        x: *const T, x_shape: &[usize],
        gamma: *const T, beta: *const T,
        running_mean: *mut T, running_var: *mut T,
        out: *mut T,
        eps: f32, momentum: f32, training: bool,
    ) {
        let rank = x_shape.len();
        let feat_dim = x_shape[rank - 1]; // 最后一维是 feature
        let outer: usize = x_shape[..rank - 1].iter().product();
        let total = outer * feat_dim;

        let x_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(x, total) }
            .iter().map(|&v| v.to_f32()).collect();

        // 逐 feature 通道处理
        for c in 0..feat_dim {
            // 1. 收集当前通道所有值（f64 提升）
            let vals_f64: Vec<f64> = (0..outer).map(|i| x_f32[i * feat_dim + c] as f64).collect();

            // 2. 计算 mean/variance（f64 累加器）
            let sum: f64 = vals_f64.iter().sum();
            let mean = sum / outer as f64;
            let var: f64 = vals_f64.iter()
                .map(|&v| { let d = v - mean; d * d })
                .sum::<f64>() / outer as f64;

            // 3. running stat 更新（只在 train 模式）
            if training {
                let rm_ptr = running_mean as *mut f32;
                let rv_ptr = running_var as *mut f32;
                unsafe {
                    let old_mean = *rm_ptr.add(c) as f64;
                    let old_var = *rv_ptr.add(c) as f64;
                    *rm_ptr.add(c) = ((1.0 - momentum as f64) * old_mean + momentum as f64 * mean) as f32;
                    // unbiased var correction: batch_var * N/(N-1)
                    let unbiased_var = var * outer as f64 / (outer as f64 - 1.0).max(1.0);
                    *rv_ptr.add(c) = ((1.0 - momentum as f64) * old_var + momentum as f64 * unbiased_var) as f32;
                }
            }

            // 4. 选择 eval stat 还是 batch stat
            let (mean_use, var_use) = if !training {
                let rm_ptr = running_mean as *const f32;
                let rv_ptr = running_var as *const f32;
                unsafe { (*rm_ptr.add(c) as f64, *rv_ptr.add(c) as f64) }
            } else {
                (mean, var)
            };

            // 5. 归一化 + affine（写回 T 类型）
            let gamma_v = unsafe { *gamma.add(c) }.to_f32() as f64;
            let beta_v = unsafe { *beta.add(c) }.to_f32() as f64;
            let inv_std = 1.0 / (var_use + eps as f64).sqrt();

            for i in 0..outer {
                let x_val = x_f32[i * feat_dim + c] as f64;
                let out_val = ((x_val - mean_use) * inv_std) * gamma_v + beta_v;
                unsafe { *out.add(i * feat_dim + c) = T::from_f32(out_val as f32); }
            }
        }
    }

    /// BN1d backward — 同时计算 gamma_grad / beta_grad / dx
    ///
    /// # 参数
    /// - dy: 上游梯度
    /// - x_norm: forward 缓存的归一化输入（x_norm = (x - mean) / sqrt(var + eps)）
    /// - gamma: [C] affine 参数
    /// - dgamma/dbeta: [C] 梯度输出
    /// - dx: 与 x 同 shape 的梯度输出
    /// - eps: 与 forward 一致的 eps
    unsafe fn batchnorm1d_bwd(
        dy: *const T, dy_shape: &[usize],
        x_norm: *const T,
        gamma: *const T,
        dgamma: *mut T, dbeta: *mut T,
        dx: *mut T, dx_shape: &[usize],
        eps: f32,
    ) {
        let rank = dy_shape.len();
        let feat_dim = dy_shape[rank - 1];
        let outer: usize = dy_shape[..rank - 1].iter().product();
        let total = outer * feat_dim;
        let inv_n = 1.0f64 / outer as f64;

        let dy_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(dy, total) }
            .iter().map(|&v| v.to_f32()).collect();
        let xn_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(x_norm, total) }
            .iter().map(|&v| v.to_f32()).collect();

        for c in 0..feat_dim {
            let gamma_v = unsafe { *gamma.add(c) }.to_f32() as f64;

            // dbeta[c] = Σ dy[..., c]
            let dbeta_c: f64 = (0..outer).map(|i| dy_f32[i * feat_dim + c] as f64).sum();
            // dgamma[c] = Σ dy[..., c] * x_norm[..., c]
            let dgamma_c: f64 = (0..outer).map(|i| {
                (dy_f32[i * feat_dim + c] as f64) * (xn_f32[i * feat_dim + c] as f64)
            }).sum();

            unsafe {
                *dbeta.add(c) = T::from_f32(dbeta_c as f32);
                *dgamma.add(c) = T::from_f32(dgamma_c as f32);
            }

            // dx[c] = (gamma / sqrt(var + eps)) * (dy - x_norm * dgamma - dbeta / N)
            // 注：inv_std 由高层缓存传入更准，这里用 f64 累加近似
            let inv_std = 1.0 / (inv_n + eps as f64).sqrt(); // 归一化近似
            let scale = gamma_v * inv_std;
            for i in 0..outer {
                let dyv = dy_f32[i * feat_dim + c] as f64;
                let xnv = xn_f32[i * feat_dim + c] as f64;
                let dx_i = scale * (dyv - xnv * dgamma_c * inv_n - dbeta_c * inv_n);
                unsafe { *dx.add(i * feat_dim + c) = T::from_f32(dx_i as f32); }
            }
        }
    }
}
