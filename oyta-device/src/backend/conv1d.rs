//! conv1d.rs：1D 卷积 Backend trait（Conv1dOps<T> 一套覆盖 f32/f16/bf16/f64）
//!
//! 支持：NCHW 格式、任意 padding/stride/dilation
//! backward：dx + dwdb（同时计算权重梯度和偏置梯度）
//! 默认实现：纯 f64 累加器循环 + Rayon 并行 batch 维

use crate::backend::base::BackendBase;
use crate::dtype::float_trait::Float;

/// 1D 卷积 Backend trait（泛型覆盖所有 dtype）
pub trait Conv1dOps<T: Float>: BackendBase {
    /// Conv1d forward（NCHW 格式，有可选 bias）
    ///
    /// # 参数
    /// - x: [N, C_in, L] 输入
    /// - w: [C_out, C_in, K] 权重
    /// - b: [C_out] 偏置（null 表示无 bias）
    /// - out: [N, C_out, L_out] 输出
    /// - padding/stride/dilation: 超参数
    ///
    /// # L_out 公式
    /// L_out = (L + 2*padding - dilation*(K-1) - 1) / stride + 1
    unsafe fn conv1d_fwd(
        x: *const T, x_dim: [usize; 3],
        w: *const T, w_dim: [usize; 3],
        b: *const T,
        out: *mut T, out_dim: [usize; 3],
        padding: usize, stride: usize, dilation: usize,
    ) {
        let n = x_dim[0]; let c_in = x_dim[1]; let l_in = x_dim[2];
        let c_out = w_dim[0]; let k = w_dim[2];
        let l_out = out_dim[2];
        let out_total = out_dim[0] * out_dim[1] * out_dim[2];

        // dtype → f32 中间缓存（统一计算精度）
        let x_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(x, n * c_in * l_in) }
            .iter().map(|&v| v.to_f32()).collect();
        let w_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(w, c_out * c_in * k) }
            .iter().map(|&v| v.to_f32()).collect();
        let b_f32: Vec<f32> = if b.is_null() { vec![0.0f32; c_out] }
            else { unsafe { std::slice::from_raw_parts(b, c_out) }.iter().map(|&v| v.to_f32()).collect() };

        let mut out_f32 = vec![0.0f32; out_total];
        // Rayon 并行 batch 维 + c_out 维
        use rayon::prelude::*;
        out_f32.par_chunks_mut(c_out * l_out).enumerate().for_each(|(ni, chunk)| {
            for co in 0..c_out {
                for lo in 0..l_out {
                    let mut acc: f64 = 0.0; // f64 累加器
                    for ci in 0..c_in {
                        for ki in 0..k {
                            let l_idx = (lo as isize * stride as isize + ki as isize * dilation as isize)
                                - padding as isize;
                            if l_idx >= 0 && l_idx < l_in as isize {
                                let xv = x_f32[ni * c_in * l_in + ci * l_in + l_idx as usize] as f64;
                                let wv = w_f32[co * c_in * k + ci * k + ki] as f64;
                                acc += xv * wv;
                            }
                        }
                    }
                    acc += b_f32[co] as f64;
                    chunk[co * l_out + lo] = acc as f32;
                }
            }
        });

        // f32 → T 写回
        unsafe { for i in 0..out_total { *out.add(i) = T::from_f32(out_f32[i]); } }
    }

    /// Conv1d backward — 输入梯度 dx
    ///
    /// # 参数
    /// - dy: [N, C_out, L_out] 上游梯度
    /// - w: [C_out, C_in, K] 权重（fwd 时的）
    /// - dx: [N, C_in, L] 输出（被累加，调用前必须清零）
    unsafe fn conv1d_bwd_dx(
        dy: *const T, dy_dim: [usize; 3],
        w: *const T, w_dim: [usize; 3],
        dx: *mut T, dx_dim: [usize; 3],
        padding: usize, stride: usize, dilation: usize,
    ) {
        let n = dy_dim[0]; let c_out = dy_dim[1]; let l_out = dy_dim[2];
        let c_in = dx_dim[1]; let l_in = dx_dim[2]; let k = w_dim[2];
        let dx_total = dx_dim[0] * dx_dim[1] * dx_dim[2];

        let dy_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(dy, n * c_out * l_out) }
            .iter().map(|&v| v.to_f32()).collect();
        let w_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(w, c_out * c_in * k) }
            .iter().map(|&v| v.to_f32()).collect();

        let mut dx_f32 = vec![0.0f32; dx_total];
        // Rayon par_chunks_mut：每个 chunk 对应一个 batch item 的完整 dx
        // 这样闭包只需 mutate chunk 内部，不需 mutate 外层 Vec
        use rayon::prelude::*;
        let inner_size = c_in * l_in; // 每个 batch item 的 dx 大小
        dx_f32.par_chunks_mut(inner_size).enumerate().for_each(|(ni, chunk)| {
            for co in 0..c_out {
                for lo in 0..l_out {
                    // dy shape: [N, C_out, L_out]，索引正确
                    let dy_val = dy_f32[ni * c_out * l_out + co * l_out + lo] as f64;
                    for ci in 0..c_in {
                        for ki in 0..k {
                            let l_idx = (lo as isize * stride as isize + ki as isize * dilation as isize)
                                - padding as isize;
                            if l_idx >= 0 && l_idx < l_in as isize {
                                let w_val = w_f32[co * c_in * k + ci * k + ki] as f64;
                                let inner_idx = ci * l_in + l_idx as usize;
                                chunk[inner_idx] = (chunk[inner_idx] as f64 + dy_val * w_val) as f32;
                            }
                        }
                    }
                }
            }
        });
        unsafe { for i in 0..dx_total { *dx.add(i) = T::from_f32(dx_f32[i]); } }
    }

    /// Conv1d backward — 权重梯度 dw + 偏置梯度 db（一次计算两个）
    ///
    /// # 参数
    /// - dy: [N, C_out, L_out] 上游梯度
    /// - x: [N, C_in, L] 前向输入
    /// - dw: [C_out, C_in, K] 权重梯度输出
    /// - db: [C_out] 偏置梯度输出
    unsafe fn conv1d_bwd_dwdb(
        dy: *const T, dy_dim: [usize; 3],
        x: *const T, x_dim: [usize; 3],
        dw: *mut T, dw_dim: [usize; 3],
        db: *mut T, db_dim: [usize; 1],
        padding: usize, stride: usize, dilation: usize,
    ) {
        let n = dy_dim[0]; let c_out = dy_dim[1]; let l_out = dy_dim[2];
        let c_in = x_dim[1]; let l_in = x_dim[2]; let k = dw_dim[2];
        let dw_total = c_out * c_in * k;
        let db_total = db_dim[0];

        let dy_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(dy, n * c_out * l_out) }
            .iter().map(|&v| v.to_f32()).collect();
        let x_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(x, n * c_in * l_in) }
            .iter().map(|&v| v.to_f32()).collect();

        // dw/db 先清零（串行累加，多块冲突）
        let mut dw_f32 = vec![0.0f32; dw_total];
        let mut db_f32 = vec![0.0f32; db_total];

        for ni in 0..n {
            for co in 0..c_out {
                for lo in 0..l_out {
                    let dy_val = dy_f32[ni * c_out * l_out + co * l_out + lo] as f64;
                    db_f32[co] = (db_f32[co] as f64 + dy_val) as f32;
                    for ci in 0..c_in {
                        for ki in 0..k {
                            let l_idx = (lo as isize * stride as isize + ki as isize * dilation as isize)
                                - padding as isize;
                            if l_idx >= 0 && l_idx < l_in as isize {
                                let xv = x_f32[ni * c_in * l_in + ci * l_in + l_idx as usize] as f64;
                                let idx = co * c_in * k + ci * k + ki;
                                dw_f32[idx] = (dw_f32[idx] as f64 + dy_val * xv) as f32;
                            }
                        }
                    }
                }
            }
        }

        unsafe {
            for i in 0..dw_total { *dw.add(i) = T::from_f32(dw_f32[i]); }
            for i in 0..db_total { *db.add(i) = T::from_f32(db_f32[i]); }
        }
    }
}
