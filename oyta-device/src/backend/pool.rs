//! pool.rs：MaxPool1d / AvgPool1d Backend trait
//!
//! MaxPool 需要 argmax 缓存（backward 用）
//! AvgPool backward 是均匀分配梯度
//! 默认实现：纯 f64 累加器，padding 自动裁剪

use crate::backend::base::BackendBase;
use crate::dtype::float_trait::Float;

/// MaxPool1d + AvgPool1d Backend trait（泛型覆盖所有 dtype）
pub trait PoolOps<T: Float>: BackendBase {
    /// MaxPool1d forward + argmax cache
    ///
    /// # 参数
    /// - x: [N, C, L] 输入
    /// - out: [N, C, L_out] 输出
    /// - argmax_cache: [N, C, L_out] 存每个输出位置对应的输入索引（backward 用）
    /// - kernel/stride/padding: 超参数
    unsafe fn maxpool1d_fwd(
        x: *const T, x_shape: &[usize],
        out: *mut T, out_shape: &[usize],
        argmax_cache: *mut i64,
        kernel: usize, stride: usize, padding: usize,
    ) {
        let n = x_shape[0]; let c = x_shape[1]; let l_in = x_shape[2];
        let l_out = out_shape[2];

        for ni in 0..n {
            for ci in 0..c {
                for li in 0..l_out {
                    let start = (li * stride).saturating_sub(padding);
                    let end = (li * stride + kernel).saturating_sub(padding).min(l_in);
                    let start = start.max(0);

                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx: i64 = -1;
                    for ki in start..end {
                        let val = unsafe { *x.add(ni * c * l_in + ci * l_in + ki) }.to_f32();
                        if val > max_val { max_val = val; max_idx = ki as i64; }
                    }
                    if max_idx < 0 { max_val = f32::NEG_INFINITY; max_idx = 0; }
                    unsafe {
                        *out.add(ni * c * l_out + ci * l_out + li) = T::from_f32(max_val);
                        *argmax_cache.add(ni * c * l_out + ci * l_out + li) = max_idx;
                    }
                }
            }
        }
    }

    /// MaxPool1d backward：根据 argmax 把梯度稀疏传回
    unsafe fn maxpool1d_bwd(
        dy: *const T, dy_shape: &[usize],
        argmax_cache: *const i64,
        dx: *mut T, dx_shape: &[usize],
    ) {
        let n = dy_shape[0]; let c = dy_shape[1]; let l_out = dy_shape[2];
        let l_in = dx_shape[2];
        let total_dx = n * c * l_in;

        // dx 先清零
        unsafe { for i in 0..total_dx { *dx.add(i) = T::from_f32(0.0); } }

        for ni in 0..n {
            for ci in 0..c {
                for li in 0..l_out {
                    let argmax_idx = unsafe { *argmax_cache.add(ni * c * l_out + ci * l_out + li) };
                    if argmax_idx >= 0 {
                        let dy_val = unsafe { *dy.add(ni * c * l_out + ci * l_out + li) };
                        let dst_idx = ni * c * l_in + ci * l_in + argmax_idx as usize;
                        unsafe {
                            let cur = *dx.add(dst_idx);
                            *dx.add(dst_idx) = T::from_f32(cur.to_f32() + dy_val.to_f32());
                        }
                    }
                }
            }
        }
    }

    /// AvgPool1d forward
    unsafe fn avgpool1d_fwd(
        x: *const T, x_shape: &[usize],
        out: *mut T, out_shape: &[usize],
        kernel: usize, stride: usize, padding: usize,
    ) {
        let n = x_shape[0]; let c = x_shape[1]; let l_in = x_shape[2];
        let l_out = out_shape[2];

        for ni in 0..n {
            for ci in 0..c {
                for li in 0..l_out {
                    let start = (li * stride).saturating_sub(padding);
                    let end = (li * stride + kernel).saturating_sub(padding).min(l_in);
                    let start = start.max(0);
                    let count = (end - start) as f64;

                    let mut sum: f64 = 0.0;
                    for ki in start..end {
                        sum += unsafe { *x.add(ni * c * l_in + ci * l_in + ki) }.to_f32() as f64;
                    }
                    unsafe {
                        *out.add(ni * c * l_out + ci * l_out + li) =
                            T::from_f32((sum / count.max(1.0)) as f32);
                    }
                }
            }
        }
    }

    /// AvgPool1d backward：梯度均匀分配到 kernel 覆盖的位置
    unsafe fn avgpool1d_bwd(
        dy: *const T, dy_shape: &[usize],
        dx: *mut T, dx_shape: &[usize],
        kernel: usize, stride: usize, padding: usize,
    ) {
        let n = dy_shape[0]; let c = dy_shape[1]; let l_out = dy_shape[2];
        let l_in = dx_shape[2];
        let total_dx = n * c * l_in;

        unsafe { for i in 0..total_dx { *dx.add(i) = T::from_f32(0.0); } }

        for ni in 0..n {
            for ci in 0..c {
                for li in 0..l_out {
                    let start = (li * stride).saturating_sub(padding);
                    let end = (li * stride + kernel).saturating_sub(padding).min(l_in);
                    let start = start.max(0);
                    let count = (end - start) as f64;
                    let dy_val = unsafe { *dy.add(ni * c * l_out + ci * l_out + li) }.to_f32() as f64;
                    let delta = dy_val / count.max(1.0);

                    for ki in start..end {
                        let idx = ni * c * l_in + ci * l_in + ki;
                        unsafe {
                            let cur = *dx.add(idx);
                            *dx.add(idx) = T::from_f32(cur.to_f32() + delta as f32);
                        }
                    }
                }
            }
        }
    }
}
