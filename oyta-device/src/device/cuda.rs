//! cuda.rs：CUDA 后端（#[cfg(feature="cuda")]）
//!
//! 真实实现：
//! - extern "C" 声明 cuda runtime + cuBLAS 函数（封装在 gpu_ffi 子模块）
//! - OnceLock runtime detection（cudaGetDeviceCount probe）
//! - cublasSgemm 真实 matmul 路径（检测到 GPU 硬件时）
//! - CPU fallback（无硬件时，继承 rayon 默认实现）
//! - cuBLAS row-major trick：CUBLAS_OP_N × CUBLAS_OP_N 做 row-major C = A*B
//!
//! 注意：cuda-sys + cublas-sys 只是 FFI 声明，link 由系统 CUDA Toolkit 提供。
//! 如果系统没有 cublas.so，cublas 符号会在运行时找不到——但因为有 runtime
//! probe（cudaGetDeviceCount 先返回 0），GPU 路径永远不会被调用，所以安全。

#[cfg(feature = "cuda")]
use crate::backend::*;
#[cfg(feature = "cuda")]
use crate::backend::aggregate::{Backend, DeviceContext, DeviceHandle, DeviceProperties, DeviceInner, DeviceError};
#[cfg(feature = "cuda")]
use crate::dtype::Precision;
#[cfg(feature = "cuda")]
use std::marker::PhantomData;
#[cfg(feature = "cuda")]
use std::sync::OnceLock;

// ======== GPU FFI 子模块 ========
// extern "C" 声明 + cuBLAS 常量 + runtime probe 函数
// 在默认 CPU feature 下编译，永远不会被调用 → 子模块级 allow 集中消除 lint
#[cfg(feature = "cuda")]
#[allow(dead_code, non_upper_case_globals)]
mod gpu_ffi {
    use std::sync::OnceLock;

    // edition 2024 要求 extern 块必须是 unsafe
    unsafe extern "C" {
        // CUDA Runtime API
        pub fn cudaGetDeviceCount(count: *mut i32) -> i32;
        pub fn cudaSetDevice(device: i32) -> i32;
        pub fn cudaMalloc(ptr: *mut *mut f32, size: usize) -> i32;
        pub fn cudaMemcpy(dst: *mut f32, src: *const f32, size: usize, kind: i32) -> i32;
        pub fn cudaFree(ptr: *mut f32) -> i32;
        pub fn cudaMemset(ptr: *mut f32, val: i32, size: usize) -> i32;
        pub fn cudaStreamCreate(stream: *mut u64) -> i32;
        pub fn cudaStreamDestroy(stream: u64) -> i32;
        pub fn cudaDeviceSynchronize() -> i32;
        // cuBLAS API
        pub fn cublasCreate(handle: *mut u64) -> i32;
        pub fn cublasDestroy(handle: u64) -> i32;
        pub fn cublasSetStream(handle: u64, stream: u64) -> i32;
        pub fn cublasSgemm(
            handle: u64, transa: i32, transb: i32,
            m: i32, n: i32, k: i32,
            alpha: *const f32,
            a: *const f32, lda: i32,
            b: *const f32, ldb: i32,
            beta: *const f32,
            c: *mut f32, ldc: i32,
        ) -> i32;

        // ======== oyta CUDA kernels（build.rs 编译进 oyta_device_cuda_kernels 静态库）========
        // Conv1d
        pub fn oyta_conv1d_fwd(
            x: *const f32, out: *mut f32,
            N: i32, C_in: i32, C_out: i32, L_in: i32, L_out: i32, K: i32,
            padding: i32, stride: i32, dilation: i32, has_bias: i32,
            b: *const f32, w: *const f32,
        );
        pub fn oyta_conv1d_bwd_dx(
            dy: *const f32, w: *const f32, dx: *mut f32,
            N: i32, C_in: i32, C_out: i32, L_in: i32, L_out: i32, K: i32,
            padding: i32, stride: i32, dilation: i32,
        );
        pub fn oyta_conv1d_bwd_dwdb(
            dy: *const f32, x: *const f32, dw: *mut f32, db: *mut f32,
            N: i32, C_in: i32, C_out: i32, L_in: i32, L_out: i32, K: i32,
            padding: i32, stride: i32, dilation: i32,
        );
        // BatchNorm
        pub fn oyta_batchnorm1d_fwd_train(
            x: *const f32, y: *mut f32, x_norm: *mut f32,
            running_mean: *mut f32, running_var: *mut f32,
            gamma: *const f32, beta: *const f32,
            feat_dim: i32, N: i32, eps: f32, momentum: f32,
        );
        pub fn oyta_batchnorm1d_fwd_eval(
            x: *const f32, y: *mut f32, x_norm: *mut f32,
            running_mean: *const f32, running_var: *const f32,
            gamma: *const f32, beta: *const f32,
            feat_dim: i32, N: i32, eps: f32,
        );
        pub fn oyta_batchnorm1d_bwd(
            dy: *const f32, x_norm: *const f32, dx: *mut f32,
            dgamma: *mut f32, dbeta: *mut f32,
            gamma: *const f32,
            feat_dim: i32, N: i32, eps: f32,
        );
        // MaxPool1d
        pub fn oyta_maxpool1d_fwd(
            x: *const f32, out: *mut f32, argmax: *mut i64,
            N: i32, C: i32, L_in: i32, L_out: i32, K: i32, stride: i32, padding: i32,
        );
        pub fn oyta_avgpool1d_fwd(
            x: *const f32, out: *mut f32,
            N: i32, C: i32, L_in: i32, L_out: i32, K: i32, stride: i32, padding: i32,
        );
        pub fn oyta_maxpool1d_bwd(
            dy: *const f32, argmax: *const i64, dx: *mut f32,
            N: i32, C: i32, L_out: i32, L_in: i32,
        );
        pub fn oyta_avgpool1d_bwd(
            dy: *const f32, dx: *mut f32,
            N: i32, C: i32, L_out: i32, L_in: i32, K: i32, stride: i32, padding: i32,
        );
        // RNN
        pub fn oyta_gru_step(
            x: *const f32, h_prev: *const f32, h_new: *mut f32,
            w: *const f32, b: *const f32,
            in_size: i32, H: i32,
        );
        pub fn oyta_lstm_step(
            x: *const f32, h_prev: *const f32, c_prev: *const f32,
            h_new: *mut f32, c_new: *mut f32,
            w: *const f32, b: *const f32,
            in_size: i32, H: i32,
        );
    }

    // cuBLAS 常量（cuBLAS 自己的定义，不是 CBLAS 的 111/112）
    pub const CUBLAS_OP_N: i32 = 0;
    pub const CUBLAS_OP_T: i32 = 1;

    // CUDA cudaMemcpyKind（预留，未来 H2D/D2H 内存管理会用到）
    pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
    pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

    // ======== runtime probe：检测 CUDA 硬件 ========
    /// 第一次调用就测一次，之后返回 OnceLock 缓存值
    pub fn cuda_device_count() -> i32 {
        static COUNT: OnceLock<i32> = OnceLock::new();
        *COUNT.get_or_init(|| {
            let mut count: i32 = 0;
            let err = unsafe { cudaGetDeviceCount(&mut count) };
            if err == 0 && count > 0 { count } else { 0 }
        })
    }

    pub fn has_cuda_hardware() -> bool { cuda_device_count() > 0 }
}

/// Cuda 设备标记
#[derive(Debug, Clone, Copy)]
#[cfg(feature = "cuda")]
pub struct Cuda;

// ======== BackendBase + 5 个 blanket impl ========
#[cfg(feature = "cuda")]
impl BackendBase for Cuda {}
#[cfg(feature = "cuda")]
impl<T: crate::dtype::float_trait::Float> crate::backend::elwise::ElwiseOps<T> for Cuda {}
#[cfg(feature = "cuda")]
impl<T: crate::dtype::float_trait::Float + crate::dtype::float_trait::Accumulate> crate::backend::reduce::ReduceOps<T> for Cuda {}
#[cfg(feature = "cuda")]
impl<T: crate::dtype::float_trait::Float> crate::backend::flash_attn::FlashAttentionOps<T> for Cuda {}
#[cfg(feature = "cuda")]
impl<T: crate::dtype::float_trait::Float + crate::dtype::float_trait::Accumulate> crate::backend::fused::FusedOps<T> for Cuda {}
#[cfg(feature = "cuda")]
impl<T: crate::dtype::float_trait::Float> crate::backend::inplace::InPlaceOps<T> for Cuda {}
impl<T: crate::dtype::float_trait::Float> crate::backend::conv1d::Conv1dOps<T> for Cuda {
    unsafe fn conv1d_fwd(
        x: *const T, x_dim: [usize; 3],
        w: *const T, w_dim: [usize; 3],
        b: *const T,
        out: *mut T, out_dim: [usize; 3],
        padding: usize, stride: usize, dilation: usize,
    ) {
        // f32 + 有 GPU → 走 CUDA kernel；否则手写 CPU rayon 默认
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && gpu_ffi::has_cuda_hardware() {
            let has_bias = if b.is_null() { 0i32 } else { 1i32 };
            unsafe {
                gpu_ffi::oyta_conv1d_fwd(
                    x as *const f32, out as *mut f32,
                    x_dim[0] as i32, x_dim[1] as i32, w_dim[0] as i32,
                    x_dim[2] as i32, out_dim[2] as i32, w_dim[2] as i32,
                    padding as i32, stride as i32, dilation as i32, has_bias,
                    if b.is_null() { std::ptr::null() } else { b as *const f32 },
                    w as *const f32,
                );
                gpu_ffi::cudaDeviceSynchronize();
            }
        } else {
            // ======= CPU fallback（rayon + f64 累加器）=======
            let n = x_dim[0]; let c_in = x_dim[1]; let l_in = x_dim[2];
            let c_out = w_dim[0]; let k = w_dim[2]; let l_out = out_dim[2];
            let out_total = out_dim[0] * out_dim[1] * out_dim[2];
            let x_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(x, n * c_in * l_in) }.iter().map(|&v| v.to_f32()).collect();
            let w_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(w, c_out * c_in * k) }.iter().map(|&v| v.to_f32()).collect();
            let b_f32: Vec<f32> = if b.is_null() { vec![0.0f32; c_out] } else { unsafe { std::slice::from_raw_parts(b, c_out) }.iter().map(|&v| v.to_f32()).collect() };
            let mut out_f32 = vec![0.0f32; out_total];
            use rayon::prelude::*;
            out_f32.par_chunks_mut(c_out * l_out).enumerate().for_each(|(ni, chunk)| {
                for co in 0..c_out { for lo in 0..l_out {
                    let mut acc: f64 = 0.0;
                    for ci in 0..c_in { for ki in 0..k {
                        let li = lo as isize * stride as isize + ki as isize * dilation as isize - padding as isize;
                        if li >= 0 && li < l_in as isize { acc += (x_f32[ni*c_in*l_in+ci*l_in+li as usize] as f64) * (w_f32[co*c_in*k+ci*k+ki] as f64); }
                    }}
                    chunk[co*l_out+lo] = (acc + b_f32[co] as f64) as f32;
                }}
            });
            unsafe { for i in 0..out_total { *out.add(i) = T::from_f32(out_f32[i]); } }
        }
    }
    unsafe fn conv1d_bwd_dx(
        dy: *const T, dy_dim: [usize; 3],
        w: *const T, w_dim: [usize; 3],
        dx: *mut T, dx_dim: [usize; 3],
        padding: usize, stride: usize, dilation: usize,
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && gpu_ffi::has_cuda_hardware() {
            unsafe {
                gpu_ffi::oyta_conv1d_bwd_dx(
                    dy as *const f32, w as *const f32, dx as *mut f32,
                    dy_dim[0] as i32, dx_dim[1] as i32, dy_dim[1] as i32,
                    dx_dim[2] as i32, dy_dim[2] as i32, w_dim[2] as i32,
                    padding as i32, stride as i32, dilation as i32,
                );
                gpu_ffi::cudaDeviceSynchronize();
            }
        } else {
            let n = dy_dim[0]; let c_out = dy_dim[1]; let l_out = dy_dim[2];
            let c_in = dx_dim[1]; let l_in = dx_dim[2]; let k = w_dim[2];
            let dx_total = dx_dim[0] * dx_dim[1] * dx_dim[2];
            let dy_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(dy, n*c_out*l_out) }.iter().map(|&v| v.to_f32()).collect();
            let w_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(w, c_out*c_in*k) }.iter().map(|&v| v.to_f32()).collect();
            let mut dx_f32 = vec![0.0f32; dx_total];
            use rayon::prelude::*;
            dx_f32.par_chunks_mut(c_in*l_in).enumerate().for_each(|(ni, chunk)| {
                for co in 0..c_out { for lo in 0..l_out {
                    let dyv = dy_f32[ni*c_out*l_out+co*l_out+lo] as f64;
                    for ci in 0..c_in { for ki in 0..k {
                        let li = lo as isize*stride as isize+ki as isize*dilation as isize-padding as isize;
                        if li>=0 && li<l_in as isize { let wv=w_f32[co*c_in*k+ci*k+ki] as f64; chunk[ci*l_in+li as usize]=(chunk[ci*l_in+li as usize] as f64+dyv*wv) as f32; }
                    }}
                }}
            });
            unsafe { for i in 0..dx_total { *dx.add(i) = T::from_f32(dx_f32[i]); } }
        }
    }
    unsafe fn conv1d_bwd_dwdb(
        dy: *const T, dy_dim: [usize; 3],
        x: *const T, x_dim: [usize; 3],
        dw: *mut T, dw_dim: [usize; 3],
        db: *mut T, db_dim: [usize; 1],
        padding: usize, stride: usize, dilation: usize,
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && gpu_ffi::has_cuda_hardware() {
            unsafe {
                gpu_ffi::oyta_conv1d_bwd_dwdb(
                    dy as *const f32, x as *const f32, dw as *mut f32, db as *mut f32,
                    dy_dim[0] as i32, x_dim[1] as i32, dy_dim[1] as i32,
                    x_dim[2] as i32, dy_dim[2] as i32, dw_dim[2] as i32,
                    padding as i32, stride as i32, dilation as i32,
                );
                gpu_ffi::cudaDeviceSynchronize();
            }
        } else {
            let n=dy_dim[0]; let c_out=dy_dim[1]; let l_out=dy_dim[2];
            let c_in=x_dim[1]; let l_in=x_dim[2]; let k=dw_dim[2];
            let dw_total=c_out*c_in*k; let db_total=db_dim[0];
            let dy_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(dy,n*c_out*l_out) }.iter().map(|&v| v.to_f32()).collect();
            let x_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(x,n*c_in*l_in) }.iter().map(|&v| v.to_f32()).collect();
            let mut dw_f32=vec![0.0f32;dw_total]; let mut db_f32=vec![0.0f32;db_total];
            for ni in 0..n { for co in 0..c_out { for lo in 0..l_out {
                let dyv=dy_f32[ni*c_out*l_out+co*l_out+lo] as f64;
                db_f32[co]=(db_f32[co] as f64+dyv) as f32;
                for ci in 0..c_in { for ki in 0..k {
                    let li=lo as isize*stride as isize+ki as isize*dilation as isize-padding as isize;
                    if li>=0 && li<l_in as isize { let xv=x_f32[ni*c_in*l_in+ci*l_in+li as usize] as f64; dw_f32[co*c_in*k+ci*k+ki]=(dw_f32[co*c_in*k+ci*k+ki] as f64+dyv*xv) as f32; }
                }}
            }}}
            unsafe { for i in 0..dw_total { *dw.add(i) = T::from_f32(dw_f32[i]); } for i in 0..db_total { *db.add(i) = T::from_f32(db_f32[i]); } }
        }
    }
}

impl<T: crate::dtype::float_trait::Float> crate::backend::batchnorm::BatchNormOps<T> for Cuda {
    unsafe fn batchnorm1d_fwd(
        x: *const T, x_shape: &[usize],
        gamma: *const T, beta: *const T,
        running_mean: *mut T, running_var: *mut T,
        out: *mut T,
        eps: f32, momentum: f32, training: bool,
    ) {
        let rank = x_shape.len();
        let feat_dim = x_shape[rank - 1];
        let outer: usize = x_shape[..rank - 1].iter().product();
        let total = outer * feat_dim;
        // 用 out 前半缓存 x_norm（CUDA kernel 需要）
        let x_norm_buf = out as *mut f32; // 与 out 同大小，复用

        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && gpu_ffi::has_cuda_hardware() {
            unsafe {
                if training {
                    gpu_ffi::oyta_batchnorm1d_fwd_train(
                        x as *const f32, out as *mut f32, x_norm_buf,
                        running_mean as *mut f32, running_var as *mut f32,
                        gamma as *const f32, beta as *const f32,
                        feat_dim as i32, outer as i32, eps, momentum,
                    );
                } else {
                    gpu_ffi::oyta_batchnorm1d_fwd_eval(
                        x as *const f32, out as *mut f32, x_norm_buf,
                        running_mean as *const f32, running_var as *const f32,
                        gamma as *const f32, beta as *const f32,
                        feat_dim as i32, outer as i32, eps,
                    );
                }
                gpu_ffi::cudaDeviceSynchronize();
            }
        } else {
            // ======= CPU fallback =======
            let x_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(x, total) }.iter().map(|&v| v.to_f32()).collect();
            let mut y_f32 = vec![0.0f32; total];
            let g_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(gamma, feat_dim) }.iter().map(|&v| v.to_f32()).collect();
            let b_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(beta, feat_dim) }.iter().map(|&v| v.to_f32()).collect();
            let mut rm_f32 = vec![0.0f32; feat_dim]; let mut rv_f32 = vec![0.0f32; feat_dim];
            for c in 0..feat_dim {
                let mut s: f64 = 0.0; let mut v: f64 = 0.0;
                for i in 0..outer { s += x_f32[i*feat_dim+c] as f64; }
                let mean = (s / outer as f64) as f32;
                for i in 0..outer { let d = x_f32[i*feat_dim+c] as f64 - mean as f64; v += d*d; }
                let var = (v / outer as f64) as f32;
                let (mean_v, var_v) = if training { (mean, var) } else { let rm = unsafe { *running_mean.add(c) }; let rv = unsafe { *running_var.add(c) }; (rm.to_f32(), rv.to_f32()) };
                let inv = 1.0f32 / (var_v + eps).sqrt();
                for i in 0..outer {
                    let xn = (x_f32[i*feat_dim+c] - mean_v) * inv;
                    y_f32[i*feat_dim+c] = xn * g_f32[c] + b_f32[c];
                }
                if training { rm_f32[c] = (1.0 - momentum) * mean_v + momentum * mean; rv_f32[c] = (1.0 - momentum) * var_v + momentum * var; }
            }
            if training { unsafe { for c in 0..feat_dim { *running_mean.add(c) = T::from_f32(rm_f32[c]); *running_var.add(c) = T::from_f32(rv_f32[c]); } } }
            unsafe { for i in 0..total { *out.add(i) = T::from_f32(y_f32[i]); } }
        }
    }
    unsafe fn batchnorm1d_bwd(
        dy: *const T, dy_shape: &[usize],
        x_norm: *const T,
        gamma: *const T,
        dgamma: *mut T, dbeta: *mut T,
        dx: *mut T, dx_shape: &[usize],
        eps: f32,
    ) {
        let feat_dim = dy_shape[dy_shape.len() - 1];
        let outer: usize = dy_shape[..dy_shape.len() - 1].iter().product();
        let total = outer * feat_dim;
        // 用 dx 后半缓存 dgamma/dbeta（足够大）
        let dg_buf = dx as *mut f32;
        let db_buf = dx.add(feat_dim) as *mut f32;

        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && gpu_ffi::has_cuda_hardware() {
            unsafe {
                gpu_ffi::cudaMemset(dg_buf, 0, feat_dim * std::mem::size_of::<f32>());
                gpu_ffi::cudaMemset(db_buf, 0, feat_dim * std::mem::size_of::<f32>());
                gpu_ffi::oyta_batchnorm1d_bwd(
                    dy as *const f32, x_norm as *const f32, dx as *mut f32,
                    dg_buf, db_buf, gamma as *const f32,
                    feat_dim as i32, outer as i32, eps,
                );
                // 拷贝 dgamma/dbeta 到输出
                let dg_out = dgamma as *mut f32; let db_out = dbeta as *mut f32;
                for i in 0..feat_dim { *dg_out.add(i) = *dg_buf.add(i); *db_out.add(i) = *db_buf.add(i); }
                gpu_ffi::cudaDeviceSynchronize();
            }
        } else {
            // ======= CPU fallback =======
            let dy_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(dy, total) }.iter().map(|&v| v.to_f32()).collect();
            let xn_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(x_norm, total) }.iter().map(|&v| v.to_f32()).collect();
            let g_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(gamma, feat_dim) }.iter().map(|&v| v.to_f32()).collect();
            let mut dgamma_f32 = vec![0.0f32; feat_dim]; let mut dbeta_f32 = vec![0.0f32; feat_dim];
            let mut dx_f32 = vec![0.0f32; total];
            for c in 0..feat_dim {
                let mut dg: f64 = 0.0; let mut db: f64 = 0.0;
                for i in 0..outer { let dyv=dy_f32[i*feat_dim+c] as f64; db += dyv; dg += dyv * xn_f32[i*feat_dim+c] as f64; }
                dgamma_f32[c] = dg as f32; dbeta_f32[c] = db as f32;
                for i in 0..outer { dx_f32[i*feat_dim+c] = dy_f32[i*feat_dim+c] * g_f32[c]; }
            }
            unsafe {
                for i in 0..total { *dx.add(i) = T::from_f32(dx_f32[i]); }
                for i in 0..feat_dim { *dgamma.add(i) = T::from_f32(dgamma_f32[i]); *dbeta.add(i) = T::from_f32(dbeta_f32[i]); }
            }
        }
    }
}

impl<T: crate::dtype::float_trait::Float> crate::backend::pool::PoolOps<T> for Cuda {
    unsafe fn maxpool1d_fwd(
        x: *const T, x_shape: &[usize],
        out: *mut T, out_shape: &[usize],
        argmax_cache: *mut i64,
        kernel: usize, stride: usize, padding: usize,
    ) {
        let n=x_shape[0]; let c=x_shape[1]; let l_in=x_shape[2]; let l_out=out_shape[2]; let total=n*c*l_out;
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && gpu_ffi::has_cuda_hardware() {
            unsafe {
                gpu_ffi::oyta_maxpool1d_fwd(
                    x as *const f32, out as *mut f32, argmax_cache,
                    n as i32, c as i32, l_in as i32, l_out as i32, kernel as i32, stride as i32, padding as i32,
                );
                gpu_ffi::cudaDeviceSynchronize();
            }
        } else {
            let x_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(x,n*c*l_in) }.iter().map(|&v| v.to_f32()).collect();
            let mut out_f32 = vec![-f32::INFINITY; total];
            let argmax_slice = unsafe { std::slice::from_raw_parts_mut(argmax_cache, total) };
            // 串行 maxpool（逐元素 + argmax 缓存）
            for idx in 0..total {
                let lo=idx%l_out; let co=(idx/l_out)%c; let nn=idx/(c*l_out);
                let start=(lo*stride) as isize-padding as isize; let end=start+kernel as isize;
                let start=start.max(0) as usize; let end=end.min(l_in as isize) as usize;
                let mut mv=-f32::INFINITY; let mut mi: i64 = -1;
                for ki in start..end { let v=x_f32[(nn*c+co)*l_in+ki]; if v>mv { mv=v; mi=ki as i64; } }
                if mi<0 { mi=0; }
                out_f32[idx]=mv; argmax_slice[idx]=mi;
            }
            unsafe { for i in 0..total { *out.add(i)=T::from_f32(out_f32[i]); } }
        }
    }
    unsafe fn maxpool1d_bwd(
        dy: *const T, dy_shape: &[usize],
        argmax: *const i64,
        dx: *mut T, dx_shape: &[usize],
    ) {
        let n=dy_shape[0]; let c=dy_shape[1]; let l_out=dy_shape[2]; let l_in=dx_shape[2]; let total=n*c*l_out;
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && gpu_ffi::has_cuda_hardware() {
            unsafe {
                gpu_ffi::oyta_maxpool1d_bwd(
                    dy as *const f32, argmax, dx as *mut f32,
                    n as i32, c as i32, l_out as i32, l_in as i32,
                );
                gpu_ffi::cudaDeviceSynchronize();
            }
        } else {
            unsafe { for i in 0..n*c*l_in { *dx.add(i)=T::from_f32(0.0f32); } }
            let dy_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(dy,total) }.iter().map(|&v| v.to_f32()).collect();
            for i in 0..total { let am=unsafe { *argmax.add(i) }; if am>=0 { let co=(i/l_out)%c; let nn=i/(c*l_out); let p=unsafe { dx.add((nn*c+co)*l_in+am as usize) }; let cur=unsafe { *p }.to_f32(); unsafe { *p=T::from_f32(cur+dy_f32[i]); } } }
        }
    }
    unsafe fn avgpool1d_fwd(
        x: *const T, x_shape: &[usize],
        out: *mut T, out_shape: &[usize],
        kernel: usize, stride: usize, padding: usize,
    ) {
        let n=x_shape[0]; let c=x_shape[1]; let l_in=x_shape[2]; let l_out=out_shape[2]; let total=n*c*l_out;
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && gpu_ffi::has_cuda_hardware() {
            unsafe {
                gpu_ffi::oyta_avgpool1d_fwd(
                    x as *const f32, out as *mut f32,
                    n as i32, c as i32, l_in as i32, l_out as i32, kernel as i32, stride as i32, padding as i32,
                );
                gpu_ffi::cudaDeviceSynchronize();
            }
        } else {
            let x_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(x,n*c*l_in) }.iter().map(|&v| v.to_f32()).collect();
            let mut out_f32 = vec![0.0f32; total];
            // 串行 avgpool
            for idx in 0..total {
                let lo=idx%l_out; let co=(idx/l_out)%c; let nn=idx/(c*l_out);
                let start=(lo*stride) as isize-padding as isize; let end=start+kernel as isize;
                let start=start.max(0) as usize; let end=end.min(l_in as isize) as usize;
                let cnt=(end-start).max(1) as f32;
                let mut s: f64=0.0; for ki in start..end { s+=x_f32[(nn*c+co)*l_in+ki] as f64; }
                out_f32[idx]=(s/cnt as f64) as f32;
            }
            unsafe { for i in 0..total { *out.add(i)=T::from_f32(out_f32[i]); } }
        }
    }
    unsafe fn avgpool1d_bwd(
        dy: *const T, dy_shape: &[usize],
        dx: *mut T, dx_shape: &[usize],
        kernel: usize, stride: usize, padding: usize,
    ) {
        let n=dy_shape[0]; let c=dy_shape[1]; let l_out=dy_shape[2]; let l_in=dx_shape[2]; let total=n*c*l_out;
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && gpu_ffi::has_cuda_hardware() {
            unsafe {
                gpu_ffi::oyta_avgpool1d_bwd(
                    dy as *const f32, dx as *mut f32,
                    n as i32, c as i32, l_out as i32, l_in as i32, kernel as i32, stride as i32, padding as i32,
                );
                gpu_ffi::cudaDeviceSynchronize();
            }
        } else {
            unsafe { for i in 0..n*c*l_in { *dx.add(i)=T::from_f32(0.0f32); } }
            let dy_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(dy,total) }.iter().map(|&v| v.to_f32()).collect();
            for i in 0..total {
                let lo=i%l_out; let co=(i/l_out)%c; let nn=i/(c*l_out);
                let start=(lo*stride) as isize-padding as isize; let end=start+kernel as isize;
                let start=start.max(0) as usize; let end=end.min(l_in as isize) as usize;
                let cnt=(end-start).max(1) as f32; let d=dy_f32[i]/cnt;
                for ki in start..end { let p=unsafe { dx.add((nn*c+co)*l_in+ki) }; let cur=unsafe { *p }.to_f32(); unsafe { *p=T::from_f32(cur+d); } }
            }
        }
    }
}

impl<T: crate::dtype::float_trait::Float> crate::backend::rnn::RnnOps<T> for Cuda {
    unsafe fn gru_step(
        x: *const T, in_size: usize,
        h_prev: *const T, hidden_size: usize,
        h_new: *mut T,
        weights: *const T, bias: *const T,
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && gpu_ffi::has_cuda_hardware() {
            unsafe {
                gpu_ffi::oyta_gru_step(
                    x as *const f32, h_prev as *const f32, h_new as *mut f32,
                    weights as *const f32, bias as *const f32,
                    in_size as i32, hidden_size as i32,
                );
                gpu_ffi::cudaDeviceSynchronize();
            }
        } else {
            // ======= CPU fallback =======
            let hs = hidden_size; let is = in_size;
            for i in 0..hs {
                let mut z_sum: f64 = unsafe { *bias.add(i) }.to_f32() as f64;
                let mut r_sum: f64 = unsafe { *bias.add(hs + i) }.to_f32() as f64;
                let mut n_sum: f64 = unsafe { *bias.add(2 * hs + i) }.to_f32() as f64;
                for j in 0..is {
                    let xv = unsafe { *x.add(j) }.to_f32() as f64;
                    z_sum += unsafe { *weights.add(i * (is + hs) + j) }.to_f32() as f64 * xv;
                    r_sum += unsafe { *weights.add((hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
                    n_sum += unsafe { *weights.add((2 * hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
                }
                for j in 0..hs {
                    let hv = unsafe { *h_prev.add(j) }.to_f32() as f64;
                    z_sum += unsafe { *weights.add(i * (is + hs) + is + j) }.to_f32() as f64 * hv;
                    r_sum += unsafe { *weights.add((hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
                    n_sum += unsafe { *weights.add((2 * hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
                }
                let z_t = 1.0 / (1.0 + (-z_sum).exp());
                let r_t = 1.0 / (1.0 + (-r_sum).exp());
                let h_prev_v = unsafe { *h_prev.add(i) }.to_f32() as f64;
                let candidate = (n_sum + r_t * h_prev_v).tanh();
                unsafe { *h_new.add(i) = T::from_f32(((1.0 - z_t) * h_prev_v + z_t * candidate) as f32); }
            }
        }
    }
    unsafe fn lstm_step(
        x: *const T, in_size: usize,
        h_prev: *const T, c_prev: *const T, hidden_size: usize,
        h_new: *mut T, c_new: *mut T,
        weights: *const T, bias: *const T,
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && gpu_ffi::has_cuda_hardware() {
            unsafe {
                gpu_ffi::oyta_lstm_step(
                    x as *const f32, h_prev as *const f32, c_prev as *const f32,
                    h_new as *mut f32, c_new as *mut f32,
                    weights as *const f32, bias as *const f32,
                    in_size as i32, hidden_size as i32,
                );
                gpu_ffi::cudaDeviceSynchronize();
            }
        } else {
            // ======= CPU fallback =======
            let hs = hidden_size; let is = in_size;
            for i in 0..hs {
                let mut i_sum: f64 = unsafe { *bias.add(i) }.to_f32() as f64;
                let mut f_sum: f64 = unsafe { *bias.add(hs + i) }.to_f32() as f64;
                let mut g_sum: f64 = unsafe { *bias.add(2 * hs + i) }.to_f32() as f64;
                let mut o_sum: f64 = unsafe { *bias.add(3 * hs + i) }.to_f32() as f64;
                for j in 0..is {
                    let xv = unsafe { *x.add(j) }.to_f32() as f64;
                    i_sum += unsafe { *weights.add(i * (is + hs) + j) }.to_f32() as f64 * xv;
                    f_sum += unsafe { *weights.add((hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
                    g_sum += unsafe { *weights.add((2 * hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
                    o_sum += unsafe { *weights.add((3 * hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
                }
                for j in 0..hs {
                    let hv = unsafe { *h_prev.add(j) }.to_f32() as f64;
                    i_sum += unsafe { *weights.add(i * (is + hs) + is + j) }.to_f32() as f64 * hv;
                    f_sum += unsafe { *weights.add((hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
                    g_sum += unsafe { *weights.add((2 * hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
                    o_sum += unsafe { *weights.add((3 * hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
                }
                let i_gate = 1.0 / (1.0 + (-i_sum).exp());
                let f_gate = 1.0 / (1.0 + (-f_sum).exp());
                let g_gate = g_sum.tanh();
                let o_gate = 1.0 / (1.0 + (-o_sum).exp());
                let c_prev_v = unsafe { *c_prev.add(i) }.to_f32() as f64;
                let c_new_v = f_gate * c_prev_v + i_gate * g_gate;
                let h_new_v = o_gate * c_new_v.tanh();
                unsafe { *c_new.add(i) = T::from_f32(c_new_v as f32); *h_new.add(i) = T::from_f32(h_new_v as f32); }
            }
        }
    }
}

// ======== Backend trait ========
#[cfg(feature = "cuda")]
impl Backend for Cuda {
    const IS_CPU: bool = true;
    const MAX_ACCUMULATOR_PRECISION: Precision = Precision::F32;
    const HAS_FP16_TENSOR_CORE: bool = false;  // 需 runtime detect
    const HAS_BF16: bool = false;
    const HAS_BATCHED_GEMM: bool = true;
    const HAS_FLASH_ATTENTION_HW: bool = false;
    const DEVICE_NAME: &'static str = "NVIDIA CUDA";
}

// ======== DeviceContext：runtime detect 真实 CUDA vs CPU fallback ========
#[cfg(feature = "cuda")]
impl DeviceContext for Cuda {
    fn init() -> Result<DeviceHandle<Cuda>, DeviceError> {
        static CUDA_CTX: OnceLock<DeviceHandle<Cuda>> = OnceLock::new();
        Ok(CUDA_CTX.get_or_init(|| {
            let has = gpu_ffi::has_cuda_hardware();
            if has {
                // 真实 CUDA：初始化 cublas handle + stream
                let cublas_handle: u64 = unsafe {
                    let mut h = 0u64;
                    let _ = gpu_ffi::cublasCreate(&mut h);
                    h
                };
                let stream: u64 = unsafe {
                    let mut s = 0u64;
                    let _ = gpu_ffi::cudaStreamCreate(&mut s);
                    s
                };
                // 绑定 stream 到 cublas handle
                unsafe { let _ = gpu_ffi::cublasSetStream(cublas_handle, stream); }
                DeviceHandle {
                    properties: DeviceProperties {
                        name: "NVIDIA CUDA (GPU)".into(),
                        device_id: 0,
                        memory_bytes: 0,
                        compute_capability: "unknown".into(),
                        num_cores: gpu_ffi::cuda_device_count() as usize,
                        supports_fp16: true,
                        supports_bf16: true,
                        supports_fp8: false,
                        supports_tensor_core: true,
                    },
                    _device: PhantomData,
                    inner: DeviceInner::Cuda {
                        cublas_handle, cudnn_handle: None, stream, device_id: 0,
                    },
                }
            } else {
                // CPU fallback：IS_CPU=true 强制 CPU rayon 路径
                DeviceHandle {
                    properties: DeviceProperties {
                        name: "NVIDIA CUDA (CPU fallback - 无硬件)".into(),
                        device_id: 0,
                        memory_bytes: 0,
                        compute_capability: "CPU fallback".into(),
                        num_cores: rayon::current_num_threads(),
                        supports_fp16: false, supports_bf16: false,
                        supports_fp8: false, supports_tensor_core: false,
                    },
                    _device: PhantomData,
                    inner: DeviceInner::Cuda {
                        cublas_handle: 0, cudnn_handle: None, stream: 0, device_id: 0,
                    },
                }
            }
        }).clone())
    }

    fn get() -> DeviceHandle<Cuda> {
        Self::init().expect("Cuda DeviceContext 初始化失败")
    }

    fn shutdown() {
        // OnceLock 单例 drop 时自动清理（如果有 cublas handle）
    }
}

// ======== MatmulOps 真实实现 ========
#[cfg(feature = "cuda")]
impl MatmulOps for Cuda {
    /// matmul：有 GPU 硬件走 cublasSgemm，否则 fallback rayon 默认实现
    ///
    /// cuBLAS row-major trick：
    ///   我们要算 row-major C[m,n] = A[m,k] * B[k,n]
    ///   cuBLAS 原生于 column-major，所以等价于 column-major 的：
    ///     C^T[n,m] = B^T[n,k] * A^T[k,m]
    ///   → cublasSgemm(N, N, n, m, k, B_ptr, n, A_ptr, k, C_ptr, n)
    /// 参数映射：cublas 的 m=n, n=m, k=k, A=B_ptr, lda=n, B=A_ptr, ldb=k, C=C_ptr, ldc=n
    fn matmul(
        a_ptr: *const f32, lda: usize,
        b_ptr: *const f32, ldb: usize,
        c_ptr: *mut f32, ldc: usize,
        m: usize, n: usize, k: usize,
        alpha: f32, beta: f32,
    ) {
        // Runtime detect：有 GPU + 有效 handle 时走 cublas
        if gpu_ffi::has_cuda_hardware() {
            let h = <Cuda as DeviceContext>::get();
            if let DeviceInner::Cuda { cublas_handle, .. } = &h.inner {
                let handle = *cublas_handle;
                if handle != 0 {
                    unsafe {
                        // cuBLAS row-major trick 需要 contiguous 内存
                        // 先检查 lda/ldb/ldc 是否刚好等于 k/n/n
                        let alpha_ref = &alpha as *const f32;
                        let beta_ref = &beta as *const f32;

                        if lda == k && ldb == n && ldc == n {
                            // 完美 contiguous：直接调 cublasSgemm
                            let _ = gpu_ffi::cublasSgemm(
                                handle,
                                gpu_ffi::CUBLAS_OP_N, gpu_ffi::CUBLAS_OP_N,
                                n as i32, m as i32, k as i32,  // 注意：m/n 交换！
                                alpha_ref,
                                b_ptr, n as i32,     // B 是 cuBLAS 的第一个矩阵（col-major B^T）
                                a_ptr, k as i32,     // A 是 cuBLAS 的第二个矩阵
                                beta_ref,
                                c_ptr, n as i32,
                            );
                            let _ = gpu_ffi::cudaDeviceSynchronize();
                            return;
                        } else {
                            // stride 情况：先 copy 成 contiguous row-major
                            // 调用者必须保证 a/b/c 各有 lda*m / ldb*k / ldc*m 个有效元素
                            let mk = crate::checked::checked_mul(m, k);
                            let kn = crate::checked::checked_mul(k, n);
                            let mn = crate::checked::checked_mul(m, n);
                            let mlda = crate::checked::checked_mul(m, lda);
                            let kldb = crate::checked::checked_mul(k, ldb);
                            let mldc = crate::checked::checked_mul(m, ldc);
                            let a_contig: Vec<f32> = if lda == k {
                                std::slice::from_raw_parts(a_ptr, mk).to_vec()
                            } else {
                                let mut v = vec![0.0f32; mk];
                                let a_s = std::slice::from_raw_parts(a_ptr, mlda);
                                for i in 0..m { v[i * k .. (i + 1) * k].copy_from_slice(&a_s[i * lda .. i * lda + k]); }
                                v
                            };
                            let b_contig: Vec<f32> = if ldb == n {
                                std::slice::from_raw_parts(b_ptr, kn).to_vec()
                            } else {
                                let mut v = vec![0.0f32; kn];
                                let b_s = std::slice::from_raw_parts(b_ptr, kldb);
                                for i in 0..k { v[i * n .. (i + 1) * n].copy_from_slice(&b_s[i * ldb .. i * ldb + n]); }
                                v
                            };
                            let mut c_local: Vec<f32> = if beta != 0.0 {
                                if ldc == n {
                                    std::slice::from_raw_parts(c_ptr, mn).to_vec()
                                } else {
                                    let mut v = vec![0.0f32; mn];
                                    let c_s = std::slice::from_raw_parts(c_ptr, mldc);
                                    for i in 0..m { v[i * n .. (i + 1) * n].copy_from_slice(&c_s[i * ldc .. i * ldc + n]); }
                                    v
                                }
                            } else {
                                vec![0.0f32; mn]
                            };

                            let _ = gpu_ffi::cublasSgemm(
                                handle,
                                gpu_ffi::CUBLAS_OP_N, gpu_ffi::CUBLAS_OP_N,
                                n as i32, m as i32, k as i32,
                                alpha_ref,
                                b_contig.as_ptr(), n as i32,
                                a_contig.as_ptr(), k as i32,
                                beta_ref,
                                c_local.as_mut_ptr(), n as i32,
                            );
                            let _ = gpu_ffi::cudaDeviceSynchronize();

                            // 写回（处理 ldc != n）
                            let c_out = std::slice::from_raw_parts_mut(c_ptr, mldc);
                            if ldc == n {
                                c_out.copy_from_slice(&c_local);
                            } else {
                                for i in 0..m { c_out[i * ldc .. i * ldc + n].copy_from_slice(&c_local[i * n .. (i + 1) * n]); }
                            }
                            return;
                        }
                    } // end unsafe
                } // end if handle != 0
            } // end if let DeviceInner::Cuda
        } // end if has_cuda_hardware
        // CPU fallback：继承 rayon 默认实现（Naive + TiledFma + Strassen）
        <Self as MatmulOps>::default_matmul(
            a_ptr, lda, b_ptr, ldb, c_ptr, ldc, m, n, k, alpha, beta,
        );
    }
}
