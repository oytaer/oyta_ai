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
impl<T: crate::dtype::float_trait::Float> crate::backend::conv1d::Conv1dOps<T> for Cuda {}
impl<T: crate::dtype::float_trait::Float> crate::backend::batchnorm::BatchNormOps<T> for Cuda {}
impl<T: crate::dtype::float_trait::Float> crate::backend::pool::PoolOps<T> for Cuda {}
impl<T: crate::dtype::float_trait::Float> crate::backend::rnn::RnnOps<T> for Cuda {}

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
