//! matmul.rs：矩阵乘法（naive→tiled_fma BM=BN=BK=128→strassen 7 公式递归）
//!
//! #[cfg(feature = "blas")] 时，大矩阵（m,n,k >= 64）优先走 cblas_sgemm。

use crate::backend::base::BackendBase;
use crate::dtype::{F16, BF16};

// 让 cargo 确认 blas-sys 这个 optional dep 被 feature = "blas" 使用，
// 避免 unused_dependencies manifest warning。运行时链接 blas-sys 提供的 cblas_sgemm 符号。
#[cfg(feature = "blas")]
use blas_sys as _;

// ============================================================
// BLAS sgemm：OpenBLAS / MKL 加速（feature gate）
// ============================================================

/// cblas_sgemm 常量（row-major + 不转置）
#[cfg(feature = "blas")]
const CBLAS_ORDER_ROW_MAJOR: i32 = 101;  // OpenBLAS 行主序
#[cfg(feature = "blas")]
const CBLAS_NO_TRANSPOSE: i32 = 111;     // 不转置 A/B

/// 真实 cblas_sgemm BLAS 加速路径（feature gate）
///
/// 调用 OpenBLAS/MKL 的 sgemm：C = α·A·B + β·C
/// row-major 下用 CBLAS_ORDER_ROW_MAJOR + 不转置，直接传 lda/ldb/ldc。
#[cfg(feature = "blas")]
#[inline]
fn matmul_blas_sgemm(
    a_ptr: *const f32, lda: usize,
    _b_ptr: *const f32, _ldb: usize,
    c_ptr: *mut f32, ldc: usize,
    m: usize, n: usize, k: usize,
    alpha: f32, beta: f32,
) {
    unsafe extern "C" {
        fn cblas_sgemm(
            order: i32, transa: i32, transb: i32,
            m: i32, n: i32, k: i32,
            alpha: *const f32,
            a: *const f32, lda: i32,
            beta: *const f32,
            c: *mut f32, ldc: i32,
        );
    }
    unsafe {
        cblas_sgemm(
            CBLAS_ORDER_ROW_MAJOR, CBLAS_NO_TRANSPOSE, CBLAS_NO_TRANSPOSE,
            m as i32, n as i32, k as i32,
            &alpha, a_ptr, lda as i32,
            &beta, c_ptr, ldc as i32,
        );
    }
}

// ============================================================
// MatmulOps 辅助函数（Tiling + FMA + Strassen 完整实现）
// ============================================================

/// Tiled matmul 分块大小（匹配 L2 缓存）
/// 三个分块 A_block + B_block + C_block ≈ 192KB ≤ L2 256KB
const MATMUL_BM: usize = 128;
const MATMUL_BN: usize = 128;
const MATMUL_BK: usize = 128;

/// 小矩阵阈值（≤32×32 naive 比分块快）
const NAIVE_THRESHOLD: usize = 32;

/// Strassen 交叉点（<128 回退到 tiled_fma）
const STRASSEN_CROSSOVER: usize = 128;

/// 朴素三重循环 matmul（极小矩阵用，编译器自动 FMA + 标量向量化）
#[inline]
fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum: f32 = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Tiled + FMA + 多线程分块 matmul（CPU 主力）
fn matmul_tiled_fma(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let num_m = (m + MATMUL_BM - 1) / MATMUL_BM;
    let num_n = (n + MATMUL_BN - 1) / MATMUL_BN;
    let num_k = (k + MATMUL_BK - 1) / MATMUL_BK;
    let total_blocks = num_m * num_n;
    
    let num_threads = rayon::current_num_threads().max(1).min(total_blocks);
    
    // 装箱裸指针为 usize（在线程间传递 Send 安全，因为块不重叠）
    let a_addr = a.as_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    let c_addr = c.as_mut_ptr() as usize;
    
    std::thread::scope(|s| {
        let blocks_per_thread = (total_blocks + num_threads - 1) / num_threads;
        for thread_idx in 0..num_threads {
            let a_a = a_addr; let b_a = b_addr; let c_a = c_addr;
            s.spawn(move || {
                let a = a_a as *const f32;
                let b = b_a as *const f32;
                let c = c_a as *mut f32;
                let start = thread_idx * blocks_per_thread;
                let end = (start + blocks_per_thread).min(total_blocks);
                for flat in start..end {
                    let mi = flat / num_n;
                    let ni = flat % num_n;
                    let ma = mi * MATMUL_BM;
                    let mb = (ma + MATMUL_BM).min(m);
                    let na = ni * MATMUL_BN;
                    let nb = (na + MATMUL_BN).min(n);
                    
                    for ki in 0..num_k {
                        let ka = ki * MATMUL_BK;
                        let kb = (ka + MATMUL_BK).min(k);
                        let is_first_k = ki == 0;
                        
                        for i in ma..mb {
                            for j in na..nb {
                                let mut sum = if is_first_k { 0.0 } else {
                                    unsafe { *c.add(i * n + j) }
                                };
                                for p in ka..kb {
                                    sum = unsafe { *a.add(i * k + p) } * unsafe { *b.add(p * n + j) } + sum;
                                }
                                unsafe { *c.add(i * n + j) = sum; }
                            }
                        }
                    }
                }
            });
        }
    });
}

/// Strassen 完整递归 matmul（7 公式 + 交叉点回退）
fn matmul_strassen_recursive(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let m_n_size = crate::checked::checked_mul(m, n);

    // 终止条件：任何维度 < 交叉点 → 直接 tiled_fma
    if m < STRASSEN_CROSSOVER || n < STRASSEN_CROSSOVER || k < STRASSEN_CROSSOVER {
        let mut c = vec![0.0f32; m_n_size];
        matmul_tiled_fma(a, b, &mut c, m, n, k);
        return c;
    }
    
    let m_half = m / 2;
    let n_half = n / 2;
    let k_half = k / 2;
    
    if m_half == 0 || n_half == 0 || k_half == 0 {
        let mut c = vec![0.0f32; m_n_size];
        matmul_tiled_fma(a, b, &mut c, m, n, k);
        return c;
    }
    
    // A, B 切成 2×2 块（直接从原始数组切片，避免拷贝）
    let a11 = &a[..m_half * k_half];
    let a12 = &a[m_half * k_half .. m_half * k][..m_half * (k - k_half)];
    let a21 = &a[m * k_half .. (m_half + m_half) * k_half];
    let a22 = &a[m * k_half ..][m_half * (k - k_half) .. m_half * (k - k_half) + m_half * (k - k_half)];
    let b11 = &b[..k_half * n_half];
    let b12 = &b[k_half * n_half .. k_half * n][..k_half * (n - n_half)];
    let b21 = &b[k_half * n .. k_half * n + k_half * n_half];
    let b22 = &b[k_half * n ..][k_half * (n - n_half) ..];
    
    // Strassen 7 公式
    let vadd = |x: &[f32], y: &[f32]| -> Vec<f32> { x.iter().zip(y.iter()).map(|(&a, &b)| a + b).collect() };
    let vsub = |x: &[f32], y: &[f32]| -> Vec<f32> { x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect() };
    
    let m1 = matmul_strassen_recursive(&vadd(a11, a22), &vadd(b11, b22), m_half, n_half, k_half);
    let m2 = matmul_strassen_recursive(&vadd(a21, a22), b11, m_half, n_half, k_half);
    let m3 = matmul_strassen_recursive(a11, &vsub(b12, b22), m_half, n_half, k_half);
    let m4 = matmul_strassen_recursive(a22, &vsub(b21, b11), m_half, n_half, k_half);
    let m5 = matmul_strassen_recursive(&vadd(a11, a12), b22, m_half, n_half, k_half);
    let m6 = matmul_strassen_recursive(&vsub(a21, a11), &vadd(b11, b12), m_half, n_half, k_half);
    let m7 = matmul_strassen_recursive(&vsub(a12, a22), &vadd(b21, b22), m_half, n_half, k_half);
    
    // C11 = M1 + M4 - M5 + M7
    // C12 = M3 + M5
    // C21 = M2 + M4
    // C22 = M1 - M2 + M3 + M6
    let mut c = vec![0.0f32; m_n_size];
    for i in 0..m_half {
        for j in 0..n_half {
            c[i * n + j] = m1[i * n_half + j] + m4[i * n_half + j] - m5[i * n_half + j] + m7[i * n_half + j];
        }
        for j in 0..n - n_half {
            c[i * n + n_half + j] = m3[i * (n - n_half) + j] + m5[i * (n - n_half) + j];
        }
    }
    for i in 0..m - m_half {
        for j in 0..n_half {
            c[(m_half + i) * n + j] = m2[i * n_half + j] + m4[i * n_half + j];
        }
        for j in 0..n - n_half {
            c[(m_half + i) * n + n_half + j] = m1[i * (n - n_half) + j] - m2[i * (n - n_half) + j] + m3[i * (n - n_half) + j] + m6[i * (n - n_half) + j];
        }
    }
    c
}

/// Strassen 算法的交叉点（超过此尺寸用 Strassen）
pub const STRASSEN_CUTOFF: usize = 512;

/// 动态算法选择器（小矩阵 naive / 中等 tiled_fma / 大矩阵 strassen）
#[inline]
fn matmul_dispatch(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
    alpha: f32, beta: f32,
) {
    if alpha == 1.0 && beta == 0.0 {
        if m <= NAIVE_THRESHOLD && n <= NAIVE_THRESHOLD && k <= NAIVE_THRESHOLD {
            matmul_naive(a, b, c, m, n, k);
        } else if m >= STRASSEN_CUTOFF && n >= STRASSEN_CUTOFF && k >= STRASSEN_CUTOFF {
            let out = matmul_strassen_recursive(a, b, m, n, k);
            c.copy_from_slice(&out);
        } else {
            matmul_tiled_fma(a, b, c, m, n, k);
        }
    } else {
        // 通用：先算 αAB，再 + βC
        let tmp_size = crate::checked::checked_mul(m, n);
        let mut tmp = vec![0.0f32; tmp_size];
        matmul_tiled_fma(a, b, &mut tmp, m, n, k);
        for i in 0..tmp_size { c[i] = alpha * tmp[i] + beta * c[i]; }
    }
}

/// 矩阵乘法算子（默认实现动态选择 naive / tiled_fma / strassen）
pub trait MatmulOps: BackendBase {
    /// 单矩阵乘法：C[m,n] = alpha * A[m,k] × B[k,n] + beta * C[m,n]
    fn matmul(a: *const f32, lda: usize, b: *const f32, ldb: usize,
              c: *mut f32, ldc: usize, m: usize, n: usize, k: usize,
              alpha: f32, beta: f32) {
        Self::default_matmul(a, lda, b, ldb, c, ldc, m, n, k, alpha, beta);
    }

    /// CPU 默认 matmul 实现（rayon tiled_fma + 可选 BLAS + Strassen）
    /// 暴露给 GPU 后端在 IS_CPU=true 时调用作为 fallback
    fn default_matmul(
        a: *const f32, lda: usize,
        b: *const f32, ldb: usize,
        c: *mut f32, ldc: usize,
        m: usize, n: usize, k: usize,
        alpha: f32, beta: f32,
    ) {
        // === 边界检查（仅 debug）===
        #[cfg(debug_assertions)]
        {
            debug_assert!(m > 0 && n > 0 && k > 0, "matmul 维度必须 > 0: m={} n={} k={}", m, n, k);
            debug_assert!(lda >= k, "matmul lda({}) 必须 >= k({}) (row-major)", lda, k);
            debug_assert!(ldb >= n, "matmul ldb({}) 必须 >= n({}) (row-major)", ldb, n);
            debug_assert!(ldc >= n, "matmul ldc({}) 必须 >= n({}) (row-major)", ldc, n);
        }
        // === 溢出防御：所有 from_raw_parts 长度参数 checked ===
        let a_len = crate::checked::checked_mul(m, lda);
        let b_len = crate::checked::checked_mul(k, ldb);
        let c_len = crate::checked::checked_mul(m, ldc);
        let a_in = unsafe { std::slice::from_raw_parts(a, a_len) };
        let b_in = unsafe { std::slice::from_raw_parts(b, b_len) };
        let c_in_out = unsafe { std::slice::from_raw_parts_mut(c, c_len) };
        
        // 如果 lda/ldb 刚好是 k/n（contiguous），直接用；否则先 copy 成 contiguous
        let a_contig: Vec<f32> = if lda == k {
            a_in.to_vec()
        } else {
            let mut v = vec![0.0f32; m * k];
            for i in 0..m { v[i * k .. (i + 1) * k].copy_from_slice(&a_in[i * lda .. i * lda + k]); }
            v
        };
        let b_contig: Vec<f32> = if ldb == n {
            b_in.to_vec()
        } else {
            let mut v = vec![0.0f32; k * n];
            for i in 0..k { v[i * n .. (i + 1) * n].copy_from_slice(&b_in[i * ldb .. i * ldb + n]); }
            v
        };
        
        // C 先 copy 出来
        let mut c_local: Vec<f32> = if ldc == n {
            c_in_out.to_vec()
        } else {
            let mut v = vec![0.0f32; m * n];
            for i in 0..m { v[i * n .. (i + 1) * n].copy_from_slice(&c_in_out[i * ldc .. i * ldc + n]); }
            v
        };

        // BLAS 快速路径：m,n,k ≥ 64 时直接走 cblas_sgemm（OpenBLAS/MKL 多线程 SIMD）
        #[cfg(feature = "blas")]
        if m >= 64 && n >= 64 && k >= 64 {
            matmul_blas_sgemm(
                a_contig.as_ptr(), k,
                b_contig.as_ptr(), n,
                c_local.as_mut_ptr(), n,
                m, n, k,
                alpha, beta,
            );
        }

        #[cfg(not(feature = "blas"))]
        matmul_dispatch(&a_contig, &b_contig, &mut c_local, m, n, k, alpha, beta);

        #[cfg(feature = "blas")]
        if !(m >= 64 && n >= 64 && k >= 64) {
            matmul_dispatch(&a_contig, &b_contig, &mut c_local, m, n, k, alpha, beta);
        }

        // 写回（处理 ldc != n）
        if ldc == n {
            c_in_out.copy_from_slice(&c_local);
        } else {
            for i in 0..m { c_in_out[i * ldc .. i * ldc + n].copy_from_slice(&c_local[i * n .. (i + 1) * n]); }
        }
    }
    
    /// batched matmul（所有 batch 元素共享相同尺寸）
    fn matmul_batched(a: *const f32, lda: usize, stride_a: isize,
                      b: *const f32, ldb: usize, stride_b: isize,
                      c: *mut f32, ldc: usize, stride_c: isize,
                      m: usize, n: usize, k: usize, batch: usize,
                      alpha: f32, beta: f32) {
        // 默认实现：循环调单 matmul（CPU 上够用）
        // GPU 设备重写此方法用 cublasGemmStridedBatchedEx
        for bi in 0..batch {
            let a_ptr = unsafe { (a as *const u8).offset(bi as isize * stride_a) as *const f32 };
            let b_ptr = unsafe { (b as *const u8).offset(bi as isize * stride_b) as *const f32 };
            let c_ptr = unsafe { (c as *mut u8).offset(bi as isize * stride_c) as *mut f32 };
            Self::matmul(a_ptr, lda, b_ptr, ldb, c_ptr, ldc, m, n, k, alpha, beta);
        }
    }
    
    /// 带 transpose 的 matmul
    fn matmul_t(a: *const f32, lda: usize, transpose_a: bool,
                b: *const f32, ldb: usize, transpose_b: bool,
                c: *mut f32, ldc: usize, m: usize, n: usize, k: usize,
                alpha: f32, beta: f32) {
        let (m_eff, k_eff, a_lda) = if transpose_a { (k, m, lda) } else { (m, k, lda) };
        let (k2_eff, n_eff, b_ldb) = if transpose_b { (n, k, ldb) } else { (k, n, ldb) };
        let _ = (m_eff, k_eff, k2_eff, n_eff);
        // 默认实现：先转置成 contiguous 再调 matmul
        let mn_prod = crate::checked::checked_mul(m, k);
        let kn_prod = crate::checked::checked_mul(k, n);
        let mut a_contig: Vec<f32> = vec![0.0; mn_prod];
        let mut b_contig: Vec<f32> = vec![0.0; kn_prod];
        // 调用者必须保证 a 指向 lda*max(m,k) 个元素的连续内存，b 同理
        let a_need = crate::checked::checked_mul(lda, if transpose_a { k } else { m });
        let b_need = crate::checked::checked_mul(ldb, if transpose_b { n } else { k });
        let a_s = unsafe { std::slice::from_raw_parts(a, a_need) };
        let b_s = unsafe { std::slice::from_raw_parts(b, b_need) };
        
        if transpose_a {
            for i in 0..k { for j in 0..m { a_contig[j * k + i] = a_s[i * lda + j]; } }
        } else {
            for i in 0..m { for j in 0..k { a_contig[i * k + j] = a_s[i * lda + j]; } }
        }
        if transpose_b {
            for i in 0..n { for j in 0..k { b_contig[j * n + i] = b_s[i * ldb + j]; } }
        } else {
            for i in 0..k { for j in 0..n { b_contig[i * n + j] = b_s[i * ldb + j]; } }
        }
        Self::matmul(a_contig.as_ptr(), k, b_contig.as_ptr(), n, c, ldc, m, n, k, alpha, beta);
        let _ = a_lda; let _ = b_ldb; // 避免警告
    }
    
    /// f16 matmul（CPU 默认 cast→f32→算→cast 回，GPU 用 Tensor Core）
    fn matmul_f16(a: *const F16, lda: usize, b: *const F16, ldb: usize,
                  c: *mut F16, ldc: usize, m: usize, n: usize, k: usize,
                  alpha: f32, beta: f32) {
        // 调用者必须保证 a/b/c 各有 lda*m / ldb*k / ldc*m 个有效元素
        let a_len = crate::checked::checked_mul(m, lda);
        let b_len = crate::checked::checked_mul(k, ldb);
        let c_len = crate::checked::checked_mul(m, ldc);
        let a_f32: Vec<f32> = (0..a_len).map(|i| unsafe { (*a.add(i)).to_f32() }).collect();
        let b_f32: Vec<f32> = (0..b_len).map(|i| unsafe { (*b.add(i)).to_f32() }).collect();
        let mut c_f32: Vec<f32> = (0..c_len).map(|i| unsafe { (*c.add(i)).to_f32() }).collect();
        Self::matmul(a_f32.as_ptr(), lda, b_f32.as_ptr(), ldb, c_f32.as_mut_ptr(), ldc, m, n, k, alpha, beta);
        unsafe { for i in 0..c_len { *c.add(i) = F16::from_f32(c_f32[i]); } }
    }
    
    /// bf16 matmul（CPU 默认 cast→f32，GPU 重写用硬件 BF16）
    fn matmul_bf16(a: *const BF16, lda: usize, b: *const BF16, ldb: usize,
                   c: *mut BF16, ldc: usize, m: usize, n: usize, k: usize,
                   alpha: f32, beta: f32) {
        // 调用者必须保证 a/b/c 各有 lda*m / ldb*k / ldc*m 个有效元素
        let a_len = crate::checked::checked_mul(m, lda);
        let b_len = crate::checked::checked_mul(k, ldb);
        let c_len = crate::checked::checked_mul(m, ldc);
        let a_f32: Vec<f32> = (0..a_len).map(|i| unsafe { (*a.add(i)).to_f32() }).collect();
        let b_f32: Vec<f32> = (0..b_len).map(|i| unsafe { (*b.add(i)).to_f32() }).collect();
        let mut c_f32: Vec<f32> = (0..c_len).map(|i| unsafe { (*c.add(i)).to_f32() }).collect();
        Self::matmul(a_f32.as_ptr(), lda, b_f32.as_ptr(), ldb, c_f32.as_mut_ptr(), ldc, m, n, k, alpha, beta);
        unsafe { for i in 0..c_len { *c.add(i) = BF16::from_f32(c_f32[i]); } }
    }
}

/// Strassen 算法 matmul（递归 2×2 分块，交叉点以下回退朴素）
pub fn matmul_strassen<D: crate::backend::aggregate::Backend>(
    a: *const f32, lda: usize,
    b: *const f32, ldb: usize,
    c: *mut f32, ldc: usize,
    m: usize, n: usize, k: usize,
    alpha: f32, beta: f32,
) {
    // === 边界检查 ===
    #[cfg(debug_assertions)]
    {
        debug_assert!(m > 0 && n > 0 && k > 0, "matmul_strassen 维度必须 > 0: m={} n={} k={}", m, n, k);
        debug_assert!(lda >= k, "matmul_strassen lda({}) 必须 >= k({})", lda, k);
        debug_assert!(ldb >= n, "matmul_strassen ldb({}) 必须 >= n({})", ldb, n);
        debug_assert!(ldc >= n, "matmul_strassen ldc({}) 必须 >= n({})", ldc, n);
    }
    // 交叉点以下回退默认 Backend::matmul
    if m < STRASSEN_CUTOFF || n < STRASSEN_CUTOFF || k < STRASSEN_CUTOFF {
        return D::matmul(a, lda, b, ldb, c, ldc, m, n, k, alpha, beta);
    }

    // 先把 A/B/C 转成 contiguous Vec
    // 调用者必须保证 a/b/c 指向 lda/ldb/ldc 步长的有效矩阵内存
    let a_need = crate::checked::checked_mul(m, lda);
    let b_need = crate::checked::checked_mul(k, ldb);
    let c_need = crate::checked::checked_mul(m, ldc);
    let a_in = unsafe { std::slice::from_raw_parts(a, a_need) };
    let b_in = unsafe { std::slice::from_raw_parts(b, b_need) };
    let c_in = unsafe { std::slice::from_raw_parts(c, c_need) };

    let a_contig: Vec<f32> = if lda == k {
        a_in[..m * k].to_vec()
    } else {
        let mut v = vec![0.0f32; m * k];
        for i in 0..m { v[i * k .. (i + 1) * k].copy_from_slice(&a_in[i * lda .. i * lda + k]); }
        v
    };
    let b_contig: Vec<f32> = if ldb == n {
        b_in[..k * n].to_vec()
    } else {
        let mut v = vec![0.0f32; k * n];
        for i in 0..k { v[i * n .. (i + 1) * n].copy_from_slice(&b_in[i * ldb .. i * ldb + n]); }
        v
    };
    let c_local: Vec<f32> = if beta != 0.0 {
        if ldc == n {
            c_in[..m * n].to_vec()
        } else {
            let mut v = vec![0.0f32; m * n];
            for i in 0..m { v[i * n .. (i + 1) * n].copy_from_slice(&c_in[i * ldc .. i * ldc + n]); }
            v
        }
    } else {
        vec![0.0f32; m * n]
    };

    // 调用已有的完整 7 公式递归 Strassen
    let out = matmul_strassen_recursive(&a_contig, &b_contig, m, n, k);

    // 写回：alpha * AB + beta * C
    let mut result = if beta != 0.0 {
        c_local.into_iter().map(|v| beta * v).collect::<Vec<_>>()
    } else {
        vec![0.0f32; m * n]
    };
    for i in 0..m * n { result[i] += alpha * out[i]; }

    // 写回非 contiguous 的情况
    let c_out_need = crate::checked::checked_mul(m, ldc);
    let c_out = unsafe { std::slice::from_raw_parts_mut(c, c_out_need) };
    if ldc == n {
        c_out[..m * n].copy_from_slice(&result);
    } else {
        for i in 0..m { c_out[i * ldc .. i * ldc + n].copy_from_slice(&result[i * n .. (i + 1) * n]); }
    }
}
