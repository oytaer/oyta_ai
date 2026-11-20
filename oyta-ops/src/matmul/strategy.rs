//! matmul：矩阵乘法算子（§7.2 + §8.1 + §30.1）
//!
//! 完整实现：
//! - matmul: 完整算法选择器（小矩阵 naive / 大矩阵 strassen / 默认 Backend::matmul）
//! - matmul_t: 带可选转置的矩阵乘法
//! - matmul_naive: 朴素三重循环（小矩阵快）
//! - matmul_tiled_fma: Tiled + FMA + rayon 分块（CPU 主力）
//! - matmul_strassen: 7 公式完整递归（大矩阵 O(n^log₂7)）
//! - dot_product_flat: 精确点积（1×1 matmul 特化）
//! - move_dim_to_last / move_dim_back: 维度置换工具函数

use oyta_core::{Tensor, DynamicShape};
use oyta_device::Backend;

// ============================================================
// 常量
// ============================================================

/// Tiled matmul 分块大小（匹配 L2 缓存）
const MATMUL_BM: usize = 128;
const MATMUL_BN: usize = 128;
const MATMUL_BK: usize = 128;

/// 小矩阵阈值（≤32×32 naive 比分块快）
const NAIVE_THRESHOLD: usize = 32;

/// Strassen 交叉点（<128 回退到 tiled_fma）
const STRASSEN_CROSSOVER: usize = 128;

/// 进入 Strassen 模式的大矩阵阈值
const STRASSEN_ENTRY: usize = 1024;

// ============================================================
// move_dim_to_last / move_dim_back（与 reduce.rs 共享实现，这里公开）
// ============================================================

/// 把指定 dim 移动到最后一维（permute 零拷贝）
///
/// # 用途
/// reduce / softmax / matmul_batched 里都需要把某个目标维移到末尾
pub fn move_dim_to_last<T: Clone, D: Backend>(
    tensor: &Tensor<T, D>, dim: usize,
) -> Tensor<T, D> {
    let rank = tensor.rank();
    #[cfg(debug_assertions)]
    debug_assert!(dim < rank, "move_dim_to_last: dim {} 越界", dim);
    if dim == rank - 1 { return tensor.clone(); }
    let mut order = Vec::with_capacity(rank);
    for i in 0..rank {
        if i != dim { order.push(i); }
    }
    order.push(dim);
    tensor.permute(&order)
}

/// 把末尾维移回 original_dim 位置（move_dim_to_last 的逆）
pub fn move_dim_back<T: Clone, D: Backend>(
    tensor: &Tensor<T, D>, original_dim: usize,
) -> Tensor<T, D> {
    let rank = tensor.rank();
    #[cfg(debug_assertions)]
    debug_assert!(original_dim < rank);
    if original_dim == rank - 1 { return tensor.clone(); }
    let last = rank - 1;
    let mut order = Vec::with_capacity(rank);
    for i in 0..rank {
        if i != last { order.push(i); }
        if i == original_dim { order.push(last); }
    }
    if order.len() < rank { order.push(last); }
    tensor.permute(&order)
}

// ============================================================
// matmul_naive：朴素三重循环
// ============================================================

/// 朴素三重循环 matmul（小矩阵用，编译器自动 FMA + 标量向量化）
///
/// # 算法
/// C[i*n + j] = sum_{p=0..k-1} A[i*k + p] * B[p*n + j]
///
/// # 参数
/// - a: 行主序 m×k 矩阵
/// - b: 行主序 k×n 矩阵
/// - m, n, k: 维度
///
/// # 返回
/// m×n 的 f32 数组（行主序）
///
/// # 前置条件
/// m*n 不溢出 usize（由调用方保证，通常在 matmul() 里统一检查）
pub fn matmul_naive(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let total = oyta_core::checked::checked_mul(m, n).unwrap_or(usize::MAX);
    let mut c = vec![0.0f32; total];
    for i in 0..m {
        for j in 0..n {
            let mut sum: f32 = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// ============================================================
// dot_product_flat：精确点积（1×1 matmul 特化）
// ============================================================

/// 精确点积（两个 k 维向量的点积）
///
/// # 用途
/// matmul 当 m=1, n=1 时退化为点积，直接调用更高效
///
/// # 实现
/// 编译器 FMA 友好写法：sum = a*b + sum
/// 现代 CPU 上自动融合成单条 FMA 指令
pub fn dot_product_flat(a: &[f32], b: &[f32], k: usize) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 0..k {
        sum = a[i] * b[i] + sum; // FMA 友好写法
    }
    sum
}

// ============================================================
// matmul_tiled_fma：Tiled + FMA + rayon
// ============================================================

/// Tiled + FMA + rayon 分块 matmul（CPU 主力，§8.1）
///
/// # 核心思想
/// - 把 C 分成 BM×BN 的 tile，每个 tile 独立计算（块间无冲突）
/// - std::thread::scope 并行（每个线程独占一块 C 的区域写入，天然无冲突）
/// - 裸指针装箱成 usize 在线程间传递（unsafe 但安全，因为块不重叠）
/// - 内循环编译器自动 FMA（sum = a*b + sum）
/// - BM/BN/BK=128 匹配 L2 缓存
///
/// # 算法
/// for each (mi, ni) tile in C:
///   for ki in k_tiles:
///     C_block[mi, ni] += A_block[mi, ki] × B_block[ki, ni]
///
/// # 返回
/// m×n 的 f32 数组（行主序）
pub fn matmul_tiled_fma(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let num_m = (m + MATMUL_BM - 1) / MATMUL_BM;
    let num_n = (n + MATMUL_BN - 1) / MATMUL_BN;
    let num_k = (k + MATMUL_BK - 1) / MATMUL_BK;
    let total_blocks = num_m * num_n;

    let num_threads = rayon::current_num_threads().max(1).min(total_blocks);

    let c_total = oyta_core::checked::checked_mul(m, n)
        .unwrap_or(usize::MAX);
    let mut c = vec![0.0f32; c_total];

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
                                let mut sum = if is_first_k {
                                    0.0
                                } else {
                                    unsafe { *c.add(i * n + j) }
                                };
                                for p in ka..kb {
                                    // sum = a*b + sum → 编译器自动 FMA！
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

    c
}

// ============================================================
// matmul_strassen：完整 7 公式递归实现（§30.1）
// ============================================================

/// Strassen 算法 matmul（递归 2×2 分块）
///
/// # 核心公式
/// A = [[A11, A12], [A21, A22]]   B = [[B11, B12], [B21, B22]]
///
/// M1 = (A11 + A22) × (B11 + B22)
/// M2 = (A21 + A22) × B11
/// M3 = A11 × (B12 - B22)
/// M4 = A22 × (B21 - B11)
/// M5 = (A11 + A12) × B22
/// M6 = (A21 - A11) × (B11 + B12)
/// M7 = (A12 - A22) × (B21 + B22)
///
/// C11 = M1 + M4 - M5 + M7
/// C12 = M3 + M5
/// C21 = M2 + M4
/// C22 = M1 - M2 + M3 + M6
///
/// # 复杂度
/// 7 次乘法 + 18 次加法 → O(n^log₂7) ≈ O(n^2.807)
///
/// # 交叉点
/// 小于 STRASSEN_CROSSOVER 的维度直接回退 tiled_fma（分块开销太小不值得递归）
pub fn matmul_strassen(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    strassen_recursive(a, b, m, n, k)
}

/// Strassen 递归内部实现
fn strassen_recursive(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    // ---- 终止条件：任何维度 < 交叉点 → 回退 tiled_fma ----
    if m < STRASSEN_CROSSOVER || n < STRASSEN_CROSSOVER || k < STRASSEN_CROSSOVER {
        return matmul_tiled_fma(a, b, m, n, k);
    }

    // ---- 分块 ----
    let m_half = m / 2;
    let n_half = n / 2;
    let k_half = k / 2;

    // 如果任何 half 是 0，直接回退
    if m_half == 0 || n_half == 0 || k_half == 0 {
        return matmul_tiled_fma(a, b, m, n, k);
    }

    // ---- A, B 切成 2×2 子块 ----
    // A[m, k] → A11[m_half, k_half], A12[m_half, k-k_half]
    //          A21[m-m_half, k_half], A22[m-m_half, k-k_half]
    let a11 = slice_2d(a, m, k, 0, 0, m_half, k_half);
    let a12 = slice_2d(a, m, k, 0, k_half, m_half, k - k_half);
    let a21 = slice_2d(a, m, k, m_half, 0, m - m_half, k_half);
    let a22 = slice_2d(a, m, k, m_half, k_half, m - m_half, k - k_half);

    let b11 = slice_2d(b, k, n, 0, 0, k_half, n_half);
    let b12 = slice_2d(b, k, n, 0, n_half, k_half, n - n_half);
    let b21 = slice_2d(b, k, n, k_half, 0, k - k_half, n_half);
    let b22 = slice_2d(b, k, n, k_half, n_half, k - k_half, n - n_half);

    // ---- 辅助函数：逐元素加减 ----
    let vadd = |x: &[f32], y: &[f32]| -> Vec<f32> { x.iter().zip(y.iter()).map(|(&a, &b)| a + b).collect() };
    let vsub = |x: &[f32], y: &[f32]| -> Vec<f32> { x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect() };

    // ---- 7 次乘法 ----
    let m1 = strassen_recursive(
        &vadd(&a11, &a22), &vadd(&b11, &b22),
        m_half, n_half, k_half,
    );
    let m2 = strassen_recursive(
        &vadd(&a21, &a22), &b11,
        m - m_half, n_half, k_half,
    );
    let m3 = strassen_recursive(
        &a11, &vsub(&b12, &b22),
        m_half, n - n_half, k_half,
    );
    let m4 = strassen_recursive(
        &a22, &vsub(&b21, &b11),
        m - m_half, n_half, k - k_half,
    );
    let m5 = strassen_recursive(
        &vadd(&a11, &a12), &b22,
        m_half, n - n_half, k - k_half,
    );
    let m6 = strassen_recursive(
        &vsub(&a21, &a11), &vadd(&b11, &b12),
        m - m_half, n_half, k_half,
    );
    let m7 = strassen_recursive(
        &vsub(&a12, &a22), &vadd(&b21, &b22),
        m_half, n - n_half, k - k_half,
    );

    // ---- 组合 C 的 4 个子块 ----
    // C11 = M1 + M4 - M5 + M7
    // C12 = M3 + M5
    // C21 = M2 + M4
    // C22 = M1 - M2 + M3 + M6
    let mn_size = oyta_core::checked::checked_mul(m, n).unwrap_or(usize::MAX);
    let mut c = vec![0.0f32; mn_size];

    // C11: [0..m_half, 0..n_half]
    for i in 0..m_half {
        for j in 0..n_half {
            let idx = i * n + j;
            c[idx] = m1[i * n_half + j] + m4[i * n_half + j] - m5[i * (n - n_half) + j] + m7[i * (n - n_half) + j];
        }
    }

    // C12: [0..m_half, n_half..n]
    for i in 0..m_half {
        for j in 0..(n - n_half) {
            let idx = i * n + (n_half + j);
            c[idx] = m3[i * (n - n_half) + j] + m5[i * (n - n_half) + j];
        }
    }

    // C21: [m_half..m, 0..n_half]
    for i in 0..(m - m_half) {
        for j in 0..n_half {
            let idx = (m_half + i) * n + j;
            c[idx] = m2[i * n_half + j] + m4[i * n_half + j];
        }
    }

    // C22: [m_half..m, n_half..n]
    for i in 0..(m - m_half) {
        for j in 0..(n - n_half) {
            let idx = (m_half + i) * n + (n_half + j);
            c[idx] = m1[i * n_half + j] - m2[i * n_half + j] + m3[i * (n - n_half) + j] + m6[i * n_half + j];
        }
    }

    c
}

/// 从一个 m×k 行主序矩阵中切出子矩阵
///
/// # 参数
/// - data: 原始 flat 数组
/// - rows: 原矩阵行数
/// - cols: 原矩阵列数
/// - row_start, col_start: 子矩阵左上角坐标
/// - row_len, col_len: 子矩阵大小
///
/// # 返回
/// 切出的子矩阵（新分配的 flat Vec）
fn slice_2d(
    data: &[f32], _rows: usize, cols: usize,
    row_start: usize, col_start: usize,
    row_len: usize, col_len: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(row_len * col_len);
    for i in row_start..row_start + row_len {
        for j in col_start..col_start + col_len {
            out.push(data[i * cols + j]);
        }
    }
    out
}

// ============================================================
// matmul：完整算法选择入口
// ============================================================

/// 自动选择最优算法的矩阵乘法
///
/// # 算法选择逻辑
/// 1. m=1 && n=1 → dot_product_flat（点积特化）
/// 2. m,n,k 都 ≤ NAIVE_THRESHOLD (32) → matmul_naive
/// 3. m,n,k 都 ≥ STRASSEN_ENTRY (1024) && !D::IS_CPU → Backend::matmul（GPU cuBLAS）
/// 4. m,n,k 都 ≥ STRASSEN_ENTRY && D::IS_CPU → matmul_strassen
/// 5. 默认 → Backend::matmul（内部会选 tiled_fma 或 strassen）
///
/// # 预处理
/// 两个输入先 contiguous（非连续输入会 copy 成新 Arc<Vec<T>>）
/// 然后提取 m, n, k，根据 rank 判断是否 batched matmul
pub fn matmul<D: Backend>(a: &Tensor<f32, D>, b: &Tensor<f32, D>) -> Tensor<f32, D> {
    // ---- 先 contiguous ----
    let a_c = a.contiguous();
    let b_c = b.contiguous();

    // ---- 提取矩阵维度 ----
    let rank_a = a_c.rank();
    let rank_b = b_c.rank();
    #[cfg(debug_assertions)]
    debug_assert!(rank_a >= 2 && rank_b >= 2, "matmul 需要 rank >= 2");
    let m = a_c.shape.dims[rank_a - 2];
    let k = a_c.shape.dims[rank_a - 1];
    let n = b_c.shape.dims[rank_b - 1];
    #[cfg(debug_assertions)]
    debug_assert_eq!(b_c.shape.dims[rank_b - 2], k, "matmul 维度不匹配");

    // ---- 计算 batch dims ----
    // 取两个 rank 里较大的那个的前 (rank-2) 维
    let batch_dims: Vec<usize> = if rank_a >= rank_b {
        a_c.shape.dims[..rank_a - 2].to_vec()
    } else {
        b_c.shape.dims[..rank_b - 2].to_vec()
    };
    let batch_size: usize = oyta_core::checked::checked_product(batch_dims.iter().copied())
        .unwrap_or(usize::MAX);

    // m*k 和 k*n 也先校验（避免后续 a_arr/b_arr from_raw_parts 长度溢出）
    let _a_total = oyta_core::checked::checked_mul(m, k)
        .unwrap_or(usize::MAX);
    let _b_total = oyta_core::checked::checked_mul(k, n)
        .unwrap_or(usize::MAX);

    // ---- 选算法并执行 ----
    let a_data = a_c.data_slice();
    let b_data = b_c.data_slice();

    // ---- 计算 out_data 大小（checked 三重乘积） ----
    let total_out = oyta_core::checked::checked_mul3(batch_size, m, n)
        .unwrap_or(usize::MAX);
    let mut out_data = vec![0.0f32; total_out];
    let _a_ptr = a_data.as_ptr();
    let _b_ptr = b_data.as_ptr();
    let _c_ptr = out_data.as_mut_ptr();

    // 单个矩阵乘法的辅助闭包
    let mk = oyta_core::checked::checked_mul(m, k).unwrap_or(usize::MAX);
    let kn = oyta_core::checked::checked_mul(k, n).unwrap_or(usize::MAX);
    let mn = oyta_core::checked::checked_mul(m, n).unwrap_or(usize::MAX);

    let matmul_one = |a_ptr: *const f32, b_ptr: *const f32, c_ptr: *mut f32| {
        // 算法选择
        if m == 1 && n == 1 {
            // 点积特化
            let a_arr = unsafe { std::slice::from_raw_parts(a_ptr, k) };
            let b_arr = unsafe { std::slice::from_raw_parts(b_ptr, k) };
            unsafe { *c_ptr = dot_product_flat(a_arr, b_arr, k); }
        } else if m <= NAIVE_THRESHOLD && n <= NAIVE_THRESHOLD && k <= NAIVE_THRESHOLD {
            // 小矩阵：naive
            let a_arr = unsafe { std::slice::from_raw_parts(a_ptr, mk) };
            let b_arr = unsafe { std::slice::from_raw_parts(b_ptr, kn) };
            let result = matmul_naive(a_arr, b_arr, m, n, k);
            unsafe {
                let c_s = std::slice::from_raw_parts_mut(c_ptr, mn);
                c_s.copy_from_slice(&result);
            }
        } else if m >= STRASSEN_ENTRY && n >= STRASSEN_ENTRY && k >= STRASSEN_ENTRY {
            if D::IS_CPU {
                // CPU 上大矩阵：Strassen
                let a_arr = unsafe { std::slice::from_raw_parts(a_ptr, mk) };
                let b_arr = unsafe { std::slice::from_raw_parts(b_ptr, kn) };
                let result = matmul_strassen(a_arr, b_arr, m, n, k);
                unsafe {
                    let c_s = std::slice::from_raw_parts_mut(c_ptr, mn);
                    c_s.copy_from_slice(&result);
                }
            } else {
                // GPU 上大矩阵：Backend::matmul（GPU 重写用 cuBLAS）
                D::matmul(
                    a_ptr, k, b_ptr, n,
                    c_ptr, n, m, n, k,
                    1.0, 0.0,
                );
            }
        } else {
            // 默认：用 Backend::matmul（内部根据大小选 tiled_fma 或 strassen）
            D::matmul(
                a_ptr, k, b_ptr, n,
                c_ptr, n, m, n, k,
                1.0, 0.0,
            );
        }
    };

    if batch_size == 1 {
        // 普通 2D matmul
        matmul_one(
            a_data.as_ptr(), b_data.as_ptr(),
            out_data.as_mut_ptr(),
        );
    } else {
        // batched matmul：循环调单 matmul（CPU 上够用；GPU 如果启用 batched gemm 会更快）
        let a_stride = mk;
        let b_stride = kn;
        let c_stride = mn;
        for bi in 0..batch_size {
            // 调用者必须保证 bi*stride 不溢出且各 batch 内存连续
            let a_off = oyta_core::checked::checked_mul(bi, a_stride).unwrap_or(usize::MAX);
            let b_off = oyta_core::checked::checked_mul(bi, b_stride).unwrap_or(usize::MAX);
            let c_off = oyta_core::checked::checked_mul(bi, c_stride).unwrap_or(usize::MAX);
            matmul_one(
                unsafe { a_data.as_ptr().add(a_off) },
                unsafe { b_data.as_ptr().add(b_off) },
                unsafe { out_data.as_mut_ptr().add(c_off) },
            );
        }
    }

    // ---- 构造输出 shape ----
    let mut out_shape_dims = batch_dims;
    out_shape_dims.push(m);
    out_shape_dims.push(n);
    Tensor::from_data(out_data, DynamicShape::new(out_shape_dims)).expect("Tensor 构造失败")
}

// ============================================================
// matmul_t：带可选转置的矩阵乘法
// ============================================================

/// 带 transpose 标志的矩阵乘法
///
/// # 参数
/// - transa: true → A 先转置再乘（A^T 在 row-major 下相当于交换 m/k）
/// - transb: true → B 先转置再乘
///
/// # 实现
/// 行主序下 transpose 一个矩阵 = 交换其 rows 和 cols
/// 先把 A^T 和 B^T 变成 contiguous（可能改变 lda/ldb 解释），再调 matmul()
pub fn matmul_t<D: Backend>(
    a: &Tensor<f32, D>, b: &Tensor<f32, D>,
    transa: bool, transb: bool,
) -> Tensor<f32, D> {
    // ---- 先提取原始维度 ----
    let rank_a = a.rank();
    let rank_b = b.rank();
    let m_orig = a.shape.dims[rank_a - 2];
    let k_orig_a = a.shape.dims[rank_a - 1];
    let k_orig_b = b.shape.dims[rank_b - 2];
    let n_orig = b.shape.dims[rank_b - 1];

    // ---- 根据 transpose 计算有效维度 ----
    let (m_eff, k_eff_a) = if transa { (k_orig_a, m_orig) } else { (m_orig, k_orig_a) };
    let (k_eff_b, n_eff) = if transb { (n_orig, k_orig_b) } else { (k_orig_b, n_orig) };
    assert_eq!(k_eff_a, k_eff_b, "matmul_t 内部维度不匹配");

    // ---- 先 contiguous ----
    let a_c = a.contiguous();
    let b_c = b.contiguous();

    let a_data = a_c.data_slice();
    let b_data = b_c.data_slice();

    // ---- 处理 transpose：构造 contiguous 的 A_eff 和 B_eff ----
    let k = k_eff_a;
    let m = m_eff;
    let n = n_eff;

    // transa=true 时：A_eff[k, m] 是原 A[m_orig, k] 的转置
    let a_eff: Vec<f32> = if transa {
        let mut v = vec![0.0f32; m * k]; // m_eff × k = k_orig × m_orig
        for i in 0..m_orig {
            for j in 0..k_orig_a {
                v[j * m_orig + i] = a_data[i * k_orig_a + j]; // 转置
            }
        }
        v
    } else {
        a_data.to_vec()
    };

    // transb=true 时：B_eff[k, n] 是原 B[k_orig_b, n_orig] 的转置
    let b_eff: Vec<f32> = if transb {
        let mut v = vec![0.0f32; k * n]; // k × n = k_orig_b × n_orig
        for i in 0..k_orig_b {
            for j in 0..n_orig {
                v[j * k_orig_b + i] = b_data[i * n_orig + j]; // 转置
            }
        }
        v
    } else {
        b_data.to_vec()
    };

    // ---- 构造 Tensor 然后调 matmul ----
    let batch_dims_a: Vec<usize> = if rank_a > 2 { a.shape.dims[..rank_a - 2].to_vec() } else { vec![] };
    let batch_dims_b: Vec<usize> = if rank_b > 2 { b.shape.dims[..rank_b - 2].to_vec() } else { vec![] };
    let batch_dims: Vec<usize> = if rank_a >= rank_b { batch_dims_a } else { batch_dims_b };

    let mut out_shape_a = batch_dims.clone();
    out_shape_a.push(m);
    out_shape_a.push(k);
    let a_t = Tensor::<f32, D>::from_data(a_eff, DynamicShape::new(out_shape_a)).expect("matmul_batch a_eff 构造失败");

    let mut out_shape_b = batch_dims;
    out_shape_b.push(k);
    out_shape_b.push(n);
    let b_t = Tensor::<f32, D>::from_data(b_eff, DynamicShape::new(out_shape_b)).expect("matmul_batch b_eff 构造失败");

    matmul(&a_t, &b_t)
}
