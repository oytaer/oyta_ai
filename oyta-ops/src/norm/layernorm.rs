//! layernorm：LayerNorm + RMSNorm（§7.6）
//!
//! 完整实现：
//! - layernorm: 调用 Backend::layernorm（γ, β, eps）
//! - rmsnorm: 调用 Backend::rmsnorm（γ, eps）
//!
//! # LayerNorm vs RMSNorm
//! LayerNorm: 中心化 + 缩放（减 mean → 除 std → ×γ + β）
//! RMSNorm: 只缩放（除 RMS → ×γ），更快，效果差不多（LLaMA 架构必须）
//!
//! # Backend trait 默认实现（纯 Rust）
//! - LayerNorm: 每行计算 mean/var，仿射变换
//! - RMSNorm: 每行计算 rms，直接缩放
//! GPU 后端重写为单个 kernel（省中间 buffer）

use oyta_core::Tensor;
use oyta_device::Backend;

// ============================================================
// layernorm：调用 Backend::layernorm
// ============================================================

/// LayerNorm：归一化 + 仿射变换
///
/// # 公式
/// out[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
///
/// # 参数
/// - x: 输入 Tensor（最后一维是归一化维）
/// - gamma: 缩放参数（长度 = 最后一维 size）
/// - beta: 偏移参数（长度 = 最后一维 size）
/// - eps: 防除零小量（通常 1e-5 或 1e-6）
///
/// # 输出 shape
/// 与输入完全相同
pub fn layernorm<D: Backend>(
    x: &Tensor<f32, D>,
    gamma: &Tensor<f32, D>,
    beta: &Tensor<f32, D>,
    eps: f32,
) -> Tensor<f32, D> {
    // ---- contiguous 预处理 ----
    let x_c = x.contiguous();
    let g_c = gamma.contiguous();
    let b_c = beta.contiguous();

    // ---- 提取维度 ----
    let rank = x_c.rank();
    let inner = x_c.shape.dims[rank - 1]; // 归一化维 size
    let outer = x_c.numel() / inner;       // 行数

    // ---- 分配输出 ----
    let mut out_data = vec![0.0f32; x_c.numel()];

    // ---- 调 Backend::layernorm 默认实现 ----
    D::layernorm(
        x_c.data_ptr(),       // a
        out_data.as_mut_ptr(), // out
        g_c.data_ptr(),        // gamma
        b_c.data_ptr(),        // beta
        outer,                 // outer
        inner,                 // inner
        eps,                   // eps
    );

    Tensor::from_data(out_data, x_c.shape.clone()).expect("Tensor 构造失败")
}

// ============================================================
// rmsnorm：调用 Backend::rmsnorm
// ============================================================

/// RMSNorm：只做缩放（比 LayerNorm 快，效果差不多）
///
/// # 公式
/// out[i] = x[i] / sqrt(mean(x²) + eps) * gamma[i]
///
/// # 参数
/// - x: 输入 Tensor（最后一维是归一化维）
/// - gamma: 缩放参数（长度 = 最后一维 size）
/// - eps: 防除零小量（通常 1e-5 或 1e-6）
///
/// # 输出 shape
/// 与输入完全相同
pub fn rmsnorm<D: Backend>(
    x: &Tensor<f32, D>,
    gamma: &Tensor<f32, D>,
    eps: f32,
) -> Tensor<f32, D> {
    // ---- contiguous 预处理 ----
    let x_c = x.contiguous();
    let g_c = gamma.contiguous();

    // ---- 提取维度 ----
    let rank = x_c.rank();
    let inner = x_c.shape.dims[rank - 1];
    let outer = x_c.numel() / inner;

    // ---- 分配输出 ----
    let mut out_data = vec![0.0f32; x_c.numel()];

    // ---- 调 Backend::rmsnorm 默认实现 ----
    D::rmsnorm(
        x_c.data_ptr(),         // a
        out_data.as_mut_ptr(),  // out
        g_c.data_ptr(),         // gamma
        outer,                  // outer
        inner,                  // inner
        eps,                    // eps
    );

    Tensor::from_data(out_data, x_c.shape.clone()).expect("Tensor 构造失败")
}


