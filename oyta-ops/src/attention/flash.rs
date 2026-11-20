//! flash_attn：FlashAttention（§9.2 在线 softmax + 分块）
//!
//! 完整实现：
//! - flash_attention: 单头 FlashAttention（调用 Backend::flash_attention）
//! - flash_attention_multihead: 多头 FlashAttention（循环调单头或用 Backend::flash_attention_batched）
//!
//! # FlashAttention v2 核心思想
//! 标准 attention: O(seq² × d) 内存（存整个 S=QK^T）
//! FlashAttention: O(seq × d) 内存（在线 softmax，不存 S）
//!   - 逐行处理 Q 的 query
//!   - 维护 m_val（当前 max）和 l_val（当前 sum exp）
//!   - 每来一个 K 块就在线更新 m, l, o_acc
//!
//! Backend trait 已有 flash_attention 默认实现（逐 query + 在线 softmax + f64 累加器）
//! 这里的函数只是 Tensor 层包装（shape 提取 + contiguous + 输出构造）

use oyta_core::{Tensor, DynamicShape};
use oyta_device::Backend;

// ============================================================
// flash_attention：单头 FlashAttention
// ============================================================

/// 单头 FlashAttention（调用 Backend::flash_attention 默认实现）
///
/// # 输入 shape
/// Q: [seq_q, d_k]  （单头）
/// K: [seq_k, d_k]
/// V: [seq_k, d_v]
///
/// # 输出 shape
/// O: [seq_q, d_v]
///
/// # 参数
/// - causal: true → causal mask（只看 [0, qi] 范围的 K）
/// - scale: 通常是 1/sqrt(d_k)
///
/// # 实现
/// 直接调用 Backend::flash_attention(q, k, v, out, seq_q, seq_k, d_k, d_v, scale, causal)
/// Backend 默认实现用在线 softmax + f64 累加器保证精度
pub fn flash_attention<D: Backend>(
    q: &Tensor<f32, D>, k: &Tensor<f32, D>, v: &Tensor<f32, D>,
    causal: bool, scale: f32,
) -> Tensor<f32, D> {
    // ---- contiguous ----
    let q_c = q.contiguous();
    let k_c = k.contiguous();
    let v_c = v.contiguous();

    // ---- 提取 shape ----
    #[cfg(debug_assertions)]
    debug_assert!(q.rank() >= 2, "flash_attention 需要 Q rank >= 2");
    let seq_q = q.shape.dims[q.rank() - 2];
    let d_k = q.shape.dims[q.rank() - 1];
    #[cfg(debug_assertions)]
    debug_assert_eq!(k.shape.dims[k.rank() - 2], v.shape.dims[v.rank() - 2], "K 和 V 的 seq 必须相同");
    let seq_k = k.shape.dims[k.rank() - 2];
    let d_v = v.shape.dims[v.rank() - 1];

    // ---- 分配输出 ----
    let total = oyta_core::checked::checked_mul(seq_q, d_v)
        .unwrap_or(usize::MAX);
    let mut out_data = vec![0.0f32; total];

    // ---- 调 Backend 默认实现 ----
    D::flash_attention(
        q_c.data_ptr(),     // q: *const f32
        k_c.data_ptr(),     // k: *const f32
        v_c.data_ptr(),     // v: *const f32
        out_data.as_mut_ptr(), // out: *mut f32
        seq_q,              // seq_q
        seq_k,              // seq_k
        d_k,                // d_k
        d_v,                // d_v
        scale,              // scale
        causal,             // is_causal
    );

    Tensor::from_data(out_data, DynamicShape::new([seq_q, d_v])).expect("Tensor 构造失败")
}

// ============================================================
// flash_attention_multihead：多头 FlashAttention
// ============================================================

/// 多头 FlashAttention（§9.4 + §24）
///
/// # 输入 shape（packed 布局）
/// Q: [batch, heads, seq_q, d_k]
/// K: [batch, heads, seq_k, d_k]
/// V: [batch, heads, seq_k, d_v]
///
/// # 输出 shape
/// O: [batch, heads, seq_q, d_v]
///
/// # 实现流程
/// 1. contiguous 预处理
/// 2. 计算 strided 访问（packed 布局：Q[b,h,s,d] 连续存储）
/// 3. 调 Backend::flash_attention_batched（内部 rayon 并行每个 batch×head）
///
/// # 为什么不用 reshape?
/// reshape [B, H, S, D] → [B*H, S, D] 也行，但 Backend::flash_attention_batched
/// 接受 stride 参数可以避免真正 reshape（packed 布局天然连续，stride 就是 S*D）
pub fn flash_attention_multihead<D: Backend>(
    q: &Tensor<f32, D>, k: &Tensor<f32, D>, v: &Tensor<f32, D>,
    causal: bool, scale: f32,
) -> Tensor<f32, D> {
    // ---- contiguous ----
    let q_c = q.contiguous();
    let k_c = k.contiguous();
    let v_c = v.contiguous();

    // ---- 提取 shape（假设 [batch, heads, seq, d] 格式）----
    #[cfg(debug_assertions)]
    debug_assert_eq!(q.rank(), 4, "multihead attention 需要 rank=4: [batch, heads, seq, d]");
    let batch = q.shape.dims[0];
    let heads = q.shape.dims[1];
    let seq_q = q.shape.dims[2];
    let d_k = q.shape.dims[3];
    let seq_k = k.shape.dims[2];
    let d_v = v.shape.dims[3];

    let bh = oyta_core::checked::checked_mul(batch, heads)
        .unwrap_or(usize::MAX);
    let total_out = oyta_core::checked::checked_mul3(bh, seq_q, d_v)
        .unwrap_or(usize::MAX);

    // ---- 分配输出 ----
    let mut out_data = vec![0.0f32; total_out];

    // ---- stride 计算 ----
    // packed 布局 [B, H, S, D] 下，每个 (batch, head) 的 Q 块大小 = seq * d
    let per_head_size_q = oyta_core::checked::checked_mul(seq_q, d_k)
        .unwrap_or(usize::MAX);
    let per_head_size_kv = oyta_core::checked::checked_mul(seq_k, d_k)
        .unwrap_or(usize::MAX);
    let per_head_size_out = oyta_core::checked::checked_mul(seq_q, d_v)
        .unwrap_or(usize::MAX);

    // 调 Backend::flash_attention_batched（内部 rayon 并行每个 bh 块）
    D::flash_attention_batched(
        q_c.data_ptr(),                     // q
        k_c.data_ptr(),                     // k
        v_c.data_ptr(),                     // v
        out_data.as_mut_ptr(),              // out
        batch,                              // batch
        heads,                              // heads
        seq_q,                              // seq_q
        seq_k,                              // seq_k
        d_k,                                // d_k
        d_v,                                // d_v
        scale,                              // scale
        causal,                             // is_causal
        per_head_size_q as isize,           // stride_q_outer
        per_head_size_kv as isize,          // stride_kv_outer（K 和 V 的 stride 相同）
        per_head_size_out as isize,         // stride_out_outer
    );
    let _ = bh; // 消除未使用警告

    Tensor::from_data(
        out_data,
        DynamicShape::new([batch, heads, seq_q, d_v]),
    ).expect("flash_attention_multihead 输出构造失败")
}
