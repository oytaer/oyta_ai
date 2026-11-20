//! flash_attn.rs：FlashAttention v2（泛型版本，在线 softmax + causal mask）
//! 累加器统一用 f64（不管 T 是 f32/f16/bf16），保证数值稳定

use crate::backend::base::BackendBase;
use crate::dtype::float_trait::Float;

pub trait FlashAttentionOps<T: Float>: BackendBase {
    /// 单 batch FlashAttention（q/k/v/out 都接受 T 类型指针）
    fn flash_attention(q: *const T, k: *const T, v: *const T,
                       out: *mut T, seq_q: usize, seq_k: usize,
                       d_k: usize, d_v: usize, scale: f32, is_causal: bool) {
        #[cfg(debug_assertions)] {
            debug_assert!(seq_q > 0, "flash_attn seq_q > 0");
            debug_assert!(seq_k > 0, "flash_attn seq_k > 0");
            debug_assert!(d_k >= 16, "flash_attn d_k({}) >= 16", d_k);
            debug_assert!(d_v >= 16, "flash_attn d_v({}) >= 16", d_v);
        }
        let q_len = crate::checked::checked_mul(seq_q, d_k);
        let k_len = crate::checked::checked_mul(seq_k, d_k);
        let v_len = crate::checked::checked_mul(seq_k, d_v);
        let o_len = crate::checked::checked_mul(seq_q, d_v);
        let q_s = unsafe { std::slice::from_raw_parts(q, q_len) };
        let k_s = unsafe { std::slice::from_raw_parts(k, k_len) };
        let v_s = unsafe { std::slice::from_raw_parts(v, v_len) };
        let o_s = unsafe { std::slice::from_raw_parts_mut(out, o_len) };
        let scale_f64 = scale as f64;
        for qi in 0..seq_q {
            let q_row = &q_s[qi * d_k .. (qi + 1) * d_k];
            let mut m_val: f64 = f64::NEG_INFINITY;
            let mut l_val: f64 = 0.0;
            let mut o_acc: Vec<f64> = vec![0.0f64; d_v];
            let k_end = if is_causal { (qi + 1).min(seq_k) } else { seq_k };
            for ki in 0..k_end {
                let k_row = &k_s[ki * d_k .. (ki + 1) * d_k];
                let s: f64 = q_row.iter().zip(k_row.iter())
                    .map(|(&a, &b)| (a.to_f32() as f64) * (b.to_f32() as f64))
                    .sum::<f64>() * scale_f64;
                let m_new = m_val.max(s);
                let correction = (m_val - m_new).exp();
                l_val *= correction;
                for d in 0..d_v { o_acc[d] *= correction; }
                let exp_s = (s - m_new).exp();
                l_val += exp_s;
                let v_row = &v_s[ki * d_v .. (ki + 1) * d_v];
                for d in 0..d_v { o_acc[d] += exp_s * (v_row[d].to_f32() as f64); }
                m_val = m_new;
            }
            let inv_l = 1.0 / l_val;
            let o_row = &mut o_s[qi * d_v .. (qi + 1) * d_v];
            for d in 0..d_v { o_row[d] = T::from_f32((o_acc[d] * inv_l) as f32); }
        }
    }

    /// batched multi-head FlashAttention
    fn flash_attention_batched(q: *const T, k: *const T, v: *const T,
                               out: *mut T, batch: usize, heads: usize,
                               seq_q: usize, seq_k: usize,
                               d_k: usize, d_v: usize,
                               scale: f32, is_causal: bool,
                               stride_q_outer: isize, stride_kv_outer: isize,
                               stride_out_outer: isize) {
        use rayon::prelude::*;
        let bh = crate::checked::checked_mul(batch, heads);
        let q_addr = q as usize; let k_addr = k as usize; let v_addr = v as usize; let o_addr = out as usize;
        let sq = seq_q; let sk = seq_k;
        let scale_f64 = scale as f64;
        (0usize..bh).into_par_iter().for_each(move |bhi: usize| {
            let qo = crate::checked::checked_mul(bhi, stride_q_outer as usize);
            let ko = crate::checked::checked_mul(bhi, stride_kv_outer as usize);
            let vo = crate::checked::checked_mul(bhi, stride_kv_outer as usize);
            let oo = crate::checked::checked_mul(bhi, stride_out_outer as usize);
            let q_h = crate::checked::checked_add(q_addr, qo) as *const T;
            let k_h = crate::checked::checked_add(k_addr, ko) as *const T;
            let v_h = crate::checked::checked_add(v_addr, vo) as *const T;
            let o_h = crate::checked::checked_add(o_addr, oo) as *mut T;
            let q_len = crate::checked::checked_mul(sq, d_k);
            let k_len = crate::checked::checked_mul(sk, d_k);
            let v_len = crate::checked::checked_mul(sk, d_v);
            let o_len = crate::checked::checked_mul(sq, d_v);
            let q_s = unsafe { std::slice::from_raw_parts(q_h, q_len) };
            let k_s = unsafe { std::slice::from_raw_parts(k_h, k_len) };
            let v_s = unsafe { std::slice::from_raw_parts(v_h, v_len) };
            let o_s = unsafe { std::slice::from_raw_parts_mut(o_h, o_len) };
            for qi in 0..sq {
                let q_row = &q_s[qi * d_k .. (qi + 1) * d_k];
                let mut m_val: f64 = f64::NEG_INFINITY;
                let mut l_val: f64 = 0.0;
                let mut o_acc: Vec<f64> = vec![0.0f64; d_v];
                let k_end = if is_causal { (qi + 1).min(sk) } else { sk };
                for ki in 0..k_end {
                    let k_row = &k_s[ki * d_k .. (ki + 1) * d_k];
                    let s: f64 = q_row.iter().zip(k_row.iter())
                        .map(|(&a, &b)| (a.to_f32() as f64) * (b.to_f32() as f64))
                        .sum::<f64>() * scale_f64;
                    let m_new = m_val.max(s);
                    let correction = (m_val - m_new).exp();
                    l_val *= correction;
                    for d in 0..d_v { o_acc[d] *= correction; }
                    let exp_s = (s - m_new).exp();
                    l_val += exp_s;
                    let v_row = &v_s[ki * d_v .. (ki + 1) * d_v];
                    for d in 0..d_v { o_acc[d] += exp_s * (v_row[d].to_f32() as f64); }
                    m_val = m_new;
                }
                let inv_l = 1.0 / l_val;
                for d in 0..d_v { o_s[qi * d_v + d] = T::from_f32((o_acc[d] * inv_l) as f32); }
            }
        });
    }
}
