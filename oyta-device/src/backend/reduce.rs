//! reduce.rs：泛型规约算子（ReduceOps<T> 一套覆盖 f32/f16/bf16）
//!
//! Accumulate trait 自动提升累加器精度：
//! - f32 的 Accum = f64 → sum/mean/softmax/layernorm/rmsnorm 全用 f64 累加
//! - f16/bf16 的 Accum = f32 → 提升到 f32 做数值稳定的统计

use crate::backend::base::BackendBase;
use crate::dtype::float_trait::{Float, Accumulate};

// ============================================================
// 默认常量
// ============================================================

pub const LAYERNORM_EPS: f32 = 1e-5;
pub const RMSNORM_EPS: f32 = 1e-6;

// ============================================================
// 辅助：pairwise sum 递归二分归约（数值稳定）
// ============================================================

pub fn pairwise_sum<A: Copy + Clone>(data: &[A], to_f64: fn(A) -> f64) -> f64 {
    let len = data.len();
    if len == 0 { return 0.0f64; }
    if len == 1 { return to_f64(data[0]); }
    let half = len / 2;
    pairwise_sum(&data[..half], to_f64) + pairwise_sum(&data[half..], to_f64)
}

// ============================================================
// ReduceOps<T>：泛型规约算子（ReduceOps<T> 一套覆盖 f32/f16/bf16）
// 默认实现统一先 to_f64() 提升精度 → 算 → from_f64() 写回
// ============================================================

pub trait ReduceOps<T: Float + Accumulate>: BackendBase {
    // -------- sum --------
    fn reduce_sum(a: *const T, out: *mut T, outer: usize, inner: usize) {
        let a_len = crate::checked::checked_mul(outer, inner);
        let a_s = unsafe { std::slice::from_raw_parts(a, a_len) };
        let out_s = unsafe { std::slice::from_raw_parts_mut(out, outer) };
        use rayon::prelude::*;
        out_s.par_iter_mut().enumerate().for_each(|(oi, o)| {
            let row = &a_s[oi * inner .. (oi + 1) * inner];
            // 用 Float::to_f32 转原始类型再 cast 到 f64（比 to_accum 灵活）
            let acc: f64 = row.iter().map(|&x| x.to_f32() as f64).sum();
            *o = T::from_f32(acc as f32);
        });
    }

    // -------- mean --------
    fn reduce_mean(a: *const T, out: *mut T, outer: usize, inner: usize) {
        let a_len = crate::checked::checked_mul(outer, inner);
        let a_s = unsafe { std::slice::from_raw_parts(a, a_len) };
        let out_s = unsafe { std::slice::from_raw_parts_mut(out, outer) };
        let inv_n = 1.0 / inner as f64;
        use rayon::prelude::*;
        out_s.par_iter_mut().enumerate().for_each(|(oi, o)| {
            let row = &a_s[oi * inner .. (oi + 1) * inner];
            let acc: f64 = row.iter().map(|&x| x.to_f32() as f64).sum();
            *o = T::from_f32((acc * inv_n) as f32);
        });
    }

    // -------- max --------
    fn reduce_max(a: *const T, out: *mut T, outer: usize, inner: usize) {
        let a_len = crate::checked::checked_mul(outer, inner);
        let a_s = unsafe { std::slice::from_raw_parts(a, a_len) };
        let out_s = unsafe { std::slice::from_raw_parts_mut(out, outer) };
        let neg_inf = T::from_f32(f32::NEG_INFINITY);
        for oi in 0..outer {
            let row = &a_s[oi * inner .. (oi + 1) * inner];
            let mut m = neg_inf;
            for &x in row { if x.to_f32() > m.to_f32() { m = x; } }
            out_s[oi] = m;
        }
    }

    // -------- min --------
    fn reduce_min(a: *const T, out: *mut T, outer: usize, inner: usize) {
        let a_len = crate::checked::checked_mul(outer, inner);
        let a_s = unsafe { std::slice::from_raw_parts(a, a_len) };
        let out_s = unsafe { std::slice::from_raw_parts_mut(out, outer) };
        let inf = T::from_f32(f32::INFINITY);
        for oi in 0..outer {
            let row = &a_s[oi * inner .. (oi + 1) * inner];
            let mut m = inf;
            for &x in row { if x.to_f32() < m.to_f32() { m = x; } }
            out_s[oi] = m;
        }
    }

    // -------- softmax（减 max → f64 exp → f64 sum → 归一化）--------
    fn softmax(a: *const T, out: *mut T, outer: usize, inner: usize) {
        let a_len = crate::checked::checked_mul(outer, inner);
        let a_s = unsafe { std::slice::from_raw_parts(a, a_len) };
        let out_s = unsafe { std::slice::from_raw_parts_mut(out, a_len) };
        for oi in 0..outer {
            let row_in = &a_s[oi * inner .. (oi + 1) * inner];
            let row_out = &mut out_s[oi * inner .. (oi + 1) * inner];
            // 1. f64 max
            let mut max_f64 = f64::NEG_INFINITY;
            for &x in row_in { let v = x.to_f32() as f64; if v > max_f64 { max_f64 = v; } }
            // 2. f64 exp + f64 sum
            let mut sum_exp = 0.0f64;
            let mut exp_vals = vec![0.0f64; inner];
            for i in 0..inner {
                let v = (row_in[i].to_f32() as f64 - max_f64).exp();
                exp_vals[i] = v;
                sum_exp += v;
            }
            // 3. 归一化
            let inv = 1.0f64 / sum_exp;
            for i in 0..inner {
                row_out[i] = T::from_f32((exp_vals[i] * inv) as f32);
            }
        }
    }

    // -------- layernorm（f64 mean/var + f64 inv_std）--------
    fn layernorm(a: *const T, out: *mut T, gamma: *const T, beta: *const T,
                 outer: usize, inner: usize, eps: f32) {
        let total = crate::checked::checked_mul(outer, inner);
        let a_s = unsafe { std::slice::from_raw_parts(a, total) };
        let out_s = unsafe { std::slice::from_raw_parts_mut(out, total) };
        let g_s = unsafe { std::slice::from_raw_parts(gamma, inner) };
        let b_s = unsafe { std::slice::from_raw_parts(beta, inner) };
        let inv_n = 1.0f64 / inner as f64;
        let eps_f64 = eps as f64;
        for oi in 0..outer {
            let row_in = &a_s[oi * inner .. (oi + 1) * inner];
            let row_out = &mut out_s[oi * inner .. (oi + 1) * inner];
            let sum: f64 = row_in.iter().map(|&x| x.to_f32() as f64).sum();
            let mean = sum * inv_n;
            let var: f64 = row_in.iter().map(|&x| {
                let d = (x.to_f32() as f64) - mean;
                d * d
            }).sum::<f64>() * inv_n;
            let inv_std = 1.0 / (var + eps_f64).sqrt();
            for i in 0..inner {
                let normed = ((row_in[i].to_f32() as f64) - mean) * inv_std;
                let res = normed * (g_s[i].to_f32() as f64) + (b_s[i].to_f32() as f64);
                row_out[i] = T::from_f32(res as f32);
            }
        }
    }

    // -------- rmsnorm --------
    fn rmsnorm(a: *const T, out: *mut T, gamma: *const T,
               outer: usize, inner: usize, eps: f32) {
        let total = crate::checked::checked_mul(outer, inner);
        let a_s = unsafe { std::slice::from_raw_parts(a, total) };
        let out_s = unsafe { std::slice::from_raw_parts_mut(out, total) };
        let g_s = unsafe { std::slice::from_raw_parts(gamma, inner) };
        let inv_n = 1.0f64 / inner as f64;
        let eps_f64 = eps as f64;
        for oi in 0..outer {
            let row_in = &a_s[oi * inner .. (oi + 1) * inner];
            let row_out = &mut out_s[oi * inner .. (oi + 1) * inner];
            let sum_sq: f64 = row_in.iter().map(|&x| {
                let v = x.to_f32() as f64; v * v
            }).sum();
            let rms = (sum_sq * inv_n + eps_f64).sqrt();
            let inv_rms = 1.0f64 / rms;
            for i in 0..inner {
                let res = (row_in[i].to_f32() as f64) * inv_rms * (g_s[i].to_f32() as f64);
                row_out[i] = T::from_f32(res as f32);
            }
        }
    }

    // -------- layernorm_inplace --------
    fn layernorm_inplace(a: *mut T, gamma: *const T, beta: *const T,
                         outer: usize, inner: usize, eps: f32) {
        let total = crate::checked::checked_mul(outer, inner);
        let a_s = unsafe { std::slice::from_raw_parts_mut(a, total) };
        let g_s = unsafe { std::slice::from_raw_parts(gamma, inner) };
        let b_s = unsafe { std::slice::from_raw_parts(beta, inner) };
        let inv_n = 1.0f64 / inner as f64;
        let eps_f64 = eps as f64;
        for oi in 0..outer {
            let row = &mut a_s[oi * inner .. (oi + 1) * inner];
            let sum: f64 = row.iter().map(|&x| x.to_f32() as f64).sum();
            let mean = sum * inv_n;
            let var: f64 = row.iter().map(|&x| {
                let d = (x.to_f32() as f64) - mean;
                d * d
            }).sum::<f64>() * inv_n;
            let inv_std = 1.0 / (var + eps_f64).sqrt();
            for i in 0..inner {
                let normed = ((row[i].to_f32() as f64) - mean) * inv_std;
                let res = normed * (g_s[i].to_f32() as f64) + (b_s[i].to_f32() as f64);
                row[i] = T::from_f32(res as f32);
            }
        }
    }
}
