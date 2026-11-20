//! elwise.rs：泛型逐元素算子（一套 trait 覆盖 f32/f16/bf16）
//!
//! 默认实现用 Float trait 的 to_f32/from_f32 自动类型提升：
//! - f32 → identity（编译器单态化后消除为零开销）
//! - f16/bf16 → to_f32 → 算 → from_f32（IEEE754 编解码开销小）
//! GPU 后端重写：用 Tensor Core / wgsl compute shader

use crate::backend::base::BackendBase;
use crate::dtype::float_trait::Float;

// ============================================================
// 精度：标准正态分布误差函数（精确 GELU 专用）
// f32 专用——f16/bf16 的 GELU 默认实现也是 f32 精度算完 cast 回去
// ============================================================

pub fn error_function(x: f32) -> f32 {
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;
    const P: f32 = 0.3275911;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

// ============================================================
// ElwiseOps<T>：泛型逐元素算子（一套覆盖 f32/f16/bf16/f64）
// 默认实现统一先 .to_f32() 提升精度 → 算 → .from_f32() 写回
// GPU 后端重写这些方法用硬件加速
// ============================================================

pub trait ElwiseOps<T: Float>: BackendBase {
    // -------- 二元算术 --------
    fn elwise_add(a: *const T, b: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let b_s = unsafe { std::slice::from_raw_parts(b, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter()).zip(b_s.par_iter())
            .for_each(|((o, &x), &y)| { *o = T::from_f32(x.to_f32() + y.to_f32()); });
    }
    fn elwise_sub(a: *const T, b: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let b_s = unsafe { std::slice::from_raw_parts(b, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter()).zip(b_s.par_iter())
            .for_each(|((o, &x), &y)| { *o = T::from_f32(x.to_f32() - y.to_f32()); });
    }
    fn elwise_mul(a: *const T, b: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let b_s = unsafe { std::slice::from_raw_parts(b, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter()).zip(b_s.par_iter())
            .for_each(|((o, &x), &y)| { *o = T::from_f32(x.to_f32() * y.to_f32()); });
    }
    fn elwise_div(a: *const T, b: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let b_s = unsafe { std::slice::from_raw_parts(b, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter()).zip(b_s.par_iter())
            .for_each(|((o, &x), &y)| { *o = T::from_f32(x.to_f32() / y.to_f32()); });
    }

    // -------- 一元算术 --------
    fn elwise_neg(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter())
            .for_each(|(o, &x)| { *o = T::from_f32(-x.to_f32()); });
    }
    fn elwise_abs(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter())
            .for_each(|(o, &x)| { *o = T::from_f32(x.to_f32().abs()); });
    }
    fn elwise_exp(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter())
            .for_each(|(o, &x)| { *o = T::from_f32(x.to_f32().exp()); });
    }
    fn elwise_log(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter())
            .for_each(|(o, &x)| { *o = T::from_f32(x.to_f32().ln()); });
    }
    fn elwise_sqrt(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter())
            .for_each(|(o, &x)| { *o = T::from_f32(x.to_f32().sqrt()); });
    }
    fn elwise_sin(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter())
            .for_each(|(o, &x)| { *o = T::from_f32(x.to_f32().sin()); });
    }
    fn elwise_cos(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter())
            .for_each(|(o, &x)| { *o = T::from_f32(x.to_f32().cos()); });
    }

    // -------- 激活函数 --------
    fn elwise_relu(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter())
            .for_each(|(o, &x)| { *o = T::from_f32(if x.to_f32() > 0.0 { x.to_f32() } else { 0.0 }); });
    }
    /// 精确 GELU：x * 0.5 * (1 + erf(x / sqrt(2)))
    fn elwise_gelu(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter()).for_each(|(o, &x)| {
            let v = x.to_f32();
            *o = T::from_f32(v * 0.5 * (1.0 + error_function(v / std::f32::consts::SQRT_2)));
        });
    }
    /// GELU 近似：0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn elwise_gelu_fast(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter()).for_each(|(o, &x)| {
            let v = x.to_f32();
            let inner = (v * 0.7978845608028654) * (1.0 + 0.044715 * v * v);
            *o = T::from_f32(0.5 * v * (1.0 + inner.tanh()));
        });
    }
    fn elwise_sigmoid(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter())
            .for_each(|(o, &x)| { *o = T::from_f32(1.0 / (1.0 + (-x.to_f32()).exp())); });
    }
    fn elwise_tanh(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter())
            .for_each(|(o, &x)| { *o = T::from_f32(x.to_f32().tanh()); });
    }
    fn elwise_silu(a: *const T, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter())
            .for_each(|(o, &x)| { *o = T::from_f32(x.to_f32() / (1.0 + (-x.to_f32()).exp())); });
    }

    // -------- 带标量参数的幂 --------
    fn elwise_pow(a: *const T, exp: f32, c: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts(a, n) };
        let c_s = unsafe { std::slice::from_raw_parts_mut(c, n) };
        use rayon::prelude::*;
        c_s.par_iter_mut().zip(a_s.par_iter())
            .for_each(|(o, &x)| { *o = T::from_f32(x.to_f32().powf(exp)); });
    }

    // -------- 广播版 add（完整 strided index 遍历）--------
    fn elwise_add_broadcast(
        a: *const T, a_strides: *const usize, a_shape: *const usize, a_rank: usize,
        b: *const T, b_strides: *const usize, b_shape: *const usize, b_rank: usize,
        out: *mut T, out_shape: *const usize, out_rank: usize,
    ) {
        use crate::backend::aggregate::broadcast_strides;
        let out_dims = unsafe { std::slice::from_raw_parts(out_shape, out_rank) };
        let out_strides_c = broadcast_strides(out_dims);
        let a_dims = unsafe { std::slice::from_raw_parts(a_shape, a_rank) };
        let b_dims = unsafe { std::slice::from_raw_parts(b_shape, b_rank) };
        let out_numel: usize = out_dims.iter().product();
        let a_numel: usize = crate::checked::checked_mul3_from_iter(a_dims.iter().copied());
        let b_numel: usize = crate::checked::checked_mul3_from_iter(b_dims.iter().copied());
        let out_s = unsafe { std::slice::from_raw_parts_mut(out, out_numel) };
        let a_s = unsafe { std::slice::from_raw_parts(a, a_numel) };
        let b_s = unsafe { std::slice::from_raw_parts(b, b_numel) };
        let a_strides_s = unsafe { std::slice::from_raw_parts(a_strides, a_rank) };
        let b_strides_s = unsafe { std::slice::from_raw_parts(b_strides, b_rank) };
        let out_strides_s = out_strides_c.as_slice();

        let mut idx = vec![0usize; out_rank];
        for flat_i in 0..out_numel {
            let mut a_idx = 0usize;
            let mut b_idx = 0usize;
            for d in 0..out_rank {
                let ai = if d < a_rank && a_dims[a_rank - 1 - d] != 1 { idx[out_rank - 1 - d] } else { 0 };
                let bi = if d < b_rank && b_dims[b_rank - 1 - d] != 1 { idx[out_rank - 1 - d] } else { 0 };
                if d < a_rank { a_idx += ai * a_strides_s[a_rank - 1 - d]; }
                if d < b_rank { b_idx += bi * b_strides_s[b_rank - 1 - d]; }
            }
            out_s[flat_i] = T::from_f32(a_s[a_idx].to_f32() + b_s[b_idx].to_f32());

            let mut d = out_rank;
            while d > 0 {
                d -= 1;
                idx[d] += 1;
                if idx[d] < out_dims[d] { break; }
                idx[d] = 0;
            }
            let _ = out_strides_s;
        }
    }

    // -------- In-place --------
    fn elwise_add_inplace(a: *mut T, b: *const T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts_mut(a, n) };
        let b_s = unsafe { std::slice::from_raw_parts(b, n) };
        use rayon::prelude::*;
        a_s.par_iter_mut().zip(b_s.par_iter()).for_each(|(o, &x)| {
            *o = T::from_f32(o.to_f32() + x.to_f32());
        });
    }
    fn elwise_mul_inplace(a: *mut T, b: *const T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts_mut(a, n) };
        let b_s = unsafe { std::slice::from_raw_parts(b, n) };
        use rayon::prelude::*;
        a_s.par_iter_mut().zip(b_s.par_iter()).for_each(|(o, &x)| {
            *o = T::from_f32(o.to_f32() * x.to_f32());
        });
    }
}
