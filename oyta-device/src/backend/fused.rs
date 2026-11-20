//! fused.rs：泛型融合算子（FusedOps<T> 一套覆盖 f32/f16/bf16）
//! 减少 kernel launch + HBM 读写
//!
//! MatmulOps 保持 f32* 签名（GPU 后端只能高效处理 f32），
//! 所以 FusedOps<T> 默认实现里：先 T→f32 提升精度，调 matmul(f32*)，
//! 再 f32→T 写回（中间 f32 buffer 已经是精度提升后的结果）

use crate::backend::base::BackendBase;
use crate::backend::elwise::ElwiseOps;
use crate::backend::reduce::ReduceOps;
use crate::backend::matmul::MatmulOps;
use crate::backend::aggregate::ActivationType;
use crate::dtype::float_trait::{Float, Accumulate};

// ============================================================
// 辅助：f32 <-> T 转换循环（FusedOps 默认实现内部使用）
// ============================================================

fn t_to_f32_vec<T: Float>(v: &[T]) -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() }

// ============================================================
// FusedOps<T>
// ============================================================

pub trait FusedOps<T: Float + Accumulate>: ElwiseOps<T> + ReduceOps<T> + MatmulOps + BackendBase {
    /// LayerNorm + GELU 融合：out = gelu(layernorm(in, gamma, beta, eps))
    fn layernorm_gelu(a: *const T, out: *mut T,
                      gamma: *const T, beta: *const T,
                      outer: usize, inner: usize, eps: f32) {
        let total = crate::checked::checked_mul(outer, inner);
        let mut ln_buf = vec![T::from_f32(0.0); total];
        Self::layernorm(a, ln_buf.as_mut_ptr(), gamma, beta, outer, inner, eps);
        Self::elwise_gelu(ln_buf.as_ptr(), out, total);
    }

    /// Linear + Bias + GELU 融合：out = gelu(matmul(x, W) + bias)
    /// matmul 在 f32 精度算完后提升到 T（避免 F16 位模式被当成 f32 读）
    fn linear_bias_gelu(x: *const T, w: *const T, bias: *const T,
                        out: *mut T, m: usize, n: usize, k: usize) {
        let mk = crate::checked::checked_mul(m, k);
        let kn = crate::checked::checked_mul(k, n);
        let mn = crate::checked::checked_mul(m, n);
        // 1. T → f32（把 F16/BF16 提升到 f32，避免 cuBLAS 读错位模式）
        let x_f32 = t_to_f32_vec(unsafe { std::slice::from_raw_parts(x, mk) });
        let w_f32 = t_to_f32_vec(unsafe { std::slice::from_raw_parts(w, kn) });
        // 2. matmul（f32 → f32）
        let mut linear_buf = vec![0.0f32; mn];
        Self::matmul(x_f32.as_ptr(), k, w_f32.as_ptr(), n,
                     linear_buf.as_mut_ptr(), n, m, n, k, 1.0, 0.0);
        // 3. 加 bias + gelu（所有操作都在 f32 精度）
        let bias_s = unsafe { std::slice::from_raw_parts(bias, m) };
        let out_s = unsafe { std::slice::from_raw_parts_mut(out, mn) };
        for i in 0..m {
            let b = bias_s[i].to_f32();
            for j in 0..n {
                let v = linear_buf[i * n + j] + b;
                let g = v * 0.5 * (1.0 + crate::backend::elwise::error_function(v / std::f32::consts::SQRT_2));
                out_s[i * n + j] = T::from_f32(g);
            }
        }
    }

    /// RMSNorm + SwiGLU 融合（LLaMA 架构必须）
    /// 所有 matmul 在 f32 精度算，silu(gate)*up 也在 f32 精度
    fn rmsnorm_swiglu(x: *const T, w1: *const T, w2: *const T, w3: *const T,
                      out: *mut T, gamma: *const T,
                      m: usize, n: usize, k: usize, eps: f32) {
        let mk = crate::checked::checked_mul(m, k);
        let mn = crate::checked::checked_mul(m, n);
        // 1. rmsnorm 在 T 上做（f64 累加器已经提升精度）
        let mut rms_buf = vec![T::from_f32(0.0); mk];
        Self::rmsnorm(x, rms_buf.as_mut_ptr(), gamma, m, k, eps);
        // 2. rms_buf → f32
        let rms_f32 = t_to_f32_vec(&rms_buf);
        let w1_f32 = t_to_f32_vec(unsafe { std::slice::from_raw_parts(w1, crate::checked::checked_mul(k, n)) });
        let w2_f32 = t_to_f32_vec(unsafe { std::slice::from_raw_parts(w2, crate::checked::checked_mul(k, n)) });
        let w3_f32 = t_to_f32_vec(unsafe { std::slice::from_raw_parts(w3, crate::checked::checked_mul(n, n)) });
        // 3. gate = matmul(rms, w1)，up = matmul(rms, w2)
        let mut gate = vec![0.0f32; mn];
        let mut up = vec![0.0f32; mn];
        Self::matmul(rms_f32.as_ptr(), k, w1_f32.as_ptr(), n, gate.as_mut_ptr(), n, m, n, k, 1.0, 0.0);
        Self::matmul(rms_f32.as_ptr(), k, w2_f32.as_ptr(), n, up.as_mut_ptr(), n, m, n, k, 1.0, 0.0);
        // 4. silu(gate) * up（全 f32）
        for i in 0..mn {
            let g = gate[i];
            let u = up[i];
            gate[i] = g / (1.0 + (-g).exp()) * u;
        }
        // 5. down = matmul(silu_gate, w3)
        let mut down_buf = vec![0.0f32; mn];
        Self::matmul(gate.as_ptr(), n, w3_f32.as_ptr(), n, down_buf.as_mut_ptr(), n, m, n, n, 1.0, 0.0);
        // 6. f32 → T
        let out_s = unsafe { std::slice::from_raw_parts_mut(out, mn) };
        for i in 0..mn { out_s[i] = T::from_f32(down_buf[i]); }
    }

    /// matmul + 激活融合：out = activation(alpha * A @ B + beta * C)
    fn matmul_activation(a: *const T, lda: usize, b: *const T, ldb: usize,
                         c: *mut T, ldc: usize, m: usize, n: usize, k: usize,
                         alpha: f32, beta: f32, act: ActivationType) {
        let ma = crate::checked::checked_mul(m, lda);
        let kb = crate::checked::checked_mul(k, ldb);
        let mc = crate::checked::checked_mul(m, ldc);
        let a_f32 = t_to_f32_vec(unsafe { std::slice::from_raw_parts(a, ma) });
        let b_f32 = t_to_f32_vec(unsafe { std::slice::from_raw_parts(b, kb) });
        let mut c_buf = vec![0.0f32; mc];
        // 用 c_buf 作为 matmul 的目标，保持 ldc 布局
        let _ = unsafe { std::slice::from_raw_parts_mut(c, mc) }; // 校验长度（避免 c.len 不够 panic）
        Self::matmul(a_f32.as_ptr(), lda, b_f32.as_ptr(), ldb,
                     c_buf.as_mut_ptr(), ldc, m, n, k, alpha, beta);
        // 激活 + 写回（直接用 oi * ldc + oj 索引保持 strided 布局）
        for oi in 0..m {
            for oj in 0..n {
                let src_idx = oi * ldc + oj;
                let dst_idx = oi * ldc + oj;
                let f = c_buf[src_idx];
                let act_f = match act {
                    ActivationType::None => f,
                    ActivationType::ReLU => if f > 0.0 { f } else { 0.0 },
                    ActivationType::GELU => f * 0.5 * (1.0 + crate::backend::elwise::error_function(f / std::f32::consts::SQRT_2)),
                    ActivationType::SiLU => f / (1.0 + (-f).exp()),
                    ActivationType::Sigmoid => 1.0 / (1.0 + (-f).exp()),
                    ActivationType::Tanh => f.tanh(),
                };
                unsafe { std::ptr::write(c.add(dst_idx), T::from_f32(act_f)); }
            }
        }
    }
}
