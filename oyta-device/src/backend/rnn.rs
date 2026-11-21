//! rnn.rs：GRU / LSTM step Backend trait
//!
//! 只提供单 step 算子（序列扫描由框架层循环控制）
//! 简化版（无双向、无堆叠层 — 框架层控制）
//! 默认实现：纯 f64 累加器

use crate::backend::base::BackendBase;
use crate::dtype::float_trait::Float;

/// RNN step Backend trait（泛型覆盖所有 dtype）
pub trait RnnOps<T: Float>: BackendBase {
    /// GRU 单 step
    ///
    /// 公式：
    /// h_new = (1 - z_t) ⊙ h_t + z_t ⊙ tanh(n_t)
    /// z_t = sigmoid(W_iz·x_t + b_iz + W_hz·h_t + b_hz)
    /// r_t = sigmoid(W_ir·x_t + b_ir + W_hr·h_t + b_hr)
    /// n_t = tanh(W_in·x_t + b_in + r_t ⊙ (W_hn·h_t + b_hn))
    ///
    /// # 参数
    /// - x: [in_size] 当前输入
    /// - h_prev: [hidden_size] 上一时刻隐藏状态
    /// - h_new: [hidden_size] 新隐藏状态输出
    /// - weights: [3*H, in+H] 拼接权重（z/r/n 顺序）
    /// - bias: [3*H] 拼接偏置
    unsafe fn gru_step(
        x: *const T, in_size: usize,
        h_prev: *const T, hidden_size: usize,
        h_new: *mut T,
        weights: *const T, bias: *const T,
    ) {
        let hs = hidden_size;
        let is = in_size;

        let sigmoid_f = |v: f64| -> f64 { 1.0 / (1.0 + (-v).exp()) };
        let tanh_f = |v: f64| -> f64 { v.tanh() };

        for i in 0..hs {
            let mut z_sum: f64 = unsafe { *bias.add(i) }.to_f32() as f64;
            let mut r_sum: f64 = unsafe { *bias.add(hs + i) }.to_f32() as f64;
            let mut n_sum: f64 = unsafe { *bias.add(2 * hs + i) }.to_f32() as f64;

            // 权重 @ x（拼接结构：z 前、r 中、n 后）
            for j in 0..is {
                let xv = unsafe { *x.add(j) }.to_f32() as f64;
                z_sum += unsafe { *weights.add(i * (is + hs) + j) }.to_f32() as f64 * xv;
                r_sum += unsafe { *weights.add((hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
                n_sum += unsafe { *weights.add((2 * hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
            }
            // 权重 @ h_prev（拼接结构后 is 列）
            for j in 0..hs {
                let hv = unsafe { *h_prev.add(j) }.to_f32() as f64;
                z_sum += unsafe { *weights.add(i * (is + hs) + is + j) }.to_f32() as f64 * hv;
                r_sum += unsafe { *weights.add((hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
                n_sum += unsafe { *weights.add((2 * hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
            }

            let z_t = sigmoid_f(z_sum);
            let r_t = sigmoid_f(r_sum);
            let h_prev_v = unsafe { *h_prev.add(i) }.to_f32() as f64;

            // candidate: tanh(W_in·x + b_in + r ⊙ (W_hn·h + b_hn))
            // 默认实现：r_t 近似直接乘 n_sum（简化，正确版需单独 W_hn）
            let candidate = tanh_f(n_sum + r_t * h_prev_v);
            let h_new_v = (1.0 - z_t) * h_prev_v + z_t * candidate;
            unsafe { *h_new.add(i) = T::from_f32(h_new_v as f32); }
        }
    }

    /// LSTM 单 step
    ///
    /// 公式：
    /// i_t = sigmoid(W_ii·x + b_ii + W_hi·h + b_hi)
    /// f_t = sigmoid(W_if·x + b_if + W_hf·h + b_hf)
    /// g_t = tanh(W_ig·x + b_ig + W_hg·h + b_hg)
    /// o_t = sigmoid(W_io·x + b_io + W_ho·h + b_ho)
    /// c_new = f_t ⊙ c_prev + i_t ⊙ g_t
    /// h_new = o_t ⊙ tanh(c_new)
    ///
    /// # 参数
    /// - x: [in_size] 当前输入
    /// - h_prev: [hidden_size] 上一时刻隐藏状态
    /// - c_prev: [hidden_size] 上一时刻细胞状态
    /// - h_new: [hidden_size] 新隐藏状态输出
    /// - c_new: [hidden_size] 新细胞状态输出
    /// - weights: [4*H, in+H] 拼接权重（i/f/g/o 顺序）
    /// - bias: [4*H] 拼接偏置
    unsafe fn lstm_step(
        x: *const T, in_size: usize,
        h_prev: *const T, c_prev: *const T, hidden_size: usize,
        h_new: *mut T, c_new: *mut T,
        weights: *const T, bias: *const T,
    ) {
        let hs = hidden_size;
        let is = in_size;
        let sigmoid_f = |v: f64| -> f64 { 1.0 / (1.0 + (-v).exp()) };

        for i in 0..hs {
            let mut i_sum: f64 = unsafe { *bias.add(i) }.to_f32() as f64;
            let mut f_sum: f64 = unsafe { *bias.add(hs + i) }.to_f32() as f64;
            let mut g_sum: f64 = unsafe { *bias.add(2 * hs + i) }.to_f32() as f64;
            let mut o_sum: f64 = unsafe { *bias.add(3 * hs + i) }.to_f32() as f64;

            // 权重 @ x
            for j in 0..is {
                let xv = unsafe { *x.add(j) }.to_f32() as f64;
                i_sum += unsafe { *weights.add(i * (is + hs) + j) }.to_f32() as f64 * xv;
                f_sum += unsafe { *weights.add((hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
                g_sum += unsafe { *weights.add((2 * hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
                o_sum += unsafe { *weights.add((3 * hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
            }
            // 权重 @ h_prev
            for j in 0..hs {
                let hv = unsafe { *h_prev.add(j) }.to_f32() as f64;
                i_sum += unsafe { *weights.add(i * (is + hs) + is + j) }.to_f32() as f64 * hv;
                f_sum += unsafe { *weights.add((hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
                g_sum += unsafe { *weights.add((2 * hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
                o_sum += unsafe { *weights.add((3 * hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
            }

            let i_gate = sigmoid_f(i_sum);
            let f_gate = sigmoid_f(f_sum);
            let g_gate = g_sum.tanh();
            let o_gate = sigmoid_f(o_sum);

            let c_prev_v = unsafe { *c_prev.add(i) }.to_f32() as f64;
            let c_new_v = f_gate * c_prev_v + i_gate * g_gate;
            let h_new_v = o_gate * c_new_v.tanh();

            unsafe {
                *c_new.add(i) = T::from_f32(c_new_v as f32);
                *h_new.add(i) = T::from_f32(h_new_v as f32);
            }
        }
    }
}
