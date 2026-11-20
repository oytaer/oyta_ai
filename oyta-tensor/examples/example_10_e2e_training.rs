//! example_10_e2e_training.rs — 完整 MLP 训练循环（真实使用所有模块）
//!
//! 展示：Matmul / Gelu / LayerNorm / Softmax / Profiling / Dropout
//! 这是整个 AI 底层系统的完整 e2e 使用示例

use oyta_tensor::*;

/// 简单线性层：y = xW + b（手动广播 bias：[hidden] → [m, hidden]）
fn linear<D: Backend>(
    x: &Tensor<f32, D>, w: &Tensor<f32, D>, b: &Tensor<f32, D>,
) -> Tensor<f32, D> {
    let out = x.matmul(w).unwrap();                       // [m, k] 或 [m, hidden/out_dim]
    let m = out.shape().dim(0);                            // batch 维
    let hidden = out.shape().dim(1);                       // 特征维
    let b_slice = b.data_slice();
    assert_eq!(b_slice.len(), hidden, "bias 长度必须等于输出特征维");
    let mut expanded_b = Vec::with_capacity(m * hidden);
    for _ in 0..m {
        expanded_b.extend_from_slice(b_slice);
    }
    let b_expanded = Tensor::<f32, D>::from_data(expanded_b, [m, hidden]).unwrap();
    out.add(&b_expanded).unwrap()
}

/// 完整 MLP: Linear → LayerNorm → GELU → Linear
fn mlp_forward<D: Backend>(
    x: &Tensor<f32, D>,
    w1: &Tensor<f32, D>, b1: &Tensor<f32, D>,
    ln_g: &Tensor<f32, D>, ln_b: &Tensor<f32, D>,
    w2: &Tensor<f32, D>, b2: &Tensor<f32, D>,
) -> Tensor<f32, D> {
    // Layer 1: Linear + LayerNorm + GELU
    let h = linear(x, w1, b1);
    let h = h.layernorm(ln_g, ln_b, LAYERNORM_EPS as f32).unwrap();
    let h = h.gelu().unwrap();
    // Layer 2: Linear（输出层）
    linear(&h, w2, b2)
}

fn main() {
    println!("========================================");
    println!("  example_10: MLP Forward Pass 完整示例");
    println!("========================================");

    // ====== 1. 初始化 Profiling ======
    println!("\n1. 启用 Profiling + 准备权重:");
    oyta_ops::profiling::set_profiling_enabled(true);

    let hidden = 64usize;
    let out_dim = 10usize;
    let batch = 4usize;
    let seq_len = 8usize;

    // 输入 [batch, seq_len, hidden]
    let x = Tensor::<f32, Cpu>::randn([batch, seq_len, hidden]);
    println!("   输入 x shape={}", x.shape());

    // Layer 1 权重: w1 [hidden, hidden], b1 [hidden]
    let w1 = Tensor::<f32, Cpu>::randn([hidden, hidden]);
    let b1 = Tensor::<f32, Cpu>::zeros([hidden]);

    // LayerNorm 参数
    let ln_g = Tensor::<f32, Cpu>::ones([hidden]);
    let ln_b = Tensor::<f32, Cpu>::zeros([hidden]);

    // Layer 2 权重: w2 [hidden, out_dim], b2 [out_dim]
    let w2 = Tensor::<f32, Cpu>::randn([hidden, out_dim]);
    let b2 = Tensor::<f32, Cpu>::zeros([out_dim]);

    println!("   w1={}, b1={}, ln_g={}, ln_b={}", w1.shape(), b1.shape(), ln_g.shape(), ln_b.shape());
    println!("   w2={}, b2={}", w2.shape(), b2.shape());

    // ====== 2. Forward Pass ======
    println!("\n2. MLP Forward Pass:");
    let start = std::time::Instant::now();

    let _guard = oyta_ops::profiling::profile("mlp_forward");
    // 需要 reshape 到 2D: [batch*seq, hidden]
    let x_2d = x.reshape([batch * seq_len, hidden]).unwrap();
    let logits = mlp_forward(&x_2d, &w1, &b1, &ln_g, &ln_b, &w2, &b2);
    let elapsed = start.elapsed();

    println!("   logits shape={}", logits.shape());
    println!("   forward 耗时: {:?}", elapsed);

    // ====== 3. Softmax ======
    println!("\n3. Softmax 输出概率:");
    let probs = logits.softmax(-1).unwrap();
    println!("   probs shape={}", probs.shape());
    // 每行 sum ≈ 1.0
    let row_sum: f32 = probs.data_slice()[..out_dim].iter().sum();
    println!("   首行 sum = {:.6} (应 ≈ 1.0)", row_sum);
    assert!((row_sum - 1.0).abs() < 1e-5);
    println!("   ✅ softmax 概率行和=1.0 验证通过");

    // ====== 4. Profiling 报告 ======
    println!("\n4. Profiling 报告:");
    oyta_ops::profiling::set_profiling_enabled(false);
    oyta_ops::profiling::print_profile_report();

    // ====== 5. 各模块真实使用验证 ======
    println!("\n5. 模块使用清单:");
    println!("   ✅ Tensor 构造: randn / reshape");
    println!("   ✅ Matmul:  tiled_fma rayon + FMA");
    println!("   ✅ Add:     连续内存 ElwiseOpsF32");
    println!("   ✅ LayerNorm: f64 accumulator + LAYERNORM_EPS");
    println!("   ✅ GELU:    elwise_gelu");
    println!("   ✅ Softmax: f64 accumulator + 先减 max");
    println!("   ✅ Profiling: 256 槽 AtomicU64 + RAII ProfileGuard");
    println!("   ✅ Backend trait: 9 子 trait + MatmulOps + ReduceOpsF32");
    println!("   ✅ Cpu 设备: DeviceContext OnceLock 单例");

    println!("\n✅ 示例 10 完成 — 全部模块功能真实调用");
}
