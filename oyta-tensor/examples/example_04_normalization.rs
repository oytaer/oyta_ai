//! example_04_normalization.rs — LayerNorm / RMSNorm / Norm 融合
//!
//! 展示：完整 LayerNorm / RMSNorm / f64 accumulator / pairwise_sum 精度

use oyta_tensor::*;

fn main() {
    println!("========================================");
    println!("  example_04: LayerNorm / RMSNorm");
    println!("========================================");

    // 1) LayerNorm 完整公式演示
    println!("\n1. LayerNorm 完整公式: out = (x - mean) / sqrt(var + eps) * gamma + beta");
    println!("   LAYERNORM_EPS = {}, RMSNORM_EPS = {}", LAYERNORM_EPS, RMSNORM_EPS);

    let x = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
    let gamma = Tensor::<f32, Cpu>::ones([3]);
    let beta = Tensor::<f32, Cpu>::zeros([3]);

    // 手动计算第一行 LayerNorm 验证
    let row0 = &x.data_slice()[0..3];
    let mean0: f64 = row0.iter().map(|&v| v as f64).sum::<f64>() / 3.0;
    let var0: f64 = row0.iter().map(|&v| ((v as f64) - mean0).powi(2)).sum::<f64>() / 3.0;
    let std0 = (var0 + LAYERNORM_EPS as f64).sqrt();
    let manual_ln0: Vec<f32> = row0.iter().map(|&v| ((v as f64 - mean0) / std0) as f32).collect();
    println!("   手动计算 row[0] LayerNorm: {:?}", manual_ln0);

    let ln = x.layernorm(&gamma, &beta, LAYERNORM_EPS as f32).unwrap();
    println!("   Backend layernorm 输出:    {:?}", ln.data_slice());

    let ln0 = &ln.data_slice()[0..3];
    for i in 0..3 {
        assert!((ln0[i] - manual_ln0[i]).abs() < 1e-4, "LayerNorm 精确值不符");
    }
    println!("   ✅ LayerNorm 精确值手动验证通过");

    // 2) RMSNorm — 比 LayerNorm 少 mean 计算（更简单更快）
    println!("\n2. RMSNorm 公式: out = x / sqrt(mean(x²) + eps) * gamma");
    use oyta_device::ReduceOps;

    let rms_eps = RMSNORM_EPS as f32;
    let gamma = Tensor::<f32, Cpu>::from_data(vec![1.0, 1.0, 1.0, 1.0], [4]).unwrap();
    let x2 = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0], [1, 4]).unwrap();

    let mut rms_out = vec![0.0f32; 4];
    <Cpu as ReduceOps<f32>>::rmsnorm(
        x2.data_ptr(), rms_out.as_mut_ptr(),
        gamma.data_ptr(),
        1, 4, rms_eps,
    );
    println!("   RMSNorm 输入:  {:?}", x2.data_slice());
    println!("   RMSNorm 输出:  {:?}", rms_out);

    // 手动验证：sqrt(mean(1+4+9+16) + eps) = sqrt(7.5 + 1e-6) ≈ 2.7386
    let mean_sq = (1.0_f32 + 4.0 + 9.0 + 16.0) / 4.0; // = 7.5
    let rms = (mean_sq + RMSNORM_EPS as f32).sqrt();
    let manual: Vec<f32> = [1.0, 2.0, 3.0, 4.0].iter().map(|v| v / rms).collect();
    println!("   手动 RMSNorm:  {:?}", manual);
    for i in 0..4 {
        assert!((rms_out[i] - manual[i]).abs() < 1e-4, "RMSNorm 精确值不符");
    }
    println!("   ✅ RMSNorm 精确值手动验证通过");

    // 3) pairwise_sum 精度 — 大数组两两递归归约
    println!("\n3. pairwise_sum 递归二分归约（大数组精度保证）:");
    use oyta_device::pairwise_sum;

    // 构造 1M 个 1.0 的数组：理想 sum = 1e6
    let big: Vec<f32> = vec![1.0f32; 1_000_000];
    let sum_pairwise = pairwise_sum(&big, |x| x as f64);
    let naive_sum: f32 = big.iter().sum();
    println!("   理想 sum = 1_000_000.0");
    println!("   顺序累加: {:.1} (f32 可能丢失精度)", naive_sum);
    println!("   pairwise: {:.1} (f64 累加器保证)", sum_pairwise);
    println!("   pairwise 误差: {:.1}", (sum_pairwise - 1_000_000.0).abs());

    // 4) layernorm + gelu 融合（FusedOps）
    println!("\n4. 融合算子 layernorm_gelu（一次调用两个操作）:");
    use oyta_device::FusedOps;

    let x3 = Tensor::<f32, Cpu>::from_data(
        vec![-1.0, 0.0, 1.0, 2.0, -2.0, 3.0], [2, 3]
    ).unwrap();
    let g = Tensor::<f32, Cpu>::ones([3]);
    let b = Tensor::<f32, Cpu>::zeros([3]);
    let mut fused_out = vec![0.0f32; 6];

    <Cpu as FusedOps<f32>>::layernorm_gelu(
        x3.data_ptr(), fused_out.as_mut_ptr(),
        g.data_ptr(), b.data_ptr(),
        2, 3, LAYERNORM_EPS as f32,
    );
    println!("   融合 layernorm+gelu 输出: {:?}", fused_out);

    // 手动分步验证
    let ln_out = x3.layernorm(&g, &b, LAYERNORM_EPS as f32).unwrap();
    let step_by_step = ln_out.gelu().unwrap();
    println!("   分步 layernorm→gelu:      {:?}", step_by_step.data_slice());

    for i in 0..6 {
        assert!((fused_out[i] - step_by_step.data_slice()[i]).abs() < 1e-4);
    }
    println!("   ✅ 融合算子与分步计算结果一致");

    println!("\n✅ 示例 04 完成");
}
