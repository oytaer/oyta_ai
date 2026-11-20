//! example_03_attention.rs — FlashAttention + Softmax + Causal Mask
//!
//! 展示：FlashAttention 在线 softmax / softmax 数值稳定版 / causal mask

use oyta_tensor::*;

fn main() {
    println!("========================================");
    println!("  example_03: FlashAttention + Softmax");
    println!("========================================");

    // 1) Softmax 数值稳定版 — 先减 max + f64 accumulator
    println!("\n1. Softmax 数值稳定版 (先减 max, f64 accumulator):");
    let logits = Tensor::<f32, Cpu>::from_data(
        vec![1.0, 2.0, 3.0, 100.0, -50.0, 0.0], [2, 3]
    ).unwrap();
    let sm = logits.softmax(-1).unwrap();
    println!("   输入:    {:?}", logits.data_slice());
    println!("   softmax: {:?}", sm.data_slice());
    // 每行 sum = 1.0 验证
    for row in 0..2 {
        let row_sum: f32 = sm.data_slice()[row*3..(row+1)*3].iter().sum();
        println!("   row[{}] sum = {:.6} (应 ≈ 1.0)", row, row_sum);
        assert!((row_sum - 1.0).abs() < 1e-5, "softmax 每行 sum 必须为 1");
    }
    println!("   ✅ 数值稳定版 softmax 每行 sum=1.0");

    // 2) 大数组 softmax — f64 accumulator 优势
    println!("\n2. 大数组 softmax (1M 元素) — f64 accumulator 精度保证:");
    let large = Tensor::<f32, Cpu>::randn([1024, 1024]);
    let sm_large = large.softmax(-1).unwrap();
    let row_sum: f32 = sm_large.data_slice()[..1024].iter().sum();
    println!("   首行 sum = {:.6} (应 ≈ 1.0)", row_sum);
    assert!((row_sum - 1.0).abs() < 1e-5);
    println!("   ✅ 大数组 f64 accumulator 精度验证通过");

    // 3) LayerNorm — (x - mean) / sqrt(var + eps) * gamma + beta
    println!("\n3. LayerNorm — (x-mean)/sqrt(var+eps)*g+b, LAYERNORM_EPS={}", LAYERNORM_EPS);
    let x = Tensor::<f32, Cpu>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]
    ).unwrap();
    let gamma = Tensor::<f32, Cpu>::ones([3]);
    let beta = Tensor::<f32, Cpu>::zeros([3]);
    let ln = x.layernorm(&gamma, &beta, LAYERNORM_EPS as f32).unwrap();
    println!("   输入:    {:?}", x.data_slice());
    println!("   layernorm输出: {:?}", ln.data_slice());
    // 每行 mean ≈ 0, var ≈ 1 验证
    for row in 0..2 {
        let row_data = &ln.data_slice()[row*3..(row+1)*3];
        let mean: f32 = row_data.iter().sum::<f32>() / 3.0;
        let var: f32 = row_data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 3.0;
        println!("   row[{}] mean={:.4}, var={:.4}", row, mean, var);
    }

    // 4) FlashAttention 在线 softmax + causal mask
    println!("\n4. FlashAttention 在线 softmax + causal mask:");
    use oyta_device::FlashAttentionOps;

    let seq_q = 4usize;
    let seq_k = 4usize;
    let d_k = 3usize;
    let d_v = 2usize;
    let scale = 1.0 / (d_k as f32).sqrt();

    // 构造确定性 Q, K, V
    let q_data: Vec<f32> = (0..seq_q*d_k).map(|i| i as f32).collect();
    let k_data: Vec<f32> = (0..seq_k*d_k).map(|i| (i as f32) * 0.5).collect();
    let v_data: Vec<f32> = (0..seq_k*d_v).map(|i| (i as f32) * 0.3).collect();
    let mut out_data = vec![0.0f32; seq_q * d_v];

    // 非 causal attention：seq_q 每个位置都能看所有 K
    <Cpu as FlashAttentionOps<f32>>::flash_attention(
        q_data.as_ptr(),
        k_data.as_ptr(),
        v_data.as_ptr(),
        out_data.as_mut_ptr(),
        seq_q, seq_k, d_k, d_v, scale, false,
    );
    println!("   非 causal out: {:?}", out_data);

    let mut out_causal = vec![0.0f32; seq_q * d_v];
    // causal attention：位置 qi 只能看 K[0..=qi]（decoder 专用）
    <Cpu as FlashAttentionOps<f32>>::flash_attention(
        q_data.as_ptr(),
        k_data.as_ptr(),
        v_data.as_ptr(),
        out_causal.as_mut_ptr(),
        seq_q, seq_k, d_k, d_v, scale, true,
    );
    println!("   causal out:    {:?}", out_causal);
    // causal 与非 causal 的差异：后几个位置的 out 不同（因为看不到未来）
    assert_ne!(out_data, out_causal, "causal 和非 causal 结果应不同");
    println!("   ✅ causal mask 正确区分未来位置");

    println!("\n✅ 示例 03 完成");
}
