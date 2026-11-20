//! example_09_dropout.rs — Dropout 训练/推理切换
//!
//! 展示：AtomicBool 全局训练态 / 1/(1-p) scale / rayon 并行

use oyta_tensor::*;
use oyta_ops::dropout::*;

fn main() {
    println!("========================================");
    println!("  example_09: Dropout 训练/推理切换");
    println!("========================================");

    // 1) 检查初始状态
    println!("\n1. Dropout 全局状态:");
    println!("   is_training_mode() = {}", is_training_mode()); // 默认 false（推理态）

    // 2) 推理态 → 直接恒等（不做任何 dropout）
    println!("\n2. 推理态 dropout (p=0.5) — 恒等不变:");
    let x = Tensor::<f32, Cpu>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]
    ).unwrap();
    let after_infer = dropout(&x, 0.5);
    println!("   输入: {:?}", x.data_slice());
    println!("   推理态 dropout(p=0.5): {:?}", after_infer.data_slice());
    assert_eq!(after_infer.data_slice(), x.data_slice());
    println!("   ✅ 推理态恒等验证通过");

    // 3) 训练态 → 随机置零 + 1/(1-p) scale
    println!("\n3. 训练态 dropout (p=0.5) — 随机置零 + scale=2.0:");
    set_training_mode(true);
    assert!(is_training_mode());

    // 多次运行观察不同 mask
    let mut zero_counts = Vec::new();
    for run in 0..5 {
        let after_train = dropout(&x, 0.5);
        let zeros = after_train.data_slice().iter().filter(|&&v| v == 0.0).count();
        zero_counts.push(zeros);
        println!("   run[{}]: {:?}  (零值数={})", run, after_train.data_slice(), zeros);
    }
    println!("   5 次 run 零值数: {:?}", zero_counts);
    // 6 元素 × p=0.5，期望零值 ≈ 3
    let avg_zeros: f32 = zero_counts.iter().sum::<usize>() as f32 / zero_counts.len() as f32;
    println!("   平均零值数: {:.1} (期望 ≈ 3)", avg_zeros);

    // 4) Scale 验证 — 非零元素应该 ≈ 原元素 × 2.0
    println!("\n4. Scale 1/(1-p) = 2.0 验证:");
    let big = Tensor::<f32, Cpu>::ones([10000]);
    let dropped = dropout(&big, 0.5);
    // 非零元素的平均值应该 ≈ 2.0
    let non_zero: Vec<f32> = dropped.data_slice().iter().filter(|&&v| v != 0.0).copied().collect();
    if !non_zero.is_empty() {
        let mean: f32 = non_zero.iter().sum::<f32>() / non_zero.len() as f32;
        println!("   非零元素数: {}", non_zero.len());
        println!("   非零元素均值: {:.4} (期望 ≈ 2.0)", mean);
        assert!((mean - 2.0).abs() < 0.1, "scale 验证失败");
        println!("   ✅ scale = 1/(1-p) 验证通过");
    }

    // 5) 回到推理态
    println!("\n5. 切换回推理态:");
    set_training_mode(false);
    assert!(!is_training_mode());
    let after_back = dropout(&x, 0.5);
    assert_eq!(after_back.data_slice(), x.data_slice());
    println!("   ✅ 推理态恒等验证通过");

    println!("\n✅ 示例 09 完成");
}
