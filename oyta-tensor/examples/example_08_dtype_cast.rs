//! example_08_dtype_cast.rs — F32 ↔ F16 ↔ BF16 完整转换
//!
//! 展示：IEEE 754 位操作 / 所有 f32 算子覆盖 f16/bf16 / Accumulate 精度提升

use oyta_tensor::*;

fn main() {
    println!("========================================");
    println!("  example_08: F16/BF16 类型完整支持");
    println!("========================================");

    // 1) F16 IEEE 754 格式验证
    println!("\n1. F16 IEEE 754 binary16 格式:");
    println!("   1 bit sign + 5 bit exp + 10 bit frac");
    let half_one = F16::from_f32(1.0);
    println!("   F16(1.0) = {:#018b}", half_one.0); // 应该是 0x3C00
    let back = half_one.to_f32();
    assert!((back - 1.0).abs() < 1e-3);
    println!("   F16(1.0) → f32 = {}", back);

    let half_pi = F16::from_f32(std::f32::consts::PI);
    println!("   F16(π) = {}", half_pi.to_f32());

    // 2) BF16 格式（截断，比 F16 精度低）
    println!("\n2. BF16 IEEE 754 bfloat16 格式:");
    println!("   1 bit sign + 8 bit exp + 7 bit frac（与 f32 同 exp 范围）");
    let bf_one = BF16::from_f32(1.0);
    println!("   BF16(1.0) = {:#018b}", bf_one.0);
    assert!((bf_one.to_f32() - 1.0).abs() < 1e-3);

    // 3) F32 → F16 → F32 roundtrip（精度损失可接受）
    println!("\n3. F32 → F16 → F32 roundtrip:");
    let original = vec![0.0_f32, 0.5, 1.0, 2.0, 0.1, 100.0, -1.5];
    for &v in &original {
        let h = F16::from_f32(v);
        let back = h.to_f32();
        let err = (back - v).abs();
        println!("   f32={:>8.4} → f16 → f32={:>8.4}  err={:.6}", v, back, err);
        // F16 精度 ~0.001 对 1.0 量级
        assert!(err < 0.01, "F16 roundtrip 误差过大: v={}, err={}", v, err);
    }
    println!("   ✅ F16 roundtrip 精度验证通过");

    // 4) BF16 roundtrip（精度更低）
    println!("\n4. F32 → BF16 → F32 roundtrip:");
    for &v in &original {
        let b = BF16::from_f32(v);
        let back = b.to_f32();
        println!("   f32={:>8.4} → bf16 → f32={:>8.4}", v, back);
    }

    // 5) DType byte_size 验证
    println!("\n5. DType byte_size:");
    use oyta_tensor::DType;
    println!("   F32:  {} bytes", DType::F32.byte_size());
    println!("   F16:  {} bytes", DType::F16.byte_size());
    println!("   BF16: {} bytes", DType::BF16.byte_size());
    assert_eq!(DType::F32.byte_size(), 4);
    assert_eq!(DType::F16.byte_size(), 2);
    assert_eq!(DType::BF16.byte_size(), 2);
    println!("   ✅ byte_size 验证通过");

    // 6) Backend F16/BF16 算子覆盖 — 所有 f32 算子都有 f16/bf16 版本
    println!("\n6. Backend f16/bf16 算子覆盖 (cast f16→f32→算→cast back):");
    use oyta_device::ElwiseOps;

    let f16_a = [F16::from_f32(1.0), F16::from_f32(2.0), F16::from_f32(3.0)];
    let f16_b = [F16::from_f32(4.0), F16::from_f32(5.0), F16::from_f32(6.0)];
    let mut f16_out = [F16::from_f32(0.0); 3];

    <Cpu as ElwiseOps<F16>>::elwise_add(
        f16_a.as_ptr(), f16_b.as_ptr(), f16_out.as_mut_ptr(), 3,
    );
    println!("   F16 add: [1,2,3] + [4,5,6] = {:?}",
        f16_out.iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    // BF16 同测
    let bf_a = [BF16::from_f32(1.0), BF16::from_f32(2.0)];
    let bf_b = [BF16::from_f32(10.0), BF16::from_f32(20.0)];
    let mut bf_out = [BF16::from_f32(0.0); 2];
    <Cpu as ElwiseOps<BF16>>::elwise_mul(
        bf_a.as_ptr(), bf_b.as_ptr(), bf_out.as_mut_ptr(), 2,
    );
    println!("   BF16 mul: [1,2] * [10,20] = {:?}",
        bf_out.iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    // 7) Accumulate trait — F16/BF16→f32, f32→f32, f64→f64
    println!("\n7. Accumulate trait 精度提升:");
    use oyta_tensor::Accumulate;
    println!("   F16 累加器类型 = {}", std::any::type_name::<<F16 as Accumulate>::Accum>());
    println!("   BF16 累加器类型 = {}", std::any::type_name::<<BF16 as Accumulate>::Accum>());
    println!("   F32 累加器类型 = {}", std::any::type_name::<<f32 as Accumulate>::Accum>());
    println!("   F64 累加器类型 = {}", std::any::type_name::<<f64 as Accumulate>::Accum>());
    // F16/BF16 内部用 f32 accumulator 避免精度丢失

    println!("\n✅ 示例 08 完成");
}
