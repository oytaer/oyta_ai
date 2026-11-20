//! example_05_profiling.rs — Profiling 256 槽无锁 AtomicU64
//!
//! 展示：ProfileGuard RAII / OP_NAMES 49 预分配 / enable/disable/reset

use oyta_tensor::*;

fn main() {
    println!("========================================");
    println!("  example_05: Profiling 无锁 AtomicU64");
    println!("========================================");

    // 1) 启用 profiling
    println!("\n1. 启用 profiling:");
    set_profiling_enabled(true);
    println!("   is_profiling_enabled() = {}", is_profiling_enabled());

    // 2) RAII ProfileGuard 自动计时
    println!("\n2. ProfileGuard RAII 自动埋点:");
    {
        let _guard = profile("matmul");
        // 模拟一些工作
        let a = Tensor::<f32, Cpu>::randn([64, 64]);
        let b = Tensor::<f32, Cpu>::randn([64, 64]);
        let _ = a.matmul(&b);
    } // drop 时自动记录耗时

    {
        let _guard = profile("softmax");
        let t = Tensor::<f32, Cpu>::randn([32, 128]);
        let _ = t.softmax(-1);
    }

    {
        let _guard = profile("add");
        let a = Tensor::<f32, Cpu>::ones([100, 100]);
        let b = Tensor::<f32, Cpu>::ones([100, 100]);
        let _ = a.add(&b);
    }

    // 3) 查看统计 — 固定 256 槽，AtomicU64 无锁
    println!("\n3. 查看统计 (256 槽 AtomicU64 无锁实现):");
    let stats = snapshot();
    println!("   总算子种类: {}", stats.len());
    for (name, s) in stats.iter().take(5) {
        println!("   {:<16} count={:<6} total_ns={:<12} avg_ns={}",
            name, s.count, s.total_ns, s.avg_ns());
    }

    // 4) reset 清空
    println!("\n4. reset_profile() 清空统计:");
    reset_profile();
    let stats_after = snapshot();
    println!("   reset 后算子种类: {}", stats_after.len());
    assert!(stats_after.is_empty());
    println!("   ✅ reset 成功");

    // 5) profiling 禁用后不再记录
    println!("\n5. set_profiling_enabled(false) 后不再记录:");
    set_profiling_enabled(false);
    assert!(!is_profiling_enabled());

    {
        let _guard = profile("should_not_record");
        let a = Tensor::<f32, Cpu>::ones([10, 10]);
        let _ = a.sum();
    }
    let stats = snapshot();
    let recorded = stats.iter().any(|(name, _)| *name == "should_not_record");
    println!("   disabled 期间记录了 should_not_record: {}", recorded);
    assert!(!recorded);
    println!("   ✅ disable 正确阻止记录");

    // 6) 49 个预分配算子名检查
    println!("\n6. 预分配算子名常量 (49 个 OP_NAMES):");
    use oyta_ops::profiling::OP_NAMES;
    println!("   MAX_OPS = {}", MAX_OPS);
    println!("   OP_NAMES 前 5: {:?}", &OP_NAMES[..5.min(OP_NAMES.len())]);
    println!("   OP_NAMES 总长度: {}", OP_NAMES.len());

    // 7) matmul 真实 profiling
    println!("\n7. 大矩阵 matmul profiling 真实耗时:");
    set_profiling_enabled(true);
    let start = std::time::Instant::now();
    {
        let _guard = profile("matmul_1024");
        let a = Tensor::<f32, Cpu>::randn([1024, 1024]);
        let b = Tensor::<f32, Cpu>::randn([1024, 1024]);
        let _ = a.matmul(&b);
    }
    println!("   wall clock: {:?}", start.elapsed());

    let stats = snapshot();
    for (name, s) in stats {
        if name == "matmul_1024" {
            println!("   ProfileGuard 记录: count={}, total_ns={} ({:.1}ms)",
                s.count, s.total_ns, s.total_ns as f64 / 1e6);
        }
    }

    println!("\n✅ 示例 05 完成");
}
