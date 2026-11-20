//! example_11_wgpu_verify.rs：WGPU 真实 GPU matmul 验证
//!
//! 运行：cargo run --example example_11_wgpu_verify --features wgpu

#[cfg(not(feature = "wgpu"))]
fn main() {
    println!("此 example 需要 --features wgpu 才能运行");
}

#[cfg(feature = "wgpu")]
fn main() {
    use rand::SeedableRng;
    use rand::RngExt;
    use rand_distr::Uniform;
    use std::time::Instant;
    use oyta_device::{Backend, DeviceContext};
    use oyta_tensor::{DynamicShape, Cpu, Tensor, Wgpu};

    println!("{}", "=".repeat(60));
    println!("  WGPU 真实 GPU matmul 验证");
    println!("{}", "=".repeat(60));

    // 1. CPU 参考基准
    let handle_cpu = <Cpu as DeviceContext>::get();
    println!("\n[CPU] Device: {}", handle_cpu.properties.name);

    // 2. WGPU
    let handle_wgpu = <Wgpu as DeviceContext>::get();
    println!("[WGPU] Device: {}", handle_wgpu.properties.name);

    // 3. 逐尺寸对比
    println!("\n{}", "-".repeat(60));
    println!("  正确性验证（CPU vs WGPU）");
    println!("{}", "-".repeat(60));

    let sizes = [16usize, 32, 64, 128];
    let mut all_pass = true;

    for size in sizes {
        let m = size;
        let n = size;
        let k = size;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42 + size as u64);
        let uniform = Uniform::new(-1.0f32, 1.0).expect("Uniform");
        let a_data: Vec<f32> = (0..m * k).map(|_| rng.sample(uniform)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|_| rng.sample(uniform)).collect();

        // CPU
        let a_cpu = Tensor::<f32, Cpu>::from_data(a_data.clone(), DynamicShape::new([m, k])).expect("a_cpu");
        let b_cpu = Tensor::<f32, Cpu>::from_data(b_data.clone(), DynamicShape::new([k, n])).expect("b_cpu");
        let c_cpu = a_cpu.matmul(&b_cpu).expect("cpu matmul");
        let cpu_result = c_cpu.data_slice();

        // WGPU
        let a_wgpu = Tensor::<f32, Wgpu>::from_data(a_data.clone(), DynamicShape::new([m, k])).expect("a_wgpu");
        let b_wgpu = Tensor::<f32, Wgpu>::from_data(b_data.clone(), DynamicShape::new([k, n])).expect("b_wgpu");
        let c_wgpu = a_wgpu.matmul(&b_wgpu).expect("wgpu matmul");
        let wgpu_result = c_wgpu.data_slice();

        let mut max_diff: f32 = 0.0;
        let mut rel_violations = 0usize;
        for (&c, &w) in cpu_result.iter().zip(wgpu_result.iter()) {
            let diff = (c - w).abs();
            if diff > max_diff { max_diff = diff; }
            let denom = c.abs().max(1.0);
            if diff / denom > 1e-4 { rel_violations += 1; }
        }

        let pass = rel_violations == 0 && max_diff < 1e-2;
        if !pass { all_pass = false; }

        // GFLOPS 计时
        let cpu_start = Instant::now();
        for _ in 0..10 {
            let _ = a_cpu.matmul(&b_cpu).expect("cpu");
        }
        let cpu_elapsed = cpu_start.elapsed().as_secs_f64() / 10.0;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let cpu_gflops = flops / cpu_elapsed / 1e9;

        let label = if pass { "PASS" } else { "FAIL" };
        println!(
            "  [{}] {}×{} × {}×{} → max_diff={:.2e} rel_violations={} CPU={:.2}GFLOPS",
            label, m, k, k, n, max_diff, rel_violations, cpu_gflops,
        );
    }

    println!("\n{}", "=".repeat(60));
    if all_pass {
        println!("  ✅ 全部通过！WGPU GPU matmul 结果与 CPU 一致");
    } else {
        println!("  ❌ 有失败！需调试");
    }
    println!("{}", "=".repeat(60));
}
