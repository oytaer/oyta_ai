//! criterion benchmark：matmul + flash_attn + Wgpu 后端
//!
//! 运行：
//!   cargo bench --bench matmul_bench
//!   cargo bench --bench matmul_bench --features wgpu
//!
//! 报告 GFLOPS：
//!   matmul:       flops = 2 * m * n * k / time_sec / 1e9
//!   flash_attn:   flops ≈ 4 * batch * heads * seq * seq * d / time_sec / 1e9
//!
//! 预期 GFLOPS（Intel Xeon 16 cores, AVX2）：
//!   matmul 1024:  ~15-20 GFLOPS
//!   matmul 4096:  ~20-25 GFLOPS（tiled_fma + rayon 多线程 + 可能的 Strassen）
//!   flash_attn seq=512 d=128 heads=8 batch=1: ~1-3 GFLOPS（f64 在线 softmax + 非 fused）

use criterion::{criterion_group, criterion_main, Criterion};
use oyta_core::{Tensor, DynamicShape};
use oyta_device::Cpu;
use rand::rngs::StdRng;
use rand::{SeedableRng, RngExt};
use rand_distr::Uniform;
use std::time::Instant;

// ============================================================
// 辅助：生成随机 Tensor
// ============================================================

fn rand_tensor_2d(rng: &mut StdRng, rows: usize, cols: usize) -> Tensor<f32, Cpu> {
    let uniform = Uniform::new(-1.0f32, 1.0f32).unwrap();
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.sample(uniform)).collect();
    Tensor::<f32, Cpu>::from_data(data, DynamicShape::new(vec![rows, cols])).unwrap()
}

#[cfg(feature = "wgpu")]
fn rand_tensor_4d_wgpu(
    rng: &mut StdRng,
    batch: usize, heads: usize, seq: usize, d: usize,
) -> Tensor<f32, oyta_device::Wgpu> {
    let uniform = Uniform::new(-1.0f32, 1.0f32).unwrap();
    let data: Vec<f32> = (0..batch * heads * seq * d).map(|_| rng.sample(uniform)).collect();
    Tensor::<f32, oyta_device::Wgpu>::from_data(
        data,
        DynamicShape::new(vec![batch, heads, seq, d]),
    ).unwrap()
}

// ============================================================
// Matmul benchmark — CPU
// ============================================================

fn matmul_benchmark_cpu(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    // 覆盖从小到大 + 大矩阵
    let sizes: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096];

    for &size in sizes {
        let a = rand_tensor_2d(&mut rng, size, size);
        let b = rand_tensor_2d(&mut rng, size, size);

        let flops = 2.0 * (size as f64).powi(3);

        c.bench_function(&format!("matmul_cpu_{}", size), |bencher| {
            bencher.iter_custom(|iters| {
                // warmup
                let _ = a.matmul(&b).unwrap();
                let start = Instant::now();
                for _ in 0..iters {
                    let _ = a.matmul(&b).unwrap();
                }
                let elapsed = start.elapsed().as_secs_f64() / iters as f64;
                let gflops = flops / elapsed / 1e9;
                println!("  matmul_cpu_{}: {:.1} GFLOPS (avg {:.3} µs/iter)", size, gflops, elapsed * 1e6);
                std::time::Duration::from_secs_f64(elapsed)
            });
        });
    }
}

// ============================================================
// Matmul benchmark — WGPU（feature gate）
// ============================================================

#[cfg(feature = "wgpu")]
fn matmul_benchmark_wgpu(c: &mut Criterion) {
    use oyta_device::Wgpu;
    let mut rng = StdRng::seed_from_u64(43);

    // WGPU 在小尺寸上 CPU fallback 开销大，从中等尺寸开始
    let sizes: &[usize] = &[256, 512, 1024, 2048];

    for &size in sizes {
        let a_cpu = rand_tensor_2d(&mut rng, size, size);
        let b_cpu = rand_tensor_2d(&mut rng, size, size);

        // Wgpu Tensor 可以用 from_data 构造（数据本身在 CPU，算子调用走 Backend trait）
        let a = Tensor::<f32, Wgpu>::from_data(
            a_cpu.data_slice().to_vec(),
            DynamicShape::new(vec![size, size]),
        ).unwrap();
        let b = Tensor::<f32, Wgpu>::from_data(
            b_cpu.data_slice().to_vec(),
            DynamicShape::new(vec![size, size]),
        ).unwrap();

        let flops = 2.0 * (size as f64).powi(3);

        c.bench_function(&format!("matmul_wgpu_{}", size), |bencher| {
            bencher.iter_custom(|iters| {
                // warmup
                let _ = a.matmul(&b).unwrap();
                let start = Instant::now();
                for _ in 0..iters {
                    let _ = a.matmul(&b).unwrap();
                }
                let elapsed = start.elapsed().as_secs_f64() / iters as f64;
                let gflops = flops / elapsed / 1e9;
                println!("  matmul_wgpu_{}: {:.1} GFLOPS (avg {:.3} µs/iter)", size, gflops, elapsed * 1e6);
                std::time::Duration::from_secs_f64(elapsed)
            });
        });
    }
}

// ============================================================
// FlashAttention benchmark — CPU
// ============================================================

fn flash_attn_benchmark_cpu(c: &mut Criterion) {
    use oyta_ops::attention::flash::flash_attention_multihead;

    let mut rng = StdRng::seed_from_u64(99);

    // 组合：seq × d × heads
    let seqs: &[usize] = &[128, 256, 512];
    let ds: &[usize] = &[64, 128];
    let heads_list: &[usize] = &[4, 8];
    let batch = 1usize;
    let causal = false;

    for &seq in seqs {
        for &d in ds {
            for &heads in heads_list {
                let q = Tensor::<f32, Cpu>::from_data(
                    (0..batch * heads * seq * d)
                        .map(|_| rng.sample(Uniform::new(-1.0f32, 1.0f32).unwrap()))
                        .collect(),
                    DynamicShape::new(vec![batch, heads, seq, d]),
                ).unwrap();
                let k = Tensor::<f32, Cpu>::from_data(
                    (0..batch * heads * seq * d)
                        .map(|_| rng.sample(Uniform::new(-1.0f32, 1.0f32).unwrap()))
                        .collect(),
                    DynamicShape::new(vec![batch, heads, seq, d]),
                ).unwrap();
                let v = Tensor::<f32, Cpu>::from_data(
                    (0..batch * heads * seq * d)
                        .map(|_| rng.sample(Uniform::new(-1.0f32, 1.0f32).unwrap()))
                        .collect(),
                    DynamicShape::new(vec![batch, heads, seq, d]),
                ).unwrap();

                let scale = 1.0 / (d as f32).sqrt();
                let bench_name = format!(
                    "flash_attn_cpu_b{}_h{}_s{}_d{}", batch, heads, seq, d,
                );

                // flash_attn FLOPs ≈ 4 * batch * heads * seq² * d
                let flops = 4.0 * (batch as f64) * (heads as f64)
                    * (seq as f64).powi(2) * (d as f64);

                c.bench_function(&bench_name, |bencher| {
                    bencher.iter_custom(|iters| {
                        // warmup
                        let _ = flash_attention_multihead::<Cpu>(&q, &k, &v, causal, scale);
                        let start = Instant::now();
                        for _ in 0..iters {
                            let _ = flash_attention_multihead::<Cpu>(&q, &k, &v, causal, scale);
                        }
                        let elapsed = start.elapsed().as_secs_f64() / iters as f64;
                        let gflops = flops / elapsed / 1e9;
                        println!(
                            "  flash_attn_cpu b{}_h{}_s{}_d{}: {:.2} GFLOPS (avg {:.3} µs/iter)",
                            batch, heads, seq, d, gflops, elapsed * 1e6,
                        );
                        std::time::Duration::from_secs_f64(elapsed)
                    });
                });
            }
        }
    }
}

// ============================================================
// FlashAttention benchmark — WGPU
// ============================================================

#[cfg(feature = "wgpu")]
fn flash_attn_benchmark_wgpu(c: &mut Criterion) {
    use oyta_device::Wgpu;
    use oyta_ops::attention::flash::flash_attention_multihead;

    let mut rng = StdRng::seed_from_u64(100);

    let seqs: &[usize] = &[128, 256];
    let ds: &[usize] = &[64, 128];
    let heads_list: &[usize] = &[4, 8];
    let batch = 1usize;
    let causal = false;

    for &seq in seqs {
        for &d in ds {
            for &heads in heads_list {
                let q = rand_tensor_4d_wgpu(&mut rng, batch, heads, seq, d);
                let k = rand_tensor_4d_wgpu(&mut rng, batch, heads, seq, d);
                let v = rand_tensor_4d_wgpu(&mut rng, batch, heads, seq, d);

                let scale = 1.0 / (d as f32).sqrt();
                let bench_name = format!(
                    "flash_attn_wgpu_b{}_h{}_s{}_d{}", batch, heads, seq, d,
                );
                let flops = 4.0 * (batch as f64) * (heads as f64)
                    * (seq as f64).powi(2) * (d as f64);

                c.bench_function(&bench_name, |bencher| {
                    bencher.iter_custom(|iters| {
                        let _ = flash_attention_multihead::<Wgpu>(&q, &k, &v, causal, scale);
                        let start = Instant::now();
                        for _ in 0..iters {
                            let _ = flash_attention_multihead::<Wgpu>(&q, &k, &v, causal, scale);
                        }
                        let elapsed = start.elapsed().as_secs_f64() / iters as f64;
                        let gflops = flops / elapsed / 1e9;
                        println!(
                            "  flash_attn_wgpu b{}_h{}_s{}_d{}: {:.2} GFLOPS (avg {:.3} µs/iter)",
                            batch, heads, seq, d, gflops, elapsed * 1e6,
                        );
                        std::time::Duration::from_secs_f64(elapsed)
                    });
                });
            }
        }
    }
}

// ============================================================
// criterion_group / criterion_main
// ============================================================

#[cfg(not(feature = "wgpu"))]
criterion_group!(
    benches,
    matmul_benchmark_cpu,
    flash_attn_benchmark_cpu,
);

#[cfg(feature = "wgpu")]
criterion_group!(
    benches,
    matmul_benchmark_cpu,
    matmul_benchmark_wgpu,
    flash_attn_benchmark_cpu,
    flash_attn_benchmark_wgpu,
);

criterion_main!(benches);
