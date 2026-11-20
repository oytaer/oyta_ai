//! example_02_matmul.rs — Matmul 算法选择 + BLAS 加速
//!
//! 展示：matmul 四档算法选择（naive / tiled_fma / strassen / BLAS）
//! 精确分块参数 BM=BN=BK=128、Strassen crossover=128

use oyta_tensor::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn rand_tensor(shape: &[usize], seed: u64) -> Tensor<f32, Cpu> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    Tensor::<f32, Cpu>::from_data(data, shape).unwrap()
}

fn main() {
    println!("========================================");
    println!("  example_02: Matmul 算法选择");
    println!("========================================");

    // 1) 极小矩阵 → naive 三重循环
    println!("\n1. 极小矩阵 (m=n=k=16) → naive 三重循环:");
    let a = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let b = Tensor::<f32, Cpu>::from_data(vec![5.0, 6.0, 7.0, 8.0], [2, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    println!("   [1,2;3,4] × [5,6;7,8] = {:?}", c.data_slice());
    // 手动验证: [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8] = [19, 22; 43, 50]
    assert_eq!(c.data_slice(), &[19.0, 22.0, 43.0, 50.0]);
    println!("   ✅ 手动验证正确");

    // 2) 点积特化路径 — [1,k] × [k,1]
    println!("\n2. 点积特化 [1,k] × [k,1] → dot_product_flat:");
    let v = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0], [1, 3]).unwrap();
    let u = Tensor::<f32, Cpu>::from_data(vec![4.0, 5.0, 6.0], [3, 1]).unwrap();
    let dot = v.matmul(&u).unwrap();
    println!("   [1,2,3] · [4,5,6] = {}", dot.item());
    assert!((dot.item() - 32.0).abs() < 1e-5);

    // 3) 非方阵 matmul — [1,3] × [3,2] → [1,2]
    println!("\n3. 非方阵 [1,3] × [3,2] → [1,2]:");
    let a = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0], [1, 3]).unwrap();
    let b = Tensor::<f32, Cpu>::from_data(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    println!("   result = [{:.0}, {:.0}]", c.data_slice()[0], c.data_slice()[1]);
    // [1*4+2*6+3*8=40, 1*5+2*7+3*9=46]
    assert!((c.data_slice()[0] - 40.0).abs() < 1e-5);
    assert!((c.data_slice()[1] - 46.0).abs() < 1e-5);

    // 4) 中等矩阵 — tiled_fma BM=BN=BK=128 + rayon 并行 + FMA 内循环
    println!("\n4. 中等矩阵 [256,256] → tiled_fma rayon + FMA:");
    let a = rand_tensor(&[256, 256], 42);
    let b = rand_tensor(&[256, 256], 43);
    let start = std::time::Instant::now();
    let c = a.matmul(&b).unwrap();
    let elapsed = start.elapsed();
    let flops = (256u64).pow(3) * 2; // m*n*k*2
    let gflops = flops as f64 / elapsed.as_secs_f64() / 1e9;
    println!("   耗时: {:?}", elapsed);
    println!("   估算: {:.2} GFLOPS", gflops);
    println!("   输出 shape={}, numel={}", c.shape(), c.numel());

    // 5) 大矩阵 [1024,1024] — strassen 7 公式递归 (CPU) 或 cblas_sgemm (BLAS feature)
    println!("\n5. 大矩阵 [1024,1024] → strassen / BLAS:");
    let a = rand_tensor(&[1024, 1024], 123);
    let b = rand_tensor(&[1024, 1024], 124);
    let start = std::time::Instant::now();
    let c = a.matmul(&b).unwrap();
    let elapsed = start.elapsed();
    let flops = (1024u64).pow(3) * 2;
    let gflops = flops as f64 / elapsed.as_secs_f64() / 1e9;
    println!("   耗时: {:?}", elapsed);
    println!("   估算: {:.2} GFLOPS", gflops);
    println!("   输出 shape={}, numel={}", c.shape(), c.numel());

    // 6) 单位矩阵验证 — I × A = A
    println!("\n6. 单位矩阵 I × A = A:");
    let eye = Tensor::<f32, Cpu>::from_data(
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3]
    ).unwrap();
    let a = Tensor::<f32, Cpu>::from_data(
        vec![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0], [3, 3]
    ).unwrap();
    let c = eye.matmul(&a).unwrap();
    assert_eq!(c.data_slice(), a.data_slice());
    println!("   ✅ I × A = A 验证通过");

    // 7) BLAS feature 验证（如果启用）
    #[cfg(feature = "blas")]
    {
        println!("\n7. BLAS feature 启用 — cblas_sgemm 真实路径:");
        let a = rand_tensor(&[512, 512], 99);
        let b = rand_tensor(&[512, 512], 100);
        let start = std::time::Instant::now();
        let c = a.matmul(&b).unwrap();
        let elapsed = start.elapsed();
        println!("   cblas_sgemm 路径耗时: {:?}", elapsed);
        println!("   输出 shape={}", c.shape());
    }

    println!("\n✅ 示例 02 完成");
}
