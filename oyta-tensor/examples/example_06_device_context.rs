//! example_06_device_context.rs — DeviceContext OnceLock 单例 + GPU fallback
//!
//! 展示：Device trait / DeviceContext / OnceLock 单例 / DeviceProperties / GPU fallback

use oyta_tensor::*;

fn main() {
    println!("========================================");
    println!("  example_06: DeviceContext + GPU fallback");
    println!("========================================");

    // 1) Backend trait — 编译期类型标记
    println!("\n1. Backend trait 编译期标记:");
    use oyta_device::Backend;
    println!("   DEVICE_NAME = {}", <Cpu as Backend>::DEVICE_NAME);
    println!("   IS_CPU = {}", <Cpu as Backend>::IS_CPU);

    // 2) DeviceContext OnceLock 单例 — Cpu
    println!("\n2. DeviceContext OnceLock 单例 (Cpu):");
    let ctx = Cpu::get();
    println!("   DeviceProperties:");
    println!("     name={}", ctx.properties.name);
    println!("     device_id={}", ctx.properties.device_id);
    println!("     compute_capability={}", ctx.properties.compute_capability);
    println!("     num_cores={}", ctx.properties.num_cores);
    println!("     supports_fp16={}", ctx.properties.supports_fp16);
    println!("     supports_bf16={}", ctx.properties.supports_bf16);

    // 3) 再次 get() — 返回同一个 OnceLock 引用
    let ctx2 = Cpu::get();
    println!("\n3. OnceLock 单例验证 (Cpu):");
    println!("   ctx properties.name = {}", ctx.properties.name);
    println!("   ctx2 properties.name = {}", ctx2.properties.name);
    // 验证 OnceLock 单例 — get() 返回同一个内部值引用
    // ctx 和 ctx2 指向 DeviceContext<Cpu> 里的数据成员应该一致
    assert_eq!(ctx.properties.name, ctx2.properties.name);
    assert_eq!(ctx.properties.num_cores, ctx2.properties.num_cores);
    println!("   ✅ OnceLock 单例内容验证通过（name/num_cores 一致）");

    // 4) Backend trait 入口：Backend 所有方法直接通过 Cpu 暴露
    println!("\n4. Backend trait 入口 — MatmulOps:");
    use oyta_device::MatmulOps;
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0_f32, 6.0, 7.0, 8.0];
    let mut c = [0.0_f32; 4];
    <Cpu as MatmulOps>::matmul(
        a.as_ptr(), 2, b.as_ptr(), 2, c.as_mut_ptr(), 2,
        2, 2, 2, 1.0, 0.0,
    );
    let result = c;
    println!("   Cpu::matmul 直接调用 = {:?}", result);

    // 5) GPU feature 启用时的真实 fallback（硬件不存在时走 CPU 路径）
    println!("\n5. GPU 后端 feature-gated + CPU fallback:");

    #[cfg(feature = "cuda")]
    {
        println!("   [cfg(feature=cuda)] Cuda 设备可用 — 真实 Backend 实现（CPU fallback）");
        use oyta_device::Backend;
        println!("   Cuda IS_CPU={}", <Cuda as Backend>::IS_CPU);
        // Cuda::matmul 真实可工作（因为 Backend 默认 impl 继承 rayon 并行）
        let a = Tensor::<f32, Cuda>::ones([32, 32]);
        let b = Tensor::<f32, Cuda>::ones([32, 32]);
        let _c = a.matmul(&b).unwrap();
        println!("   ✅ Cuda feature enable: matmul CPU fallback 路径工作正常");
    }

    #[cfg(feature = "wgpu")]
    {
        println!("   [cfg(feature=wgpu)] Wgpu 设备可用 — 真实 Backend 实现（CPU fallback）");
        use oyta_device::Backend;
        println!("   Wgpu IS_CPU={}", <Wgpu as Backend>::IS_CPU);
        let a = Tensor::<f32, Wgpu>::ones([32, 32]);
        let b = Tensor::<f32, Wgpu>::ones([32, 32]);
        let _c = a.matmul(&b).unwrap();
        println!("   ✅ Wgpu feature enable: matmul CPU fallback 路径工作正常");
    }

    #[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "metal")))]
    {
        println!("   GPU feature 未启用 — Cuda/Metal/Wgpu 类型不可用（编译期裁剪）");
    }

    // 6) set_runtime_device / get_runtime_device — AnyBackend 运行时切换
    println!("\n6. AnyBackend 运行时动态设备切换:");
    println!("   当前 runtime device: {:?}", get_runtime_device());
    // set_runtime_device(DeviceRuntimeId::Cuda);  // 如果 feature 启用且硬件存在

    println!("\n✅ 示例 06 完成");
}
