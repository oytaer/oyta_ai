//! wgpu.rs：Wgpu 跨平台后端（#[cfg(feature="wgpu")]）
//!
//! 核心设计：
//! 1. `WgpuContext` OnceLock 全局单例 — pollster::block_on 初始化 Instance → Adapter → Device + Queue
//! 2. 有 wgpu adapter → 真实 GPU matmul（WGSL compute shader）
//! 3. 无 adapter → fallback `<Self as MatmulOps>::default_matmul(...)`（CPU rayon）
//!
//! wgpu v30 + edition 2024。

#[cfg(feature = "wgpu")]
use crate::backend::*;
#[cfg(feature = "wgpu")]
use crate::backend::aggregate::{Backend, DeviceContext, DeviceHandle, DeviceProperties, DeviceInner, DeviceError};
#[cfg(feature = "wgpu")]
use crate::dtype::Precision;
#[cfg(feature = "wgpu")]
use std::marker::PhantomData;
#[cfg(feature = "wgpu")]
use std::sync::OnceLock;

// ============================================================
// 设备标记
// ============================================================

/// Wgpu 设备标记
#[cfg(feature = "wgpu")]
#[derive(Copy, Clone, Debug, Default)]
pub struct Wgpu;

// ======== BackendBase + 5 个 blanket impl ========
#[cfg(feature = "wgpu")]
impl BackendBase for Wgpu {}
#[cfg(feature = "wgpu")]
impl<T: crate::dtype::float_trait::Float> crate::backend::elwise::ElwiseOps<T> for Wgpu {}
#[cfg(feature = "wgpu")]
impl<T: crate::dtype::float_trait::Float + crate::dtype::float_trait::Accumulate> crate::backend::reduce::ReduceOps<T> for Wgpu {}
#[cfg(feature = "wgpu")]
impl<T: crate::dtype::float_trait::Float> crate::backend::flash_attn::FlashAttentionOps<T> for Wgpu {}
#[cfg(feature = "wgpu")]
impl<T: crate::dtype::float_trait::Float + crate::dtype::float_trait::Accumulate> crate::backend::fused::FusedOps<T> for Wgpu {}
#[cfg(feature = "wgpu")]
impl<T: crate::dtype::float_trait::Float> crate::backend::inplace::InPlaceOps<T> for Wgpu {}

// 运行时 probe：检测 GPU adapter
#[cfg(feature = "wgpu")]
fn has_wgpu_adapter() -> bool {
    static EXISTS: OnceLock<bool> = OnceLock::new();
    *EXISTS.get_or_init(|| {
        if std::path::Path::new("/dev/dxg").exists() { return true; }
        if std::path::Path::new("/dev/dri/card0").exists() { return true; }
        false
    })
}

// Backend impl（空 impl，自动继承所有泛型 trait 默认实现）
// MatmulOps 有 GPU 重写，其余全部继承泛型默认
#[cfg(feature = "wgpu")]
impl Backend for Wgpu {
    const IS_CPU: bool = true;
    const MAX_ACCUMULATOR_PRECISION: Precision = Precision::F32;
    const HAS_FP16_TENSOR_CORE: bool = false;
    const HAS_BF16: bool = false;
    const HAS_BATCHED_GEMM: bool = false;
    const HAS_FLASH_ATTENTION_HW: bool = false;
    const DEVICE_NAME: &'static str = "WGPU";
}

// DeviceContext for Wgpu
#[cfg(feature = "wgpu")]
impl DeviceContext for Wgpu {
    fn init() -> Result<DeviceHandle<Wgpu>, DeviceError> {
        static WGPU_CTX: OnceLock<DeviceHandle<Wgpu>> = OnceLock::new();
        Ok(WGPU_CTX.get_or_init(|| {
            let has_adapter = has_wgpu_adapter();
            let (name, compute_cap, fp16, bf16, tensor_core) = if has_adapter {
                (
                    "WGPU (GPU detected)".into(),
                    "wgpu-core".into(),
                    true, true, true,
                )
            } else {
                (
                    "WGPU (CPU fallback - 无适配器)".into(),
                    "CPU fallback".into(),
                    false, false, false,
                )
            };
            DeviceHandle {
                properties: DeviceProperties {
                    name,
                    device_id: 0,
                    memory_bytes: 0,
                    compute_capability: compute_cap,
                    num_cores: rayon::current_num_threads(),
                    supports_fp16: fp16,
                    supports_bf16: bf16,
                    supports_fp8: false,
                    supports_tensor_core: tensor_core,
                },
                _device: PhantomData,
                inner: DeviceInner::Wgpu { device_id: 0 },
            }
        }).clone())
    }

    fn get() -> DeviceHandle<Wgpu> {
        Self::init().expect("WGPU 上下文初始化失败")
    }

    fn shutdown() { /* 空操作 */ }
}

// ============================================================
// WgpuContext：Device + Queue 全局单例
// ============================================================

#[cfg(feature = "wgpu")]
struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[cfg(feature = "wgpu")]
impl WgpuContext {
    fn try_get() -> Option<&'static WgpuContext> {
        static CTX: OnceLock<Option<WgpuContext>> = OnceLock::new();
        let slot = CTX.get_or_init(|| {
            if !has_wgpu_adapter() {
                return None;
            }
            // pollster::block_on 初始化 wgpu instance → adapter → device + queue
            pollster::block_on(async {
                let instance = wgpu::Instance::default();
                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions::default())
                    .await
                    .ok()?;
                let (device, queue) = adapter
                    .request_device(&wgpu::DeviceDescriptor::default())
                    .await
                    .ok()?;
                Some(WgpuContext { device, queue })
            })
        });
        slot.as_ref()
    }
}

// ============================================================
// GPU matmul：WGSL compute shader
// ============================================================

/// WGSL shader source（行主序 matmul，16×16 workgroup，WSL2 DX GPU 兼容）
#[cfg(feature = "wgpu")]
const MATMUL_WGSL: &str = include_str!("../wgsl/matmul.wgsl");

/// Wgpu 实现 MatmulOps：有 adapter 走 GPU compute shader，否则 fallback CPU
#[cfg(feature = "wgpu")]
impl MatmulOps for Wgpu {
    fn matmul(
        a: *const f32, lda: usize,
        b: *const f32, ldb: usize,
        c: *mut f32, ldc: usize,
        m: usize, n: usize, k: usize,
        alpha: f32, beta: f32,
    ) {
        // 快速路径：alpha=1, beta=0, GPU 可用
        if alpha == 1.0 && beta == 0.0 {
            if let Some(ctx) = WgpuContext::try_get() {
                if let Ok(()) = gpu_matmul_impl(ctx, a, lda, b, ldb, c, ldc, m, n, k) {
                    return;
                }
            }
        }
        // 通用 fallback：CPU
        <Self as MatmulOps>::default_matmul(a, lda, b, ldb, c, ldc, m, n, k, alpha, beta);
    }
}

/// GPU matmul 实现
///
/// 流程：
/// 1. 整理 contiguous（如果 lda/ldb/ldc 不匹配 k/n/n，先 copy）
/// 2. 创建 GPU 端的 input/output storage buffers
/// 3. 创建 dims uniform buffer
/// 4. 创建 ComputePipeline + BindGroup
/// 5. 拷贝 A/B → dispatch matmul → 拷贝 C 回 host
/// 6. device.poll wait + buffer.map_async 读回
#[cfg(feature = "wgpu")]
fn gpu_matmul_impl(
    ctx: &WgpuContext,
    a_ptr: *const f32, lda: usize,
    b_ptr: *const f32, ldb: usize,
    c_ptr: *mut f32, ldc: usize,
    m: usize, n: usize, k: usize,
) -> Result<(), ()> {
    use wgpu::util::DeviceExt;

    // === 整理 contiguous ===
    let a_len = crate::checked::checked_mul(m, lda);
    let b_len = crate::checked::checked_mul(k, ldb);
    let c_len = crate::checked::checked_mul(m, ldc);
    let a_in = unsafe { std::slice::from_raw_parts(a_ptr, a_len) };
    let b_in = unsafe { std::slice::from_raw_parts(b_ptr, b_len) };
    let c_out = unsafe { std::slice::from_raw_parts_mut(c_ptr, c_len) };

    let a_contig: Vec<f32> = if lda == k {
        a_in[..m * k].to_vec()
    } else {
        let mut v = vec![0.0f32; m * k];
        for i in 0..m {
            v[i * k..(i + 1) * k].copy_from_slice(&a_in[i * lda..i * lda + k]);
        }
        v
    };
    let b_contig: Vec<f32> = if ldb == n {
        b_in[..k * n].to_vec()
    } else {
        let mut v = vec![0.0f32; k * n];
        for i in 0..k {
            v[i * n..(i + 1) * n].copy_from_slice(&b_in[i * ldb..i * ldb + n]);
        }
        v
    };

    let out_size = m * n;
    if out_size == 0 {
        return Ok(());
    }

    let c_size = out_size * std::mem::size_of::<f32>();

    // === 创建 GPU buffer ===
    let gpu_a = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matmul-gpu-a"),
        contents: bytemuck::cast_slice(&a_contig),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let gpu_b = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matmul-gpu-b"),
        contents: bytemuck::cast_slice(&b_contig),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // C buffer：STORAGE | COPY_SRC | COPY_DST
    let gpu_c = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul-gpu-c"),
        size: c_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // dims storage buffer
    let dims = [m as u32, n as u32, k as u32, 0u32];
    let gpu_dims = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matmul-gpu-dims"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // host-side read buffer：MAP_READ | COPY_DST
    let host_c = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul-host-c"),
        size: c_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // === 创建 shader module ===
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul-wgsl"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MATMUL_WGSL)),
    });

    // === Compute pipeline ===
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matmul-pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // === Bind group layout & bind group ===
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gpu_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gpu_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gpu_c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: gpu_dims.as_entire_binding() },
        ],
    });

    // === Command encoder ===
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul-encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        let block = 16u32;  // 16×16 = 256 invocations（WSL2 DX GPU 限制 ≤ 256）
        let work_x = (n as u32 + block - 1) / block;
        let work_y = (m as u32 + block - 1) / block;
        cpass.dispatch_workgroups(work_x, work_y, 1);
    }

    // 拷贝结果回 host buffer
    encoder.copy_buffer_to_buffer(&gpu_c, 0, &host_c, 0, c_size as u64);

    // === Submit ===
    let _submit_idx = ctx.queue.submit(Some(encoder.finish()));

    // === Wait + map read ===
    // 先 poll 确保 GPU 完成
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    // map 读回 host buffer
    let (tx, rx) = flume::bounded(1);
    host_c.map_async(wgpu::MapMode::Read, 0..c_size as u64, move |result| {
        let _ = tx.send(result);
    });
    // poll 等 map callback
    loop {
        let _ = ctx.device.poll(wgpu::PollType::Poll);
        if let Ok(res) = rx.try_recv() {
            res.map_err(|_| ())?;
            break;
        }
        std::thread::yield_now();
    }

    // 读回数据
    let view = host_c.get_mapped_range(0..c_size as u64).map_err(|_| ())?;
    let result: &[f32] = bytemuck::cast_slice(&view);

    // 写回 c（处理 ldc != n）
    if ldc == n {
        c_out[..out_size].copy_from_slice(result);
    } else {
        for i in 0..m {
            c_out[i * ldc..i * ldc + n].copy_from_slice(&result[i * n..(i + 1) * n]);
        }
    }
    drop(view);
    host_c.unmap();

    Ok(())
}
