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

// ======== Conv1dOps — runtime probe + WGSL dispatch ========
#[cfg(feature = "wgpu")]
impl<T: crate::dtype::float_trait::Float> crate::backend::conv1d::Conv1dOps<T> for Wgpu {
    unsafe fn conv1d_fwd(
        x: *const T, x_dim: [usize; 3],
        w: *const T, w_dim: [usize; 3],
        b: *const T,
        out: *mut T, out_dim: [usize; 3],
        padding: usize, stride: usize, dilation: usize,
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && has_wgpu_adapter() {
            let has_bias = !b.is_null();
            let b_ptr = if b.is_null() { std::ptr::null::<f32>() } else { b as *const f32 };
            if gpu_conv1d_fwd(
                x as *const f32, x_dim, w as *const f32, w_dim,
                b_ptr, has_bias as u32,
                out as *mut f32, out_dim, padding, stride, dilation,
            ).is_ok() { return; }
        }
        // fallback：继承 trait 默认
        <Self as crate::backend::conv1d::Conv1dOps<T>>::conv1d_fwd(
            x, x_dim, w, w_dim, b, out, out_dim, padding, stride, dilation,
        );
    }
    unsafe fn conv1d_bwd_dx(
        dy: *const T, dy_dim: [usize; 3],
        w: *const T, w_dim: [usize; 3],
        dx: *mut T, dx_dim: [usize; 3],
        padding: usize, stride: usize, dilation: usize,
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && has_wgpu_adapter() {
            if gpu_conv1d_bwd_dx(
                dy as *const f32, dy_dim, w as *const f32, w_dim,
                dx as *mut f32, dx_dim, padding, stride, dilation,
            ).is_ok() { return; }
        }
        <Self as crate::backend::conv1d::Conv1dOps<T>>::conv1d_bwd_dx(
            dy, dy_dim, w, w_dim, dx, dx_dim, padding, stride, dilation,
        );
    }
    unsafe fn conv1d_bwd_dwdb(
        dy: *const T, dy_dim: [usize; 3],
        x: *const T, x_dim: [usize; 3],
        dw: *mut T, dw_dim: [usize; 3],
        db: *mut T, db_dim: [usize; 1],
        padding: usize, stride: usize, dilation: usize,
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && has_wgpu_adapter() {
            if gpu_conv1d_bwd_dwdb(
                dy as *const f32, dy_dim, x as *const f32, x_dim,
                dw as *mut f32, dw_dim, db as *mut f32, db_dim,
                padding, stride, dilation,
            ).is_ok() { return; }
        }
        <Self as crate::backend::conv1d::Conv1dOps<T>>::conv1d_bwd_dwdb(
            dy, dy_dim, x, x_dim, dw, dw_dim, db, db_dim, padding, stride, dilation,
        );
    }
}

// ======== BatchNormOps — runtime probe + WGSL dispatch ========
#[cfg(feature = "wgpu")]
impl<T: crate::dtype::float_trait::Float> crate::backend::batchnorm::BatchNormOps<T> for Wgpu {
    unsafe fn batchnorm1d_fwd(
        x: *const T, x_shape: &[usize],
        gamma: *const T, beta: *const T,
        running_mean: *mut T, running_var: *mut T,
        out: *mut T,
        eps: f32, momentum: f32, training: bool,
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && has_wgpu_adapter() {
            if gpu_bn_fwd(
                x as *const f32, x_shape, gamma as *const f32, beta as *const f32,
                running_mean as *mut f32, running_var as *mut f32, out as *mut f32,
                eps, momentum, training,
            ).is_ok() { return; }
        }
        <Self as crate::backend::batchnorm::BatchNormOps<T>>::batchnorm1d_fwd(
            x, x_shape, gamma, beta, running_mean, running_var, out, eps, momentum, training,
        );
    }
    unsafe fn batchnorm1d_bwd(
        dy: *const T, dy_shape: &[usize],
        x_norm: *const T,
        gamma: *const T,
        dgamma: *mut T, dbeta: *mut T,
        dx: *mut T, dx_shape: &[usize],
        eps: f32,
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && has_wgpu_adapter() {
            if gpu_bn_bwd(
                dy as *const f32, dy_shape, x_norm as *const f32,
                gamma as *const f32, dgamma as *mut f32, dbeta as *mut f32,
                dx as *mut f32, dx_shape, eps,
            ).is_ok() { return; }
        }
        <Self as crate::backend::batchnorm::BatchNormOps<T>>::batchnorm1d_bwd(
            dy, dy_shape, x_norm, gamma, dgamma, dbeta, dx, dx_shape, eps,
        );
    }
}

// ======== PoolOps — runtime probe + WGSL dispatch ========
#[cfg(feature = "wgpu")]
impl<T: crate::dtype::float_trait::Float> crate::backend::pool::PoolOps<T> for Wgpu {
    unsafe fn maxpool1d_fwd(
        x: *const T, x_shape: &[usize],
        out: *mut T, out_shape: &[usize],
        argmax_cache: *mut i64,
        kernel: usize, stride: usize, padding: usize,
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && has_wgpu_adapter() {
            if gpu_maxpool1d_fwd(
                x as *const f32, x_shape, out as *mut f32, out_shape,
                argmax_cache, kernel, stride, padding,
            ).is_ok() { return; }
        }
        <Self as crate::backend::pool::PoolOps<T>>::maxpool1d_fwd(
            x, x_shape, out, out_shape, argmax_cache, kernel, stride, padding,
        );
    }
    unsafe fn maxpool1d_bwd(
        dy: *const T, dy_shape: &[usize],
        argmax: *const i64,
        dx: *mut T, dx_shape: &[usize],
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && has_wgpu_adapter() {
            if gpu_maxpool1d_bwd(
                dy as *const f32, dy_shape, argmax, dx as *mut f32, dx_shape,
            ).is_ok() { return; }
        }
        <Self as crate::backend::pool::PoolOps<T>>::maxpool1d_bwd(
            dy, dy_shape, argmax, dx, dx_shape,
        );
    }
    unsafe fn avgpool1d_fwd(
        x: *const T, x_shape: &[usize],
        out: *mut T, out_shape: &[usize],
        kernel: usize, stride: usize, padding: usize,
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && has_wgpu_adapter() {
            if gpu_avgpool1d_fwd(
                x as *const f32, x_shape, out as *mut f32, out_shape,
                kernel, stride, padding,
            ).is_ok() { return; }
        }
        <Self as crate::backend::pool::PoolOps<T>>::avgpool1d_fwd(
            x, x_shape, out, out_shape, kernel, stride, padding,
        );
    }
    unsafe fn avgpool1d_bwd(
        dy: *const T, dy_shape: &[usize],
        dx: *mut T, dx_shape: &[usize],
        kernel: usize, stride: usize, padding: usize,
    ) {
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() && has_wgpu_adapter() {
            if gpu_avgpool1d_bwd(
                dy as *const f32, dy_shape, dx as *mut f32, dx_shape,
                kernel, stride, padding,
            ).is_ok() { return; }
        }
        <Self as crate::backend::pool::PoolOps<T>>::avgpool1d_bwd(
            dy, dy_shape, dx, dx_shape, kernel, stride, padding,
        );
    }
}

// ======== RnnOps — runtime probe + WGSL dispatch ========
#[cfg(feature = "wgpu")]
impl<T: crate::dtype::float_trait::Float> crate::backend::rnn::RnnOps<T> for Wgpu {
    unsafe fn gru_step(
        input: *const T, in_size: usize,
        h_prev: *const T, hidden_size: usize,
        h_new: *mut T,
        weights: *const T, bias: *const T,
    ) {
        // GRU/LSTM step 很小（通常 hidden_size < 1024），WGSL dispatch 开销 > 计算量
        // 保持继承 trait 默认（足够快）
        <Self as crate::backend::rnn::RnnOps<T>>::gru_step(
            input, in_size, h_prev, hidden_size, h_new, weights, bias,
        );
    }
    unsafe fn lstm_step(
        input: *const T, in_size: usize,
        h_prev: *const T, c_prev: *const T, hidden_size: usize,
        h_new: *mut T, c_new: *mut T,
        weights: *const T, bias: *const T,
    ) {
        <Self as crate::backend::rnn::RnnOps<T>>::lstm_step(
            input, in_size, h_prev, c_prev, hidden_size, h_new, c_new, weights, bias,
        );
    }
}

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
const CONV1D_WGSL: &str = include_str!("../wgsl/conv1d.wgsl");
const BATCHNORM_WGSL: &str = include_str!("../wgsl/batchnorm.wgsl");
const POOL1D_WGSL: &str = include_str!("../wgsl/pool1d.wgsl");

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

// ============================================================
// 通用 WGPU dispatch 辅助：shader → pipeline → bind group → dispatch → 读回
// ============================================================
#[cfg(feature = "wgpu")]
fn wgpu_dispatch_one_buffer_out(
    ctx: &WgpuContext,
    wgsl_source: &str,
    entry: &str,
    n_workgroups: u32,
    // bind group entries（顺序：@group(0) 所有 @binding）
    bg_entries: &[wgpu::BindGroupEntry],
    // output 信息：(output_buffer_index_in_bg, size_bytes, host_ptr)
    output_idx: usize,
    output_size: usize,
    output_host: *mut f32,
) -> Result<(), ()> {
    use wgpu::util::DeviceExt;

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_source)),
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bg_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(entry),
        layout: &bg_layout,
        entries: bg_entries,
    });
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(entry) });
    {
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(entry), timestamp_writes: None });
        cp.set_pipeline(&pipeline);
        cp.set_bind_group(0, &bind_group, &[]);
        cp.dispatch_workgroups(n_workgroups, 1, 1);
    }

    // 输出 buffer：从 bg_entries 里取（output_idx 指向第几个）
    let out_buf = match &bg_entries[output_idx].resource {
        wgpu::BindingResource::Buffer(b) => b.buffer.clone(),
        _ => return Err(()),
    };

    let out_size64 = output_size as u64;
    let host_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("host-read"), size: out_size64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&out_buf, 0, &host_out, 0, out_size64);
    let _ = ctx.queue.submit(Some(encoder.finish()));
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let (tx, rx) = flume::bounded(1);
    host_out.map_async(wgpu::MapMode::Read, 0..out_size64, move |r| { let _ = tx.send(r); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().map_err(|_| ())?.map_err(|_| ())?;
    let view = host_out.get_mapped_range(0..out_size64).map_err(|_| ())?;
    let result: &[f32] = bytemuck::cast_slice(&view);
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), output_host, output_size / 4); }
    drop(view); host_out.unmap();
    Ok(())
}

// ============================================================
// Pool1d WGPU dispatch
// ============================================================

#[cfg(feature = "wgpu")]
fn gpu_maxpool1d_fwd(
    x: *const f32, x_shape: &[usize],
    out: *mut f32, out_shape: &[usize],
    argmax: *mut i64,
    kernel: usize, stride: usize, padding: usize,
) -> Result<(), ()> {
    use wgpu::util::DeviceExt;
    let ctx = WgpuContext::try_get().ok_or(())?;
    let n=x_shape[0]; let c=x_shape[1]; let l_in=x_shape[2]; let l_out=out_shape[2];
    let nc = n*c; let total = nc*l_out;
    let x_slice = unsafe { std::slice::from_raw_parts(x, nc*l_in) };
    let out_size = total * 4;
    let argmax_size = total * 4; // WGSL 用 i32，host i64 写回时扩

    let gx = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mp-x"), contents: bytemuck::cast_slice(x_slice), usage: wgpu::BufferUsages::STORAGE,
    });
    let gout = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mp-out"), size: out_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let gargmax = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mp-am"), size: argmax_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let dims = [nc as u32, l_in as u32, l_out as u32, kernel as u32, stride as u32, padding as u32];
    let gdims = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mp-dims"), contents: bytemuck::cast_slice(&dims), usage: wgpu::BufferUsages::STORAGE,
    });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pool1d"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(POOL1D_WGSL)),
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mp-pipe"), layout: None, module: &module, entry_point: Some("maxpool_fwd"),
        compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
    });
    let bg_layout = pipeline.get_bind_group_layout(0);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mp-bg"), layout: &bg_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gout.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gargmax.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: gdims.as_entire_binding() },
        ],
    });

    let n_wg = ((total as u32) + 255) / 256;
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mp-enc") });
    { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("mp-pass"), timestamp_writes: None });
      cp.set_pipeline(&pipeline); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups(n_wg, 1, 1); }

    // 读 out + argmax
    let host_out = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("ho"), size: out_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let host_am = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("ham"), size: argmax_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_buffer_to_buffer(&gout, 0, &host_out, 0, out_size as u64);
    enc.copy_buffer_to_buffer(&gargmax, 0, &host_am, 0, argmax_size as u64);
    let _ = ctx.queue.submit(Some(enc.finish()));
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let (tx1, rx1) = flume::bounded(1);
    host_out.map_async(wgpu::MapMode::Read, 0..out_size as u64, move |r| { let _ = tx1.send(r); });
    let (tx2, rx2) = flume::bounded(1);
    host_am.map_async(wgpu::MapMode::Read, 0..argmax_size as u64, move |r| { let _ = tx2.send(r); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx1.recv().map_err(|_| ())?.map_err(|_| ())?;
    rx2.recv().map_err(|_| ())?.map_err(|_| ())?;

    let vo = host_out.get_mapped_range(0..out_size as u64).map_err(|_| ())?;
    let result: &[f32] = bytemuck::cast_slice(&vo);
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), out, total); }
    drop(vo); host_out.unmap();

    let vam = host_am.get_mapped_range(0..argmax_size as u64).map_err(|_| ())?;
    // WGSL 写 i32，host 要 i64 —— 逐元素转换
    let am_i32: &[i32] = bytemuck::cast_slice(&vam);
    unsafe {
        for i in 0..total {
            *argmax.add(i) = am_i32[i] as i64;
        }
    }
    drop(vam); host_am.unmap();
    Ok(())
}

#[cfg(feature = "wgpu")]
fn gpu_avgpool1d_fwd(
    x: *const f32, x_shape: &[usize],
    out: *mut f32, out_shape: &[usize],
    kernel: usize, stride: usize, padding: usize,
) -> Result<(), ()> {
    use wgpu::util::DeviceExt;
    let ctx = WgpuContext::try_get().ok_or(())?;
    let n=x_shape[0]; let c=x_shape[1]; let l_in=x_shape[2]; let l_out=out_shape[2];
    let nc=n*c; let total=nc*l_out;
    let x_slice = unsafe { std::slice::from_raw_parts(x, nc*l_in) };

    let gx = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ap-x"), contents: bytemuck::cast_slice(x_slice), usage: wgpu::BufferUsages::STORAGE,
    });
    let gout = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ap-out"), size: (total*4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let dims = [nc as u32, l_in as u32, l_out as u32, kernel as u32, stride as u32, padding as u32];
    let gdims = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ap-dims"), contents: bytemuck::cast_slice(&dims), usage: wgpu::BufferUsages::STORAGE,
    });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pool1d"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(POOL1D_WGSL)),
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ap-pipe"), layout: None, module: &module, entry_point: Some("avgpool_fwd"),
        compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ap-bg"), layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gout.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gdims.as_entire_binding() },
        ],
    });

    let n_wg = ((total as u32) + 255) / 256;
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ap-enc") });
    { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ap-pass"), timestamp_writes: None });
      cp.set_pipeline(&pipeline); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups(n_wg, 1, 1); }

    let host_out = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("ho2"), size: (total*4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_buffer_to_buffer(&gout, 0, &host_out, 0, (total*4) as u64);
    let _ = ctx.queue.submit(Some(enc.finish()));
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let (tx, rx) = flume::bounded(1);
    host_out.map_async(wgpu::MapMode::Read, 0..(total*4) as u64, move |r| { let _ = tx.send(r); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().map_err(|_| ())?.map_err(|_| ())?;
    let view = host_out.get_mapped_range(0..(total*4) as u64).map_err(|_| ())?;
    let result: &[f32] = bytemuck::cast_slice(&view);
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), out, total); }
    drop(view); host_out.unmap();
    Ok(())
}

// pool bwd 类似，但需要清零 dx buffer 再 dispatch（WGSL atomicAdd）
#[cfg(feature = "wgpu")]
fn gpu_maxpool1d_bwd(
    dy: *const f32, dy_shape: &[usize],
    argmax: *const i64,
    dx: *mut f32, dx_shape: &[usize],
) -> Result<(), ()> {
    use wgpu::util::DeviceExt;
    let ctx = WgpuContext::try_get().ok_or(())?;
    let n=dy_shape[0]; let c=dy_shape[1]; let l_out=dy_shape[2]; let l_in=dx_shape[2];
    let nc=n*c; let dy_total=nc*l_out; let dx_total=nc*l_in;

    let dy_slice = unsafe { std::slice::from_raw_parts(dy, dy_total) };
    // argmax 转 i32
    let am_i32: Vec<i32> = unsafe { std::slice::from_raw_parts(argmax, dy_total) }.iter().map(|&v| v as i32).collect();
    let zero_dx = vec![0.0f32; dx_total];

    let gdy = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mpb-dy"), contents: bytemuck::cast_slice(dy_slice), usage: wgpu::BufferUsages::STORAGE,
    });
    let gam = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mpb-am"), contents: bytemuck::cast_slice(&am_i32), usage: wgpu::BufferUsages::STORAGE,
    });
    let gdx = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mpb-dx"), contents: bytemuck::cast_slice(&zero_dx),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let dims = [nc as u32, l_out as u32, l_in as u32, 0u32, 0u32, 0u32];
    let gdims = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mpb-dims"), contents: bytemuck::cast_slice(&dims), usage: wgpu::BufferUsages::STORAGE,
    });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pool1d"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(POOL1D_WGSL)),
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mpb-pipe"), layout: None, module: &module, entry_point: Some("maxpool_bwd"),
        compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mpb-bg"), layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gdy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gam.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gdx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: gdims.as_entire_binding() },
        ],
    });

    let n_wg = ((dy_total as u32) + 255) / 256;
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mpb-enc") });
    { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("mpb-pass"), timestamp_writes: None });
      cp.set_pipeline(&pipeline); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups(n_wg, 1, 1); }

    let host_dx = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("hmb"), size: (dx_total*4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_buffer_to_buffer(&gdx, 0, &host_dx, 0, (dx_total*4) as u64);
    let _ = ctx.queue.submit(Some(enc.finish()));
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let (tx, rx) = flume::bounded(1);
    host_dx.map_async(wgpu::MapMode::Read, 0..(dx_total*4) as u64, move |r| { let _ = tx.send(r); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().map_err(|_| ())?.map_err(|_| ())?;
    let view = host_dx.get_mapped_range(0..(dx_total*4) as u64).map_err(|_| ())?;
    let result: &[f32] = bytemuck::cast_slice(&view);
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), dx, dx_total); }
    drop(view); host_dx.unmap();
    Ok(())
}

#[cfg(feature = "wgpu")]
fn gpu_avgpool1d_bwd(
    dy: *const f32, dy_shape: &[usize],
    dx: *mut f32, dx_shape: &[usize],
    kernel: usize, stride: usize, padding: usize,
) -> Result<(), ()> {
    use wgpu::util::DeviceExt;
    let ctx = WgpuContext::try_get().ok_or(())?;
    let n=dy_shape[0]; let c=dy_shape[1]; let l_out=dy_shape[2]; let l_in=dx_shape[2];
    let nc=n*c; let dy_total=nc*l_out; let dx_total=nc*l_in;

    let dy_slice = unsafe { std::slice::from_raw_parts(dy, dy_total) };
    let zero_dx = vec![0.0f32; dx_total];

    let gdy = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("apb-dy"), contents: bytemuck::cast_slice(dy_slice), usage: wgpu::BufferUsages::STORAGE,
    });
    let gdx = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("apb-dx"), contents: bytemuck::cast_slice(&zero_dx),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let dims = [nc as u32, l_out as u32, l_in as u32, kernel as u32, stride as u32, padding as u32];
    let gdims = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("apb-dims"), contents: bytemuck::cast_slice(&dims), usage: wgpu::BufferUsages::STORAGE,
    });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pool1d"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(POOL1D_WGSL)),
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("apb-pipe"), layout: None, module: &module, entry_point: Some("avgpool_bwd"),
        compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("apb-bg"), layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gdy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gdx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gdims.as_entire_binding() },
        ],
    });

    let n_wg = ((dy_total as u32) + 255) / 256;
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("apb-enc") });
    { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("apb-pass"), timestamp_writes: None });
      cp.set_pipeline(&pipeline); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups(n_wg, 1, 1); }

    let host_dx = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("hab"), size: (dx_total*4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_buffer_to_buffer(&gdx, 0, &host_dx, 0, (dx_total*4) as u64);
    let _ = ctx.queue.submit(Some(enc.finish()));
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let (tx, rx) = flume::bounded(1);
    host_dx.map_async(wgpu::MapMode::Read, 0..(dx_total*4) as u64, move |r| { let _ = tx.send(r); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().map_err(|_| ())?.map_err(|_| ())?;
    let view = host_dx.get_mapped_range(0..(dx_total*4) as u64).map_err(|_| ())?;
    let result: &[f32] = bytemuck::cast_slice(&view);
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), dx, dx_total); }
    drop(view); host_dx.unmap();
    Ok(())
}

// ============================================================
// Conv1d WGPU dispatch — conv1d.wgsl（3 个 entry point 共用一个 shader module）
// ============================================================
#[cfg(feature = "wgpu")]
fn gpu_conv1d_fwd(
    x: *const f32, x_dim: [usize; 3], w: *const f32, w_dim: [usize; 3],
    b: *const f32, has_bias: u32,
    out: *mut f32, out_dim: [usize; 3],
    padding: usize, stride: usize, dilation: usize,
) -> Result<(), ()> {
    use wgpu::util::DeviceExt;
    let ctx = WgpuContext::try_get().ok_or(())?;
    let n=x_dim[0]; let c_in=x_dim[1]; let l_in=x_dim[2]; let c_out=w_dim[0]; let k=w_dim[2]; let l_out=out_dim[2];
    let total=n*c_out*l_out;
    let x_slice = unsafe { std::slice::from_raw_parts(x, n*c_in*l_in) };
    let w_slice = unsafe { std::slice::from_raw_parts(w, c_out*c_in*k) };
    let out_size = total * 4;

    let gx = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("c1-x"), contents: bytemuck::cast_slice(x_slice), usage: wgpu::BufferUsages::STORAGE });
    let gw = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("c1-w"), contents: bytemuck::cast_slice(w_slice), usage: wgpu::BufferUsages::STORAGE });
    let gb = if has_bias != 0 {
        let b_slice = unsafe { std::slice::from_raw_parts(b, c_out) };
        ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("c1-b"), contents: bytemuck::cast_slice(b_slice), usage: wgpu::BufferUsages::STORAGE })
    } else { ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("c1-b"), size: (c_out*4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }) };
    let gout = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("c1-out"), size: out_size as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let pack = padding as u32 | ((stride as u32) << 8) | ((dilation as u32) << 16) | (has_bias << 24);
    let dims = [n as u32, c_in as u32, c_out as u32, l_in as u32, l_out as u32, k as u32, pack];
    let gdims = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("c1-d"), contents: bytemuck::cast_slice(&dims), usage: wgpu::BufferUsages::STORAGE });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("c1-mod"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(CONV1D_WGSL)) });
    let pipe = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("c1-pipe"), layout: None, module: &module, entry_point: Some("main"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("c1-bg"), layout: &pipe.get_bind_group_layout(0),
        entries: &[ wgpu::BindGroupEntry { binding: 0, resource: gx.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: gw.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: gb.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: gout.as_entire_binding() }, wgpu::BindGroupEntry { binding: 4, resource: gdims.as_entire_binding() } ] });

    let n_wg = ((total as u32) + 255) / 256;
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("c1-enc") });
    { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("c1-pass"), timestamp_writes: None }); cp.set_pipeline(&pipe); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups(n_wg, 1, 1); }

    let host_out = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("c1-ho"), size: out_size as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_buffer_to_buffer(&gout, 0, &host_out, 0, out_size as u64);
    let _ = ctx.queue.submit(Some(enc.finish())); let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let (tx, rx) = flume::bounded(1); host_out.map_async(wgpu::MapMode::Read, 0..out_size as u64, move |r| { let _ = tx.send(r); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely()); rx.recv().map_err(|_| ())?.map_err(|_| ())?;
    let view = host_out.get_mapped_range(0..out_size as u64).map_err(|_| ())?;
    let res: &[f32] = bytemuck::cast_slice(&view);
    unsafe { std::ptr::copy_nonoverlapping(res.as_ptr(), out, total); }
    drop(view); host_out.unmap(); Ok(())
}

#[cfg(feature = "wgpu")]
fn gpu_conv1d_bwd_dx(
    dy: *const f32, dy_dim: [usize; 3], w: *const f32, w_dim: [usize; 3],
    dx: *mut f32, dx_dim: [usize; 3],
    padding: usize, stride: usize, dilation: usize,
) -> Result<(), ()> {
    use wgpu::util::DeviceExt;
    let ctx = WgpuContext::try_get().ok_or(())?;
    let n=dy_dim[0]; let c_out=dy_dim[1]; let l_out=dy_dim[2]; let c_in=dx_dim[1]; let l_in=dx_dim[2]; let k=w_dim[2];
    let dy_total = n*c_out*l_out; let dx_total = n*c_in*l_in;
    let dy_slice = unsafe { std::slice::from_raw_parts(dy, dy_total) };
    let w_slice = unsafe { std::slice::from_raw_parts(w, c_out*c_in*k) };

    let gdy = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("c1b-dy"), contents: bytemuck::cast_slice(dy_slice), usage: wgpu::BufferUsages::STORAGE });
    let gw = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("c1b-w"), contents: bytemuck::cast_slice(w_slice), usage: wgpu::BufferUsages::STORAGE });
    let gdx = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("c1b-dx"), size: (dx_total*4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let pack = padding as u32 | ((stride as u32) << 8) | ((dilation as u32) << 16);
    let dims = [n as u32, c_in as u32, c_out as u32, l_in as u32, l_out as u32, k as u32, pack];
    let gdims = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("c1b-d"), contents: bytemuck::cast_slice(&dims), usage: wgpu::BufferUsages::STORAGE });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("c1-mod"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(CONV1D_WGSL)) });
    let pipe = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("c1b-pipe"), layout: None, module: &module, entry_point: Some("conv1d_bwd_dx"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("c1b-bg"), layout: &pipe.get_bind_group_layout(0),
        entries: &[ wgpu::BindGroupEntry { binding: 0, resource: gdy.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: gw.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: gdx.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: gdims.as_entire_binding() } ] });

    let n_wg = ((dx_total as u32) + 255) / 256;
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("c1b-enc") });
    { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("c1b-pass"), timestamp_writes: None }); cp.set_pipeline(&pipe); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups(n_wg, 1, 1); }

    let host_dx = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("c1b-hd"), size: (dx_total*4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_buffer_to_buffer(&gdx, 0, &host_dx, 0, (dx_total*4) as u64);
    let _ = ctx.queue.submit(Some(enc.finish())); let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let (tx, rx) = flume::bounded(1); host_dx.map_async(wgpu::MapMode::Read, 0..(dx_total*4) as u64, move |r| { let _ = tx.send(r); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely()); rx.recv().map_err(|_| ())?.map_err(|_| ())?;
    let view = host_dx.get_mapped_range(0..(dx_total*4) as u64).map_err(|_| ())?;
    let res: &[f32] = bytemuck::cast_slice(&view);
    unsafe { std::ptr::copy_nonoverlapping(res.as_ptr(), dx, dx_total); }
    drop(view); host_dx.unmap(); Ok(())
}

#[cfg(feature = "wgpu")]
fn gpu_conv1d_bwd_dwdb(
    dy: *const f32, dy_dim: [usize; 3], x: *const f32, x_dim: [usize; 3],
    dw: *mut f32, dw_dim: [usize; 3], db: *mut f32, db_dim: [usize; 1],
    padding: usize, stride: usize, dilation: usize,
) -> Result<(), ()> {
    use wgpu::util::DeviceExt;
    let ctx = WgpuContext::try_get().ok_or(())?;
    let n=dy_dim[0]; let c_out=dy_dim[1]; let l_out=dy_dim[2]; let c_in=x_dim[1]; let l_in=x_dim[2]; let k=dw_dim[2];
    let dy_total = n*c_out*l_out; let x_total = n*c_in*l_in; let dw_total = c_out*c_in*k; let db_total = c_out;
    let dy_slice = unsafe { std::slice::from_raw_parts(dy, dy_total) };
    let x_slice = unsafe { std::slice::from_raw_parts(x, x_total) };

    let gdy = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("c1w-dy"), contents: bytemuck::cast_slice(dy_slice), usage: wgpu::BufferUsages::STORAGE });
    let gx = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("c1w-x"), contents: bytemuck::cast_slice(x_slice), usage: wgpu::BufferUsages::STORAGE });
    let gdw = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("c1w-dw"), size: (dw_total*4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let gdb = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("c1w-db"), size: (db_total*4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let pack = padding as u32 | ((stride as u32) << 8) | ((dilation as u32) << 16);
    let dims = [n as u32, c_in as u32, c_out as u32, l_in as u32, l_out as u32, k as u32, pack];
    let gdims = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("c1w-d"), contents: bytemuck::cast_slice(&dims), usage: wgpu::BufferUsages::STORAGE });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("c1-mod"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(CONV1D_WGSL)) });
    let pipe = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("c1w-pipe"), layout: None, module: &module, entry_point: Some("conv1d_bwd_dwdb"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("c1w-bg"), layout: &pipe.get_bind_group_layout(0),
        entries: &[ wgpu::BindGroupEntry { binding: 0, resource: gdy.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: gx.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: gdw.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: gdb.as_entire_binding() }, wgpu::BindGroupEntry { binding: 4, resource: gdims.as_entire_binding() } ] });

    let n_wg = ((dw_total as u32) + 255) / 256;
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("c1w-enc") });
    { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("c1w-pass"), timestamp_writes: None }); cp.set_pipeline(&pipe); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups(n_wg, 1, 1); }

    let host_dw = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("c1w-hd"), size: (dw_total*4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let host_db = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("c1w-hb"), size: (db_total*4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_buffer_to_buffer(&gdw, 0, &host_dw, 0, (dw_total*4) as u64);
    enc.copy_buffer_to_buffer(&gdb, 0, &host_db, 0, (db_total*4) as u64);
    let _ = ctx.queue.submit(Some(enc.finish())); let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let (tx1, rx1) = flume::bounded(1); host_dw.map_async(wgpu::MapMode::Read, 0..(dw_total*4) as u64, move |r| { let _ = tx1.send(r); });
    let (tx2, rx2) = flume::bounded(1); host_db.map_async(wgpu::MapMode::Read, 0..(db_total*4) as u64, move |r| { let _ = tx2.send(r); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx1.recv().map_err(|_| ())?.map_err(|_| ())?;
    rx2.recv().map_err(|_| ())?.map_err(|_| ())?;
    let vdw = host_dw.get_mapped_range(0..(dw_total*4) as u64).map_err(|_| ())?;
    let vdb = host_db.get_mapped_range(0..(db_total*4) as u64).map_err(|_| ())?;
    let rdw: &[f32] = bytemuck::cast_slice(&vdw); let rdb: &[f32] = bytemuck::cast_slice(&vdb);
    unsafe { std::ptr::copy_nonoverlapping(rdw.as_ptr(), dw, dw_total); std::ptr::copy_nonoverlapping(rdb.as_ptr(), db, db_total); }
    drop(vdw); host_dw.unmap(); drop(vdb); host_db.unmap(); Ok(())
}

// ============================================================
// BatchNorm WGPU dispatch
// ============================================================
#[cfg(feature = "wgpu")]
fn gpu_bn_fwd(
    x: *const f32, x_shape: &[usize], gamma: *const f32, beta: *const f32,
    rm: *mut f32, rv: *mut f32, out: *mut f32,
    eps: f32, mom: f32, train: bool,
) -> Result<(), ()> {
    use wgpu::util::DeviceExt;
    let ctx = WgpuContext::try_get().ok_or(())?;
    let rank = x_shape.len();
    let feat_dim = x_shape[rank - 1];
    let outer: usize = x_shape[..rank - 1].iter().product();
    let total = outer * feat_dim;
    let x_slice = unsafe { std::slice::from_raw_parts(x, total) };
    let g_slice = unsafe { std::slice::from_raw_parts(gamma, feat_dim) };
    let b_slice = unsafe { std::slice::from_raw_parts(beta, feat_dim) };
    let eps_bits = eps.to_bits(); let mom_bits = mom.to_bits();

    let gx = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bn-x"), contents: bytemuck::cast_slice(x_slice), usage: wgpu::BufferUsages::STORAGE });
    let gy = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("bn-y"), size: (total*4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let grm = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bn-rm"), contents: bytemuck::cast_slice(unsafe { std::slice::from_raw_parts(rm, feat_dim) }), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC });
    let grv = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bn-rv"), contents: bytemuck::cast_slice(unsafe { std::slice::from_raw_parts(rv, feat_dim) }), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC });
    let ggamma = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bn-g"), contents: bytemuck::cast_slice(g_slice), usage: wgpu::BufferUsages::STORAGE });
    let gbeta = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bn-b"), contents: bytemuck::cast_slice(b_slice), usage: wgpu::BufferUsages::STORAGE });
    let dims = [outer as u32, feat_dim as u32, eps_bits, mom_bits];
    let gdims = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bn-d"), contents: bytemuck::cast_slice(&dims), usage: wgpu::BufferUsages::STORAGE });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("bn-mod"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BATCHNORM_WGSL)) });
    let entry = if train { "bn_fwd_train" } else { "bn_fwd_eval" };
    let pipe = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("bn-pipe"), layout: None, module: &module, entry_point: Some(entry), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None });
    // bind group layout 由 shader @group(N) 决定，train 用 group(0)，eval 用 group(1)
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("bn-bg"), layout: &pipe.get_bind_group_layout(0),
        entries: &[ wgpu::BindGroupEntry { binding: 0, resource: gx.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: gy.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: grm.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: grv.as_entire_binding() }, wgpu::BindGroupEntry { binding: 4, resource: ggamma.as_entire_binding() }, wgpu::BindGroupEntry { binding: 5, resource: gbeta.as_entire_binding() }, wgpu::BindGroupEntry { binding: 6, resource: gdims.as_entire_binding() } ] });

    let n_wg = if train { feat_dim as u32 } else { ((total as u32) + 255) / 256 };
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bn-enc") });
    { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bn-pass"), timestamp_writes: None }); cp.set_pipeline(&pipe); cp.set_bind_group(0, &bg, &[]); cp.dispatch_workgroups(n_wg, 1, 1); }

    let host_y = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("bn-hy"), size: (total*4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_buffer_to_buffer(&gy, 0, &host_y, 0, (total*4) as u64);
    if train {
        let host_rm = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("bn-hrm"), size: (feat_dim*4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let host_rv = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("bn-hrv"), size: (feat_dim*4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        enc.copy_buffer_to_buffer(&grm, 0, &host_rm, 0, (feat_dim*4) as u64);
        enc.copy_buffer_to_buffer(&grv, 0, &host_rv, 0, (feat_dim*4) as u64);
        let _ = ctx.queue.submit(Some(enc.finish())); let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

        let (t1, r1) = flume::bounded(1); host_y.map_async(wgpu::MapMode::Read, 0..(total*4) as u64, move |r| { let _ = t1.send(r); });
        let (t2, r2) = flume::bounded(1); host_rm.map_async(wgpu::MapMode::Read, 0..(feat_dim*4) as u64, move |r| { let _ = t2.send(r); });
        let (t3, r3) = flume::bounded(1); host_rv.map_async(wgpu::MapMode::Read, 0..(feat_dim*4) as u64, move |r| { let _ = t3.send(r); });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        r1.recv().map_err(|_| ())?.map_err(|_| ())?;
        r2.recv().map_err(|_| ())?.map_err(|_| ())?;
        r3.recv().map_err(|_| ())?.map_err(|_| ())?;
        let vy = host_y.get_mapped_range(0..(total*4) as u64).map_err(|_| ())?;
        let vrm = host_rm.get_mapped_range(0..(feat_dim*4) as u64).map_err(|_| ())?;
        let vrv = host_rv.get_mapped_range(0..(feat_dim*4) as u64).map_err(|_| ())?;
        let ry: &[f32] = bytemuck::cast_slice(&vy); let rrm: &[f32] = bytemuck::cast_slice(&vrm); let rrv: &[f32] = bytemuck::cast_slice(&vrv);
        unsafe { std::ptr::copy_nonoverlapping(ry.as_ptr(), out, total); std::ptr::copy_nonoverlapping(rrm.as_ptr(), rm, feat_dim); std::ptr::copy_nonoverlapping(rrv.as_ptr(), rv, feat_dim); }
        drop(vy); drop(vrm); drop(vrv); host_y.unmap(); host_rm.unmap(); host_rv.unmap();
    } else {
        let _ = ctx.queue.submit(Some(enc.finish())); let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let (tx, rx) = flume::bounded(1); host_y.map_async(wgpu::MapMode::Read, 0..(total*4) as u64, move |r| { let _ = tx.send(r); });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely()); rx.recv().map_err(|_| ())?.map_err(|_| ())?;
        let vy = host_y.get_mapped_range(0..(total*4) as u64).map_err(|_| ())?;
        let ry: &[f32] = bytemuck::cast_slice(&vy);
        unsafe { std::ptr::copy_nonoverlapping(ry.as_ptr(), out, total); }
        drop(vy); host_y.unmap();
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
fn gpu_bn_bwd(
    dy: *const f32, dy_shape: &[usize], xn: *const f32,
    gamma: *const f32, dg: *mut f32, _db: *mut f32,
    dx: *mut f32, dx_shape: &[usize], eps: f32,
) -> Result<(), ()> {
    // 简化版：BN bwd 在 batchnorm.wgsl 里用 group(2) bindings，
    // 但 batchnorm1d_bwd trait 签名里没有 x_norm_shape/dx_shape 在正确位置的对齐——
    // 这里返回 Err 触发 CPU fallback，batchnorm.wgsl 的 bn_bwd entry point 已存在
    // 若需 GPU 加速请调用 batchnorm1d_bwd trait 默认实现
    Err(())
}
