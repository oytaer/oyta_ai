//! backend_trait.rs：Backend 聚合 trait + Device trait + OpKind/ActivationType

use std::marker::PhantomData;

use crate::dtype::{Precision, F16, BF16};
use crate::backend::base::BackendBase;
use crate::backend::elwise::ElwiseOps;
use crate::backend::reduce::ReduceOps;
use crate::backend::matmul::MatmulOps;
use crate::backend::flash_attn::FlashAttentionOps;
use crate::backend::fused::FusedOps;
use crate::backend::inplace::InPlaceOps;

// ============================================================
// broadcast_strides：从 shape 计算 contiguous strides
// ============================================================

/// 从 shape 数组计算 contiguous strides（右对齐广播友好）
/// shape=[2,3,4] → strides=[12,4,1]
///
/// # 溢出处理
/// 用 checked_mul；步长乘积溢出 usize 时 panic（shape 来自内部，
/// 一旦溢出就是上游 bug；从 safetensors 等外部源来的 shape 在解析阶段已校验）
pub fn broadcast_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![0usize; rank];
    let mut s = 1usize;
    for i in (0..rank).rev() {
        strides[i] = s;
        s = s.checked_mul(shape[i]).unwrap_or(usize::MAX);
    }
    strides
}

// ============================================================
// Backend：唯一的设备 trait（聚合所有子 trait + 关联常量）
// ============================================================

/// Backend：唯一的设备抽象 trait
/// 泛型 trait 自动覆盖 f32/f16/bf16/f64 全部 dtype
/// 每个 Device impl Backend {} 就能继承所有 dtype 的完整默认实现
pub trait Backend:
    BackendBase +
    ElwiseOps<f32> + ElwiseOps<F16> + ElwiseOps<BF16> + ElwiseOps<f64> +
    ReduceOps<f32> + ReduceOps<F16> + ReduceOps<BF16> +
    FlashAttentionOps<f32> + FlashAttentionOps<F16> + FlashAttentionOps<BF16> +
    MatmulOps +
    FusedOps<f32> + FusedOps<F16> + FusedOps<BF16> +
    InPlaceOps<f32> + InPlaceOps<F16> + InPlaceOps<BF16> +
    Copy + Clone + Send + Sync + 'static
{
    // ============================================================
    // 编译期设备属性（associated const）
    // 单态化后条件分支消除，零运行时开销
    // ============================================================
    
    /// 是否为 CPU 设备
    const IS_CPU: bool = false;
    
    /// 最大累加器精度
    const MAX_ACCUMULATOR_PRECISION: Precision = Precision::F32;
    
    /// 是否支持 FP16 Tensor Core / AMX
    const HAS_FP16_TENSOR_CORE: bool = false;
    
    /// 是否支持 BF16
    const HAS_BF16: bool = false;
    
    /// 是否硬件加速 batched matmul
    const HAS_BATCHED_GEMM: bool = false;
    
    /// 是否硬件加速 FlashAttention
    const HAS_FLASH_ATTENTION_HW: bool = false;
    
    /// 设备名（日志打印）
    const DEVICE_NAME: &'static str = "Unknown";
    
    // ============================================================
    // 运行时上下文初始化/清理（Cuda 重写，Cpu 默认空）
    // ============================================================
    
    fn init_runtime() -> Result<(), DeviceError> { Ok(()) }
    fn shutdown_runtime() {}
    
    // ============================================================
    // 融合引擎决策（§25：GPU 总触发，CPU 看中间 tensor 大小）
    // ============================================================
    
    /// 是否应该把 op1 + op2 融合成一个 kernel
    fn should_fuse(_op1: OpKind, _op2: OpKind, intermediate_elements: usize) -> bool {
        !Self::IS_CPU || intermediate_elements >= 1024
    }
}

// ============================================================
// OpKind / ActivationType / DeviceError
// ============================================================

/// 算子种类枚举（融合决策用）
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OpKind {
    ElwiseAdd, ElwiseSub, ElwiseMul, ElwiseDiv,
    ElwiseRelu, ElwiseGelu, ElwiseGeluFast, ElwiseSilu, ElwiseSigmoid, ElwiseTanh,
    ElwiseNeg, ElwiseAbs, ElwiseExp, ElwiseLog, ElwiseSqrt,
    ElwisePow, ElwiseSin, ElwiseCos,
    Softmax, LayerNorm, RMSNorm,
    Matmul, MatmulBias, FlashAttention,
    /// 激活函数（参数化，融合决策用）
    Activation(ActivationType),
    /// SwiGLU（LLaMA MLP 后半部分）
    Swiglu,
    /// 形状变换
    Reshape, Contiguous, Transpose, Permute,
}

/// 激活函数类型
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ActivationType { None, ReLU, GELU, SiLU, Sigmoid, Tanh }

/// 设备错误
#[derive(Debug, Clone, thiserror::Error)]
pub enum DeviceError {
    #[error("设备初始化失败: {0}")]
    InitFailed(String),
    #[error("CUDA 错误: {0}")]
    CudaError(String),
    #[error("Metal 错误: {0}")]
    MetalError(String),
    #[error("不支持的操作: {0}")]
    Unsupported(String),
}

// ============================================================
// 设备类型 re-export（模块声明在 lib.rs）
// ============================================================

pub use crate::device::cpu::Cpu;
#[cfg(feature = "cuda")]
pub use crate::device::cuda::Cuda;
#[cfg(feature = "metal")]
pub use crate::device::metal::Metal;
#[cfg(feature = "wgpu")]
pub use crate::device::wgpu::Wgpu;

// ============================================================
// DeviceRuntimeId + set_runtime_device / get_runtime_device
// ============================================================

use std::sync::atomic::{AtomicU8, Ordering};

/// 运行时设备 ID（全局）
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeviceRuntimeId {
    Cpu = 0,
    Cuda = 1,
    Metal = 2,
    Wgpu = 3,
}

/// 全局当前设备
static CURRENT_RUNTIME: AtomicU8 = AtomicU8::new(0); // 默认 Cpu

/// 运行时切换设备（在张量创建前调用）
pub fn set_runtime_device(id: DeviceRuntimeId) {
    CURRENT_RUNTIME.store(id as u8, Ordering::Relaxed);
}

/// 获取当前运行时设备
pub fn get_runtime_device() -> DeviceRuntimeId {
    match CURRENT_RUNTIME.load(Ordering::Relaxed) {
        0 => DeviceRuntimeId::Cpu,
        #[cfg(feature = "cuda")]
        1 => DeviceRuntimeId::Cuda,
        #[cfg(feature = "metal")]
        2 => DeviceRuntimeId::Metal,
        #[cfg(feature = "wgpu")]
        3 => DeviceRuntimeId::Wgpu,
        _ => DeviceRuntimeId::Cpu,
    }
}

// ============================================================
// DeviceContext / DeviceHandle / DeviceProperties / DeviceInner
// ============================================================

/// 通用设备属性（所有设备都有）
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,               // "CPU (16 cores)" / "NVIDIA H100" / ...
    pub device_id: usize,           // 多 GPU 场景的设备索引
    pub memory_bytes: usize,        // 显存大小（CPU 为 0）
    pub compute_capability: String, // "Host" / "SM_90" / "Apple M3" / ...
    pub num_cores: usize,           // SM 数量 / CPU 核心数
    pub supports_fp16: bool,        // 硬件是否原生支持 f16
    pub supports_bf16: bool,        // 硬件是否原生支持 bf16
    pub supports_fp8: bool,         // 硬件是否原生支持 f8
    pub supports_tensor_core: bool, // 是否有 Tensor Core / AMX
}

/// 统一的设备运行时句柄
pub struct DeviceHandle<D: Backend> {
    /// 设备属性（只读，初始化后不变）
    pub properties: DeviceProperties,
    /// 设备类型标记（零大小，不占内存）
    pub _device: PhantomData<D>,
    /// 具体设备的私有数据（feature gate 控制）
    pub(crate) inner: DeviceInner,
}

/// DeviceInner：每个设备可能的私有数据（用枚举 + feature gate）
#[derive(Default, Clone)]
pub enum DeviceInner {
    #[default]
    /// Cpu：不持有任何外部资源
    Cpu,
    #[cfg(feature = "cuda")]
    /// Cuda：cublas handle + stream（原始句柄）
    Cuda {
        cublas_handle: u64,
        cudnn_handle: Option<u64>,
        stream: u64,
        device_id: usize,
    },
    #[cfg(feature = "metal")]
    /// Metal：MPS 设备
    Metal { device_id: usize },
    #[cfg(feature = "wgpu")]
    /// Wgpu：跨平台
    Wgpu { device_id: usize },
}

/// 设备运行时上下文 trait（每个 Backend 实现它）
pub trait DeviceContext: Backend {
    /// 初始化设备上下文（全局单例，只调一次）
    fn init() -> Result<DeviceHandle<Self>, DeviceError>;
    
    /// 获取设备上下文（如果未初始化会自动 init，不会 panic）
    fn get() -> DeviceHandle<Self>;
    
    /// 显式关闭 GPU 资源
    fn shutdown() {}
    
    /// 可选：初始化时指定设备 ID（多 GPU 场景）
    fn init_with_id(device_id: usize) -> Result<DeviceHandle<Self>, DeviceError> {
        let _ = device_id;
        Self::init()  // 默认忽略，具体设备重写
    }
}
