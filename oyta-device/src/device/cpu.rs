//! cpu.rs：Cpu 设备实现（默认 feature）
//!
//! Cpu 是空设备类型标记（零大小 struct），所有算子通过 Backend trait 默认实现完成。
//! 默认实现使用：rayon 并行 + 128³ Tiling + FMA + 动态算法选择器（naive/tiled/strassen）。
//! Backend::IS_CPU = true 让运行时决策器跳过不必要的融合判断。

use crate::*;
use std::marker::PhantomData;
use std::sync::OnceLock;

/// 空设备类型标记（零大小，仅 PhantomData 风格的类型标签）
#[derive(Copy, Clone, Debug, Default)]
pub struct Cpu;

// BackendBase：Backend bound 的第一个 trait（必须显式 impl，无默认方法）
impl BackendBase for Cpu {}

// ======== 泛型 blanket impl 一行覆盖所有 dtype ========
// T=Any Float → f32/F16/BF16/f64 全满足 Backend bound
use crate::dtype::float_trait::{Float, Accumulate};
impl<T: Float> crate::backend::elwise::ElwiseOps<T> for Cpu {}
impl<T: Float + Accumulate> crate::backend::reduce::ReduceOps<T> for Cpu {}
impl<T: Float> crate::backend::flash_attn::FlashAttentionOps<T> for Cpu {}
impl<T: Float + Accumulate> crate::backend::fused::FusedOps<T> for Cpu {}
impl<T: Float> crate::backend::inplace::InPlaceOps<T> for Cpu {}
// MatmulOps 无泛型参数（f32* 签名）
impl crate::backend::matmul::MatmulOps for Cpu {}

// Backend 聚合 impl（关联常量提供编译期设备属性）
// 上面 6 个 blanket impl + MatmulOps 已满足 Backend bound 的所有泛型实例
impl Backend for Cpu {
    const IS_CPU: bool = true;
    const MAX_ACCUMULATOR_PRECISION: Precision = Precision::F64;
    const DEVICE_NAME: &'static str = "CPU";
}

// DeviceContext for Cpu（OnceLock 全局单例）
impl DeviceContext for Cpu {
    fn init() -> Result<DeviceHandle<Cpu>, DeviceError> {
        static CPU_CTX: OnceLock<DeviceHandle<Cpu>> = OnceLock::new();
        Ok(CPU_CTX.get_or_init(|| DeviceHandle {
            properties: DeviceProperties {
                name: format!("CPU ({} cores)", rayon::current_num_threads()),
                device_id: 0,
                memory_bytes: 0,
                compute_capability: "Host".into(),
                num_cores: rayon::current_num_threads(),
                supports_fp16: false,
                supports_bf16: false,
                supports_fp8: false,
                supports_tensor_core: false,
            },
            _device: PhantomData,
            inner: DeviceInner::Cpu,
        }).clone())
    }

    fn get() -> DeviceHandle<Cpu> {
        Self::init().expect("CPU 上下文初始化失败")
    }

    fn shutdown() { /* Cpu 无外部资源，空操作 */ }
}

// DeviceHandle Clone（PhantomData 自动 Clone，DeviceProperties/DeviceInner 手动 derive）
impl<D: Backend> Clone for DeviceHandle<D> {
    fn clone(&self) -> Self {
        Self {
            properties: self.properties.clone(),
            _device: PhantomData,
            inner: self.inner.clone(),
        }
    }
}
