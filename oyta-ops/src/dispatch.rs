//! dispatch：算子调度器（§7.4）
//!
//! 简化实现：核心只做策略选择，contiguous 检查委托 Tensor.contiguous()
//! 泛型 Backend 约束在具体模块（elwise/reduce/matmul）里解决

use oyta_core::{Tensor, DynamicShape};
use oyta_device::Backend;

// ============================================================
// matmul 算法选择（§24.3）
// ============================================================

/// matmul 策略
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatmulStrategy { Naive, TiledFma, Strassen, GpuTensorCore }

/// 动态选择 matmul 算法
pub fn select_matmul_strategy<D: Backend>(m: usize, k: usize, n: usize) -> MatmulStrategy {
    let max_dim = m.max(k).max(n);
    if max_dim < 32 { return MatmulStrategy::Naive; }
    if max_dim < 128 { return MatmulStrategy::TiledFma; }
    if D::HAS_FP16_TENSOR_CORE || D::HAS_BF16 || !D::IS_CPU {
        MatmulStrategy::GpuTensorCore
    } else {
        MatmulStrategy::Strassen
    }
}

/// reduce 策略
#[derive(Copy, Clone, Debug)]
pub enum ReduceStrategy { Rayon, Sequential }

pub fn select_reduce_strategy(numel: usize) -> ReduceStrategy {
    if numel < 1024 { ReduceStrategy::Sequential } else { ReduceStrategy::Rayon }
}

/// 广播兼容检查
pub fn broadcast_target_shape<D: Backend>(
    a: &Tensor<f32, D>, b: &Tensor<f32, D>,
) -> Option<DynamicShape> {
    if a.shape.broadcast_compatible(&b.shape) {
        Some(a.shape.broadcast_with(&b.shape))
    } else {
        None
    }
}
