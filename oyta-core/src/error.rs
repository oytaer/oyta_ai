//! TensorError：Tensor 层错误类型
//!
//! §3.1 目录结构 → error.rs
//! §14.1 完整错误定义（7 种变体覆盖所有场景）
//!
//! 用 thiserror 派生 std::error::Error，统一 Tensor 层错误处理

use thiserror::Error; // 错误派生宏

/// Tensor 操作错误
#[derive(Debug, Clone, Error)]
pub enum TensorError {
    /// 形状不匹配（二元算子输入 shape 不一致）
    #[error("形状不匹配: 期望 {expected:?}, 实际 {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    /// 矩阵乘法维度错误
    #[error("matmul 维度不匹配: A[{m},{k1}] × B[{k2},{n}], k1≠k2")]
    MatMulDimMismatch { m: usize, k1: usize, k2: usize, n: usize },

    /// 无法广播（两个 shape 广播不兼容）
    #[error("广播失败: {shape_a:?} 和 {shape_b:?} 不兼容")]
    BroadcastFailed { shape_a: Vec<usize>, shape_b: Vec<usize> },

    /// reshape 不可能（元素总数不匹配）
    #[error("reshape 不可能: 原 shape {old:?}, 新 shape {new:?}, 元素总数不一致")]
    ReshapeImpossible { old: Vec<usize>, new: Vec<usize> },

    /// 数据量不匹配（from_data 元素数 ≠ shape.numel()）
    #[error("数据量不匹配: 需要 {need} 个元素, 实际 {have} 个")]
    DataSizeMismatch { need: usize, have: usize },

    /// 索引越界（多维索引超出 shape 范围）
    #[error("索引越界: shape {shape:?}, 索引 {index:?}")]
    IndexOutOfBounds { shape: Vec<usize>, index: Vec<usize> },

    /// Tensor 非 contiguous，操作需要 contiguous
    #[error("Tensor 不 contiguous，操作需要 contiguous")]
    NotContiguous,

    /// shape 含零元素
    #[error("shape 含零元素: {0:?}")]
    ZeroShape(Vec<usize>),

    /// 不支持的操作
    #[error("不支持的操作: {0}")]
    Unsupported(String),

    /// 设备层错误（GPU 分配失败等不可恢复错误）
    #[error("设备错误: {msg}")]
    DeviceError { msg: String },

    /// 整数溢出（shape.numel/stride/乘积等溢出 usize）
    #[error("整数溢出: {0}")]
    Overflow(String),
}

/// 便捷 Result 类型（全局 Result 别名）
pub type Result<T> = std::result::Result<T, TensorError>;
