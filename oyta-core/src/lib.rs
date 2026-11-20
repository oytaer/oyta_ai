//! oyta-core：Tensor 核心数据结构
//!
//! 目录结构：
//!   ├── lib.rs          crate 文档 + 模块声明 + re-export
//!   ├── tensor/         Tensor 子域（base/shape/data/ops/reduce/factory/dtype/device/tests）
//!   ├── shape.rs        DynamicShape（运行时形状 + -1 reshape + 广播 + From 实现）
//!   ├── strides.rs      Strides（步长 + contiguous 检查 + transpose/permute）
//!   ├── index.rs        IndexEngine（多维→线性偏移转换，广播+非contiguous通用路径）
//!   ├── dtype.rs        从 oyta-device re-export
//!   ├── device.rs       Device trait + Cpu/Cuda/Metal/Wgpu 类型标记
//!   ├── error.rs        TensorError（thiserror，7 种变体 + Result 别名）
//!   ├── alloc.rs        AlignedAlloc（对齐分配器包装）
//!   ├── bytes.rs        bytes 级序列化
//!   └── safetensors.rs  HuggingFace safetensors 文件格式读写

// ============================================================
// 模块声明
// ============================================================
pub mod shape;          // DynamicShape（dim/-1 reshape/广播）
pub mod strides;        // Strides（contiguous/transpose/permute）
pub mod index;          // IndexEngine（多维→线性偏移）
pub mod dtype;          // DType + Float trait + F16/BF16 重导出
pub mod device;         // Device trait + Cpu/Cuda/Metal/Wgpu（feature gate）
pub mod error;          // TensorError（thiserror 10 种变体）
pub mod tensor;         // Tensor<T, D>（子目录聚合：base/shape/data/ops/reduce/factory/dtype/device）
pub mod alloc;          // 对齐分配器包装（预留）
pub mod bytes;          // bytes 级序列化
pub mod safetensors;    // HuggingFace safetensors 文件格式读写
pub mod checked;        // 整数乘积安全工具（checked_mul / checked_product）

// ============================================================
// 便捷重导出（外部用 oyta_core::Tensor 而不是 oyta_core::tensor::Tensor）
// ============================================================
pub use shape::DynamicShape;
pub use strides::Strides;
pub use index::IndexEngine;
pub use device::Device;
pub use error::{TensorError, Result};
pub use tensor::Tensor;
pub use safetensors::{save_safetensors, load_safetensors, SafetensorsHeader, TensorHeader};
pub use bytes::{
    f32_to_bytes, bytes_to_f32, f64_to_bytes, bytes_to_f64,
    f16_to_bytes, bytes_to_f16, bf16_to_bytes, bytes_to_bf16,
    tensor_byte_size, write_bytes, read_bytes,
};
pub use dtype::{DType, Float, F16, BF16};
pub use alloc::AlignedAlloc;
