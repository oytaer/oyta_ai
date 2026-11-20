//! DType + Float + F16/BF16：从 oyta-device re-export
//!
//! §3.1 目录结构 → dtype.rs
//! DType/byte_size/from_str/to_str 全部在 oyta-device 里定义（因为 orphan rule）

// 从 oyta-device 重导出所有 dtype 相关类型
pub use oyta_device::{DType, Float, F16, BF16};
