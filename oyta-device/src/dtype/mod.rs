//! dtype 子域：数据类型定义

pub mod f16;
pub mod bf16;
pub mod float_trait;

pub use f16::F16;
pub use bf16::BF16;
pub use float_trait::{Float, Accumulate, DType, Precision};
