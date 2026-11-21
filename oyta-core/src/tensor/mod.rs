//! tensor 子域：Tensor<T, D> 核心数据结构 + 所有 impl 块

pub mod base;
pub mod shape;
pub mod data;
pub mod ops;
pub mod tensor_ml;
pub mod reduce;
pub mod factory;
pub mod dtype;
pub mod device;
#[cfg(test)]
pub mod tests;

pub use base::Tensor;
