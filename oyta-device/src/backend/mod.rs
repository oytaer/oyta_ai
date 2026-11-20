//! backend 子域：Backend 算子域

pub mod base;
pub mod elwise;
pub mod reduce;
pub mod matmul;
pub mod flash_attn;
pub mod fused;
pub mod inplace;
pub mod aggregate;
pub mod any_backend;

pub use base::BackendBase;
pub use elwise::{ElwiseOps, error_function};
pub use reduce::ReduceOps;
pub use matmul::MatmulOps;
pub use flash_attn::FlashAttentionOps;
pub use fused::FusedOps;
pub use inplace::InPlaceOps;
pub use aggregate::{Backend, OpKind, ActivationType, DeviceError, DeviceRuntimeId};
pub use any_backend::AnyBackend;
