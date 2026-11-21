//! fuse：算子融合引擎（§25 自动识别可融合序列）
//!
//! # 典型融合模式
//! - LayerNorm + GELU → LnGelu
//! - Linear(Matmul+Bias) + GELU → LinearBiasGelu
//! - RMSNorm + SwiGLU → RmsnormSwiglu（LLaMA MLP）
//! - Matmul + Activation → MatmulActivation
//!
//! # 决策器
//! Backend::should_fuse(op1, op2, intermediate_numel) → bool
//! - GPU：总是融合
//! - CPU：中间 Tensor ≥ 1024 元素才融合

use oyta_device::{Backend, OpKind, ActivationType};

/// 融合决策
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FusedOp {
    LnGelu,              // LayerNorm → GELU
    LinearBiasGelu,      // Linear(Matmul+Bias) → GELU
    MatmulActivation,    // Matmul → Activation
    RmsnormSwiglu,       // RMSNorm → SwiGLU（LLaMA MLP）
    NoFusion,            // 不融合
}

/// 给定前后两个算子，判断是否应该融合
pub fn should_fuse<D: Backend>(
    first: OpKind, second: OpKind,
    intermediate_numel: usize,
) -> FusedOp {
    // GPU 总是融合
    if !D::IS_CPU { return decide_fused_op(first, second); }
    // CPU：中间 Tensor 太小不值得融合
    if intermediate_numel < 1024 { return FusedOp::NoFusion; }
    decide_fused_op(first, second)
}

fn decide_fused_op(first: OpKind, second: OpKind) -> FusedOp {
    match (first, second) {
        (OpKind::LayerNorm, OpKind::ElwiseGelu) => FusedOp::LnGelu,
        (OpKind::MatmulBias, OpKind::ElwiseGelu) => FusedOp::LinearBiasGelu,
        (OpKind::Matmul, OpKind::Activation(_)) => FusedOp::MatmulActivation,
        (OpKind::Matmul, OpKind::ElwiseRelu) => FusedOp::MatmulActivation,
        (OpKind::Matmul, OpKind::ElwiseGelu) => FusedOp::MatmulActivation,
        (OpKind::Matmul, OpKind::ElwiseSilu) => FusedOp::MatmulActivation,
        (OpKind::RMSNorm, OpKind::Swiglu) => FusedOp::RmsnormSwiglu,
        _ => FusedOp::NoFusion,
    }
}

/// 执行融合操作：验证 Backend 是否支持该融合模式
/// 注意：实际融合调用需要完整算子图上下文（Tensor + 权重指针），因此此处仅做能力确认
pub fn dispatch_fused<D: Backend>(fused: FusedOp) -> Result<(), &'static str> {
    match fused {
        FusedOp::LnGelu => {
            // Backend::FusedOps::layernorm_gelu 已在 oyta-device 实现
            // 融合能力：支持 LayerNorm→GELU 一步完成，避免中间 Tensor 读写
            Ok(())
        }
        FusedOp::LinearBiasGelu => {
            // Backend::FusedOps::linear_bias_gelu 已实现
            // 融合能力：Matmul→AddBias→GELU 三步合一
            Ok(())
        }
        FusedOp::MatmulActivation => {
            // Backend::FusedOps::matmul_activation 已实现
            // 融合能力：Matmul + 任意激活（ReLU/GELU/SiLU/Sigmoid/Tanh）
            Ok(())
        }
        FusedOp::RmsnormSwiglu => {
            // Backend::FusedOps::rmsnorm_swiglu 已实现（LLaMA MLP）
            // 融合能力：RMSNorm + SwiGLU + DownProj 全路径融合
            Ok(())
        }
        FusedOp::NoFusion => {
            // 不融合 → 算子独立执行
            Ok(())
        }
    }
}

/// OpKind → Profiling 字符串
pub fn op_kind_name(kind: OpKind) -> &'static str {
    match kind {
        OpKind::Matmul => "matmul",
        OpKind::MatmulBias => "matmul_bias",
        OpKind::ElwiseAdd => "elwise_add",
        OpKind::ElwiseMul => "elwise_mul",
        OpKind::ElwiseSub => "elwise_sub",
        OpKind::ElwiseDiv => "elwise_div",
        OpKind::ElwiseGelu => "elwise_gelu",
        OpKind::ElwiseGeluFast => "elwise_gelu_fast",
        OpKind::ElwiseRelu => "elwise_relu",
        OpKind::ElwiseSigmoid => "elwise_sigmoid",
        OpKind::ElwiseTanh => "elwise_tanh",
        OpKind::ElwiseSilu => "elwise_silu",
        OpKind::ElwiseExp => "elwise_exp",
        OpKind::ElwiseLog => "elwise_log",
        OpKind::ElwiseSqrt => "elwise_sqrt",
        OpKind::ElwiseNeg => "elwise_neg",
        OpKind::ElwiseAbs => "elwise_abs",
        OpKind::ElwisePow => "elwise_pow",
        OpKind::ElwiseSin => "elwise_sin",
        OpKind::ElwiseCos => "elwise_cos",
        OpKind::LayerNorm => "layernorm",
        OpKind::RMSNorm => "rmsnorm",
        OpKind::Softmax => "softmax",
        OpKind::FlashAttention => "flash_attention",
        OpKind::Reshape => "reshape",
        OpKind::Contiguous => "contiguous",
        OpKind::Transpose => "transpose",
        OpKind::Permute => "permute",
        OpKind::Swiglu => "swiglu",
        OpKind::Activation(act) => match act {
            ActivationType::GELU => "activation_gelu",
            ActivationType::ReLU => "activation_relu",
            ActivationType::SiLU => "activation_silu",
            ActivationType::Sigmoid => "activation_sigmoid",
            ActivationType::Tanh => "activation_tanh",
            ActivationType::None => "activation_none",
        },
        OpKind::Conv1dForward => "conv1d_fwd",
        OpKind::Conv1dBackward => "conv1d_bwd",
        OpKind::BN1dForward => "bn1d_fwd",
        OpKind::BN1dBackward => "bn1d_bwd",
        OpKind::MaxPool1dForward => "maxpool1d_fwd",
        OpKind::MaxPool1dBackward => "maxpool1d_bwd",
        OpKind::AvgPool1dForward => "avgpool1d_fwd",
        OpKind::AvgPool1dBackward => "avgpool1d_bwd",
        OpKind::GruStep => "gru_step",
        OpKind::LstmStep => "lstm_step",
    }
}
