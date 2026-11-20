//! float_trait.rs：Float trait + Accumulate + DType + Precision

use super::{f16::F16, bf16::BF16};

// ============================================================
// Accumulate：累加器精度 trait
// ============================================================

/// 累加器精度 trait（f32→f64、f16→f32、bf16→f32）
/// 泛型 backend 用它自动在 reduce/matmul/softmax 里提升精度
pub trait Accumulate: Copy + Clone {
    type Accum: Copy + Clone;
    /// 把自身值转成累加器类型（用于进入高精度计算）
    fn to_accum(self) -> Self::Accum;
    /// 把累加器值转回自身类型（用于输出）
    fn from_accum(a: Self::Accum) -> Self;
}

impl Accumulate for f32 {
    type Accum = f64;
    #[inline] fn to_accum(self) -> f64 { self as f64 }
    #[inline] fn from_accum(a: f64) -> Self { a as f32 }
}
impl Accumulate for f64 {
    type Accum = f64;
    #[inline] fn to_accum(self) -> f64 { self }
    #[inline] fn from_accum(a: f64) -> Self { a }
}
impl Accumulate for F16 {
    type Accum = f32;
    #[inline] fn to_accum(self) -> f32 { self.to_f32() }
    #[inline] fn from_accum(a: f32) -> Self { F16::from_f32(a) }
}
impl Accumulate for BF16 {
    type Accum = f32;
    #[inline] fn to_accum(self) -> f32 { self.to_f32() }
    #[inline] fn from_accum(a: f32) -> Self { BF16::from_f32(a) }
}

// ============================================================
// Float：通用浮点 trait
// ============================================================

/// 通用浮点 trait（简化版，替代 num_traits::Float）
/// 加 Send + Sync 让默认实现能用 rayon par_iter
pub trait Float: Clone + Copy + Default + PartialEq + PartialOrd + std::fmt::Debug + Send + Sync {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
    fn zero() -> Self { Self::from_f32(0.0) }
    fn one() -> Self { Self::from_f32(1.0) }
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
}

impl Float for f32 {
    #[inline] fn to_f32(self) -> f32 { self }
    #[inline] fn from_f32(v: f32) -> Self { v }
    #[inline] fn is_nan(self) -> bool { self.is_nan() }
    #[inline] fn is_infinite(self) -> bool { self.is_infinite() }
}
impl Float for f64 {
    #[inline] fn to_f32(self) -> f32 { self as f32 }
    #[inline] fn from_f32(v: f32) -> Self { v as f64 }
    #[inline] fn is_nan(self) -> bool { self.is_nan() }
    #[inline] fn is_infinite(self) -> bool { self.is_infinite() }
}

// F16/BF16 的 Float impl 在各自文件里

// ============================================================
// DType：safetensors dtype 名称
// ============================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DType { F32, F64, F16, BF16 }
impl DType {
    /// safetensors 字符串
    pub fn safetensors_name(self) -> &'static str {
        match self { DType::F32 => "F32", DType::F64 => "F64", DType::F16 => "F16", DType::BF16 => "BF16" }
    }
    /// 每个元素字节数（safetensors 计算偏移）
    pub fn byte_size(self) -> usize {
        match self { DType::F32 => 4, DType::F64 => 8, DType::F16 | DType::BF16 => 2 }
    }
    /// 从 safetensors 字符串解析
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s { "F32" => Ok(DType::F32), "F64" => Ok(DType::F64),
                  "F16" => Ok(DType::F16), "BF16" => Ok(DType::BF16),
                  other => Err(format!("未知 dtype: {}", other)) }
    }
}
impl From<f32> for DType { fn from(_: f32) -> Self { DType::F32 } }
impl From<f64> for DType { fn from(_: f64) -> Self { DType::F64 } }
impl From<F16> for DType { fn from(_: F16) -> Self { DType::F16 } }
impl From<BF16> for DType { fn from(_: BF16) -> Self { DType::BF16 } }

// ============================================================
// Precision：累加器精度枚举
// ============================================================

/// 累加器精度等级
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Precision { F32, F64 }
