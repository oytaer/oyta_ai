//! bf16.rs：BF16 半精度浮点数
//!
//! 与 F16 相比：
//! - 和 f32 同指数范围（8 位指数），7 位尾数
//! - 范围与 f32 相同，精度比 F16 更低
//! - 常用于大模型训练节省显存

use std::fmt;
use super::float_trait::Float;

/// BF16 半精度浮点数（1 符号 + 8 指数 + 7 尾数，与 f32 同指数范围）
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct BF16(pub u16);

impl BF16 {
    // ======== 常量 ========
    /// +0.0
    pub const ZERO: Self = BF16(0x0000);
    /// +1.0
    pub const ONE: Self = BF16(0x3F80);
    /// +Infinity
    pub const INFINITY: Self = BF16(0x7F80);
    /// -Infinity
    pub const NEG_INFINITY: Self = BF16(0xFF80);
    /// NaN（quiet）
    pub const NAN: Self = BF16(0x7FC0);

    // ======== BF16 → f32 ========
    /// bf16 → f32：把 16 位左移 16 位拼成 32 位 float
    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    // ======== BF16 → f64 ========
    /// bf16 → f64：先转 f32 再提升为 f64
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    // ======== f32 → BF16（round-to-nearest-even）========
    /// f32 → bf16，round-to-nearest-even 四舍五入
    ///
    /// 截断 16 位尾数（f32 frac 23 位 → bf16 frac 7 位）。
    /// 直接 bits>>16 截断，然后 round-to-nearest-even 决定是否 +1。
    /// +1 如果造成尾数溢出会自然进位到指数，不需要特殊处理。
    ///
    /// # 注意
    /// debug 模式下要求输入是有限值（finite）；NaN/Inf 会触发 panic。
    /// release 模式下 NaN/Inf 输入会被正确转换为 BF16 对应的 NaN/Inf。
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        #[cfg(debug_assertions)]
        {
            debug_assert!(v.is_finite(), "BF16::from_f32 输入 {} 不是有限值", v);
        }
        let bits = v.to_bits();
        let exp = (bits >> 23) & 0xFF;
        let frac = bits & 0x7FFFFF;

        // NaN：保留 quiet NaN（尾数非零）
        if exp == 0xFF {
            let bf16_frac = if frac == 0 { 0 } else { 0x40 }; // quiet NaN: 尾数最高位置 1
            return BF16(((bits >> 16) & 0xFF00) as u16 | bf16_frac);
        }

        // 直接截断：bits >> 16 = [符号:1][指数:8][尾数:7] 恰好是 bf16 的 16 位布局
        let truncated = bits >> 16;

        // round-to-nearest-even：看被舍入的最高位和剩余位
        let round_bit = (frac >> 15) & 1;    // 被舍入的最高位（第 16 位）
        let remainder = frac & 0x7FFF;        // 被舍入的剩余 15 位
        let truncated_frac = truncated & 0x7F; // 截断后的 7 位尾数（判断奇偶）

        let rounded = if round_bit == 1 && (remainder != 0 || (truncated_frac & 1) != 0) {
            truncated + 1 // 进位，尾数满了自然进指数
        } else {
            truncated
        };

        BF16(rounded as u16)
    }

    // ======== f64 → BF16 ========
    /// f64 → bf16：先转 f32 再 from_f32
    #[inline]
    pub fn from_f64(v: f64) -> Self {
        BF16::from_f32(v as f32)
    }

    // ======== 数学属性 ========
    /// 判断是否 NaN
    #[inline]
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 7) & 0xFF;
        let frac = self.0 & 0x7F;
        exp == 0xFF && frac != 0
    }

    /// 判断是否无穷
    #[inline]
    pub fn is_infinite(self) -> bool {
        self.0 == 0x7F80 || self.0 == 0xFF80
    }

    /// 取绝对值
    #[inline]
    pub fn abs(self) -> Self { BF16(self.0 & 0x7FFF) }

    /// 取反
    #[inline]
    pub fn neg(self) -> Self { BF16(self.0 ^ 0x8000) }
}

impl Float for BF16 {
    #[inline] fn to_f32(self) -> f32 { self.to_f32() }
    #[inline] fn from_f32(v: f32) -> Self { BF16::from_f32(v) }
    #[inline] fn is_nan(self) -> bool { self.is_nan() }
    #[inline] fn is_infinite(self) -> bool { self.is_infinite() }
}

impl PartialOrd for BF16 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl fmt::Display for BF16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("{}", self.to_f32()))
    }
}
