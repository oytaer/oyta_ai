//! f16.rs：FP16 半精度浮点数（IEEE 754 binary16）
//!
//! 完整实现：
//! - F16 → f32 / f64 位操作转换
//! - f32 / f64 → F16 round-to-nearest-even 四舍五入（LSB 判断）
//! - 常量 ZERO / ONE / MAX / INFINITY / NAN
//! - Display / PartialOrd / is_nan / is_infinite

use std::fmt;
use super::float_trait::Float;

/// FP16 半精度浮点数（IEEE 754 binary16，1 符号 + 5 指数 + 10 尾数）
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct F16(pub u16);

impl F16 {
    // ======== 常量 ========
    /// +0.0
    pub const ZERO: Self = F16(0x0000);
    /// +1.0
    pub const ONE: Self = F16(0x3C00);
    /// +Infinity
    pub const INFINITY: Self = F16(0x7C00);
    /// -Infinity
    pub const NEG_INFINITY: Self = F16(0xFC00);
    /// NaN（quiet，最高尾数位置 = 1）
    pub const NAN: Self = F16(0x7E00);
    /// 最大有限正数：65504.0
    pub const MAX: Self = F16(0x7BFF);
    /// 最小规格化正数：2^(-14) ≈ 6.1035e-5
    pub const MIN_POSITIVE: Self = F16(0x0400);

    // ======== F16 → f32 ========
    /// f16 → f32：把 16 位包装成 32 位 float（指数加 112 偏移）
    #[inline]
    pub fn to_f32(self) -> f32 {
        let u = self.0 as u32;
        let sign = (u >> 15) & 1;           // 1 位符号
        let exp = (u >> 10) & 0x1F;         // 5 位指数
        let frac = u & 0x3FF;               // 10 位尾数
        let f_bits = (sign << 31)           // f32 符号位
            | ((exp + 112) << 23)           // f32 指数偏移 = 127-15 = 112
            | (frac << 13);                 // f32 尾数对齐（32-23=9？不，frac 10 位，左移 13 对齐到 23 位）
        f32::from_bits(f_bits)
    }

    // ======== F16 → f64 ========
    /// f16 → f64：先转 f32 再提升为 f64（保持精度）
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    // ======== f32 → F16（round-to-nearest-even）========
    /// f32 → f16，IEEE 754 round-to-nearest-even 四舍五入
    ///
    /// 截断 13 位尾数（f32 frac 23 位 → f16 frac 10 位），
    /// 舍入规则：被舍入位最高位 = 1 且 (剩余位非零 或 新尾数 LSB = 1) 时 +1。
    ///
    /// # 注意
    /// debug 模式下要求输入是有限值（finite）；NaN/Inf 会触发 panic。
    /// release 模式下 NaN/Inf 输入会被正确转换为 F16 对应的 NaN/Inf。
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        #[cfg(debug_assertions)]
        {
            // 上游算子（matmul/softmax/layernorm）正常输出应该都是 finite；
            // 遇到 NaN/Inf 通常意味着上游 bug，debug 下尽早暴露。
            debug_assert!(v.is_finite(), "F16::from_f32 输入 {} 不是有限值", v);
        }
        let bits = v.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let frac = bits & 0x7FFFFF;

        // NaN 原样保留 quiet NaN
        if exp == 0xFF {
            let f16_frac = if frac == 0 { 0 } else { 0x200 }; // quiet NaN
            return F16(((sign << 15) | 0x7C00 | f16_frac) as u16);
        }
        // 零（含 subnormal 太小）
        if exp == 0 {
            // subnormal 处理：f32 exp=0 已经是非规格化，f16 也只能是 0
            return F16((sign << 15) as u16);
        }

        let new_exp = exp - 127 + 15; // 指数偏移：f32 127 → f16 15，差 112
        // subnormal f16（new_exp <= 0 时尾数隐式 1 要挪出来）
        if new_exp <= 0 {
            if new_exp <= -14 { return F16((sign << 15) as u16); } // 太小直接下溢到 0
            // subnormal: frac 前加隐含 1，然后右移到把最高位放到 frac 的对应位置
            let implied_frac = frac | 0x800000; // 加 1.xxx 的 1
            let shift = (-new_exp) + 1; // 需要右移位数
            let truncated = implied_frac >> (13 + shift);
            return F16(((sign << 15) | truncated) as u16);
        }
        // 太大溢出到 inf
        if new_exp >= 0x1F {
            return F16(((sign << 15) | 0x7C00) as u16);
        }

        // 正常规格化数：截断 13 位尾数 + round-to-nearest-even
        let truncated_frac = frac >> 13;           // 截断后的 10 位尾数
        let round_bit = (frac >> 12) & 1;          // 被舍入的最高位（第 13 位）
        let remainder = frac & 0xFFF;              // 被舍入的剩余 12 位

        // round-to-nearest-even 规则：
        // - round_bit = 0 → 直接截断
        // - round_bit = 1 且 remainder > 0 → +1（严格大于 0.5）
        // - round_bit = 1 且 remainder = 0 → 恰好 0.5：看 truncated_frac 最低位，奇数 +1，偶数不变
        let rounded_frac = if round_bit == 1 && (remainder != 0 || (truncated_frac & 1) != 0) {
            truncated_frac + 1
        } else {
            truncated_frac
        };

        // 尾数加 1 可能溢出（如 0x3FF + 1 = 0x400），需要进一位指数
        if rounded_frac >= 0x400 {
            // 尾数满了，进指数
            let final_exp = (new_exp as u32) + 1;
            if final_exp >= 0x1F {
                return F16(((sign << 15) | 0x7C00) as u16); // 溢出到 inf
            }
            return F16(((sign << 15) | (final_exp << 10) | 0) as u16);
        }

        F16(((sign << 15) | ((new_exp as u32) << 10) | rounded_frac) as u16)
    }

    // ======== f64 → F16 ========
    /// f64 → f16：先转 f32（保持精确）再 from_f32
    #[inline]
    pub fn from_f64(v: f64) -> Self {
        F16::from_f32(v as f32)
    }

    // ======== 数学属性 ========
    /// 判断是否 NaN
    #[inline]
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let frac = self.0 & 0x3FF;
        exp == 0x1F && frac != 0
    }

    /// 判断是否无穷
    #[inline]
    pub fn is_infinite(self) -> bool {
        self.0 == 0x7C00 || self.0 == 0xFC00
    }

    /// 取绝对值
    #[inline]
    pub fn abs(self) -> Self { F16(self.0 & 0x7FFF) }

    /// 取反
    #[inline]
    pub fn neg(self) -> Self { F16(self.0 ^ 0x8000) }
}

impl Float for F16 {
    #[inline] fn to_f32(self) -> f32 { self.to_f32() }
    #[inline] fn from_f32(v: f32) -> Self { F16::from_f32(v) }
    #[inline] fn is_nan(self) -> bool { self.is_nan() }
    #[inline] fn is_infinite(self) -> bool { self.is_infinite() }
}

impl PartialOrd for F16 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl fmt::Display for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("{}", self.to_f32()))
    }
}
