//! checked.rs：整数乘积安全工具（防止 usize 隐式溢出）
//!
//! 所有对 shape 维度、stride、numel 的乘积都必须走这里。
//! 一旦溢出就返回 Err(TensorError::Overflow)，拒绝继续执行。

use crate::error::TensorError;

/// 两个 usize 的 checked 乘法
#[inline]
pub fn checked_mul(a: usize, b: usize) -> Result<usize, TensorError> {
    a.checked_mul(b).ok_or_else(|| TensorError::Overflow(format!("{} × {} 溢出 usize", a, b)))
}

/// 两个 usize 的 checked 加法
#[inline]
pub fn checked_add(a: usize, b: usize) -> Result<usize, TensorError> {
    a.checked_add(b).ok_or_else(|| TensorError::Overflow(format!("{} + {} 溢出 usize", a, b)))
}

/// 一个迭代器里所有 usize 的 checked 乘积
#[inline]
pub fn checked_product<I: Iterator<Item = usize>>(iter: I) -> Result<usize, TensorError> {
    let mut acc: usize = 1;
    for v in iter {
        acc = checked_mul(acc, v)?;
    }
    Ok(acc)
}

/// 便捷：三个数的 checked 乘积（batch_size * m * n 等常见场景）
#[inline]
pub fn checked_mul3(a: usize, b: usize, c: usize) -> Result<usize, TensorError> {
    checked_mul(checked_mul(a, b)?, c)
}

/// 便捷：4 个数的 checked 乘积
#[inline]
pub fn checked_mul4(a: usize, b: usize, c: usize, d: usize) -> Result<usize, TensorError> {
    checked_mul(checked_mul3(a, b, c)?, d)
}
