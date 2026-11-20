//! checked.rs：整数安全工具（防止 usize 隐式溢出）
//!
//! 后端层（oyta-device）的 checked_mul / checked_mul3 工具。
//! oyta-core 也有同名模块，两边代码基本一致但不共享依赖（避免循环依赖）。
//!
//! 溢出策略：saturating（饱和截断到 usize::MAX）
//! 比 panic 好：一旦上游传了超大 shape，saturating 保证分配一个合法（虽然巨大）Vec，
//! 不会静默 wrapping 导致小 buffer + 写越界（最危险）。
//! 用户侧（oyta-core）应该返回 Err(TensorError::Overflow)。

/// 两个 usize 的 checked 乘法，溢出 saturating 到 usize::MAX
#[inline]
pub fn checked_mul(a: usize, b: usize) -> usize {
    a.checked_mul(b).unwrap_or(usize::MAX)
}

/// 两个 usize 的 checked 加法，溢出 saturating 到 usize::MAX
#[inline]
pub fn checked_add(a: usize, b: usize) -> usize {
    a.checked_add(b).unwrap_or(usize::MAX)
}

/// 三个 usize 的 checked 乘积，溢出 saturating
#[inline]
pub fn checked_mul3(a: usize, b: usize, c: usize) -> usize {
    checked_mul(checked_mul(a, b), c)
}

/// 迭代器中所有 usize 的 checked 乘积，溢出 saturating 到 usize::MAX
#[inline]
pub fn checked_product<I: Iterator<Item = usize>>(iter: I) -> usize {
    let mut acc: usize = 1;
    for v in iter {
        acc = checked_mul(acc, v);
    }
    acc
}

/// 便捷：从迭代器算 checked product（别名）
#[inline]
pub fn checked_mul3_from_iter<I: Iterator<Item = usize>>(iter: I) -> usize {
    checked_product(iter)
}
