//! Strides：Tensor 步长（支持零拷贝 view）
//!
//! §3.1 目录结构 → strides.rs
//! 步长 = 某维度移动一个元素时，底层 flat 数组需要跳过的元素数
//!
//! 完整实现：§4.4 Strides 结构（contiguous 检查、transpose、permute）

use crate::shape::DynamicShape; // 需要 DynamicShape 做 is_contiguous 检查

/// 步长向量（和 shape.dims 一一对应）
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Strides(pub Vec<usize>); // 元组结构体，内部是各维度步长

impl Strides {
    /// 从 Vec 构造
    pub fn new(dims: impl Into<Vec<usize>>) -> Self { Self(dims.into()) }

    /// 从 shape 生成 C-contiguous 步长
    pub fn contiguous(shape: &DynamicShape) -> Self {
        Self(shape.contiguous_strides()) // shape 的 contiguous_strides() 返回 Vec<usize>
    }

    /// 转只读切片（给外部比较/迭代）
    pub fn as_slice(&self) -> &[usize] { &self.0 }

    /// 是否 C-contiguous（最后一维 stride=1，且逐层 dim 乘积匹配）
    ///
    /// 遍历 strides，和 shape.dims 逐层乘积对比（从右向左）
    pub fn is_contiguous(&self, shape: &DynamicShape) -> bool {
        if self.0.len() != shape.dims.len() { return false; } // rank 不匹配
        let mut expected = 1usize;
        for i in (0..shape.dims.len()).rev() { // 从右向左验证
            if self.0[i] != expected { return false; }
            expected *= shape.dims[i]; // 更新期望值
        }
        true
    }

    /// 转置两个维度（交换 strides[i] 和 strides[j]）
    ///
    /// 只交换步长，不改变 shape 或底层数据
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        let mut s = self.0.clone();
        s.swap(dim1, dim2);
        Self(s)
    }

    /// permute：按新顺序重排所有 strides
    ///
    /// order[i] = 原维度索引 → 新的第 i 个 stride = 旧 strides[order[i]]
    pub fn permute(&self, order: &[usize]) -> Self {
        let mut s = vec![0usize; order.len()];
        for (i, &old_idx) in order.iter().enumerate() {
            s[i] = self.0[old_idx];
        }
        Self(s)
    }
}
