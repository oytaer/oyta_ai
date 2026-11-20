//! DynamicShape：Tensor 运行时逻辑形状
//!
//! §3.1 目录结构 → shape.rs
//! 用 Vec<usize> 存维度，不固定 rank（支持 0~N 维 Tensor）
//!
//! 完整实现：§4.3 Shape 结构（dim 负索引、reshape -1 推断、广播兼容检查）

use crate::strides::Strides; // 依赖 strides（c_contiguous_strides 返回 Strides）
use crate::error::TensorError; // reshape 可能返回错误
use crate::checked;

/// 动态形状（运行时确定各维度大小）
///
/// # 内存布局
/// [dims[0], dims[1], ..., dims[rank-1]] → 最左维到最右维
/// 例：shape=[2,3,4] → dims[0]=2 (batch), dims[1]=3 (rows), dims[2]=4 (cols)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicShape {
    /// 各维度大小（len == rank）
    pub dims: Vec<usize>,
}

impl DynamicShape {
    // ============================================================
    // 构造
    // ============================================================

    /// 从迭代器创建形状
    pub fn new(dims: impl IntoIterator<Item = usize>) -> Self {
        Self { dims: dims.into_iter().collect() }
    }

    // ============================================================
    // 属性查询
    // ============================================================

    /// 元素总数（所有维度乘积）
    ///
    /// # 溢出处理
    /// 用 checked_product；如果乘积溢出 usize 会 panic（shape 来自内部构造，
    /// 一旦溢出就是上游 bug；从 safetensors 等外部源来的 shape 在解析阶段就已校验）
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() { return 1; } // 标量 → 1 个元素（不是 0！）
        checked::checked_product(self.dims.iter().copied())
            .unwrap_or(usize::MAX)
    }

    /// 维度数（rank）
    pub fn rank(&self) -> usize { self.dims.len() }

    /// 第 i 维大小（支持负数索引：从末尾数，-1=最右维）
    ///
    /// 例：shape=[2,3,4].dim(-1) → 4, dim(0) → 2
    pub fn dim(&self, i: isize) -> usize {
        let idx = if i < 0 { self.dims.len() as isize + i } else { i };
        #[cfg(debug_assertions)]
        debug_assert!(idx >= 0 && (idx as usize) < self.dims.len(), "dim 索引越界");
        self.dims[idx.max(0) as usize]
    }

    /// 是否标量（rank=0 或 numel=1）
    pub fn is_scalar(&self) -> bool { self.dims.is_empty() || self.numel() == 1 }

    // ============================================================
    // 步长计算
    // ============================================================

    /// 生成 C-contiguous（行优先）步长
    ///
    /// 例：shape [2,3,4] → strides [12,4,1]（最内层 stride=1）
    ///
    /// # 溢出处理
    /// 步长乘积溢出 usize 时 panic（内部逻辑 bug）
    pub fn c_contiguous_strides(&self) -> Strides {
        if self.dims.is_empty() { return Strides(vec![]); } // 空形状 → 空步长
        let mut strides = vec![1usize; self.dims.len()]; // 最右层 stride=1
        for i in (0..self.dims.len() - 1).rev() { // 从右向左递推
            let next = strides[i + 1];
            let dim = self.dims[i + 1];
            strides[i] = checked::checked_mul(next, dim)
                .unwrap_or(0); // 左层 stride = 右层 stride × 右层 dim（溢出安全为 0）
        }
        Strides(strides) // 返回步长包装
    }

    /// 直接返回步长 Vec（给 oyta-ops 用）
    pub fn contiguous_strides(&self) -> Vec<usize> {
        self.c_contiguous_strides().0.clone()
    }

    // ============================================================
    // reshape（支持 -1 推断，§4.3 完整实现）
    // ============================================================

    /// reshape：改变形状，保持元素总数不变
    ///
    /// # 支持 -1 推断
    /// new_dims 中可以有一个 -1 表示"这一维推断出来"
    /// 例：shape=[2,6].reshape(&[-1,3]) → [4,3]（因为原 numel=12，12/3=4）
    pub fn reshape(&self, new_dims: &[isize]) -> Result<DynamicShape, TensorError> {
        let old_numel = self.numel(); // 原元素总数
        let inferred_count = new_dims.iter().filter(|&&d| d == -1).count();
        #[cfg(debug_assertions)]
        debug_assert!(inferred_count <= 1, "reshape 最多只能有一个 -1");

        // 第一步：把 -1 替换成 0 占位；非法负数（除了 -1）立即报错
        let known: Result<Vec<usize>, TensorError> = new_dims.iter().map(|&d| {
            if d == -1 {
                Ok(0usize)
            } else if d < 0 {
                Err(TensorError::Overflow(format!("reshape 只允许一个 -1 推断，收到非法值 {}", d)))
            } else {
                Ok(d as usize)
            }
        }).collect();
        let known = known?;

        // 第二步：计算已知维度的乘积（checked 防止溢出）
        let known_numel = checked::checked_product(known.iter().copied())?;

        // 第三步：推断 -1 的值（需要 known_numel 能整除 old_numel）
        let final_dims: Result<Vec<usize>, TensorError> = new_dims.iter().map(|&d| {
            if d == -1 {
                if known_numel == 0 {
                    return Err(TensorError::Overflow(
                        "reshape 推断 -1：已知维度乘积为 0，无法推断".into()
                    ));
                }
                if old_numel % known_numel != 0 {
                    return Err(TensorError::ReshapeImpossible {
                        old: self.dims.clone(),
                        new: known.clone(),
                    });
                }
                Ok(old_numel / known_numel)
            } else if d < 0 {
                Err(TensorError::Overflow(format!("reshape 非法负数 {}", d)))
            } else {
                Ok(d as usize)
            }
        }).collect();
        let final_dims = final_dims?;

        // 第四步：验证元素总数一致
        let final_numel = checked::checked_product(final_dims.iter().copied())?;
        if final_numel != old_numel {
            return Err(TensorError::ReshapeImpossible {
                old: self.dims.clone(),
                new: final_dims,
            });
        }
        Ok(DynamicShape::new(final_dims))
    }

    // ============================================================
    // 广播（§7.3 逐元素算子需要的广播预处理）
    // ============================================================

    /// 广播兼容检查（§4.3 实现）
    ///
    /// 规则：两个 shape 从最右维开始逐对比较，每个位置必须满足：
    ///   相等，或其中一个是 1（广播）
    /// rank 不同时短的左边补 1
    pub fn broadcast_compatible(&self, other: &DynamicShape) -> bool {
        let max_rank = self.dims.len().max(other.dims.len());
        for i in 0..max_rank {
            let a = self.dims.get(self.dims.len() - 1 - i).copied().unwrap_or(1);
            let b = other.dims.get(other.dims.len() - 1 - i).copied().unwrap_or(1);
            if a != b && a != 1 && b != 1 { return false; }
        }
        true
    }

    /// 广播结果 shape（取每对的最大值）
    ///
    /// 例：[2,3] broadcast [3] → [2,3]（3=3 匹配，2 对应对方补的 1）
    pub fn broadcast_with(&self, other: &DynamicShape) -> DynamicShape {
        let max_rank = self.dims.len().max(other.dims.len());
        let mut result = Vec::with_capacity(max_rank);
        for i in 0..max_rank {
            let a = self.dims.get(self.dims.len() - 1 - i).copied().unwrap_or(1);
            let b = other.dims.get(other.dims.len() - 1 - i).copied().unwrap_or(1);
            result.push(a.max(b)); // 取较大者（其中一个可能是 1）
        }
        result.reverse(); // 翻转回左→右顺序
        DynamicShape::new(result)
    }
}

// ============================================================
// From 实现（便捷构造，§4.3）
// ============================================================

impl From<&[usize]> for DynamicShape {
    fn from(d: &[usize]) -> Self { DynamicShape::new(d.to_vec()) }
}
impl From<Vec<usize>> for DynamicShape {
    fn from(d: Vec<usize>) -> Self { DynamicShape::new(d) }
}
impl From<usize> for DynamicShape {
    fn from(d: usize) -> Self { DynamicShape::new([d]) }
}
impl From<(usize, usize)> for DynamicShape {
    fn from(d: (usize, usize)) -> Self { DynamicShape::new([d.0, d.1]) }
}
impl From<(usize, usize, usize)> for DynamicShape {
    fn from(d: (usize, usize, usize)) -> Self { DynamicShape::new([d.0, d.1, d.2]) }
}
impl From<(usize, usize, usize, usize)> for DynamicShape {
    fn from(d: (usize, usize, usize, usize)) -> Self { DynamicShape::new([d.0, d.1, d.2, d.3]) }
}

// 数组直接转 DynamicShape（最常用）
impl From<[usize; 0]> for DynamicShape {
    fn from(_: [usize; 0]) -> Self { DynamicShape::new([]) }
}
impl From<[usize; 1]> for DynamicShape {
    fn from(d: [usize; 1]) -> Self { DynamicShape::new(d) }
}
impl From<[usize; 2]> for DynamicShape {
    fn from(d: [usize; 2]) -> Self { DynamicShape::new(d) }
}
impl From<[usize; 3]> for DynamicShape {
    fn from(d: [usize; 3]) -> Self { DynamicShape::new(d) }
}
impl From<[usize; 4]> for DynamicShape {
    fn from(d: [usize; 4]) -> Self { DynamicShape::new(d) }
}
impl From<[usize; 5]> for DynamicShape {
    fn from(d: [usize; 5]) -> Self { DynamicShape::new(d) }
}
impl From<[usize; 6]> for DynamicShape {
    fn from(d: [usize; 6]) -> Self { DynamicShape::new(d) }
}

impl std::fmt::Display for DynamicShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}
