//! IndexEngine：多维逻辑索引 → 线性偏移转换
//!
//! §3.1 目录结构 → index.rs
//! §7.3 逐元素算子（elwise_generic）需要处理非 contiguous 输入时，
//! 需要把逻辑坐标 (i,j,k,...) 映射到底层 flat 数组的物理偏移 offset。
//!
//! # 核心公式
//! flat_offset = sum(logic_index[i] * strides[i])
//!
//! 例：shape=[2,3,4] strides=[12,4,1]
//! 逻辑坐标 (1,2,3) → 1*12 + 2*4 + 3*1 = 23 → 底层 data[23]
//!
//! # 为什么单独封装
//! 广播场景 stride 里有 0（broadcast 维度），此时
//! flat_offset 会忽略那个维度（因为 logic_index * 0 = 0）

use crate::shape::DynamicShape; // shape 维度
use crate::strides::Strides;     // strides 步长

/// 多维→线性偏移转换器
#[derive(Clone, Debug)]
pub struct IndexEngine {
    shape: DynamicShape,  // 逻辑形状
    strides: Vec<usize>,  // 步长向量（复制出来避免借用检查）
}

impl IndexEngine {
    /// 从 shape + strides 创建
    pub fn new(shape: DynamicShape, strides: &Strides) -> Self {
        Self { shape, strides: strides.0.clone() }
    }

    /// 给定 flat 线性索引，返回逻辑各维坐标
    ///
    /// flat→logic：按 C-contiguous 规则分解
    pub fn logic_coords(&self, flat_idx: usize) -> Vec<usize> {
        let mut remaining = flat_idx;
        let mut coords = Vec::with_capacity(self.shape.rank());
        // 从右向左分解（最右维 stride=1）
        for i in (0..self.shape.rank()).rev() {
            let dim_size = self.shape.dims[i];
            coords.push(remaining % dim_size);
            remaining /= dim_size;
        }
        coords.reverse(); // 翻回左→右顺序
        coords
    }

    /// 给定 flat 索引，返回在底层 data 里的物理偏移
    ///
    /// # 公式
    /// physics_offset = sum(logic_coords[i] * strides[i])
    ///
    /// # 广播场景
    /// 如果 strides[i] == 0（该维被广播），那 logic_coords[i] * 0 = 0，
    /// 物理偏移会忽略这个维度 → 正确实现广播
    pub fn offset(&self, flat_idx: usize) -> usize {
        let coords = self.logic_coords(flat_idx);
        let mut off = 0usize;
        for (i, &c) in coords.iter().enumerate() {
            off += c * self.strides[i]; // stride=0 时自动跳过
        }
        off
    }

    /// 给定直接的逻辑坐标数组，返回物理偏移（跳过 flat→logic 分解）
    pub fn offset_from_coords(&self, coords: &[usize]) -> usize {
        let mut off = 0usize;
        for (i, &c) in coords.iter().enumerate() {
            off += c * self.strides[i];
        }
        off
    }
}
