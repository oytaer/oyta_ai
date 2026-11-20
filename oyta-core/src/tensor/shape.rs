//! tensor_shape.rs：Tensor 形状变换 + 切片（零拷贝 view）
use super::Tensor;
use crate::shape::DynamicShape;
use crate::strides::Strides;
use crate::error::TensorError;
use oyta_device::Backend;

impl<T: Clone, D: Backend> Tensor<T, D> {
    // -- 形状变换（全部零拷贝） --

    /// reshape（新 shape 必须 numel 相同）
    pub fn reshape(&self, new_dims: impl IntoIterator<Item = usize>) -> Result<Self, TensorError> {
        let new_shape = DynamicShape::new(new_dims);
        #[cfg(debug_assertions)]
        debug_assert!(
            new_shape.numel() == self.shape.numel(),
            "reshape numel 不匹配：旧 numel={}, 新 numel={}",
            self.shape.numel(), new_shape.numel(),
        );
        if new_shape.numel() != self.shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.shape.numel()],
                actual: vec![new_shape.numel()],
            });
        }
        let new_strides = new_shape.c_contiguous_strides(); // reshape 后默认行优先
        Ok(Self::from_shared(self.data.clone(), new_shape, new_strides, self.offset))
    }

    /// 交换两个维度（转置）
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let rank = self.shape.rank();
        #[cfg(debug_assertions)]
        debug_assert!(dim0 < rank && dim1 < rank); // 边界检查
        let mut new_dims = self.shape.dims.clone();
        let mut new_strides = self.strides.0.clone();
        new_dims.swap(dim0, dim1); // 交换维度大小
        new_strides.swap(dim0, dim1); // 同步交换步长
        Self::from_shared(self.data.clone(), DynamicShape::new(new_dims), Strides(new_strides), self.offset)
    }

    /// 按给定顺序排列所有维度（广义 permute）
    pub fn permute(&self, order: &[usize]) -> Self {
        let rank = self.shape.rank();
        #[cfg(debug_assertions)]
        debug_assert_eq!(order.len(), rank); // 维度数匹配
        let mut new_dims = Vec::with_capacity(rank);
        let mut new_strides = Vec::with_capacity(rank);
        for &d in order {
            #[cfg(debug_assertions)]
            debug_assert!(d < rank); // 每个目标维度合法
            new_dims.push(self.shape.dims[d]); // 按新顺序收集 dim
            new_strides.push(self.strides.0[d]); // 同步收集 stride
        }
        Self::from_shared(self.data.clone(), DynamicShape::new(new_dims), Strides(new_strides), self.offset)
    }

    /// 在指定位置插入一个 size=1 的维度（unsqueeze）
    pub fn unsqueeze(&self, dim: isize) -> Self {
        let rank = self.shape.rank();
        let dim = if dim < 0 { (rank as isize + dim) as usize } else { dim as usize }; // 负索引转正
        #[cfg(debug_assertions)]
        debug_assert!(dim <= rank); // 允许插到末尾
        let mut new_dims = self.shape.dims.clone();
        let mut new_strides = self.strides.0.clone();
        new_dims.insert(dim, 1); // 插入 size=1
        new_strides.insert(dim, 0); // stride=0（broadcast 语义）
        Self::from_shared(self.data.clone(), DynamicShape::new(new_dims), Strides(new_strides), self.offset)
    }

    /// 去掉所有 size=1 的维度（squeeze）
    pub fn squeeze(&self) -> Self {
        let mut new_dims = Vec::new();
        let mut new_strides = Vec::new();
        for (d, &s) in self.shape.dims.iter().zip(self.strides.0.iter()) {
            if *d != 1 { // 跳过 size=1 的维度
                new_dims.push(*d);
                new_strides.push(s);
            }
        }
        Self::from_shared(self.data.clone(), DynamicShape::new(new_dims), Strides(new_strides), self.offset)
    }

    // -- 切片（零拷贝 + select 非零拷贝） --

    /// 按维度切片（零拷贝：offset 前进 + shape 缩小）
    /// 例：shape=[4,6].slice(0, 1..3) → shape=[2,6]
    pub fn slice(&self, dim: usize, range: std::ops::Range<usize>) -> Self {
        #[cfg(debug_assertions)]
        debug_assert!(range.start <= range.end && range.end <= self.shape.dims[dim], "slice 范围越界");
        let step = self.strides.0[dim]; // 该维 stride
        let new_offset = self.offset + range.start * step;
        let mut new_dims = self.shape.dims.clone();
        new_dims[dim] = range.end - range.start;
        Self::from_shared(self.data.clone(), DynamicShape::new(new_dims), self.strides.clone(), new_offset)
    }

    /// 多维度切片（每维一个 Range）
    pub fn multi_slice(&self, ranges: &[std::ops::Range<usize>]) -> Self {
        #[cfg(debug_assertions)]
        debug_assert_eq!(ranges.len(), self.shape.rank());
        let mut new_offset = self.offset;
        let mut new_dims = Vec::with_capacity(ranges.len());
        for (i, r) in ranges.iter().enumerate() {
            #[cfg(debug_assertions)]
            debug_assert!(r.start <= r.end && r.end <= self.shape.dims[i], "multi_slice 范围越界");
            new_offset += r.start * self.strides.0[i];
            new_dims.push(r.end - r.start);
        }
        Self::from_shared(self.data.clone(), DynamicShape::new(new_dims), self.strides.clone(), new_offset)
    }

    /// 在指定维度选择索引（非零拷贝：indices 不连续时必须重排）
    pub fn select(&self, dim: usize, indices: &[usize]) -> Self {
        #[cfg(debug_assertions)]
        debug_assert!(indices.iter().all(|&i| i < self.shape.dims[dim]), "select 索引越界");
        let mut new_dims = self.shape.dims.clone();
        new_dims[dim] = indices.len();
        let dim_stride = self.strides.0[dim];
        let mut new_data = Vec::with_capacity(indices.len() * (self.numel() / self.shape.dims[dim]));
        // 对每个选中的 slice，拷贝其所有元素（C-contiguous 假设）
        let inner_count = self.numel() / self.shape.dims[dim];
        let outer_shape_dims: Vec<usize> = self.shape.dims.iter().enumerate()
            .filter(|(i, _)| *i != dim).map(|(_, &s)| s).collect();
        // 先切出所有 "dim 左侧" 维的外循环，"dim 右侧" 维的内循环
        let left_size: usize = outer_shape_dims.iter().take(dim).product();
        let right_size: usize = outer_shape_dims.iter().skip(dim).product();
        // 对每个 (left_idx, selected_idx, right_idx) 三循环
        for li in 0..left_size {
            for &sel_idx in indices {
                for ri in 0..right_size {
                    // 计算逻辑坐标 → 物理偏移
                    let mut logic_offset = self.offset;
                    // 左半
                    let mut remaining = li;
                    for i in 0..dim {
                        let stride = self.strides.0[i];
                        let dim_size = self.shape.dims[i];
                        logic_offset += (remaining % dim_size) * stride;
                        remaining /= dim_size;
                    }
                    // 选中维
                    logic_offset += sel_idx * dim_stride;
                    // 右半
                    let mut rem_right = ri;
                    for i in (dim + 1)..self.shape.rank() {
                        let stride = self.strides.0[i];
                        let dim_size = self.shape.dims[i];
                        logic_offset += (rem_right % dim_size) * stride;
                        rem_right /= dim_size;
                    }
                    new_data.push(self.data[logic_offset].clone());
                }
            }
        }
        let out = Tensor::from_data(new_data, DynamicShape::new(new_dims)).expect("select 输出构造失败");
        let _ = inner_count; // 消除未使用警告
        out
    }
}
