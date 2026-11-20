//! reduce：规约算子（§7.5）
//!
//! 完整实现：sum / mean / max / min / prod 沿任意 axis 规约 + keepdim 支持
//!           softmax 沿指定 axis（先用 move_dim_to_last 再调 Backend::softmax）
//!
//! 核心技巧：所有规约都用 move_dim_to_last 把目标维移到最后，
//!           这样外层数量 outer = numel / inner（inner = 被规约维 size）
//!           Backend trait 默认实现的 reduce_* 函数签名正好是 (a, out, outer, inner)

use oyta_core::{Tensor, DynamicShape};
use oyta_device::Backend;

// ============================================================
// 辅助函数：move_dim_to_last / move_dim_back
// ============================================================

/// 把指定 dim 移动到最后一维（用 permute 零拷贝实现）
///
/// # 用途
/// 所有 reduce / softmax 操作都把目标维移到最后，
/// 这样 inner = shape.dims[rank-1]，outer = numel / inner，
/// 正好匹配 Backend::reduce_* 和 Backend::softmax 的签名
///
/// # 示例
/// shape=[2,3,4], dim=0 → 新 shape=[3,4,2]（batch 维移到最后）
///                       dim=1 → 新 shape=[2,4,3]
pub fn move_dim_to_last<T: Clone, D: Backend>(
    tensor: &Tensor<T, D>, dim: usize,
) -> Tensor<T, D> {
    let rank = tensor.rank();
    #[cfg(debug_assertions)]
    debug_assert!(dim < rank, "move_dim_to_last: dim {} 越界（rank={}）", dim, rank);
    if dim == rank - 1 {
        return tensor.clone(); // 已经在末尾，直接返回
    }
    // 构造 permute order：[0..dim) 不动，[dim+1..rank) 移到前面，dim 放到最后
    let mut order = Vec::with_capacity(rank);
    for i in 0..rank {
        if i != dim { order.push(i); }
    }
    order.push(dim);
    tensor.permute(&order)
}

/// 把原来在末尾的 dim 移回到 original_dim 位置（permute 逆操作）
///
/// # 原理
/// move_dim_to_last 的逆 permute：把最后一维插回 original_dim 位置
pub fn move_dim_back<T: Clone, D: Backend>(
    tensor: &Tensor<T, D>, original_dim: usize,
) -> Tensor<T, D> {
    let rank = tensor.rank();
    #[cfg(debug_assertions)]
    debug_assert!(original_dim < rank, "move_dim_back: 目标 dim 越界");
    if original_dim == rank - 1 {
        return tensor.clone(); // 已经在末尾
    }
    let last = rank - 1;
    // permute order：先收集 original_dim 之前的维，然后是最后一维，然后是 original_dim 之后的维
    let mut order = Vec::with_capacity(rank);
    for i in 0..rank {
        if i != last { order.push(i); }
        if i == original_dim { order.push(last); }
    }
    // 如果 original_dim 在最后（不该发生），补一下
    if order.len() < rank { order.push(last); }
    tensor.permute(&order)
}

// ============================================================
// 输出 shape 辅助：规约后 shape
// ============================================================

/// 计算 reduce 后的输出 shape
///
/// # 参数
/// - rank: 原 rank（move_dim_to_last 后 rank 不变）
/// - target_dim: 移到最后的那个"目标维"在原 shape 中的位置
/// - reduce_after_move: 是否是 reduce 操作（移除了该维）
/// - keepdim: true → 保留 size=1；false → 去掉该维
///
/// 注：调用方先 move_dim_to_last，reduce 之后 inner 维消失，
///     out 的 rank = rank_before_reduce - (if keepdim { 0 } else { 1 })
///     再 move_dim_back 回原位
fn reduced_shape(src_dims: &[usize], axis: usize, keepdim: bool) -> Vec<usize> {
    let mut out_dims = src_dims.to_vec();
    if keepdim {
        out_dims[axis] = 1; // 保留 size=1
    } else {
        out_dims.remove(axis); // 直接移除
    }
    out_dims
}

// ============================================================
// sum：沿 axis 求和
// ============================================================

/// 沿指定 axis 求和（完整实现，委托 Backend::reduce_sum）
///
/// # 参数
/// - a: 输入 Tensor
/// - dim: Some(axis) 沿指定维规约；None → 全部元素求和（返回标量 Tensor）
/// - keepdim: true → 规约维保留为 size=1；false → 直接移除该维
///
/// # 实现流程
/// 1. dim=None → 直接所有元素相加（rayon pairwise sum 精确）
/// 2. dim=Some(axis) → move_dim_to_last → contiguous → Backend::reduce_sum → 构造输出 shape → move_dim_back
pub fn sum<D: Backend>(
    a: &Tensor<f32, D>,
    dim: Option<usize>,
    keepdim: bool,
) -> Tensor<f32, D> {
    match dim {
        None => {
            // 全量求和 → 标量 Tensor
            let a_c = a.contiguous();
            let src = a_c.data_slice();
            // pairwise sum（二分递归，比顺序累加更数值稳定）
            let val = pairwise_sum_rust(src);
            Tensor::from_data(vec![val], DynamicShape::new([])).expect("reduce scalar 构造失败")
        }
        Some(axis) => {
            let rank = a.rank();
            let axis = if axis >= rank { axis - rank } else { axis }; // 支持负数
            let axis = if axis > rank { rank - (axis - rank) } else { axis };
            let axis = axis.min(rank - 1); // clamp

            // move_dim_to_last 让 axis 维排到末尾
            let moved = move_dim_to_last(a, axis);
            let moved_c = moved.contiguous();

            let inner = moved_c.shape.dims[moved_c.rank() - 1]; // 被规约的维 size
            let outer = moved_c.numel() / inner; // 外层数量

            // 调 Backend::reduce_sum
            let mut out_data = vec![0.0f32; outer];
            D::reduce_sum(
                moved_c.data_ptr(),
                out_data.as_mut_ptr(),
                outer,
                inner,
            );

            // 构造输出 shape：keepdim=true 时保留最后一维 size=1；false 时移除
            let moved_out_shape_dims: Vec<usize> = if keepdim {
                let mut d = moved_c.shape.dims.clone();
                let last = d.len() - 1; // 先算索引，避免和下方可变借用冲突
                d[last] = 1;
                d
            } else {
                let mut d = moved_c.shape.dims.clone();
                d.pop();
                d
            };
            let moved_out = Tensor::from_data(out_data, DynamicShape::new(moved_out_shape_dims)).expect("reduce moved_out 构造失败");

            // 再 move_dim_back 回原位置
            let result = if keepdim {
                // keepdim=true 时末尾有个 size=1 的维，move_dim_back 把它插回原 axis
                move_dim_back(&moved_out, axis)
            } else {
                // keepdim=false：被规约维已经移除，现在要把新末尾（原 axis 后面那个）移回原 axis
                // move_dim_back 在 rank 变化的情况下不太好用，手动构造
                let final_dims = reduced_shape(&a.shape.dims, axis, keepdim);
                Tensor::from_data(moved_out.data_slice().to_vec(), DynamicShape::new(final_dims)).expect("reduce final_dims 构造失败")
            };
            result
        }
    }
}

// ============================================================
// mean：沿 axis 求均值
// ============================================================

/// 沿指定 axis 求均值
///
/// 实现方式：sum / inner_size（或直接调 Backend::reduce_mean）
pub fn mean<D: Backend>(
    a: &Tensor<f32, D>,
    dim: Option<usize>,
    keepdim: bool,
) -> Tensor<f32, D> {
    match dim {
        None => {
            // 全量求均值
            let a_c = a.contiguous();
            let src = a_c.data_slice();
            let n = src.len();
            let val = if n == 0 { 0.0 } else { pairwise_sum_rust(src) / n as f32 };
            Tensor::from_data(vec![val], DynamicShape::new([])).expect("reduce scalar 构造失败")
        }
        Some(axis) => {
            let rank = a.rank();
            let axis = axis.min(rank - 1);

            let moved = move_dim_to_last(a, axis);
            let moved_c = moved.contiguous();
            let inner = moved_c.shape.dims[moved_c.rank() - 1];
            let outer = moved_c.numel() / inner;

            let mut out_data = vec![0.0f32; outer];
            D::reduce_mean(
                moved_c.data_ptr(),
                out_data.as_mut_ptr(),
                outer,
                inner,
            );

            let final_dims = reduced_shape(&a.shape.dims, axis, keepdim);
            Tensor::from_data(out_data, DynamicShape::new(final_dims)).expect("reduce 输出构造失败")
        }
    }
}

// ============================================================
// max：沿 axis 求最大值
// ============================================================

/// 沿指定 axis 求最大值
pub fn max<D: Backend>(
    a: &Tensor<f32, D>,
    dim: Option<usize>,
    keepdim: bool,
) -> Tensor<f32, D> {
    match dim {
        None => {
            let a_c = a.contiguous();
            let val = a_c.data_slice().iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            Tensor::from_data(vec![val], DynamicShape::new([])).expect("reduce scalar 构造失败")
        }
        Some(axis) => {
            let rank = a.rank();
            let axis = axis.min(rank - 1);

            let moved = move_dim_to_last(a, axis);
            let moved_c = moved.contiguous();
            let inner = moved_c.shape.dims[moved_c.rank() - 1];
            let outer = moved_c.numel() / inner;

            let mut out_data = vec![0.0f32; outer];
            D::reduce_max(
                moved_c.data_ptr(),
                out_data.as_mut_ptr(),
                outer,
                inner,
            );

            let final_dims = reduced_shape(&a.shape.dims, axis, keepdim);
            Tensor::from_data(out_data, DynamicShape::new(final_dims)).expect("reduce 输出构造失败")
        }
    }
}

// ============================================================
// min：沿 axis 求最小值
// ============================================================

/// 沿指定 axis 求最小值
pub fn min<D: Backend>(
    a: &Tensor<f32, D>,
    dim: Option<usize>,
    keepdim: bool,
) -> Tensor<f32, D> {
    match dim {
        None => {
            let a_c = a.contiguous();
            let val = a_c.data_slice().iter().cloned().fold(f32::INFINITY, f32::min);
            Tensor::from_data(vec![val], DynamicShape::new([])).expect("reduce scalar 构造失败")
        }
        Some(axis) => {
            let rank = a.rank();
            let axis = axis.min(rank - 1);

            let moved = move_dim_to_last(a, axis);
            let moved_c = moved.contiguous();
            let inner = moved_c.shape.dims[moved_c.rank() - 1];
            let outer = moved_c.numel() / inner;

            let mut out_data = vec![0.0f32; outer];
            D::reduce_min(
                moved_c.data_ptr(),
                out_data.as_mut_ptr(),
                outer,
                inner,
            );

            let final_dims = reduced_shape(&a.shape.dims, axis, keepdim);
            Tensor::from_data(out_data, DynamicShape::new(final_dims)).expect("reduce 输出构造失败")
        }
    }
}

// ============================================================
// prod：沿 axis 求乘积（Backend trait 没有 prod，纯 Rust 实现）
// ============================================================

/// 沿指定 axis 求乘积
///
/// Backend trait 没有 reduce_prod，这里用纯 Rust 实现
pub fn prod<D: Backend>(
    a: &Tensor<f32, D>,
    dim: Option<usize>,
    keepdim: bool,
) -> Tensor<f32, D> {
    match dim {
        None => {
            let a_c = a.contiguous();
            let val = a_c.data_slice().iter().cloned().fold(1.0f32, |acc, v| acc * v);
            Tensor::from_data(vec![val], DynamicShape::new([])).expect("reduce scalar 构造失败")
        }
        Some(axis) => {
            let rank = a.rank();
            let axis = axis.min(rank - 1);

            let moved = move_dim_to_last(a, axis);
            let moved_c = moved.contiguous();
            let inner = moved_c.shape.dims[moved_c.rank() - 1];
            let outer = moved_c.numel() / inner;

            let src = moved_c.data_slice();
            let mut out_data = vec![1.0f32; outer];
            for o in 0..outer {
                let base = o * inner;
                let mut acc = 1.0f32;
                for i in 0..inner {
                    acc *= src[base + i];
                }
                out_data[o] = acc;
            }

            let final_dims = reduced_shape(&a.shape.dims, axis, keepdim);
            Tensor::from_data(out_data, DynamicShape::new(final_dims)).expect("reduce 输出构造失败")
        }
    }
}

// ============================================================
// softmax：沿 dim 做 softmax
// ============================================================

/// 沿指定 axis 做 softmax（数值稳定：减 max → exp → 除 sum）
///
/// # 实现流程
/// move_dim_to_last → contiguous → Backend::softmax → move_dim_back
/// 输出 shape 和输入一样（softmax 不改变 shape）
pub fn softmax<D: Backend>(
    a: &Tensor<f32, D>,
    dim: usize,
) -> Tensor<f32, D> {
    let rank = a.rank();
    let axis = dim.min(rank - 1);

    // move_dim_to_last 让目标维在末尾
    let moved = move_dim_to_last(a, axis);
    let moved_c = moved.contiguous();

    let inner = moved_c.shape.dims[moved_c.rank() - 1];
    let outer = moved_c.numel() / inner;

    // Backend::softmax 直接写回完整大小的输出
    let mut out_data = vec![0.0f32; moved_c.numel()];
    D::softmax(
        moved_c.data_ptr(),
        out_data.as_mut_ptr(),
        outer,
        inner,
    );

    let moved_out = Tensor::from_data(out_data, moved_c.shape.clone()).expect("reduce moved_out 构造失败");

    // move_dim_back 回原位
    move_dim_back(&moved_out, axis)
}

// ============================================================
// pairwise_sum：递归二分归约（比顺序累加更数值稳定）
// ============================================================

/// pairwise sum：递归二分归约
///
/// 把数组一分为二递归求和，比顺序累加更能抵抗舍入误差
/// （顺序累加误差 O(n*eps)，二分归约 O(log n * eps)）
fn pairwise_sum_rust(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 { return 0.0; }
    if len == 1 { return data[0]; }
    let half = len / 2;
    pairwise_sum_rust(&data[..half]) + pairwise_sum_rust(&data[half..])
}
