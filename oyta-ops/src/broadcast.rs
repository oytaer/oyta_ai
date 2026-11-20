//! broadcast：广播预处理（§7.3）
//!
//! 完整实现：检查 broadcast_compatible + contiguous 预处理 + stride 调整
//! 被广播的维 stride 设为 0（底层数据不扩展，只是逻辑 stride 为 0）
//! 两个 Tensor 都用 Arc<Vec<T>> 共享底层数据，零拷贝

use oyta_core::{Tensor, DynamicShape, Strides};
use oyta_device::Backend;

/// 广播预处理：把 a 和 b 调整到可以广播计算的形式
///
/// # 核心逻辑
/// 1. 检查两个 shape 是否广播兼容（rank 短的左边补 1）
/// 2. 把两个 Tensor 都变成 contiguous（先调用 .contiguous()）
/// 3. 计算目标 shape（broadcast_with）
/// 4. 调整 strides：对于原来 size=1 的维度（被广播），stride 设为 0
///
/// # 为什么 stride=0
/// 当某个维度 stride=0 时，IndexEngine.offset() 会计算 logic_coord * 0 = 0，
/// 意味着不管该维索引是多少，物理偏移都不变 → 正确实现广播"取同一个元素"语义
///
/// # 返回
/// - 两个调整过 strides 的 Tensor（共享原底层数据，零拷贝）
/// - 目标广播 shape
///
/// # 错误
/// - shape 不兼容 → Err(String)
pub fn preprocess_binary<T: Clone, D: Backend>(
    a: &Tensor<T, D>, b: &Tensor<T, D>,
) -> Result<(Tensor<T, D>, Tensor<T, D>, DynamicShape), String> {
    // ---- 第一步：检查广播兼容 ----
    if !a.shape.broadcast_compatible(&b.shape) {
        return Err(format!(
            "broadcast 失败：shape {:?} 和 {:?} 不兼容",
            a.shape.dims, b.shape.dims
        ));
    }
    let target_shape = a.shape.broadcast_with(&b.shape);

    // ---- 第二步：先 contiguous（stride 调整基于 C-contiguous 的底层数据）----
    let a_contig = a.contiguous();
    let b_contig = b.contiguous();

    // ---- 第三步：调整各自的 strides ----
    // 核心：原来 size=1 的维度（被广播）→ stride=0
    // rank 不够的维（左边补 1）→ 也 stride=0
    let a_new_strides = compute_broadcast_strides(&a_contig.shape.dims, &target_shape.dims);
    let b_new_strides = compute_broadcast_strides(&b_contig.shape.dims, &target_shape.dims);

    // ---- 第四步：用 from_shared 构造新 Tensor（零拷贝，共享 Arc<Vec<T>>）----
    let a_broadcasted = Tensor::<T, D>::from_shared(
        a_contig.data.clone(),
        target_shape.clone(),
        Strides(a_new_strides),
        a_contig.offset,
    );
    let b_broadcasted = Tensor::<T, D>::from_shared(
        b_contig.data.clone(),
        target_shape.clone(),
        Strides(b_new_strides),
        b_contig.offset,
    );

    Ok((a_broadcasted, b_broadcasted, target_shape))
}

/// 根据源 shape 和目标 shape 计算广播后的 strides
///
/// # 规则
/// - 对于两边 rank 对齐后的每一维：
///   - 如果源该维 size == 1（被广播）→ stride = 0
///   - 如果源该维 size != 1（真实维）→ 用 contiguous stride
/// - 源没有、目标有的维（左边补的 1）→ stride = 0
///
/// # 示例
/// 源 a shape=[3] 目标=[2,3] → a_new_strides = [0, 1]
///   - 左边补的 2 对应维：stride=0（整个 batch 取同一个 [3] slice）
///   - 右边 3 对应维：原来 size=3 非 1，stride=1（contiguous）
///
/// 源 a shape=[1,4] 目标=[3,4] → a_new_strides = [0, 1]
///   - 第一维原来 size=1 被广播 → stride=0
///   - 第二维原来 size=4 → stride=1
fn compute_broadcast_strides(src_dims: &[usize], target_dims: &[usize]) -> Vec<usize> {
    let src_rank = src_dims.len();
    let target_rank = target_dims.len();
    let pad = target_rank.saturating_sub(src_rank); // 左边补的维度数

    // 先计算源的 contiguous strides（用于非广播维）
    let src_contig_strides = DynamicShape::new(src_dims.to_vec()).contiguous_strides();

    let mut new_strides = Vec::with_capacity(target_rank);
    for i in 0..target_rank {
        if i < pad {
            // 这是左边补的 1：源里根本没有这个维 → stride=0
            new_strides.push(0);
        } else {
            // 对应源的某个维度
            let src_idx = i - pad;
            let src_size = src_dims[src_idx];
            if src_size == 1 {
                // 源 size=1 → 被广播 → stride=0
                new_strides.push(0);
            } else {
                // 源 size != 1 → 真实数据维 → 用源的 contiguous stride
                new_strides.push(src_contig_strides[src_idx]);
            }
        }
    }
    new_strides
}

/// 便捷版本：只广播一个 Tensor 到目标 shape（unary 算子用）
///
/// 把 a 从 src_shape 广播到 target_shape，返回调整过 strides 的 Tensor
pub fn broadcast_to_shape<T: Clone, D: Backend>(
    a: &Tensor<T, D>, target_shape: &DynamicShape,
) -> Tensor<T, D> {
    let a_contig = a.contiguous();
    let new_strides = compute_broadcast_strides(&a_contig.shape.dims, &target_shape.dims);
    Tensor::<T, D>::from_shared(
        a_contig.data.clone(),
        target_shape.clone(),
        Strides(new_strides),
        a_contig.offset,
    )
}

/// 便捷函数：把两个 Tensor 广播到相同 shape（elwise 算子最常用入口）
///
/// 返回两个都调整好 strides 的 Tensor，底层数据都是 contiguous 的共享引用
pub fn broadcast_to_match<T: Clone, D: Backend>(
    a: &Tensor<T, D>, b: &Tensor<T, D>,
) -> Result<(Tensor<T, D>, Tensor<T, D>), String> {
    let (a_b, b_b, _target) = preprocess_binary(a, b)?;
    Ok((a_b, b_b))
}


