//! dropout：Dropout 正则化（§7.7 逐元素 mask）
//!
//! # 训练 / 推理区分
//! - 训练态：随机 mask，p=0.1 → 10% 元素置 0，其他放大 1/(1-p) 补偿均值
//! - 推理态：什么都不做（前向同 identity）
//!
//! # 实现
//! 1. 生成 Uniform(0,1) 随机数 mask
//! 2. mask < p 的位置置 0，其他乘以 scale = 1/(1-p)

use oyta_core::Tensor;
use oyta_device::Backend;

/// 全局训练态标志（默认 false）
static IS_TRAINING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 设置训练态
///
/// # Ordering
/// Release：确保此 store 之前的所有写操作（如 RNG seed 设置）对其它线程可见
pub fn set_training_mode(training: bool) {
    IS_TRAINING.store(training, std::sync::atomic::Ordering::Release);
}

/// 当前是否训练态
///
/// # Ordering
/// Acquire：确保此 load 之后的读操作能看到其它线程 Release store 之前的所有写
pub fn is_training_mode() -> bool {
    IS_TRAINING.load(std::sync::atomic::Ordering::Acquire)
}

/// Dropout 操作
///
/// # 参数
/// - x: 输入 Tensor
/// - p: 丢弃概率（0.0 ~ 1.0）
/// # 行为
/// 训练态：随机 mask + scale
/// 推理态：直接返回输入（identity）
pub fn dropout<D: Backend>(x: &Tensor<f32, D>, p: f32) -> Tensor<f32, D> {
    if !is_training_mode() { return x.clone(); } // 推理态什么都不做
    #[cfg(debug_assertions)]
    debug_assert!((0.0..1.0).contains(&p), "p 必须在 [0,1)");

    let x_c = x.contiguous();
    let n = x_c.numel();
    let scale = 1.0 / (1.0 - p); // 补偿因子：保持期望值不变
    let mut out_data = Vec::with_capacity(n);

    // 逐元素采样 + mask（CPU 实现，未来移到 Backend trait）
    use rand::RngExt;
    use rand_distr::StandardUniform;
    let rng = rand::rng();
    let mask_iter = rng.sample_iter::<f32, _>(StandardUniform).take(n);

    for (v, mask_val) in x_c.data_slice().iter().zip(mask_iter) {
        if mask_val < p {
            out_data.push(0.0); // 丢弃 → 0
        } else {
            out_data.push(*v * scale); // 保留 → 放大
        }
    }
    Tensor::from_data(out_data, x_c.shape().clone()).expect("Tensor 构造失败")
}
