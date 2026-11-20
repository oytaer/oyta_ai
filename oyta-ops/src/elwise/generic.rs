//! elwise：逐元素算子（§7.3 宏模式 + 双路径 dispatch）
//!
//! 完整实现：
//! - elwise_generic: 用 IndexEngine 处理非 contiguous + 广播 stride=0 的通用路径
//! - broadcast_to_match: 把两个 Tensor 广播到相同 shape（委托 broadcast 模块）
//! - impl_elwise_op! 宏：为每个二元算子生成"contiguous → Backend / 非 contiguous → elwise_generic"的双路径函数
//! - impl_elwise_unary! 宏：为每个一元算子生成"contiguous → Backend / 非 contiguous → 手动遍历"的双路径函数
//!
//! # 双路径设计
//! 1. 如果两个输入都是 contiguous → 直接调 Backend::elwise_*（rayon 并行、SIMD 友好）
//! 2. 否则（转置后、广播后、切片后）→ elwise_generic 用 IndexEngine 逐元素取值
//!
//! # 为什么需要 elwise_generic
//! broadcast 后的 Tensor 有 stride=0 的维，直接 Backend kernel 会把它当作完整数据重算一遍，
//! 浪费内存。IndexEngine.offset() 会正确处理 stride=0（logic_coord * 0 = 0）

use oyta_core::{Tensor, DynamicShape, IndexEngine};
use oyta_device::Backend;

// 引入 broadcast 模块的便捷函数
use crate::broadcast::preprocess_binary;

// ============================================================
// elwise_generic：通用逐元素算子（处理非 contiguous + 广播 stride=0）
// ============================================================

/// 通用逐元素算子：用 IndexEngine 正确处理所有布局
///
/// # 核心机制
/// IndexEngine::offset(flat_idx) 会：
/// 1. 把 flat_idx 分解成多维逻辑坐标
/// 2. 用 stride 计算物理偏移
///    - stride=0 的维（被广播）：logic_coord * 0 = 0 → 取同一个元素
///    - stride>0 的维：正常计算偏移
///
/// # 参数
/// - a, b: 两个 Tensor（可以有 stride=0 的广播维，可以非 contiguous）
/// - out: 输出 buffer（长度必须等于 a.numel() == b.numel()）
/// - op: (a_val, b_val) → out_val 的闭包
///
/// # 泛型约束
/// - D: Backend（Tensor 的设备类型，不影响 elwise_generic 的逻辑，但 Tensor<T,D> 必须有 D）
/// - F: Fn(f32, f32) → f32（二元闭包）
pub fn elwise_generic<D: Backend, F: Fn(f32, f32) -> f32>(
    a: &Tensor<f32, D>,
    b: &Tensor<f32, D>,
    out: &mut [f32],
    op: F,
) {
    assert_eq!(a.numel(), b.numel(), "elwise_generic: a.numel != b.numel");
    assert_eq!(out.len(), a.numel(), "elwise_generic: out.len != numel");

    let n = a.numel();
    let a_engine = IndexEngine::new(a.shape.clone(), &a.strides);
    let b_engine = IndexEngine::new(b.shape.clone(), &b.strides);
    let a_data = a.data_slice();
    let b_data = b.data_slice();

    for flat_idx in 0..n {
        let a_offset = a_engine.offset(flat_idx);
        let b_offset = b_engine.offset(flat_idx);
        let a_val = a_data[a_offset];
        let b_val = b_data[b_offset];
        out[flat_idx] = op(a_val, b_val);
    }
}

/// 一元通用路径（非 contiguous 时用）
pub fn elwise_unary_generic<D: Backend, F: Fn(f32) -> f32>(
    a: &Tensor<f32, D>,
    out: &mut [f32],
    op: F,
) {
    let n = a.numel();
    let a_engine = IndexEngine::new(a.shape.clone(), &a.strides);
    let a_data = a.data_slice();

    for flat_idx in 0..n {
        let a_offset = a_engine.offset(flat_idx);
        let val = a_data[a_offset];
        out[flat_idx] = op(val);
    }
}

// ============================================================
// 宏：生成二元逐元素算子的双路径函数
// ============================================================

/// 生成二元逐元素算子函数的宏
///
/// 为每个算子生成：
/// 1. 广播预处理（preprocess_binary）
/// 2. contiguous 检查（两个输入都 C-contiguous → Backend kernel；否则 → elwise_generic）
/// 3. 构造输出 Tensor
macro_rules! impl_elwise_op {
    ($name:ident, $backend_fn:ident, $impl_fn:expr) => {
        /// 逐元素 $name（完整双路径实现）
        pub fn $name<D: Backend>(
            a: &Tensor<f32, D>, b: &Tensor<f32, D>,
        ) -> Tensor<f32, D> {
            // ---- 第一步：广播预处理 ----
            let (a_b, b_b, _target) = preprocess_binary(a, b)
                .expect(&format!("{}: 广播失败", stringify!($name)));

            // ---- 第二步：双路径 dispatch ----
            // 广播后 stride=0 会让 is_contiguous() 返回 false（因为步长不匹配 C-contiguous）
            // 所以只有"无广播 + 两个都 contiguous"才走 Backend kernel 快速路径
            let both_contiguous = a_b.is_contiguous() && b_b.is_contiguous();

            if both_contiguous {
                // 路径 A：contiguous → Backend kernel（rayon + SIMD）
                let a_c = a_b.contiguous();
                let b_c = b_b.contiguous();
                let numel = a_c.numel();
                let mut out_data = vec![0.0f32; numel];
                D::$backend_fn(
                    a_c.data_ptr(),
                    b_c.data_ptr(),
                    out_data.as_mut_ptr(),
                    numel,
                );
                Tensor::from_data(out_data, a_c.shape.clone()).expect("Tensor 构造失败")
            } else {
                // 路径 B：非 contiguous（含广播）→ elwise_generic + IndexEngine
                let numel = a_b.numel();
                let mut out_data = vec![0.0f32; numel];
                elwise_generic(&a_b, &b_b, &mut out_data, $impl_fn);
                Tensor::from_data(out_data, a_b.shape.clone()).expect("Tensor 构造失败")
            }
        }
    };
}

/// 生成一元逐元素算子函数的宏
macro_rules! impl_elwise_unary {
    ($name:ident, $backend_fn:ident, $impl_fn:expr) => {
        /// 逐元素 $name（一元，完整双路径实现）
        pub fn $name<D: Backend>(
            a: &Tensor<f32, D>,
        ) -> Tensor<f32, D> {
            if a.is_contiguous() {
                // 路径 A：contiguous → Backend kernel
                let a_c = a.contiguous();
                let numel = a_c.numel();
                let mut out_data = vec![0.0f32; numel];
                D::$backend_fn(
                    a_c.data_ptr(),
                    out_data.as_mut_ptr(),
                    numel,
                );
                Tensor::from_data(out_data, a_c.shape.clone()).expect("Tensor 构造失败")
            } else {
                // 路径 B：非 contiguous → elwise_unary_generic + IndexEngine
                let numel = a.numel();
                let mut out_data = vec![0.0f32; numel];
                elwise_unary_generic(a, &mut out_data, $impl_fn);
                Tensor::from_data(out_data, a.shape.clone()).expect("Tensor 构造失败")
            }
        }
    };
}

// ============================================================
// 二元逐元素算子（全部用宏生成）
// ============================================================

impl_elwise_op!(add, elwise_add, |a, b| a + b);
impl_elwise_op!(sub, elwise_sub, |a, b| a - b);
impl_elwise_op!(mul, elwise_mul, |a, b| a * b);
impl_elwise_op!(div, elwise_div, |a, b| a / b);

// ============================================================
// 一元逐元素算子（全部用宏生成）
// ============================================================

impl_elwise_unary!(neg, elwise_neg, |a| -a);
impl_elwise_unary!(abs, elwise_abs, |a| a.abs());
impl_elwise_unary!(exp, elwise_exp, |a| a.exp());
impl_elwise_unary!(log, elwise_log, |a| a.ln());
impl_elwise_unary!(sqrt, elwise_sqrt, |a| a.sqrt());
impl_elwise_unary!(relu, elwise_relu, |a| if a > 0.0 { a } else { 0.0 });
impl_elwise_unary!(gelu, elwise_gelu, |a| a * 0.5 * (1.0 + erf_approx(a / std::f32::consts::SQRT_2)));
impl_elwise_unary!(gelu_fast, elwise_gelu_fast, |a| {
    let inner = (a * 0.7978845608028654) * (1.0 + 0.044715 * a * a);
    0.5 * a * (1.0 + inner.tanh())
});
impl_elwise_unary!(sigmoid, elwise_sigmoid, |a| 1.0 / (1.0 + (-a).exp()));
impl_elwise_unary!(tanh, elwise_tanh, |a| a.tanh());
impl_elwise_unary!(silu, elwise_silu, |a| a / (1.0 + (-a).exp()));

// ============================================================
// 额外算子
// ============================================================

/// 逐元素 a 的 exp 次幂（带标量参数）
pub fn pow<D: Backend>(a: &Tensor<f32, D>, exp: f32) -> Tensor<f32, D> {
    if a.is_contiguous() {
        let a_c = a.contiguous();
        let numel = a_c.numel();
        let mut out_data = vec![0.0f32; numel];
        D::elwise_pow(a_c.data_ptr(), exp, out_data.as_mut_ptr(), numel);
        Tensor::from_data(out_data, a_c.shape.clone()).expect("Tensor 构造失败")
    } else {
        let numel = a.numel();
        let mut out_data = vec![0.0f32; numel];
        let a_engine = IndexEngine::new(a.shape.clone(), &a.strides);
        let a_data = a.data_slice();
        for flat in 0..numel {
            let val = a_data[a_engine.offset(flat)];
            out_data[flat] = val.powf(exp);
        }
        Tensor::from_data(out_data, a.shape.clone()).expect("Tensor 构造失败")
    }
}

/// 逐元素 sin
pub fn sin<D: Backend>(a: &Tensor<f32, D>) -> Tensor<f32, D> {
    if a.is_contiguous() {
        let a_c = a.contiguous();
        let numel = a_c.numel();
        let mut out_data = vec![0.0f32; numel];
        D::elwise_sin(a_c.data_ptr(), out_data.as_mut_ptr(), numel);
        Tensor::from_data(out_data, a_c.shape.clone()).expect("Tensor 构造失败")
    } else {
        let numel = a.numel();
        let mut out_data = vec![0.0f32; numel];
        elwise_unary_generic(a, &mut out_data, |a| a.sin());
        Tensor::from_data(out_data, a.shape.clone()).expect("Tensor 构造失败")
    }
}

/// 逐元素 cos
pub fn cos<D: Backend>(a: &Tensor<f32, D>) -> Tensor<f32, D> {
    if a.is_contiguous() {
        let a_c = a.contiguous();
        let numel = a_c.numel();
        let mut out_data = vec![0.0f32; numel];
        D::elwise_cos(a_c.data_ptr(), out_data.as_mut_ptr(), numel);
        Tensor::from_data(out_data, a_c.shape.clone()).expect("Tensor 构造失败")
    } else {
        let numel = a.numel();
        let mut out_data = vec![0.0f32; numel];
        elwise_unary_generic(a, &mut out_data, |a| a.cos());
        Tensor::from_data(out_data, a.shape.clone()).expect("Tensor 构造失败")
    }
}

/// 填充：创建一个全 value 的 Tensor
pub fn fill<D: Backend>(shape: &DynamicShape, value: f32) -> Tensor<f32, D> {
    let n = shape.numel();
    let data = vec![value; n];
    Tensor::from_data(data, shape.clone()).expect("Tensor 构造失败")
}

/// 广播到相同 shape（直接调用 broadcast 模块的 preprocess_binary，丢弃 target shape）
pub fn broadcast_to_match<T: Clone, D: Backend>(
    a: &Tensor<T, D>, b: &Tensor<T, D>,
) -> Result<(Tensor<T, D>, Tensor<T, D>), String> {
    let (a_b, b_b, _target) = preprocess_binary(a, b)?;
    Ok((a_b, b_b))
}

// ============================================================
// erf 函数（gelu 内部用）
// ============================================================

/// 标准正态分布误差函数（Abramowitz and Stegun 近似，最大误差 1.5e-7）
fn erf_approx(x: f32) -> f32 {
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;
    const P: f32 = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}
