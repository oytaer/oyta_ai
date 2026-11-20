//! base.rs：Tensor<T, D> 核心结构
//!
//! 设计：
//!   - Arc<Vec<T>> 数据共享，Clone 零拷贝（只增 Arc 引用计数）
//!   - shape + strides + offset 描述视图，transpose/reshape/permute 都是零拷贝
//!   - contiguous() 才真实拷贝到新 Arc<Vec<T>>
//!   - PhantomData<D> 标记设备，Backend trait 抽象算子调用

use std::sync::Arc; // 线程安全引用计数（数据共享）
use std::marker::PhantomData; // 零成本类型标记
use std::fmt; // Debug trait

use oyta_device::Backend; // 后端算子 trait

use crate::shape::DynamicShape; // 动态形状
use crate::strides::Strides; // 步长
use crate::error::TensorError; // 错误类型

// ============================================================
// Tensor<T, D> struct
// ============================================================

/// 通用多维数组
///
/// T: 元素类型（f32/f64/F16/BF16）
/// D: 后端设备标记（Cpu/Cuda/Metal/Wgpu）
#[derive(Clone)] // Arc + PhantomData 都是 Clone，自动派生
pub struct Tensor<T, D: Backend> {
    /// 共享底层数据（Arc 保证线程安全 + Clone 零拷贝）
    pub data: Arc<Vec<T>>,
    /// 逻辑形状
    pub shape: DynamicShape,
    /// 步长向量（支持非 contiguous view）
    pub strides: Strides,
    /// flat 数组起始偏移（切片 view 用）
    pub offset: usize,
    /// 设备标记（零开销泛型标记）
    pub _device: PhantomData<D>,
}

// ============================================================
// Tensor 基础实现（Clone 需求）
// ============================================================

impl<T: Clone, D: Backend> Tensor<T, D> {
    // -- 构造 --

    /// 从 Vec + Shape 创建（最常用入口）
    pub fn from_data(data: Vec<T>, shape: impl Into<DynamicShape>) -> Result<Self, TensorError> {
        let shape = shape.into();
        let expected = shape.numel(); // 期望元素数
        if data.len() != expected { // 实际不匹配
            return Err(TensorError::ShapeMismatch {
                expected: vec![expected],
                actual: vec![data.len()],
            });
        }
        let strides = shape.c_contiguous_strides(); // 行优先步长
        Ok(Self {
            data: Arc::new(data), // 包进 Arc
            shape,
            strides,
            offset: 0,
            _device: PhantomData,
        })
    }

    /// 从 Arc 共享数据创建（public，给 oyta-ops 构造广播 Tensor 用）
    pub fn from_shared(data: Arc<Vec<T>>, shape: DynamicShape, strides: Strides, offset: usize) -> Self {
        Self { data, shape, strides, offset, _device: PhantomData }
    }

    // -- 属性访问 --

    /// 形状引用
    pub fn shape(&self) -> &DynamicShape { &self.shape }
    /// 步长引用
    pub fn strides(&self) -> &Strides { &self.strides }
    /// 元素总数
    pub fn numel(&self) -> usize { self.shape.numel() }
    /// 维度数
    pub fn rank(&self) -> usize { self.shape.rank() }
    /// flat 只读切片（自动按 offset 截取）
    pub fn data_slice(&self) -> &[T] { &self.data[self.offset .. self.offset + self.numel()] }

    /// 是否 C-contiguous（offset=0 且 strides 匹配行优先）
    pub fn is_contiguous(&self) -> bool {
        if self.offset != 0 { return false; } // 有偏移 → 不连续
        self.strides == self.shape.c_contiguous_strides() // 步长匹配行优先
    }

    /// 是否标量（rank=0 或 numel=1）
    pub fn is_scalar(&self) -> bool { self.numel() == 1 }

    // -- 指针/标量访问 --

    /// 底层数据裸指针（配合 Backend 算子 unsafe 调用）
    /// 指向 self.data[self.offset]，保证对齐
    /// 调用者必须保证 offset + numel <= data.len()
    pub fn data_ptr(&self) -> *const T {
        // 调用者必须保证 offset <= data.len() 且对齐
        unsafe { self.data.as_ptr().add(self.offset) } // base + offset
    }

    /// 转为 Vec（拷贝到新 Vec，所有权转移）
    pub fn to_vec(&self) -> Vec<T> where T: Copy {
        self.data_slice().to_vec() // 从切片拷贝
    }

    /// 转为标量值（rank=0 或 numel=1）
    pub fn item(&self) -> T where T: Copy {
        #[cfg(debug_assertions)]
        debug_assert!(self.numel() == 1, "item() 只能用于标量或单元素 Tensor");
        self.data_slice()[0]
    }
}

// ============================================================
// raw bytes 序列化（委托给 bytes.rs 模块）
// ============================================================

impl<T: Copy + Clone + 'static, D: Backend> Tensor<T, D> {
    /// 把 Tensor 的底层数据序列化成原始小端字节
    pub fn to_raw_bytes(&self) -> Vec<u8> {
        let s = self.data_slice();
        let byte_len = crate::checked::checked_mul(s.len(), std::mem::size_of::<T>())
            .unwrap_or(usize::MAX);
        unsafe {
            std::slice::from_raw_parts(
                s.as_ptr() as *const u8,
                byte_len,
            ).to_vec()
        }
    }

    /// 从原始小端字节构造 Tensor
    pub fn from_raw_bytes(bytes: &[u8], shape: DynamicShape) -> Result<Self, TensorError> {
        let numel = shape.numel();
        let expected_len = crate::checked::checked_mul(numel, std::mem::size_of::<T>())?;
        if bytes.len() != expected_len {
            return Err(TensorError::Unsupported(format!(
                "bytes 长度 {} ≠ numel*sizeof(T)={}",
                bytes.len(),
                expected_len
            )));
        }
        let data: Vec<T> = unsafe {
            let src = bytes.as_ptr() as *const T;
            std::slice::from_raw_parts(src, numel).to_vec()
        };
        Tensor::from_data(data, shape)
    }
}

// ============================================================
// Debug Display
// ============================================================

impl<T: fmt::Debug, D: Backend> fmt::Debug for Tensor<T, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor<{}, {}>(shape={:?})", std::any::type_name::<T>(), D::DEVICE_NAME, self.shape.dims)
    }
}
