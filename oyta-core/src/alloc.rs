//! alloc.rs：自定义内存分配器包装
//!
//! §3.1 目录结构 → alloc.rs
//! §5.4 内存对齐要求
//!
//! 设计：
//! - Tensor.from_data 用 Vec<T> 直接分配（jemalloc/tcmalloc 默认对齐 16 字节）
//! - AVX2 需要 32 字节对齐 → 用 posix_memalign / _aligned_malloc
//! - AVX-512 需要 64 字节对齐
//! - 本版简化：假设 jemalloc 默认对齐到 32 字节（现代系统已满足）
//!   如果未对齐，算子内部用 loadu 替代 load（只慢一点点）
//!
//! 预留：未来可以启用 #[feature="aligned_allocator"] 切换到严格对齐

#![allow(dead_code)] // 预留代码，当前没启用

/// 对齐分配要求（字节）
pub const ALIGN_AVX2: usize = 32;      // AVX2: 32 字节对齐
pub const ALIGN_AVX512: usize = 64;    // AVX-512: 64 字节对齐
pub const ALIGN_NEON: usize = 16;      // NEON: 16 字节对齐

/// 当前系统推荐对齐（取最大值以兼容所有 SIMD）
pub const ALIGN_DEFAULT: usize = ALIGN_AVX512;

/// 对齐分配器（预留，当前用系统默认）
/// 在 release + 特定 feature gate 下可以切换到严格对齐
pub struct AlignedAlloc;

impl AlignedAlloc {
    /// 判断指针是否对齐到指定字节数
    pub fn is_aligned<T>(ptr: *const T, align: usize) -> bool {
        (ptr as usize) % align == 0
    }

    /// 返回推荐对齐值
    pub fn recommended_alignment() -> usize { ALIGN_DEFAULT }
}
