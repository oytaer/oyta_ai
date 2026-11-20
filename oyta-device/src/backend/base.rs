//! backend_base.rs：Backend 底层工具算子（fill/copy/where_bool）

// ============================================================
// BackendBase：最底层工具 trait（不继承任何设备 trait）
// ============================================================

/// BackendBase：最底层的内存/数据工具
///
/// 职责：fill / copy / where_bool 三个最基础操作。
/// 是所有子 trait 的基础，不继承任何设备 trait。
pub trait BackendBase: Copy + Clone + Send + Sync + 'static {
    /// 用 value 填充 dst[0..n]
    /// 调用者必须保证 dst 指向至少 n 个 f32 的有效可写内存
    fn fill(dst: *mut f32, n: usize, value: f32) {
        unsafe {
            let slice = std::slice::from_raw_parts_mut(dst, n);
            slice.iter_mut().for_each(|v| *v = value);
        }
    }
    
    /// 从 src[0..n] 拷贝到 dst[0..n]
    /// 调用者必须保证 src 和 dst 各自指向至少 n 个 f32 的有效内存，且不重叠
    fn copy(src: *const f32, dst: *mut f32, n: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, n);
        }
    }
    
    /// 条件选择：out[i] = cond[i] ? a[i] : b[i]
    /// 调用者必须保证 cond/a/b/out 各自指向至少 n 个元素的有效内存
    fn where_bool(cond: *const bool, a: *const f32, b: *const f32, out: *mut f32, n: usize) {
        unsafe {
            let cond_s = std::slice::from_raw_parts(cond, n);
            let a_s = std::slice::from_raw_parts(a, n);
            let b_s = std::slice::from_raw_parts(b, n);
            let out_s = std::slice::from_raw_parts_mut(out, n);
            out_s.iter_mut().zip(cond_s.iter()).zip(a_s.iter()).zip(b_s.iter())
                .for_each(|(((o, &c), &av), &bv)| {
                    *o = if c { av } else { bv };
                });
        }
    }
}
