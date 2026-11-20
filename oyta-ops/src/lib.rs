//! oyta-ops：算子引擎
//!
//! 目录结构：
//!   ├── profiling.rs   Profiling 统计
//!   ├── dispatch.rs    调度器
//!   ├── fuse.rs        融合引擎
//!   ├── dropout.rs     Dropout 正则化
//!   ├── broadcast.rs   广播预处理
//!   ├── matmul/        矩阵乘法（strategy.rs）
//!   ├── elwise/        逐元素算子（generic.rs）
//!   ├── reduce/        规约算子（fns.rs）
//!   ├── attention/     FlashAttention（flash.rs）
//!   └── norm/          LayerNorm + RMSNorm（layernorm.rs）

// ============================================================
// 子目录模块声明
// ============================================================
pub mod matmul;     // matmul/mod.rs → matmul/strategy.rs
pub mod elwise;     // elwise/mod.rs → elwise/generic.rs
pub mod reduce;     // reduce/mod.rs → reduce/fns.rs
pub mod attention;  // attention/mod.rs → attention/flash.rs
pub mod norm;       // norm/mod.rs → norm/layernorm.rs

// ============================================================
// 单文件模块（保持根目录）
// ============================================================
pub mod profiling;
pub mod dispatch;
pub mod fuse;
pub mod dropout;
pub mod broadcast;

// ============================================================
// 模块别名（保持外部 API 零变化）
// ============================================================

// flash_attn → attention::flash 的 re-export 别名
pub mod flash_attn {
    pub use crate::attention::flash::*;
}

// layernorm → norm::layernorm 的 re-export 别名
pub mod layernorm {
    pub use crate::norm::layernorm::*;
}

// ============================================================
// Profiling 重导出
// ============================================================
pub use profiling::{
    profile, set_profiling_enabled, is_profiling_enabled,
    print_profile_report, reset_profile, op_index, ProfileGuard,
    MAX_OPS, OP_NAMES, OpStatsSnapshot, snapshot,
};

// ============================================================
// Dispatch 重导出
// ============================================================
pub use dispatch::{
    select_matmul_strategy, select_reduce_strategy, broadcast_target_shape,
    MatmulStrategy, ReduceStrategy,
};

// ============================================================
// 融合引擎重导出
// ============================================================
pub use fuse::{FusedOp, should_fuse, dispatch_fused, op_kind_name};

// ============================================================
// Dropout 训练态 API
// ============================================================
pub use dropout::{set_training_mode, is_training_mode};

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use oyta_core::Tensor;
    use oyta_device::{Cpu, MatmulOps};

    #[test]
    fn test_profiling_simple() {
        set_profiling_enabled(true);
        reset_profile();
        let _g = profile("matmul");
        let a = Tensor::<f32, Cpu>::randn([64, 64]);
        let _ = a.matmul(&a.clone()).unwrap();
        drop(_g);
        print_profile_report();
        set_profiling_enabled(false);
    }

    #[test]
    fn test_reset_works() {
        reset_profile();
        set_profiling_enabled(true);
        reset_profile();
        let _g = profile("matmul");
        drop(_g);
        reset_profile();
        set_profiling_enabled(false);
    }

    #[test]
    fn test_op_index_basic() {
        assert_eq!(op_index("matmul"), 0);
        assert_eq!(op_index("elwise_add"), 5);
        assert_eq!(op_index("softmax"), 33);
        assert_eq!(op_index("nonexistent_op_xyz"), 255);
    }

    #[test]
    fn test_dtype_cast_roundtrip() {
        let a = Tensor::<f32, Cpu>::randn([32, 32]);
        let f16_t = a.to_f16();
        let f32_back = f16_t.to_f32();
        assert_eq!(a.shape().numel(), f16_t.shape().numel());
        assert_eq!(a.shape().numel(), f32_back.shape().numel());
        for (orig, back) in a.data_slice().iter().zip(f32_back.data_slice().iter()) {
            let diff = (orig - back).abs();
            let tol = (orig.abs() * 0.001).max(0.01);
            assert!(diff <= tol, "f16 roundtrip 偏差太大");
        }
        let bf16_t = a.to_bf16();
        let f32_bf_back = bf16_t.to_f32();
        for (orig, back) in a.data_slice().iter().zip(f32_bf_back.data_slice().iter()) {
            let diff = (orig - back).abs();
            let tol = (orig.abs() * 0.01).max(0.02);
            assert!(diff <= tol, "bf16 roundtrip 偏差太大");
        }
        let f64_t = a.to_f64();
        let f32_back2 = f64_t.to_f32();
        for (orig, back) in a.data_slice().iter().zip(f32_back2.data_slice().iter()) {
            assert_eq!(orig, back, "f64 roundtrip 应该精确");
        }
    }

    #[test]
    fn test_device_context_cpu() {
        use oyta_device::DeviceContext;
        let handle = Cpu::get();
        assert!(handle.properties.name.contains("CPU"));
        assert_eq!(handle.properties.compute_capability, "Host");
        assert!(!handle.properties.supports_fp16);
        assert!(!handle.properties.supports_tensor_core);
        let handle2 = Cpu::get();
        assert_eq!(handle.properties.name, handle2.properties.name);
    }

    #[test]
    fn test_tensor_to_device_transfer() {
        let a = Tensor::<f32, Cpu>::randn([16, 16]);
        let b = a.to_cpu();
        assert_eq!(a.shape().numel(), b.shape().numel());
        let c: Tensor::<f32, Cpu> = a.to_device::<Cpu>();
        assert_eq!(a.shape().numel(), c.shape().numel());
    }

    #[test]
    fn test_anybackend_runtime_switch() {
        use oyta_device::{DeviceRuntimeId, set_runtime_device, get_runtime_device};
        assert_eq!(get_runtime_device(), DeviceRuntimeId::Cpu);
        set_runtime_device(DeviceRuntimeId::Cpu);
        assert_eq!(get_runtime_device(), DeviceRuntimeId::Cpu);
    }

    #[test]
    fn test_matmul_tiled_vs_naive_correctness() {
        let a = Tensor::<f32, Cpu>::randn([16, 16]);
        let b = Tensor::<f32, Cpu>::randn([16, 16]);
        let c = a.matmul(&b).unwrap();
        let expected = {
            let mut e = vec![0.0f32; 16 * 16];
            let as_ = a.data_slice(); let bs_ = b.data_slice();
            for i in 0..16 { for j in 0..16 { for p in 0..16 { e[i*16+j] += as_[i*16+p] * bs_[p*16+j]; } } }
            e
        };
        for (got, exp) in c.data_slice().iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-3, "naive matmul 偏差太大: {} vs {}", got, exp);
        }
        let a64 = Tensor::<f32, Cpu>::randn([64, 64]);
        let b64 = Tensor::<f32, Cpu>::randn([64, 64]);
        let c64 = a64.matmul(&b64).unwrap();
        let expected64 = {
            let mut e = vec![0.0f32; 64 * 64];
            let as_ = a64.data_slice(); let bs_ = b64.data_slice();
            for i in 0..64 { for j in 0..64 { for p in 0..64 { e[i*64+j] += as_[i*64+p] * bs_[p*64+j]; } } }
            e
        };
        let mut max_diff = 0.0f32;
        for (got, exp) in c64.data_slice().iter().zip(expected64.iter()) {
            max_diff = max_diff.max((got - exp).abs());
        }
        assert!(max_diff < 1e-2, "tiled matmul 最大偏差: {}", max_diff);
    }

    #[test]
    fn test_matmul_non_contiguous_ldc() {
        let a = Tensor::<f32, Cpu>::zeros([4, 4]);
        let a_d = a.data_slice();
        let b_d = a.data_slice();
        let mut c_buf = vec![0.0f32; 16 * 2];
        <Cpu as MatmulOps>::matmul(
            a_d.as_ptr(), 4, b_d.as_ptr(), 4,
            c_buf.as_mut_ptr(), 8,
            4, 4, 4, 1.0, 0.0,
        );
        for i in 0..4 {
            for j in 4..8 { assert_eq!(c_buf[i * 8 + j], 0.0, "非 contiguous ldc 填充应保持 0"); }
        }
    }
}
