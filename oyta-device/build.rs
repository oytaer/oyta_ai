// oyta-device/build.rs
// 只有当启用 cuda feature 且系统有 nvcc 时才编译 .cu kernels
// 否则自动跳过（libcuda.so/cublas.so 由 runtime linker 提供）

fn main() {
    #[cfg(feature = "cuda")]
    {
        use std::path::PathBuf;

        // 强制 nvcc 用 GCC 13（Ubuntu 26.04 默认 GCC 15 不兼容 CUDA 12.5）
        // cc crate 会读 CXX 环境变量传给 nvcc 的 -ccbin
        unsafe {
            std::env::set_var("CXX", "g++-13");
            std::env::set_var("CC", "gcc-13");
        }

        // 检查 nvcc 是否存在
        let nvcc = match std::process::Command::new("nvcc").arg("--version").output() {
            Ok(o) if o.status.success() => true,
            _ => false,
        };

        if nvcc {
            let dir = PathBuf::from("cuda_kernels");
            let mut build = cc::Build::new();
            build.cuda(true);
            build.include(&dir);

            for k in &[
                "elwise_add.cu",
                "softmax.cu",
                "layernorm.cu",
                "matmul.cu",
                "flash_attn.cu",
                "conv1d.cu",
                "batchnorm.cu",
                "pool1d.cu",
                "rnn.cu",
            ] {
                build.file(dir.join(k));
            }

            build
                .flag("-O3")                                    // 最高优化
                .flag("--use_fast_math")                        // 快速数学函数
                .flag("-std=c++17")                             // C++17 标准
                .compile("oyta_device_cuda_kernels");           // 静态库名

            println!("cargo:rerun-if-changed=cuda_kernels");
            println!("cargo:warning=CUDA kernels compiled with nvcc");
        } else {
            println!("cargo:warning=nvcc not found, CUDA kernels skipped (will use CPU fallback)");
        }
    }
}
