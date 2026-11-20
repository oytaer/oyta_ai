# oyta-tensor

**用户只需依赖这一个 crate**。re-export `oyta-device` + `oyta-core` + `oyta-ops`，提供 ergonomic 的 `Tensor<f32, Cpu>` 风格 API。

## Quick Start

```toml
[dependencies]
oyta-tensor = "0.1.1"
```

```rust
use oyta_tensor::prelude::*;

// 创建张量
let a = Tensor::<f32, Cpu>::randn([32, 512]);
let b = Tensor::<f32, Cpu>::randn([512, 768]);

// matmul（自动 dispatch naive→tiled→BLAS→Strassen）
let c = a.matmul(&b).unwrap();

// LayerNorm + GELU（融合算子）
let gamma = Tensor::<f32, Cpu>::ones([768]);
let beta  = Tensor::<f32, Cpu>::zeros([768]);
let out   = c.layer_norm_gelu(&gamma, &beta, 1e-5).unwrap();

// FlashAttention v2
let q = Tensor::<f32, Cpu>::randn([4096, 64]);
let k = Tensor::<f32, Cpu>::randn([4096, 64]);
let v = Tensor::<f32, Cpu>::randn([4096, 64]);
let fa = q.flash_attention(&k, &v, true).unwrap();
```

## Feature Flags

```toml
oyta-tensor = { version = "0.1.1", features = ["cuda", "wgpu", "blas"] }
```

| Feature | 默认 | 说明 |
|---|---|---|
| `cpu` | ✅ | CPU rayon 并行 |
| `cuda` | ❌ | NVIDIA CUDA + cuBLAS |
| `wgpu` | ❌ | WebGPU 跨平台 |
| `metal` | ❌ | Apple Metal |
| `blas` | ❌ | OpenBLAS/MKL |
| `debug_assertions` | ❌ | 边界检查（release 关） |

## 跨设备

```rust
let gpu = a.to_wgpu();
let result = gpu.matmul(&b.to_wgpu()).unwrap();
```

## Links

- Repository: https://github.com/oytaer/oyta_ai
- Full documentation: [docs/](https://github.com/oytaer/oyta_ai/tree/main/docs)
- Parent workspace: oyta-device + oyta-core + oyta-ops
