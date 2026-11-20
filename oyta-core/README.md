# oyta-core

Tensor core data structures for oyta-ai.

## Provides

- `Tensor<T, D>` — generic Tensor with Arc<Vec<T>> storage
- `DynamicShape` / `Strides` / `IndexEngine` — shape and stride management
- `Device` marker types: `Cpu`, `Cuda`, `Wgpu`, `Metal`
- `TensorError` — thiserror-based error enum
- `safetensors` — HuggingFace format load/save

## Quick Start

```toml
[dependencies]
oyta-core = "0.1.1"
```

```rust
use oyta_core::{Tensor, Cpu};

let t = Tensor::<f32, Cpu>::randn([32, 512]);
println!("shape: {:?}, strides: {:?}", t.shape(), t.strides());
```

## Links

- Repository: https://github.com/oytaer/oyta_ai
