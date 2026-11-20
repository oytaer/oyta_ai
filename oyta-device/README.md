# oyta-device

Generic backend abstraction layer for oyta-ai.

One blanket impl covers **all dtypes** (f32 / f16 / bf16 / f64) across
**all devices** (CPU / CUDA / WGPU / Metal).

## Features

- `ElwiseOps<T>` — 21 generic element-wise ops, one trait for all dtype
- `ReduceOps<T>` — sum / mean / max / min / softmax / layernorm / rmsnorm
- `MatmulOps` — auto dispatch naive → tiled → BLAS → Strassen
- `FlashAttentionOps<T>` — online softmax + causal mask, O(seq·d) memory
- `FusedOps<T>` — LayerNormGELU / LinearBiasGELU / RMSNormSwiglu / MatmulActivation
- `InPlaceOps<T>` — 5 in-place ops

## Quick Start

```toml
[dependencies]
oyta-device = "0.1.1"
```

```rust
use oyta_device::prelude::*;

// blanket impl: one line covers ALL dtypes
impl<T: Float> ElwiseOps<T> for Cpu {}
impl<T: Float + Accumulate> ReduceOps<T> for Cpu {}
impl<T: Float> FlashAttentionOps<T> for Cpu {}
impl<T: Float + Accumulate> FusedOps<T> for Cpu {}
impl<T: Float> InPlaceOps<T> for Cpu {}

// automatic dtype-specific code at compile-time:
// T=f32  → identity (zero overhead)
// T=F16  → to_f32() + op + from_f32() (IEEE754 decode)
// T=BF16 → same as F16
```

## Links

- Repository: https://github.com/oytaer/oyta_ai
- Parent crate (user-facing API): [oyta-tensor](https://crates.io/crates/oyta-tensor)
