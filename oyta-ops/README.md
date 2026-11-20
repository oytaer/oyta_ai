# oyta-ops

Operator engine for oyta-ai.

## Provides

- Matmul dispatch — naive → tiled (128×128) → BLAS → Strassen recursive (crossover=512)
- FlashAttention — online f64 softmax + causal mask (消灭 O(seq²) 中间张量)
- 4 fusion ops — LayerNormGELU / LinearBiasGELU / RMSNormSwiglu / MatmulActivation
- pairwise_sum — 二分递归归约，比 naive sum 精度高 2-3 个数量级
- profiling — 256 槽 AtomicU64 无锁性能计数器

## Quick Start

```toml
[dependencies]
oyta-ops = "0.1.1"
```

## Links

- Repository: https://github.com/oytaer/oyta_ai
