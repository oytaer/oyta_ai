// oyta-device/cuda_kernels/flash_attn.cu
// 简化 Flash Attention 前向：Q[B,H,S,D] · K^T · softmax · V
// 单 head 内部 block 并行（非 online softmax 分块，但走 flash 的 tile 套路）
// D 固定 <= 128；每个 block 负责一个 (batch, head) 的 row

#include <cuda_runtime.h>

// 每个 block 处理一个 query 行（一个 (b, h) 的 full attention）
// 对于 S 较大的情况，可分块 tile K/V 做 online softmax
// 此处给出 naive-flash 混合版：blockDim.x 线程并行处理 K 维度
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,   // [B, H, S, D]
    const float* __restrict__ K,   // [B, H, S, D]
    const float* __restrict__ V,   // [B, H, S, D]
    float* __restrict__ O,         // [B, H, S, D]
    int B, int H, int S, int D,
    float scale                     // 通常 = 1/sqrt(D)
) {
    const int bh = blockIdx.x;       // batch * H
    const int s  = blockIdx.y;       // query position
    const int tid = threadIdx.x;

    if (bh >= B * H || s >= S) return;

    const int stride = S * D;
    const float* qRow = Q + (long long)bh * stride + s * D;
    const float* kBase = K + (long long)bh * stride;
    const float* vBase = V + (long long)bh * stride;

    // Step 1: 计算 q·k^T（长度 S），block 内并行 reduce
    extern __shared__ float smem[];
    float* s_scores = smem;                // 空间 S（近似 blockDim.x 归约）
    float* s_max    = smem + S;
    float* s_sum    = s_max + blockDim.x;

    // 每个线程算一段 score
    float local_max = -INFINITY;
    for (int kPos = tid; kPos < S; kPos += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot = fmaf(qRow[d], kBase[kPos * D + d], dot);
        }
        dot *= scale;
        s_scores[kPos] = dot;
        local_max = fmaxf(local_max, dot);
    }
    s_max[tid] = local_max;
    __syncthreads();

    // block 内归约 max
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step)
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + step]);
        __syncthreads();
    }
    float row_max = s_max[0];

    // Step 2: exp + sum
    float local_sum = 0.0f;
    for (int kPos = tid; kPos < S; kPos += blockDim.x) {
        s_scores[kPos] = expf(s_scores[kPos] - row_max);
        local_sum += s_scores[kPos];
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step)
            s_sum[tid] += s_sum[tid + step];
        __syncthreads();
    }
    float inv_sum = 1.0f / s_sum[0];

    // Step 3: O = softmax_scores · V，每个线程负责一个 d
    float* oRow = O + (long long)bh * stride + s * D;
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int kPos = 0; kPos < S; kPos++) {
            acc = fmaf(s_scores[kPos] * inv_sum, vBase[kPos * D + d], acc);
        }
        oRow[d] = acc;
    }
}

extern "C" void launch_flash_attn_fwd_kernel(
    float* O, const float* Q, const float* K, const float* V,
    int B, int H, int S, int D, float scale, cudaStream_t stream
) {
    int bs = 256;
    if (D > bs) bs = D;
    if (bs > 1024) bs = 1024;

    // smem: S scores + blockDim max + blockDim sum
    size_t smem_bytes = (size_t)S * sizeof(float)
                      + (size_t)bs * 2 * sizeof(float);

    dim3 grid(B * H, S);
    flash_attn_fwd_kernel<<<grid, bs, smem_bytes, stream>>>(
        Q, K, V, O, B, H, S, D, scale
    );
}
