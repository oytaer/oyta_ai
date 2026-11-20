// oyta-device/cuda_kernels/softmax.cu
// 数值稳定：先减 max → exp → 除 sum（block 内共享内存归约）

#include <cuda_runtime.h>

// 每个 block 处理一行（outer 个 block，blockDim.x 线程归约）
__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int outer, int inner) {
    const int row = blockIdx.x;
    if (row >= outer) return;

    extern __shared__ float smem[];   // 动态共享内存
    float* s_max = smem;              // 前半：max 值
    float* s_sum = smem + blockDim.x; // 后半：sum 值

    const float* row_in = input + (long long)row * inner;
    float* row_out = output + (long long)row * inner;

    // Step 1: block 内并行找 max
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    s_max[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = s_max[0];

    // Step 2: 计算 exp(x-max) 并求和
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        local_sum += expf(row_in[i] - row_max);
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            s_sum[threadIdx.x] += s_sum[threadIdx.x] + s_sum[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = s_sum[0];
    float inv_sum = 1.0f / row_sum;

    // Step 3: 归一化写回
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
    }
}

extern "C" void launch_softmax_kernel(
    float* output, const float* input, int outer, int inner, cudaStream_t stream
) {
    int bs = min(inner, 1024);
    if (bs < 32) bs = 32;  // warp 对齐
    size_t smem_bytes = bs * 2 * sizeof(float);
    softmax_kernel<<<outer, bs, smem_bytes, stream>>>(input, output, outer, inner);
}
