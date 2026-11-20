// oyta-device/cuda_kernels/layernorm.cu
// LayerNorm：mean/var 用 f64 累加器（更稳定）
// gamma/beta 长度 = inner

#include <cuda_runtime.h>

__global__ void layernorm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int outer, int inner, float eps
) {
    const int row = blockIdx.x;
    if (row >= outer) return;

    const float* row_in = input + (long long)row * inner;
    float* row_out = output + (long long)row * inner;

    // f64 累加器保证数值稳定
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        double v = (double)row_in[i];
        sum += v;
        sum_sq += v * v;
    }

    // 用 block 归约（共享内存）
    extern __shared__ double smem[];
    double* s_sum = smem;
    double* s_sq = smem + blockDim.x;
    s_sum[threadIdx.x] = sum;
    s_sq[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sq[threadIdx.x] += s_sq[threadIdx.x + s];
        }
        __syncthreads();
    }
    double row_sum = s_sum[0];
    double row_sum_sq = s_sq[0];
    double mean = row_sum / (double)inner;
    double var = row_sum_sq / (double)inner - mean * mean;
    double inv_std = 1.0 / sqrt(var + (double)eps);

    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        double normed = ((double)row_in[i] - mean) * inv_std;
        row_out[i] = (float)(normed * (double)gamma[i] + (double)beta[i]);
    }
}

extern "C" void launch_layernorm_kernel(
    float* output, const float* input,
    const float* gamma, const float* beta,
    int outer, int inner, float eps, cudaStream_t stream
) {
    int bs = min(inner, 1024);
    if (bs < 32) bs = 32;
    size_t smem_bytes = (size_t)bs * 2 * sizeof(double);
    layernorm_kernel<<<outer, bs, smem_bytes, stream>>>(
        input, output, gamma, beta, outer, inner, eps
    );
}
