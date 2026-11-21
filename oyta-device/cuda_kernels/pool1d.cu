// oyta-device/cuda_kernels/pool1d.cu
// MaxPool1d + AvgPool1d forward/backward

#include <cuda_runtime.h>
#include <cfloat>

// MaxPool1d forward — 每个 thread 算一个 out 位置
__global__ void maxpool1d_fwd_kernel(
    const float* __restrict__ x,  // [N, C, L]
    float* __restrict__ out,      // [N, C, L_out]
    int64_t* __restrict__ argmax, // [N, C, L_out]
    int C, int L_in, int L_out, int K, int stride, int padding
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nco = C * L_out;
    const int NC = (int)gridDim.y;  // N 由 gridDim.y 编码
    // 简化：直接算 index
    // idx 已经是平坦索引 [N, C, L_out]
    const int co = (idx / L_out) % C;
    const int lo = idx % L_out;
    const int n = idx / (C * L_out);
    (void)NC;  // 避免 unused

    int start = lo * stride - padding;
    int end = lo * stride - padding + K;
    start = max(start, 0);
    end = min(end, L_in);

    float max_val = -FLT_MAX;
    int64_t max_idx = -1;
    for (int ki = start; ki < end; ki++) {
        float v = x[(n * C + co) * L_in + ki];
        if (v > max_val) { max_val = v; max_idx = ki; }
    }
    if (max_idx < 0) { max_val = -FLT_MAX; max_idx = 0; }
    out[idx] = max_val;
    argmax[idx] = max_idx;
}

// AvgPool1d forward
__global__ void avgpool1d_fwd_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int C, int L_in, int L_out, int K, int stride, int padding
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int co = (idx / L_out) % C;
    const int lo = idx % L_out;
    const int n = idx / (C * L_out);

    int start = lo * stride - padding;
    int end = lo * stride - padding + K;
    start = max(start, 0);
    end = min(end, L_in);
    int count = end - start;
    if (count <= 0) count = 1;

    float sum = 0.0f;
    for (int ki = start; ki < end; ki++) {
        sum += x[(n * C + co) * L_in + ki];
    }
    out[idx] = sum / count;
}

// MaxPool1d backward — 根据 argmax 把梯度稀疏传回
__global__ void maxpool1d_bwd_kernel(
    const float* __restrict__ dy,     // [N, C, L_out]
    const int64_t* __restrict__ argmax,
    float* __restrict__ dx,           // [N, C, L_in] 预先清零
    int C, int L_out, int L_in
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * L_out) return;

    const int co = (idx / L_out) % C;
    const int lo = idx % L_out;
    const int n = idx / (C * L_out);

    int64_t argmax_idx = argmax[idx];
    if (argmax_idx >= 0) {
        atomicAdd(&dx[(n * C + co) * L_in + (int)argmax_idx], dy[idx]);
    }
}

// AvgPool1d backward — 均匀分配梯度
__global__ void avgpool1d_bwd_kernel(
    const float* __restrict__ dy,
    float* __restrict__ dx,  // 预先清零
    int C, int L_out, int L_in, int K, int stride, int padding
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * L_out) return;

    const int co = (idx / L_out) % C;
    const int lo = idx % L_out;
    const int n = idx / (C * L_out);

    int start = lo * stride - padding;
    int end = lo * stride - padding + K;
    start = max(start, 0);
    end = min(end, L_in);
    int count = max(end - start, 1);
    float delta = dy[idx] / count;

    for (int ki = start; ki < end; ki++) {
        atomicAdd(&dx[(n * C + co) * L_in + ki], delta);
    }
}

// Host wrappers
extern "C" void oyta_maxpool1d_fwd(
    const float* x, float* out, int64_t* argmax,
    int N, int C, int L_in, int L_out, int K, int stride, int padding
) {
    int total = N * C * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    maxpool1d_fwd_kernel<<<blocks, threads>>>(
        x, out, argmax, C, L_in, L_out, K, stride, padding
    );
}

extern "C" void oyta_avgpool1d_fwd(
    const float* x, float* out,
    int N, int C, int L_in, int L_out, int K, int stride, int padding
) {
    int total = N * C * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    avgpool1d_fwd_kernel<<<blocks, threads>>>(
        x, out, C, L_in, L_out, K, stride, padding
    );
}

extern "C" void oyta_maxpool1d_bwd(
    const float* dy, const int64_t* argmax, float* dx,
    int N, int C, int L_out, int L_in
) {
    cudaMemset(dx, 0, (size_t)N * C * L_in * sizeof(float));
    int total = N * C * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    maxpool1d_bwd_kernel<<<blocks, threads>>>(
        dy, argmax, dx, C, L_out, L_in
    );
}

extern "C" void oyta_avgpool1d_bwd(
    const float* dy, float* dx,
    int N, int C, int L_out, int L_in, int K, int stride, int padding
) {
    cudaMemset(dx, 0, (size_t)N * C * L_in * sizeof(float));
    int total = N * C * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    avgpool1d_bwd_kernel<<<blocks, threads>>>(
        dy, dx, C, L_out, L_in, K, stride, padding
    );
}
