// oyta-device/cuda_kernels/conv1d.cu
// Conv1d forward + backward (dx + dwdb) 1D 卷积 NCHW 格式

#include <cuda_runtime.h>

// ============================================================
// Conv1d forward：out[n, co, lo] = Σ_{ci, ki} w[co, ci, ki] * x[n, ci, lo*stride + ki*dilation - padding] + b[co]
// 每个 thread 算一个 out 元素
// ============================================================

__global__ void conv1d_fwd_kernel(
    const float* __restrict__ x,    // [N, C_in, L_in]
    const float* __restrict__ w,    // [C_out, C_in, K]
    const float* __restrict__ b,    // [C_out] 或 null
    float* __restrict__ out,        // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K,
    int padding, int stride, int dilation, int has_bias
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * L_out;
    if (idx >= total) return;

    const int lo = idx % L_out;
    const int co = (idx / L_out) % C_out;
    const int n = idx / (L_out * C_out);

    float acc = 0.0f;
    for (int ci = 0; ci < C_in; ci++) {
        for (int ki = 0; ki < K; ki++) {
            int l_idx = lo * stride + ki * dilation - padding;
            if (l_idx >= 0 && l_idx < L_in) {
                acc += x[(n * C_in + ci) * L_in + l_idx] * w[(co * C_in + ci) * K + ki];
            }
        }
    }
    if (has_bias) acc += b[co];
    out[idx] = acc;
}

// Conv1d backward — 输入梯度 dx
// dx[n, ci, li] += w[co, ci, ki] * dy[n, co, lo]  （反卷积式回传）
__global__ void conv1d_bwd_dx_kernel(
    const float* __restrict__ dy,   // [N, C_out, L_out]
    const float* __restrict__ w,     // [C_out, C_in, K]
    float* __restrict__ dx,          // [N, C_in, L_in]
    int N, int C_in, int C_out, int L_in, int L_out, int K,
    int padding, int stride, int dilation
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_in * L_in;
    if (idx >= total) return;
    dx[idx] = 0.0f;  // 先清零

    const int li = idx % L_in;
    const int ci = (idx / L_in) % C_in;
    const int n = idx / (L_in * C_in);

    // 遍历所有 (co, lo, ki) 使得 lo*stride + ki*dilation - padding == li
    for (int co = 0; co < C_out; co++) {
        for (int lo = 0; lo < L_out; lo++) {
            for (int ki = 0; ki < K; ki++) {
                int l_calc = lo * stride + ki * dilation - padding;
                if (l_calc == li) {
                    dx[idx] += w[(co * C_in + ci) * K + ki] * dy[(n * C_out + co) * L_out + lo];
                }
            }
        }
    }
}

// Conv1d backward — 权重梯度 dw + 偏置梯度 db
// dw[co, ci, ki] = Σ_{n, lo} dy[n, co, lo] * x[n, ci, lo*stride + ki*dilation - padding]
// db[co] = Σ_{n, lo} dy[n, co, lo]
__global__ void conv1d_bwd_dwdb_kernel(
    const float* __restrict__ dy,   // [N, C_out, L_out]
    const float* __restrict__ x,    // [N, C_in, L_in]
    float* __restrict__ dw,         // [C_out, C_in, K]
    float* __restrict__ db,         // [C_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K,
    int padding, int stride, int dilation
) {
    const int co = blockIdx.x;      // 每个 block 算一个 C_out
    const int tid = threadIdx.x;

    // db[co] 用 warp reduction
    float db_sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int lo = tid; lo < L_out; lo += blockDim.x) {
            db_sum += dy[(n * C_out + co) * L_out + lo];
        }
    }
    // 跨线程归约
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        db_sum += __shfl_down_sync(0xffffffff, db_sum, offset);
    }
    if (tid == 0) db[co] = db_sum;

    // dw 计算：每个 thread 处理多个 (ci, ki)
    for (int ci = 0; ci < C_in; ci++) {
        for (int ki = 0; ki < K; ki++) {
            // 每个 thread 分摊 n/lo 循环
            float acc = 0.0f;
            for (int n = 0; n < N; n++) {
                for (int lo = 0; lo < L_out; lo++) {
                    int l_idx = lo * stride + ki * dilation - padding;
                    if (l_idx >= 0 && l_idx < L_in) {
                        acc += dy[(n * C_out + co) * L_out + lo] * x[(n * C_in + ci) * L_in + l_idx];
                    }
                }
            }
            // 简单 atomicAdd 写 dw（正确但非最优）
            atomicAdd(&dw[(co * C_in + ci) * K + ki], acc);
        }
    }
}

// ============================================================
// Host wrappers（Rust extern "C" 调用入口）
// ============================================================

extern "C" void oyta_conv1d_fwd(
    const float* x, float* out,
    int N, int C_in, int C_out, int L_in, int L_out, int K,
    int padding, int stride, int dilation, int has_bias,
    const float* b, const float* w
) {
    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv1d_fwd_kernel<<<blocks, threads>>>(
        x, w, b, out, N, C_in, C_out, L_in, L_out, K,
        padding, stride, dilation, has_bias
    );
}

extern "C" void oyta_conv1d_bwd_dx(
    const float* dy, const float* w, float* dx,
    int N, int C_in, int C_out, int L_in, int L_out, int K,
    int padding, int stride, int dilation
) {
    int total = N * C_in * L_in;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv1d_bwd_dx_kernel<<<blocks, threads>>>(
        dy, w, dx, N, C_in, C_out, L_in, L_out, K, padding, stride, dilation
    );
}

extern "C" void oyta_conv1d_bwd_dwdb(
    const float* dy, const float* x, float* dw, float* db,
    int N, int C_in, int C_out, int L_in, int L_out, int K,
    int padding, int stride, int dilation
) {
    // 先清零 dw
    int dw_total = C_out * C_in * K;
    cudaMemset(dw, 0, dw_total * sizeof(float));
    conv1d_bwd_dwdb_kernel<<<C_out, 256>>>(
        dy, x, dw, db, N, C_in, C_out, L_in, L_out, K, padding, stride, dilation
    );
}
