// oyta-device/cuda_kernels/rnn.cu
// GRU step + LSTM step CUDA kernel
// 每个 thread 处理一个 hidden dimension

#include <cuda_runtime.h>
#include <math.h>

// GRU step（hidden_size 每个维度一个 thread）
// 权重布局：[3*H, in_size + H]，行主序
// bias 布局：[3*H]
__global__ void gru_step_kernel(
    const float* __restrict__ x,      // [in_size]
    const float* __restrict__ h_prev,  // [H]
    float* __restrict__ h_new,         // [H]
    const float* __restrict__ w,       // [3*H, in_size+H]
    const float* __restrict__ b,        // [3*H]
    int in_size, int H
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;

    // 权重布局：z 前 H 行, r 中 H 行, n 后 H 行
    int stride = in_size + H;

    // z_gate = sigmoid(W_z·x + b_z + W_zh·h + b_z) — b_z 加两次
    float z_sum = b[i];
    float r_sum = b[H + i];
    float n_sum = b[2 * H + i];
    for (int j = 0; j < in_size; j++) {
        float xv = x[j];
        z_sum += w[i * stride + j] * xv;
        r_sum += w[(H + i) * stride + j] * xv;
        n_sum += w[(2 * H + i) * stride + j] * xv;
    }
    for (int j = 0; j < H; j++) {
        float hv = h_prev[j];
        z_sum += w[i * stride + in_size + j] * hv;
        r_sum += w[(H + i) * stride + in_size + j] * hv;
        // n gate 用候选权重 W_in（简化：直接加 h 贡献，不单独 W_hn）
        n_sum += w[(2 * H + i) * stride + in_size + j] * hv;
    }

    float z_t = 1.0f / (1.0f + expf(-z_sum));
    float r_t = 1.0f / (1.0f + expf(-r_sum));
    float h_prev_v = h_prev[i];

    // candidate (简化版：用 n_sum + r_t * h_prev 近似)
    // 正确版需要单独的 W_hn，这里保持与 CPU 默认实现一致
    float candidate = tanhf(n_sum + r_t * h_prev_v);
    h_new[i] = (1.0f - z_t) * h_prev_v + z_t * candidate;
}

// LSTM step
// 权重布局：[4*H, in_size+H]，i/f/g/o 顺序
__global__ void lstm_step_kernel(
    const float* __restrict__ x,
    const float* __restrict__ h_prev,
    const float* __restrict__ c_prev,
    float* __restrict__ h_new,
    float* __restrict__ c_new,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int in_size, int H
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;

    int stride = in_size + H;

    float i_sum = b[i];
    float f_sum = b[H + i];
    float g_sum = b[2 * H + i];
    float o_sum = b[3 * H + i];
    for (int j = 0; j < in_size; j++) {
        float xv = x[j];
        i_sum += w[i * stride + j] * xv;
        f_sum += w[(H + i) * stride + j] * xv;
        g_sum += w[(2 * H + i) * stride + j] * xv;
        o_sum += w[(3 * H + i) * stride + j] * xv;
    }
    for (int j = 0; j < H; j++) {
        float hv = h_prev[j];
        i_sum += w[i * stride + in_size + j] * hv;
        f_sum += w[(H + i) * stride + in_size + j] * hv;
        g_sum += w[(2 * H + i) * stride + in_size + j] * hv;
        o_sum += w[(3 * H + i) * stride + in_size + j] * hv;
    }

    float i_t = 1.0f / (1.0f + expf(-i_sum));
    float f_t = 1.0f / (1.0f + expf(-f_sum));
    float g_t = tanhf(g_sum);
    float o_t = 1.0f / (1.0f + expf(-o_sum));

    float c_prev_v = c_prev[i];
    float c_new_v = f_t * c_prev_v + i_t * g_t;
    float h_new_v = o_t * tanhf(c_new_v);
    c_new[i] = c_new_v;
    h_new[i] = h_new_v;
}

extern "C" void oyta_gru_step(
    const float* x, const float* h_prev, float* h_new,
    const float* w, const float* b,
    int in_size, int H
) {
    int threads = min(H, 256);
    int blocks = (H + threads - 1) / threads;
    gru_step_kernel<<<blocks, threads>>>(x, h_prev, h_new, w, b, in_size, H);
}

extern "C" void oyta_lstm_step(
    const float* x, const float* h_prev, const float* c_prev,
    float* h_new, float* c_new,
    const float* w, const float* b,
    int in_size, int H
) {
    int threads = min(H, 256);
    int blocks = (H + threads - 1) / threads;
    lstm_step_kernel<<<blocks, threads>>>(
        x, h_prev, c_prev, h_new, c_new, w, b, in_size, H
    );
}
