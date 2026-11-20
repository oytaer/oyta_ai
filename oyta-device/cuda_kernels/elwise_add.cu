// oyta-device/cuda_kernels/elwise_add.cu
// 编译：nvcc -c -O3 --use_fast_math elwise_add.cu -o elwise_add.o

#include <cuda_runtime.h>

// 每个 thread 处理一个元素（block_size=256）
__global__ void elwise_add_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { c[idx] = a[idx] + b[idx]; }
}

// C++ 封装（Rust 通过 extern "C" 调用）
extern "C" void launch_elwise_add_kernel(
    float* c, const float* a, const float* b, int n, cudaStream_t stream
) {
    const int BS = 256;
    const int GS = (n + BS - 1) / BS;
    elwise_add_kernel<<<GS, BS, 0, stream>>>(a, b, c, n);
}
