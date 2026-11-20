// oyta-device/cuda_kernels/matmul.cu
// 简化 tiled matmul：C[M,N] = A[M,K] * B[K,N]（行主序）
// BM=BN=BK=128，每个 block 负责一个 C 子块；warp tile 32x32 FMA

#include <cuda_runtime.h>

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 64;
static constexpr int TM = 8;
static constexpr int TN = 8;
// 每个 block 有 (BM/TM) * (BN/TN) = 8 * 8 = 64 threads
// shared memory 合计 smemA(16KB) + smemB(16KB) = 32KB < 48KB 默认上限

__global__ void matmul_kernel(
    const float* __restrict__ A,   // [M, K] row-major
    const float* __restrict__ B,   // [K, N] row-major
    float* __restrict__ C,         // [M, N] row-major
    int M, int N, int K
) {
    // block 坐标
    const int bx = blockIdx.x;  // col tile
    const int by = blockIdx.y;  // row tile

    // thread 在 block 内的坐标（2D: threadRow, threadCol）
    const int threadRow = threadIdx.x / (BN / TN);  // 0..BM/TM-1 = 0..15
    const int threadCol = threadIdx.x % (BN / TN);  // 0..BN/TN-1 = 0..15

    // 该 thread 负责的 C 子块起始行/列
    const int rowStart = by * BM + threadRow * TM;
    const int colStart = bx * BN + threadCol * TN;

    // 累加器寄存器
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    // 共享内存 tile
    __shared__ float smemA[BM][BK];
    __shared__ float smemB[BK][BN];

    // 遍历 K 方向
    for (int kOff = 0; kOff < K; kOff += BK) {
        // cooperative loading: 256 threads 分摊 BM*BK 和 BK*BN
        #pragma unroll
        for (int load = threadIdx.x; load < BM * BK; load += blockDim.x) {
            int lr = load / BK;
            int lk = load % BK;
            int gRow = by * BM + lr;
            int gCol = kOff + lk;
            smemA[lr][lk] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }
        #pragma unroll
        for (int load = threadIdx.x; load < BK * BN; load += blockDim.x) {
            int lk = load / BN;
            int lc = load % BN;
            int gRow = kOff + lk;
            int gCol = bx * BN + lc;
            smemB[lk][lc] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }
        __syncthreads();

        // FMA
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float aFrag[TM];
            float bFrag[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                aFrag[i] = smemA[threadRow * TM + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                bFrag[j] = smemB[kk][threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = fmaf(aFrag[i], bFrag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // 写回 C（边界检查）
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gRow = rowStart + i;
        if (gRow < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int gCol = colStart + j;
                if (gCol < N) {
                    C[gRow * N + gCol] = acc[i][j];
                }
            }
        }
    }
}

extern "C" void launch_matmul_kernel(
    float* C, const float* A, const float* B,
    int M, int N, int K, cudaStream_t stream
) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block((BM / TM) * (BN / TN));  // 256
    matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
