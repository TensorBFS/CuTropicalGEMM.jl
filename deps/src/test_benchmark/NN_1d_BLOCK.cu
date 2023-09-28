#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define OFFSET_row(row, col, height, width) ((row) * (width) + (col))
#define OFFSET_col(row, col, height, width) ((col) * (height) + (row))

#define CONCATENATE_(x, y) x##y
#define CONCATENATETHREE_(x, y, z) x##y##z

#define CONCATENATE(x, y) CONCATENATE_(x, y)
#define CONCATENATETHREE(x, y, z) CONCATENATETHREE_(x, y, z)

// #define OPERATOR_ADD(a, b) (a + b)
// #define OPERATOR_MUL(a, b) (a * b)
// #define PADDING 0
// #define FUNCNAME _plusmul

#define OPERATOR_ADD(a, b) max(a, b)
#define OPERATOR_MUL(a, b) (a + b)
#define PADDING -INFINITY
#define FUNCNAME _maxplus

#define TYPE float
#define TYPENAME FLOAT

#define TT _TT
#define TN _TN
#define NT _NT
#define NN _NN

bool check_all_NN(const float *A,
    const float *B,
    const float *C,
    const float *D,
    int m, int n, int k) {
    for (int a = 0; a < 100; ++a) {
        for (int b = 0; b < 100; ++b) {
            int i  = rand() % m;
            int j  = rand() % n;
            float sum = PADDING;
            sum = C[OFFSET_col(i, j, m, n)];
            for (int p = 0; p < k; ++p) {
                sum = OPERATOR_ADD(OPERATOR_MUL(A[OFFSET_col(i, p, m, k)], B[OFFSET_col(p, j, k, n)]), sum);
            }

            if (std::fabs(sum - D[OFFSET_col(i, j, m, n)]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, D[OFFSET_col(i, j, m, n)]);
                return false;
            }
        }
    }

    return true;
}

// template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_K, const int BLOCK_SIZE_N, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_K, const int BLOCK_SIZE_N, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void CONCATENATETHREE(TYPENAME, FUNCNAME, NN)(TYPE * __restrict__ A, TYPE * __restrict__ B, TYPE * __restrict__ C, int M, int N, int K, TYPE alpha, TYPE beta, int DIM_GRID_X, int DIM_GRID_Y) {
    // size of thread block
    const int bszy = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int bszx = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    // thread id
    const int tid = threadIdx.y * bszx + threadIdx.x;

    // shared memory

    __shared__ TYPE As[BLOCK_SIZE_M * BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ TYPE Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];
    // registers for C
    TYPE accum[THREAD_SIZE_M * THREAD_SIZE_N] = {0};
    TYPE regs_a[THREAD_SIZE_M] = {0};
    TYPE regs_b[THREAD_SIZE_N] = {0};

    // init the accum as tropical zero
    #pragma unroll
    for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
        #pragma unroll
        for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
            accum[OFFSET_row(thread_m, thread_n, THREAD_SIZE_M, THREAD_SIZE_N)] = PADDING;
        }
    }
    
    // As is load in col major way
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;
    const int A_TILE_COL = tid / BLOCK_SIZE_M;
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    
    // Bs is load in col major way
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;
    const int B_TILE_COL = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {

        // load A from global memory to shared memory, in col major
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K ; i += A_TILE_ROW_STRIDE) {
            const int row = BLOCK_SIZE_M * BLOCK_IDY + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx + i;
            if (tile_idx > K - BLOCK_SIZE_K || BLOCK_IDY == DIM_GRID_Y - 1) {
                As[OFFSET_col(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M, BLOCK_SIZE_K)] = row < M && col < K ? A[OFFSET_col(row, col, M, K)] : PADDING;
            } else {
                As[OFFSET_col(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M, BLOCK_SIZE_K)] = A[OFFSET_col(row, col, M, K)];
            }
        }

        // load B from global memory to shared memory, in col major
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_N; i += B_TILE_ROW_STRIDE) {
            const int row = tile_idx + B_TILE_ROW;
            const int col = B_TILE_COL + BLOCK_SIZE_N * BLOCK_IDX + i;
            if (BLOCK_IDX == DIM_GRID_X -1 || tile_idx > K - BLOCK_SIZE_K) {
                Bs[OFFSET_col(B_TILE_ROW, i +  B_TILE_COL, BLOCK_SIZE_K, BLOCK_SIZE_N)] = row < K && col < N ? B[OFFSET_col(row, col, K, N)] : PADDING;
            } else {
                Bs[OFFSET_col(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K, BLOCK_SIZE_N)] = B[OFFSET_col(row, col, K, N)];
            }
        }

        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {

            #pragma unroll
            for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
                regs_a[thread_m] = As[OFFSET_col(thread_m + THREAD_SIZE_M * threadIdx.x, k, BLOCK_SIZE_M, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
                regs_b[thread_n] = Bs[OFFSET_col(k, thread_n + THREAD_SIZE_N * threadIdx.y, BLOCK_SIZE_K, BLOCK_SIZE_N)];
            }

            #pragma unroll
            for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
                #pragma unroll
                for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
                    accum[OFFSET_row(thread_m, thread_n, THREAD_SIZE_M, THREAD_SIZE_N)] = OPERATOR_ADD(OPERATOR_MUL(regs_a[thread_m], regs_b[thread_n]), accum[OFFSET_row(thread_m, thread_n, THREAD_SIZE_M, THREAD_SIZE_N)]);
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
        #pragma unroll
        for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
            const int row = BLOCK_SIZE_M * BLOCK_IDY + threadIdx.x * THREAD_SIZE_M + thread_m;
            const int col = BLOCK_SIZE_N * BLOCK_IDX + threadIdx.y * THREAD_SIZE_N + thread_n;
            if (BLOCK_IDX == DIM_GRID_X - 1 || BLOCK_IDY == DIM_GRID_Y - 1) {
                if (row < M && col < N) {
                    C[OFFSET_col(row, col, M, N)] = OPERATOR_ADD(
                        OPERATOR_MUL(C[OFFSET_col(row, col, M, N)], beta), 
                        OPERATOR_MUL(accum[OFFSET_row(thread_m, thread_n, THREAD_SIZE_M, THREAD_SIZE_N)], alpha)
                        );
                }
            } else {
                C[OFFSET_col(row, col, M, N)] = OPERATOR_ADD(
                    OPERATOR_MUL(C[OFFSET_col(row, col, M, N)], beta), 
                    OPERATOR_MUL(accum[OFFSET_row(thread_m, thread_n, THREAD_SIZE_M, THREAD_SIZE_N)], alpha)
                    );
            }
        }
    }
    __syncthreads();
}

void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
        // data[i] = 1;
    }
}

int main() {
    int m = 4096;
    int n = 4096;
    int k = 4096;

    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;

    float alpha = 0.0;
    float beta = 0.0;

    printf("BM = %d, BN = %d, BK = %d\n", BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K);

    dim3 dimBlock(BLOCK_SIZE_M / THREAD_SIZE_M, BLOCK_SIZE_N / THREAD_SIZE_N);

    int DIM_GRID_X = n / BLOCK_SIZE_N;
    int DIM_GRID_Y = m / BLOCK_SIZE_M;

    if (n % BLOCK_SIZE_N != 0)
        DIM_GRID_X++;
    if (m % BLOCK_SIZE_M != 0)
        DIM_GRID_Y++;

    dim3 dimGrid(DIM_GRID_X * DIM_GRID_Y);

    float *h_A, *h_B, *h_C, *h_D;
    cudaMallocHost(&h_A, m * k * sizeof(float));
    cudaMallocHost(&h_B, k * n * sizeof(float));
    cudaMallocHost(&h_C, m * n * sizeof(float));
    cudaMallocHost(&h_D, m * n * sizeof(float));
    random_init(h_A, m * k);
    random_init(h_B, k * n);
    random_init(h_C, m * n);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyDefault);

    // warmup
    CONCATENATETHREE(TYPENAME, FUNCNAME, NN)<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k, alpha, beta, DIM_GRID_X, DIM_GRID_Y);

    cudaMemcpy(h_D, d_C, m * n * sizeof(float), cudaMemcpyDefault);
    bool chk = check_all_NN(h_A, h_B, h_C, h_D, m, n, k);
    if (!chk) {
        printf("Matrix_C check: Failed\n");
    }

    const int n_iter = 100;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        CONCATENATETHREE(TYPENAME, FUNCNAME, NN)<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k, alpha, beta, DIM_GRID_X, DIM_GRID_Y);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    long workload = n_iter * long(m) * n * k * 2;
    double tflops = (double(workload) / 1e12) / (double(ms) / 1e3);
    printf("FP32_maxplus_NN: %f TFLOPS\n", tflops);
    printf("time per iteration: %f ms\n \n", ms / n_iter);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_D);
}