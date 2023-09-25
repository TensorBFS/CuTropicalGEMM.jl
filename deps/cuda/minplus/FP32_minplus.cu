#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// The macro
#define OFFSET_row(row, col, ld) ((row) * (ld) + (col))
#define OFFSET_col(row, col, ld) ((col) * (ld) + (row))

#define OPERATOR_ADD(a, b) min(a, b)
#define OPERATOR_MUL(a, b) (a + b)
#define PADDING INFINITY

#define TYPE float
#define FUNCNAME FP32_minplus
#define kernel_TT FP32_minplus_TT
#define kernel_TN FP32_minplus_TN
#define kernel_NT FP32_minplus_NT
#define kernel_NN FP32_minplus_NN

template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_M, // height of block of C that each thread calculate
    const int THREAD_SIZE_N
    > 
__global__ void kernel_TT( 
    TYPE * __restrict__ A,
    TYPE * __restrict__ B,
    TYPE * __restrict__ C, 
    int M,
    int N,
    int K
    ) {
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = threadIdx.y * bszx + threadIdx.x;

    // shared memory

    __shared__ TYPE As[BLOCK_SIZE_M * BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ TYPE Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];
    // registers for C
    TYPE accum[THREAD_SIZE_M * THREAD_SIZE_N] = {PADDING};
    TYPE regs_a[THREAD_SIZE_M] = {0};
    TYPE regs_b[THREAD_SIZE_N] = {0};
    
    // row number and col number that needs to be loaded blockIdx.y this thread
    const int A_TILE_ROW = tid / BLOCK_SIZE_K;
    const int A_TILE_COL = tid % BLOCK_SIZE_K;
    
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;
    const int B_TILE_COL = tid % BLOCK_SIZE_N;
    
    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

    // const int A_S = BLOCK_SIZE_M / THREAD_SIZE_M;
    // const int B_S = BLOCK_SIZE_N / THREAD_SIZE_N;

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {

        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx;
            if (tile_idx > K - BLOCK_SIZE_K || blockIdx.y == gridDim.y - 1) {
                As[OFFSET_row(i + A_TILE_ROW, A_TILE_COL, BLOCK_SIZE_K)] = row < M && col < K ? A[OFFSET_row(
                    row, // row
                    col, // col
                    K )] : PADDING;
            } else {
                As[OFFSET_row(i + A_TILE_ROW, A_TILE_COL, BLOCK_SIZE_K)] = A[OFFSET_row(
                    row, // row
                    col, // col
                    K )];
            }
        }

        // load B from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            const int row = tile_idx + i + B_TILE_ROW;
            const int col = B_TILE_COL + BLOCK_SIZE_N * blockIdx.x;
            if (blockIdx.x == gridDim.x -1 || tile_idx > K - BLOCK_SIZE_K) {
                Bs[OFFSET_row(i + B_TILE_ROW, B_TILE_COL, BLOCK_SIZE_N)] = row < K && col < N ? B[OFFSET_row(
                    row, // row
                    col, // col
                    N )] : PADDING;
            } else {
                Bs[OFFSET_row(i + B_TILE_ROW, B_TILE_COL, BLOCK_SIZE_N)] = B[OFFSET_row(
                    row, // row
                    col, // col
                    N )];
            }
        }

        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {

            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_M; ++thread_y) {
                regs_a[thread_y] = As[OFFSET_row(thread_y + THREAD_SIZE_M * threadIdx.y, k, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_N; ++thread_x) {
                regs_b[thread_x] = Bs[OFFSET_row(k, thread_x + THREAD_SIZE_N * threadIdx.x, BLOCK_SIZE_N)];
            }

            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_M; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_N; ++thread_x) {
                    accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)] = OPERATOR_ADD(OPERATOR_MUL(regs_a[thread_y], regs_b[thread_x]), accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)]);
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_M; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_N; ++thread_x) {
            const int row = BLOCK_SIZE_M * blockIdx.y + THREAD_SIZE_M * threadIdx.y + thread_y;
            const int col = BLOCK_SIZE_N * blockIdx.x + THREAD_SIZE_N * threadIdx.x + thread_x;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                if (row < M && col < N) {
                    C[OFFSET_col(row, col, M)] = OPERATOR_ADD(C[OFFSET_col(row, col, M)], accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)]);
                }
            } else {
                C[OFFSET_col(row, col, M)] = OPERATOR_ADD(C[OFFSET_col(row, col, M)], accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)]);
            }
        }
    }
}

template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_M, // height of block of C that each thread calculate
    const int THREAD_SIZE_N
    > 
__global__ void kernel_TN( 
    TYPE * __restrict__ A,
    TYPE * __restrict__ B,
    TYPE * __restrict__ C, 
    int M,
    int N,
    int K
    ) {
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid_A = threadIdx.y * bszx + threadIdx.x;
    const int tid_B = threadIdx.y + threadIdx.x * bszy;

    // shared memory

    __shared__ TYPE As[BLOCK_SIZE_M * BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ TYPE Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];
    // registers for C
    TYPE accum[THREAD_SIZE_M * THREAD_SIZE_N] = {PADDING};
    TYPE regs_a[THREAD_SIZE_M] = {0};
    TYPE regs_b[THREAD_SIZE_N] = {0};
    
    // row number and col number that needs to be loaded blockIdx.y this thread
    const int A_TILE_ROW = tid_A / BLOCK_SIZE_K;
    const int A_TILE_COL = tid_A % BLOCK_SIZE_K;
    
    const int B_TILE_ROW = tid_B % BLOCK_SIZE_K;
    const int B_TILE_COL = tid_B / BLOCK_SIZE_K;
    
    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    // const int A_S = BLOCK_SIZE_M / THREAD_SIZE_M;
    // const int B_S = BLOCK_SIZE_N / THREAD_SIZE_N;

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {

        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx;
            if (tile_idx > K - BLOCK_SIZE_K || blockIdx.y == gridDim.y - 1) {
                As[OFFSET_row(i + A_TILE_ROW, A_TILE_COL, BLOCK_SIZE_K)] = row < M && col < K ? A[OFFSET_row(
                    row, // row
                    col, // col
                    K )] : PADDING;
            } else {
                As[OFFSET_row(i + A_TILE_ROW, A_TILE_COL, BLOCK_SIZE_K)] = A[OFFSET_row(
                    row, // row
                    col, // col
                    K )];
            }
        }

        // load B from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            const int row = tile_idx + B_TILE_ROW;
            const int col = B_TILE_COL + i + BLOCK_SIZE_N * blockIdx.x;
            if (blockIdx.x == gridDim.x -1 || tile_idx > K - BLOCK_SIZE_K) {
                Bs[OFFSET_row(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_N)] = row < K && col < N ? B[OFFSET_col(row, col, K)] : PADDING;
            } else {
                Bs[OFFSET_row(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_N)] = B[OFFSET_col(row, col, K)];
            }
        }

        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {

            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_M; ++thread_y) {
                regs_a[thread_y] = As[OFFSET_row(thread_y + THREAD_SIZE_M * threadIdx.y, k, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_N; ++thread_x) {
                regs_b[thread_x] = Bs[OFFSET_row(k, thread_x + THREAD_SIZE_N * threadIdx.x, BLOCK_SIZE_N)];
            }

            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_M; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_N; ++thread_x) {
                    accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)] = OPERATOR_ADD(OPERATOR_MUL(regs_a[thread_y], regs_b[thread_x]), accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)]);
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_M; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_N; ++thread_x) {
            const int row = BLOCK_SIZE_M * blockIdx.y + THREAD_SIZE_M * threadIdx.y + thread_y;
            const int col = BLOCK_SIZE_N * blockIdx.x + THREAD_SIZE_N * threadIdx.x + thread_x;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                if (row < M && col < N) {
                    C[OFFSET_col(row, col, M)] = OPERATOR_ADD(C[OFFSET_col(row, col, M)], accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)]);
                }
            } else {
                C[OFFSET_col(row, col, M)] = OPERATOR_ADD(C[OFFSET_col(row, col, M)], accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)]);
            }
        }
    }
}

template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_M, // height of block of C that each thread calculate
    const int THREAD_SIZE_N  // width of block of C that each thread calculate
    > 
__global__ void kernel_NT( 
    TYPE * __restrict__ A,
    TYPE * __restrict__ B,
    TYPE * __restrict__ C,
    int M,
    int N,
    int K
    ) {
    
    // size of thread block
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;

    const int BLOCK_SIZE_MK = BLOCK_SIZE_M * BLOCK_SIZE_K;
    const int BLOCK_SIZE_KN = BLOCK_SIZE_K * BLOCK_SIZE_N;
    const int THREAD_SIZE_MN = THREAD_SIZE_M * THREAD_SIZE_N;

    // thread id
    const int tid = threadIdx.y * bszm + threadIdx.x;

    // shared memory
    // directly use 1d shared memory to avoid the conflict of col-major and row-major
    __shared__ TYPE As[BLOCK_SIZE_MK]; // avoid bank conflict
    __shared__ TYPE Bs[BLOCK_SIZE_KN];

    // registers for C
    TYPE accum[THREAD_SIZE_MN] = {PADDING};
    TYPE regs_a[THREAD_SIZE_M] = {0};
    TYPE regs_b[THREAD_SIZE_N] = {0};
    
    // row number and col number that needs to be loaded blockIdx.y this thread
    const int A_TILE_COL = tid / BLOCK_SIZE_M;
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;

    const int B_TILE_ROW = tid / BLOCK_SIZE_N;
    const int B_TILE_COL = tid % BLOCK_SIZE_N;
    
    // col stride that thread uses to load multiple rows of a tile
    // how many cols that the threads load in one iteration
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    // const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

    for (int tile_idx = 0 ; tile_idx < K ;) {

        // load A from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K ; i += A_TILE_COL_STRIDE) {
            const int row = BLOCK_SIZE_M * blockIdx.x + A_TILE_ROW ;
            const int col = A_TILE_COL + i + tile_idx;

            if (blockIdx.x == gridDim.x -1 || tile_idx >= K - BLOCK_SIZE_K) {
                As[OFFSET_col(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = row < M && col < K ? A[OFFSET_col(row, col, M)] : PADDING;
            } else {
                As[OFFSET_col(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = A[OFFSET_col(row, col, M)];
            }
        }

        // load B from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            const int row = tile_idx + i + B_TILE_ROW;
            const int col = B_TILE_COL + BLOCK_SIZE_N * blockIdx.y;

            if (blockIdx.y == gridDim.y -1 || tile_idx > K - BLOCK_SIZE_K) {
                Bs[OFFSET_row(i + B_TILE_ROW, B_TILE_COL, BLOCK_SIZE_N)] = row < K && col < N ? B[OFFSET_row(row, col, N)] : PADDING;
            } else {
                Bs[OFFSET_row(i + B_TILE_ROW, B_TILE_COL, BLOCK_SIZE_N)] = B[OFFSET_row(row, col, N)];
            }
        }

        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; k += 1) {

            // load A and B from shared memory to registers
            #pragma unroll
            for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
                regs_a[thread_m] = As[OFFSET_col(threadIdx.x * THREAD_SIZE_M + thread_m, k, BLOCK_SIZE_M)];
            }

            #pragma unroll
            for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
                regs_b[thread_n] = Bs[OFFSET_row(k, threadIdx.y * THREAD_SIZE_N + thread_n, BLOCK_SIZE_N)];
            }

            #pragma unroll
            for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
                #pragma unroll
                for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
                    accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)] = OPERATOR_ADD(OPERATOR_MUL(regs_a[thread_m], regs_b[thread_n]), accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)]);
                }
            }
            
        }
        __syncthreads();
        tile_idx += BLOCK_SIZE_K;
    }

    // store back to C
    #pragma unroll
    for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
        #pragma unroll
        for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
            const int col = BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * threadIdx.y + thread_n;
            const int row = BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * threadIdx.x + thread_m;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                if (row < M && col < N) {
                    C[OFFSET_col(row, col, M)] = OPERATOR_ADD(accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)], C[OFFSET_col(row, col, M)]);
                }
            } else {
                C[OFFSET_col(row, col, M)] = OPERATOR_ADD(accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)], C[OFFSET_col(row, col, M)]);
            }
        }
    }
}

template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_M, // height of block of C that each thread calculate
    const int THREAD_SIZE_N  // width of block of C that each thread calculate
    > 
__global__ void kernel_NN( 
    TYPE * __restrict__ A,
    TYPE * __restrict__ B,
    TYPE * __restrict__ C,
    int M,
    int N,
    int K
    ) {
    
    // size of thread block
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;

    const int BLOCK_SIZE_MK = BLOCK_SIZE_M * BLOCK_SIZE_K;
    const int BLOCK_SIZE_KN = BLOCK_SIZE_K * BLOCK_SIZE_N;
    const int THREAD_SIZE_MN = THREAD_SIZE_M * THREAD_SIZE_N;

    // thread id
    const int tid = threadIdx.y * bszm + threadIdx.x;

    // shared memory
    // directly use 1d shared memory to avoid the conflict of col-major and row-major
    __shared__ TYPE As[BLOCK_SIZE_MK]; // avoid bank conflict
    __shared__ TYPE Bs[BLOCK_SIZE_KN];

    // registers for C
    TYPE accum[THREAD_SIZE_MN] = {PADDING};
    TYPE regs_a[THREAD_SIZE_M] = {0};
    TYPE regs_b[THREAD_SIZE_N] = {0};
    
    // row number and col number that needs to be loaded blockIdx.y this thread
    const int A_TILE_COL = tid / BLOCK_SIZE_M;
    const int A_TILE_ROW = tid % BLOCK_SIZE_M;

    const int B_TILE_COL = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;
    
    // col stride that thread uses to load multiple rows of a tile
    // how many cols that the threads load in one iteration
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    // number of threads in M and N direction (used when calculating C)
    // const int A_S = BLOCK_SIZE_M / THREAD_SIZE_M;
    // const int B_S = BLOCK_SIZE_N / THREAD_SIZE_N;

    for (int tile_idx = 0 ; tile_idx < K ;) {

        // load A from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K ; i += A_TILE_COL_STRIDE) {
            const int row = BLOCK_SIZE_M * blockIdx.x + A_TILE_ROW ;
            const int col = A_TILE_COL + i + tile_idx;

            if (blockIdx.x == gridDim.x -1 || tile_idx >= K - BLOCK_SIZE_K) {
                As[OFFSET_col(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = row < M && col < K ? A[OFFSET_col(row, col, M)] : PADDING;
            } else {
                As[OFFSET_col(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = A[OFFSET_col(row, col, M)];
            }
        }

        // load B from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            const int row = tile_idx + B_TILE_ROW;
            const int col = BLOCK_SIZE_N * blockIdx.y + i + B_TILE_COL;

            if (tile_idx >= K - BLOCK_SIZE_K || blockIdx.y == gridDim.y - 1) {
                Bs[OFFSET_col(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = row < K && col < N ? B[OFFSET_col(row, col, K)] : PADDING;
            } else {
                Bs[OFFSET_col(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = B[OFFSET_col(row, col, K)];
            }
        }

        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; k += 1) {

            // load A and B from shared memory to registers
            #pragma unroll
            for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
                regs_a[thread_m] = As[OFFSET_col(threadIdx.x * THREAD_SIZE_M + thread_m, k, BLOCK_SIZE_M)];
            }

            #pragma unroll
            for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
                regs_b[thread_n] = Bs[OFFSET_col(k, threadIdx.y * THREAD_SIZE_N + thread_n, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
                #pragma unroll
                for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
                    accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)] = OPERATOR_ADD(OPERATOR_MUL(regs_a[thread_m], regs_b[thread_n]), accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)]);
                }
            }
            
        }
        __syncthreads();
        tile_idx += BLOCK_SIZE_K;
    }

    // store back to C
    #pragma unroll
    for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
        #pragma unroll
        for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
            const int col = BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * threadIdx.y + thread_n;
            const int row = BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * threadIdx.x + thread_m;

            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                if (row < M && col < N) {
                    C[OFFSET_col(row, col, M)] = OPERATOR_ADD(accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)], C[OFFSET_col(row, col, M)]);
                }
            } else {
                C[OFFSET_col(row, col, M)] = OPERATOR_ADD(accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)], C[OFFSET_col(row, col, M)]);
            }
        }
    }
}

extern "C"
void FUNCNAME(const int m, const int n, const int k, TYPE *d_A, TYPE *d_B, TYPE *d_C, const char TA, const char TB){
    // TA and TB are 'T' or 'N'

    const char T = 'T';
    const char N = 'N';

    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;
    
    if (TA == T && TB == T) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_N, BLOCK_SIZE_M / THREAD_SIZE_M);
        dim3 dimGrid(n / BLOCK_SIZE_N, m / BLOCK_SIZE_M);
        if (n % BLOCK_SIZE_N != 0)
            dimGrid.x++;
        if (m % BLOCK_SIZE_M != 0)
            dimGrid.y++;

        kernel_TT<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
            <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k);
    }

    if (TA == T && TB == N) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_N, BLOCK_SIZE_M / THREAD_SIZE_M);
        dim3 dimGrid(n / BLOCK_SIZE_N, m / BLOCK_SIZE_M);
        if (n % BLOCK_SIZE_N != 0)
            dimGrid.x++;
        if (m % BLOCK_SIZE_M != 0)
            dimGrid.y++;
            
        kernel_TN<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k);
    }

    if (TA == N && TB == T) {
        dim3 dimBlock(BLOCK_SIZE_M / THREAD_SIZE_M, BLOCK_SIZE_N / THREAD_SIZE_N);
        dim3 dimGrid(m / BLOCK_SIZE_M, n / BLOCK_SIZE_N);
        if (m % BLOCK_SIZE_M != 0)
            dimGrid.x++;
        if (n % BLOCK_SIZE_N != 0)
            dimGrid.y++;

        kernel_NT<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
            <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k);
    }

    if (TA == N && TB == N) {
        dim3 dimBlock(BLOCK_SIZE_M / THREAD_SIZE_M, BLOCK_SIZE_N / THREAD_SIZE_N);
        dim3 dimGrid(m / BLOCK_SIZE_M, n / BLOCK_SIZE_N);
        if (m % BLOCK_SIZE_M != 0)
            dimGrid.x++;
        if (n % BLOCK_SIZE_N != 0)
            dimGrid.y++;

        kernel_NN<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
            <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k);
    }

}