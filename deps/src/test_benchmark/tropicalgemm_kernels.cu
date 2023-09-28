#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>

#define CONCATENATE_(x, y) x##y
#define CONCATENATETHREE_(x, y, z) x##y##z

#define CONCATENATE(x, y) CONCATENATE_(x, y)
#define CONCATENATETHREE(x, y, z) CONCATENATETHREE_(x, y, z)

// The Tropical algebras
#ifdef PlusMul
#define OPERATOR_ADD(a, b) (a + b)
#define OPERATOR_MUL(a, b) (a * b)
#define PADDING 0
#define FUNCNAME _plusmul
#endif

#ifdef TropicalAndOr
#define OPERATOR_ADD(a, b) (a || b)
#define OPERATOR_MUL(a, b) (a && b)
#define PADDING false
#define FUNCNAME _andor
#endif

#ifdef TropicalMaxMul
#define OPERATOR_ADD(a, b) max(a, b)
#define OPERATOR_MUL(a, b) (a * b)
#define PADDING 0
#define FUNCNAME _maxmul
#endif

#ifdef TropicalMaxPlus
#define OPERATOR_ADD(a, b) max(a, b)
#define OPERATOR_MUL(a, b) (a + b)
#define PADDING -INFINITY
#define FUNCNAME _maxplus
#endif

#ifdef TropicalMinPlus
#define OPERATOR_ADD(a, b) min(a, b)
#define OPERATOR_MUL(a, b) (a + b)
#define PADDING INFINITY
#define FUNCNAME _minplus
#endif

// Types

#ifdef Bool
#define TYPE bool
#define TYPENAME BOOL
#endif

#ifdef FP32
#define TYPE float
#define TYPENAME FLOAT
#endif

#ifdef FP64
#define TYPE double
#define TYPENAME DOUBLE
#endif

#ifdef INT32
#define TYPE int
#define TYPENAME INT
#endif

#ifdef INT64
#define TYPE long
#define TYPENAME LONG
#endif

#ifdef S_L
#define BM 32
#define BK 32
#define BN 128
#endif

// #ifdef M_M
#define BM 64
#define BK 32
#define BN 64
// #endif

#ifdef L_S
#define BM 128
#define BK 32
#define BN 32
#endif

#define TT _TT
#define TN _TN
#define NT _NT
#define NN _NN

// OFFSET calculation
#define OFFSET_row(row, col, width, height) ((row) * (width) + (col))
#define OFFSET_col(row, col, width, height) ((col) * (height) + (row))


template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_K, const int BLOCK_SIZE_N, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void CONCATENATETHREE(TYPENAME, FUNCNAME, TN)(TYPE * __restrict__ A, TYPE * __restrict__ B, TYPE * __restrict__ C, int M, int N, int K, TYPE alpha, TYPE beta, int DIM_GRID_X, int DIM_GRID_Y) {
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    // thread id
    const int tid_A = threadIdx.y * bszx + threadIdx.x;
    const int tid_B = threadIdx.y + threadIdx.x * bszy;

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
    
    // As is load in row major way
    const int A_TILE_ROW = tid_A / BLOCK_SIZE_K;
    const int A_TILE_COL = tid_A % BLOCK_SIZE_K;
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    
    // Bs is load in col major way
    const int B_TILE_ROW = tid_B % BLOCK_SIZE_K;
    const int B_TILE_COL = tid_B / BLOCK_SIZE_K;
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {

        // load A from global memory to shared memory, in row major
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            const int row = BLOCK_SIZE_M * BLOCK_IDY + i + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx;
            if (tile_idx > K - BLOCK_SIZE_K || BLOCK_IDY == DIM_GRID_Y - 1) {
                As[OFFSET_row(i + A_TILE_ROW, A_TILE_COL, BLOCK_SIZE_M, BLOCK_SIZE_K)] = row < M && col < K ? A[OFFSET_row(row, col, M, K)] : PADDING;
            } else {
                As[OFFSET_row(i + A_TILE_ROW, A_TILE_COL, BLOCK_SIZE_M, BLOCK_SIZE_K)] = A[OFFSET_row(row, col, M, K)];
            }
        }

        // load B from global memory to shared memory, in col major
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            const int row = tile_idx + B_TILE_ROW;
            const int col = B_TILE_COL + BLOCK_SIZE_N * BLOCK_IDX + i;
            if (BLOCK_IDX == DIM_GRID_X -1 || tile_idx > K - BLOCK_SIZE_K) {
                Bs[OFFSET_row(B_TILE_ROW, i +  B_TILE_COL, BLOCK_SIZE_K, BLOCK_SIZE_N)] = row < K && col < N ? B[OFFSET_col(row, col, K, N)] : PADDING;
            } else {
                Bs[OFFSET_row(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K, BLOCK_SIZE_N)] = B[OFFSET_col(row, col, K, N)];
            }
        }

        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {

            #pragma unroll
            for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
                regs_a[thread_m] = As[OFFSET_row(thread_m + THREAD_SIZE_M * threadIdx.y, k, BLOCK_SIZE_M, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
                regs_b[thread_n] = Bs[OFFSET_row(k, thread_n + THREAD_SIZE_N * threadIdx.x, BLOCK_SIZE_K, BLOCK_SIZE_N)];
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
            const int row = BLOCK_SIZE_M * BLOCK_IDY + THREAD_SIZE_M * threadIdx.y + thread_m;
            const int col = BLOCK_SIZE_N * BLOCK_IDX + THREAD_SIZE_N * threadIdx.x + thread_n;
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

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_K, const int BLOCK_SIZE_N, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void CONCATENATETHREE(TYPENAME, FUNCNAME, NT)(TYPE * __restrict__ A, TYPE * __restrict__ B, TYPE * __restrict__ C, int M, int N, int K, TYPE alpha, TYPE beta, int DIM_GRID_X, int DIM_GRID_Y) {
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
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
    
    // Bs is load in row major way
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;
    const int B_TILE_COL = tid % BLOCK_SIZE_N;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

    
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

        // load B from global memory to shared memory, in row major
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            const int row = tile_idx + i + B_TILE_ROW;
            const int col = B_TILE_COL + BLOCK_SIZE_N * BLOCK_IDX;
            if (BLOCK_IDX == DIM_GRID_X -1 || tile_idx > K - BLOCK_SIZE_K) {
                Bs[OFFSET_row(i + B_TILE_ROW, B_TILE_COL, BLOCK_SIZE_K, BLOCK_SIZE_N)] = row < K && col < N ? B[OFFSET_row(row, col, K, N)] : PADDING;
            } else {
                Bs[OFFSET_row(i + B_TILE_ROW, B_TILE_COL, BLOCK_SIZE_K, BLOCK_SIZE_N)] = B[OFFSET_row(row, col, K, N)];
            }
        }

        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {

            #pragma unroll
            for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
                regs_a[thread_m] = As[OFFSET_col(thread_m + THREAD_SIZE_M * threadIdx.y, k, BLOCK_SIZE_M, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
                regs_b[thread_n] = Bs[OFFSET_row(k, thread_n + THREAD_SIZE_N * threadIdx.x, BLOCK_SIZE_K, BLOCK_SIZE_N)];
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
            const int row = BLOCK_SIZE_M * BLOCK_IDY + THREAD_SIZE_M * threadIdx.y + thread_m;
            const int col = BLOCK_SIZE_N * BLOCK_IDX + THREAD_SIZE_N * threadIdx.x + thread_n;
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

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_K, const int BLOCK_SIZE_N, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void CONCATENATETHREE(TYPENAME, FUNCNAME, TT)(TYPE * __restrict__ A, TYPE * __restrict__ B, TYPE * __restrict__ C, int M, int N, int K, TYPE alpha, TYPE beta, int DIM_GRID_X, int DIM_GRID_Y) {
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
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
    
    // As is load in row major way
    const int A_TILE_ROW = tid / BLOCK_SIZE_K;
    const int A_TILE_COL = tid % BLOCK_SIZE_K;
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    
    // Bs is load in row major way
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;
    const int B_TILE_COL = tid % BLOCK_SIZE_N;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

    
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {

        // load A from global memory to shared memory, in row major
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            const int row = BLOCK_SIZE_M * BLOCK_IDY + i + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx;
            if (tile_idx > K - BLOCK_SIZE_K || BLOCK_IDY == DIM_GRID_Y - 1) {
                As[OFFSET_row(i + A_TILE_ROW, A_TILE_COL, BLOCK_SIZE_M, BLOCK_SIZE_K)] = row < M && col < K ? A[OFFSET_row(row, col, M, K)] : PADDING;
            } else {
                As[OFFSET_row(i + A_TILE_ROW, A_TILE_COL, BLOCK_SIZE_M, BLOCK_SIZE_K)] = A[OFFSET_row(row, col, M, K)];
            }
        }

        // load B from global memory to shared memory, in row major
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            const int row = tile_idx + i + B_TILE_ROW;
            const int col = B_TILE_COL + BLOCK_SIZE_N * BLOCK_IDX;
            if (BLOCK_IDX == DIM_GRID_X -1 || tile_idx > K - BLOCK_SIZE_K) {
                Bs[OFFSET_row(i + B_TILE_ROW, B_TILE_COL, BLOCK_SIZE_K, BLOCK_SIZE_N)] = row < K && col < N ? B[OFFSET_row(row, col, K, N)] : PADDING;
            } else {
                Bs[OFFSET_row(i + B_TILE_ROW, B_TILE_COL, BLOCK_SIZE_K, BLOCK_SIZE_N)] = B[OFFSET_row(row, col, K, N)];
            }
        }

        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {

            #pragma unroll
            for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
                regs_a[thread_m] = As[OFFSET_row(thread_m + THREAD_SIZE_M * threadIdx.y, k, BLOCK_SIZE_M, BLOCK_SIZE_K)];
            }

            #pragma unroll
            for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
                regs_b[thread_n] = Bs[OFFSET_row(k, thread_n + THREAD_SIZE_N * threadIdx.x, BLOCK_SIZE_K, BLOCK_SIZE_N)];
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
            const int row = BLOCK_SIZE_M * BLOCK_IDY + THREAD_SIZE_M * threadIdx.y + thread_m;
            const int col = BLOCK_SIZE_N * BLOCK_IDX + THREAD_SIZE_N * threadIdx.x + thread_n;
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

extern "C"{
void CONCATENATE(TYPENAME, FUNCNAME)(const int m, const int n, const int k, TYPE *d_A, TYPE *d_B, TYPE *d_C, TYPE alpha, TYPE beta, const char TA, const char TB){
    // TA and TB are 'T' or 'N'
    const char T = 'T';
    const char N = 'N';

    const int BLOCK_SIZE_M = BM;
    const int BLOCK_SIZE_K = BK;
    const int BLOCK_SIZE_N = BN;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;
    
    int DIM_GRID_X = n / BLOCK_SIZE_N;
    int DIM_GRID_Y = m / BLOCK_SIZE_M;

    if (n % BLOCK_SIZE_N != 0)
        DIM_GRID_X++;
    if (m % BLOCK_SIZE_M != 0)
        DIM_GRID_Y++;

    dim3 dimGrid(DIM_GRID_X * DIM_GRID_Y);

    if (TA == T && TB == T) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_N, BLOCK_SIZE_M / THREAD_SIZE_M);
        CONCATENATETHREE(TYPENAME, FUNCNAME, TT)<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k, alpha, beta, DIM_GRID_X, DIM_GRID_Y);
    }

    if (TA == T && TB == N) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_N, BLOCK_SIZE_M / THREAD_SIZE_M);

        CONCATENATETHREE(TYPENAME, FUNCNAME, TN)<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k, alpha, beta, DIM_GRID_X, DIM_GRID_Y);
    }

    if (TA == N && TB == T) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_N, BLOCK_SIZE_M / THREAD_SIZE_M);

        CONCATENATETHREE(TYPENAME, FUNCNAME, NT)<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k, alpha, beta, DIM_GRID_X, DIM_GRID_Y);
    }

    if (TA == N && TB == N) {
        dim3 dimBlock(BLOCK_SIZE_M / THREAD_SIZE_M, BLOCK_SIZE_N / THREAD_SIZE_N);

        CONCATENATETHREE(TYPENAME, FUNCNAME, NN)<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k, alpha, beta, DIM_GRID_X, DIM_GRID_Y);
    }

}
}