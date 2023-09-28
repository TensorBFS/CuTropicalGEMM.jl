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

#define CONCATENATE_(x, y) x##y
#define CONCATENATETHREE_(x, y, z) x##y##z

#define CONCATENATE(x, y) CONCATENATE_(x, y)
#define CONCATENATETHREE(x, y, z) CONCATENATETHREE_(x, y, z)

// The macro
#define OFFSET_row(row, col, ld) ((row) * (ld) + (col))
#define OFFSET_col(row, col, ld) ((col) * (ld) + (row))

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


#define TT _TT
#define TN _TN
#define NT _NT
#define NN _NN

template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_M, // height of block of C that each thread calculate
    const int THREAD_SIZE_N
    > 
__global__ void CONCATENATETHREE(TYPENAME, FUNCNAME, TT)( 
    TYPE * __restrict__ A,
    TYPE * __restrict__ B,
    TYPE * __restrict__ C, 
    TYPE alpha,
    TYPE beta,
    int M,
    int N,
    int K, 
    int DIM_GRID_X, 
    int DIM_GRID_Y
    ) {
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = threadIdx.y * bszx + threadIdx.x;
    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    // shared memory

    __shared__ TYPE As[BLOCK_SIZE_M * BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ TYPE Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];
    // registers for C
    TYPE accum[THREAD_SIZE_M * THREAD_SIZE_N] = {0};
    TYPE regs_a[THREAD_SIZE_M] = {0};
    TYPE regs_b[THREAD_SIZE_N] = {0};

    // init the accum as tropical zero
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_M; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_N; ++thread_x) {
            accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)] = PADDING;
        }
    }
    
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
            const int row = BLOCK_SIZE_M * BLOCK_IDY + i + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx;
            if (tile_idx > K - BLOCK_SIZE_K || BLOCK_IDY == DIM_GRID_Y - 1) {
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
            const int col = B_TILE_COL + BLOCK_SIZE_N * BLOCK_IDX;
            if (BLOCK_IDX == DIM_GRID_X -1 || tile_idx > K - BLOCK_SIZE_K) {
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
            const int row = BLOCK_SIZE_M * BLOCK_IDY + THREAD_SIZE_M * threadIdx.y + thread_y;
            const int col = BLOCK_SIZE_N * BLOCK_IDX + THREAD_SIZE_N * threadIdx.x + thread_x;
            if (BLOCK_IDX == DIM_GRID_X -1 || BLOCK_IDY == DIM_GRID_Y - 1) {
                if (row < M && col < N) {
                    C[OFFSET_col(row, col, M)] = OPERATOR_ADD(
                        OPERATOR_MUL(C[OFFSET_col(row, col, M)], beta), 
                        OPERATOR_MUL(accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)], alpha)
                        );
                }
            } else {
                C[OFFSET_col(row, col, M)] = OPERATOR_ADD(
                    OPERATOR_MUL(C[OFFSET_col(row, col, M)], beta), 
                    OPERATOR_MUL(accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)], alpha)
                    );
            }
        }
    }
    __syncthreads();
}

template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_M, // height of block of C that each thread calculate
    const int THREAD_SIZE_N
    > 
__global__ void CONCATENATETHREE(TYPENAME, FUNCNAME, TN)( 
    TYPE * __restrict__ A,
    TYPE * __restrict__ B,
    TYPE * __restrict__ C, 
    TYPE alpha,
    TYPE beta,
    int M,
    int N,
    int K, 
    int DIM_GRID_X, 
    int DIM_GRID_Y
    ) {
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid_A = threadIdx.y * bszx + threadIdx.x;
    const int tid_B = threadIdx.y + threadIdx.x * bszy;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    // shared memory

    __shared__ TYPE As[BLOCK_SIZE_M * BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ TYPE Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];
    // registers for C
    TYPE accum[THREAD_SIZE_M * THREAD_SIZE_N] = {0};
    TYPE regs_a[THREAD_SIZE_M] = {0};
    TYPE regs_b[THREAD_SIZE_N] = {0};

    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_M; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_N; ++thread_x) {
            accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)] = PADDING;
        }
    }
    
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
            const int row = BLOCK_SIZE_M * BLOCK_IDY + i + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx;
            if (tile_idx > K - BLOCK_SIZE_K || BLOCK_IDY == DIM_GRID_Y - 1) {
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
            const int col = B_TILE_COL + i + BLOCK_SIZE_N * BLOCK_IDX;
            if (BLOCK_IDX == DIM_GRID_X -1 || tile_idx > K - BLOCK_SIZE_K) {
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
            const int row = BLOCK_SIZE_M * BLOCK_IDY + THREAD_SIZE_M * threadIdx.y + thread_y;
            const int col = BLOCK_SIZE_N * BLOCK_IDX + THREAD_SIZE_N * threadIdx.x + thread_x;
            if (BLOCK_IDX == DIM_GRID_X -1 || BLOCK_IDY == DIM_GRID_Y - 1) {
                if (row < M && col < N) {
                    C[OFFSET_col(row, col, M)] = OPERATOR_ADD(
                        OPERATOR_MUL(C[OFFSET_col(row, col, M)], beta), 
                        OPERATOR_MUL(accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)], alpha)
                        );
                }
            } else {
                C[OFFSET_col(row, col, M)] = OPERATOR_ADD(
                        OPERATOR_MUL(C[OFFSET_col(row, col, M)], beta), 
                        OPERATOR_MUL(accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)], alpha)
                        );
            }
        }
    }
    __syncthreads();
}

template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_M, // height of block of C that each thread calculate
    const int THREAD_SIZE_N  // width of block of C that each thread calculate
    > 
__global__ void CONCATENATETHREE(TYPENAME, FUNCNAME, NT)( 
    TYPE * __restrict__ A,
    TYPE * __restrict__ B,
    TYPE * __restrict__ C,
    TYPE alpha,
    TYPE beta,
    int M,
    int N,
    int K, 
    int DIM_GRID_X, 
    int DIM_GRID_Y
    ) {
    
    // size of thread block
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;

    const int BLOCK_SIZE_MK = BLOCK_SIZE_M * BLOCK_SIZE_K;
    const int BLOCK_SIZE_KN = BLOCK_SIZE_K * BLOCK_SIZE_N;
    const int THREAD_SIZE_MN = THREAD_SIZE_M * THREAD_SIZE_N;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    // thread id
    const int tid = threadIdx.y * bszm + threadIdx.x;

    // shared memory
    // directly use 1d shared memory to avoid the conflict of col-major and row-major
    __shared__ TYPE As[BLOCK_SIZE_MK]; // avoid bank conflict
    __shared__ TYPE Bs[BLOCK_SIZE_KN];

    // registers for C
    TYPE accum[THREAD_SIZE_MN] = {0};
    TYPE regs_a[THREAD_SIZE_M] = {0};
    TYPE regs_b[THREAD_SIZE_N] = {0};

    #pragma unroll
    for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
        #pragma unroll
        for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
            accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)] = PADDING;
        }
    }
    
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
            const int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW ;
            const int col = A_TILE_COL + i + tile_idx;

            if (BLOCK_IDX == DIM_GRID_X -1 || tile_idx >= K - BLOCK_SIZE_K) {
                As[OFFSET_col(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = row < M && col < K ? A[OFFSET_col(row, col, M)] : PADDING;
            } else {
                As[OFFSET_col(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = A[OFFSET_col(row, col, M)];
            }
        }

        // load B from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            const int row = tile_idx + i + B_TILE_ROW;
            const int col = B_TILE_COL + BLOCK_SIZE_N * BLOCK_IDY;

            if (BLOCK_IDY == DIM_GRID_Y -1 || tile_idx > K - BLOCK_SIZE_K) {
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
            const int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + thread_n;
            const int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + thread_m;
            if (BLOCK_IDX == DIM_GRID_X -1 || BLOCK_IDY == DIM_GRID_Y - 1) {
                if (row < M && col < N) {
                    C[OFFSET_col(row, col, M)] = OPERATOR_ADD(
                        OPERATOR_MUL(accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)], alpha), 
                        OPERATOR_MUL(C[OFFSET_col(row, col, M)], beta)
                        );
                }
            } else {
                C[OFFSET_col(row, col, M)] = OPERATOR_ADD(
                        OPERATOR_MUL(accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)], alpha), 
                        OPERATOR_MUL(C[OFFSET_col(row, col, M)], beta)
                        );
            }
        }
    }
    __syncthreads();
}

template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_M, // height of block of C that each thread calculate
    const int THREAD_SIZE_N  // width of block of C that each thread calculate
    > 
__global__ void CONCATENATETHREE(TYPENAME, FUNCNAME, NN)( 
    TYPE * __restrict__ A,
    TYPE * __restrict__ B,
    TYPE * __restrict__ C,
    TYPE alpha,
    TYPE beta,
    int M,
    int N,
    int K, 
    int DIM_GRID_X, 
    int DIM_GRID_Y
    ) {
    
    // size of thread block
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;

    const int BLOCK_SIZE_MK = BLOCK_SIZE_M * BLOCK_SIZE_K;
    const int BLOCK_SIZE_KN = BLOCK_SIZE_K * BLOCK_SIZE_N;
    const int THREAD_SIZE_MN = THREAD_SIZE_M * THREAD_SIZE_N;

    int BLOCK_IDX = blockIdx.x % DIM_GRID_X;
    int BLOCK_IDY = blockIdx.x / DIM_GRID_X;

    // thread id
    const int tid = threadIdx.y * bszm + threadIdx.x;

    // shared memory
    // directly use 1d shared memory to avoid the conflict of col-major and row-major
    __shared__ TYPE As[BLOCK_SIZE_MK]; // avoid bank conflict
    __shared__ TYPE Bs[BLOCK_SIZE_KN];

    // registers for C
    TYPE accum[THREAD_SIZE_MN] = {0};
    TYPE regs_a[THREAD_SIZE_M] = {0};
    TYPE regs_b[THREAD_SIZE_N] = {0};

    #pragma unroll
    for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
        #pragma unroll
        for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
            accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)] = PADDING;
        }
    }
    
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
            const int row = BLOCK_SIZE_M * BLOCK_IDX + A_TILE_ROW ;
            const int col = A_TILE_COL + i + tile_idx;

            if (BLOCK_IDX == DIM_GRID_X -1 || tile_idx >= K - BLOCK_SIZE_K) {
                As[OFFSET_col(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = row < M && col < K ? A[OFFSET_col(row, col, M)] : PADDING;
            } else {
                As[OFFSET_col(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = A[OFFSET_col(row, col, M)];
            }
        }

        // load B from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            const int row = tile_idx + B_TILE_ROW;
            const int col = BLOCK_SIZE_N * BLOCK_IDY + i + B_TILE_COL;

            if (tile_idx >= K - BLOCK_SIZE_K || BLOCK_IDY == DIM_GRID_Y - 1) {
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
            const int col = BLOCK_SIZE_N * BLOCK_IDY + THREAD_SIZE_N * threadIdx.y + thread_n;
            const int row = BLOCK_SIZE_M * BLOCK_IDX + THREAD_SIZE_M * threadIdx.x + thread_m;

            if (BLOCK_IDX == DIM_GRID_X -1 || BLOCK_IDY == DIM_GRID_Y - 1) {
                if (row < M && col < N) {
                    C[OFFSET_col(row, col, M)] = OPERATOR_ADD(
                        OPERATOR_MUL(accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)], alpha), 
                        OPERATOR_MUL(C[OFFSET_col(row, col, M)], beta)
                        );
                }
            } else {
                C[OFFSET_col(row, col, M)] = OPERATOR_ADD(
                    OPERATOR_MUL(accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)], alpha), 
                    OPERATOR_MUL(C[OFFSET_col(row, col, M)], beta)
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

    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;

    
    if (TA == T && TB == T) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_N, BLOCK_SIZE_M / THREAD_SIZE_M);
        
        int DIM_GRID_X = n / BLOCK_SIZE_N;
        int DIM_GRID_Y = m / BLOCK_SIZE_M;
        if (n % BLOCK_SIZE_N != 0)
            DIM_GRID_X++;
        if (m % BLOCK_SIZE_M != 0)
            DIM_GRID_Y++;

        dim3 dimGrid(DIM_GRID_X * DIM_GRID_Y);

        CONCATENATETHREE(TYPENAME, FUNCNAME, TT)<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, alpha, beta, m, n, k, DIM_GRID_X, DIM_GRID_Y);
    }

    if (TA == T && TB == N) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_N, BLOCK_SIZE_M / THREAD_SIZE_M);
        
        int DIM_GRID_X = n / BLOCK_SIZE_N;
        int DIM_GRID_Y = m / BLOCK_SIZE_M;
        if (n % BLOCK_SIZE_N != 0)
            DIM_GRID_X++;
        if (m % BLOCK_SIZE_M != 0)
            DIM_GRID_Y++;

        dim3 dimGrid(DIM_GRID_X * DIM_GRID_Y);
            
        CONCATENATETHREE(TYPENAME, FUNCNAME, TN)<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, alpha, beta, m, n, k, DIM_GRID_X, DIM_GRID_Y);
    }

    if (TA == N && TB == T) {
        dim3 dimBlock(BLOCK_SIZE_M / THREAD_SIZE_M, BLOCK_SIZE_N / THREAD_SIZE_N);

        int DIM_GRID_X = m / BLOCK_SIZE_M;
        int DIM_GRID_Y = n / BLOCK_SIZE_N;
        if (m % BLOCK_SIZE_M != 0)
            DIM_GRID_X++;
        if (n % BLOCK_SIZE_N != 0)
            DIM_GRID_Y++;

        dim3 dimGrid(DIM_GRID_X * DIM_GRID_Y);

        CONCATENATETHREE(TYPENAME, FUNCNAME, NT)<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
            <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, alpha, beta, m, n, k, DIM_GRID_X, DIM_GRID_Y);
    }

    if (TA == N && TB == N) {
        dim3 dimBlock(BLOCK_SIZE_M / THREAD_SIZE_M, BLOCK_SIZE_N / THREAD_SIZE_N);
        
        int DIM_GRID_X = m / BLOCK_SIZE_M;
        int DIM_GRID_Y = n / BLOCK_SIZE_N;
        if (m % BLOCK_SIZE_M != 0)
            DIM_GRID_X++;
        if (n % BLOCK_SIZE_N != 0)
            DIM_GRID_Y++;

        dim3 dimGrid(DIM_GRID_X * DIM_GRID_Y);

        CONCATENATETHREE(TYPENAME, FUNCNAME, NN)<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
            <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, alpha, beta, m, n, k, DIM_GRID_X, DIM_GRID_Y);
    }

}
}