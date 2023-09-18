#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>


#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// rule for Tropical Max Mul
#define OPERATOR_ADD(a, b) max(a, b)
#define OPERATOR_MUT(a, b) (a * b)
#define PADDING_FP 0
#define PADDING_INT 0


template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X
    > 
__global__ void FP32_maxmul_kernel( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int N,
    const int K
    ) {
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = threadIdx.y * bszx + threadIdx.x;

    // shared memory

    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {PADDING_FP};
    
    // row number and col number that needs to be loaded blockIdx.y this thread
    const int A_TILE_ROW = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;

    const int A_TILE_COL = tid % BLOCK_SIZE_K;
    const int B_TILE_COL = tid % BLOCK_SIZE_N;
    
    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

    const int A_S = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int B_S = BLOCK_SIZE_N / THREAD_SIZE_X;

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {

        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx;
            if (tile_idx > K - BLOCK_SIZE_K || blockIdx.y == gridDim.y - 1) {
                As[i + A_TILE_ROW ][A_TILE_COL] = row < M && col < K ? A[OFFSET(
                    row, // row
                    col, // col
                    K )] : PADDING_FP;
            } else {
                As[i + A_TILE_ROW ][A_TILE_COL] = A[OFFSET(
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
                Bs[i + B_TILE_ROW][B_TILE_COL] = row < K && col < N ? B[OFFSET(
                    row, // row
                    col, // col
                    N )] : PADDING_FP;
            } else {
                Bs[i + B_TILE_ROW][B_TILE_COL] = B[OFFSET(
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
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] = OPERATOR_ADD(OPERATOR_MUT(As[thread_y * A_S + threadIdx.y][k], Bs[k][thread_x * B_S + threadIdx.x]), accum[thread_y][thread_x]);
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
            const int row = BLOCK_SIZE_M * blockIdx.y + thread_y * A_S + threadIdx.y;
            const int col = BLOCK_SIZE_N * blockIdx.x + thread_x * B_S + threadIdx.x;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                if (row < M && col < N) {
                    C[OFFSET(row, col, N)] = OPERATOR_ADD(C[OFFSET(row, col, N)], accum[thread_y][thread_x]);
                }
            } else {
                C[OFFSET(row, col, N)] = OPERATOR_ADD(C[OFFSET(row, col, N)], accum[thread_y][thread_x]);
            }
        }
    }
}

template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X
    > 
__global__ void FP64_maxmul_kernel( 
    double * __restrict__ A,
    double * __restrict__ B,
    double * __restrict__ C, 
    const int M,
    const int N,
    const int K
    ) {
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = threadIdx.y * bszx + threadIdx.x;

    // shared memory

    __shared__ double As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ double Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    double accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {PADDING_FP};
    
    // row number and col number that needs to be loaded blockIdx.y this thread
    const int A_TILE_ROW = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;

    const int A_TILE_COL = tid % BLOCK_SIZE_K;
    const int B_TILE_COL = tid % BLOCK_SIZE_N;
    
    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

    const int A_S = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int B_S = BLOCK_SIZE_N / THREAD_SIZE_X;

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {

        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx;
            if (tile_idx > K - BLOCK_SIZE_K || blockIdx.y == gridDim.y - 1) {
                As[i + A_TILE_ROW ][A_TILE_COL] = row < M && col < K ? A[OFFSET(
                    row, // row
                    col, // col
                    K )] : PADDING_FP;
            } else {
                As[i + A_TILE_ROW ][A_TILE_COL] = A[OFFSET(
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
                Bs[i + B_TILE_ROW][B_TILE_COL] = row < K && col < N ? B[OFFSET(
                    row, // row
                    col, // col
                    N )] : PADDING_FP;
            } else {
                Bs[i + B_TILE_ROW][B_TILE_COL] = B[OFFSET(
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
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] = OPERATOR_ADD(OPERATOR_MUT(As[thread_y * A_S + threadIdx.y][k], Bs[k][thread_x * B_S + threadIdx.x]), accum[thread_y][thread_x]);
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
            const int row = BLOCK_SIZE_M * blockIdx.y + thread_y * A_S + threadIdx.y;
            const int col = BLOCK_SIZE_N * blockIdx.x + thread_x * B_S + threadIdx.x;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                if (row < M && col < N) {
                    C[OFFSET(row, col, N)] = OPERATOR_ADD(C[OFFSET(row, col, N)], accum[thread_y][thread_x]);
                }
            } else {
                C[OFFSET(row, col, N)] = OPERATOR_ADD(C[OFFSET(row, col, N)], accum[thread_y][thread_x]);
            }
        }
    }
}

template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X
    > 
__global__ void INT32_maxmul_kernel( 
    int * __restrict__ A,
    int * __restrict__ B,
    int * __restrict__ C, 
    const int M,
    const int N,
    const int K
    ) {
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = threadIdx.y * bszx + threadIdx.x;

    // shared memory

    __shared__ int As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ int Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    int accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {PADDING_INT};
    
    // row number and col number that needs to be loaded blockIdx.y this thread
    const int A_TILE_ROW = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;

    const int A_TILE_COL = tid % BLOCK_SIZE_K;
    const int B_TILE_COL = tid % BLOCK_SIZE_N;
    
    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

    const int A_S = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int B_S = BLOCK_SIZE_N / THREAD_SIZE_X;

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {

        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx;
            if (tile_idx > K - BLOCK_SIZE_K || blockIdx.y == gridDim.y - 1) {
                As[i + A_TILE_ROW ][A_TILE_COL] = row < M && col < K ? A[OFFSET(
                    row, // row
                    col, // col
                    K )] : PADDING_INT;
            } else {
                As[i + A_TILE_ROW ][A_TILE_COL] = A[OFFSET(
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
                Bs[i + B_TILE_ROW][B_TILE_COL] = row < K && col < N ? B[OFFSET(
                    row, // row
                    col, // col
                    N )] : PADDING_INT;
            } else {
                Bs[i + B_TILE_ROW][B_TILE_COL] = B[OFFSET(
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
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] = OPERATOR_ADD(OPERATOR_MUT(As[thread_y * A_S + threadIdx.y][k], Bs[k][thread_x * B_S + threadIdx.x]), accum[thread_y][thread_x]);
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
            const int row = BLOCK_SIZE_M * blockIdx.y + thread_y * A_S + threadIdx.y;
            const int col = BLOCK_SIZE_N * blockIdx.x + thread_x * B_S + threadIdx.x;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                if (row < M && col < N) {
                    C[OFFSET(row, col, N)] = OPERATOR_ADD(C[OFFSET(row, col, N)], accum[thread_y][thread_x]);
                }
            } else {
                C[OFFSET(row, col, N)] = OPERATOR_ADD(C[OFFSET(row, col, N)], accum[thread_y][thread_x]);
            }
        }
    }
}

template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X
    > 
__global__ void INT64_maxmul_kernel( 
    long * __restrict__ A,
    long * __restrict__ B,
    long * __restrict__ C, 
    const int M,
    const int N,
    const int K
    ) {
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = threadIdx.y * bszx + threadIdx.x;

    // shared memory

    __shared__ long As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ long Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    long accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {PADDING_INT};
    
    // row number and col number that needs to be loaded blockIdx.y this thread
    const int A_TILE_ROW = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;

    const int A_TILE_COL = tid % BLOCK_SIZE_K;
    const int B_TILE_COL = tid % BLOCK_SIZE_N;
    
    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

    const int A_S = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int B_S = BLOCK_SIZE_N / THREAD_SIZE_X;

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {

        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx;
            if (tile_idx > K - BLOCK_SIZE_K || blockIdx.y == gridDim.y - 1) {
                As[i + A_TILE_ROW ][A_TILE_COL] = row < M && col < K ? A[OFFSET(
                    row, // row
                    col, // col
                    K )] : PADDING_INT;
            } else {
                As[i + A_TILE_ROW ][A_TILE_COL] = A[OFFSET(
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
                Bs[i + B_TILE_ROW][B_TILE_COL] = row < K && col < N ? B[OFFSET(
                    row, // row
                    col, // col
                    N )] : PADDING_INT;
            } else {
                Bs[i + B_TILE_ROW][B_TILE_COL] = B[OFFSET(
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
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] = OPERATOR_ADD(OPERATOR_MUT(As[thread_y * A_S + threadIdx.y][k], Bs[k][thread_x * B_S + threadIdx.x]), accum[thread_y][thread_x]);
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
            const int row = BLOCK_SIZE_M * blockIdx.y + thread_y * A_S + threadIdx.y;
            const int col = BLOCK_SIZE_N * blockIdx.x + thread_x * B_S + threadIdx.x;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                if (row < M && col < N) {
                    C[OFFSET(row, col, N)] = OPERATOR_ADD(C[OFFSET(row, col, N)], accum[thread_y][thread_x]);
                }
            } else {
                C[OFFSET(row, col, N)] = OPERATOR_ADD(C[OFFSET(row, col, N)], accum[thread_y][thread_x]);
            }
        }
    }
}

extern "C"
void FP32_maxmul(const int M, const int N, const int K, float *d_A, float *d_B, float *d_C){

    const int BLOCK_SIZE_M = 96;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_Y = 6;
    const int THREAD_SIZE_X = 4;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    if (N % BLOCK_SIZE_N != 0)
        dimGrid.x++;
    if (M % BLOCK_SIZE_M != 0)
        dimGrid.y++;

    FP32_maxmul_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);

}

extern "C"
void FP64_maxmul(const int M, const int N, const int K, double *d_A, double *d_B, double *d_C){

    const int BLOCK_SIZE_M = 96;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_Y = 6;
    const int THREAD_SIZE_X = 4;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    if (N % BLOCK_SIZE_N != 0)
        dimGrid.x++;
    if (M % BLOCK_SIZE_M != 0)
        dimGrid.y++;

    FP64_maxmul_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);

}

extern "C"
void INT32_maxmul(const int M, const int N, const int K, int *d_A, int *d_B, int *d_C){

    const int BLOCK_SIZE_M = 96;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_Y = 6;
    const int THREAD_SIZE_X = 4;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    if (N % BLOCK_SIZE_N != 0)
        dimGrid.x++;
    if (M % BLOCK_SIZE_M != 0)
        dimGrid.y++;

    INT32_maxmul_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);

}

extern "C"
void INT64_maxmul(const int M, const int N, const int K, long *d_A, long *d_B, long *d_C){

    const int BLOCK_SIZE_M = 96;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_Y = 6;
    const int THREAD_SIZE_X = 4;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    if (N % BLOCK_SIZE_N != 0)
        dimGrid.x++;
    if (M % BLOCK_SIZE_M != 0)
        dimGrid.y++;

    INT64_maxmul_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);

}