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


#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define OPERATOR_ADD(a, b) (a || b)
#define OPERATOR_MUT(a, b) (a && b)
#define PADDING_FP false


template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X
    > 
__global__ void MatMul( 
    bool * __restrict__ A,
    bool * __restrict__ B,
    bool * __restrict__ C, 
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

    __shared__ bool As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ bool Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    bool accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {PADDING_FP};
    
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

void random_init(bool *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = bool(rand() % 2);
    }
}

int main() {
    
    printf("Benchmarking AndOrBool. \n");
    
    int m = 4096;
    int n = 4096;
    int k = 4096;
    printf("benchmarking by large matrix %d, %d, %d\n", m, n, k);

    const int BLOCK_SIZE_M = 96;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_Y = 6;
    const int THREAD_SIZE_X = 4;

    bool *h_A, *h_B, *h_C, *h_D;

    bool *d_A, *d_B, *d_C;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(n / BLOCK_SIZE_N, m / BLOCK_SIZE_M);
    if (n % BLOCK_SIZE_N != 0)
        dimGrid.x++;
    if (m % BLOCK_SIZE_M != 0)
        dimGrid.y++;

    cudaMallocHost(&h_A, m * k * sizeof(bool));
    cudaMallocHost(&h_B, k * n * sizeof(bool));
    cudaMallocHost(&h_C, m * n * sizeof(bool));
    cudaMallocHost(&h_D, m * n * sizeof(bool));
    random_init(h_A, m * k);
    random_init(h_B, k * n);
    random_init(h_C, m * n);

    cudaMalloc(&d_A, m * k * sizeof(bool));
    cudaMalloc(&d_B, k * n * sizeof(bool));
    cudaMalloc(&d_C, m * n * sizeof(bool));

    cudaMemcpy(d_A, h_A, m * k * sizeof(bool), cudaMemcpyDefault);
    cudaMemcpy(d_B, h_B, k * n * sizeof(bool), cudaMemcpyDefault);
    cudaMemcpy(d_C, h_C, m * n * sizeof(bool), cudaMemcpyDefault);

    MatMul<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    const int n_iter = 100;
    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        MatMul<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    long workload = n_iter * long(m) * n * k * 2;
    double tflops = (double(workload) / 1e12) / (double(ms) / 1e3);
    printf("SGEMM with dynamic Matrix size: %f TFLOPS\n", tflops);
    printf("time per iteration: %f ms\n", ms / n_iter);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_D);
}
