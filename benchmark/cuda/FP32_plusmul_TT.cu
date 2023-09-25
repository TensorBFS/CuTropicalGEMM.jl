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


#define OFFSET_row(row, col, ld) ((row) * (ld) + (col))
#define OFFSET_col(row, col, ld) ((col) * (ld) + (row))

#define OPERATOR_ADD(a, b) (a + b)
#define OPERATOR_MUT(a, b) (a * b)
#define PADDING_FP 0
#define PADDING_INT 0


template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_M, // height of block of C that each thread calculate
    const int THREAD_SIZE_N
    > 
__global__ void kernel_TTN( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
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

    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_M * THREAD_SIZE_N] = {0};
    float regs_a[THREAD_SIZE_M] = {0};
    float regs_b[THREAD_SIZE_N] = {0};
    
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
                    K )] : 0;
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
                    N )] : 0;
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
                    accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)] = OPERATOR_ADD(regs_a[thread_y] * regs_b[thread_x], accum[OFFSET_row(thread_y, thread_x, THREAD_SIZE_N)]);
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

void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

bool check(const float *A,
    const float *B,
    const float *C,
    const float *D,
    int m, int n, int k) {
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            int a = rand() % m;
            int b = rand() % n;
            float sum = 0.f;
            sum = C[OFFSET_col(a, b, m)];
            for (int p = 0; p < k; ++p) {
                sum = OPERATOR_ADD(sum, OPERATOR_MUT(A[OFFSET_row(a, p, k)], B[OFFSET_row(p, b, n)]));
            }

            if (std::fabs(sum - D[OFFSET_col(a, b, m)]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", a, b, sum, D[OFFSET_col(a, b, m)]);
                return false;
            }
        }
    }

    return true;
}

bool check_all(const float *A,
    const float *B,
    const float *C,
    const float *D,
    int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        int a = i;
        for (int j = 0; j < n; ++j) {
            int b = j;
            float sum = 0.f;
            sum = C[OFFSET_col(a, b, m)];
            for (int p = 0; p < k; ++p) {
                sum = OPERATOR_ADD(sum, OPERATOR_MUT(A[OFFSET_row(a, p, k)], B[OFFSET_row(p, b, n)]));
            }

            if (std::fabs(sum - D[OFFSET_col(a, b, m)]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", a, b, sum, D[OFFSET_col(a, b, m)]);
                return false;
            }
        }
    }

    return true;
}

int main() {
    
    // printf("Benchmarking MutAddFP32. \n");
    // 
    // first check-all for small matirx
    int m = 4096;
    int n = 4096;
    int k = 4096;
    // printf("check_all by small matrix %d, %d, %d\n", m, n, k);

    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_N, BLOCK_SIZE_M / THREAD_SIZE_M);
    dim3 dimGrid(n / BLOCK_SIZE_N, m / BLOCK_SIZE_M);
    if (n % BLOCK_SIZE_N != 0)
        dimGrid.x++;
    if (m % BLOCK_SIZE_M != 0)
        dimGrid.y++;

    float *h_A, *h_B, *h_C, *h_D;
    float *d_A, *d_B, *d_C;

    // printf("benchmarking by large matrix %d, %d, %d\n", m, n, k);


    cudaMallocHost(&h_A, m * k * sizeof(float));
    cudaMallocHost(&h_B, k * n * sizeof(float));
    cudaMallocHost(&h_C, m * n * sizeof(float));
    cudaMallocHost(&h_D, m * n * sizeof(float));
    random_init(h_A, m * k);
    random_init(h_B, k * n);
    random_init(h_C, m * n);

    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyDefault);

    kernel_TTN<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(h_D, d_C, m * n * sizeof(float), cudaMemcpyDefault);
    bool chk = check(h_A, h_B, h_C, h_D, m, n, k);
    // printf("Matrix_C random check: %s\n", chk ? "OK" : "Failed");

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    const int n_iter = 100;
    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        kernel_TTN<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
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
    printf("FP32_plusmul_TT: %f TFLOPS\n", tflops);
    printf("time per iteration: %f ms\n \n", ms / n_iter);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_D);
}
