// this script is a common SGEMM method for 

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


// cal OFFSET_col from row col and ld , in col-major matrix, ld is the height of the matrix
#define OFFSET_col(row, col, ld) ((col) * (ld) + (row))
#define OFFSET_row(row, col, ld) ((row) * (ld) + (col))

// #define OPERATOR_ADD(a, b) (max(a, b))
// #define OPERATOR_MUT(a, b) (a + b)
// #define PADDING -INFINITY

#define OPERATOR_ADD(a, b) max(a, b)
#define OPERATOR_MUT(a, b) (a + b)
#define PADDING -INFINITY

// GEMM for Col-Major matrix
// default of julia is Col-Major and default of C++ is Row-Major
template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_M, // height of block of C that each thread calculate
    const int THREAD_SIZE_N  // width of block of C that each thread calculate
    > 
__global__ void Tropical_Gemm_kernel( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
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
    __shared__ float As[BLOCK_SIZE_MK]; // avoid bank conflict
    __shared__ float Bs[BLOCK_SIZE_KN];

    // registers for C
    float accum[THREAD_SIZE_MN] = {0};
    float A_reg[THREAD_SIZE_M] = {0};
    float B_reg[THREAD_SIZE_N] = {0};
    
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
                A_reg[thread_m] = As[OFFSET_col(threadIdx.x * THREAD_SIZE_M + thread_m, k, BLOCK_SIZE_M)];
            }

            #pragma unroll
            for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
                B_reg[thread_n] = Bs[OFFSET_row(k, threadIdx.y * THREAD_SIZE_N + thread_n, BLOCK_SIZE_N)];
            }

            #pragma unroll
            for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
                #pragma unroll
                for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
                    accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)] = OPERATOR_ADD(OPERATOR_MUT(A_reg[thread_m], B_reg[thread_n]), accum[OFFSET_col(thread_m, thread_n, THREAD_SIZE_M)]);
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

void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
        // data[i] = 1;
    }
}

bool check(const float *A,
    const float *B,
    const float *C,
    const float *D,
    int m, int n, int k) {
    for (int i = 0; i < 100; ++i) {
        int a = rand() % m;
        for (int j = 0; j < 100; ++j) {
            int b = rand() % n;
            float sum = 0.f;
            sum = C[OFFSET_col(a, b, m)];
            for (int p = 0; p < k; ++p) {
                sum = OPERATOR_ADD(OPERATOR_MUT(A[OFFSET_col(a, p, m)], B[OFFSET_row(p, b, n)]), sum);
            }

            if (std::fabs(sum - D[OFFSET_col(a, b, m)]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", a, b, sum, D[OFFSET_col(a, b, m)]);
                // return false;
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
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            sum = C[OFFSET_col(i, j, m)];
            for (int p = 0; p < k; ++p) {
                sum = OPERATOR_ADD(OPERATOR_MUT(A[OFFSET_col(i, p, m)], B[OFFSET_row(p, j, n)]), sum);
            }

            if (std::fabs(sum - D[OFFSET_col(i, j, m)]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, D[OFFSET_col(i, j, m)]);
                // return false;
            }
        }
    }

    return true;
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

    dim3 dimBlock(BLOCK_SIZE_M / THREAD_SIZE_M, BLOCK_SIZE_N / THREAD_SIZE_N);
    dim3 dimGrid(m / BLOCK_SIZE_M, n / BLOCK_SIZE_N);
    if (m % BLOCK_SIZE_M != 0)
        dimGrid.x++;
    if (n % BLOCK_SIZE_N != 0)
        dimGrid.y++;

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

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // warmup
    Tropical_Gemm_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(h_D, d_C, m * n * sizeof(float), cudaMemcpyDefault);
    bool chk = check(h_A, h_B, h_C, h_D, m, n, k);
    // printf("Matrix_C check: %s\n", chk ? "OK" : "Failed");


    // code for benchmarking

    const int n_iter = 100;
    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        Tropical_Gemm_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
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
    printf("FP32_maxplus_NT: %f TFLOPS\n", tflops);
    printf("time per iteration: %f ms\n \n", ms / n_iter);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_D);
}
