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


// cal offset from row col and ld , in col-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((col) * (ld) + (row))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


// GEMM for Col-Major matrix
// default of julia is Col-Major and default of C++ is Row-Major
template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_M, // height of block of C that each thread calculate
    const int THREAD_SIZE_N,  // width of block of C that each thread calculate
    const int M, // M, N, K are size of the matrix
    const int K,
    const int N,
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void MatMul( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C
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
    
    // row number and col number that needs to be loaded blockIdx.y this thread
    const int A_TILE_COL = tid / BLOCK_SIZE_M;
    const int B_TILE_COL = tid / BLOCK_SIZE_K;

    const int A_TILE_ROW = tid % BLOCK_SIZE_M;
    const int B_TILE_ROW = tid % BLOCK_SIZE_K;
    
    // col stride that thread uses to load multiple rows of a tile
    // how many cols that the threads load in one iteration
    const int A_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_M;
    const int B_TILE_COL_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;

    // number of threads in M and N direction (used when calculating C)
    // const int A_S = BLOCK_SIZE_M / THREAD_SIZE_M;
    // const int B_S = BLOCK_SIZE_N / THREAD_SIZE_N;

    // (can not unroll since K can not be determined at this point) -- now it is determined XD
    #pragma unroll
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {

        // load A from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K ; i += A_TILE_COL_STRIDE) {
            const int row = BLOCK_SIZE_M * blockIdx.x + A_TILE_ROW ;
            const int col = A_TILE_COL + i + tile_idx;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                As[OFFSET(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = row < M && col < K ? A[OFFSET(row, col, M)] : 0;
            } else {
                As[OFFSET(A_TILE_ROW, i + A_TILE_COL, BLOCK_SIZE_M)] = A[OFFSET(row, col, M)];
            }
        }

        // load B from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_N; i += B_TILE_COL_STRIDE) {
            const int row = tile_idx + B_TILE_ROW;
            const int col = BLOCK_SIZE_N * blockIdx.y + i + B_TILE_COL;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                Bs[OFFSET(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = row < K && col < N ? B[OFFSET(row, col, K)] : 0;
            } else {
                Bs[OFFSET(B_TILE_ROW, i + B_TILE_COL, BLOCK_SIZE_K)] = B[OFFSET(row, col, K)];
            }
        }

        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            #pragma unroll
            for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
                #pragma unroll
                for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
                    accum[OFFSET(thread_m, thread_n, THREAD_SIZE_M)] += As[OFFSET(threadIdx.x * THREAD_SIZE_M + thread_m, k, BLOCK_SIZE_M)] * Bs[OFFSET(k, threadIdx.y * THREAD_SIZE_N + thread_n, BLOCK_SIZE_K)];
                    // accum[OFFSET(thread_m, thread_n, THREAD_SIZE_M)] = max(As[OFFSET(threadIdx.x * THREAD_SIZE_M + thread_m, k, BLOCK_SIZE_M)] + Bs[OFFSET(k, threadIdx.y * THREAD_SIZE_N + thread_n, BLOCK_SIZE_K)], accum[OFFSET(thread_m, thread_n, THREAD_SIZE_M)]);
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_n = 0; thread_n < THREAD_SIZE_N; ++thread_n) {
        #pragma unroll
        for (int thread_m = 0; thread_m < THREAD_SIZE_M; ++thread_m) {
            const int col = BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * threadIdx.y + thread_n;
            const int row = BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * threadIdx.x + thread_m;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                if (row < M && col < N) {
                    // C[OFFSET(row, col, M)] = max(accum[OFFSET(thread_m, thread_n, THREAD_SIZE_M)], C[OFFSET(row, col, M)]);
                    C[OFFSET(row, col, M)] += accum[OFFSET(thread_m, thread_n, THREAD_SIZE_M)];
                }
            } else {
                // C[OFFSET(row, col, M)] = max(accum[OFFSET(thread_m, thread_n, THREAD_SIZE_M)], C[OFFSET(row, col, M)]);
                C[OFFSET(row, col, M)] += accum[OFFSET(thread_m, thread_n, THREAD_SIZE_M)];
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
    for (int i = 0; i < 20; ++i) {
        int a = rand() % m;
        for (int j = 0; j < 20; ++j) {
            int b = rand() % n;
            float sum = 0.f;
            sum = C[OFFSET(a, b, m)];
            for (int p = 0; p < k; ++p) {
                sum +=  A[OFFSET(a, p, m)] * B[OFFSET(p, b, k)];
            }

            if (std::fabs(sum - D[OFFSET(a, b, m)]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", a, b, sum, D[OFFSET(a, b, m)]);
                return false;
            }
        }
    }

    return true;
}

int main() {
    const int m = 4096;
    const int n = 4096;
    const int k = 4096;
    const int n_iter = 100;

    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 96;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 6;
    const bool ENABLE_DOUBLE_BUFFER = false;

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
    MatMul<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N, m, n, k, ENABLE_DOUBLE_BUFFER> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C);

    cudaMemcpy(h_D, d_C, m * n * sizeof(float), cudaMemcpyDefault);
    bool chk = check(h_A, h_B, h_C, h_D, m, n, k);
    printf("Matrix_C check: %s\n", chk ? "OK" : "Failed");

    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        MatMul<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N, m, n, k, ENABLE_DOUBLE_BUFFER> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    long workload = n_iter * long(m) * n * k * 2;
    double tflops = (double(workload) / 1e12) / (double(ms) / 1e3);
    printf("SGEMM with const type Matrix size: %f TFLOPS\n", tflops);
    printf("time per iteration: %f ms\n", ms / n_iter);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_D);
}

// However, it is shown we will not be able to unroll the loop correctly if we want to use this kernel as a C function correctly...
// but in this script, we did not use any float4 type of some thing like that, it will be great to try to write a julia version of it
// extern "C"
// void SGemmMatMul(const int m, const int n, const int k, const int BM, const int BN, const int BK, const int TM, const int TN, float *d_A, float *d_B, float *d_C){

//     dim3 dimBlock(BM / TM, BN / TN);
//     dim3 dimGrid(m / BM, n / BN);
//     if (m % BM != 0)
//         dimGrid.x++;
//     if (n % BN != 0)
//         dimGrid.y++;

//     const bool ENABLE_DOUBLE_BUFFER = false;

//     MatMul<BM, BK, BN, TM, TN, m, n, k, ENABLE_DOUBLE_BUFFER> 
//     <<< dimGrid, dimBlock >>>(d_A, d_B, d_C);

// }