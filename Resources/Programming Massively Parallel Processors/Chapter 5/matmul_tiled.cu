#include <cuda_runtime.h>
#include <stdlib.h>     // rand
#include <stdio.h>      // printf
#include <float.h>      // FLT_EPSILON

#include <string.h>     // memset

#define TILE_WIDTH 16


__global__ void matrixMulKernel(float* M, float* N, float* P, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;    int by = blockIdx.y;
    int tx = threadIdx.x;   int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph)
    {
        // Collaborative loading of M and N tiles into shared memory
        Mds[ty][tx] = M[Row*Width +  ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx]; // blockDim is TILE_WIDTH x TILE_WIDTH
        __syncthreads();
    }
    P[Row*Width + Col] = Pvalue;
}

__global__ void matrixMulKernel_boundries(float* M, float* N, float* P, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;    int by = blockIdx.y;
    int tx = threadIdx.x;   int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph)
    {
        // Collaborative loading of M and N tiles into shared memory
        if (Row < Width && (ph*TILE_WIDTH + tx < Width))
            Mds[ty][tx] = M[Row*Width +  ph*TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;

        if ((ph*TILE_WIDTH + ty) < Width && Col < Width)
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        else
            Nds[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx]; // blockDim is TILE_WIDTH x TILE_WIDTH
        __syncthreads();
    }
    if (Row < Width && Col < Width)
        P[Row*Width + Col] = Pvalue;
}

void matrixMul(float* M, float* N, float* P, int Width) {
    int size = Width * Width * sizeof(float);

    float *M_d, *N_d, *P_d;
    cudaMalloc((void **)&M_d, size);
    cudaMalloc((void **)&N_d, size);
    cudaMalloc((void **)&P_d, size);

    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(Width / (float)blockDim.x), ceil(Width / (float)blockDim.y));

    printf("TILE_WIDTH: %d\n", TILE_WIDTH);
    printf("blockDim: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

    matrixMulKernel<<<gridDim, blockDim>>>(M_d, N_d, P_d, Width);

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

static void matrixMul_timed(float* M, float* N, float* P, int Width);


__global__ void matrixMulKernel_external_smem(float* M, float* N, float* P, int Width, unsigned Mds_sz, unsigned Nds_sz)
{
    extern __shared__ char Mds_Nds[];
    
    float* Mds = (float *) Mds_Nds;
    float* Nds = (float *) Mds_Nds + (TILE_WIDTH*TILE_WIDTH);

    int bx = blockIdx.x;    int by = blockIdx.y;
    int tx = threadIdx.x;   int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph)
    {
        // Collaborative loading of M and N tiles into shared memory
        Mds[ty*TILE_WIDTH + tx] = M[Row*Width +  ph*TILE_WIDTH + tx];
        Nds[ty*TILE_WIDTH + tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty*TILE_WIDTH + k] * Nds[k*TILE_WIDTH + tx]; // blockDim is TILE_WIDTH x TILE_WIDTH
        __syncthreads();
    }
    P[Row*Width + Col] = Pvalue;
}



void matrixMul_dynamic_smem(float* M, float* N, float* P, int Width)
{
    size_t size = Width * Width * sizeof(float);
    
    size_t smem_size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
    printf("smem bytes: %zu\n", smem_size);

    float *M_d, *N_d, *P_d;
    cudaMalloc((void **)&M_d, size);
    cudaMalloc((void **)&N_d, size);
    cudaMalloc((void **)&P_d, size);

    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(Width / (float)blockDim.x), ceil(Width / (float)blockDim.y));

    printf("TILE_WIDTH: %d\n", TILE_WIDTH);
    printf("blockDim: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);


    matrixMulKernel_external_smem<<<gridDim, blockDim, smem_size>>>(M_d, N_d, P_d, Width, smem_size/2, smem_size/2);

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}



void matrixMulCPU(float* M, float* N, float* P, int Width) {
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < Width; ++k) {
                const float a = M[row*Width + k];
                const float b = N[k*Width + col];
                sum += a*b;
            }
            P[row*Width + col] = sum;
        }
    }
}

// void matrixMul(float* M, float* N, float* P, int J, int K, int L) { }

// void matrixMulCPU(const float* restrict M, const float* restrict N, float* P, int J, int K, int L) {
//     // M: J x K
//     // N: K x L
//     // P: J x L

//     for (int row = 0; row < J; ++row) {
//         for (int col = 0; col < L; ++col) {
//             float sum = 0.0f;

//             // The shared (contracted) dimension is K:
//             // M[row, k] * N[k, col], k < K
//             for (int k = 0; k < K; ++k) {
//                 const float a = M[row*K + k];
//                 const float b = N[k*L + col];
//                 sum += a*b;
//             }
//             P[row*L + col] = sum;
//         }
//     }
// }

// Kernel to use for timing function
#ifndef KERNEL
#define KERNEL matrixMulKernel_boundries
#endif

int main(int argc, char* argv[])
{
    // The maximum number of threads in a block is 1024.
    const int Width = argc > 1 ? atoi(argv[1]) : 16;
    printf("Matrix size: %d x %d\n", Width, Width);

    int size = Width * Width * sizeof(float);
    
    float* M = (float*)malloc(size);
    float* N = (float*)malloc(size);
    float* P_gpu = (float*)calloc(Width * Width, sizeof(float));
    float* P_cpu = (float*)calloc(Width * Width, sizeof(float));

    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            const int index = row * Width + col;
            M[index] = rand() / (float)RAND_MAX; //2.0 //row * Width + col;
            N[index] =rand() / (float)RAND_MAX; //4.0 //row * Width + col;
        }
    }

    matrixMul_dynamic_smem(M, N, P_gpu, Width);
    matrixMulCPU(M, N, P_cpu, Width);

    // const float abs_tol = 1e-6f;    // 1e-3
    // const float rel_tol = 1e-5f;    // 1e-3

    const float eps = FLT_EPSILON;                   // ~1.19e-7
    float max_val = 0.f;
    for (int i = 0; i < Width*Width; ++i) max_val = fmaxf(max_val, fabs(P_cpu[i]));
    float abs_tol = eps * Width * max_val * 10.f;    // scale factor (10) as safety margin
    float rel_tol = fmaxf(1e-5f, eps * Width * 10.f);
    printf("abs_tol: %f, rel_tol: %f\n\n", abs_tol, rel_tol);

    bool match = true;
    float max_abs_diff = 0; float max_rel_diff = 0;
    for (int i = 0; i < Width * Width; ++i) {
        const float abs_diff = fabs(P_gpu[i] - P_cpu[i]);
        const float rel_diff = abs_diff / fmaxf(fmaxf(fabs(P_gpu[i]), fabs(P_cpu[i])), 1e-8f);

        max_abs_diff = fmaxf(max_abs_diff, abs_diff);
        max_rel_diff = fmaxf(max_rel_diff, rel_diff);
        if (abs_diff > abs_tol && rel_diff > rel_tol) {
            printf("Mismatch at (%d, %d)\n", (int)(i/Width), i%Width);
            printf("abs_diff: %f, rel_diff: %f\n\n", abs_diff, rel_diff);
            printf("P_gpu=%f, P_cpu=%f\n", P_gpu[i], P_cpu[i]);
            match = false;
            break;
        }
    }

    if (match) {
        printf("GPU and CPU results match\n");
        printf("max_abs_diff=%f, max_rel_diff=%f\n", max_abs_diff, max_rel_diff);
    }
    else {
        printf("GPU and CPU results do not match!\n");
    }

    free(M);
    free(N);
    free(P_gpu);
    free(P_cpu);

    return 0;
}

void matrixMul_timed(float* M, float* N, float* P, int Width) {
    int size = Width * Width * sizeof(float);

    float *M_d, *N_d, *P_d;
    cudaMalloc((void **)&M_d, size);
    cudaMalloc((void **)&N_d, size);
    cudaMalloc((void **)&P_d, size);

    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(Width / (float)blockDim.x), ceil(Width / (float)blockDim.y));

    printf("TILE_WIDTH: %d\n", TILE_WIDTH);
    printf("blockDim: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);


    // Warm
    const int warmups = 1;

    for (int i = 0; i < warmups; ++i)
        KERNEL<<<gridDim, blockDim>>>(M_d, N_d, P_d, Width);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 5;
    cudaEventRecord(start);
    KERNEL<<<gridDim, blockDim>>>(M_d, N_d, P_d, Width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0;
    cudaEventElapsedTime(&ms, start, stop);
    double avg_ms = ms / iters;

    double flops = 2.0 * (double)Width * (double)Width * (double)Width;
    double gflops = (flops / (avg_ms / 1e3)) / 1e9;

    printf("matrixMulKernel\n");
    printf("Avg GEMM time: %.3f ms\n", avg_ms);
    printf("Throughput:    %.1f GFLOP/s\n", gflops);


    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}
