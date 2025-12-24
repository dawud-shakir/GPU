#include <stdio.h>      // printf
#include <stdlib.h>     // atoi
#include <string.h>     // memset
#include <cuda.h>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                    \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)


__global__ void gpu_matmul(float* A, float* B, float* C, int n)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col >= n || row >= n)
        return;

    float val = 0;  // Register variable
    val = C[row*n + col];
    for (int k = 0; k < n; ++k)
        val += A[row*n + k] * B[k*n + col];
    C[row*n + col] = val;
}

// Prof's
// __global__ void gpu_matmul(int n, float* A, float* B, float* C)
// {
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     float val = 0;

//     if (row < n && col < n) {
//         val = C[row * n + col];
//         for (int k = 0; k < n; k++)
//             val += A[row * n + k] * B[k * n + col];
//         C[row * n + col] = val;
//     }
// }

void cpu_matmul(int n, float* A, float* B, float* C)
{
    float val;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            val = C[row * n + col];
            for (int k = 0; k < n; k++)
                val += A[row * n + k] * B[k * n + col];
            C[row * n + col] = val;
        }
    }
}
double sum(float* C, int n)
{
    double s = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            s += C[i*n+j];
    return s;
}

void print_matrix(float* A, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
            printf("%.1f ", A[i*n + j]);
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        printf("Usage: %s <n>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int bytes = sizeof(float) * n * n;
    float *A, *B, *C;
    float *comparisonResult;

    CUDA_CHECK(cudaMallocManaged(&A, bytes));
    CUDA_CHECK(cudaMallocManaged(&B, bytes));
    CUDA_CHECK(cudaMallocManaged(&C, bytes));
    comparisonResult = (float*)malloc(bytes);

    for (int i = 0; i < bytes; ++i)
    {
        A[i] = 2.0;
        B[i] = 0.5;
    }

    // printf("A:\n");
    // print_matrix(A, n);
    // printf("\nB:\n");
    // print_matrix(B, n);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;   // integer ceil

    // Warmup
    gpu_matmul<<<blocks, threads>>>(A, B, C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    int iters = 50;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaMemset(C, 0, bytes));
        // Launch kernel
        gpu_matmul<<<blocks, threads>>>(A, B, C, n);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float ms_per = ms / iters;

    printf("n=%d, blocks=%d, threads=%d\n", n, blocks, threads);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", ms_per, iters);

    // Perform computation serially on CPU for comparison
    cpu_matmul(A, B, comparisonResult, n);

    // Confirm that CPU and GPU got the same answer
    double gpu_sum = sum(C, n);
    double cpu_sum = sum(comparisonResult, n);
    printf("GPU: %.0f, CPU: %.0f\n", gpu_sum, cpu_sum);

    // n=4 output: GPU: 16, CPU 64
    // n=256 output: GPU: 392704, CPU: 67108864


    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    free(comparisonResult);

    return 0;
}