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


__device__ __inline__ void gpu_matmul(float* A, float* B, float* C, int n)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col >= n || row >= n)
        return;

    float val = 0.0f;  // registered
    for (int k = 0; k < n; ++k)
        val += A[row*n + k] * B[k*n + col];
    C[row*n + col] = val;
}

// // Prof's
// __global__ void gpu_matmul(float* A, float* B, float* C, int n)
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

#ifndef TILE_SIZE
// Prefer thread counts that are multiples of the warp size (32) for good occupancy.
#define TILE_SIZE 32
#endif

#if TILE_SIZE <= 0 || (TILE_SIZE * TILESIZE) > 1024
#error "THREADS_PER_BLOCK (TILE_SIZE * TILESIZE) must be in (0,1024]"
#endif

// #ifndef THREADS_PER_BLOCK_X
// #define THREADS_PER_BLOCK_X 32
// #endif
// #ifndef THREADS_PER_BLOCK_Y
// #define THREADS_PER_BLOCK_Y 8
// #endif

// #if (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y) > 1024
// #error "THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y must be <= 1024"
// #endif


__global__ void gpu_matmul_tiled_nonsquare(float* A, float* B, float* C, int n)
{
    __shared__ float A_shared[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X];
    __shared__ float B_shared[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X];

    int ty    = threadIdx.y;
    int tx    = threadIdx.x;

    const int row = ty + blockIdx.y * blockDim.y;
    const int col = tx + blockIdx.x * blockDim.x;

    if (row >= n || col >= n)
        return;

    // if (n < TILE_SIZE)
    // {
    //     // Use the regular matmul when n is smaller than the tile.
    //     gpu_matmul(A, B, C, n);
    //     return;
    // }

    // if (n % TILE_SIZE != 0)
    // {
    //     // For simplicity, this kernel assumes n is a multiple of TILE_SIZE
    //     // In practice, you would handle the boundary conditions here

    //     // Here's a more efficient way would involve padding or conditional loading

    // }

    float val = 0.0f;  // registered
    int n_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int i = 0; i < n_tiles; ++i)
    {
        int a_col = i * TILE_SIZE + tx;
        int b_row = i * TILE_SIZE + ty;

        // Pad with zeros if block is less than tile size
        A_shared[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        B_shared[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads();    // wait for all threads before reading shared memory

        for (int k = 0; k < TILE_SIZE; ++k)
            val += A_shared[ty][k] * B_shared[k][tx];
        __syncthreads();
    }
    C[row*n + col] = val;
}

__global__ void gpu_matmul_tiled(float* A, float* B, float* C, int n)
{
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    int ty    = threadIdx.y;
    int tx    = threadIdx.x;

    const int row = ty + blockIdx.y * blockDim.y;
    const int col = tx + blockIdx.x * blockDim.x;

    if (row >= n || col >= n)
        return;

    if (n < TILE_SIZE)
    {
        // Use the regular matmul when n is smaller than the tile.
        gpu_matmul(A, B, C, n);
        return;
    }

    if (n % TILE_SIZE != 0)
    {
        // For simplicity, this kernel assumes n is a multiple of TILE_SIZE
        // In practice, you would handle the boundary conditions here

        // Here's a more efficient way would involve padding or conditional loading

    }

    float val = 0.0f;  // registered
    for (int i = 0; i < n / TILE_SIZE; ++i)
    {
        A_shared[ty][tx] = A[row*n + i*TILE_SIZE + tx];
        B_shared[ty][tx] = B[(i*TILE_SIZE + ty)*n + col];
        __syncthreads();    // wait for all threads before reading shared memory

        for (int k = 0; k < TILE_SIZE; ++k)
            val += A_shared[ty][k] * B_shared[k][tx];
        __syncthreads();
    }
    C[row*n + col] = val;
}

void cpu_matmul(float* A, float* B, float* C, int n)
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
    double s = 0.0;
    for (int i = 0; i < n * n; ++i)
        s += C[i];
    return s;
}

bool approximatelyEqual(float* A, float* B, int n, float epsilon=0.00001)
{
    for (int i = 0; i < n; ++i)
    {
        if (fabs(A[i] - B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
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
    float *A, *B, *gpu_C;
    float *cpu_C;

    CUDA_CHECK(cudaMallocManaged(&A, bytes));
    CUDA_CHECK(cudaMallocManaged(&B, bytes));
    CUDA_CHECK(cudaMallocManaged(&gpu_C, bytes));

    cpu_C = (float*)malloc(bytes);
    if (0 == cpu_C) {
        fprintf(stderr, "malloc error\n");
        return 1;
    }
    memset(cpu_C, 0, bytes);

    for (int i = 0; i < n * n; ++i)
    {
        A[i] = 2.0f;
        B[i] = 0.5f;
    }

    int threads = TILE_SIZE;   // warps are 32 threads
    dim3 blockDim(threads, threads);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);

    // Warmup
    gpu_matmul_tiled_nonsquare<<<gridDim, blockDim>>>(A, B, gpu_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    int iters = 50;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaMemset(gpu_C, 0, bytes));
        // Launch kernel
        gpu_matmul_tiled_nonsquare<<<gridDim, blockDim>>>(A, B, gpu_C, n);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float ms_per = ms / iters;

    printf("n=%d, blocks=(%d,%d), threads=(%d,%d)\n", n, gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", ms_per, iters);

    // Perform computation serially on CPU for comparison
    cpu_matmul(A, B, cpu_C, n);

    // Confirm that GPU and CPU got the same answer
    // double gpu_sum = sum(gpu_C, n);
    // double cpu_sum = sum(cpu_C, n);

    float gpu_sum = 0.0;
    float cpu_sum = 0.0;
    for (int i = 0; i < n * n; ++i)
    {
        gpu_sum += gpu_C[i];
        cpu_sum += cpu_C[i];
    }


    printf("Checksum\nGPU: %.0f, CPU: %.0f\n", gpu_sum, cpu_sum);

    if (approximatelyEqual(gpu_C, cpu_C, n*n))
        printf("Tiling: GPU and CPU answers match\n");
    else
        printf("Tiling: Error - GPU and CPU answers do not match\n");

    // Clean Up
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(gpu_C));
    free(cpu_C);

    return 0;
}