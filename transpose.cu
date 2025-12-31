/*
transpose.cu
----------------
A minimal CUDA C++ implementation of matrix transpose.

Compile: nvcc -O3 -gencode arch=compute_75,code=sm_75 <cuda file> -o <executable>
Run: ./<executable>
*/

#include <cblas.h> // for verification with cblas_dgemm
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>  // memset
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                 \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)


/* macro to index a 1D memory array with 2D indices in column-major order */
/* ld is the leading dimension, i.e. the number of rows in the matrix     */

#define INDX(row, col, ld) (((col) * (ld)) + (row))

__global__ void gpu_transpose_1(const float* __restrict__ A,
    int n, int m, float* __restrict__ A_T)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (ty < n && tx < m) {
        // A_T[tx * m + ty] = A[ty * n + tx];  // strided access
        A_T[ty * n + tx] = A[tx * m + ty];  // coalesced/non-strided access (faster)
    }
    return;
}


// CUDA kernel for naive matrix transpose
__global__ void gpu_transpose_2(const float* A, int n, int m, float* A_T)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;

    if (ty < n && tx < m)
        A_T[INDX(tx, ty, n)] = A[INDX(ty, tx, m)];
    
    return;
}

/* CUDA kernel for shared memory matrix transpose */

/* definitions of thread block size in X and Y directions */
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
__global__ void gpu_transpose_tiled( const float* a, 
                                int n, int m,
                                float *c )
{

    /* declare a statically allocated shared memory array */

    __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];

    /* determine my row and column indices for the error checking code */

    const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
    const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

    /* determine my row tile and column tile index */

    const int tileX = blockDim.x * blockIdx.x;
    const int tileY = blockDim.y * blockIdx.y;

    if( myRow < n && myCol < m )
    {
        /* read from global memory into shared memory array */
        smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileX + threadIdx.x, tileY + threadIdx.y, m )];
    } /* end if */

    /* synchronize the threads in the thread block */
    __syncthreads();

    if( myRow < n && myCol < m )
    {
        /* write the result from shared memory to global memory */
        c[INDX( tileY + threadIdx.x, tileX + threadIdx.y, n )] = smemArray[threadIdx.y][threadIdx.x];
    } /* end if */
    return;
}


// void call_transpose_fast(int n, float* a, float* c)
// {
//     // Kernel launch config
//     // int threads = 32;   // 32 threads per block x AND 32 threads per block y
//     // int blocks = (n + threads - 1) / threads;

//     // dim3 threadsPerBlock(32, 32);
//     // dim3 blocksPerGrid( (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
//     //                    (n + threadsPerBlock.y - 1) / threadsPerBlock.y );

//     // int n = width * height;
//     int threadsPerBlock = 1024;
//     int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

//     // Warmup
//     transpose_fast<<<blocksPerGrid, threadsPerBlock>>>(n, a, c);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // Timed runs
//     const int iters = 50;
//     cudaEvent_t start, stop;
//     CUDA_CHECK(cudaEventCreate(&start));
//     CUDA_CHECK(cudaEventCreate(&stop));

//     CUDA_CHECK(cudaEventRecord(start));
//     for (int it = 0; it < iters; ++it) {
//         transpose_fast<<<blocksPerGrid, threadsPerBlock>>>(n, a, c);
//     }
//     CUDA_CHECK(cudaEventRecord(stop));
//     CUDA_CHECK(cudaEventSynchronize(stop));
//     CUDA_CHECK(cudaGetLastError());

//     printf("Completed %d iterations.\n", iters);
//     float ms = 0.0f;
//     CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
//     float ms_per = ms / iters;
//     printf("%s (n=%d): %.3f ms (%.3f ms per run)\n",
//         __func__, n, ms, ms_per);

//     CUDA_CHECK(cudaEventDestroy(start));
//     CUDA_CHECK(cudaEventDestroy(stop));
// }

// void call_transpose_slow(int width, int height, const float* __restrict__ input, float* __restrict__ output)
// {
//     // Kernel launch config
//     int n = width * height;
//     int threads = 256;
//     int blocks = (n + threads - 1) / threads;

//     // Warmup
//     transpose_slow<<<blocks, threads>>>(input, output, width, height);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // Timed runs
//     const int iters = 50;
//     cudaEvent_t start, stop;
//     CUDA_CHECK(cudaEventCreate(&start));
//     CUDA_CHECK(cudaEventCreate(&stop));

//     CUDA_CHECK(cudaEventRecord(start));
//     for (int it = 0; it < iters; ++it) {
//         transpose_slow<<<blocks, threads>>>(input, output, width, height);
//     }
//     CUDA_CHECK(cudaEventRecord(stop));
//     CUDA_CHECK(cudaEventSynchronize(stop));
//     CUDA_CHECK(cudaGetLastError());

//     printf("Completed %d iterations.\n", iters);
//     float ms = 0.0f;
//     CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
//     float ms_per = ms / iters;
//     printf("%s (n=%d): %.3f ms (%.3f ms per run)\n",
//         __func__, n, ms, ms_per);

//     CUDA_CHECK(cudaEventDestroy(start));
//     CUDA_CHECK(cudaEventDestroy(stop));
// }

void cpu_transpose(const float* A, int n, int m, float* AT)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            AT[j * n + i] = A[i * m + j];
}

#define STR1(x) #x
#define STR(x)  STR1(x)

#ifndef FUNCTION_NAME
#define FUNCTION_NAME gpu_transpose_tiled
#endif

int main(int argc, char** argv)
{
    // Problem size (default 1024)
    int n = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int m = n;

    // Host allocations

    float *A = nullptr, *gpu_result = nullptr;
    CUDA_CHECK(cudaMallocManaged(&A, n * m * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&gpu_result, m * n * sizeof(float)));

    // Data
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            A[i * m + j] = (float)(i * m + j);
        }
    }

    // Zero output matrix
    std::memset(gpu_result, 0, m * n * sizeof(float));

    // Kernel launch config
    dim3 threads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);

    // Warmup
    FUNCTION_NAME<<<blocks, threads>>>(A, n, m, gpu_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    const int iters = 50;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        FUNCTION_NAME<<<blocks, threads>>>(A, n, m, gpu_result);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    printf("Completed %d iterations.\n", iters);
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float ms_per = ms / iters;
    printf("%s (n=%d): %.3f ms (%.3f ms per run)\n",
        STR(FUNCTION_NAME), n, ms, ms_per);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Verify
    float* cpu_result = nullptr;
    CUDA_CHECK(cudaMallocManaged(&cpu_result, m * n * sizeof(float)));
    cpu_transpose(A, n, m, cpu_result);
    bool match = true;
    for (int i = 0; i < m * n; ++i) {
        if (std::fabs(cpu_result[i] - gpu_result[i]) > 1e-5) {
            match = false;
            printf("Mismatch at index %d: CPU %f, GPU %f\n", i, cpu_result[i], gpu_result[i]);
            break;
        }
    }
    if (match) {
        printf("Transpose result matches CPU reference.\n");
    } else {
        printf("Transpose result does NOT match CPU reference.\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(cpu_result));
    CUDA_CHECK(cudaFree(gpu_result));

    return 0;
}
