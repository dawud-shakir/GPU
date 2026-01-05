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

#define INDX_C(row, col, ld) ( ( (col) * (ld) ) + (row) ) // column-major order

#define INDX_R(row, col, ld) ( ( (row) * (ld) ) + (col) ) // row-major order

#define INDX INDX_C

// __global__ void gpu_transpose_1(const float* __restrict__ A,
//     int n, int m, float* __restrict__ A_T)
// {
//     int tx = blockIdx.x * blockDim.x + threadIdx.x;
//     int ty = blockIdx.y * blockDim.y + threadIdx.y;

//     if (ty < n && tx < m) {
//         // A_T[tx * m + ty] = A[ty * n + tx];  // strided access
//         A_T[ty * n + tx] = A[tx * m + ty];  // coalesced/non-strided access (faster)
//     }
//     return;
// }

// Column major version
__global__ void gpu_transpose_cm(const float* __restrict__ A,
    int n, int m, float* __restrict__ A_T)
{
    int col = blockIdx.x * blockDim.x;
    int tx = threadIdx.x;

    int row = blockIdx.y * blockDim.y;
    int ty = threadIdx.y;
    

    if (ty < n && tx < m)
        // Reads are strided, but writes are contiguous across threads (coalesced)
        // -> fewer write transactions
        A_T[(row + ty) * n + (col + tx)] = A[(col + tx) * m + (row + ty)]; 
    
    return;
}

// Row major version
__global__ void gpu_transpose_rm(const float* __restrict__ A,
    int n, int m, float* __restrict__ A_T)
{
    int col = blockIdx.x * blockDim.x;
    int tx = threadIdx.x;

    int row = blockIdx.y * blockDim.y;
    int ty = threadIdx.y;
    

    if (ty < n && tx < m) {
        // Reads are contiguous but writes are strided 
        //      ->   expensive write traffic (often incurs extra memory ops)
        A_T[(col + tx) * n + (row + ty)] = A[(row + ty) * m + (col + tx)];  
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


// Tiled version: threads read contiguous input rows into a shared tile
// then write the transposed tile so both reads and writes are coalesced 
// (this is the standard fast approach).
__global__ void gpu_transpose_tiled( const float* a, 
                                int n, int m,
                                float *c )
{

    /* declare a statically allocated shared memory array */


    // A 32x32 tile pattern causes heavy bank conflicts unless the second dimension is padded.
    // Add +1 to the second dimension to avoid bank conflicts.
    __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y+1]; 

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
    }

    /* synchronize the threads in the thread block */
    __syncthreads();

    if( myRow < n && myCol < m )
    {
        /* write the result from shared memory to global memory */
        c[INDX( tileY + threadIdx.x, tileX + threadIdx.y, n )] = smemArray[threadIdx.y][threadIdx.x];
    }
    return;
}

void cpu_transpose(const float* A, int n, int m, float* AT)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            AT[j * n + i] = A[i * m + j];
}

#define STR1(x) #x
#define STR(x)  STR1(x)

#ifndef FUNCTION
#define FUNCTION gpu_transpose_tiled
#endif

int main(int argc, char** argv)
{
    // Problem size (defaults 1024)
    int n = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int m = (argc > 2) ? std::atoi(argv[2]) : 1024;

    // Host allocations

    // Matrix (n x m) to transpose
    float *A = nullptr;
    CUDA_CHECK(cudaMallocManaged(&A, n * m * sizeof(float)));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            A[i * m + j] = (float)(i * m + j);
        }
    }

    float* gpu_result = nullptr;
    CUDA_CHECK(cudaMallocManaged(&gpu_result, m * n * sizeof(float)));
    // Zero output matrix
    std::memset(gpu_result, 0, m * n * sizeof(float));

    // Kernel launch config
    dim3 threads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);

    // Warmup
    FUNCTION<<<blocks, threads>>>(A, n, m, gpu_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    const int iters = 50;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        FUNCTION<<<blocks, threads>>>(A, n, m, gpu_result);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    printf("Completed %d iterations.\n", iters);
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float ms_per = ms / iters;
    printf("%s (n=%d, m=%d): %.3f ms (%.3f ms per run)\n",
        STR(FUNCTION), n, m, ms, ms_per);

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
        printf("\x1b[31mTranspose result does NOT match CPU reference.\x1b[0m\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(cpu_result));
    CUDA_CHECK(cudaFree(gpu_result));

    return 0;
}
