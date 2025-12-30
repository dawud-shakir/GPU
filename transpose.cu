/*
transpose.cu
----------------
A minimal CUDA C++ implementation of matrix transpose.

Compile: nvcc -O3 -gencode arch=compute_75,code=sm_75 <cuda file> -o <executable>
Run: ./<executable>
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda/atomic>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                 \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

__global__ void transpose_slow(const float* __restrict__ input,
                          float* __restrict__ output,
                          int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;  

    if (x < width && y < height) {
        int input_idx = y * width + x;
        int output_idx = x * height + y;
        output[output_idx] = input[input_idx];
    }
}



/* macro to index a 1D memory array with 2D indices in column-major order */
/* ld is the leading dimension, i.e. the number of rows in the matrix     */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )


// // CUDA kernel for naive matrix transpose
// __global__ void transpose_fast(int m, const float* a, float* c )
// {
//     int myCol = blockDim.x * blockIdx.x + threadIdx.x;
//     int myRow = blockDim.y * blockIdx.y + threadIdx.y;

//     if( myRow < m && myCol < m )
//     {
//         c[INDX( myCol, myRow, m )] = a[INDX( myRow, myCol, m )];
//     } /* end if */
//     return;
// } // end naive_cuda_transpose

/* CUDA kernel for shared memory matrix transpose */

/* definitions of thread block size in X and Y directions */

// #define THREADS_PER_BLOCK_X 32
// #define THREADS_PER_BLOCK_Y 32
__global__ void transpose_fast( int m,
                                float *a,
                                float *c )
{

    /* declare a statically allocated shared memory array */

    __shared__ float smemArray[32][32];

    /* determine my row and column indices for the error checking code */

    const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
    const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

    /* determine my row tile and column tile index */

    const int tileX = blockDim.x * blockIdx.x;
    const int tileY = blockDim.y * blockIdx.y;

    if( myRow < m && myCol < m )
    {
        /* read from global memory into shared memory array */
        smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileX + threadIdx.x, tileY + threadIdx.y, m )];
    } /* end if */

    /* synchronize the threads in the thread block */
    __syncthreads();

    if( myRow < m && myCol < m )
    {
        /* write the result from shared memory to global memory */
        c[INDX( tileY + threadIdx.x, tileX + threadIdx.y, m )] = smemArray[threadIdx.y][threadIdx.x];
    } /* end if */
    return;

} /* end transpose_fast */


void call_transpose_fast(int n, float* a, float* c)
{
    // Kernel launch config
    // int threads = 32;   // 32 threads per block x AND 32 threads per block y
    // int blocks = (n + threads - 1) / threads;
    
    dim2 threadsPerBlock(32, 32);
    dim2 numBlocks( (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y );



    // Warmup
    transpose_fast<<<numBlocks, threadsPerBlock>>>(n, a, c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    const int iters = 50;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        transpose_fast<<<numBlocks, threadsPerBlock>>>(n, a, c);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    printf("Completed %d iterations.\n", iters);
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float ms_per = ms / iters;
    printf("%s (n=%d): %.3f ms (%.3f ms per run)\n",
           __func__, n, ms, ms_per);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}



void call_transpose_slow(int width, int height, const float* __restrict__ input, float* __restrict__ output)
{
    // Kernel launch config
    int n = width * height;
    int threadsPerBlock = 256;
    int blocks = (n + threads - 1) / threads;

    // Warmup
    transpose_slow<<<blocks, threads>>>(input, output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    const int iters = 50;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        transpose_slow<<<blocks, threads>>>(input, output, width, height);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    printf("Completed %d iterations.\n", iters);
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float ms_per = ms / iters;
    printf("%s (n=%d): %.3f ms (%.3f ms per run)\n",
           __func__, n, ms, ms_per);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}


int main(int argc, char** argv)
{
    // Problem size (default 1<<10)
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 10);    // 1024

    // Host allocations

    float *A = nullptr, *A_T = nullptr; 
    CUDA_CHECK(cudaMallocHost(&A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&A_T, n * n * sizeof(float)));

    // Initialize host data
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = (float)(i * n + j);
            A_T[j * n + i] = 0.0f;
        }
    }

    // Device allocations
    float *d_A = nullptr, *d_AT = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_AT, n * n * sizeof(float)));

    // H2D copies
    CUDA_CHECK(cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_AT, A_T, n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Time transposes
    call_transpose_slow(n, n, d_A, d_AT);
    call_transpose_fast(n, d_A, d_AT);

    // Cleanup

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_AT));
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(A_T));

    return 0;
}
