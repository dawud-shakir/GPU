

/*

LSQR: A minimal CUDA C++ implementation of the SAXPY operation.

Compile: nvcc -O3 -gencode arch=compute_75,code=sm_75 <cuda file> -o <executable>
Run: ./<executable>
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                 \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

// static void matvec_local(const std::vector<double>& A_local,
//                          std::int64_t mloc, std::int64_t n,
//                          const double* x, double* y)
// {
//     for (std::int64_t ii = 0; ii < nloc; ++ii) {
//         const double* row = &A_local[(size_t)ii * (size_t)m];
//         double sum = 0.0;
//         for (std::int64_t j = 0; j < m; ++j) sum += row[j] * x[j];
//         y[ii] = sum;
//     }
// }

__global__ void gpu_matvec(const float* A, int n, int m, const float* x, float* y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n)
        return;

    const float* row = &A[(size_t)tid * (size_t)m];
    float sum = 0.0f;
    for (int j = 0; j < m; ++j)
        sum += row[j] * x[j];
    y[tid] = sum;
}

__global__ void sumsq(const float* x, int n, float* result)
{
   


    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

cuda::atomic_ref<float, cuda::thread_scope_device> result_ref(result);
if (tid < n)    
    result_ref.fetch_add(x[tid])

    float val = 0.0f;
    if (idx < n)
        val = x[idx] * x[idx];

    sdata[tid] = val;
    __syncthreads();

    // reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // one thread per block accumulates
    if (tid == 0)
        atomicAdd(result, sdata[0]);
}

// __global__ void sumsq_gridstride(const float* __restrict__ x,
//                                           int n,
//                                           float* __restrict__ sumsq_out)
// {
//     extern __shared__ float sdata[];

//     int tid  = threadIdx.x;
//     int gtid = blockIdx.x * blockDim.x + tid;
//     int stride = blockDim.x * gridDim.x;

//     // grid-stride accumulation of sum of squares
//     float local = 0.0f;
//     for (int i = gtid; i < n; i += stride) {
//         float v = x[i];
//         local += v * v;
//     }

//     // reduce within block in shared memory
//     sdata[tid] = local;
//     __syncthreads();

//     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s)
//             sdata[tid] += sdata[tid + s];
//         __syncthreads();
//     }

//     // one atomic per block
//     if (tid == 0)
//         atomicAdd(sumsq_out, sdata[0]);
// }



float gpu_norm(const float* d_x, int n)
{
    float* sum = nullptr;
    CUDA_CHECK(cudaMallocManaged(&sum, sizeof(float)));
    CUDA_CHECK(cudaMemset(sum, 0, sizeof(float)));

    int threadsPerBlock = 256;
    int blocksPerGrid = getBlockSize(n, threadsPerBlock);

    sumsq<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_x, n, sum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float norm = std::sqrt(*sum);   
    CUDA_CHECK(cudaFree(sum));

    return norm;
}

float cpu_norm(const float* x, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double v = x[i];
        sum += v * v;
    }
    return std::sqrt(sum);
}

void cpu_matvec(const float* A, int n, int m, const float* x, float* y)
{
    for (int ii = 0; ii < n; ++ii) {
        const float* row = &A[(size_t)ii * (size_t)m];
        float sum = 0.0f;
        for (int j = 0; j < m; ++j)
            sum += row[j] * x[j];
        y[ii] = sum;
    }
}

/* definitions of thread block size in X and Y directions */

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

/* macro to index a 1D memory array with 2D indices in column-major order */
/* ld is the leading dimension, i.e. the number of rows in the matrix     */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* CUDA kernel for shared memory matrix transpose */

__global__ void smem_cuda_transpose(int m,
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

} /* end smem_cuda_transpose */

int main(int argc, char** argv)
{
    // Problem size (default 1<<10)
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 10);    // 1024

    // Host allocations

    float *A = nullptr, *x = nullptr, *y = nullptr; 
    CUDA_CHECK(cudaMallocHost(&A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&x, n * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&y, n * sizeof(float)));

    // Initialize host data
    for (int i = 0; i < n; ++i) {
        x[i] = i * 1.0f;
        for (int j = 0; j < n; ++j)
            A[i * n + j] = 2.0f;
    }
        printf("At line %d\n", __LINE__);


    CUDA_CHECK(cudaMemset(y, 0, n * sizeof(float)));

    // Device allocations
    float *d_A = nullptr, *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));

    // H2D copies
    CUDA_CHECK(cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch config
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    // Warmup
    gpu_matvec<<<blocks, threads>>>(d_A, n, n, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    const int iters = 50;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        gpu_matvec<<<blocks, threads>>>(d_A, n, n, d_x, d_y);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    printf("Completed %d iterations.\n", iters);
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float ms_per = ms / iters;
    printf("GPU matrix-vector multiplication (n=%d): %.3f ms (%.3f ms per run)\n",
           n, ms, ms_per);

    // Verify results against CPU
    double gpu_sum = 0.0;
    CUDA_CHECK(cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost)); // D2H copy
    for (int i = 0; i < n; ++i)
        gpu_sum += y[i];

    cpu_matvec(A, n, n, x, y);
    double cpu_sum = 0.0;
    for (int i = 0; i < n; ++i)
        cpu_sum += y[i];

    double rel_err = std::abs(gpu_sum - cpu_sum) / std::abs(cpu_sum);
    if (rel_err > 1e-5) {
        printf("Result verification failed! rel_err = %.6e\n", rel_err);
        return 1;
    }
    printf("Result verification passed! rel_err = %.6e\n", rel_err);

    printf("GPU Norm of y: %.6f\n", gpu_norm(d_y, n));
    printf("CPU Norm of y: %.6f\n", cpu_norm(y, n));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(x));
    CUDA_CHECK(cudaFreeHost(y));

    return 0;
}
