/*

2.2.3.2. Shared Memory

Shared memory is a memory space that is accessible by all threads in a
thread block. It is physically located on each SM and uses the same
physical resource as the L1 cache, the unified data cache. The data in
shared memory persists throughout the kernel execution. Shared memory can
be considered a user-managed scratchpad for use during kernel execution.
*/

#include <stdio.h>
#include <cuda.h>

// assuming blockDim.x is 128
__global__ void example_syncthreads(int* input_data, int* output_data) {
    __shared__ int shared_data[128];
    // Every thread writes to a distinct element of 'shared_data':
    shared_data[threadIdx.x] = input_data[threadIdx.x];

    // All threads synchronize, guaranteeing all writes to 'shared_data' are ordered 
    // before any thread is unblocked from '__syncthreads()':
    __syncthreads();

    // A single thread safely reads 'shared_data':
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_data[i];
        }
        output_data[blockIdx.x] = sum;
    }
}

int main() {
    const int numBlocks = 10;
    const int blockSize = 128;
    int *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, numBlocks * blockSize * sizeof(int));
    cudaMalloc(&d_output, numBlocks * sizeof(int));

    // Launch kernel
    example_syncthreads<<<numBlocks, blockSize>>>(d_input, d_output);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}