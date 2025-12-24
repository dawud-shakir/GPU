/*
2.1.10.1. Launching with Clusters in Triple Chevron Notation

A thread block cluster can be enabled in a kernel either using a
compile-time kernel attribute using __cluster_dims__(X,Y,Z) or using the
CUDA kernel launch API cudaLaunchKernelEx. 

The example below shows how to launch a cluster using a compile-time kernel 
attribute. The cluster size specified by the kernel attribute is fixed at compile 
time and the kernel can then be launched using the classical <<< , >>>.
If a kernel uses a compile-time cluster size, the cluster size cannot
be modified when launching the kernel.

*/

#include <stdio.h>
#include <cuda.h>

#define N 1024

// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}