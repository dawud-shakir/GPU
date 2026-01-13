
#include <cuda_runtime.h>

#include <stdio.h>

int main(int argc, char* argv[])
{
    int devCount;
    cudaGetDeviceCount(&devCount);

    cudaDeviceProp devProp;
    for (unsigned int i = 0; i < devCount; ++i) {
        cudaDeviceProperties(&devProp, i);

        // Decide if device has sufficient resources/capabilities

        
        printf("Maximum number of threads per block: %d\n", devProp.maxThreadsPerBlock);
    }
}