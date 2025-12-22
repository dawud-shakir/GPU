#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("Detected CUDA Device:\n");
    printf("  Name: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);

    // printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    // printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    // printf("  CUDA Cores: %d\n", prop.multiProcessorCount * _ConvertSMVer2Cores(prop.major, prop.minor));
    
    // printf("  Clock Rate: %.2f GHz\n", prop.clockRate / 1.0e6);
    // printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1.0e6);
    // printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    // printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);


    // printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    // printf("  Max Threads Dim: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    // printf("  Max Grid Size: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);   

    // printf("  Shared Memory per Block: %d KB\n", prop.sharedMemPerBlock / 1024);
    // printf("  Registers per Block: %d\n", prop.regsPerBlock);
    // printf("  Warp Size: %d\n", prop.warpSize);

    
    return 0;
}