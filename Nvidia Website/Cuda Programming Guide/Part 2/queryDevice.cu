#include <stdio.h>
#include <cuda.h>

int main() {
    int device_id;
    cudaDeviceProp prop;


    // Query the shared memory size per SM and per thread block
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);

    printf("Shared Memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);

    // Shared Memory per SM: 65536 bytes
    // Shared Memory per Block: 49152 bytes

    //cudaFuncSetCacheConfig()


    printf("Registers Per Multiprocessor (SM): %d\n", prop.regsPerMultiprocessor);
    printf("Registers Per Block: %d\n", prop.regsPerBlock);

    return 0;
}