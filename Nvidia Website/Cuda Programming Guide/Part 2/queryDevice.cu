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

    //cudaFuncSetCacheConfig()


    return 0;
}