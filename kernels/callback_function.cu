// callback_function.cu
// demo of CUDA host callback function

#include <cuda_runtime.h>
#include <cstdio>
#include "utils.h"

__global__ void kernel(unsigned long long cycles) {
    // some GPU work
    unsigned long long start = clock64();
    while (clock64() - start < cycles) {
        // intentional spin
    }
}

// macro only needed for windows compatibility
void CUDART_CB host_callback(void* userData)
{
    int id = *reinterpret_cast<int*>(userData);
    printf("Host callback executed after kernel %d\n", id);
}

int main()
{
    // // ~1e9 cycles ≈ 0.5–1 second depending on GPU clock
    // unsigned long long cycles = 1ULL << 30;

    const unsigned long long cycles_per_ms = 
            (unsigned long long)getDeviceProperties().clockRate; // kHz = cycles/ms
    

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int callback_id = 0;



    kernel<<<256, 256, 0, stream>>>(cycles_per_ms * 2000); // ~2 seconds
    CUDA_CHECK(cudaGetLastError());

    // Enqueue host function in *this* stream
    CUDA_CHECK(cudaLaunchHostFunc(stream, host_callback, &callback_id));

    // More work in the same stream (will wait for callback)
    kernel<<<256, 256, 0, stream>>>(cycles_per_ms * 0); // ~0 seconds
    CUDA_CHECK(cudaGetLastError());

    // Let everything finish
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
