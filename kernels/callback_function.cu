// callback_function.cu
// demo of CUDA host callback function

#include <cuda_runtime.h>
#include <cstdio>

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
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int callback_id = 0;

    // ~1e9 cycles ≈ 0.5–1 second depending on GPU clock
    unsigned long long cycles = 1ULL << 30;

    kernel<<<256, 256, 0, stream>>>(cycles);

    // Enqueue host function in *this* stream
    cudaLaunchHostFunc(stream, host_callback, &callback_id);

    // More work in the same stream (will wait for callback)
    kernel<<<256, 256, 0, stream>>>(0);

    // Let everything finish
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    return 0;
}
