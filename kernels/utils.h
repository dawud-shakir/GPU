#ifndef KERNELS_UTILS_H
#define KERNELS_UTILS_H
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                 \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)


cudaDeviceProp getDeviceProperties(int dev = 0) {
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    return prop;
}

#endif // KERNELS_UTILS_H

