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

#endif // KERNELS_UTILS_H

