#include <cuda_runtime.h>

void vecAddSequential(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; ++i) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

// Compute sum C = A + B
// Each thread performs one pair-wise addition

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n)
{
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    
    // Launch ceil(n/256) blocks of 256 threads each
    vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
     
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(int argc, char* argv[])
{
    int n = argc > 1 ? atoi(argv[1]) : 200000;


    int size = n * sizeof(float);
    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);

    float* C_cpu = (float*)malloc(size);
    float* C_gpu = (float*)malloc(size);

    for (int i = 0; i < n; ++i) {
        A[i] = (float)i;
        B[i] = (float)(i * 2);
    }

    vecAddSequential(A, B, C_cpu, n);
    vecAdd(A, B, C_gpu, n);

    for (int i = 0; i < n; ++i) {
        if (C_cpu[i] != C_gpu[i]) {
            printf("Mismatch at index %d: CPU %f, GPU %f\n", i, C_cpu[i], C_gpu[i]);
            return -1;
        }
    }
    printf("Results match!\n");

    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);

    return 0;
}