#include <stdio.h>      // printf
#include <stdlib.h>     // atoi
// #include <cuda.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                    \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

__global__ void matmul(float* A, float* B, float* C, int n)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col >= n || row >= n)
        return;

    float sum = C[row*n + col];
    for (int k = 0; k < n; ++k)
        sum += A[row*n + k] * B[k*n + col];
    C[row*n + col] = sum;
}

void run_matmat(float* A, float* B, float* C, int n)
{

}

void print_matrix(float* A, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
            printf("%d ", A[i*n + j]);
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        printf("Usage: %s n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int bytes = sizeof(float) * n * n;
    float *A, *B, *C;

    CUDA_CHECK(cudaManagedMalloc(&A, bytes));
    CUDA_CHECK(cudaManagedMalloc(&B, bytes));
    CUDA_CHECK(cudaManagedMalloc(&C, bytes));
    
    CUDA_CHECK(cudaMemSet(C, 0, bytes));

    for (int i = 0; i < bytes; ++i)
    {
        A[i] = 2.0;
        B[i] = 0.5;
    }

    printf("A:\n");
    print_matrix(A, n);
    printf("\nB:\n");
    print_matrix(B, n);

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    

    return 0;
}