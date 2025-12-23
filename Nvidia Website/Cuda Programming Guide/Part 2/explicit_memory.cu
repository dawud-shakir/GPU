// Compile: nvcc -O3 -gencode arch=compute_75,code=sm_75 explicit_memory.cu -o explicit_memory

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
// #include <cuda/cmath>    // only in newer versions
#include <cmath>

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

// //explicit-memory-begin
// void explicitMemExample(int vectorLength)
// {
//     // Pointers for host memory
//     float* A = nullptr;
//     float* B = nullptr;
//     float* C = nullptr;
//     float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    
//     // Pointers for device memory
//     float* devA = nullptr;
//     float* devB = nullptr;
//     float* devC = nullptr;

//     //Allocate Host Memory using cudaMallocHost API. This is best practice
//     // when buffers will be used for copies between CPU and GPU memory
//     cudaMallocHost(&A, vectorLength*sizeof(float));
//     cudaMallocHost(&B, vectorLength*sizeof(float));
//     cudaMallocHost(&C, vectorLength*sizeof(float));

//     // Initialize vectors on the host
//     initArray(A, vectorLength);
//     initArray(B, vectorLength);

//     // start-allocate-and-copy
//     // Allocate memory on the GPU
//     cudaMalloc(&devA, vectorLength*sizeof(float));
//     cudaMalloc(&devB, vectorLength*sizeof(float));
//     cudaMalloc(&devC, vectorLength*sizeof(float));

//     // Copy data to the GPU
//     cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
//     cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
//     cudaMemset(devC, 0, vectorLength*sizeof(float));
//     // end-allocate-and-copy

//     // Launch the kernel
//     int threads = 256;
//     // int blocks = cuda::ceil_div(vectorLength, threads);
//     int blocks = (vectorLength + threads - 1) / threads;
    
//     vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
//     // wait for kernel execution to complete
//     cudaDeviceSynchronize();

//     // Copy results back to host
//     cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

//     // Perform computation serially on CPU for comparison
//     serialVecAdd(A, B, comparisonResult, vectorLength);

//     // Confirm that CPU and GPU got the same answer
//     if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
//     {
//         printf("Explicit Memory: CPU and GPU answers match\n");
//     }
//     else
//     {
//         printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
//     }

//     // clean up
//     cudaFree(devA);
//     cudaFree(devB);
//     cudaFree(devC);
//     cudaFreeHost(A);
//     cudaFreeHost(B);
//     cudaFreeHost(C);
//     free(comparisonResult);
// }
// //explicit-memory-end

//explicit-memory-begin
void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    
    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    CUDA_CHECK(cudaMallocHost(&A, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&B, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&C, vectorLength*sizeof(float)));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));

    // Copy data to the GPU
    CUDA_CHECK(cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemset(devC, 0, vectorLength*sizeof(float)));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    // int blocks = cuda::ceil_div(vectorLength, threads);
    int blocks = (vectorLength + threads - 1) / threads;
    
    // Warmup
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    // check error sate after kernel launch
    CUDA_CHECK(cudaGetLastError());
    // wait for kernel execution to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    const int iters = 50;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);        
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float ms_per = ms / iters;

    printf("n=%d, blocks=%d, threads=%d\n", vectorLength, blocks, threads);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", ms_per, iters);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault));

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFree(devC));
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(B));
    CUDA_CHECK(cudaFreeHost(C));
    free(comparisonResult);
}
//explicit-memory-end


int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    explicitMemExample(vectorLength);		
    return 0;
}