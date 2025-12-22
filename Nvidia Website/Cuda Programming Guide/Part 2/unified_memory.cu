// Compile: nvcc -O3 -gencode arch=compute_75,code=sm_75 unified_memory.cu -o unified_memory

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
// #include <cuda/cmath> // only in newer versions
#include <cmath>

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

// //unified-memory-begin
// void unifiedMemExample(int vectorLength)
// {
//     // Pointers to memory vectors
//     float* A = nullptr;
//     float* B = nullptr;
//     float* C = nullptr;
//     float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

//     // Use unified memory to allocate buffers
//     cudaMallocManaged(&A, vectorLength*sizeof(float));
//     cudaMallocManaged(&B, vectorLength*sizeof(float));
//     cudaMallocManaged(&C, vectorLength*sizeof(float));

//     // Initialize vectors on the host
//     initArray(A, vectorLength);
//     initArray(B, vectorLength);

//     // Launch the kernel. Unified memory will make sure A, B, and C are
//     // accessible to the GPU
//     int threads = 256;
//     // int blocks = cuda::ceil_div(vectorLength, threads);
//     int blocks = (vectorLength + threads - 1) / threads;
    
//     vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
//     // Wait for the kernel to complete execution
//     cudaDeviceSynchronize();

//     // Perform computation serially on CPU for comparison
//     serialVecAdd(A, B, comparisonResult, vectorLength);

//     // Confirm that CPU and GPU got the same answer
//     if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
//     {
//         printf("Unified Memory: CPU and GPU answers match\n");
//     }
//     else
//     {
//         printf("Unified Memory: Error - CPU and GPU answers do not match\n");
//     }

//     // Clean Up
//     cudaFree(A);
//     cudaFree(B);
//     cudaFree(C);
//     free(comparisonResult);

// }
// //unified-memory-end


//unified-memory-begin
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    // int blocks = cuda::ceil_div(vectorLength, threads);
    int blocks = (vectorLength + threads - 1) / threads;
    
    // Warmup
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Timed runs
    const int iters = 50;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int it = 0; it < iters; ++it) {
        vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);        
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float ms_per = ms / iters;

    printf("n=%d, blocks=%d, threads=%d\n", vectorLength, blocks, threads);
    printf("Avg kernel time: %.4f ms (over %d iters)\n", ms_per, iters);

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

}
//unified-memory-end

int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemExample(vectorLength);		
    return 0;
}