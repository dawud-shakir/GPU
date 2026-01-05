// async_streams.cu

// Demo two streams running asynchronously 

// #include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"

__global__ void kernel1()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
        printf("Exiting kernel1...\n");
}

__global__ void kernel2()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
        printf("Exiting kernel2...\n");
}

static bool cpu_work_done = false;
__host__ bool allCPUWorkDone()
{
    if ( cpu_work_done ) 
        return true;
    printf("Starting CPU work...\n");
    struct timespec ts = { .tv_sec = 5, .tv_nsec = 0 };
    nanosleep(&ts, NULL);
    printf("CPU work done.\n");
    cpu_work_done = true;
    return true;
}

int main(int argc, char* argv[])
{
    char* host = NULL;
    size_t bytes = 1000;
    CUDA_CHECK(cudaMallocHost(&host, bytes * sizeof(char)));

    char* device = NULL;
    CUDA_CHECK(cudaMalloc(&device, bytes * sizeof(char)));

    cudaStream_t compute_stream;    // processing stream
    cudaStream_t copying_stream;    // memory transfer stream
    bool copyStarted = false;

    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&copying_stream));


    int blocks = 32; 
    int grid = (bytes + blocks - 1) / blocks;

    kernel1<<<grid, blocks, 0, compute_stream>>>();
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t event; 
    CUDA_CHECK(cudaEventCreate(&event));                
    CUDA_CHECK(cudaEventRecord(event, compute_stream));

    kernel2<<<grid, blocks, 0, compute_stream>>>();
    CUDA_CHECK(cudaGetLastError());

    printf("Entering while loop...\n");

    while ( not allCPUWorkDone() || not copyStarted )
    {
        // Peek to see if kernel1 has completed on compute stream
        if ( cudaEventQuery(event) == cudaSuccess )
        {
            // Add a asynchronous copy operation to the other stream
            printf("Starting async copy from device to host...\n");
            CUDA_CHECK(cudaMemcpyAsync(host, device, bytes*sizeof(char), cudaMemcpyDeviceToHost, copying_stream));
            copyStarted = true;
        }
    }

    printf("Exited while loop...\n");
    
    // Wait for both streams to finish
    CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaStreamDestroy(copying_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaFree(device));
    CUDA_CHECK(cudaFreeHost(host));

    return 0;
}

