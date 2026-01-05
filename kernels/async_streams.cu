// async_streams.cu

// Demo two streams running asynchronously 

// #include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__global__ void kernel1()
{

}

__global__ void kernel2()
{

}

__host__ bool allCPUWorkDone()
{
    struct timespec ts = { .tv_sec = 2, .tv_nsec = 0 };
    nanosleep(&ts, NULL);
    return false;
}

int main(int argc, char* argv[])
{
    char* host = NULL;
    size_t bytes = 1000;
    cudaMallocHost(&host, bytes * sizeof(char));

    char* device = NULL;
    cudaMalloc(&device, bytes * sizeof(char));

    cudaStream_t compute_stream;    // processing stream
    cudaStream_t copying_stream;    // memory transfer stream
    bool copyStarted = false;

    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&copying_stream);


    int blocks = 32; 
    int grid = (bytes + blocks - 1) / blocks;

    kernel1<<<grid, blocks, 0, compute_stream>>>();

    cudaEvent_t event; 
    cudaEventCreate(&event);                
    cudaEventRecord(event, compute_stream);

    kernel2<<<grid, blocks, 0, compute_stream>>>();


    while ( not allCPUWorkDone() || not copyStarted )
    {
        // Peek to see if kernel1 has completed on compute stream
        if ( cudaEventQuery(event) == cudaSuccess )
        {
            // Add a asynchronous copy operation to the other stream
            cudaMemcpyAsync(host, device, bytes*sizeof(char), cudaMemcpyDeviceToHost, copying_stream);
            copyStarted = true;
        }
    }

    
    // Wait for both streams to finish
    cudaStreamDestroy(copying_stream);
    cudaStreamDestroy(compute_stream);

    return 0;
}

