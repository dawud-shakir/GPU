// coorperative_kernel.cu (by Charlie)

#include <cuda_runtime.h>
#include <cstdio>

static void checkCuda(cudaError_t e, const char* msg)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

// l2_norm_cooperative.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Kernel: compute sum of squares, then (after grid sync) take sqrt on GPU.
__global__ void l2_norm_coop_kernel(const float* x, const int n, float* out_norm)
{
    cg::grid_group grid = cg::this_grid();

    // dynamic shared memory: one float per thread
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop so we can use fewer blocks than needed for full coverage
    float local = 0.0f;
    for (int i = gtid; i < n; i += stride) {
        float v = x[i];
        local += v * v;
    }

    // Reduce within block into sdata[0]
    sdata[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // One atomic add per block into out_norm (temporarily holds sumsq)
    if (tid == 0) atomicAdd(out_norm, sdata[0]);

    // ✅ Grid-wide barrier: wait until *all blocks* finish atomicAdd
    grid.sync();

    // One thread finalizes sqrt (and writes final norm back to out_norm)
    if (grid.thread_rank() == 0) {
        *out_norm = sqrtf(*out_norm);
    }
}


int main()
{
    // Example: allocate a vector on device (fill however you want)
    const int n = 1 << 20;
    float* d_x = nullptr;
    float* d_norm = nullptr;
    checkCuda(cudaMalloc(&d_x, n * sizeof(float)), "cudaMalloc d_x");
    checkCuda(cudaMalloc(&d_norm, sizeof(float)), "cudaMalloc d_norm");

    // IMPORTANT: out_norm is used first as sumsq accumulator, so zero it
    checkCuda(cudaMemset(d_norm, 0, sizeof(float)), "cudaMemset d_norm");

    // Check cooperative launch support
    cudaDeviceProp prop{};
    int dev = 0;
    checkCuda(cudaGetDevice(&dev), "cudaGetDevice");
    checkCuda(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");

    if (!prop.cooperativeLaunch) {
        std::fprintf(stderr, "Device does NOT support cooperativeLaunch.\n");
        return 0;
    }

    // Choose launch size.
    // With cooperative launch, gridDim may be limited (must fit concurrently).
    int threads = 256;

    // A safe, simple choice: use the maximum number of resident blocks
    // per SM for this kernel is complicated; start conservative:
    int blocks = prop.multiProcessorCount;  // one block per SM

    // Cooperative launch requires using cudaLaunchCooperativeKernel
    void* args[] = {
        (void*)&d_x,     // const float*
        (void*)&n,       // int
        (void*)&d_norm   // float*
    };

    size_t shmem_bytes = threads * sizeof(float);

    checkCuda(cudaLaunchCooperativeKernel(
                  (void*)l2_norm_coop_kernel,
                  blocks, threads,
                  args, shmem_bytes, nullptr),
              "cudaLaunchCooperativeKernel");

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    float h_norm = 0.0f;
    checkCuda(cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy d_norm->h_norm");

    std::printf("L2 norm = %f\n", h_norm);

    cudaFree(d_x);
    cudaFree(d_norm);
    return 0;
}
