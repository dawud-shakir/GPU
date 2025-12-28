/*

 A minimal Colab-ready CUDA C++ template.

Compile: nvcc -O3 -gencode arch=compute_75,code=sm_75 <cuda file> -o <executable>
Run: ./<executable>
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                    \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

__global__ void saxpy(const float a, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main(int argc, char** argv) {
  // Problem size (default 1<<20)
  int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
  float a = 2.5f;

  // Host allocations
  float* h_x = (float*)std::malloc(n * sizeof(float));
  float* h_y = (float*)std::malloc(n * sizeof(float));
  if (!h_x || !h_y) {
    fprintf(stderr, "Host malloc failed\n");
    return 1;
  }

  // Initialize host data
  for (int i = 0; i < n; ++i) {
    h_x[i] = std::sin(i) * 0.5f;
    h_y[i] = std::cos(i) * 0.5f;
  }

  // Device allocations
  float *d_x = nullptr, *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));

  // H2D copies
  CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice));

  // Kernel launch config
  int threads = 256;
  int blocks  = (n + threads - 1) / threads;

  // Warmup
  saxpy<<<blocks, threads>>>(a, d_x, d_y, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed runs
  const int iters = 50;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int it = 0; it < iters; ++it) {
    saxpy<<<blocks, threads>>>(a, d_x, d_y, n);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  float ms_per = ms / iters;

  // D2H copy back a small slice to verify
  CUDA_CHECK(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

  // Print a few values
  printf("n=%d, blocks=%d, threads=%d\n", n, blocks, threads);
  printf("Avg kernel time: %.4f ms (over %d iters)\n", ms_per, iters);
  printf("Sample y[0..4]: ");
  for (int i = 0; i < 5; ++i) printf("%.6f ", h_y[i]);
  printf("\n");

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  std::free(h_x);
  std::free(h_y);

  return 0;
}
