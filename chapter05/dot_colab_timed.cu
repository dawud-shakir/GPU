// Colab-friendly (no GLUT/CPUAnimBitmap), with CUDA event timing.
// Recommended build: nvcc -O3 -std=c++17 <file>.cu -o <exe>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <string>

#define CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); \
  } \
} while (0)

static void write_ppm_rgb_from_rgba(const char* filename, const unsigned char* rgba, int w, int h) {
  FILE* f = std::fopen(filename, "wb");
  if (!f) { perror("fopen"); std::exit(1); }
  std::fprintf(f, "P6\n%d %d\n255\n", w, h);
  // Input is RGBA (4 bytes per pixel). Write RGB.
  const size_t n = (size_t)w * (size_t)h;
  for (size_t i = 0; i < n; ++i) {
    const unsigned char* p = rgba + 4*i;
    std::fputc(p[0], f);
    std::fputc(p[1], f);
    std::fputc(p[2], f);
  }
  std::fclose(f);
}

#define N 33792

__global__ void dot( float *a, float *b, float *c ) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}


int main(void) {
  // allocate and initialize host data
  float *a = (float*)std::malloc(N*sizeof(float));
  float *b = (float*)std::malloc(N*sizeof(float));
  float *partial_c = (float*)std::malloc(64*sizeof(float));
  if (!a || !b || !partial_c) {
    std::fprintf(stderr,"malloc failed\n"); return 1;
  }

  for (int i=0; i<N; i++) {
    a[i] = (float)i;
    b[i] = (float)(i*2);
  }

  float *dev_a=nullptr, *dev_b=nullptr, *dev_partial_c=nullptr;
  CUDA_CHECK(cudaMalloc(&dev_a, N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_b, N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_partial_c, 64*sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice));

  const int threadsPerBlock = 256;
  const int blocksPerGrid   = 64;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());

  float ms=0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::printf("Kernel time: %.6f ms\n", ms);

  CUDA_CHECK(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));

  // finalize on CPU
  float c = 0.0f;
  for (int i=0;i<blocksPerGrid;i++) c += partial_c[i];

  // expected: sum_i (i * 2i) = 2 * sum_i i^2, i=0..N-1
  auto sum_squares = [](float x) -> float { return x*(x+1.0f)*(2.0f*x+1.0f)/6.0f; };
  float expected = 2.0f * sum_squares((float)(N-1));
  std::printf("Dot product: %.6g\nExpected:   %.6g\n", c, expected);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_partial_c));
  std::free(a); std::free(b); std::free(partial_c);
  return 0;
}
