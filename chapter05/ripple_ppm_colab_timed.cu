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

#define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernel( unsigned char *ptr, int ticks ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // now calculate the value at that position
    float fx = x - DIM/2;
    float fy = y - DIM/2;
    float d = sqrtf( fx * fx + fy * fy );
    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                         cos(d/10.0f - ticks/7.0f) /
                                         (d/10.0f + 1.0f));    
    ptr[offset*4 + 0] = grey;
    ptr[offset*4 + 1] = grey;
    ptr[offset*4 + 2] = grey;
    ptr[offset*4 + 3] = 255;
}


int main(int argc, char** argv) {
  const char* out = (argc >= 2) ? argv[1] : "/content/drive/MyDrive/CUDA/ripple.ppm";
  int ticks = (argc >= 3) ? std::atoi(argv[2]) : 0;

  const size_t bytes = (size_t)DIM * (size_t)DIM * 4;
  unsigned char* dev = nullptr;
  CUDA_CHECK(cudaMalloc(&dev, bytes));

  dim3 threads(16,16);
  dim3 grids((DIM + threads.x - 1)/threads.x, (DIM + threads.y - 1)/threads.y);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  kernel<<<grids, threads>>>(dev, ticks);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());

  float ms=0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::printf("Kernel time: %.6f ms\n", ms);

  unsigned char* host = (unsigned char*)std::malloc(bytes);
  if (!host) { std::fprintf(stderr,"malloc failed\n"); return 1; }
  CUDA_CHECK(cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost));

  write_ppm_rgb_from_rgba(out, host, DIM, DIM);
  std::printf("Wrote %s\n", out);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dev));
  std::free(host);
  return 0;
}
