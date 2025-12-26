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

#define N (33 * 1024)

__global__ void add(const int* a, const int* b, int* c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // grid covers exactly N in the original example: 128 blocks * 128 threads = 16384 = 33*1024
  if (tid < N) c[tid] = a[tid] + b[tid];
}

int main(int argc, char** argv) {
  int *a = (int*)std::malloc(N*sizeof(int));
  int *b = (int*)std::malloc(N*sizeof(int));
  int *c = (int*)std::malloc(N*sizeof(int));
  if (!a || !b || !c) { std::fprintf(stderr,"malloc failed\n"); return 1; }

  for (int i=0;i<N;i++){ a[i]=i; b[i]=i*2; }

  int *da=nullptr,*db=nullptr,*dc=nullptr;
  CUDA_CHECK(cudaMalloc(&da, N*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&db, N*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dc, N*sizeof(int)));

  CUDA_CHECK(cudaMemcpy(da, a, N*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b, N*sizeof(int), cudaMemcpyHostToDevice));

  dim3 blocks(128);
  dim3 threads(128);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  add<<<blocks, threads>>>(da, db, dc);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());

  float ms=0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::printf("Kernel time: %.6f ms\n", ms);

  CUDA_CHECK(cudaMemcpy(c, dc, N*sizeof(int), cudaMemcpyDeviceToHost));

  bool ok=true;
  for (int i=0;i<N;i++){
    if (c[i] != a[i]+b[i]) { ok=false; std::printf("Mismatch at %d: %d + %d != %d\n", i, a[i], b[i], c[i]); break; }
  }
  std::printf("Verification: %s\n", ok ? "PASS" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  std::free(a); std::free(b); std::free(c);
  return ok ? 0 : 2;
}
