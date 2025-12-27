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

#define N (33 * 1024)   // 33792 elements (use -DN=<value> to override)

__global__ void add(const int* a, const int* b, int* c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // grid covers exactly N in the original example: 128 blocks * 128 threads = 16384 = 33*1024
  if (tid < N) c[tid] = a[tid] + b[tid];
}

/* 
    This kernel uses a loop with stride to cover all N elements,
    even if the grid size is less than N.

    Each thread handles:

            tid, tid + gridSize, tid + 2·gridSize, ...

    This allows for fewer blocks and threads to cover all elements.


    1. Improved flexibility: Can run with smaller grids/blocks.
    2. Better resource utilization: More work per thread can lead to better occupancy.
    3. Reduced launch overhead: Fewer threads/blocks means less overhead.

    This was especially important when:

        Grid size was limited

        Launch overhead mattered more

        Hardware was less flexible

 */
__global__ void strided_add( int *a, int *b, int *c ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

void test_strided_add(int *da, int *db, int *dc) {
    dim3 blocks(32);   // Fewer blocks
    dim3 threads(64);  // Fewer threads

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    strided_add<<<blocks, threads>>>(da, db, dc);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::printf("N=%d, Strided Kernel time: %.6f ms\n", N, ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
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
  std::printf("N=%d, Kernel time: %.6f ms\n", N, ms);

  CUDA_CHECK(cudaMemcpy(c, dc, N*sizeof(int), cudaMemcpyDeviceToHost));

  bool ok=true;
  for (int i=0;i<N;i++){
    if (c[i] != a[i]+b[i]) { ok=false; std::printf("Mismatch at %d: %d + %d != %d\n", i, a[i], b[i], c[i]); break; }
  }

  std::printf("Verification: %s\n", ok ? "PASS" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));


//     // Now test strided_add with fewer blocks/threads
//     CUDA_CHECK(cudaMemset(dc, 0, N*sizeof(int))); // clear output
//     test_strided_add(da, db, dc);
  
//   bool strided_ok=true;
//   for (int i=0;i<N;i++){
//     if (c[i] != a[i]+b[i]) { strided_ok=false; std::printf("Mismatch at %d: %d + %d != %d\n", i, a[i], b[i], c[i]); break; }
//   }
//   std::printf("Verification: %s\n", strided_ok ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  std::free(a); std::free(b); std::free(c);
  return ok & strided_ok ? 0 : 2;
}
