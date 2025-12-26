// julia_ppm_colab.cu
// CUDA By Example "Julia set" demo adapted for Google Colab (headless):
// - Removes CPUBitmap/GLUT dependency
// - Writes output to a binary PPM (P6) file
//
// Build (Colab):
//   nvcc -O3 -std=c++17 julia_ppm_colab.cu -o julia_ppm
// Run:
//   ./julia_ppm              # writes julia.ppm
//   ./julia_ppm out.ppm 800  # optional: output filename + DIM
//
// View in Colab (Python):
//   from PIL import Image; import matplotlib.pyplot as plt
//   img = Image.open("julia.ppm"); plt.imshow(img); plt.axis("off")

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#ifndef DEFAULT_DIM
#define DEFAULT_DIM 1000
#endif

// ---------- error checking ----------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(_e));         \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// ---------- complex arithmetic ----------
struct cuComplex {
    float r;
    float i;

    __host__ __device__ cuComplex(float a, float b) : r(a), i(b) {}

    __device__ float magnitude2() const { return r * r + i * i; }

    __device__ cuComplex operator*(const cuComplex& a) const {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a) const {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y, int dim) {
    const float scale = 1.5f;
    float jx = scale * (float)(dim / 2 - x) / (dim / 2);
    float jy = scale * (float)(dim / 2 - y) / (dim / 2);

    cuComplex c(-0.8f, 0.156f);
    cuComplex a(jx, jy);

    // iteration cap from the book
    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000.0f) return 0;
    }
    return 1;
}

__global__ void kernel_rgba(uint8_t* rgba, int dim) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= dim || y >= dim) return;
    int offset = x + y * dim;

    int v = julia(x, y, dim);

    rgba[offset * 4 + 0] = (uint8_t)(255 * v);  // R
    rgba[offset * 4 + 1] = 0;                  // G
    rgba[offset * 4 + 2] = 0;                  // B
    rgba[offset * 4 + 3] = 255;                // A
}

static void write_ppm_rgb(const char* filename, const uint8_t* rgba, int dim) {
    // PPM P6 is RGB; we drop alpha.
    std::FILE* f = std::fopen(filename, "wb");
    if (!f) {
        std::perror("fopen");
        std::exit(1);
    }

    std::fprintf(f, "P6\n%d %d\n255\n", dim, dim);

    for (int i = 0; i < dim * dim; i++) {
        const uint8_t r = rgba[i * 4 + 0];
        const uint8_t g = rgba[i * 4 + 1];
        const uint8_t b = rgba[i * 4 + 2];
        std::fputc(r, f);
        std::fputc(g, f);
        std::fputc(b, f);
    }

    std::fclose(f);
}

int main(int argc, char** argv) {
    const char* out = (argc >= 2) ? argv[1] : "julia.ppm";
    int dim = (argc >= 3) ? std::atoi(argv[2]) : DEFAULT_DIM;

    if (dim <= 0 || dim > 8192) {
        std::fprintf(stderr, "DIM must be in (0, 8192]. Got %d\n", dim);
        return 1;
    }

    const size_t bytes = (size_t)dim * (size_t)dim * 4;

    uint8_t* d_rgba = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_rgba, bytes));

    // More modern launch: 16x16 threads, grid covers the image
    dim3 block(16, 16);
    dim3 grid((dim + block.x - 1) / block.x,
              (dim + block.y - 1) / block.y);
    kernel_rgba<<<grid, block>>>(d_rgba, dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint8_t* h_rgba = (uint8_t*)std::malloc(bytes);
    if (!h_rgba) {
        std::fprintf(stderr, "malloc failed\n");
        return 1;
    }

    CUDA_CHECK(cudaMemcpy(h_rgba, d_rgba, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_rgba));

    write_ppm_rgb(out, h_rgba, dim);

    std::free(h_rgba);

    std::printf("Wrote %s (%dx%d)\n", out, dim, dim);
    return 0;
}
