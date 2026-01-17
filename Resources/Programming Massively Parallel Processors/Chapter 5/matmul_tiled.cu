#include <cuda_runtime.h>
#include <stdlib.h>

#define TILE_WIDTH 16


__global__ void matrixMulKernel(float* M, float* N, float* P, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;    int by = blockIdx.y;
    int tx = threadIdx.x;   int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph)
    {
        // Collaborative loading of M and N tiles into shared memory
        Mds[ty][tx] = M[Row*Width +  ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx]; // blockDim is TILE_WIDTH x TILE_WIDTH
        syncthreads();
    }
    P[Row*Width + Col] = Pvalue;
}

void matrixMul(float* M, float* N, float* P, int Width) {
    int size = Width * Width * sizeof(float);

    float *M_d, *N_d, *P_d;
    cudaMalloc((void **)&M_d, size);
    cudaMalloc((void **)&N_d, size);
    cudaMalloc((void **)&P_d, size);

    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil(Width / (float)blockDim.x), ceil(Width / (float)blockDim.y));

    printf("TILE_WIDTH: %d\n", TILE_WIDTH);
    printf("blockDim: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

    matrixMulKernel<<<gridDim, blockDim>>>(M_d, N_d, P_d, Width);

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

void matrixMulCPU(float* M, float* N, float* P, int Width) {
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < Width; ++k) {
                const float a = M[row*Width + k];
                const float b = N[k*Width + col];
                sum += a*b;
            }
            P[row*Width + col] = sum;
        }
    }
}

int main(int argc, char* argv[])
{
    // The maximum number of threads in a block is 1024.
    const int Width = argc > 1 ? atoi(argv[1]) : 16;
    printf("Matrix size: %d x %d\n", Width, Width);

    int size = Width * Width * sizeof(float);
    
    float* M = (float*)malloc(size);
    float* N = (float*)malloc(size);
    float* P_gpu = (float*)calloc(Width * Width, sizeof(float));
    float* P_cpu = (float*)calloc(Width * Width, sizeof(float));

    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            const int index = row * Width + col;
            M[index] = rand() / (float)RAND_MAX; //2.0 //row * Width + col;
            N[index] =rand() / (float)RAND_MAX; //4.0 //row * Width + col;
        }
    }

    matrixMul(M, N, P_gpu, Width);
    matrixMulCPU(M, N, P_cpu, Width);

    bool match = true;
    for (int i = 0; i < Width * Width; ++i) {
        const float diff = fabs(P_gpu[i] - P_cpu[i]);
        if (diff > 0.00001) {
            printf("Mismatch at (%d, %d)\n", (int)(i/Width), i%Width);
            printf("diff: %f\n", diff);
            printf("P_gpu=%f, P_cpu=%f\n", P_gpu[i], P_cpu[i]);
            match = false;
            break;
        }
    }

    if (match) {
        printf("GPU and CPU results match\n");
    }
    else {
        printf("GPU and CPU results do not match!\n");
    }

    free(M);
    free(N);
    free(P_gpu);
    free(P_cpu);

    return 0;
}