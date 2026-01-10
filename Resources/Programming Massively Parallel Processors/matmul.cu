#include <cuda_runtime.h>
#include <stdio.h>

__global__
void MatrixMulKernel(float* M, float* N,
                     float* P, int Width) {

    
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < Width && col < Width) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row*Width + k] * N[k*Width + col];
        }

        P[row*Width + col] = Pvalue;
    }
}

void MatrixMul(float* M, float* N,
               float* P, int Width) {
    int size = Width * Width * sizeof(float);

    float *M_d, *N_d, *P_d;
    cudaMalloc((void **)&M_d, size);
    cudaMalloc((void **)&N_d, size);
    cudaMalloc((void **)&P_d, size);

    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(32, 32);
    dim3 gridDim(ceil(Width / (float)blockDim.x), ceil(Width / (float)blockDim.y));

    printf("blockDim: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

    MatrixMulKernel<<<gridDim, blockDim>>>(M_d, N_d, P_d, Width);

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);
}

void MatrixMulCPU(float* M, float* N,
                  float* P, int Width) {
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
    float* P_gpu = (float*)malloc(size);
    float* P_cpu = (float*)malloc(size);

    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            const int index = row * Width + col;
            M[index] = 1; //row * Width + col;
            N[index] = 1; //row * Width + col;
            P_gpu[index] = 0.0f;
            P_cpu[index] = 0.0f;

        }
    }

    MatrixMul(M, N, P_gpu, Width);
    MatrixMulCPU(M, N, P_cpu, Width);

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