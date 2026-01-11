#include <cuda_runtime.h>
#include <stdio.h>

/*****
 Ch. 3, Exercise 2
*****/

__global__
void MatrixVectorMulKernel(float* A, float* B,
                     float* C, int Width) {

    
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < Width && col < Width) {
        
        float sum = 0.0f;

        for (int j = 0; j < Width; ++j) {
            sum += B[row*Width + j] * C[j];
        }
        A[row] = sum;        
    }
}

void MatrixVectorMul(float* A, float* B,
               float* C, int Width) {
    
    int size_vector = Width * sizeof(float);
    int size_matrix = Width * Width * sizeof(float);

    float *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, size_vector);
    cudaMalloc((void **)&B_d, size_matrix);
    cudaMalloc((void **)&C_d, size_vector);

    cudaMemcpy(B_d, B, siz_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, size_vector, cudaMemcpyHostToDevice);
    
    dim3 blockDim(1, 32);
    dim3 gridDim(1, ceil(Width / (float)blockDim.y));

    printf("blockDim: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

    MatrixVectorMulKernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, Width);

    cudaMemcpy(A, A_d, size_vector, cudaMemcpyDeviceToHost);
}

void MatrixVectorMulCPU(float* A, float* B,
                  float* C, int Width) {
    for (int row = 0; row < Width; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < Width; ++col) {
            const float b = B[row*Width + col];
            const float c = C[col];
            sum += b*c;
        }
        A[row] = sum;
    }
}

int main(int argc, char* argv[])
{
    // The maximum number of threads in a block is 1024.
    const int Width = argc > 1 ? atoi(argv[1]) : 16;
    printf("Vector size: %d x 1\n", Width);
    printf("Matrix size: %d x %d\n", Width, Width);

    int size_vector = Width * sizeof(float);
    int size_matrix = Width * Width * sizeof(float);
    
    float* A_gpu = (float*)malloc(size_vector);
    float* A_cpu = (float*)malloc(size_vector);
    float* B = (float*)malloc(size_matrix);
    float* C = (float*)malloc(size_vector);
    
    for (int row = 0; row < Width; ++row) {
        C[row] = 2.0f; //row * Width + col;
        A_gpu[row] = 0.0f;
        A_cpu[row] = 0.0f;
    }

    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            B[row * Width + col] = 4.0f; //row * Width + col;
        }
    }

    MatrixVectorMul_Cols(A, B, C_gpu, Width);
    MatrixVectorMulCPU(A, B, C_cpu, Width);

    bool match = true;
    for (int i = 0; i < Width * Width; ++i) {
        const float diff = fabs(C_gpu[i] - C_cpu[i]);
        if (diff > 0.00001) {
            printf("Mismatch at (%d, %d)\n", (int)(i/Width), i%Width);
            printf("diff: %f\n", diff);
            printf("C_gpu=%f, C_cpu=%f\n", C_gpu[i], C_cpu[i]);
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

    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);

    return 0;
}