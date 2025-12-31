
/*
Compile with:
gcc blas_matrix_multiply.c \
  -I/usr/local/opt/openblas/include \
  -L/usr/local/opt/openblas/lib \
  -lopenblas \
  -O3
*/

#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int N = 1024;
    double *A = malloc(N*N*sizeof(double));
    double *B = malloc(N*N*sizeof(double));
    double *C = malloc(N*N*sizeof(double));

    for (int i = 0; i < N*N; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 0.0;
    }
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0, A, N,
                     B, N,
                0.0, C, N);

    printf("C[0] = %f\n", C[0]);
    return 0;
}
