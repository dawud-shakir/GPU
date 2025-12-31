// transpose_via_blas.c
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  const int m = 2, n = 3;            // A is m x n, AT is n x m
  double A[m*n] = { 1,2,3, 4,5,6 };  // row-major: [ [1 2 3], [4 5 6] ]
  double *I = calloc((size_t)m*m, sizeof(double));
  double *AT = calloc((size_t)n*m, sizeof(double));

  for (int i = 0; i < m; ++i) I[i*m + i] = 1.0;

  // AT (n x m) = A^T (n x m) * I (m x m)
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              n, m, m, 1.0, A, n, I, m, 0.0, AT, m);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) printf("%g ", AT[i*m + j]);
    printf("\n");
  }

  free(I); free(AT);
  return 0;
}