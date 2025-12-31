// transpose_via_blas.c
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  const int m = 2, n = 3;            // A is m x n, AT is n x m
  double A[] = { 1, 2, 3, 4, 5, 6 }; // row-major: [ [1 2 3], [4 5 6] ]
  double *Id = calloc((size_t)m * (size_t)m, sizeof(double));
  double *AT = calloc((size_t)n*m, sizeof(double));

  if (!Id || !AT) {
    fprintf(stderr, "calloc failed\n");
    free(Id);
    free(AT);
    return 1;
  }

  for (int i = 0; i < m; ++i) Id[i*m + i] = 1.0;

  // AT (n x m) = A^T (n x m) * I (m x m)
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              n, m, m, 1.0, A, n, Id, m, 0.0, AT, m);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) printf("%g ", AT[i*m + j]);
    printf("\n");
  }

  free(Id);
  free(AT);
  return 0;
}
