/*
gcc lapack_solve_linear_system.c -O3 \
  -I/usr/local/opt/lapack/include \
  -L/usr/local/opt/lapack/lib -llapacke -llapack \
  -L/usr/local/opt/openblas/lib -lopenblas
*/

#include <stdio.h>
#include <lapacke.h>

int main() {
    // Solve A x = b for x, overwriting b with the solution.
    // LAPACKE uses column-major by default when you pass LAPACK_COL_MAJOR.

    int n = 2, nrhs = 1;
    int ipiv[2];

    // A = [[3,1],
    //      [1,2]]
    // Column-major storage:
    double A[4] = {3.0, 1.0,
                   1.0, 2.0};

    // b = [9,8]
    double b[2] = {9.0, 8.0};

    int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, A, n, ipiv, b, n);

    if (info == 0) {
        printf("x = [%f, %f]\n", b[0], b[1]);
    } else {
        printf("dgesv failed, info = %d\n", info);
    }
    return 0;
}
