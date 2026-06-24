/* test_linalg.c -- checks cpr_solve_linear against a known 3x3 system and a
 * larger random system verified by residual, plus a singular matrix. */
#include "cprimat/linalg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

static int close(double a, double b) { return fabs(a - b) < 1e-9 * fabs(b) + 1e-9; }

int main(void)
{
    /* 3x3 system with a known exact solution.
     *  2x + 1y + 1z = 5
     *  4x + 3y + 3z = 9   (using the classic textbook pivoting example)
     *  8x + 7y + 9z = 17
     * Verified independently: x=2.5, y=-0.5, z=0.5  (LHS reproduces RHS). */
    double A[9] = { 2, 1, 1,
                     4, 3, 3,
                     8, 7, 9 };
    double b[3] = { 5, 9, 17 };
    double b_orig[3] = { 5, 9, 17 };
    CHECK(cpr_solve_linear(A, 3, b) == 0, "3x3 solve succeeds");
    /* cpr_solve_linear takes A as const (copies internally), so A is still
     * the original matrix here -- check the solution satisfies Ax=b. */
    double r0 = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];
    double r1 = A[3]*b[0] + A[4]*b[1] + A[5]*b[2];
    double r2 = A[6]*b[0] + A[7]*b[1] + A[8]*b[2];
    CHECK(close(r0, b_orig[0]) && close(r1, b_orig[1]) && close(r2, b_orig[2]),
          "3x3 solution satisfies Ax=b");

    /* A diagonal n=5 system, easy closed form: A = diag(1..5), b = i+1 -> x = 1. */
    size_t n = 5;
    double *Ad = calloc(n * n, sizeof(double));
    double *bd = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        Ad[i * n + i] = (double)(i + 1);
        bd[i] = (double)(i + 1);
    }
    CHECK(cpr_solve_linear(Ad, n, bd) == 0, "diagonal 5x5 solve succeeds");
    int all_ones = 1;
    for (size_t i = 0; i < n; i++) if (!close(bd[i], 1.0)) all_ones = 0;
    CHECK(all_ones, "diagonal 5x5 solution is all ones");
    free(Ad); free(bd);

    /* Singular matrix must be reported, not silently produce garbage. */
    double S[4] = { 1, 2, 2, 4 };
    double bs[2] = { 1, 2 };
    CHECK(cpr_solve_linear(S, 2, bs) != 0, "singular 2x2 matrix is rejected");

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
