/* linalg.c -- see cprimat/linalg.h. Doolittle LU with partial pivoting. */
#include "cprimat/linalg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define AT(M, i, j) (M[(i) * n + (j)])

int cpr_lu_factor(double *A, size_t n, size_t *piv)
{
    for (size_t k = 0; k < n; k++) {
        /* Partial pivoting: swap in the largest-magnitude entry in column k
         * among rows k..n-1, to keep the elimination numerically stable. */
        size_t p = k;
        double best = fabs(AT(A, k, k));
        for (size_t i = k + 1; i < n; i++) {
            double v = fabs(AT(A, i, k));
            if (v > best) { best = v; p = i; }
        }
        piv[k] = p;
        if (p != k) {
            for (size_t j = 0; j < n; j++) {
                double tmp = AT(A, k, j);
                AT(A, k, j) = AT(A, p, j);
                AT(A, p, j) = tmp;
            }
        }

        double pivot = AT(A, k, k);
        if (fabs(pivot) < 1e-300)
            return 1; /* singular to working precision */

        for (size_t i = k + 1; i < n; i++) {
            double m = AT(A, i, k) / pivot;
            AT(A, i, k) = m; /* store the multiplier in L's lower part */
            for (size_t j = k + 1; j < n; j++)
                AT(A, i, j) -= m * AT(A, k, j);
        }
    }
    return 0;
}

void cpr_lu_solve(const double *LU, size_t n, const size_t *piv, double *b)
{
    /* Apply the recorded row swaps to b (P*b), matching the swaps that
     * were applied to A's rows during factorisation. */
    for (size_t k = 0; k < n; k++) {
        size_t p = piv[k];
        if (p != k) {
            double tmp = b[k];
            b[k] = b[p];
            b[p] = tmp;
        }
    }
    /* Forward substitution: L*y = P*b (L unit lower triangular). */
    for (size_t i = 1; i < n; i++) {
        double s = b[i];
        for (size_t j = 0; j < i; j++)
            s -= AT(LU, i, j) * b[j];
        b[i] = s;
    }
    /* Back substitution: U*x = y. */
    for (size_t ii = n; ii-- > 0;) {
        double s = b[ii];
        for (size_t j = ii + 1; j < n; j++)
            s -= AT(LU, ii, j) * b[j];
        b[ii] = s / AT(LU, ii, ii);
    }
}

int cpr_solve_linear(const double *A, size_t n, double *b)
{
    double *Acopy = malloc(n * n * sizeof(double));
    size_t *piv = malloc(n * sizeof(size_t));
    memcpy(Acopy, A, n * n * sizeof(double));
    int rc = cpr_lu_factor(Acopy, n, piv);
    if (!rc)
        cpr_lu_solve(Acopy, n, piv, b);
    free(Acopy);
    free(piv);
    return rc;
}
