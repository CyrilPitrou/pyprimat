/* linalg.h -- dense LU with partial pivoting.
 *
 * Matrices are n x n, row-major (A[i*n+j]). Sized for the BDF Newton
 * corrector's Jacobian, up to ~60x60 for the `large` network -- dense is
 * fine at this size, no sparse solver needed.
 */
#ifndef CPRIMAT_LINALG_H
#define CPRIMAT_LINALG_H

#include <stddef.h>

/* Factorises A in place into L (unit lower, implicit diagonal) and U
 * (upper), with partial pivoting recorded in piv[i] = the row swapped into
 * position i during elimination (so applying the same swaps to a
 * right-hand side reproduces P*b). Returns 0 on success, nonzero if a
 * pivot is (numerically) zero -- the matrix is singular to working
 * precision. */
int cpr_lu_factor(double *A, size_t n, size_t *piv);

/* Solves A*x = b given the LU factors and pivot array from cpr_lu_factor;
 * `b` is overwritten with the solution x in place. */
void cpr_lu_solve(const double *LU, size_t n, const size_t *piv, double *b);

/* Convenience one-shot solve: factorises a copy of A (caller's A is left
 * untouched) and solves A*x = b in place on `b`. Returns 0 on success,
 * nonzero if singular. For repeated solves against the same matrix (the
 * BDF Newton corrector's common case), call cpr_lu_factor once and
 * cpr_lu_solve per right-hand side instead. */
int cpr_solve_linear(const double *A, size_t n, double *b);

#endif /* CPRIMAT_LINALG_H */
