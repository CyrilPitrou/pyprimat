/* quad.h -- 1D adaptive quadrature (CPLAN.md S3.3: "Gauss-Kronrod 21/43-point
 * or adaptive Simpson with error estimate + bisection" -- adaptive Simpson
 * chosen here for simplicity). Used for the e+- thermodynamic integrals
 * and the Born weak-rate phase-space integral once those land (Phase 3a).
 */
#ifndef CPRIMAT_QUAD_H
#define CPRIMAT_QUAD_H

typedef double (*CPRQuadFunc)(double x, void *ctx);

/* Integrates f over [a, b] via adaptive Simpson with recursive bisection:
 * compares the whole-interval Simpson estimate against the sum of the two
 * half-interval estimates: if they agree to within `tol` (Richardson-
 * extrapolated, the classic adaptive-Simpson error criterion), accepts the
 * refined (half-interval) estimate; otherwise bisects further. `max_depth`
 * bounds the recursion (a guard against pathological integrands; 30 is a
 * generous default -- 2^30 subintervals is far beyond what any smooth
 * physical integrand here needs). Returns the integral estimate; if
 * `err_estimate` is non-NULL, *err_estimate receives the absolute error
 * estimate of the top-level call (the sum of accepted leaf errors). */
double cpr_quad_adaptive(CPRQuadFunc f, void *ctx, double a, double b,
                          double tol, int max_depth, double *err_estimate);

/* Fills `nodes`/`weights` (each caller-allocated, length n) with the n-point
 * Gauss-Legendre quadrature rule on [-1, 1] -- mirrors
 * numpy.polynomial.legendre.leggauss(n), used by weak_rates.c's fixed-order
 * rate-integral grid (_quad_grid / _N_GL = 160 in the Python source). Finds
 * the roots of the degree-n Legendre polynomial by Newton's method from the
 * standard asymptotic initial guess cos(pi*(i+0.75)/(n+0.5)) (Numerical
 * Recipes S4.6, "gauleg"), then sets weight[i] = 2 / ((1-x_i^2) P_n'(x_i)^2).
 * Nodes are returned ascending. Double precision converges in <=5 Newton
 * iterations for any n used here. */
void cpr_gauss_legendre(int n, double *nodes, double *weights);

#endif /* CPRIMAT_QUAD_H */
