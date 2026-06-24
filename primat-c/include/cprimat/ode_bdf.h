/* ode_bdf.h -- variable-order (1-5), variable-step BDF integrator for
 * stiff ODEs (CPLAN.md S3.5/S8): the MT/LT-era nuclear network once
 * network_builder.c lands (Phase 4), validated for now against the
 * textbook stiff benchmarks in S8 (Robertson, Van der Pol).
 *
 * Implementation note (a deliberate simplification vs. scipy's BDF/
 * MATLAB's ode15s): those use a Nordsieck-vector representation that
 * supports changing the step size *and* growing the order using the same
 * history. Here, because the classical Gear/Krogh backward-difference
 * formula at order k requires k+1 *uniformly spaced* points
 * (alpha_0 y_{n+1} + alpha_1 y_n + ... = h f_{n+1}, coefficients tabulated
 * for constant h), a change in h resets the order to 1 and the order is
 * then allowed to climb back up to `max_order` only after enough
 * consecutive accepted steps at the new (now-fixed) h have accumulated
 * enough history. This costs some efficiency (more steps than a true
 * variable-step/order method) but is far simpler to get right, and
 * CPLAN.md S8 explicitly accepts a different step sequence as long as
 * solution accuracy matches scipy's BDF at the same tolerance.
 */
#ifndef CPRIMAT_ODE_BDF_H
#define CPRIMAT_ODE_BDF_H

#include <stddef.h>
#include "cprimat/ode_rk.h" /* reuses CPRODEFunc, CPRODEStepCB */

/* Analytic Jacobian callback: writes df_i/dy_j into J (row-major, n*n) at
 * (t, y). If NULL is passed to cpr_ode_bdf, a forward-difference Jacobian
 * is computed internally (sufficient for validation against the S8
 * benchmarks; network_builder.c's analytic `_jac_kernel` port, Phase 4,
 * is a drop-in replacement once available, for speed on the real ~60-
 * species network). */
typedef int (*CPRODEJacFunc)(double t, const double *y, double *J, void *ctx);

typedef struct {
    double rtol;          /* relative tolerance, error norm as in ode_rk.h */
    double atol;          /* absolute tolerance */
    double h_init;         /* initial step guess; 0 lets the integrator pick one */
    double h_min;          /* smallest step before giving up (0 disables the check) */
    double h_max;          /* largest step (0 disables the check) */
    int max_order;         /* 1..5, default 5 */
    int max_steps;         /* hard cap on accepted+rejected steps */
    int max_newton_iter;   /* Newton corrector iteration cap per attempt, default 10 */
} CPRBDFOpts;

/* Default options: rtol=1e-7, atol=1e-12, h_init=0 (auto), h_min=h_max=0
 * (unbounded), max_order=5, max_steps=200000, max_newton_iter=10. */
CPRBDFOpts cpr_ode_bdf_default_opts(void);

/* Integrates dy/dt = f(t, y) from t0 to t1, in place in `y` (length n),
 * using the implicit BDF corrector with Newton iteration (dense LU via
 * linalg.c for the n x n Newton system -- fine up to the ~60-species
 * `large` network this targets). `jac` may be NULL (finite-difference
 * fallback). Returns 0 on success, nonzero on failure with *errmsg set
 * (caller frees it). `step_cb`/`cb_ctx` may be NULL. */
int cpr_ode_bdf(CPRODEFunc f, CPRODEJacFunc jac, void *ctx,
                 double t0, double t1, double *y, size_t n,
                 CPRBDFOpts opts, CPRODEStepCB step_cb, void *cb_ctx, char **errmsg);

#endif /* CPRIMAT_ODE_BDF_H */
