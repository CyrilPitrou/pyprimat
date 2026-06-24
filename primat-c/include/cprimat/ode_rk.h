/* ode_rk.h -- embedded Dormand-Prince RK45 with adaptive step size, for
 * non-stiff ODEs (CPLAN.md S3.5): the HT-era n<->p ODE and the two
 * background ODEs (a(T_gamma) entropy-conservation, t(a) Hubble
 * integration). Python uses scipy's LSODA for these, which auto-switches
 * to BDF only if stiffness is detected; these problems are smooth enough
 * that a fixed non-stiff method suffices, so a dedicated RK45 is simpler
 * than porting LSODA's method-switching logic.
 */
#ifndef CPRIMAT_ODE_RK_H
#define CPRIMAT_ODE_RK_H

#include <stddef.h>

/* RHS callback: writes dy/dt into ydot given (t, y). Returns 0 on success,
 * nonzero to signal the integrator should abort (e.g. on a domain error). */
typedef int (*CPRODEFunc)(double t, const double *y, double *ydot, void *ctx);

/* Called after each *accepted* step, if non-NULL (e.g. to record a dense
 * time-evolution table); ydot is not necessarily recomputed for the
 * accepted point so it is not made available here. */
typedef void (*CPRODEStepCB)(double t, const double *y, size_t n, void *ctx);

typedef struct {
    double rtol;        /* relative tolerance per component (scipy-style mixed rtol/atol error norm) */
    double atol;        /* absolute tolerance per component */
    double h_init;       /* initial step size guess; 0 lets the integrator pick one */
    double h_min;        /* smallest step allowed before giving up (0 disables the check) */
    double h_max;        /* largest step allowed (0 disables the check, i.e. unbounded) */
    int max_steps;       /* hard cap on accepted+rejected steps combined */
} CPRRKOpts;

/* Default options: rtol=1e-7, atol=1e-12 (matching PyPRIMAT's
 * `numerical_precision` default and a representative abundance floor),
 * h_init=0 (auto), h_min=0, h_max=0, max_steps=100000. */
CPRRKOpts cpr_ode_rk_default_opts(void);

/* Integrates dy/dt = f(t, y) from t0 to t1, in place in `y` (length n).
 * Returns 0 on success, nonzero on failure (step-size underflow, f()
 * returning nonzero, or max_steps exceeded) with *errmsg set (caller
 * frees it). `step_cb`/`cb_ctx` may be NULL if no per-step callback is
 * needed. */
int cpr_ode_rk45(CPRODEFunc f, void *ctx, double t0, double t1, double *y, size_t n,
                  CPRRKOpts opts, CPRODEStepCB step_cb, void *cb_ctx, char **errmsg);

#endif /* CPRIMAT_ODE_RK_H */
