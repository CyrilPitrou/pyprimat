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

/* Called after each *accepted* step, if non-NULL, with the RAW ingredients
 * needed to reconstruct the Dormand-Prince dense-output (continuous
 * extension) polynomial for that step at any theta in [0,1] later, via
 * cpr_ode_dense_eval below: the step's starting state `y_old` (length n,
 * NOT yet advanced), its span `hh` (signed, t_new = t_old + hh), and the
 * stage derivatives k1,k3,k4,k5,k6,k7 (each length n; k2 is unused, see
 * cpr_ode_dense_eval's docstring). Unlike `t_eval`/`y_eval` below (which
 * needs the full query grid known BEFORE the solve starts), this lets a
 * caller record every step cheaply and evaluate arbitrary, not-yet-known
 * query points at full dense-output accuracy AFTER the solve completes
 * (e.g. background.c's combined a(T)/t(T) ODE, which needs a second set of
 * query points only determined once the first pass's solution has been
 * inverted -- see combined_bg_rhs's docstring in background.c). */
typedef void (*CPRODEDenseStepCB)(double t_old, double hh, const double *y_old,
                                    const double *k1, const double *k3, const double *k4,
                                    const double *k5, const double *k6, const double *k7,
                                    size_t n, void *ctx);

typedef struct {
    double rtol;        /* relative tolerance per component (scipy-style mixed rtol/atol error norm) */
    double atol;        /* absolute tolerance per component */
    double h_init;       /* initial step size guess; 0 lets the integrator pick one */
    double h_min;        /* smallest step allowed before giving up (0 disables the check) */
    double h_max;        /* largest step allowed (0 disables the check, i.e. unbounded) */
    int max_steps;       /* hard cap on accepted+rejected steps combined */

    /* Optional dense-output evaluation grid, known in full BEFORE the solve
     * starts: when t_eval != NULL, cpr_ode_rk45 fills y_eval[j*n .. j*n+n-1]
     * with the solution at t_eval[j] (for every j < n_eval) as it steps,
     * using the Dormand-Prince continuous-extension polynomial (accurate to
     * the same order as the step itself -- no extra f() evaluations, no
     * separate interpolation error to accumulate on top of the ODE's own
     * accuracy). `t_eval` MUST be sorted in the same direction as the
     * integration (ascending if t1>t0, descending if t1<t0) and lie within
     * [t0, t1]; `y_eval` is caller-allocated, length n_eval*n, row-major
     * (component i of point j at y_eval[j*n+i]). Leave t_eval NULL (the
     * default from cpr_ode_rk_default_opts) to disable -- the solver then
     * only returns the final state in `y`, as before. */
    const double *t_eval;
    size_t n_eval;
    double *y_eval;

    /* Optional per-step raw-ingredient recorder (see CPRODEDenseStepCB);
     * NULL disables (the default). Independent of step_cb/t_eval -- any
     * combination may be used simultaneously in the same call. */
    CPRODEDenseStepCB dense_cb;
    void *dense_ctx;
} CPRRKOpts;

/* Default options: rtol=1e-7, atol=1e-12 (matching primat's
 * `numerical_precision` default and a representative abundance floor),
 * h_init=0 (auto), h_min=0, h_max=0, max_steps=100000, t_eval/dense_cb
 * disabled (NULL). */
CPRRKOpts cpr_ode_rk_default_opts(void);

/* Integrates dy/dt = f(t, y) from t0 to t1, in place in `y` (length n).
 * Returns 0 on success, nonzero on failure (step-size underflow, f()
 * returning nonzero, or max_steps exceeded) with *errmsg set (caller
 * frees it). `step_cb`/`cb_ctx` may be NULL if no per-step callback is
 * needed. See CPRRKOpts for the optional t_eval/y_eval dense-output grid
 * and dense_cb per-step recorder. */
int cpr_ode_rk45(CPRODEFunc f, void *ctx, double t0, double t1, double *y, size_t n,
                  CPRRKOpts opts, CPRODEStepCB step_cb, void *cb_ctx, char **errmsg);

/* Evaluates the Dormand-Prince dense-output (continuous extension)
 * polynomial at a single `theta` in [0,1] (theta = (t_interp-t_old)/hh)
 * within one step, for all `n` components, writing into `out` (caller-
 * allocated, length n). `y_old`, `hh`, and the stage derivatives
 * k1,k3,k4,k5,k6,k7 are exactly the values a CPRODEDenseStepCB receives for
 * that step (k2 is not needed: its dense-output weight is identically zero
 * for this Butcher tableau, matching A72=0). Exposed publicly so a caller
 * that has recorded steps via dense_cb (e.g. background.c's combined ODEs)
 * can replay this exact interpolant for query points only known after the
 * solve completes -- see CPRODEDenseStepCB's docstring. theta=0 reproduces
 * y_old exactly; theta=1 reproduces the step's accepted (5th-order) state
 * exactly (both to round-off, verified in test_ode_rk.c). */
void cpr_ode_dense_eval(double theta, double hh, const double *y_old,
                         const double *k1, const double *k3, const double *k4,
                         const double *k5, const double *k6, const double *k7,
                         size_t n, double *out);

#endif /* CPRIMAT_ODE_RK_H */
