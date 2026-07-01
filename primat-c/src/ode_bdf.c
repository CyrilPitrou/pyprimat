/* ode_bdf.c -- see ode_bdf.h.
 *
 * Variable-order (1-5), variable-step BDF/NDF integrator using the
 * Nordsieck-vector (scaled-derivative array) representation, following
 * Byrne & Hindmarsh, "A Polyalgorithm for the Numerical Solution of
 * Ordinary Differential Equations", ACM TOMS 1(1), 1975, and Shampine &
 * Reichelt, "The MATLAB ODE Suite", SIAM J. Sci. Comput. 18(1), 1997 --
 * the same algorithm family scipy.integrate.BDF implements
 * (scipy/integrate/_ivp/bdf.py). This file is a direct, step-for-step
 * port of that scipy module (read alongside this file if anything below
 * is unclear): same Nordsieck update formula, same NDF kappa coefficients,
 * same Newton convergence-rate test, same order/step selection logic --
 * deliberately, since the acceptance bar here is *solution accuracy*
 * matching scipy's BDF at the same tolerance, and the most reliable way to
 * hit that bar is to reproduce the same well-tested heuristics rather than
 * invent new ones.
 *
 * This supersedes an earlier, simpler "constant-step-with-restart"
 * version (changing h or growing the order both required rebuilding a
 * fixed-spacing finite-difference history from scratch) that worked but
 * needed a much tighter nominal rtol than scipy's BDF to reach the same
 * actual accuracy -- see the project memory note on this gap. The
 * Nordsieck representation here changes step size *and* order using the
 * same continuously-maintained history (via change_D below), which is
 * exactly what makes scipy's BDF efficient at a given tolerance.
 */
#include "ode_bdf.h"
#include "linalg.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ORDER_CAP 5
#define NEWTON_MAXITER 4   /* scipy bdf.py's NEWTON_MAXITER */
#define MIN_FACTOR 0.2     /* scipy bdf.py's MIN_FACTOR */
#define MAX_FACTOR 10.0    /* scipy bdf.py's MAX_FACTOR */

/* RMS error norm sqrt(mean(x_i^2)) -- same convention as ode_rk.c and
 * scipy's own `norm()` helper (_ivp/common.py). */
static double rms_norm_scaled(const double *x, const double *scale, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; i++) { double e = x[i] / scale[i]; s += e * e; }
    return sqrt(s / (double)n);
}

/* NDF (numerical differentiation formula) correction kappa[order],
 * order=1..5 (kappa[0] unused, kappa[5]=0 i.e. order 5 is plain BDF --
 * the NDF modification is not stable enough there): Shampine & Reichelt
 * S3, "The MATLAB ODE Suite". gamma[q] = sum_{i=1}^{q} 1/i;
 * alpha[q] = (1-kappa[q])*gamma[q] is the corrector's effective step
 * coefficient (the `c = h/alpha[order]` used below); error_const[q]
 * estimates the local truncation error from the Nordsieck correction term
 * `d` once Newton has converged at order q. */
static const double KAPPA[MAX_ORDER_CAP + 1] = { 0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0 };

typedef struct {
    double gamma[MAX_ORDER_CAP + 1];
    double alpha[MAX_ORDER_CAP + 1];
    double error_const[MAX_ORDER_CAP + 1];
} BDFCoeffs;

static void bdf_coeffs_init(BDFCoeffs *c)
{
    c->gamma[0] = 0.0;
    for (int i = 1; i <= MAX_ORDER_CAP; i++) c->gamma[i] = c->gamma[i - 1] + 1.0 / (double)i;
    for (int i = 0; i <= MAX_ORDER_CAP; i++) {
        c->alpha[i] = (1.0 - KAPPA[i]) * c->gamma[i];
        c->error_const[i] = KAPPA[i] * c->gamma[i] + 1.0 / (double)(i + 1);
    }
}

/* compute_R: the (order+1)x(order+1) matrix mapping a Nordsieck array's
 * divided-difference rows, built at one step size, onto the same
 * underlying interpolating polynomial's divided differences at a step
 * size rescaled by `factor` (scipy bdf.py's compute_R: a column-wise
 * cumulative product of M, where M[0][j]=1 and, for i,j>=1,
 * M[i][j]=(i-1-factor*j)/i). R is written row-major into a caller-supplied
 * (order+1)*(order+1) buffer. */
static void compute_R(int order, double factor, double *R)
{
    int m = order + 1;
    for (int j = 0; j < m; j++) R[j] = 1.0; /* row 0 */
    for (int i = 1; i < m; i++) {
        R[i * m + 0] = 0.0;
        for (int j = 1; j < m; j++) {
            double Mij = ((double)i - 1.0 - factor * (double)j) / (double)i;
            R[i * m + j] = R[(i - 1) * m + j] * Mij;
        }
    }
}

/* change_D: rescales the Nordsieck array's first `order+1` rows in place
 * for a step-size change by `factor` (scipy bdf.py's change_D: D[:order+1]
 * <- (R.U)^T . D[:order+1], U = compute_R(order, 1.0)). D is a flat
 * row-major array with fixed row stride `n` (MAX_ORDER_CAP+3 rows
 * allocated by the caller; only rows 0..order are touched here). */
static void change_D(double *D, size_t n, int order, double factor)
{
    int m = order + 1;
    double R[(MAX_ORDER_CAP + 1) * (MAX_ORDER_CAP + 1)];
    double U[(MAX_ORDER_CAP + 1) * (MAX_ORDER_CAP + 1)];
    double RU[(MAX_ORDER_CAP + 1) * (MAX_ORDER_CAP + 1)];
    compute_R(order, factor, R);
    compute_R(order, 1.0, U);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            double s = 0.0;
            for (int k = 0; k < m; k++) s += R[i * m + k] * U[k * m + j];
            RU[i * m + j] = s;
        }
    /* newD[i] = sum_k RU[k][i] * oldD[k] (i.e. RU^T applied); buffer the
     * new rows before overwriting, since every new row reads every old
     * row. */
    double *tmp = malloc((size_t)m * n * sizeof(double));
    for (int i = 0; i < m; i++)
        for (size_t c = 0; c < n; c++) {
            double s = 0.0;
            for (int k = 0; k < m; k++) s += RU[k * m + i] * D[(size_t)k * n + c];
            tmp[(size_t)i * n + c] = s;
        }
    memcpy(D, tmp, (size_t)m * n * sizeof(double));
    free(tmp);
}

/* Forward-difference Jacobian fallback (used when the caller has no
 * analytic one -- network_builder.c's analytic `_jac_kernel` port is the
 * normal case for the real network; this remains the fallback path and
 * the one validated against the textbook S8 benchmarks below). Standard
 * sqrt(machine epsilon)-scaled perturbation (Dennis & Schnabel). */
static int finite_diff_jac(CPRODEFunc f, void *ctx, double t, const double *y, size_t n,
                            double *f0, double *J, double *ytmp, double *f1)
{
    if (f(t, y, f0, ctx)) return 1;
    const double eps = 1.4901161193847656e-08; /* sqrt(DBL_EPSILON) */
    memcpy(ytmp, y, n * sizeof(double));
    for (size_t j = 0; j < n; j++) {
        double yj = y[j];
        double dy = eps * fmax(1.0, fabs(yj));
        ytmp[j] = yj + dy;
        if (f(t, ytmp, f1, ctx)) return 1;
        ytmp[j] = yj;
        for (size_t i = 0; i < n; i++)
            J[i * n + j] = (f1[i] - f0[i]) / dy;
    }
    return 0;
}

/* select_initial_step: Hairer/Norsett/Wanner I S II.4's empirical
 * heuristic for a good first step (scipy _ivp/common.py's
 * select_initial_step, order=1 since BDF always starts at order 1). */
static double select_initial_step(CPRODEFunc f, void *ctx, double t0, const double *y0,
                                    double t1, double h_max_user, const double *f0,
                                    int dir, double rtol, double atol, size_t n,
                                    double *scratch_y1, double *scratch_f1)
{
    double interval_length = fabs(t1 - t0);
    if (interval_length == 0.0) return 0.0;

    double d0 = 0.0, d1 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double scale = atol + fabs(y0[i]) * rtol;
        double e0 = y0[i] / scale, e1 = f0[i] / scale;
        d0 += e0 * e0; d1 += e1 * e1;
    }
    d0 = sqrt(d0 / (double)n);
    d1 = sqrt(d1 / (double)n);

    double h0;
    if (d0 < 1e-5 || d1 < 1e-5) h0 = 1e-6;
    else h0 = 0.01 * d0 / d1;
    if (h0 > interval_length) h0 = interval_length;

    for (size_t i = 0; i < n; i++) scratch_y1[i] = y0[i] + h0 * (double)dir * f0[i];
    if (f(t0 + h0 * (double)dir, scratch_y1, scratch_f1, ctx)) return h0; /* f() failure: caller's next real eval will catch it */

    double d2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double scale = atol + fabs(y0[i]) * rtol;
        double e2 = (scratch_f1[i] - f0[i]) / scale;
        d2 += e2 * e2;
    }
    d2 = sqrt(d2 / (double)n) / h0;

    double h1;
    if (d1 <= 1e-15 && d2 <= 1e-15) h1 = fmax(1e-6, h0 * 1e-3);
    else h1 = pow(0.01 / fmax(d1, d2), 1.0 / 2.0); /* order=1 -> exponent 1/(order+1)=1/2 */

    double h = fmin(fmin(100.0 * h0, h1), interval_length);
    if (h_max_user > 0.0 && h > h_max_user) h = h_max_user;
    return h;
}

CPRBDFOpts cpr_ode_bdf_default_opts(void)
{
    CPRBDFOpts o;
    o.rtol = 1e-7;
    o.atol = 1e-12;
    o.h_init = 0.0;
    o.h_min = 0.0;
    o.h_max = 0.0;
    o.max_order = 5;
    o.max_steps = 200000;
    o.max_newton_iter = NEWTON_MAXITER;
    return o;
}

/* solve_bdf_system: simplified-Newton corrector for the BDF/NDF algebraic
 * equation c*f(t_new,y) - psi - d = 0, d = y - y_predict (scipy bdf.py's
 * solve_bdf_system). Unlike a fixed-residual-tolerance Newton test (the
 * previous version of this file's design, which broke down when
 * components share one atol across many decades of magnitude -- see the
 * project memory note on the BDF accuracy gap), convergence is judged by
 * the *correction* dy shrinking at a consistent geometric rate (`rate`),
 * extrapolated forward to predict whether continuing would reach `tol`:
 * this is scale-and-magnitude-robust by construction, since `dy` itself
 * (not the raw residual) is what's compared against `scale`. Returns 0 if
 * converged (sets *y_out, *d_out, *n_iter_out), 1 if f() failed (hard
 * error), 2 if Newton did not converge (caller shrinks h and retries; not
 * a hard error). */
static int solve_bdf_system(CPRODEFunc f, void *ctx, double t_new, const double *y_predict,
                              double c, const double *psi, const double *LU, const size_t *piv,
                              const double *scale, double tol, size_t n,
                              double *y_out, double *d_out, int *n_iter_out,
                              double *fwork, double *dy)
{
    memcpy(y_out, y_predict, n * sizeof(double));
    memset(d_out, 0, n * sizeof(double));
    double dy_norm_old = -1.0; /* sentinel for Python's `None` */
    int converged = 0;
    int iters = 0;

    for (int k = 0; k < NEWTON_MAXITER; k++) {
        iters = k + 1;
        if (f(t_new, y_out, fwork, ctx)) return 1;
        int all_finite = 1;
        for (size_t i = 0; i < n; i++) if (!isfinite(fwork[i])) { all_finite = 0; break; }
        if (!all_finite) break;

        for (size_t i = 0; i < n; i++) dy[i] = c * fwork[i] - psi[i] - d_out[i];
        cpr_lu_solve(LU, n, piv, dy);
        double dy_norm = rms_norm_scaled(dy, scale, n);

        double rate = -1.0;
        if (dy_norm_old >= 0.0) rate = dy_norm / dy_norm_old;

        if (rate >= 0.0 && (rate >= 1.0 ||
              pow(rate, (double)(NEWTON_MAXITER - k)) / (1.0 - rate) * dy_norm > tol))
            break;

        for (size_t i = 0; i < n; i++) { y_out[i] += dy[i]; d_out[i] += dy[i]; }

        if (dy_norm == 0.0 || (rate >= 0.0 && rate / (1.0 - rate) * dy_norm < tol)) {
            converged = 1;
            break;
        }
        dy_norm_old = dy_norm;
    }
    *n_iter_out = iters;
    return converged ? 0 : 2;
}

int cpr_ode_bdf(CPRODEFunc f, CPRODEJacFunc jac, void *ctx,
                 double t0, double t1, double *y, size_t n,
                 CPRBDFOpts opts, CPRODEStepCB step_cb, void *cb_ctx, char **errmsg)
{
    int max_order = opts.max_order;
    if (max_order < 1) max_order = 1;
    if (max_order > MAX_ORDER_CAP) max_order = MAX_ORDER_CAP;

    int dir = (t1 >= t0) ? 1 : -1;
    double span = fabs(t1 - t0);
    if (span == 0.0) return 0;

    BDFCoeffs co;
    bdf_coeffs_init(&co);

    /* Nordsieck array: MAX_ORDER_CAP+3 rows (indices 0..order+2 are live
     * at any given order<=MAX_ORDER_CAP), each of length n, row stride n
     * throughout (see change_D's docstring). */
    double *D = calloc((size_t)(MAX_ORDER_CAP + 3) * n, sizeof(double));
    double *f0 = malloc(n * sizeof(double));
    double *scratch_y1 = malloc(n * sizeof(double));
    double *scratch_f1 = malloc(n * sizeof(double));

    memcpy(D, y, n * sizeof(double)); /* D[0] = y0 */
    if (f(t0, y, f0, ctx)) {
        *errmsg = strdup("cpr_ode_bdf: f() failed at t0");
        free(D); free(f0); free(scratch_y1); free(scratch_f1);
        return 1;
    }

    double h_abs;
    if (opts.h_init > 0.0) {
        h_abs = opts.h_init;
        if (h_abs > span) h_abs = span;
    } else {
        h_abs = select_initial_step(f, ctx, t0, y, t1, opts.h_max, f0, dir,
                                      opts.rtol, opts.atol, n, scratch_y1, scratch_f1);
    }
    for (size_t i = 0; i < n; i++) D[1 * n + i] = f0[i] * h_abs * (double)dir; /* D[1] = f0*h*dir */

    int order = 1;
    int n_equal_steps = 0;

    double *J = malloc(n * n * sizeof(double));
    double *Jnewton = malloc(n * n * sizeof(double));
    size_t *piv = malloc(n * sizeof(size_t));
    int lu_valid = 0; /* mirrors scipy's `self.LU is None` */

    /* Initial Jacobian, computed once up front (mirrors scipy's
     * `_validate_jac` evaluating J at construction time). */
    if (jac) {
        if (jac(t0, y, J, ctx)) {
            *errmsg = strdup("cpr_ode_bdf: jac() failed at t0");
            free(D); free(f0); free(scratch_y1); free(scratch_f1);
            free(J); free(Jnewton); free(piv);
            return 1;
        }
    } else {
        double *ytmp = malloc(n * sizeof(double));
        double *f1tmp = malloc(n * sizeof(double));
        double *f0tmp = malloc(n * sizeof(double));
        int jrc = finite_diff_jac(f, ctx, t0, y, n, f0tmp, J, ytmp, f1tmp);
        free(ytmp); free(f1tmp); free(f0tmp);
        if (jrc) {
            *errmsg = strdup("cpr_ode_bdf: f() failed during initial finite-diff Jacobian");
            free(D); free(f0); free(scratch_y1); free(scratch_f1);
            free(J); free(Jnewton); free(piv);
            return 1;
        }
    }

    double newton_tol = fmax(10.0 * DBL_EPSILON / opts.rtol, fmin(0.03, sqrt(opts.rtol)));

    double *y_predict = malloc(n * sizeof(double));
    double *psi = malloc(n * sizeof(double));
    double *scale = malloc(n * sizeof(double));
    double *y_new = malloc(n * sizeof(double));
    double *d = malloc(n * sizeof(double));
    double *fwork = malloc(n * sizeof(double));
    double *dywork = malloc(n * sizeof(double));
    double *error = malloc(n * sizeof(double));

    double t = t0;
    int rc = 0;
    int steps = 0;

    while (dir * (t1 - t) > 1e-15 * span) {
        if (steps++ >= opts.max_steps) {
            *errmsg = strdup("cpr_ode_bdf: max_steps exceeded");
            rc = 1; goto done;
        }

        /* min_step: the smallest representable step from `t` in this
         * direction, scaled by 10 -- scipy bdf.py's floating-point floor
         * (distinct from opts.h_min, the user-requested hard-error floor
         * checked further below). max_step: opts.h_max (0 = unbounded). */
        double min_step_fp = 10.0 * fabs(nextafter(t, (dir > 0) ? INFINITY : -INFINITY) - t);
        double max_step = (opts.h_max > 0.0) ? opts.h_max : INFINITY;
        if (h_abs > max_step) {
            change_D(D, n, order, max_step / h_abs);
            h_abs = max_step;
            n_equal_steps = 0;
            lu_valid = 0;
        } else if (h_abs < min_step_fp) {
            change_D(D, n, order, min_step_fp / h_abs);
            h_abs = min_step_fp;
            n_equal_steps = 0;
            lu_valid = 0;
        }

        int current_jac = 0; /* "is J fresh as of this step attempt" -- see this file's
                                 module docstring discussion of scipy's current_jac flag;
                                 always starts false here since cpr_ode_bdf always has a
                                 real Jacobian function (analytic or finite-difference),
                                 never the "constant array, can't refresh" case. */

        int step_accepted = 0;
        double err_norm = 0.0;
        int n_iter = 0;
        while (!step_accepted) {
            if (h_abs < min_step_fp) {
                *errmsg = strdup("cpr_ode_bdf: step size underflowed below machine precision");
                rc = 1; goto done;
            }
            if (opts.h_min > 0.0 && h_abs < opts.h_min) {
                *errmsg = strdup("cpr_ode_bdf: step size underflowed below h_min");
                rc = 1; goto done;
            }

            double hh = (double)dir * h_abs;
            double t_new = t + hh;
            if (dir * (t_new - t1) > 0.0) {
                t_new = t1;
                change_D(D, n, order, fabs(t_new - t) / h_abs);
                n_equal_steps = 0;
                lu_valid = 0;
            }
            hh = t_new - t;
            h_abs = fabs(hh);

            for (size_t i = 0; i < n; i++) {
                double s = 0.0;
                for (int k = order; k >= 0; k--) s += D[(size_t)k * n + i];
                y_predict[i] = s;
            }
            for (size_t i = 0; i < n; i++) scale[i] = opts.atol + opts.rtol * fabs(y_predict[i]);
            for (size_t i = 0; i < n; i++) {
                double s = 0.0;
                for (int k = 1; k <= order; k++) s += D[(size_t)k * n + i] * co.gamma[k];
                psi[i] = s / co.alpha[order];
            }

            double cc = hh / co.alpha[order];
            int converged = 0;
            for (;;) {
                if (!lu_valid) {
                    for (size_t i = 0; i < n; i++)
                        for (size_t jx = 0; jx < n; jx++)
                            Jnewton[i * n + jx] = ((i == jx) ? 1.0 : 0.0) - cc * J[i * n + jx];
                    if (cpr_lu_factor(Jnewton, n, piv)) {
                        /* Singular Newton matrix: treat exactly like non-convergence
                         * (shrink h below), not as a hard error -- a transient
                         * near-singularity at one (h,J) combination is recoverable. */
                        converged = 0;
                        lu_valid = 0;
                        break;
                    }
                    lu_valid = 1;
                }
                int src = solve_bdf_system(f, ctx, t_new, y_predict, cc, psi, Jnewton, piv,
                                             scale, newton_tol, n, y_new, d, &n_iter, fwork, dywork);
                if (src == 1) { *errmsg = strdup("cpr_ode_bdf: f() failed"); rc = 1; goto done; }
                converged = (src == 0);
                if (!converged) {
                    if (current_jac) break;
                    if (jac) {
                        if (jac(t_new, y_predict, J, ctx)) {
                            *errmsg = strdup("cpr_ode_bdf: jac() failed"); rc = 1; goto done;
                        }
                    } else {
                        double *ytmp = malloc(n * sizeof(double));
                        double *f1tmp = malloc(n * sizeof(double));
                        double *f0tmp = malloc(n * sizeof(double));
                        int jrc = finite_diff_jac(f, ctx, t_new, y_predict, n, f0tmp, J, ytmp, f1tmp);
                        free(ytmp); free(f1tmp); free(f0tmp);
                        if (jrc) {
                            *errmsg = strdup("cpr_ode_bdf: f() failed during finite-diff Jacobian");
                            rc = 1; goto done;
                        }
                    }
                    lu_valid = 0;
                    current_jac = 1;
                    continue;
                }
                break;
            }

            if (!converged) {
                double factor = 0.5;
                change_D(D, n, order, factor);
                h_abs *= factor;
                n_equal_steps = 0;
                lu_valid = 0;
                continue;
            }

            double safety = 0.9 * (2.0 * NEWTON_MAXITER + 1.0) / (2.0 * NEWTON_MAXITER + (double)n_iter);

            for (size_t i = 0; i < n; i++) scale[i] = opts.atol + opts.rtol * fabs(y_new[i]);
            for (size_t i = 0; i < n; i++) error[i] = co.error_const[order] * d[i];
            err_norm = rms_norm_scaled(error, scale, n);

            if (err_norm > 1.0) {
                double factor = fmax(MIN_FACTOR, safety * pow(err_norm, -1.0 / (double)(order + 1)));
                change_D(D, n, order, factor);
                h_abs *= factor;
                n_equal_steps = 0;
                /* LU intentionally left as-is: Newton convergence was fine, only the
                 * error estimate was too large, so the cached Jacobian (just not this
                 * exact h's iteration matrix) is still a good approximation -- matches
                 * scipy bdf.py's explicit "we don't reset LU here" comment. Since `cc`
                 * is recomputed fresh from the new h_abs at the top of this inner loop,
                 * lu_valid staying 1 here means the NEXT pass will (deliberately) solve
                 * with a one-h-change-stale iteration matrix rather than refactoring --
                 * a standard modified-Newton tradeoff, not a bug. */
                continue;
            }
            step_accepted = 1;

            t = t_new;
            memcpy(y, y_new, n * sizeof(double));

            /* Nordsieck update from the converged correction `d`
             * (scipy bdf.py's "D^{j+1} y_n = D^j y_n - D^j y_{n-1}"
             * comment): fold the corrector's residual into the
             * higher-order difference rows, then propagate down. */
            for (size_t i = 0; i < n; i++) {
                D[(size_t)(order + 2) * n + i] = d[i] - D[(size_t)(order + 1) * n + i];
                D[(size_t)(order + 1) * n + i] = d[i];
            }
            for (int kk = order; kk >= 0; kk--)
                for (size_t i = 0; i < n; i++)
                    D[(size_t)kk * n + i] += D[(size_t)(kk + 1) * n + i];

            if (step_cb) step_cb(t, y, n, cb_ctx);

            n_equal_steps++;
            if (n_equal_steps < order + 1) break; /* not enough settled history yet to reconsider order/h */

            /* Order/step selection: estimate the local error that accepting at
             * order-1 / order (already known: err_norm) / order+1 would have
             * given, pick whichever maximises the resulting step-growth
             * factor (scipy bdf.py: `delta_order = argmax(factors) - 1`).
             * pow(INFINITY, -x) correctly evaluates to 0 for x>0 (IEEE754),
             * which is exactly what's wanted when order is already at its
             * floor/ceiling (error_m_norm/error_p_norm default to +inf
             * below, making that direction's factor 0 and so never the
             * argmax) -- no special-casing needed. */
            double error_m_norm = INFINITY, error_p_norm = INFINITY;
            if (order > 1) {
                for (size_t i = 0; i < n; i++) error[i] = co.error_const[order - 1] * D[(size_t)order * n + i];
                error_m_norm = rms_norm_scaled(error, scale, n);
            }
            if (order < max_order) {
                for (size_t i = 0; i < n; i++) error[i] = co.error_const[order + 1] * D[(size_t)(order + 2) * n + i];
                error_p_norm = rms_norm_scaled(error, scale, n);
            }
            double factors[3];
            factors[0] = pow(error_m_norm, -1.0 / (double)order);
            factors[1] = pow(err_norm, -1.0 / (double)(order + 1));
            factors[2] = pow(error_p_norm, -1.0 / (double)(order + 2));
            int best_idx = 0;
            for (int idx = 1; idx < 3; idx++) if (factors[idx] > factors[best_idx]) best_idx = idx;
            int delta_order = best_idx - 1;

            order += delta_order;
            if (order < 1) order = 1;
            if (order > max_order) order = max_order;

            double factor = fmin(MAX_FACTOR, safety * factors[best_idx]);
            change_D(D, n, order, factor);
            h_abs *= factor;
            n_equal_steps = 0;
            lu_valid = 0;
        }
    }

done:
    free(D); free(f0); free(scratch_y1); free(scratch_f1);
    free(J); free(Jnewton); free(piv);
    free(y_predict); free(psi); free(scale); free(y_new); free(d);
    free(fwork); free(dywork); free(error);
    return rc;
}
