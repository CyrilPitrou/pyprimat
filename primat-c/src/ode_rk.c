/* ode_rk.c -- see cprimat/ode_rk.h. Dormand-Prince RK5(4)7M (Dormand & Prince
 * 1980, "A family of embedded Runge-Kutta formulae", J. Comput. Appl. Math.
 * 6) with standard PI-ish step-size control (the classic "err^(-1/5) with
 * safety factor and min/max clamps" rule, e.g. Hairer/Norsett/Wanner II.4).
 */
#include "cprimat/ode_rk.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Butcher tableau (Dormand-Prince, the same one scipy's RK45 uses). */
static const double C2 = 1.0 / 5.0, C3 = 3.0 / 10.0, C4 = 4.0 / 5.0, C5 = 8.0 / 9.0;

static const double A21 = 1.0 / 5.0;
static const double A31 = 3.0 / 40.0, A32 = 9.0 / 40.0;
static const double A41 = 44.0 / 45.0, A42 = -56.0 / 15.0, A43 = 32.0 / 9.0;
static const double A51 = 19372.0 / 6561.0, A52 = -25360.0 / 2187.0,
                     A53 = 64448.0 / 6561.0, A54 = -212.0 / 729.0;
static const double A61 = 9017.0 / 3168.0, A62 = -355.0 / 33.0,
                     A63 = 46732.0 / 5247.0, A64 = 49.0 / 176.0, A65 = -5103.0 / 18656.0;
static const double A71 = 35.0 / 384.0, A73 = 500.0 / 1113.0, A74 = 125.0 / 192.0,
                     A75 = -2187.0 / 6784.0, A76 = 11.0 / 84.0;

/* The 5th-order solution weights equal row 7 of A (FSAL: k7 of this step
 * is k1 of the next, though we recompute it for simplicity rather than
 * carrying it across steps -- one extra f() eval per step, negligible
 * next to the other 6), so y5 below is built directly from A71/A73/.../A76
 * with no separate B-weight constants needed. */
/* 4th-order solution weights (the embedded lower-order estimate). */
static const double B1s = 5179.0 / 57600.0, B3s = 7571.0 / 16695.0, B4s = 393.0 / 640.0,
                     B5s = -92097.0 / 339200.0, B6s = 187.0 / 2100.0, B7s = 1.0 / 40.0;

CPRRKOpts cpr_ode_rk_default_opts(void)
{
    CPRRKOpts o;
    o.rtol = 1e-7;
    o.atol = 1e-12;
    o.h_init = 0.0;
    o.h_min = 0.0;
    o.h_max = 0.0;
    o.max_steps = 100000;
    return o;
}

int cpr_ode_rk45(CPRODEFunc f, void *ctx, double t0, double t1, double *y, size_t n,
                  CPRRKOpts opts, CPRODEStepCB step_cb, void *cb_ctx, char **errmsg)
{
    int dir = (t1 >= t0) ? 1 : -1;
    double span = fabs(t1 - t0);
    if (span == 0.0) return 0;

    double *k1 = malloc(n * sizeof(double));
    double *k2 = malloc(n * sizeof(double));
    double *k3 = malloc(n * sizeof(double));
    double *k4 = malloc(n * sizeof(double));
    double *k5 = malloc(n * sizeof(double));
    double *k6 = malloc(n * sizeof(double));
    double *k7 = malloc(n * sizeof(double));
    double *ytmp = malloc(n * sizeof(double));
    double *y5 = malloc(n * sizeof(double));
    double *y4 = malloc(n * sizeof(double));

    double t = t0;
    /* Crude initial step: a small fraction of the total span, refined by
     * the controller within the first couple of steps regardless. */
    double h = (opts.h_init > 0.0) ? opts.h_init : span / 100.0;
    if (opts.h_max > 0.0 && h > opts.h_max) h = opts.h_max;

    int rc = 0;
    int steps = 0;

    if (f(t, y, k1, ctx)) { *errmsg = strdup("cpr_ode_rk45: f() failed at t0"); rc = 1; goto done; }

    while (dir * (t1 - t) > 1e-15 * span) {
        if (steps++ >= opts.max_steps) {
            *errmsg = strdup("cpr_ode_rk45: max_steps exceeded");
            rc = 1; goto done;
        }
        /* Do not step past t1. */
        double hh = dir * h;
        if (dir * (t + hh - t1) > 0.0) hh = t1 - t;
        h = fabs(hh);

        for (size_t i = 0; i < n; i++) ytmp[i] = y[i] + hh * A21 * k1[i];
        if (f(t + C2 * hh, ytmp, k2, ctx)) { *errmsg = strdup("cpr_ode_rk45: f() failed (stage 2)"); rc = 1; goto done; }

        for (size_t i = 0; i < n; i++) ytmp[i] = y[i] + hh * (A31 * k1[i] + A32 * k2[i]);
        if (f(t + C3 * hh, ytmp, k3, ctx)) { *errmsg = strdup("cpr_ode_rk45: f() failed (stage 3)"); rc = 1; goto done; }

        for (size_t i = 0; i < n; i++) ytmp[i] = y[i] + hh * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        if (f(t + C4 * hh, ytmp, k4, ctx)) { *errmsg = strdup("cpr_ode_rk45: f() failed (stage 4)"); rc = 1; goto done; }

        for (size_t i = 0; i < n; i++)
            ytmp[i] = y[i] + hh * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        if (f(t + C5 * hh, ytmp, k5, ctx)) { *errmsg = strdup("cpr_ode_rk45: f() failed (stage 5)"); rc = 1; goto done; }

        for (size_t i = 0; i < n; i++)
            ytmp[i] = y[i] + hh * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
        if (f(t + hh, ytmp, k6, ctx)) { *errmsg = strdup("cpr_ode_rk45: f() failed (stage 6)"); rc = 1; goto done; }

        for (size_t i = 0; i < n; i++)
            y5[i] = y[i] + hh * (A71 * k1[i] + A73 * k3[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i]);
        if (f(t + hh, y5, k7, ctx)) { *errmsg = strdup("cpr_ode_rk45: f() failed (stage 7)"); rc = 1; goto done; }

        for (size_t i = 0; i < n; i++)
            y4[i] = y[i] + hh * (B1s * k1[i] + B3s * k3[i] + B4s * k4[i] + B5s * k5[i] + B6s * k6[i] + B7s * k7[i]);

        /* scipy/Hairer-style mixed error norm: RMS of (y5-y4) scaled by
         * atol + rtol*max(|y_old|, |y_new|). */
        double err2 = 0.0;
        for (size_t i = 0; i < n; i++) {
            double scale = opts.atol + opts.rtol * fmax(fabs(y[i]), fabs(y5[i]));
            double e = (y5[i] - y4[i]) / scale;
            err2 += e * e;
        }
        double err_norm = sqrt(err2 / (double)n);

        if (err_norm <= 1.0 || h <= (opts.h_min > 0.0 ? opts.h_min : 0.0) * 1.0001) {
            /* Accept the step. */
            t += hh;
            memcpy(y, y5, n * sizeof(double));
            memcpy(k1, k7, n * sizeof(double)); /* FSAL: reuse k7 as next k1 */
            if (step_cb) step_cb(t, y, n, cb_ctx);
        }
        /* else: reject -- t, y, k1 unchanged, only h shrinks below. */

        /* Step-size update (PI-like single-step controller, safety factor
         * 0.9, order 5 error exponent 1/5, clamped to avoid wild swings). */
        double fac;
        if (err_norm == 0.0) fac = 5.0;
        else fac = 0.9 * pow(1.0 / err_norm, 1.0 / 5.0);
        if (fac > 5.0) fac = 5.0;
        if (fac < 0.2) fac = 0.2;
        h *= fac;
        if (opts.h_max > 0.0 && h > opts.h_max) h = opts.h_max;
        if (opts.h_min > 0.0 && h < opts.h_min) {
            if (err_norm > 1.0) {
                *errmsg = strdup("cpr_ode_rk45: step size underflowed below h_min");
                rc = 1; goto done;
            }
            h = opts.h_min;
        }
    }

done:
    free(k1); free(k2); free(k3); free(k4); free(k5); free(k6); free(k7);
    free(ytmp); free(y5); free(y4);
    return rc;
}
