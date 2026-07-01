/* ode_rk.c -- see ode_rk.h. Dormand-Prince RK5(4)7M (Dormand & Prince
 * 1980, "A family of embedded Runge-Kutta formulae", J. Comput. Appl. Math.
 * 6) with standard PI-ish step-size control (the classic "err^(-1/5) with
 * safety factor and min/max clamps" rule, e.g. Hairer/Norsett/Wanner II.4).
 */
#include "ode_rk.h"

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

/* Dense-output ("free" continuous extension) coefficients for the Dormand-
 * Prince RK5(4)7 pair, order 4 accurate, requiring no extra f() evaluations
 * beyond the 7 stage derivatives k1..k7 already computed for the step. This
 * is the standard quartic-in-theta interpolant for this Butcher tableau
 * (Shampine 1986; the same construction underlies scipy's
 * `scipy.integrate.RK45` dense output, `scipy/integrate/_ivp/rk.py`'s `P`
 * matrix for the `RK45` class -- BSD-licensed, and reproduced here from
 * that primary source's exact rational coefficients to guarantee bit-
 * faithful agreement with a well-tested reference implementation):
 *
 *   y(t_old + theta*h) = y_old + h * sum_{s in {1,3,4,5,6,7}} k_s * P_s(theta)
 *
 * where P_s(theta) = P[s][0]*theta + P[s][1]*theta^2 + P[s][2]*theta^3
 * + P[s][3]*theta^4, theta = (t - t_old)/h in [0,1]. Row 2 (k2) is
 * identically zero, matching A72=0 in the Butcher tableau above, so k2 is
 * never needed here. By construction P_s(0)=0 for every s (so theta=0
 * reduces to y_old exactly) and sum_s P_s(1) equals the 5th-order weights
 * A71/A73/.../A76 above (so theta=1 reduces to y5 exactly) -- verified
 * numerically in test_ode_rk.c. */
static const double DOP_P1[4] = { 1.0, -8048581381.0 / 2820520608.0,
                                   8663915743.0 / 2820520608.0, -12715105075.0 / 11282082432.0 };
static const double DOP_P3[4] = { 0.0, 131558114200.0 / 32700410799.0,
                                   -68118460800.0 / 10900136933.0, 87487479700.0 / 32700410799.0 };
static const double DOP_P4[4] = { 0.0, -1754552775.0 / 470086768.0,
                                   14199869525.0 / 1410260304.0, -10690763975.0 / 1880347072.0 };
static const double DOP_P5[4] = { 0.0, 127303824393.0 / 49829197408.0,
                                   -318862633887.0 / 49829197408.0, 701980252875.0 / 199316789632.0 };
static const double DOP_P6[4] = { 0.0, -282668133.0 / 205662961.0,
                                   2019193451.0 / 616988883.0, -1453857185.0 / 822651844.0 };
static const double DOP_P7[4] = { 0.0, 40617522.0 / 29380423.0,
                                   -110615467.0 / 29380423.0, 69997945.0 / 29380423.0 };

void cpr_ode_dense_eval(double theta, double hh, const double *y_old,
                         const double *k1, const double *k3, const double *k4,
                         const double *k5, const double *k6, const double *k7,
                         size_t n, double *out)
{
    double th1 = theta, th2 = th1 * theta, th3 = th2 * theta, th4 = th3 * theta;
    double b1 = DOP_P1[0] * th1 + DOP_P1[1] * th2 + DOP_P1[2] * th3 + DOP_P1[3] * th4;
    double b3 = DOP_P3[0] * th1 + DOP_P3[1] * th2 + DOP_P3[2] * th3 + DOP_P3[3] * th4;
    double b4 = DOP_P4[0] * th1 + DOP_P4[1] * th2 + DOP_P4[2] * th3 + DOP_P4[3] * th4;
    double b5 = DOP_P5[0] * th1 + DOP_P5[1] * th2 + DOP_P5[2] * th3 + DOP_P5[3] * th4;
    double b6 = DOP_P6[0] * th1 + DOP_P6[1] * th2 + DOP_P6[2] * th3 + DOP_P6[3] * th4;
    double b7 = DOP_P7[0] * th1 + DOP_P7[1] * th2 + DOP_P7[2] * th3 + DOP_P7[3] * th4;
    for (size_t i = 0; i < n; i++)
        out[i] = y_old[i] + hh * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i]
                                   + b5 * k5[i] + b6 * k6[i] + b7 * k7[i]);
}

CPRRKOpts cpr_ode_rk_default_opts(void)
{
    CPRRKOpts o;
    o.rtol = 1e-7;
    o.atol = 1e-12;
    o.h_init = 0.0;
    o.h_min = 0.0;
    o.h_max = 0.0;
    o.max_steps = 100000;
    o.t_eval = NULL;
    o.n_eval = 0;
    o.y_eval = NULL;
    o.dense_cb = NULL;
    o.dense_ctx = NULL;
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
    /* Cursor into opts.t_eval: points strictly before t_eval[eval_idx] (in
     * the integration direction) have already been filled into y_eval.
     * t_eval is required to be sorted in the same direction as the
     * integration, so a single forward-only cursor suffices -- no need to
     * re-scan from the start on every step. */
    size_t eval_idx = 0;
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
            /* Accept the step. Fill any requested dense-output points that
             * fall within [t, t+hh] (this step's span) and/or invoke
             * dense_cb, *before* t/y are advanced to the new state, since
             * both need the step's starting point y_old=y (still un-
             * overwritten here) and its stage derivatives k1..k7 (k1 not
             * yet FSAL-recycled into k7 below). */
            if (opts.t_eval != NULL) {
                double t_new = t + hh;
                /* eps: small tolerance (relative to the total integration
                 * span) absorbing roundoff in t_new (built by repeated
                 * float addition) vs. a t_eval value intended to land
                 * exactly on it -- e.g. the very last point, where t_new
                 * is forced equal to t1 bit-for-bit by the "do not step
                 * past t1" clamp above, but earlier points accumulate the
                 * usual ULP-level drift. */
                double eps = 1e-9 * span;
                while (eval_idx < opts.n_eval
                       && dir * (opts.t_eval[eval_idx] - t_new) <= eps) {
                    double theta = (opts.t_eval[eval_idx] - t) / hh;
                    cpr_ode_dense_eval(theta, hh, y, k1, k3, k4, k5, k6, k7, n,
                                       opts.y_eval + eval_idx * n);
                    eval_idx++;
                }
            }
            if (opts.dense_cb) opts.dense_cb(t, hh, y, k1, k3, k4, k5, k6, k7, n, opts.dense_ctx);
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
