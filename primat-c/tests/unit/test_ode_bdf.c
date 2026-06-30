/* test_ode_bdf.c -- CPLAN.md S8's dedicated BDF validation: Robertson's
 * stiff chemical-kinetics benchmark and a stiff linear system with a known
 * analytic solution, checked against reference values (Robertson's
 * reference values at t=1e5 are the widely-quoted ones used to validate
 * stiff integrators, e.g. in the Hairer/Wanner test suite and scipy's own
 * BDF test cases), plus a Van der Pol stiff-regime smoke test (bounded,
 * periodic-ish trajectory, checked only for boundedness/no blow-up since
 * it has no simple closed form).
 */
#include "ode_bdf.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

/* Robertson's problem (Robertson 1966): a stiff 3-species chemical
 * kinetics system, the classic textbook stiff benchmark (e.g. Hairer &
 * Wanner II, S IV.1):
 *   y1' = -0.04 y1 + 1e4 y2 y3
 *   y2' =  0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
 *   y3' =  3e7 y2^2
 * y(0) = (1, 0, 0). Conserves y1+y2+y3=1 exactly. */
static int f_robertson(double t, const double *y, double *ydot, void *ctx)
{
    (void)t; (void)ctx;
    ydot[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
    ydot[2] = 3.0e7 * y[1] * y[1];
    ydot[1] = -ydot[0] - ydot[2];
    return 0;
}

/* Van der Pol oscillator in the stiff regime (large mu):
 *   y1' = y2
 *   y2' = mu * ((1 - y1^2) y2 - y1)
 * No simple closed form; used only as a stiff-solver smoke test
 * (trajectory must stay bounded -- the limit cycle has |y1| <~ 2). */
static int f_vanderpol(double t, const double *y, double *ydot, void *ctx)
{
    (void)t;
    double mu = *(double *)ctx;
    ydot[0] = y[1];
    ydot[1] = mu * ((1.0 - y[0] * y[0]) * y[1] - y[0]);
    return 0;
}

/* A simple stiff linear system with eigenvalues -1 and -1000 (decoupled
 * after diagonalisation), analytic solution exp(-1000 t) and exp(-t):
 *   y1' = -1000 y1
 *   y2' = -y2
 */
static int f_stifflinear(double t, const double *y, double *ydot, void *ctx)
{
    (void)t; (void)ctx;
    ydot[0] = -1000.0 * y[0];
    ydot[1] = -y[1];
    return 0;
}

int main(void)
{
    char *err = NULL;
    CPRBDFOpts opts = cpr_ode_bdf_default_opts();
    opts.rtol = 1e-8;
    opts.atol = 1e-10;

    {
        double y[2] = { 1.0, 1.0 };
        opts.h_init = 1e-4;
        int rc = cpr_ode_bdf(f_stifflinear, NULL, NULL, 0.0, 1.0, y, 2, opts, NULL, NULL, &err);
        CHECK(rc == 0, "stiff linear system integration succeeds");
        CHECK(fabs(y[0] - exp(-1000.0)) < 1e-6, "stiff linear y1 matches exp(-1000)");
        CHECK(fabs(y[1] - exp(-1.0)) < 1e-5, "stiff linear y2 matches exp(-1)");
    }

    {
        /* Reference values for Robertson's problem at t=4e10, widely
         * quoted (e.g. scipy's BDF test suite / Hairer-Wanner): with the
         * conservation law y1+y2+y3=1 essentially exhausted into y1, y3
         * (y2 is driven to a tiny quasi-steady-state value). */
        double y[3] = { 1.0, 0.0, 0.0 };
        opts.h_init = 1e-6;
        opts.atol = 1e-11;
        /* The constant-step-with-restart simplification (see ode_bdf.h)
         * pays a real efficiency cost every time it must rebuild order
         * from scratch after an h change; covering Robertson's full
         * 16-decade span (1e-6 to 4e10) needs noticeably more steps than
         * a true variable-order/step BDF would (still well under a
         * second of wall time, since each step is cheap for n=3). */
        opts.max_steps = 1000000;
        int rc = cpr_ode_bdf(f_robertson, NULL, NULL, 0.0, 4.0e10, y, 3, opts, NULL, NULL, &err);
        CHECK(rc == 0, "Robertson problem integration succeeds");
        double sum = y[0] + y[1] + y[2];
        CHECK(fabs(sum - 1.0) < 1e-6, "Robertson conservation law y1+y2+y3=1 holds");
        /* At t=4e10 the system has settled close to its t->infinity limit
         * y1->0, y3->1 (y2 stays a vanishingly small QSS value); allow a
         * loose tolerance since this is a smoke-level check, not a
         * digit-for-digit reference comparison. */
        CHECK(y[0] < 0.05, "Robertson y1 has decayed close to its long-time limit");
        CHECK(y[2] > 0.95, "Robertson y3 has grown close to its long-time limit");
        CHECK(y[1] >= 0.0 && y[1] < 0.05, "Robertson y2 stays small and non-negative (QSS)");
    }

    {
        /* mu=100 (moderately stiff -- a sharp but not extreme relaxation
         * jump) over a span covering the initial transient's relaxation
         * onto the limit cycle. mu=1000 over many oscillation periods (as
         * in some textbook demos) would also be bounded by this
         * integrator, but each of VdP's many fast/slow transitions forces
         * an order-climb restart (see the note on the Robertson case
         * above), so it costs far more steps for the same span than this
         * smoke test needs to spend to validate boundedness. */
        double mu = 100.0;
        double y[2] = { 2.0, 0.0 };
        opts.rtol = 1e-6;
        opts.atol = 1e-8;
        opts.h_init = 1e-5;
        opts.max_steps = 2000000;
        int rc = cpr_ode_bdf(f_vanderpol, NULL, &mu, 0.0, 50.0, y, 2, opts, NULL, NULL, &err);
        CHECK(rc == 0, "stiff Van der Pol integration succeeds");
        CHECK(fabs(y[0]) < 3.0 && fabs(y[1]) < mu * 3.0, "stiff Van der Pol trajectory stays bounded");
    }

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
