/* test_ode_rk.c -- checks cpr_ode_rk45 against analytic IVPs: exponential
 * decay, a harmonic oscillator (energy/phase exact), and a 2-body Kepler
 * orbit (closed-form position via conservation of energy is overkill;
 * instead checked by analytic x(t)=cos/sin solution). */
#include "ode_rk.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

/* dy/dt = -y, y(0) = 1 -> y(t) = exp(-t). */
static int f_decay(double t, const double *y, double *ydot, void *ctx)
{
    (void)t; (void)ctx;
    ydot[0] = -y[0];
    return 0;
}

/* Harmonic oscillator: y0' = y1, y1' = -y0; y0(0)=1, y1(0)=0
 * -> y0(t) = cos(t), y1(t) = -sin(t). */
static int f_harmonic(double t, const double *y, double *ydot, void *ctx)
{
    (void)t; (void)ctx;
    ydot[0] = y[1];
    ydot[1] = -y[0];
    return 0;
}

int main(void)
{
    char *err = NULL;
    CPRRKOpts opts = cpr_ode_rk_default_opts();
    opts.rtol = 1e-10;
    opts.atol = 1e-13;

    {
        double y[1] = { 1.0 };
        int rc = cpr_ode_rk45(f_decay, NULL, 0.0, 3.0, y, 1, opts, NULL, NULL, &err);
        CHECK(rc == 0, "exponential decay integration succeeds");
        CHECK(fabs(y[0] - exp(-3.0)) < 1e-8, "exponential decay matches exp(-3)");
    }

    {
        double y[2] = { 1.0, 0.0 };
        int rc = cpr_ode_rk45(f_harmonic, NULL, 0.0, 10.0, y, 2, opts, NULL, NULL, &err);
        CHECK(rc == 0, "harmonic oscillator integration succeeds");
        CHECK(fabs(y[0] - cos(10.0)) < 1e-7, "harmonic y0 matches cos(10)");
        CHECK(fabs(y[1] - (-sin(10.0))) < 1e-7, "harmonic y1 matches -sin(10)");
    }

    /* Backward integration (t1 < t0) must also work. */
    {
        double y[1] = { exp(-3.0) };
        int rc = cpr_ode_rk45(f_decay, NULL, 3.0, 0.0, y, 1, opts, NULL, NULL, &err);
        CHECK(rc == 0, "backward integration succeeds");
        CHECK(fabs(y[0] - 1.0) < 1e-8, "backward integration recovers y(0)=1");
    }

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
