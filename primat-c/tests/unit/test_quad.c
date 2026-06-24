/* test_quad.c -- checks cpr_quad_adaptive against closed-form integrals:
 * a polynomial (exact for Simpson regardless of tolerance), sin(x), and a
 * Gaussian (transcendental, needs genuine adaptivity to converge). */
#include "cprimat/quad.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

static double f_cubic(double x, void *ctx) { (void)ctx; return 3.0 * x * x - 2.0 * x + 1.0; }
static double f_sin(double x, void *ctx) { (void)ctx; return sin(x); }
static double f_gauss(double x, void *ctx) { (void)ctx; return exp(-x * x); }

int main(void)
{
    double err;

    /* Integral of (3x^2 - 2x + 1) from 0 to 2 = [x^3 - x^2 + x] = 8-4+2 = 6. */
    double r1 = cpr_quad_adaptive(f_cubic, NULL, 0.0, 2.0, 1e-12, 20, &err);
    CHECK(fabs(r1 - 6.0) < 1e-9, "cubic integral matches closed form");

    /* Integral of sin(x) from 0 to pi = 2. */
    double r2 = cpr_quad_adaptive(f_sin, NULL, 0.0, M_PI, 1e-10, 30, &err);
    CHECK(fabs(r2 - 2.0) < 1e-8, "sin integral over [0, pi] matches 2");

    /* Integral of exp(-x^2) from -6 to 6 ~ sqrt(pi) to high accuracy
     * (tails beyond x=6 are utterly negligible: erfc(6) ~ 2e-17, well
     * below the comparison tolerance below). */
    double r3 = cpr_quad_adaptive(f_gauss, NULL, -6.0, 6.0, 1e-10, 30, &err);
    CHECK(fabs(r3 - sqrt(M_PI)) < 1e-8, "gaussian integral matches sqrt(pi)");

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
