/* test_vegas.c -- convergence checks for cpr_vegas_integrate against known
 * 2D integrals: a smooth separable Gaussian, and a sharply localised
 * Gaussian-times-step feature narrow compared to its integration domain
 * (the same failure mode weak_rates.c's deterministic-quadrature comments
 * warn about for naive single-pass quadrature -- VEGAS's adaptive
 * importance sampling should instead zoom in on it automatically). */
#include "cprimat/vegas.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

/* f(x,y) = exp(-x^2 - y^2) over [-3,3]x[-3,3]: analytic integral over all of
 * R^2 is pi; over [-3,3]^2 the tails are negligible (<1e-4), so pi is also
 * an excellent reference value here. */
static double gaussian2d(const double x[2], void *ctx)
{
    (void)ctx;
    return exp(-x[0] * x[0] - x[1] * x[1]);
}

/* A narrow peak (width ~0.05) sitting inside a much wider domain
 * ([-10,10]x[-10,10]), normalised so its analytic integral is exactly 1
 * (a 2D Gaussian of amplitude 1/(2*pi*sigma^2) and std sigma). */
static double narrow_peak(const double x[2], void *ctx)
{
    (void)ctx;
    const double sigma = 0.05;
    double r2 = x[0] * x[0] + x[1] * x[1];
    return exp(-r2 / (2.0 * sigma * sigma)) / (2.0 * M_PI * sigma * sigma);
}

int main(void)
{
    {
        double lo[2] = {-3.0, -3.0}, hi[2] = {3.0, 3.0};
        CPRVegasResult r = cpr_vegas_integrate(gaussian2d, NULL, lo, hi,
                                                5000, 10, 10, 1234);
        double rel_err = fabs(r.mean - M_PI) / M_PI;
        printf("gaussian2d: mean=%.6f sigma=%.6f (expected pi=%.6f, rel_err=%.4f)\n",
               r.mean, r.sigma, M_PI, rel_err);
        CHECK(rel_err < 0.01, "separable Gaussian integral within 1% of pi");
    }

    {
        double lo[2] = {-10.0, -10.0}, hi[2] = {10.0, 10.0};
        CPRVegasResult r = cpr_vegas_integrate(narrow_peak, NULL, lo, hi,
                                                20000, 15, 15, 5678);
        double rel_err = fabs(r.mean - 1.0);
        printf("narrow_peak: mean=%.6f sigma=%.6f (expected 1.0, abs_err=%.4f)\n",
               r.mean, r.sigma, rel_err);
        CHECK(rel_err < 0.02, "narrow peak (width << domain) integral within 2% of 1.0");
    }

    {
        /* Determinism: same seed must reproduce the same result bit-for-bit. */
        double lo[2] = {-3.0, -3.0}, hi[2] = {3.0, 3.0};
        CPRVegasResult r1 = cpr_vegas_integrate(gaussian2d, NULL, lo, hi, 1000, 5, 5, 999);
        CPRVegasResult r2 = cpr_vegas_integrate(gaussian2d, NULL, lo, hi, 1000, 5, 5, 999);
        CHECK(r1.mean == r2.mean && r1.sigma == r2.sigma, "same seed reproduces the same result");
    }

    printf(failures == 0 ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    return failures == 0 ? 0 : 1;
}
