/* test_spline.c -- checks cubic-spline fitting against an exact closed-form
 * function (a cubic spline through samples of a cubic polynomial must
 * reproduce it exactly, both for natural and not-a-knot boundary
 * conditions), linear interpolation, and the rate-table resampler against
 * a known power law. */
#include "cprimat/spline.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

static int close(double a, double b, double tol) { return fabs(a - b) < tol * fabs(b) + tol; }

/* f(x) = 1 + 2x - 3x^2 + 0.5x^3 -- any not-a-knot or natural cubic spline
 * through samples of a true cubic must reproduce it exactly (up to
 * round-off) away from the natural-spline's curvature-zero boundary
 * artifact, which is why the not-a-knot check below uses interior points
 * while the natural-spline check is restricted to where it actually holds. */
static double cubic(double x) { return 1.0 + 2.0 * x - 3.0 * x * x + 0.5 * x * x * x; }

int main(void)
{
    char *err = NULL;

    /* Not-a-knot: exact for any cubic everywhere (that's the point of the
     * boundary condition -- it doesn't impose curvature=0 like natural). */
    {
        size_t n = 8;
        double x[8], y[8];
        for (size_t i = 0; i < n; i++) { x[i] = (double)i - 2.0; y[i] = cubic(x[i]); }
        CPRCubicSpline sp;
        CHECK(cpr_cubic_spline_fit_notaknot(x, y, n, &sp, &err) == 0, "notaknot fit succeeds");
        int ok = 1;
        for (double xq = x[0]; xq <= x[n - 1]; xq += 0.137) {
            double got = cpr_cubic_spline_eval(&sp, xq);
            double want = cubic(xq);
            if (!close(got, want, 1e-9)) { ok = 0; break; }
        }
        CHECK(ok, "notaknot spline reproduces exact cubic on whole domain");
        cpr_cubic_spline_free(&sp);
    }

    /* Natural: exact for a *linear* function (zero curvature everywhere,
     * so the natural spline's zero-curvature boundary assumption is exact
     * everywhere, not just at the boundary). */
    {
        size_t n = 6;
        double x[6], y[6];
        for (size_t i = 0; i < n; i++) { x[i] = (double)i; y[i] = 3.0 - 1.5 * x[i]; }
        CPRCubicSpline sp;
        CHECK(cpr_cubic_spline_fit_natural(x, y, n, &sp, &err) == 0, "natural fit succeeds");
        int ok = 1;
        for (double xq = 0.0; xq <= 5.0; xq += 0.31) {
            double got = cpr_cubic_spline_eval(&sp, xq);
            double want = 3.0 - 1.5 * xq;
            if (!close(got, want, 1e-9)) { ok = 0; break; }
        }
        CHECK(ok, "natural spline reproduces exact line on whole domain");
        cpr_cubic_spline_free(&sp);
    }

    /* Linear interpolation sanity check. */
    {
        double x[3] = { 0.0, 1.0, 3.0 };
        double y[3] = { 0.0, 2.0, 10.0 };
        CHECK(close(cpr_interp_linear(x, y, 3, 0.5, CPR_EXTRAP_LINEAR), 1.0, 1e-12),
              "linear interp midpoint of first segment");
        CHECK(close(cpr_interp_linear(x, y, 3, 2.0, CPR_EXTRAP_LINEAR), 6.0, 1e-12),
              "linear interp midpoint of second segment");
        CHECK(close(cpr_interp_linear(x, y, 3, 10.0, CPR_EXTRAP_CONSTANT), 10.0, 1e-12),
              "constant extrapolation clamps above range");
        CHECK(close(cpr_interp_linear(x, y, 3, -5.0, CPR_EXTRAP_CONSTANT), 0.0, 1e-12),
              "constant extrapolation clamps below range");
    }

    /* Rate-table resampling: a power-law rate(T9) = T9^-2 is a straight
     * line in log-log space, so the log-log not-a-knot branch must
     * reproduce it (almost) exactly. */
    {
        size_t n_src = 10, n_dst = 5;
        double T9_src[10], rate_src[10];
        for (size_t i = 0; i < n_src; i++) {
            T9_src[i] = pow(10.0, -2.0 + 0.5 * (double)i);
            rate_src[i] = pow(T9_src[i], -2.0);
        }
        double T9_dst[5] = { 1e-1, 3e-1, 1.0, 3.0, 10.0 };
        double rate_dst[5];
        CHECK(cpr_resample_rate_table(T9_src, rate_src, n_src, T9_dst, rate_dst, n_dst, &err) == 0,
              "resample power-law table succeeds");
        int ok = 1;
        for (size_t i = 0; i < n_dst; i++)
            if (!close(rate_dst[i], pow(T9_dst[i], -2.0), 1e-6)) ok = 0;
        CHECK(ok, "resampled power-law rate matches analytic T9^-2");
    }

    /* Non-positive rate column (e.g. an all-zero error column) must fall
     * back to linear interpolation in log10(T9) without crashing. */
    {
        double T9_src[4] = { 1e-2, 1e-1, 1.0, 10.0 };
        double rate_src[4] = { 0.0, 0.0, 0.0, 0.0 };
        double T9_dst[2] = { 5e-2, 5.0 };
        double rate_dst[2];
        CHECK(cpr_resample_rate_table(T9_src, rate_src, 4, T9_dst, rate_dst, 2, &err) == 0,
              "resample all-zero table succeeds (fallback path)");
        CHECK(close(rate_dst[0], 0.0, 1e-12) && close(rate_dst[1], 0.0, 1e-12),
              "resampled all-zero table stays zero");
    }

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
