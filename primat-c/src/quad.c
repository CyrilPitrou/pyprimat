/* quad.c -- see quad.h. Classic adaptive Simpson (Richardson-
 * extrapolated error estimate), e.g. as in Burden & Faires S4.6. */
#include "quad.h"
#include <math.h>

static double simpson(double a, double b, double fa, double fm, double fb)
{
    return (b - a) / 6.0 * (fa + 4.0 * fm + fb);
}

static double fabsd(double x) { return x < 0.0 ? -x : x; }

static double recurse(CPRQuadFunc f, void *ctx, double a, double b,
                       double fa, double fm, double fb, double whole,
                       double tol, int depth, double *err_accum)
{
    double m = 0.5 * (a + b);
    double lm = 0.5 * (a + m), rm = 0.5 * (m + b);
    double flm = f(lm, ctx), frm = f(rm, ctx);
    double left = simpson(a, m, fa, flm, fm);
    double right = simpson(m, b, fm, frm, fb);
    double refined = left + right;

    /* Richardson extrapolation: for Simpson's rule the error of the
     * refined (two-panel) estimate is ~(refined - whole)/15, so the
     * standard adaptive-Simpson stopping test compares |refined - whole|
     * against 15*tol. */
    if (depth <= 0 || fabsd(refined - whole) <= 15.0 * tol) {
        *err_accum += (refined - whole) / 15.0;
        return refined + (refined - whole) / 15.0;
    }
    double tol_half = tol / 2.0;
    double s1 = recurse(f, ctx, a, m, fa, flm, fm, left, tol_half, depth - 1, err_accum);
    double s2 = recurse(f, ctx, m, b, fm, frm, fb, right, tol_half, depth - 1, err_accum);
    return s1 + s2;
}

double cpr_quad_adaptive(CPRQuadFunc f, void *ctx, double a, double b,
                          double tol, int max_depth, double *err_estimate)
{
    double m = 0.5 * (a + b);
    double fa = f(a, ctx), fm = f(m, ctx), fb = f(b, ctx);
    double whole = simpson(a, b, fa, fm, fb);
    double err_accum = 0.0;
    double result = recurse(f, ctx, a, b, fa, fm, fb, whole, tol, max_depth, &err_accum);
    if (err_estimate) *err_estimate = fabsd(err_accum);
    return result;
}

void cpr_gauss_legendre(int n, double *nodes, double *weights)
{
    const double pi = 3.14159265358979323846;
    int m = (n + 1) / 2; /* roots come in +-pairs; only solve for half */
    for (int i = 0; i < m; i++) {
        /* Asymptotic initial guess for the i-th root (0-indexed from the
         * positive end), then polish by Newton's method on P_n. */
        double z = cos(pi * (i + 0.75) / (n + 0.5));
        double z_prev;
        double pn = 0.0, pn1;
        for (int iter = 0; iter < 100; iter++) {
            /* Three-term recurrence (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x),
             * built up from P_0=1, P_1=x, to get P_n(z) and its derivative. */
            double p0 = 1.0, p1 = z;
            for (int k = 2; k <= n; k++) {
                double p2 = ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            pn = p1;
            pn1 = n * (z * p1 - p0) / (z * z - 1.0); /* P_n'(z) */
            z_prev = z;
            z = z_prev - pn / pn1;
            if (fabs(z - z_prev) < 1e-15) break;
        }
        double w = 2.0 / ((1.0 - z * z) * pn1 * pn1);
        /* Store ascending: root i (from the cos() ordering, descending in z)
         * maps to position (n-1-i) from the left, and its mirror -z to i. */
        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
}
