/* spline.c -- see cprimat/spline.h. */
#include "cprimat/spline.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Binary search for the segment i such that x[i] <= xq <= x[i+1] (clamped
 * to [0, n-2] outside the table -- the caller then evaluates with an
 * unclamped dx to extrapolate using that boundary segment). */
static size_t find_segment(const double *x, size_t n, double xq)
{
    if (xq <= x[0]) return 0;
    if (xq >= x[n - 1]) return n - 2;
    size_t lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        size_t mid = (lo + hi) / 2;
        if (x[mid] <= xq) lo = mid; else hi = mid;
    }
    return lo;
}

double cpr_interp_linear(const double *x, const double *y, size_t n, double xq,
                          CPRExtrapMode mode)
{
    if (mode == CPR_EXTRAP_CONSTANT) {
        if (xq <= x[0]) return y[0];
        if (xq >= x[n - 1]) return y[n - 1];
    }
    size_t i = find_segment(x, n, xq);
    double t = (xq - x[i]) / (x[i + 1] - x[i]);
    return y[i] + t * (y[i + 1] - y[i]);
}

double cpr_interp_quadratic_local(const double *x, const double *y, size_t n, double xq)
{
    size_t i = find_segment(x, n, xq);          /* bracketing segment [i, i+1] */
    /* 3-point window {k, k+1, k+2} centred on the segment, clamped so it
     * stays inside [0, n-1]. */
    size_t k = (i == 0) ? 0 : i - 1;
    if (k + 2 > n - 1) k = n - 3;
    double x0 = x[k], x1 = x[k + 1], x2 = x[k + 2];
    double y0 = y[k], y1 = y[k + 1], y2 = y[k + 2];
    double L0 = (xq - x1) * (xq - x2) / ((x0 - x1) * (x0 - x2));
    double L1 = (xq - x0) * (xq - x2) / ((x1 - x0) * (x1 - x2));
    double L2 = (xq - x0) * (xq - x1) / ((x2 - x0) * (x2 - x1));
    return y0 * L0 + y1 * L1 + y2 * L2;
}

/* -------------------------------------------------------------------- */
/* Thomas (tridiagonal) solver: lower[k]*x[k-1] + diag[k]*x[k] +
 * upper[k]*x[k+1] = rhs[k]; lower[0] and upper[m-1] are unused. Solves in
 * place (overwrites diag/rhs), result in x[0..m-1]. */
static void thomas_solve(double *lower, double *diag, double *upper,
                          double *rhs, double *x, size_t m)
{
    for (size_t i = 1; i < m; i++) {
        double w = lower[i] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    x[m - 1] = rhs[m - 1] / diag[m - 1];
    for (size_t i = m - 1; i-- > 0;)
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / diag[i];
}

/* Builds the per-segment (a,b,c,d) coefficients from the knot second
 * derivatives M[0..n-1] (the standard cubic-spline coefficient formulas;
 * see e.g. Burden & Faires, "Numerical Analysis", S3.5). */
static void coeffs_from_M(const double *x, const double *y, const double *M,
                           size_t n, CPRCubicSpline *out)
{
    out->x = malloc(n * sizeof(double));
    out->a = malloc((n - 1) * sizeof(double));
    out->b = malloc((n - 1) * sizeof(double));
    out->c = malloc((n - 1) * sizeof(double));
    out->d = malloc((n - 1) * sizeof(double));
    out->n = n;
    memcpy(out->x, x, n * sizeof(double));
    for (size_t i = 0; i + 1 < n; i++) {
        double h = x[i + 1] - x[i];
        out->a[i] = y[i];
        out->c[i] = M[i] / 2.0;
        out->d[i] = (M[i + 1] - M[i]) / (6.0 * h);
        out->b[i] = (y[i + 1] - y[i]) / h - h * (2.0 * M[i] + M[i + 1]) / 6.0;
    }
}

int cpr_cubic_spline_fit_natural(const double *x, const double *y, size_t n,
                                   CPRCubicSpline *out, char **errmsg)
{
    if (n < 3) {
        *errmsg = strdup("cpr_cubic_spline_fit_natural: need at least 3 knots");
        return 1;
    }
    double *h = malloc((n - 1) * sizeof(double));
    for (size_t i = 0; i + 1 < n; i++) h[i] = x[i + 1] - x[i];

    size_t m = n - 2; /* unknowns M_1..M_{n-2} */
    double *lower = calloc(m, sizeof(double));
    double *diag = calloc(m, sizeof(double));
    double *upper = calloc(m, sizeof(double));
    double *rhs = calloc(m, sizeof(double));
    double *Msub = malloc(m * sizeof(double));

    for (size_t k = 0; k < m; k++) {
        size_t i = k + 1;
        double h0 = h[i - 1], h1 = h[i];
        if (k > 0) lower[k] = h0;
        diag[k] = 2.0 * (h0 + h1);
        if (k + 1 < m) upper[k] = h1;
        rhs[k] = 6.0 * ((y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0);
    }
    thomas_solve(lower, diag, upper, rhs, Msub, m);

    double *M = calloc(n, sizeof(double)); /* M[0] = M[n-1] = 0 */
    for (size_t k = 0; k < m; k++) M[k + 1] = Msub[k];

    coeffs_from_M(x, y, M, n, out);

    free(h); free(lower); free(diag); free(upper); free(rhs); free(Msub); free(M);
    return 0;
}

int cpr_cubic_spline_fit_notaknot(const double *x, const double *y, size_t n,
                                    CPRCubicSpline *out, char **errmsg)
{
    if (n < 4) {
        *errmsg = strdup("cpr_cubic_spline_fit_notaknot: need at least 4 knots");
        return 1;
    }
    double *h = malloc((n - 1) * sizeof(double));
    for (size_t i = 0; i + 1 < n; i++) h[i] = x[i + 1] - x[i];

    size_t m = n - 2; /* unknowns M_1..M_{n-2} */
    double *lower = calloc(m, sizeof(double));
    double *diag = calloc(m, sizeof(double));
    double *upper = calloc(m, sizeof(double));
    double *rhs = calloc(m, sizeof(double));
    double *Msub = malloc(m * sizeof(double));

    for (size_t k = 0; k < m; k++) {
        size_t i = k + 1;
        double h0 = h[i - 1], h1 = h[i];
        rhs[k] = 6.0 * ((y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0);
        if (k == 0) {
            /* M_0 eliminated via the left not-a-knot relation; see
             * spline.h / the derivation in this file's accompanying notes. */
            diag[k] = h0 * (h0 + h1) / h1 + 2.0 * (h0 + h1);
            upper[k] = (m > 1) ? h1 - h0 * h0 / h1 : 0.0;
        } else if (k == m - 1) {
            lower[k] = h0 - h1 * h1 / h0;
            diag[k] = h1 * (h0 + h1) / h0 + 2.0 * (h0 + h1);
        } else {
            lower[k] = h0;
            diag[k] = 2.0 * (h0 + h1);
            upper[k] = h1;
        }
    }
    thomas_solve(lower, diag, upper, rhs, Msub, m);

    double *M = malloc(n * sizeof(double));
    for (size_t k = 0; k < m; k++) M[k + 1] = Msub[k];
    /* Back-substitute the eliminated boundary second derivatives. */
    M[0] = ((h[0] + h[1]) * M[1] - h[0] * M[2]) / h[1];
    M[n - 1] = ((h[n - 3] + h[n - 2]) * M[n - 2] - h[n - 2] * M[n - 3]) / h[n - 3];

    coeffs_from_M(x, y, M, n, out);

    free(h); free(lower); free(diag); free(upper); free(rhs); free(Msub); free(M);
    return 0;
}

double cpr_cubic_spline_eval(const CPRCubicSpline *s, double xq)
{
    size_t i = find_segment(s->x, s->n, xq);
    double dx = xq - s->x[i];
    return s->a[i] + dx * (s->b[i] + dx * (s->c[i] + dx * s->d[i]));
}

void cpr_cubic_spline_free(CPRCubicSpline *s)
{
    free(s->x); free(s->a); free(s->b); free(s->c); free(s->d);
    s->x = s->a = s->b = s->c = s->d = NULL;
    s->n = 0;
}

/* -------------------------------------------------------------------- */

int cpr_resample_rate_table(const double *T9_src, const double *rate_src, size_t n_src,
                              const double *T9_dst, double *rate_dst, size_t n_dst,
                              char **errmsg)
{
    double *lx_src = malloc(n_src * sizeof(double));
    double *lx_dst = malloc(n_dst * sizeof(double));
    for (size_t i = 0; i < n_src; i++) lx_src[i] = log10(T9_src[i]);
    for (size_t i = 0; i < n_dst; i++) lx_dst[i] = log10(T9_dst[i]);

    int all_positive = 1;
    for (size_t i = 0; i < n_src; i++)
        if (!(rate_src[i] > 0.0)) { all_positive = 0; break; }

    if (all_positive) {
        double *log_rate = malloc(n_src * sizeof(double));
        for (size_t i = 0; i < n_src; i++) log_rate[i] = log10(rate_src[i]);

        CPRCubicSpline sp;
        if (cpr_cubic_spline_fit_notaknot(lx_src, log_rate, n_src, &sp, errmsg)) {
            free(log_rate); free(lx_src); free(lx_dst);
            return 1;
        }
        for (size_t i = 0; i < n_dst; i++)
            rate_dst[i] = pow(10.0, cpr_cubic_spline_eval(&sp, lx_dst[i]));
        cpr_cubic_spline_free(&sp);
        free(log_rate);
    } else {
        /* Linear interpolation of rate vs log10(T9), matching Python's
         * fallback for non-positive values (e.g. an error column with
         * zeros) to avoid taking log of zero. */
        for (size_t i = 0; i < n_dst; i++)
            rate_dst[i] = cpr_interp_linear(lx_src, rate_src, n_src, lx_dst[i],
                                              CPR_EXTRAP_LINEAR);
    }

    free(lx_src); free(lx_dst);
    return 0;
}
