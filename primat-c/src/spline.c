/* spline.c -- see cprimat/spline.h. */
#include "spline.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Binary search for the segment i such that x[i] <= xq <= x[i+1] (clamped
 * to [0, n-2] outside the table -- the caller then evaluates with an
 * unclamped dx to extrapolate using that boundary segment). Exposed
 * (non-static, declared in spline.h) so callers with a "cold" query (no
 * prior nearby lookup, e.g. the very first call of a solve) and the fuzz
 * test in test_spline.c can use it as the ground truth that
 * cpr_find_segment_monotone below must always reproduce exactly. */
size_t cpr_find_segment(const double *x, size_t n, double xq)
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

/* Maximum number of grid cells the forward/backward scan below will walk
 * away from *hint before giving up and falling back to a full binary
 * search. Small on purpose: the whole point of the hinted lookup is that
 * consecutive queries (e.g. successive BDF evaluations as T9 decreases
 * monotonically through a rate table during nuclear-network integration)
 * land within a cell or two of the previous one, so a handful of steps
 * covers the overwhelmingly common case while keeping the worst-case cost
 * of a wrong/stale hint bounded (a few wasted comparisons, then the same
 * O(log n) binary search find_segment would have done anyway). */
#define CPR_HINT_MAX_STEPS 8

/* Hinted variant of cpr_find_segment: identical return value for every
 * input (x, n, xq), including all extrapolation/boundary edge cases, but
 * normally O(1) instead of O(log n) when xq is close to the segment found
 * by the previous call, since BDF integration queries rate tables at a
 * slowly-and-monotonically-varying T9 millions of times per solve.
 *
 * `*hint` is read as the starting guess and overwritten with the segment
 * actually returned, so the caller just needs one size_t of persistent
 * per-table state (initialised to anything -- 0 works fine, see below) and
 * to keep reusing the same hint across consecutive queries of the same
 * table. `hint == NULL` always falls back to the plain binary search
 * (e.g. for one-off, non-repeated lookups with no state to cache into).
 *
 * Correctness invariant (relied on by the fuzz test in test_spline.c):
 * the two extrapolation cases (xq <= x[0], xq >= x[n-1]) are handled
 * exactly as in cpr_find_segment regardless of the hint. For interior xq,
 * the forward-then-backward walk from the hint can only return early once
 * it has re-established find_segment's own bracketing invariant
 * (x[i] <= xq < x[i+1]); if it can't do that within CPR_HINT_MAX_STEPS
 * cells (a wildly wrong or uninitialised hint), it gives up and calls the
 * full binary search instead -- so a bad hint costs a little speed, never
 * correctness. */
size_t cpr_find_segment_monotone(const double *x, size_t n, double xq, size_t *hint)
{
    /* Boundary/extrapolation cases: identical to cpr_find_segment, checked
     * up front so the scan below only ever has to deal with the interior,
     * strictly-bracketing case. */
    if (xq <= x[0]) { if (hint) *hint = 0; return 0; }
    if (xq >= x[n - 1]) { if (hint) *hint = n - 2; return n - 2; }

    if (!hint) return cpr_find_segment(x, n, xq);

    size_t i = *hint;
    /* Stale/uninitialised/out-of-range hint (e.g. the grid was reloaded
     * with a different n since the hint was last set) -- the clamp below
     * would silently misbehave, so just fall back instead of guessing. */
    if (i > n - 2) {
        i = cpr_find_segment(x, n, xq);
        *hint = i;
        return i;
    }

    size_t steps = 0;
    while (i < n - 2 && x[i + 1] <= xq) {
        if (steps++ >= CPR_HINT_MAX_STEPS) { i = cpr_find_segment(x, n, xq); *hint = i; return i; }
        i++;
    }
    steps = 0;
    while (i > 0 && x[i] > xq) {
        if (steps++ >= CPR_HINT_MAX_STEPS) { i = cpr_find_segment(x, n, xq); *hint = i; return i; }
        i--;
    }

    *hint = i;
    return i;
}

double cpr_interp_linear(const double *x, const double *y, size_t n, double xq,
                          CPRExtrapMode mode)
{
    if (mode == CPR_EXTRAP_CONSTANT) {
        if (xq <= x[0]) return y[0];
        if (xq >= x[n - 1]) return y[n - 1];
    }
    size_t i = cpr_find_segment(x, n, xq);
    double t = (xq - x[i]) / (x[i + 1] - x[i]);
    return y[i] + t * (y[i + 1] - y[i]);
}

double cpr_interp_quadratic_local(const double *x, const double *y, size_t n, double xq)
{
    size_t i = cpr_find_segment(x, n, xq);          /* bracketing segment [i, i+1] */
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
    size_t i = cpr_find_segment(s->x, s->n, xq);
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
