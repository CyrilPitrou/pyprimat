/* spline.h -- interpolation.
 *
 * Covers `np.interp`/`interp1d(kind="linear")` (cpr_interp_linear) and
 * natural/not-a-knot cubic splines (CPRCubicSpline, used by the QED-
 * pressure tables, the electron-thermo cache, and
 * network_data._resample_rate_table's log-log resampling -- see
 * cpr_resample_rate_table).
 */
#ifndef CPRIMAT_SPLINE_H
#define CPRIMAT_SPLINE_H

#include <stddef.h>

typedef enum { CPR_EXTRAP_CONSTANT, CPR_EXTRAP_LINEAR } CPRExtrapMode;

/* Binary search for the segment i such that x[i] <= xq <= x[i+1] (clamped
 * to [0, n-2] outside the table). Exposed as the ground-truth reference
 * cpr_find_segment_monotone below must always agree with. */
size_t cpr_find_segment(const double *x, size_t n, double xq);

/* Hinted variant of cpr_find_segment for repeated lookups in the same
 * table at a slowly, monotonically varying xq (e.g. rate-table lookups
 * during BDF integration, where T9 decreases smoothly across millions of
 * evaluations): normally O(1) instead of O(log n), but ALWAYS returns
 * exactly what a cold cpr_find_segment for every (x, n, xq) -- correctness
 * never depends on the hint being right, only on its absence costing more.
 * `*hint` is read as a starting guess and overwritten with the segment
 * actually returned; the caller owns one size_t of persistent state (any
 * initial value is safe, including 0 or garbage) and must not share it
 * across independent, unrelated query sequences (e.g. two rate tables, or
 * two threads) -- each needs its own hint. `hint == NULL` always falls
 * back to plain cpr_find_segment. */
size_t cpr_find_segment_monotone(const double *x, size_t n, double xq, size_t *hint);

/* Evaluates the piecewise-linear interpolant through (x[i], y[i]) at xq;
 * `x` must be strictly increasing, length n >= 2. Outside [x[0], x[n-1]],
 * either holds the boundary value constant (CPR_EXTRAP_CONSTANT) or
 * extends the boundary segment's slope (CPR_EXTRAP_LINEAR, == `np.interp`'s
 * implicit linear extrapolation when fed sorted x). */
double cpr_interp_linear(const double *x, const double *y, size_t n, double xq,
                          CPRExtrapMode mode);

/* Local quadratic interpolant: the Lagrange quadratic through the 3
 * consecutive data points {x[k],x[k+1],x[k+2]} whose middle segment
 * brackets (or, outside the table, is nearest to) xq -- a stand-in for
 * scipy.interpolate.interp1d(kind='quadratic') (a global FITPACK B-spline)
 * used by weak_rates.c's n<->p rate-table interpolants. On the smooth,
 * densely-sampled grids those tables use (one point per
 * 1/sampling_nTOp_per_decade of a T-decade), a local quadratic through the
 * nearest 3 points agrees with the global B-spline to <~1e-6 relative in
 * the interior (verified against live Python output in test_weak_rates.c);
 * it is not a bit-exact replication of FITPACK's knot placement, which
 * would require a full B-spline solver for a difference unobservable at
 * this grid density. Requires n >= 3. Outside [x[0], x[n-1]], extrapolates
 * with the boundary window's quadratic (matches `fill_value="extrapolate"`). */
double cpr_interp_quadratic_local(const double *x, const double *y, size_t n, double xq);

/* A fitted piecewise-cubic interpolant: y(x) = a[i] + b[i]*dx + c[i]*dx^2 +
 * d[i]*dx^3 on segment i = [x[i], x[i+1]], dx = x - x[i]. */
typedef struct {
    double *x, *a, *b, *c, *d;
    size_t n; /* number of knots; n-1 segments */
} CPRCubicSpline;

/* Natural boundary (second derivative = 0 at both ends), the standard
 * tridiagonal Thomas-algorithm solve. Requires n >= 3. */
int cpr_cubic_spline_fit_natural(const double *x, const double *y, size_t n,
                                   CPRCubicSpline *out, char **errmsg);

/* "Not-a-knot" boundary (third derivative continuous across the second and
 * second-to-last knots, i.e. the first two and last two segments are each a
 * single cubic) -- mirrors scipy's `interp1d(kind="cubic")` default used by
 * _resample_rate_table. Requires n >= 4. */
int cpr_cubic_spline_fit_notaknot(const double *x, const double *y, size_t n,
                                    CPRCubicSpline *out, char **errmsg);

/* Evaluates the spline at xq. Outside [x[0], x[n-1]], extrapolates by
 * extending the boundary segment's cubic polynomial (matching
 * scipy's `fill_value="extrapolate"`), i.e. clamps the *segment* but not
 * `dx`. */
double cpr_cubic_spline_eval(const CPRCubicSpline *s, double xq);

void cpr_cubic_spline_free(CPRCubicSpline *s);

/* Port of network_data._resample_rate_table: resamples a rate table from
 * its source T9 grid onto the master T9 grid (T9_dst), using not-a-knot
 * cubic interpolation in log10(T9)-log10(rate) space when every rate_src
 * value is positive, falling back to linear interpolation of rate vs
 * log10(T9) when any value is non-positive (e.g. an error column that may
 * contain zeros) -- exactly Python's two-branch logic. Writes n_dst values
 * into `rate_dst` (caller-allocated). Returns 0 on success, nonzero with
 * *errmsg set (caller frees) if n_src < 4 (not-a-knot's minimum) in the
 * positive branch. */
int cpr_resample_rate_table(const double *T9_src, const double *rate_src, size_t n_src,
                              const double *T9_dst, double *rate_dst, size_t n_dst,
                              char **errmsg);

#endif /* CPRIMAT_SPLINE_H */
