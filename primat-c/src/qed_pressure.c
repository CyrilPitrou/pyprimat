/* qed_pressure.c -- see qed_pressure.h. */
#include "qed_pressure.h"
#include "quad.h"
#include "spline.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Mirrors qed_pressure.py's module-level constants exactly (see that
 * file's comments for the rationale of each). */
#define X_NONREL_CUTOFF 50.0  /* x = me/T above which dP is set to 0 */
#define P_UPPER         500.0 /* upper momentum-integration limit */
#define QUAD_TOL        1e-13 /* matches scipy.quad's epsabs=epsrel=1e-13 */
#define QUAD_MAX_DEPTH  30

static double i01_integrand(double p, void *ctx)
{
    double x = *(double *)ctx;
    double E = sqrt(p * p + x * x);
    /* At p=x=0 both numerator and denominator vanish (0/0); the true
     * limit is 0 (numerator ~ p^2, denominator ~ p, so the ratio ~ p -> 0).
     * scipy.quad never samples this exact point (Gauss-Kronrod nodes
     * exclude the interval endpoints), so qed_pressure.py never hits it;
     * cpr_quad_adaptive's Simpson rule does sample the endpoint, so this
     * guard is needed for the x=0 case to avoid a NaN. */
    if (E == 0.0) return 0.0;
    return p * p / (E * (exp(E) + 1.0));
}

static double i2m1_integrand(double p, void *ctx)
{
    double x = *(double *)ctx;
    double E = sqrt(p * p + x * x);
    return E / (exp(E) + 1.0);
}

/* Fixed breakpoints, geometrically widening from 0 up to P_UPPER. Both
 * integrands here are unimodal Boltzmann-like bumps p^a exp(-sqrt(p^2+x^2))
 * whose peak location grows with x (roughly sqrt(2x) for x >~ few, by
 * maximising p^2 exp(-sqrt(p^2+x^2))); for the largest x this module ever
 * evaluates (the X_NONREL_CUTOFF=50 cutoff above), the peak sits around
 * p ~ 10. A single cpr_quad_adaptive call across [0, P_UPPER] with P_UPPER
 * = 500 fails here: its first-level Simpson sample sits at {0, 250, 500},
 * entirely past the peak, where the integrand has already underflowed to
 * 0 -- coarse and refined estimates then agree (both ~0) and the
 * recursion accepts a result of 0, silently missing the entire peak
 * (verified: this reproduces exactly as I01(0) -> 0 instead of pi^2/12).
 * Splitting into pre-defined sub-intervals densest near 0 guarantees every
 * sub-interval's own first-level sample lands inside or adjacent to the
 * region where the integrand actually varies, for any x in [0, 50]. */
static const double BREAKPOINTS[] = {
    0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, P_UPPER
};
#define N_BREAKPOINTS (sizeof(BREAKPOINTS) / sizeof(BREAKPOINTS[0]))

static double integrate_segmented(CPRQuadFunc f, void *ctx)
{
    double total = 0.0;
    double seg_tol = QUAD_TOL / (double)(N_BREAKPOINTS - 1);
    for (size_t i = 0; i + 1 < N_BREAKPOINTS; i++)
        total += cpr_quad_adaptive(f, ctx, BREAKPOINTS[i], BREAKPOINTS[i + 1],
                                    seg_tol, QUAD_MAX_DEPTH, NULL);
    return total;
}

double cpr_qed_I01(double x)
{
    if (x > X_NONREL_CUTOFF) return 0.0;
    return integrate_segmented(i01_integrand, &x);
}

double cpr_qed_I2m1(double x)
{
    if (x > X_NONREL_CUTOFF) return 0.0;
    return integrate_segmented(i2m1_integrand, &x);
}

double cpr_qed_dPa(double T, double alpha, double me)
{
    double x = me / T;
    double I01 = cpr_qed_I01(x);
    return alpha / M_PI * pow(T, 4.0) * (-2.0 / 3.0 * I01 - 2.0 / (M_PI * M_PI) * I01 * I01);
}

double cpr_qed_dPe3(double T, double alpha, double me)
{
    double x = me / T;
    double I01  = cpr_qed_I01(x);
    double I2m1 = cpr_qed_I2m1(x);
    double combo = (I01 + I2m1) / (M_PI * M_PI);
    if (combo <= 0.0) return 0.0;
    return pow(alpha, 1.5) * (4.0 / 3.0) * sqrt(2.0 * M_PI) * pow(T, 4.0) * pow(combo, 1.5);
}

int cpr_qed_compute_tables(double T_min, double T_max, size_t n_pts,
                            double alpha, double me,
                            CPRQEDTables *out, char **errmsg)
{
    if (n_pts < 4) {
        *errmsg = strdup("cpr_qed_compute_tables: n_pts must be >= 4 (not-a-knot spline minimum)");
        return 1;
    }

    memset(out, 0, sizeof(*out));
    out->n = n_pts;
    out->T              = malloc(n_pts * sizeof(double));
    out->dP_e2           = malloc(n_pts * sizeof(double));
    out->dP_e3           = malloc(n_pts * sizeof(double));
    out->d_dP_e2_dT      = malloc(n_pts * sizeof(double));
    out->d_dP_e3_dT      = malloc(n_pts * sizeof(double));
    out->d2_dP_e2_dT2    = malloc(n_pts * sizeof(double));
    out->d2_dP_e3_dT2    = malloc(n_pts * sizeof(double));

    /* Log-spaced grid (np.logspace(log10(T_min), log10(T_max), n_pts)). */
    double log_min = log10(T_min), log_max = log10(T_max);
    for (size_t i = 0; i < n_pts; i++) {
        double frac = (n_pts == 1) ? 0.0 : (double)i / (double)(n_pts - 1);
        out->T[i] = pow(10.0, log_min + frac * (log_max - log_min));
        out->dP_e2[i] = cpr_qed_dPa(out->T[i], alpha, me);
        out->dP_e3[i] = cpr_qed_dPe3(out->T[i], alpha, me);
    }

    /* Differentiate numerically via not-a-knot cubic splines (matches
     * scipy.interpolate.CubicSpline's default boundary condition, used by
     * compute_qed_pressure_tables). */
    CPRCubicSpline spl_e2, spl_e3;
    if (cpr_cubic_spline_fit_notaknot(out->T, out->dP_e2, n_pts, &spl_e2, errmsg)) {
        cpr_qed_tables_free(out);
        return 1;
    }
    if (cpr_cubic_spline_fit_notaknot(out->T, out->dP_e3, n_pts, &spl_e3, errmsg)) {
        cpr_cubic_spline_free(&spl_e2);
        cpr_qed_tables_free(out);
        return 1;
    }
    for (size_t i = 0; i < n_pts; i++) {
        /* CPRCubicSpline stores per-segment polynomial coefficients
         * directly (b = 1st deriv, 2c = 2nd deriv, at the left knot of
         * each segment); evaluate analytically at each knot rather than
         * via cpr_cubic_spline_eval (which is the y(x) evaluator, not a
         * derivative one) by locating the segment whose left knot is T[i]
         * (every grid point is itself a knot, so dx=0 there and the
         * derivatives are exactly b[i] and 2*c[i] -- except at the last
         * knot, which belongs to segment n-2). */
        size_t seg = (i < n_pts - 1) ? i : n_pts - 2;
        double dx = out->T[i] - spl_e2.x[seg];
        out->d_dP_e2_dT[i]   = spl_e2.b[seg] + 2.0 * spl_e2.c[seg] * dx + 3.0 * spl_e2.d[seg] * dx * dx;
        out->d2_dP_e2_dT2[i] = 2.0 * spl_e2.c[seg] + 6.0 * spl_e2.d[seg] * dx;
        dx = out->T[i] - spl_e3.x[seg];
        out->d_dP_e3_dT[i]   = spl_e3.b[seg] + 2.0 * spl_e3.c[seg] * dx + 3.0 * spl_e3.d[seg] * dx * dx;
        out->d2_dP_e3_dT2[i] = 2.0 * spl_e3.c[seg] + 6.0 * spl_e3.d[seg] * dx;
    }
    cpr_cubic_spline_free(&spl_e2);
    cpr_cubic_spline_free(&spl_e3);
    return 0;
}

void cpr_qed_tables_free(CPRQEDTables *t)
{
    free(t->T); free(t->dP_e2); free(t->dP_e3);
    free(t->d_dP_e2_dT); free(t->d_dP_e3_dT);
    free(t->d2_dP_e2_dT2); free(t->d2_dP_e3_dT2);
    memset(t, 0, sizeof(*t));
}

/* Writes two 4-column files, one per order in e, matching
 * numpy.savetxt(fmt="%.6E") output from Python's save_qed_tables. Each
 * file's columns are: T [MeV]  dP [MeV^4]  d(dP)/dT [MeV^3]  d2(dP)/dT2
 * [MeV^2]. All '#'-prefixed header lines are skipped by cpr_table_read. */
static int write_one_qed_file(const char *path, const char *src_tag,
                               const char *phys_tag, const char *ref_tag,
                               const char *col_hdr,
                               const double *T, const double *dP,
                               const double *ddP, const double *d2dP,
                               size_t n, char **errmsg)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        char buf[512];
        snprintf(buf, sizeof(buf), "cpr_qed_save_tables: cannot open %s for writing", path);
        *errmsg = strdup(buf);
        return 1;
    }
    fprintf(fp, "# Source: %s\n", src_tag);
    fprintf(fp, "# %s\n", phys_tag);
    fprintf(fp, "# Reference: %s\n", ref_tag);
    fprintf(fp, "# %s\n", col_hdr);
    for (size_t i = 0; i < n; i++)
        fprintf(fp, "%.6E %.6E %.6E %.6E\n", T[i], dP[i], ddP[i], d2dP[i]);
    fclose(fp);
    return 0;
}

int cpr_qed_save_tables(const CPRQEDTables *t, const char *plasma_dir, char **errmsg)
{
    char path_e2[1024], path_e3[1024];
    snprintf(path_e2, sizeof(path_e2), "%s/QED_pressure_correction_e2.txt", plasma_dir);
    snprintf(path_e3, sizeof(path_e3), "%s/QED_pressure_correction_e3.txt", plasma_dir);

    if (write_one_qed_file(path_e2,
            "CPRIMAT qed_pressure.c -- QED plasma-pressure correction delta_P_a(T)",
            "delta_P_a: O(e^2), one-loop (Frenkel-Galitskii-Migdal)",
            "Pitrou et al., Phys. Rep. (2018), eq. 47; PRIMAT-Main.m: dPa",
            "T [MeV]       dP_a [MeV^4]      d(dP_a)/dT [MeV^3]  d2(dP_a)/dT2 [MeV^2]",
            t->T, t->dP_e2, t->d_dP_e2_dT, t->d2_dP_e2_dT2, t->n, errmsg))
        return 1;

    if (write_one_qed_file(path_e3,
            "CPRIMAT qed_pressure.c -- QED plasma-pressure correction delta_P_e3(T)",
            "delta_P_e3: O(e^3), ring/plasmon (Blaizot-Zinn-Justin)",
            "Pitrou et al., Phys. Rep. (2018), eq. 47; PRIMAT-Main.m: dPe3",
            "T [MeV]       dP_e3 [MeV^4]     d(dP_e3)/dT [MeV^3]  d2(dP_e3)/dT2 [MeV^2]",
            t->T, t->dP_e3, t->d_dP_e3_dT, t->d2_dP_e3_dT2, t->n, errmsg))
        return 1;

    return 0;
}
