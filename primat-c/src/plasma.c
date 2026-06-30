/* plasma.c -- see cprimat/plasma.h. */
#include "cprimat/plasma.h"
#include "cprimat/constants.h"
#include "cprimat/qed_pressure.h"
#include "cprimat/quad.h"
#include "cprimat/table_io.h"
#include "cprimat/cache.h"
#include "cprimat/log.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Below Tg = me / _ELEC_THERMO_LOWT_RATIO the e+- number density is
 * Boltzmann-suppressed by exp(-me/Tg) < exp(-30) ~ 1e-13 relative to
 * photons, so all four e+- quantities are exactly 0 (Phys. Rep. App. A.2
 * non-relativistic limit) -- avoids integrating a negligible, potentially
 * slow tail. Matches plasma.py's _ELEC_THERMO_LOWT_RATIO. */
#define ELEC_THERMO_LOWT_RATIO 30.0

/* High-T limit of spl/Tg^3 (photon + e+-): Phys. Rep. Eq. 25d/26d give
 * sbar_g = 4 pi^2/45, sbar_e+- = 7/8 sbar_g each, so
 * sbar_pl = sbar_g + 2*(7/8)*sbar_g = (11/4) * 4pi^2/45 = 11 pi^2/45. */
#define SIGMA_INF (11.0 * M_PI * M_PI / 45.0)

#define ELECTRON_THERMO_FORMAT_VERSION 1

double cpr_rho_g(double Tg)    { return 2.0 * (M_PI * M_PI / 30.0) * pow(Tg, 4.0); }
double cpr_drho_g_dT(double Tg) { return 4.0 * cpr_rho_g(Tg) / Tg; }
double cpr_rho_nu(double Tnu)  { return 2.0 * (7.0 / 8.0) * (M_PI * M_PI / 30.0) * pow(Tnu, 4.0); }
double cpr_drho_nu_dT(double Tnu) { return 4.0 * cpr_rho_nu(Tnu) / Tnu; }

/* Per-flavour (nu+nubar) energy-density excess from a genuine reduced chemical
 * potential c = mu/Tnu: rho(c) - rho(0) = Tnu^4 (c^2/4 + c^4/(8 pi^2)). Even in
 * c (the antineutrino carries -c). See cpr_rho_nu_chempot_excess in plasma.h. */
double cpr_rho_nu_chempot_excess(double Tnu, double c)
{
    return pow(Tnu, 4.0) * (c * c / 4.0 + pow(c, 4.0) / (8.0 * M_PI * M_PI));
}

/* ---------------------------------------------------------------------
 * CPRInterp1D: linear-from-file or not-a-knot-spline-from-computation.
 * ------------------------------------------------------------------- */

double cpr_interp1d_eval(const CPRInterp1D *itp, double xq)
{
    if (itp->is_spline) return cpr_cubic_spline_eval(&itp->spl, xq);
    return cpr_interp_linear(itp->x, itp->y, itp->n, xq, CPR_EXTRAP_LINEAR);
}

void cpr_interp1d_free(CPRInterp1D *itp)
{
    if (itp->is_spline) cpr_cubic_spline_free(&itp->spl);
    else { free(itp->x); free(itp->y); }
    memset(itp, 0, sizeof(*itp));
}

static int file_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

/* ---------------------------------------------------------------------
 * QED interaction-pressure correction tables (plasma.Plasma._load_tables).
 * ------------------------------------------------------------------- */

static int load_qed_tables(CPRPlasma *pl, const CPRConfig *cfg, char **errmsg)
{
    if (!cfg->QED_corrections) {
        pl->qed_active = 0;
        memset(&pl->P_QED, 0, sizeof(pl->P_QED));
        memset(&pl->dP_QED, 0, sizeof(pl->dP_QED));
        memset(&pl->d2P_QED, 0, sizeof(pl->d2P_QED));
        return 0;
    }
    pl->qed_active = 1;

    char plasma_dir[4096], new_file[4200];
    /* Legacy 3-file names for backward compat with old cached copies. */
    char p_file_leg[4200], dp_file_leg[4200], d2p_file_leg[4200];
    snprintf(plasma_dir,   sizeof(plasma_dir),   "%s/plasma", cfg->data_dir);
    snprintf(new_file,     sizeof(new_file),      "%s/QED_tables.txt", plasma_dir);
    snprintf(p_file_leg,   sizeof(p_file_leg),    "%s/QED_P_int.txt", plasma_dir);
    snprintf(dp_file_leg,  sizeof(dp_file_leg),   "%s/QED_dP_intdT.txt", plasma_dir);
    snprintf(d2p_file_leg, sizeof(d2p_file_leg),  "%s/QED_d2P_intdT2.txt", plasma_dir);

    int new_present    = file_exists(new_file);
    int legacy_present = file_exists(p_file_leg) && file_exists(dp_file_leg)
                         && file_exists(d2p_file_leg);
    int files_present  = new_present || legacy_present;
    int recompute = cfg->recompute_qed_corrections;

    if (recompute || !files_present) {
        /* Analytic path: compute on a fresh 500-point grid (~0.3 s) and
         * build not-a-knot cubic-spline interpolants directly from the
         * computed arrays -- smoother than the linear interpolation used
         * when loading from a file (mirrors Python's choice exactly). */
        cpr_log(cfg, "init", "Computing QED plasma-pressure tables (%s)...",
                 recompute ? "recompute requested" : "files not found");
        CPRQEDTables t;
        if (cpr_qed_compute_tables(1e-3, 1e2, 500, g_const.alphaem, g_const.me, &t, errmsg))
            return 1;
        if (recompute) {
            if (cpr_qed_save_tables(&t, plasma_dir, errmsg)) { cpr_qed_tables_free(&t); return 1; }
        }
        double *sumP = malloc(t.n * sizeof(double));
        double *sumdP = malloc(t.n * sizeof(double));
        double *sumd2P = malloc(t.n * sizeof(double));
        for (size_t i = 0; i < t.n; i++) {
            sumP[i]   = t.dP_e2[i] + t.dP_e3[i];
            sumdP[i]  = t.d_dP_e2_dT[i] + t.d_dP_e3_dT[i];
            sumd2P[i] = t.d2_dP_e2_dT2[i] + t.d2_dP_e3_dT2[i];
        }
        pl->P_QED.is_spline = pl->dP_QED.is_spline = pl->d2P_QED.is_spline = 1;
        int rc = cpr_cubic_spline_fit_notaknot(t.T, sumP, t.n, &pl->P_QED.spl, errmsg)
              || cpr_cubic_spline_fit_notaknot(t.T, sumdP, t.n, &pl->dP_QED.spl, errmsg)
              || cpr_cubic_spline_fit_notaknot(t.T, sumd2P, t.n, &pl->d2P_QED.spl, errmsg);
        free(sumP); free(sumdP); free(sumd2P);
        cpr_qed_tables_free(&t);
        return rc;
    }

    /* File mode: load and sum the e^2/e^3 columns, linear interpolation
     * with linear extrapolation outside the table (matches
     * interp1d(kind='linear', fill_value="extrapolate")). */
    CPRTable tab;
    if (new_present) {
        /* New 7-column format: T, dP_a, dP_e3, d(dP_a)/dT, d(dP_e3)/dT,
         * d2(dP_a)/dT2, d2(dP_e3)/dT2. */
        if (cpr_table_read(new_file, 7, &tab, errmsg)) return 1;
        /* col indices: 0=T, 1=dP_a, 2=dP_e3, 3=ddP_a/dT, 4=ddP_e3/dT,
         *              5=d2dP_a/dT2, 6=d2dP_e3/dT2 */
        int col_pairs[3][2] = { {1,2}, {3,4}, {5,6} };
        CPRInterp1D *targets[3] = { &pl->P_QED, &pl->dP_QED, &pl->d2P_QED };
        for (int k = 0; k < 3; k++) {
            targets[k]->is_spline = 0;
            targets[k]->n = tab.n_rows;
            targets[k]->x = malloc(tab.n_rows * sizeof(double));
            targets[k]->y = malloc(tab.n_rows * sizeof(double));
            for (size_t i = 0; i < tab.n_rows; i++) {
                targets[k]->x[i] = tab.cols[0][i];
                targets[k]->y[i] = tab.cols[col_pairs[k][0]][i]
                                  + tab.cols[col_pairs[k][1]][i];
            }
        }
        cpr_table_free(&tab);
    } else {
        /* Legacy 3-file format: backward compat with old cached copies. */
        const char *files[3]    = { p_file_leg, dp_file_leg, d2p_file_leg };
        CPRInterp1D *targets[3] = { &pl->P_QED, &pl->dP_QED, &pl->d2P_QED };
        for (int k = 0; k < 3; k++) {
            if (cpr_table_read(files[k], 3, &tab, errmsg)) return 1;
            targets[k]->is_spline = 0;
            targets[k]->n = tab.n_rows;
            targets[k]->x = malloc(tab.n_rows * sizeof(double));
            targets[k]->y = malloc(tab.n_rows * sizeof(double));
            for (size_t i = 0; i < tab.n_rows; i++) {
                targets[k]->x[i] = tab.cols[0][i];
                targets[k]->y[i] = tab.cols[1][i] + tab.cols[2][i];
            }
            cpr_table_free(&tab);
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------
 * e+- exact integrands and quadrature (plasma.Plasma._*_exact).
 *
 * Each integrand is evaluated over the dimensionless energy variable
 * E = eps/Tg, lower bound x = me/Tg, fixed upper bound 100 (well past
 * where exp(-E) makes any further contribution negligible at double
 * precision). As in qed_pressure.c, a single cpr_quad_adaptive call over
 * the full [x, 100] risks missing the E~O(1-5) peak when x is small and
 * the domain is wide (the coarse first-level Simpson sample would then
 * land entirely in the exponentially-suppressed tail) -- so integration
 * is split into breakpoints anchored at x and widening geometrically,
 * exactly the same fix as cpr_qed_I01/I2m1.
 * ------------------------------------------------------------------- */

typedef double (*ElecIntegrand)(double E, double x);

static double rho_e_intgd(double E, double x)     { 
    if (E <= x) return 0.0; 
    return E * E * sqrt(E * E - x * x) / (exp(E) + 1.0); 
}
static double drho_e_dT_intgd(double E, double x) { 
    if (E <= x) return 0.0; 
    return E * E * E * sqrt(E * E - x * x) / pow(cosh(E / 2.0), 2.0); 
}
static double p_e_intgd(double E, double x)       { 
    if (E <= x) return 0.0; 
    return pow(E * E - x * x, 1.5) / (exp(E) + 1.0); 
}
static double dp_e_dT_intgd(double E, double x)   { 
    if (E <= x) return 0.0; 
    return E * pow(E * E - x * x, 1.5) / pow(cosh(E / 2.0), 2.0); 
}

typedef struct { ElecIntegrand fn; double x; } ElecCtx;

static double elec_quad_wrapper(double E, void *ctx)
{
    ElecCtx *c = (ElecCtx *)ctx;
    return c->fn(E, c->x);
}

static const double ELEC_OFFSETS[] = { 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0 };
#define N_ELEC_OFFSETS (sizeof(ELEC_OFFSETS) / sizeof(ELEC_OFFSETS[0]))

static double elec_integrate(ElecIntegrand fn, double x, double upper)
{
    ElecCtx ctx = { fn, x };
    double total = 0.0;
    double prev = x;
    double seg_tol = 1e-12 / (double)N_ELEC_OFFSETS;
    for (size_t i = 1; i < N_ELEC_OFFSETS; i++) {
        double next = x + ELEC_OFFSETS[i];
        if (next >= upper) { next = upper; }
        if (next > prev)
            total += cpr_quad_adaptive(elec_quad_wrapper, &ctx, prev, next, seg_tol, 30, NULL);
        prev = next;
        if (prev >= upper) break;
    }
    if (prev < upper)
        total += cpr_quad_adaptive(elec_quad_wrapper, &ctx, prev, upper, seg_tol, 30, NULL);
    return total;
}

static double rho_e_exact(double Tg)
{
    double me = g_const.me;
    if (Tg < me / ELEC_THERMO_LOWT_RATIO) return 0.0;
    double r = elec_integrate(rho_e_intgd, me / Tg, 100.0);
    return 4.0 / (2.0 * M_PI * M_PI) * pow(Tg, 4.0) * r;
}

static double drho_e_dT_exact(double Tg)
{
    double me = g_const.me;
    if (Tg < me / ELEC_THERMO_LOWT_RATIO) return 0.0;
    double r = elec_integrate(drho_e_dT_intgd, me / Tg, 100.0);
    return 1.0 / (2.0 * M_PI * M_PI) * pow(Tg, 3.0) * r;
}

static double p_e_exact(double Tg)
{
    double me = g_const.me;
    if (Tg < me / ELEC_THERMO_LOWT_RATIO) return 0.0;
    double r = elec_integrate(p_e_intgd, me / Tg, 100.0);
    return 4.0 / (6.0 * M_PI * M_PI) * pow(Tg, 4.0) * r;
}

static double dp_e_dT_exact(double Tg)
{
    double me = g_const.me;
    if (Tg < me / ELEC_THERMO_LOWT_RATIO) return 0.0;
    double r = elec_integrate(dp_e_dT_intgd, me / Tg, 100.0);
    return 1.0 / (6.0 * M_PI * M_PI) * pow(Tg, 3.0) * r;
}

/* ---------------------------------------------------------------------
 * e+- pre-tabulation with a fingerprinted on-disk cache
 * (plasma.Plasma._build_electron_tables).
 * ------------------------------------------------------------------- */

static int build_electron_tables(CPRPlasma *pl, const CPRConfig *cfg, char **errmsg)
{
    char cache_path[4224];
    snprintf(cache_path, sizeof(cache_path), "%s/plasma/electron_thermo_cache.txt", cfg->data_dir);

    double Tmin = g_const.me / ELEC_THERMO_LOWT_RATIO;
    double Tmax = fmax(cfg->T_start_cosmo_MeV, 100.0) * 1.5;
    size_t npts = (size_t)cfg->n_electron_table;

    CPRFPField fields[3];
    fields[0] = (CPRFPField){ "format_version", { CPR_INT, { .i = ELECTRON_THERMO_FORMAT_VERSION } } };
    fields[1] = (CPRFPField){ "n_electron_table", { CPR_INT, { .i = cfg->n_electron_table } } };
    fields[2] = (CPRFPField){ "T_start_cosmo_MeV", { CPR_DOUBLE, { .d = cfg->T_start_cosmo_MeV } } };
    char *fp_hash = cpr_fingerprint_hash(fields, 3);

    if (!cfg->recompute_electron_thermo) {
        char *cached_hash = cpr_cache_read_fingerprint_hash(cache_path);
        if (cached_hash && strcmp(cached_hash, fp_hash) == 0) {
            free(cached_hash);
            CPRTable tab;
            if (cpr_table_read(cache_path, 5, &tab, errmsg) == 0) {
                int rc = cpr_cubic_spline_fit_notaknot(tab.cols[0], tab.cols[1], tab.n_rows, &pl->rho_e_tab, errmsg)
                      || cpr_cubic_spline_fit_notaknot(tab.cols[0], tab.cols[2], tab.n_rows, &pl->p_e_tab, errmsg)
                      || cpr_cubic_spline_fit_notaknot(tab.cols[0], tab.cols[3], tab.n_rows, &pl->drho_e_dT_tab, errmsg)
                      || cpr_cubic_spline_fit_notaknot(tab.cols[0], tab.cols[4], tab.n_rows, &pl->dp_e_dT_tab, errmsg);
                cpr_table_free(&tab);
                free(fp_hash);
                if (rc == 0)
                    cpr_log(cfg, "init", "Electron-thermo tables loaded from cache (%d points).",
                             cfg->n_electron_table);
                return rc;
            }
            /* Fall through to recompute if the cache file turned out to
             * be unreadable despite a matching fingerprint header
             * (matches Python's try/except warn-and-recompute path). */
        }
        free(cached_hash);
    }

    double *grid = malloc(npts * sizeof(double));
    double *rho_e_arr = malloc(npts * sizeof(double));
    double *p_e_arr = malloc(npts * sizeof(double));
    double *drho_e_dT_arr = malloc(npts * sizeof(double));
    double *dp_e_dT_arr = malloc(npts * sizeof(double));

    double log_min = log10(Tmin), log_max = log10(Tmax);
    for (size_t i = 0; i < npts; i++) {
        double frac = (npts == 1) ? 0.0 : (double)i / (double)(npts - 1);
        grid[i] = pow(10.0, log_min + frac * (log_max - log_min));
        rho_e_arr[i] = rho_e_exact(grid[i]);
        p_e_arr[i] = p_e_exact(grid[i]);
        drho_e_dT_arr[i] = drho_e_dT_exact(grid[i]);
        dp_e_dT_arr[i] = dp_e_dT_exact(grid[i]);
    }

    int rc = cpr_cubic_spline_fit_notaknot(grid, rho_e_arr, npts, &pl->rho_e_tab, errmsg)
          || cpr_cubic_spline_fit_notaknot(grid, p_e_arr, npts, &pl->p_e_tab, errmsg)
          || cpr_cubic_spline_fit_notaknot(grid, drho_e_dT_arr, npts, &pl->drho_e_dT_tab, errmsg)
          || cpr_cubic_spline_fit_notaknot(grid, dp_e_dT_arr, npts, &pl->dp_e_dT_tab, errmsg);

    if (rc == 0) {
        double *columns[5] = { grid, rho_e_arr, p_e_arr, drho_e_dT_arr, dp_e_dT_arr };
        /* A cache-write failure is non-fatal (matches Python's warn-and-
         * continue): the tables we just built in memory are still valid
         * for this run, only the on-disk cache for future runs is stale. */
        cpr_cache_write(cache_path, fields, 3, "grid rho_e p_e drho_e_dT dp_e_dT",
                         columns, 5, npts, NULL);
    }

    free(grid); free(rho_e_arr); free(p_e_arr); free(drho_e_dT_arr); free(dp_e_dT_arr);
    free(fp_hash);
    if (rc == 0)
        cpr_log(cfg, "init", "Electron-thermo tables built (%d points).", cfg->n_electron_table);
    return rc;
}

/* ---------------------------------------------------------------------
 * Public API.
 * ------------------------------------------------------------------- */

int cpr_plasma_init(CPRPlasma *pl, const CPRConfig *cfg, char **errmsg)
{
    memset(pl, 0, sizeof(*pl));
    pl->cfg = cfg;
    if (load_qed_tables(pl, cfg, errmsg)) return 1;
    if (build_electron_tables(pl, cfg, errmsg)) {
        cpr_interp1d_free(&pl->P_QED);
        cpr_interp1d_free(&pl->dP_QED);
        cpr_interp1d_free(&pl->d2P_QED);
        return 1;
    }
    cpr_log(cfg, "init", "QED pressure corrections tables loaded.");
    return 0;
}

void cpr_plasma_free(CPRPlasma *pl)
{
    if (pl->qed_active) {
        cpr_interp1d_free(&pl->P_QED);
        cpr_interp1d_free(&pl->dP_QED);
        cpr_interp1d_free(&pl->d2P_QED);
    }
    cpr_cubic_spline_free(&pl->rho_e_tab);
    cpr_cubic_spline_free(&pl->p_e_tab);
    cpr_cubic_spline_free(&pl->drho_e_dT_tab);
    cpr_cubic_spline_free(&pl->dp_e_dT_tab);
    memset(pl, 0, sizeof(*pl));
}

static double qed_P(const CPRPlasma *pl, double Tg)   { return pl->qed_active ? cpr_interp1d_eval(&pl->P_QED, Tg) : 0.0; }
static double qed_dP(const CPRPlasma *pl, double Tg)  { return pl->qed_active ? cpr_interp1d_eval(&pl->dP_QED, Tg) : 0.0; }
static double qed_d2P(const CPRPlasma *pl, double Tg) { return pl->qed_active ? cpr_interp1d_eval(&pl->d2P_QED, Tg) : 0.0; }

double cpr_plasma_rho_e(const CPRPlasma *pl, double Tg)
{
    if (Tg < g_const.me / ELEC_THERMO_LOWT_RATIO) return 0.0;
    return cpr_cubic_spline_eval(&pl->rho_e_tab, Tg);
}

double cpr_plasma_drho_e_dT(const CPRPlasma *pl, double Tg)
{
    if (Tg < g_const.me / ELEC_THERMO_LOWT_RATIO) return 0.0;
    return cpr_cubic_spline_eval(&pl->drho_e_dT_tab, Tg);
}

double cpr_plasma_p_e(const CPRPlasma *pl, double Tg)
{
    if (Tg < g_const.me / ELEC_THERMO_LOWT_RATIO) return 0.0;
    return cpr_cubic_spline_eval(&pl->p_e_tab, Tg);
}

double cpr_plasma_dp_e_dT(const CPRPlasma *pl, double Tg)
{
    if (Tg < g_const.me / ELEC_THERMO_LOWT_RATIO) return 0.0;
    return cpr_cubic_spline_eval(&pl->dp_e_dT_tab, Tg);
}

double cpr_plasma_rho_nu_extra(const CPRPlasma *pl, double Tg)
{
    if (pl->cfg->DeltaNeff == 0.0) return 0.0;
    double Tnu_dec = cpr_plasma_T_nu_decoupling(pl, Tg);
    return pl->cfg->DeltaNeff * 2.0 * (7.0 / 8.0) * (M_PI * M_PI / 30.0) * pow(Tnu_dec, 4.0);
}

double cpr_plasma_rho_SM(const CPRPlasma *pl, double Tg, double Tnue, double Tnumu)
{
    double rho_qed = Tg * qed_dP(pl, Tg) - qed_P(pl, Tg);
    return cpr_rho_g(Tg) + cpr_plasma_rho_e(pl, Tg) + rho_qed
         + cpr_rho_nu(Tnue) + 2.0 * cpr_rho_nu(Tnumu)
         + cpr_plasma_rho_nu_extra(pl, Tg);
}

double cpr_plasma_p_SM(const CPRPlasma *pl, double Tg, double Tnue, double Tnumu)
{
    return cpr_rho_g(Tg) / 3.0 + cpr_plasma_p_e(pl, Tg) + qed_P(pl, Tg)
         + (cpr_rho_nu(Tnue) + 2.0 * cpr_rho_nu(Tnumu)) / 3.0
         + cpr_plasma_rho_nu_extra(pl, Tg) / 3.0;
}

double cpr_plasma_spl(const CPRPlasma *pl, double Tg)
{
    double rho_pl = cpr_rho_g(Tg) + cpr_plasma_rho_e(pl, Tg);
    double p_pl   = cpr_rho_g(Tg) / 3.0 + cpr_plasma_p_e(pl, Tg);
    double rho_qed = Tg * qed_dP(pl, Tg) - qed_P(pl, Tg);
    double p_qed   = qed_P(pl, Tg);
    return (rho_pl + p_pl + rho_qed + p_qed) / Tg;
}

void cpr_plasma_spl_and_dspl_dT(const CPRPlasma *pl, double Tg, double *s, double *ds_dT)
{
    double rho_g_val = cpr_rho_g(Tg);
    double rho_e_val = cpr_plasma_rho_e(pl, Tg);
    double p_e_val   = cpr_plasma_p_e(pl, Tg);
    double P_val   = qed_P(pl, Tg);
    double dP_val  = qed_dP(pl, Tg);
    double d2P_val = qed_d2P(pl, Tg);

    double rho_pl = rho_g_val + rho_e_val;
    double p_pl   = rho_g_val / 3.0 + p_e_val;
    double rho_qed = Tg * dP_val - P_val;
    double p_qed   = P_val;
    *s = (rho_pl + p_pl + rho_qed + p_qed) / Tg;

    double drho_g_val = cpr_drho_g_dT(Tg);
    double drho_pl_dT = drho_g_val + cpr_plasma_drho_e_dT(pl, Tg);
    double dp_pl_dT   = drho_g_val / 3.0 + cpr_plasma_dp_e_dT(pl, Tg);
    double drho_qed_dT = Tg * d2P_val;  /* d/dT[T dP/dT - P] = T d^2P/dT^2 */
    double dp_qed_dT   = dP_val;        /* d/dT[P] = dP/dT */
    *ds_dT = (drho_pl_dT + dp_pl_dT + drho_qed_dT + dp_qed_dT) / Tg - *s / Tg;
}

double cpr_plasma_dspl_dT(const CPRPlasma *pl, double Tg)
{
    double s, ds_dT;
    cpr_plasma_spl_and_dspl_dT(pl, Tg, &s, &ds_dT);
    return ds_dT;
}

double cpr_plasma_T_nu_decoupling(const CPRPlasma *pl, double Tg)
{
    return Tg * pow(cpr_plasma_spl(pl, Tg) / (SIGMA_INF * pow(Tg, 3.0)), 1.0 / 3.0);
}
