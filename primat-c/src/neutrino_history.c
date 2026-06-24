/* neutrino_history.c -- see cprimat/neutrino_history.h. */
#include "cprimat/neutrino_history.h"
#include "cprimat/constants.h"
#include "cprimat/table_io.h"
#include "cprimat/spline.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void cpr_resolve_nevo_path(const CPRConfig *cfg, const char *override,
                            const char *default_filename, char *out, size_t out_size)
{
    const char *fname = override ? override : default_filename;
    if (fname[0] == '/') {
        snprintf(out, out_size, "%s", fname);
    } else {
        snprintf(out, out_size, "%s/rates/NEVO/%s", cfg->data_dir, fname);
    }
}

/* Index i such that asc[i] <= xq <= asc[i+1] (clamped to [0, n-2]); caller
 * has already checked xq is within [asc[0], asc[n-1]]. Plain linear scan:
 * every table here has at most a few hundred rows, called O(1) times per
 * weak-rate/background evaluation, so this is not a hot loop worth a
 * binary search. */
static size_t bracket(const double *asc, size_t n, double xq)
{
    size_t i = 0;
    while (i + 2 < n && asc[i + 1] < xq) i++;
    return i;
}

static double interp_asc(const double *x_asc, const double *y_asc, size_t n, double xq, CPRExtrapMode mode)
{
    return cpr_interp_linear(x_asc, y_asc, n, xq, mode);
}

/* ---------------------------------------------------------------------
 * CPR_NU_NEVO_TABLE construction.
 * ------------------------------------------------------------------- */

static int build_nevo_table(CPRNeutrinoHistory *nh, const CPRConfig *cfg, char **errmsg)
{
    char path[4224];
    const char *prefix = cfg->nevo_file_prefix ? cfg->nevo_file_prefix : "NEVOPRIMAT";
    char default_file[256];
    snprintf(default_file, sizeof(default_file), "%s%s_col_1_7.csv", prefix,
             cfg->QED_corrections ? "" : "_NoQED");
    cpr_resolve_nevo_path(cfg, cfg->nevo_file, default_file, path, sizeof(path));

    CPRTable tab;
    if (cpr_table_read(path, 7, &tab, errmsg)) return 1;

    size_t n = tab.n_rows;
    double *x      = tab.cols[0];
    double *z      = tab.cols[1];
    double *Tnue_r = tab.cols[2], *Tnumu_r = tab.cols[3], *Tnutau_r = tab.cols[4];
    double *N_r    = tab.cols[5];

    double me = g_const.me;
    double *Tg = malloc(n * sizeof(double));
    double *Tnue = malloc(n * sizeof(double)), *Tnumu = malloc(n * sizeof(double)), *Tnutau = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        Tg[i]     = me * z[i] / x[i];
        Tnue[i]   = Tnue_r[i]   * me / x[i];
        Tnumu[i]  = Tnumu_r[i]  * me / x[i];
        Tnutau[i] = Tnutau_r[i] * me / x[i];
    }

    /* Table is naturally high->low Tg (descending); we want ascending for
     * cpr_interp_linear, so detect and reverse exactly like Python's
     * "if Tg_tab[0] < Tg_tab[-1]: reverse" (there, reversal makes it
     * descending; here we want the opposite sense -- ascending). */
    int need_reverse = (n >= 2 && Tg[0] > Tg[n - 1]);

    nh->n_tab = n;
    nh->Tg_asc        = malloc(n * sizeof(double));
    nh->ratio_ue_asc  = malloc(n * sizeof(double));
    nh->ratio_umu_asc = malloc(n * sizeof(double));
    nh->ratio_utau_asc = malloc(n * sizeof(double));
    nh->N_asc         = malloc(n * sizeof(double));
    /* Also keep an ascending-x_NEVO copy for x_of_Tg below. */
    double *x_for_xofTg = malloc(n * sizeof(double));

    for (size_t i = 0; i < n; i++) {
        size_t src = need_reverse ? (n - 1 - i) : i;
        nh->Tg_asc[i]        = Tg[src];
        nh->ratio_ue_asc[i]  = Tnue[src]   / Tg[src];
        nh->ratio_umu_asc[i] = Tnumu[src]  / Tg[src];
        nh->ratio_utau_asc[i] = Tnutau[src] / Tg[src];
        nh->N_asc[i]         = N_r[src];
        x_for_xofTg[i]       = x[src];
    }

    nh->n_x = n;
    nh->logTg_x_asc = malloc(n * sizeof(double));
    nh->logx_asc    = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        nh->logTg_x_asc[i] = log(nh->Tg_asc[i]);
        nh->logx_asc[i]    = log(x_for_xofTg[i]);
    }
    nh->x_Tg_min = nh->Tg_asc[0];
    nh->x_Tg_max = nh->Tg_asc[n - 1];
    nh->x_at_Tg_min = x_for_xofTg[0];
    nh->x_at_Tg_max = x_for_xofTg[n - 1];

    free(Tg); free(Tnue); free(Tnumu); free(Tnutau); free(x_for_xofTg);
    cpr_table_free(&tab);

    /* Spectral distortion from the full 86-column NEVO spectrum table
     * (cfg->spectral_distortions && !cfg->analytic_distortions only --
     * the analytic mu/y decorator is out of scope, see neutrino_history.h). */
    nh->has_distortion = 0;
    if (!(cfg->spectral_distortions && !cfg->analytic_distortions))
        return 0;

    char full_path[4224], grid_path[4224];
    char default_full[256];
    snprintf(default_full, sizeof(default_full), "%s%s.csv", prefix,
             cfg->QED_corrections ? "" : "_NoQED");
    cpr_resolve_nevo_path(cfg, cfg->nevo_spectral_file, default_full, full_path, sizeof(full_path));
    cpr_resolve_nevo_path(cfg, cfg->nevo_grid_file, "NEVOGrid.csv", grid_path, sizeof(grid_path));

    CPRTable full_tab, grid_tab;
    if (cpr_table_read(full_path, 86, &full_tab, errmsg)) return 1;
    if (cpr_table_read(grid_path, 1, &grid_tab, errmsg)) { cpr_table_free(&full_tab); return 1; }

    size_t nr = full_tab.n_rows;
    size_t ny = grid_tab.n_rows;
    double *x_NEVO_raw = full_tab.cols[0];
    double *z_NEVO_raw = full_tab.cols[1];

    int rev = (nr >= 2 && x_NEVO_raw[0] > x_NEVO_raw[nr - 1]);

    nh->n_dist_rows = nr;
    nh->n_y = ny;
    nh->y_nodes = malloc(ny * sizeof(double));
    memcpy(nh->y_nodes, grid_tab.cols[0], ny * sizeof(double));
    nh->y_min = nh->y_nodes[0];
    nh->y_max = nh->y_nodes[ny - 1];

    double *xNEVO_asc = malloc(nr * sizeof(double));
    double *x_table_unsorted = malloc(nr * sizeof(double));
    nh->df_table = malloc(nr * ny * sizeof(double));
    for (size_t i = 0; i < nr; i++) {
        size_t src = rev ? (nr - 1 - i) : i;
        xNEVO_asc[i] = x_NEVO_raw[src];
        x_table_unsorted[i] = x_NEVO_raw[src] / z_NEVO_raw[src];
        for (size_t j = 0; j < ny; j++)
            nh->df_table[i * ny + j] = full_tab.cols[6 + j][src];
    }
    nh->logxNEVO_asc = malloc(nr * sizeof(double));
    for (size_t i = 0; i < nr; i++) nh->logxNEVO_asc[i] = log(xNEVO_asc[i]);

    nh->x_min_table = x_table_unsorted[0];
    nh->x_max_table = x_table_unsorted[0];
    for (size_t i = 1; i < nr; i++) {
        if (x_table_unsorted[i] < nh->x_min_table) nh->x_min_table = x_table_unsorted[i];
        if (x_table_unsorted[i] > nh->x_max_table) nh->x_max_table = x_table_unsorted[i];
    }

    /* idx_sort: sort x_table_unsorted ascending (matches np.argsort), for
     * the 1D x_table -> x_NEVO interpolant (linear-extrapolated). Simple
     * insertion sort on index array -- nr is a few hundred rows. */
    size_t *idx = malloc(nr * sizeof(size_t));
    for (size_t i = 0; i < nr; i++) idx[i] = i;
    for (size_t i = 1; i < nr; i++) {
        size_t key = idx[i];
        double keyval = x_table_unsorted[key];
        size_t j = i;
        while (j > 0 && x_table_unsorted[idx[j - 1]] > keyval) { idx[j] = idx[j - 1]; j--; }
        idx[j] = key;
    }
    nh->x_table_sorted = malloc(nr * sizeof(double));
    nh->xNEVO_of_xtable_sorted = malloc(nr * sizeof(double));
    for (size_t i = 0; i < nr; i++) {
        nh->x_table_sorted[i] = x_table_unsorted[idx[i]];
        nh->xNEVO_of_xtable_sorted[i] = xNEVO_asc[idx[i]];
    }
    free(idx); free(xNEVO_asc); free(x_table_unsorted);
    cpr_table_free(&full_tab);
    cpr_table_free(&grid_tab);

    nh->has_distortion = 1;
    return 0;
}

/* ---------------------------------------------------------------------
 * CPR_NU_INSTANTANEOUS construction.
 * ------------------------------------------------------------------- */

static void build_instantaneous(CPRNeutrinoHistory *nh, const CPRConfig *cfg)
{
    /* sbar_ref: high-T limit of spl(T)/T^3 -- see neutrino_history.py's
     * InstantaneousDecoupling._build_temperatures comment for the
     * Dodelson & Turner 1992 / Heckler 1994 perturbative QED formula. */
    if (cfg->QED_corrections) {
        double alpha = g_const.alphaem;
        double ratio3 = 11.0 / 4.0
                      - 25.0 * alpha / (8.0 * M_PI)
                      + 10.0 * pow(alpha, 1.5) * sqrt(M_PI / 3.0) / (M_PI * M_PI);
        nh->sbar_ref = ratio3 * (4.0 * M_PI * M_PI / 45.0);
    } else {
        nh->sbar_ref = 11.0 * M_PI * M_PI / 45.0;
    }
}

int cpr_neutrino_history_init(CPRNeutrinoHistory *nh, const CPRConfig *cfg,
                               const CPRPlasma *plasma, char **errmsg)
{
    memset(nh, 0, sizeof(*nh));
    nh->cfg = cfg;
    nh->plasma = plasma;

    if (cfg->incomplete_decoupling) {
        nh->kind = CPR_NU_NEVO_TABLE;
        if (build_nevo_table(nh, cfg, errmsg)) return 1;
    } else {
        nh->kind = CPR_NU_INSTANTANEOUS;
        build_instantaneous(nh, cfg);
    }
    return 0;
}

void cpr_neutrino_history_free(CPRNeutrinoHistory *nh)
{
    free(nh->Tg_asc); free(nh->ratio_ue_asc); free(nh->ratio_umu_asc); free(nh->ratio_utau_asc);
    free(nh->N_asc);
    free(nh->logTg_x_asc); free(nh->logx_asc);
    free(nh->x_table_sorted); free(nh->xNEVO_of_xtable_sorted);
    free(nh->logxNEVO_asc); free(nh->df_table); free(nh->y_nodes);
    memset(nh, 0, sizeof(*nh));
}

/* ---------------------------------------------------------------------
 * Public evaluators.
 * ------------------------------------------------------------------- */

static double nevo_ratio(const CPRNeutrinoHistory *nh, const double *ratio_asc, double Tg)
{
    /* fill_value=(ratio[-1], ratio[0]) in Python's descending-Tg storage
     * is exactly the edge value of our ascending arrays -- constant
     * extrapolation. */
    return interp_asc(nh->Tg_asc, ratio_asc, nh->n_tab, Tg, CPR_EXTRAP_CONSTANT);
}

double cpr_nu_Tnue_of_Tg(const CPRNeutrinoHistory *nh, double Tg)
{
    if (nh->kind == CPR_NU_INSTANTANEOUS)
        return pow(cpr_plasma_spl(nh->plasma, Tg) / nh->sbar_ref, 1.0 / 3.0);
    return nevo_ratio(nh, nh->ratio_ue_asc, Tg) * Tg;
}

double cpr_nu_Tnumu_of_Tg(const CPRNeutrinoHistory *nh, double Tg)
{
    if (nh->kind == CPR_NU_INSTANTANEOUS)
        return pow(cpr_plasma_spl(nh->plasma, Tg) / nh->sbar_ref, 1.0 / 3.0);
    return nevo_ratio(nh, nh->ratio_umu_asc, Tg) * Tg;
}

double cpr_nu_Tnutau_of_Tg(const CPRNeutrinoHistory *nh, double Tg)
{
    if (nh->kind == CPR_NU_INSTANTANEOUS)
        return pow(cpr_plasma_spl(nh->plasma, Tg) / nh->sbar_ref, 1.0 / 3.0);
    return nevo_ratio(nh, nh->ratio_utau_asc, Tg) * Tg;
}

double cpr_nu_N_NEVO_of_Tg(const CPRNeutrinoHistory *nh, double Tg)
{
    if (nh->kind == CPR_NU_INSTANTANEOUS) return 0.0;
    /* fill_value=(0.,0.): exactly 0 outside the table, NOT edge-clamped. */
    if (Tg < nh->Tg_asc[0] || Tg > nh->Tg_asc[nh->n_tab - 1]) return 0.0;
    return interp_asc(nh->Tg_asc, nh->N_asc, nh->n_tab, Tg, CPR_EXTRAP_CONSTANT);
}

double cpr_nu_x_of_Tg(const CPRNeutrinoHistory *nh, double Tg)
{
    if (nh->kind != CPR_NU_NEVO_TABLE) return 0.0;
    if (Tg < nh->x_Tg_min) return nh->x_at_Tg_min * nh->x_Tg_min / Tg;
    if (Tg > nh->x_Tg_max) return nh->x_at_Tg_max * nh->x_Tg_max / Tg;
    double logTg = log(Tg);
    double logx = interp_asc(nh->logTg_x_asc, nh->logx_asc, nh->n_x, logTg, CPR_EXTRAP_CONSTANT);
    return exp(logx);
}

/* Bilinear lookup on the (logxNEVO_asc, y_nodes) grid, fill_value=0 outside
 * either axis's range (matches RegularGridInterpolator(..., fill_value=0.)). */
static double df_2d_lookup(const CPRNeutrinoHistory *nh, double log_xNEVO, double y)
{
    size_t nr = nh->n_dist_rows, ny = nh->n_y;
    if (log_xNEVO < nh->logxNEVO_asc[0] || log_xNEVO > nh->logxNEVO_asc[nr - 1]) return 0.0;
    if (y < nh->y_nodes[0] || y > nh->y_nodes[ny - 1]) return 0.0;

    size_t i = bracket(nh->logxNEVO_asc, nr, log_xNEVO);
    size_t j = bracket(nh->y_nodes, ny, y);
    double x0 = nh->logxNEVO_asc[i], x1 = nh->logxNEVO_asc[i + 1];
    double y0 = nh->y_nodes[j], y1 = nh->y_nodes[j + 1];
    double tx = (x1 > x0) ? (log_xNEVO - x0) / (x1 - x0) : 0.0;
    double ty = (y1 > y0) ? (y - y0) / (y1 - y0) : 0.0;

    double v00 = nh->df_table[i * ny + j];
    double v10 = nh->df_table[(i + 1) * ny + j];
    double v01 = nh->df_table[i * ny + (j + 1)];
    double v11 = nh->df_table[(i + 1) * ny + (j + 1)];
    return v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty)
         + v01 * (1 - tx) * ty       + v11 * tx * ty;
}

static double dFDneu_raw(const CPRNeutrinoHistory *nh, double en, double x, double znu)
{
    static const double EXP_CUT = 3e2;
    if (x < nh->x_min_table || x > nh->x_max_table) return 0.0;

    double xNEV = interp_asc(nh->x_table_sorted, nh->xNEVO_of_xtable_sorted, nh->n_dist_rows, x, CPR_EXTRAP_LINEAR);
    double en_ph = fabs(en);
    double y = en_ph * xNEV;
    if (y < nh->y_min || y > nh->y_max) return 0.0;

    double df = df_2d_lookup(nh, log(xNEV), y);
    double arg_y = y, arg_nu = en_ph * znu;
    double f_nevo  = (arg_y  > EXP_CUT) ? 0.0 : (1.0 + df) / (exp(arg_y) + 1.0);
    double f_fd_nu = (arg_nu > EXP_CUT) ? 0.0 : 1.0 / (exp(arg_nu) + 1.0);
    double delta_f = f_nevo - f_fd_nu;

    return (en < 0.0) ? -delta_f : delta_f;
}

double cpr_nu_dFDneu(const CPRNeutrinoHistory *nh, double en, double x, double znu, double sgnq)
{
    (void)sgnq; /* the NEVO-table distortion does not depend on sgnq (unlike
                 * the analytic mu/y decorator, out of scope here) */
    if (!nh->has_distortion) return 0.0;
    return dFDneu_raw(nh, en, x, znu);
}
