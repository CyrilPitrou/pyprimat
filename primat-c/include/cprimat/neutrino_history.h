/* neutrino_history.h -- pluggable neutrino-sector background (port of
 * pyprimat/neutrino_history.py).
 *
 * The neutrino sector entering the cosmological background is fully
 * described, as a function of the photon temperature Tg [MeV], by:
 *
 *   - the three flavour temperatures Tnue/Tnumu/Tnutau(Tg),
 *   - the NEVO heating function N(Tg) (entropy injected into neutrinos
 *     during e+e- annihilation; drives the a(Tg) ODE in background.c),
 *   - the neutrino spectral-distortion correction dFDneu(en, x, znu, sgnq)
 *     to the n<->p weak-rate integrand (0 when there is none),
 *   - x_of_Tg(Tg), the NEVO table's scale-factor proxy (used only by
 *     external_scale_factor mode in background.c).
 *
 * Two regimes are ported (CPLAN.md S0/S6 table): CPR_NU_NEVO_TABLE
 * (incomplete/non-instantaneous decoupling, reading the pre-computed NEVO
 * tables, cfg->incomplete_decoupling) and CPR_NU_INSTANTANEOUS (complete
 * decoupling, Tnu fixed by EM entropy conservation, N=0). The analytic
 * mu/y-type distortion decorator (neutrino_history.AnalyticDistortion,
 * cfg->analytic_distortions) is explicitly OUT OF SCOPE (CPLAN.md S0) and
 * not ported; the full NEVO-spectrum-based distortion
 * (cfg->spectral_distortions with analytic_distortions=False, the default)
 * IS in scope and is ported in CPR_NU_NEVO_TABLE's distortion table.
 *
 * Reference: Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095).
 */
#ifndef CPRIMAT_NEUTRINO_HISTORY_H
#define CPRIMAT_NEUTRINO_HISTORY_H

#include "cprimat/config.h"
#include "cprimat/plasma.h"

/* Resolves a rates/NEVO/ data file, honouring a config override (mirrors
 * neutrino_history.resolve_nevo_path). `override` is one of
 * cfg->nevo_file/nevo_spectral_file/nevo_grid_file (NULL = unset); when set
 * it names the file to use instead of `default_filename`, either an
 * absolute path or a filename resolved relative to rates/NEVO/. Writes the
 * resolved absolute path into `out` (caller-allocated, size out_size). */
void cpr_resolve_nevo_path(const CPRConfig *cfg, const char *override,
                            const char *default_filename, char *out, size_t out_size);

typedef enum { CPR_NU_NEVO_TABLE, CPR_NU_INSTANTANEOUS } CPRNuHistKind;

typedef struct {
    CPRNuHistKind kind;
    const CPRConfig *cfg;     /* borrowed */
    const CPRPlasma *plasma;  /* borrowed; only used by CPR_NU_INSTANTANEOUS */

    /* ---- CPR_NU_NEVO_TABLE: flavour-temperature-ratio and heating tables,
     * ascending in Tg, with the same boundary conventions as Python's
     * interp1d calls (see neutrino_history.c for the exact match per
     * field). ---- */
    double *Tg_asc;                              /* length n_tab, ascending */
    double *ratio_ue_asc, *ratio_umu_asc, *ratio_utau_asc; /* Tnu_a/Tg */
    double *N_asc;                                /* N_NEVO */
    size_t n_tab;

    /* x_of_Tg: log-log table of the NEVO scale-factor proxy x = me/(kB Tg),
     * plus the radiation-domination extrapolation anchors. */
    double *logTg_x_asc, *logx_asc;
    size_t n_x;
    double x_Tg_min, x_Tg_max;   /* table edges (ascending Tg) */
    double x_at_Tg_min, x_at_Tg_max; /* x value at those edges */

    /* NEVO-spectrum spectral distortion (cfg->spectral_distortions &&
     * !cfg->analytic_distortions only; has_distortion is 0 otherwise). */
    int has_distortion;
    double x_min_table, x_max_table; /* range of x = me/(kB Tg) covered */
    double *x_table_sorted, *xNEVO_of_xtable_sorted; /* 1D interp1d, x ascending */
    size_t n_dist_rows;
    double *logxNEVO_asc; /* ascending x_NEVO axis of the 2D table, length n_dist_rows */
    double *df_table;     /* row-major [n_dist_rows][n_y], same x_NEVO order as logxNEVO_asc */
    double *y_nodes;       /* ascending, length n_y */
    size_t n_y;
    double y_min, y_max;

    /* ---- CPR_NU_INSTANTANEOUS: high-T limit of spl(T)/T^3. ---- */
    double sbar_ref;
} CPRNeutrinoHistory;

/* Builds the neutrino history selected by cfg->incomplete_decoupling
 * (mirrors make_neutrino_history, minus the AnalyticDistortion wrap --
 * out of scope, see this header's top comment). `plasma` must already be
 * initialised (cpr_plasma_init) and must outlive `nh`. Returns 0 on
 * success (caller must cpr_neutrino_history_free the result), nonzero with
 * *errmsg set (caller frees) on failure (e.g. a malformed/missing NEVO
 * table file). */
int cpr_neutrino_history_init(CPRNeutrinoHistory *nh, const CPRConfig *cfg,
                               const CPRPlasma *plasma, char **errmsg);

void cpr_neutrino_history_free(CPRNeutrinoHistory *nh);

double cpr_nu_Tnue_of_Tg(const CPRNeutrinoHistory *nh, double Tg);
double cpr_nu_Tnumu_of_Tg(const CPRNeutrinoHistory *nh, double Tg);
double cpr_nu_Tnutau_of_Tg(const CPRNeutrinoHistory *nh, double Tg);

/* NEVO heating function N(Tg) [dimensionless]: 0 identically for
 * CPR_NU_INSTANTANEOUS; for CPR_NU_NEVO_TABLE, 0 outside the table range
 * (matches interp1d(..., fill_value=(0.,0.)), NOT edge-value clamping). */
double cpr_nu_N_NEVO_of_Tg(const CPRNeutrinoHistory *nh, double Tg);

/* NEVO table scale-factor proxy x(Tg) = me/(kB Tg) up to a normalisation
 * (x ∝ a); only meaningful for CPR_NU_NEVO_TABLE (used by
 * external_scale_factor mode in background.c). Radiation-domination
 * extrapolation (x*Tg = const) outside the table. */
double cpr_nu_x_of_Tg(const CPRNeutrinoHistory *nh, double Tg);

/* Neutrino spectral-distortion correction to the n<->p weak-rate
 * integrand: dFDneu(en, x, znu, sgnq) = f_actual - f_FD at the shifted
 * neutrino energy. 0 identically when nh->has_distortion is 0 (analytic-
 * distortion mode and CPR_NU_INSTANTANEOUS without spectral_distortions
 * both fall here). en: electron energy / me (sign encodes initial-state
 * vs Pauli-blocking, see neutrino_history.py's dFDneu_func docstring); x:
 * me/(kB Tg); znu: me/(kB Tnu); sgnq: +1 (n->p) or -1 (p->n). */
double cpr_nu_dFDneu(const CPRNeutrinoHistory *nh, double en, double x, double znu, double sgnq);

#endif /* CPRIMAT_NEUTRINO_HISTORY_H */
