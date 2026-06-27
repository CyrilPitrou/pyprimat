/* neutrino_history.h -- pluggable neutrino-sector background (port of
 * primat/neutrino_history.py).
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
 * decoupling, Tnu fixed by EM entropy conservation, N=0). The full
 * NEVO-spectrum-based distortion (cfg->spectral_distortions with
 * analytic_distortions=False, the default) is ported in CPR_NU_NEVO_TABLE's
 * distortion table. The analytic y/gray-type distortion decorator
 * (neutrino_history.AnalyticDistortion, cfg->analytic_distortions=True,
 * which PRIMATConfig pairs with incomplete_decoupling=False i.e.
 * CPR_NU_INSTANTANEOUS) is also ported: cpr_nu_dFDneu dispatches to the
 * closed-form y-type (SZ/Compton) + gray-type distortion of
 * AnalyticDistortion._dFDneu_analytic, and cpr_nu_dFDneu_moment provides the
 * eight en-moment derivatives (AnalyticDistortion.dFDneu_moments) feeding
 * the SD-FM finite-nucleon-mass weak-rate term in weak_rates.c.
 *
 * Reference: Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095).
 */
#ifndef CPRIMAT_NEUTRINO_HISTORY_H
#define CPRIMAT_NEUTRINO_HISTORY_H

#include "cprimat/config.h"
#include "cprimat/plasma.h"

/* Resolves a data/NEVO/ data file, honouring a config override (mirrors
 * neutrino_history.resolve_nevo_path). `override` is one of
 * cfg->nevo_file/nevo_spectral_file/nevo_grid_file (NULL = unset); when set
 * it names the file to use instead of `default_filename`, either an
 * absolute path or a filename resolved relative to data/NEVO/. Writes the
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

    /* Analytic y/gray-type distortion (cfg->analytic_distortions=True;
     * only ever set when kind==CPR_NU_INSTANTANEOUS, since PRIMATConfig
     * pairs analytic_distortions with incomplete_decoupling=False -- see
     * neutrino_history.AnalyticDistortion). xi_nu is the genuine reduced
     * chemical potential (cfg->munuOverTnu) the y-type piece sits on;
     * y_sz/y_gray are cfg->y_SZ/cfg->y_gray. */
    int has_analytic_distortion;
    double xi_nu, y_sz, y_gray;
} CPRNeutrinoHistory;

/* Selects which of the 8 en-moments of the analytic y/gray distortion
 * (AnalyticDistortion.dFDneu_moments's dict keys "e2p0".."e4p2") to
 * evaluate; see cpr_nu_dFDneu_moment. */
typedef enum {
    CPR_DFD_E2P0, CPR_DFD_E3P0,
    CPR_DFD_E2P1, CPR_DFD_E3P1, CPR_DFD_E4P1,
    CPR_DFD_E2P2, CPR_DFD_E3P2, CPR_DFD_E4P2
} CPRDFDneuMomentKind;

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
 * neutrino energy. 0 identically when neither nh->has_distortion (NEVO-
 * table mode) nor nh->has_analytic_distortion (analytic mode) is set. en:
 * electron energy / me (sign encodes initial-state vs Pauli-blocking, see
 * neutrino_history.py's dFDneu_func docstring); x: me/(kB Tg) (unused in
 * analytic mode, present for interface parity); znu: me/(kB Tnu); sgnq: +1
 * (n->p) or -1 (p->n). Dispatches to AnalyticDistortion._dFDneu_analytic's
 * en>=0/en<0 antisymmetric form when nh->has_analytic_distortion. */
double cpr_nu_dFDneu(const CPRNeutrinoHistory *nh, double en, double x, double znu, double sgnq);

/* Extra neutrino energy density [MeV^4] from the analytic y/gray
 * distortion's Friedmann-equation contribution (neutrino_history.
 * AnalyticDistortion's rho_nu_SD, see _rho_nu_SD_from_int's docstring for
 * the N_nu=3 normalisation). 0 identically when nh->has_analytic_distortion
 * is 0 (NEVO-table distortion mode needs no such correction: NEVO
 * temperatures are already the energy-equivalent FD temperature). `Tnu_avg`
 * is the energy-weighted mean flavour temperature, ((Tnue^4+Tnumu^4+
 * Tnutau^4)/3)^(1/4) -- mirrors background.py's call sites. */
double cpr_nu_rho_nu_SD(const CPRNeutrinoHistory *nh, double Tnu_avg);

/* The k-th en-derivative of en^n times the analytic y/gray distortion
 * (AnalyticDistortion.dFDneu_moments["e{n}p{k}"]), used only by the SD-FM
 * (finite-nucleon-mass x spectral-distortion) weak-rate term in
 * weak_rates.c. Nonzero only when nh->has_analytic_distortion; 0
 * identically otherwise (the NEVO-table distortion has no closed-form
 * en-derivative, mirrors dFDneu_moments being None in that mode). Formulas
 * transcribed from neutrino_history.AnalyticDistortion._build_analytic_
 * distortion's _raw_M{n}p{k}/_make_moment closures (themselves generated by
 * scratch/derive_sd_fm_distortions.py) -- see neutrino_history.c for the
 * verbatim transcription. */
double cpr_nu_dFDneu_moment(const CPRNeutrinoHistory *nh, CPRDFDneuMomentKind kind,
                             double en, double x, double znu, double sgnq);

#endif /* CPRIMAT_NEUTRINO_HISTORY_H */
