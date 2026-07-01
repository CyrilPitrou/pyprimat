/* background.h -- cosmological background (port of primat/background.py).
 *
 * A "background" encapsulates everything the nuclear-network integration
 * needs about the expanding Universe: T_gamma(t)/t(T_gamma), the baryon
 * mass density rhoB_BBN(t) [g/cm^3] (the prefactor for nuclear reaction
 * rates), and the normalised n<->p weak rates weak_nTOp_frwrd/bkwrd(T)
 * [s^-1]. Two concrete kinds are ported:
 *
 *   CPR_BG_STANDARD: builds the full a<->t<->T relations and Friedmann
 *     Hubble rate from cfg's neutrino-decoupling mode (NEVO table or
 *     instantaneous, via neutrino_history.c), optional ΛCDM (Omegach2/h)
 *     and Early Dark Energy (fEDE/zcEDE/wnEDE) contributions to the
 *     Friedmann equation, and the n<->p weak rates (weak_rates.c).
 *
 *   CPR_BG_CUSTOM: reads a user-supplied (T, t, a) table (cfg->custom_background)
 *     and uses the instantaneous-decoupling approximation for neutrino
 *     temperatures and weak rates; Neff is estimated indirectly from the
 *     Friedmann equation by finite-differencing the table's late-time a(t).
 *
 * Both eventually expose the same query surface (cpr_bg_T_of_t etc.); the
 * `kind` field only matters for cpr_bg_init_* (which one to call) and
 * cpr_bg_omeganuh2_relnu/nrnu and cpr_bg_write_time_evolution (richer
 * output for CPR_BG_STANDARD).
 *
 * Out of scope, not ported: analytic mu/y-type spectral
 * distortions (rho_nu_SD is always NULL/inactive here, since
 * neutrino_history.c never sets has_distortion's analytic-mode sibling --
 * see neutrino_history.h's top comment); decay_era.
 *
 * Reference: Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095).
 */
#ifndef CPRIMAT_BACKGROUND_H
#define CPRIMAT_BACKGROUND_H

#include "config.h"
#include "plasma.h"
#include "neutrino_history.h"
#include "weak_rates.h"
#include <stddef.h>

typedef enum { CPR_BG_STANDARD, CPR_BG_CUSTOM } CPRBackgroundKind;

typedef struct {
    CPRBackgroundKind kind;
    const CPRConfig *cfg;   /* borrowed */
    const CPRPlasma *plasma; /* borrowed */

    /* ---- Friedmann extra-energy-density plug-ins (StandardBackground only).
     * Python keeps a generic list of rho(Tg) callables (extra_rho); since
     * the only in-scope members are CDM, Lambda, and EDE (no runtime
     * plug-in mechanism is otherwise exposed here), a fixed flag-
     * gated trio replaces the generic list -- behaviourally identical,
     * simpler in C. */
    int has_lcdm;          /* Omegach2/h both finite (always true for CPRConfig's
                             * non-nullable double fields -- see _setup_LCDM's
                             * docstring in background.c for the deviation note) */
    double rhocdm_a3;      /* Omega_c h^2 * rhocrit100 [MeV^4], rho_CDM = rhocdm_a3/a^3 */
    double rholambda;      /* Omega_Lambda h^2 * rhocrit100 [MeV^4], constant */
    int has_ede;           /* fEDE != 0 */
    double TcEDE;          /* [MeV] */
    double rhocEDEac;      /* [MeV^4] */
    double EDE_exponent;   /* 3*wnEDE + 3 */
    /* Historical note (pre-Branch E): CDM used to use the radiation-
     * domination approximation a(T)~=T0CMB/T while the a(T)/t(a) ODEs were
     * being solved sequentially (cpr_bg_Hubble had no way to know `a`
     * exactly until the first ODE -- a(T) -- was fully solved and splined),
     * then switched to the exact a_of_T once available. Since the combined
     * a(T)/t(T) 2D ODE (setup_background_and_cosmo) always carries `a` in
     * its own state vector (x = ln(a*T)), cpr_bg_Hubble now takes `a`
     * directly as an explicit parameter and is always exact -- no bootstrap,
     * no `lcdm_use_exact` flag needed any more. */

    /* ---- Neutrino sector (StandardBackground: NEVO table or instantaneous,
     * via neutrino_history.c; CustomBackground: always instantaneous,
     * built directly without an owned CPRNeutrinoHistory -- see
     * cpr_bg_init_custom's docstring). ---- */
    CPRNeutrinoHistory nh;       /* owned; valid only when kind == CPR_BG_STANDARD */
    int nh_owned;

    /* ---- a<->t<->T tables, built once by cpr_bg_init_standard /
     * cpr_bg_init_custom and queried by cpr_bg_*_of_*. All arrays below
     * are owned. ---- */
    double *t_vec, *Tg_vec, *Tnue_vec, *Tnumu_vec, *Tnutau_vec, *Tnu_vec; /* avg flavour T */
    double *a_vec;           /* scale factor on the same t_vec grid (CPR_BG_STANDARD only) */
    size_t n_bg;             /* length of the arrays above */
    /* Tg_vec is built time-ascending, hence T-*descending* (T falls as the
     * universe expands); cpr_interp_linear requires an ascending x array,
     * so cpr_bg_t_of_T keeps this reversed (T-ascending) copy instead of
     * reusing Tg_vec/t_vec directly. */
    double *Tg_asc, *t_by_Tg_asc;
    int has_scale_factor;
    int has_heating_table;   /* CPR_BG_STANDARD && cfg->incomplete_decoupling */

    /* a(T): either the closed-form external_scale_factor branch (K_ext *
     * cpr_nu_x_of_Tg), or a linear interpolant over the ODE solution
     * (lnT_sol/lna_sol, ascending in T). T_of_a is always a linear
     * interpolant over (a_sol_asc, T_sol_asc), built ascending in `a`
     * (the opposite order from lnT_sol/lna_sol -- see background.c). */
    int external_scale_factor;
    double K_ext;
    double *lnT_sol, *lna_sol;          /* ascending in T; used iff !external_scale_factor */
    size_t n_Tsol;
    double *a_sol_asc, *T_sol_asc;       /* ascending in a (descending in T) */

    /* ---- n<->p weak rates (both kinds). ---- */
    CPRWeakRates wr;
    double norm_weak_rates;

    /* ---- CPR_BG_CUSTOM table-mode raw arrays. Python's CustomBackground
     * builds three separately-sorted copies of the same (T,t,a) triple,
     * one per independent variable, each feeding a `kind='linear'`,
     * `fill_value='extrapolate'` interp1d (LINEAR extrapolation
     * throughout -- unlike CPR_BG_STANDARD's a_of_t/t_of_a, which use
     * CONSTANT extrapolation, see cpr_bg_a_of_t/_t_of_a in background.c).
     * t_asc/T_by_t/a_by_t (t-ascending) doubles as the array
     * cpr_bg_rho_nu_total_final's finite-difference Neff estimate reads. */
    double *t_asc, *T_by_t, *a_by_t;       /* t-ascending */
    double *T_asc, *t_by_T, *a_by_T;       /* T-ascending */
    double *a_sort, *T_by_a, *t_by_a;      /* a-ascending */
    size_t n_custom;
} CPRBackground;

/* StandardBackground: builds extra_rho (CDM/Lambda/EDE), the neutrino
 * history, the a(T)/t(a) ODE solutions, derived relic-neutrino Omegas, and
 * the n<->p weak rates -- mirrors StandardBackground.__init__'s call
 * sequence (_setup_LCDM, _setup_EDE, _setup_background_and_cosmo,
 * _setup_derived_cosmo, _setup_weak_rates; _replace_LCDM_with_exact has no
 * separate step here, see background.c). `plasma` must already be
 * initialised (cpr_plasma_init) and must outlive `bg`. Returns 0 on
 * success (caller must cpr_background_free the result), nonzero with
 * *errmsg set (caller frees) on failure. */
int cpr_bg_init_standard(CPRBackground *bg, const CPRConfig *cfg, const CPRPlasma *plasma,
                          char **errmsg);

/* CustomBackground: reads (T,t,a) from `filename` (tab- or comma-delimited,
 * header row naming columns, T/t/a required, extra columns ignored; rows
 * may be in any order), builds instantaneous-decoupling neutrino
 * temperatures and the n<->p weak rates over the table's T range. Mirrors
 * CustomBackground.__init__. Returns 0 on success (caller must
 * cpr_background_free), nonzero with *errmsg set (caller frees) on
 * failure (missing required column, non-positive T/t/a value). */
int cpr_bg_init_custom(CPRBackground *bg, const CPRConfig *cfg, const CPRPlasma *plasma,
                        const char *filename, char **errmsg);

void cpr_background_free(CPRBackground *bg);

/* ---- Compulsory interface (both kinds). ---- */
double cpr_bg_T_of_t(const CPRBackground *bg, double t);   /* [MeV] */
double cpr_bg_t_of_T(const CPRBackground *bg, double T);   /* [s] */
double cpr_bg_rhoB_BBN(const CPRBackground *bg, double t); /* [g/cm^3] */
double cpr_bg_weak_nTOp_frwrd(const CPRBackground *bg, double T_K); /* [s^-1] */
double cpr_bg_weak_nTOp_bkwrd(const CPRBackground *bg, double T_K); /* [s^-1] */

/* ---- Scale-factor interface (both kinds set has_scale_factor=1). ---- */
double cpr_bg_a_of_T(const CPRBackground *bg, double T);
double cpr_bg_T_of_a(const CPRBackground *bg, double a);
double cpr_bg_a_of_t(const CPRBackground *bg, double t);
double cpr_bg_t_of_a(const CPRBackground *bg, double a);

/* ---- Friedmann expansion rate H [s^-1] at Tg and the three flavour
 * neutrino temperatures [MeV] (CPR_BG_STANDARD only -- CustomBackground
 * has no Hubble()/extra_rho machinery, mirroring Python). `a` is the scale
 * factor at this Tg, supplied explicitly by the caller (always known
 * exactly -- either read off the combined a(T)/t(T) ODE's state vector
 * while it is being solved, or looked up via cpr_bg_a_of_T once solved;
 * see background.c's setup_background_and_cosmo). ---- */
double cpr_bg_Hubble(const CPRBackground *bg, double Tg, double Tnue, double Tnumu,
                      double Tnutau, double a);

/* Per-flavour neutrino temperature [MeV] at cosmic time t [s] -- mirrors
 * Background.Tnu_of_t (background.py). For CPR_BG_STANDARD, linearly
 * interpolates/extrapolates bg->Tnue_vec/Tnumu_vec/Tnutau_vec over
 * bg->t_vec, exactly like Python's StandardBackground.Tnu_of_t
 * (interp1d(..., kind='linear', fill_value='extrapolate')); writes Tnue/
 * Tnumu/Tnutau (via the output pointers) and returns 1. For CPR_BG_CUSTOM,
 * which tracks no time-indexed neutrino sector, writes nothing and
 * returns 0 -- mirroring
 * Python's base Background.Tnu_of_t returning None there (callers fill
 * NaN, see cpr_nuclear_network_sample_time_evolution). */
int cpr_bg_Tnu_of_t(const CPRBackground *bg, double t, double *Tnue, double *Tnumu,
                     double *Tnutau);

/* ---- Derived cosmology (optional in Python; both kinds implement it
 * here, writing into Tg_final/rho_nu_tot_final). Returns 0 (always
 * available for both ported kinds; Background's own "None" default has no
 * C analogue since both v1 kinds override it). ---- */
int cpr_bg_rho_nu_total_final(const CPRBackground *bg, double *Tg_final, double *rho_nu_tot_final);

/* Neff = rho_nu_tot / rho_g(Tg) / ((7/8)(4/11)^(4/3)) -- generic formula,
 * identical for both kinds (mirrors Background.N_eff). */
double cpr_bg_N_eff(const CPRBackground *bg, double Tg, double rho_nu_tot);

/* Relic-neutrino Omega_nu h^2 x 1e-6, in the relativistic-today and
 * non-relativistic-today conventions respectively. CPR_BG_STANDARD only
 * (CustomBackground does not track a separate relic-neutrino calculation,
 * mirroring Python's CustomBackground which does not override these and
 * so inherits Background's `None` default); `*out` is left untouched and
 * this returns nonzero when kind != CPR_BG_STANDARD. */
int cpr_bg_Omeganuh2_relnu(const CPRBackground *bg, double *out);
int cpr_bg_Omeganuh2_nrnu(const CPRBackground *bg, double *out);

/* Writes the background time evolution to a TSV file, mirroring
 * Python's StandardBackground.write_time_evolution. Columns: T [MeV], t [s],
 * a [1], H [s^-1], Tnue [MeV], Tnumu [MeV], Tnutau [MeV], Nheating [1] (if
 * has_heating_table), rho_plasma [MeV^4], rho_nu_tot [MeV^4],
 * rho_extra [MeV^4] (if has_extra), rho_tot [MeV^4]. Returns 0 on success,
 * nonzero with *errmsg set (caller frees) on failure. */
int cpr_bg_write_time_evolution(const CPRBackground *bg, const char *path, int n_points, char **errmsg);

#endif /* CPRIMAT_BACKGROUND_H */
