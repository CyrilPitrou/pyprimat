/* weak_rates.h -- n<->p weak-rate tables (port of pyprimat/weak_rates/).
 *
 * CPLAN.md S7 scope (S7a, "always-needed pieces"): the non-thermal n<->p
 * rate (Born/CCR + finite-nucleon-mass + spectral-distortion corrections),
 * computed from scratch via fixed Gauss-Legendre quadrature when no cache
 * file matches the current configuration's fingerprint, or loaded directly
 * from data/weak/nTOp_<hash>.txt otherwise (cache.c already ports the
 * fingerprint/hash/cache-file machinery, see cache.h). The finite-
 * temperature radiative correction (CCRTh, Brown & Sawyer 2001) is S7b:
 * cpr_weak_rates_init *loads* its cache file
 * (data/weak/nTOp_thermal_<hash>.txt) when cfg->thermal_corrections is set
 * and a matching file exists, and otherwise recomputes it from scratch via
 * the same algorithm Python's `corrections.py` uses -- VEGAS adaptive
 * Monte Carlo (vegas.h) for the three 2D sub-integrals, deterministic 1D
 * quadrature for the one 1D sub-integral -- see weak_rates.c's CCRTh
 * section.
 *
 * The SD-FM correction terms (_L_SD_FMCCR/_L_SD_FMNoCCR in the Python
 * source) are analytic-distortion-mode only (cfg->analytic_distortions)
 * and ARE ported: see chi_func_sd_fm_v and the L_SD_FMCCR/L_SD_FMNOCCR
 * LKind cases in weak_rates.c, wired in nonthermal_rate_term whenever
 * cfg->analytic_distortions && cfg->spectral_distortions &&
 * cfg->finite_mass_corrections.
 *
 * Reference: Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018
 * (arXiv:1806.11095), cited below as "Phys. Rep.".
 */
#ifndef CPRIMAT_WEAK_RATES_H
#define CPRIMAT_WEAK_RATES_H

#include "cprimat/config.h"
#include "cprimat/neutrino_history.h"
#include <stddef.h>

/* Fermi-Coulomb factor F(b) (Phys. Rep. S III.D); see corrections.FermiCoulomb. */
double cpr_fermi_coulomb(double b, const CPRConfig *cfg);

/* Resummed T=0 radiative correction R(b,y,en) (Phys. Rep. Eq. 101-105);
 * see corrections.RadCorrResum. b = v/c, y = E_nu/me, en = E_e/me. */
double cpr_rad_corr_resum(double b, double y, double en, const CPRConfig *cfg);

/* Neutron-decay phase-space integral Fn (Phys. Rep. Eq. 89-91); normalises
 * K = 1/(tau_n * Fn). See corrections.ComputeFn. */
double cpr_compute_fn(const CPRConfig *cfg);

/* The non-thermal rate table plus, when available, the separately-cached
 * thermal (CCRTh) correction -- mirrors RecomputeWeakRates's two-piece
 * return value, but stored as raw arrays (quadratic-interpolated on
 * evaluation, see cpr_interp_quadratic_local) rather than closures. Both
 * tables are in units of 1/tau_n (the caller multiplies by 1/cfg->tau_n,
 * or by the corresponding K, to get the physical rate in s^-1). */
typedef struct {
    double *T, *frwrd, *bkwrd; /* nonthermal table: T[K], Gamma_nTOp, Gamma_pTOn */
    size_t n;
    double *T_th, *Lnth, *Lpth; /* thermal correction table, only if has_thermal */
    size_t n_th;
    int has_thermal;
} CPRWeakRates;

/* Builds the n<->p weak-rate tables for the given background, mirroring
 * weak_rates.RecomputeWeakRates([Tg_vec, Tnu_vec], cfg, dFDneu_func=...).
 *
 * Tg_MeV/Tnu_MeV (length n_bg): photon and (electron-flavour) neutrino
 * temperatures in MeV, e.g. PyPR._setup_background_and_cosmo's Tg_vec/
 * Tnue_vec -- despite ComputeWeakRates's Python docstring saying "Kelvin",
 * background.py actually passes MeV arrays (_build_rate_context converts
 * via cfg.MeV_to_Kelvin); confirmed by reading the caller in background.py.
 * Used only to build the T_nu(T_gamma)/T_gamma ratio interpolant feeding
 * the rate integrands -- not stored.
 *
 * nh: neutrino history (cpr_neutrino_history_init), supplies the NEVO
 * spectral-distortion correction dFDneu when cfg->spectral_distortions.
 *
 * On a fingerprint cache hit (cfg->weak_rate_cache and a matching
 * data/weak/nTOp_<hash>.txt exists), the nonthermal table is loaded
 * directly (no integration). Otherwise it is computed via the
 * Gauss-Legendre rate integrals (Born/CCR/FM/SD) and, if cfg->save_nTOp,
 * written to that cache file. The thermal correction is loaded from
 * data/weak/nTOp_thermal_<hash>.txt when cfg->thermal_corrections is set
 * and that file exists (`has_thermal` is then 1); if thermal_corrections is
 * set but no matching file exists, this returns nonzero (the from-scratch
 * thermal computation is Phase 3b, not yet ported).
 *
 * Returns 0 on success (caller must cpr_weak_rates_free), nonzero with
 * *errmsg set (caller frees) otherwise. */
int cpr_weak_rates_init(CPRWeakRates *wr, const double *Tg_MeV, const double *Tnu_MeV,
                         size_t n_bg, const CPRConfig *cfg, const CPRNeutrinoHistory *nh,
                         char **errmsg);

void cpr_weak_rates_free(CPRWeakRates *wr);

/* Gamma_{n->p}(T)/Gamma_{p->n}(T) in units of 1/tau_n, the sum of the
 * (quadratic-interpolated) nonthermal table and, when present, the thermal
 * correction table -- mirrors RecomputeWeakRates's returned closures. */
double cpr_weak_rate_nTOp(const CPRWeakRates *wr, double T_K);
double cpr_weak_rate_pTOn(const CPRWeakRates *wr, double T_K);

#endif /* CPRIMAT_WEAK_RATES_H */
