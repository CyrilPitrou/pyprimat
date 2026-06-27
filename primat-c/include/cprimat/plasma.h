/* plasma.h -- SM plasma thermodynamics (port of primat/plasma.py).
 *
 * Photon, e+-, and neutrino thermodynamics used in the background a(T)/
 * t(T) evolution and Friedmann equation during BBN. All quantities follow
 * the Fermi-Dirac/Bose-Einstein integrals of Phys. Rep. App. A (Eq. A1-A7):
 *
 *   rho = g T^4/(2 pi^2) I+-^{(2,1)}(x)      (Eq. A4b)
 *   p   = g T^4/(6 pi^2) I+-^{(0,3)}(x)      (Eq. A4c)
 *   s   = (rho + p) / T                      (Eq. 21/24)
 *
 * with x = m/T, g the spin degeneracy (g=2 photons, g=4 e+-, g=1 nu).
 *
 * QED interaction-pressure corrections (Phys. Rep. S2.E, Eq. 47-49) are
 * loaded from data/plasma/QED_*.txt (see qed_pressure.h) when present, or
 * computed analytically on the fly otherwise -- exactly the three modes
 * documented on plasma.Plasma._load_tables (file / analytic-fallback /
 * recompute).
 *
 * CPRPlasma is a per-instance bundle exactly like Python's Plasma class:
 * one CPRPlasma per CPRConfig, no shared mutable module state, so e.g. a
 * QED_corrections=True and a QED_corrections=False instance never
 * interfere with each other (cfg.h's own comment on this design choice
 * applies equally here).
 */
#ifndef CPRIMAT_PLASMA_H
#define CPRIMAT_PLASMA_H

#include "cprimat/config.h"
#include "cprimat/spline.h"

/* ---------------------------------------------------------------------
 * Photons and SM neutrinos: pure functions of temperature, no cfg
 * dependence (mirrors plasma.py's module-level rho_g/drho_g_dT/rho_nu/
 * drho_nu_dT).
 * ------------------------------------------------------------------- */

/* Photon energy density [MeV^4]: rho_g = (pi^2/15) Tg^4 (Phys. Rep. Eq. 25b). */
double cpr_rho_g(double Tg);
/* d(rho_g)/dTg [MeV^3] = 4 rho_g / Tg. */
double cpr_drho_g_dT(double Tg);
/* Energy density of one SM neutrino flavour (nu+nubar) [MeV^4]:
 * rho_nu = 2 (7/8) (pi^2/30) Tnu^4 (Phys. Rep. Eq. 26b). */
double cpr_rho_nu(double Tnu);
/* d(rho_nu)/dTnu [MeV^3] = 4 rho_nu / Tnu. */
double cpr_drho_nu_dT(double Tnu);
/* Extra (nu+nubar) energy density [MeV^4] PER FLAVOUR from a genuine reduced
 * neutrino chemical potential c = mu/Tnu (antineutrino carries -c). The energy
 * density rho(c) = Tnu^4 (7pi^2/120 + c^2/4 + c^4/(8pi^2)) is even in c; this
 * returns just the excess over c=0, Tnu^4 (c^2/4 + c^4/(8pi^2)). The caller
 * sums it over the three flavours. Mirrors primat.plasma.rho_nu_chempot_excess.
 * A neutrino chemical potential is NOT a spectral distortion: this energy feeds
 * the expansion rate / Neff directly. */
double cpr_rho_nu_chempot_excess(double Tnu, double c);

/* ---------------------------------------------------------------------
 * Per-config plasma instance.
 * ------------------------------------------------------------------- */

/* A generic 1D interpolant used for the QED dP/dT-derivative callables:
 * either piecewise-linear over loaded-from-file arrays (the "file mode"
 * path, matching interp1d(kind='linear', fill_value="extrapolate")), or a
 * not-a-knot cubic spline built from freshly computed arrays (the
 * "analytic fallback"/"recompute" path, matching
 * scipy.interpolate.CubicSpline -- smoother than the file-mode linear
 * interpolant, exactly as Python's _load_tables comments explain). */
typedef struct {
    int is_spline;
    double *x, *y;       /* owned; used when !is_spline */
    size_t n;
    CPRCubicSpline spl;   /* used when is_spline */
} CPRInterp1D;

double cpr_interp1d_eval(const CPRInterp1D *itp, double xq);
void cpr_interp1d_free(CPRInterp1D *itp);

typedef struct {
    const CPRConfig *cfg; /* borrowed; must outlive this CPRPlasma */

    /* QED interaction-pressure correction dP(Tg) and its first two
     * Tg-derivatives [MeV^4, MeV^3, MeV^2]. When cfg->QED_corrections is
     * false, qed_active is 0 and all three evaluate to exactly 0 without
     * touching the interpolants below (mirrors the Python "lambda T: 0."
     * zero-function shortcut). */
    int qed_active;
    CPRInterp1D P_QED, dP_QED, d2P_QED;

    /* e+- thermodynamics: cubic-spline interpolants over a log-Tg grid
     * (mirrors interp1d(kind='cubic'), built once in cpr_plasma_init via
     * _build_electron_tables's on-disk fingerprinted cache or a fresh
     * quadrature pass). */
    CPRCubicSpline rho_e_tab, p_e_tab, drho_e_dT_tab, dp_e_dT_tab;
} CPRPlasma;

/* Builds every table/interpolant cfg needs (QED pressure corrections,
 * e+- thermodynamics), following exactly the three QED modes and the
 * electron-thermo fingerprinted-cache logic of plasma.Plasma.__init__.
 * Returns 0 on success (caller must cpr_plasma_free the result), nonzero
 * with *errmsg set (caller frees) on a hard failure (e.g. a present-but-
 * malformed QED/cache file; a missing file is not an error -- it falls
 * back to the analytic/recompute path exactly like Python). */
int cpr_plasma_init(CPRPlasma *pl, const CPRConfig *cfg, char **errmsg);

void cpr_plasma_free(CPRPlasma *pl);

/* e+- energy density [MeV^4] (Phys. Rep. Eq. A4b, g=4, x=me/Tg). Exactly
 * 0 for Tg < me/30 (Boltzmann-suppressed below double precision -- see
 * _ELEC_THERMO_LOWT_RATIO in plasma.c). */
double cpr_plasma_rho_e(const CPRPlasma *pl, double Tg);
double cpr_plasma_drho_e_dT(const CPRPlasma *pl, double Tg);
/* e+- pressure [MeV^4] (Phys. Rep. Eq. A4c). */
double cpr_plasma_p_e(const CPRPlasma *pl, double Tg);
double cpr_plasma_dp_e_dT(const CPRPlasma *pl, double Tg);

/* Energy density of DeltaNeff extra decoupled relativistic species
 * [MeV^4]; 0 when cfg->DeltaNeff == 0. */
double cpr_plasma_rho_nu_extra(const CPRPlasma *pl, double Tg);

/* Total SM energy/pressure density during BBN [MeV^4] (Phys. Rep.
 * Eq. 43): photons + e+- + QED correction + 3 nu flavours (Tnue for
 * nu_e, Tnumu shared by nu_mu/nu_tau) + DeltaNeff extra species. */
double cpr_plasma_rho_SM(const CPRPlasma *pl, double Tg, double Tnue, double Tnumu);
double cpr_plasma_p_SM(const CPRPlasma *pl, double Tg, double Tnue, double Tnumu);

/* EM plasma (photons + e+- + QED) entropy density [MeV^3] (Phys. Rep.
 * Eq. 21/24/30): spl = (rho_pl + p_pl)/Tg. */
double cpr_plasma_spl(const CPRPlasma *pl, double Tg);

/* Computes spl and dspl/dTg together (sharing intermediate e+-/QED
 * evaluations -- mirrors spl_and_dspl_dT, the efficient combined path
 * background.c will call from its a(T)-ODE right-hand side). */
void cpr_plasma_spl_and_dspl_dT(const CPRPlasma *pl, double Tg, double *s, double *ds_dT);
double cpr_plasma_dspl_dT(const CPRPlasma *pl, double Tg);

/* Neutrino temperature in the instantaneous-decoupling limit [MeV]
 * (Phys. Rep. Eqs. 30-33): Tnu(Tg) = Tg (spl(Tg)/(sigma_inf Tg^3))^{1/3}.
 * Self-consistent only when cfg->QED_corrections is false -- see the
 * caveat on plasma.Plasma.T_nu_decoupling's docstring. */
double cpr_plasma_T_nu_decoupling(const CPRPlasma *pl, double Tg);

#endif /* CPRIMAT_PLASMA_H */
