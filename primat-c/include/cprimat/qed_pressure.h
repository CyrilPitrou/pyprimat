/* qed_pressure.h -- analytical QED plasma-pressure corrections (port of
 * pyprimat/qed_pressure.py). See that module's docstring for the physics
 * background; brief summary here:
 *
 * The finite-temperature QED interaction-pressure correction dP(T) to the
 * photon+e+- plasma is decomposed (Phys. Rep. S2.E, Eq. 47-49; PRIMAT-Main.m
 * dPa/dPe3) into:
 *
 *   dP_a(T)  [O(alpha)]      -- leading one-loop correction (Frenkel-
 *                                Galitskii-Migdal), the dominant (negative)
 *                                term.
 *   dP_e3(T) [O(alpha^3/2)]  -- ring/plasmon contribution (Blaizot-Zinn-
 *                                Justin), positive, ~10x smaller.
 *
 * (The O(alpha^2) two-loop exchange term dPb is not ported: PyPRIMAT's
 * shipped tables never include it, and CPLAN.md S7a scopes this module to
 * the pieces plasma.c actually loads.)
 *
 * Both terms are built from two Fermi-Dirac phase-space integrals, I01(x)
 * and I2m1(x) with x = me/T (cpr_qed_I01, cpr_qed_I2m1 below).
 *
 * cpr_qed_compute_tables evaluates dP_a, dP_e3 and their first two T-
 * derivatives (via a not-a-knot cubic spline fit, exactly mirroring
 * scipy.interpolate.CubicSpline's default in qed_pressure.py) on a log-
 * spaced T grid -- this is the "analytic fallback"/"recompute" path of
 * plasma.Plasma._load_tables; the normal "file mode" path instead reads
 * the pre-saved data/plasma/QED_*.txt files directly via table_io.c (no
 * qed_pressure.c involvement at all in that path, matching Python).
 */
#ifndef CPRIMAT_QED_PRESSURE_H
#define CPRIMAT_QED_PRESSURE_H

#include <stddef.h>

/* Fermi-Dirac phase-space integral I01(x) [dimensionless]:
 *
 *   I01(x) = int_0^inf p^2 / [sqrt(p^2+x^2) (exp(sqrt(p^2+x^2))+1)] dp
 *
 * (PRIMAT: Imn[1][0,1][x]). The p-space form is non-singular at p=0, so
 * plain adaptive quadrature over [0, P_UPPER] suffices -- no substitution
 * needed (matching qed_pressure._I01's choice of integration variable).
 * Returns 0 for x > 50 (T < me/50, where e+- are Boltzmann-suppressed
 * below double precision -- the same _X_NONREL_CUTOFF as Python). */
double cpr_qed_I01(double x);

/* Fermi-Dirac phase-space integral I2m1(x) [dimensionless]:
 *
 *   I2m1(x) = int_0^inf sqrt(p^2+x^2) / (exp(sqrt(p^2+x^2))+1) dp
 *
 * (PRIMAT: Imn[1][2,-1][x]). Same cutoff convention as cpr_qed_I01. */
double cpr_qed_I2m1(double x);

/* O(alpha) QED interaction-pressure correction dP_a(T) [MeV^4]:
 *
 *   dP_a = (alpha/pi) T^4 [-(2/3) I01(x) - (2/pi^2) I01(x)^2],   x = me/T
 *
 * (PRIMAT: dPa). Negative -- the dominant correction, lowering the
 * plasma pressure relative to the free e+- gas. */
double cpr_qed_dPa(double T, double alpha, double me);

/* O(alpha^3/2) QED interaction-pressure correction dP_e3(T) [MeV^4]:
 *
 *   dP_e3 = alpha^{3/2} (4/3) sqrt(2 pi) T^4 [(I01(x)+I2m1(x))/pi^2]^{3/2}
 *
 * (PRIMAT: dPe3). Positive; returns 0 if the bracketed combination is
 * non-positive (guards the fractional power, matching qed_pressure._dPe3's
 * `if combo <= 0: return 0.`). */
double cpr_qed_dPe3(double T, double alpha, double me);

/* Tabulated dP_a(T), dP_e3(T) and their first two T-derivatives on a log-
 * spaced grid of n points -- the layout plasma.c needs to rebuild the
 * PQEDofT/dPQEDdT/d2PQEDdT2 interpolants in "analytic fallback"/"recompute"
 * mode (see qed_pressure.h's top comment). All seven arrays have length n
 * and are owned by this struct (free with cpr_qed_tables_free). */
typedef struct {
    double *T;             /* grid [MeV], log-spaced T_min..T_max */
    double *dP_e2;          /* dP_a(T) [MeV^4] */
    double *dP_e3;          /* dP_e3(T) [MeV^4] */
    double *d_dP_e2_dT;     /* d(dP_a)/dT [MeV^3] */
    double *d_dP_e3_dT;     /* d(dP_e3)/dT [MeV^3] */
    double *d2_dP_e2_dT2;   /* d^2(dP_a)/dT^2 [MeV^2] */
    double *d2_dP_e3_dT2;   /* d^2(dP_e3)/dT^2 [MeV^2] */
    size_t n;
} CPRQEDTables;

/* Computes dP_a, dP_e3 on a log-spaced grid of n_pts points in
 * [T_min, T_max] [MeV], then differentiates each numerically via a not-
 * a-knot cubic spline fit (mirrors qed_pressure.compute_qed_pressure_tables;
 * see that function's docstring for why spline differentiation is
 * preferred over an analytic route: ~7x fewer quadratures, agrees with
 * finite differences to <0.01%). `alpha`/`me` are typically g_const.alphaem
 * / g_const.me (constants.h), passed explicitly here to keep this module
 * self-contained/standalone-testable like its Python counterpart. Returns
 * 0 on success (caller must cpr_qed_tables_free the result), nonzero with
 * *errmsg set (caller frees) if n_pts < 4 (not-a-knot's minimum). */
int cpr_qed_compute_tables(double T_min, double T_max, size_t n_pts,
                            double alpha, double me,
                            CPRQEDTables *out, char **errmsg);

void cpr_qed_tables_free(CPRQEDTables *t);

/* Writes the three data/plasma/QED_*.txt files (QED_P_int.txt,
 * QED_dP_intdT.txt, QED_d2P_intdT2.txt) in the same 3-column (T, e^2-order,
 * e^3-order) whitespace-separated format Python's save_qed_tables produces
 * (and table_io.c/plasma.c's "file mode" loader reads back) -- so a
 * recompute-and-save cycle through this function is byte-for-byte
 * interchangeable with the Python one. `plasma_dir` is the path to
 * data/plasma/ (no trailing slash required). Returns 0 on success,
 * nonzero with *errmsg set (caller frees) on a file-write failure. */
int cpr_qed_save_tables(const CPRQEDTables *t, const char *plasma_dir, char **errmsg);

#endif /* CPRIMAT_QED_PRESSURE_H */
