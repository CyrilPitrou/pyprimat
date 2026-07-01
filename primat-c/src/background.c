/* background.c -- see cprimat/background.h.
 *
 * Reference: Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095),
 * cited below as "Phys. Rep.".
 */
#include "background.h"
#include "constants.h"
#include "spline.h"
#include "ode_rk.h"
#include "log.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

/* Riemann zeta(3) (Apery's constant) -- see constants.c's identical
 * literal for cpr_n0CMB(); duplicated here (rather than exposed from
 * constants.h) since it is needed only by the two small Omeganuh2_*
 * closures below. */
#define ZETA3 1.2020569031595942854

/* Noise floor for the raw n<->p weak rates, in 1/tau_n units -- mirrors
 * background.py's _WEAK_RATE_FLOOR. Below this scale the rate is
 * exp(-Q/T)-suppressed and the sum of phase-space correction terms has
 * lost all significant digits to cancellation (alternating sign around
 * zero); the quadratic interpolant used by cpr_weak_rate_nTOp/pTOn can
 * also overshoot slightly negative between cached nodes even when every
 * node value is >= 0. A physical rate is never negative, so both that
 * interpolation noise and any genuine sub-floor value are replaced by 0. */
#define WEAK_RATE_FLOOR 1.0e-28

/* Fixed (not cfg->numerical_precision-derived) tolerance for the
 * background ODE(s) below (the combined a(T)/t(T) 2D ODE, Branch E) --
 * see the first call site's comment for the empirical justification
 * (closes a ~1-3% end-to-end BBN-abundance gap traced to RK45-vs-LSODA
 * discretization at the Python-nominal tolerance).
 *
 * Historically this had to be set as tight as 1e-15 because the solution
 * at the desired output grid was obtained by *piecewise-linear*
 * interpolation of the accepted RK45 steps: with a loose tolerance the
 * stepper takes large steps, and O(h^2) linear-interpolation error
 * between sparse points corrupted a(T)/t(a) downstream. Forcing rtol/atol
 * to 1e-15 hid that interpolation error at the cost of a hugely
 * oversampled stepper.
 *
 * Branch A: cpr_ode_rk45 now supports dense-output evaluation (ode_rk.h,
 * `dense_cb`/cpr_ode_dense_eval), used by setup_background_and_cosmo's
 * CPRDensePath to evaluate the solution at any query point via the
 * Dormand-Prince continuous extension polynomial -- locally as accurate
 * as the step itself, no separate interpolation error to hide. This lets
 * rtol/atol be loosened by ~5 orders of magnitude: empirically, 1e-7
 * through 1e-12 were tried against the small/large+amax=8 reference
 * numbers (CLAUDE.md's "Validation before committing"); 1e-7 already
 * nudges D/H outside CLAUDE.md's +-3e-9 bound, while 1e-8 through 1e-12
 * stay comfortably within +-1.2e-9. 1e-10 is chosen as a safety margin
 * below that edge rather than chasing the exact breakeven, since the
 * combined ODE is cheap regardless (2-D, smooth, dense-output already
 * removing the dominant cost) and the BBN-abundance sensitivity to
 * a(T)/t(a) discretization error documented above leaves little room for
 * complacency. */
#define BG_ODE_RTOL 1.0e-10
#define BG_ODE_ATOL 1.0e-10

static double clamp_raw_weak_rate(double rate)
{
    return (rate < WEAK_RATE_FLOOR) ? 0.0 : rate;
}

/* Mirrors weak_rates.c's identical static n_points_per_decade (not
 * exposed via weak_rates.h, since it is a generic small grid-sizing
 * helper rather than weak-rate-specific physics) -- number of log-spaced
 * grid points spanning [T_lo, T_hi] at `per_decade` points per decade,
 * floored at 2. */
static int n_points_per_decade(double per_decade, double T_lo, double T_hi)
{
    double decades = log10(T_hi / T_lo);
    int n = (int)lround(per_decade * decades);
    return n < 2 ? 2 : n;
}

/* ---------------------------------------------------------------------
 * QED plasma-pressure correction accessors. plasma.h stores P_QED/dP_QED
 * as raw CPRInterp1D fields (qed_active gates them to exactly 0, mirroring
 * the Python "lambda T: 0." shortcut in plasma.Plasma._load_tables) rather
 * than behind PQEDofT/dPQEDdT-named wrappers, so background.c -- the only
 * other module that needs them, via Hubble's plasma EOS term -- supplies
 * its own thin accessors here.
 * ------------------------------------------------------------------- */
static double bg_PQEDofT(const CPRPlasma *pl, double Tg)
{
    return pl->qed_active ? cpr_interp1d_eval(&pl->P_QED, Tg) : 0.0;
}
static double bg_dPQEDdT(const CPRPlasma *pl, double Tg)
{
    return pl->qed_active ? cpr_interp1d_eval(&pl->dP_QED, Tg) : 0.0;
}

/* ---------------------------------------------------------------------
 * Growable per-step recorder for the background ODEs below (the combined
 * a(T)/t(T) 2D ODE, and the external_scale_factor mode's 1D t(a) ODE):
 * stores the RAW Dormand-Prince dense-output ingredients (y_old, hh,
 * k1,k3,k4,k5,k6,k7) for every accepted step, via ode_rk.h's dense_cb
 * mechanism, so the solution can be evaluated AFTER the solve completes at
 * full dense-output accuracy (locally as accurate as the step itself -- no
 * separate, lower-order interpolation error on top of the ODE's own
 * accuracy) at query points that are not necessarily known in advance
 * (unlike ode_rk.h's t_eval/y_eval, which needs the whole grid up front).
 * This is needed here because the second output grid this file queries
 * (bg->t_vec, over a uniform-in-log(a) grid) is only determined AFTER the
 * first grid (bg->lnT_sol) has been solved and inverted via T_of_a -- see
 * combined_bg_rhs's docstring below. Both ODEs integrate strictly
 * ascending (dir=+1), so steps are recorded in increasing t0 order and a
 * plain binary search brackets any query. */
typedef struct {
    double *t0, *hh;              /* per-step: start time, step span; length nsteps */
    double *y_old, *k1, *k3, *k4, *k5, *k6, *k7; /* per-step, n components each; length nsteps*n */
    size_t n;                      /* ODE dimension (1 or 2 here) */
    size_t nsteps, cap;
} CPRDensePath;

static void dense_path_init(CPRDensePath *p, size_t n)
{
    memset(p, 0, sizeof(*p));
    p->n = n;
}

static void dense_path_push(double t0, double hh, const double *y_old,
                              const double *k1, const double *k3, const double *k4,
                              const double *k5, const double *k6, const double *k7,
                              size_t n, void *ctx)
{
    CPRDensePath *p = ctx;
    (void)n; /* == p->n by construction (one CPRDensePath per ODE dimension) */
    if (p->nsteps == p->cap) {
        p->cap = p->cap ? p->cap * 2 : 64;
        p->t0 = realloc(p->t0, p->cap * sizeof(double));
        p->hh = realloc(p->hh, p->cap * sizeof(double));
        p->y_old = realloc(p->y_old, p->cap * p->n * sizeof(double));
        p->k1 = realloc(p->k1, p->cap * p->n * sizeof(double));
        p->k3 = realloc(p->k3, p->cap * p->n * sizeof(double));
        p->k4 = realloc(p->k4, p->cap * p->n * sizeof(double));
        p->k5 = realloc(p->k5, p->cap * p->n * sizeof(double));
        p->k6 = realloc(p->k6, p->cap * p->n * sizeof(double));
        p->k7 = realloc(p->k7, p->cap * p->n * sizeof(double));
    }
    size_t i = p->nsteps;
    p->t0[i] = t0;
    p->hh[i] = hh;
    memcpy(&p->y_old[i * p->n], y_old, p->n * sizeof(double));
    memcpy(&p->k1[i * p->n], k1, p->n * sizeof(double));
    memcpy(&p->k3[i * p->n], k3, p->n * sizeof(double));
    memcpy(&p->k4[i * p->n], k4, p->n * sizeof(double));
    memcpy(&p->k5[i * p->n], k5, p->n * sizeof(double));
    memcpy(&p->k6[i * p->n], k6, p->n * sizeof(double));
    memcpy(&p->k7[i * p->n], k7, p->n * sizeof(double));
    p->nsteps++;
}

static void dense_path_free(CPRDensePath *p)
{
    free(p->t0); free(p->hh); free(p->y_old);
    free(p->k1); free(p->k3); free(p->k4); free(p->k5); free(p->k6); free(p->k7);
    memset(p, 0, sizeof(*p));
}

/* Evaluates the recorded path at `tq` (anywhere in [t0[0], t0[last]+hh[last]])
 * via the exact Dormand-Prince dense-output polynomial for the bracketing
 * step, writing `p->n` components into `out` (caller-allocated). Binary
 * search for the bracketing step (t0 ascending, since both ODEs using this
 * integrate with dir=+1); theta is clamped to [0,1] to absorb roundoff at
 * the very first/last query (mirrors cpr_ode_rk45's own t_eval `eps`
 * handling of the same boundary-roundoff issue). */
static void dense_path_eval(const CPRDensePath *p, double tq, double *out)
{
    size_t lo = 0, hi = p->nsteps - 1;
    while (lo < hi) {
        size_t mid = (lo + hi + 1) / 2;
        if (p->t0[mid] <= tq) lo = mid; else hi = mid - 1;
    }
    double theta = (tq - p->t0[lo]) / p->hh[lo];
    if (theta < 0.0) theta = 0.0;
    if (theta > 1.0) theta = 1.0;
    cpr_ode_dense_eval(theta, p->hh[lo], &p->y_old[lo * p->n], &p->k1[lo * p->n],
                        &p->k3[lo * p->n], &p->k4[lo * p->n], &p->k5[lo * p->n],
                        &p->k6[lo * p->n], &p->k7[lo * p->n], p->n, out);
}

/* ---------------------------------------------------------------------
 * StandardBackground: Lambda-CDM + EDE setup.
 * ------------------------------------------------------------------- */

/* _setup_LCDM (Phys. Rep. background; see background.py's docstring for
 * the full derivation): comoving CDM amplitude rhocdm_a3 = Omega_c h^2 *
 * rhocrit100 (rho_CDM = rhocdm_a3/a^3) and the flatness-condition
 * cosmological constant rholambda = (h^2 - Omegabh2 - Omegach2)*rhocrit100.
 *
 * Deviation from Python: CPRConfig's Omegach2/h fields are plain (non-
 * nullable) doubles -- there is no C analogue of Python's "Omegach2 is
 * None disables this contribution" escape hatch, since DEFAULT_PARAMS
 * itself ships a concrete float for both (0.11933/0.6766), making that
 * branch already dead code under any config reachable through
 * cpr_config_set_by_name. has_lcdm is therefore unconditionally 1. */
static void setup_lcdm(CPRBackground *bg)
{
    const CPRConfig *cfg = bg->cfg;
    double rhocrit100 = cpr_config_rhocOverh2(cfg);

    bg->has_lcdm = 1;
    bg->rhocdm_a3 = cfg->Omegach2 * rhocrit100;

    double Omegalambdah2 = cfg->h * cfg->h - cpr_config_get_Omegabh2(cfg) - cfg->Omegach2;
    bg->rholambda = Omegalambdah2 * rhocrit100;
    if (Omegalambdah2 < 0.0) {
        fprintf(stderr,
                "[bckg] warning: Omega_Lambda h^2 = %.4g < 0 (h=%g, Omegabh2=%g, "
                "Omegach2=%g). Cosmological constant is negative -- non-standard "
                "cosmology.\n",
                Omegalambdah2, cfg->h, cpr_config_get_Omegabh2(cfg), cfg->Omegach2);
    }
}

/* _setup_EDE: builds rho_EDE(Tg) = 2*rhocEDEac / (1 + (TcEDE/Tg)^(3*wnEDE+3)),
 * active only when cfg->fEDE != 0 (Phys. Rep.-adjacent EDE parametrisation;
 * see background.py's _setup_EDE docstring for the acEDE/amaxEDE/TmaxEDE/
 * TcEDE/rhocEDEac derivation, reproduced verbatim below). */
static void setup_ede(CPRBackground *bg)
{
    const CPRConfig *cfg = bg->cfg;
    bg->has_ede = (cfg->fEDE != 0.0);
    if (!bg->has_ede) return;

    double acEDE = 1.0 / (1.0 + cfg->zcEDE);
    double amaxEDE = acEDE * pow(4.0 / (3.0 * cfg->wnEDE - 1.0), 1.0 / (3.0 * cfg->wnEDE + 3.0));
    double T0CMB_MeV = g_const.T0CMB / cpr_MeV_to_Kelvin();
    double TmaxEDE = T0CMB_MeV / amaxEDE;
    bg->TcEDE = T0CMB_MeV / acEDE;

    bg->rhocEDEac = (cfg->fEDE / (1.0 - cfg->fEDE)
                      * cpr_rho_g(TmaxEDE)
                      * (1.0 + g_const.Neff_SM * 7.0 / 8.0 * pow(4.0 / 11.0, 4.0 / 3.0))
                      / 2.0
                      * (1.0 + 4.0 / (3.0 * cfg->wnEDE - 1.0)));
    bg->EDE_exponent = 3.0 * cfg->wnEDE + 3.0;
}

/* ---------------------------------------------------------------------
 * Friedmann expansion rate (StandardBackground.Hubble).
 * ------------------------------------------------------------------- */
double cpr_bg_Hubble(const CPRBackground *bg, double Tg, double Tnue, double Tnumu,
                      double Tnutau, double a)
{
    const CPRConfig *cfg = bg->cfg;
    const CPRPlasma *thermo = bg->plasma;

    double rho_pl = cpr_rho_g(Tg) + cpr_plasma_rho_e(thermo, Tg)
                     - bg_PQEDofT(thermo, Tg) + Tg * bg_dPQEDdT(thermo, Tg);
    double rho_3nu = cpr_rho_nu(Tnue) + cpr_rho_nu(Tnumu) + cpr_rho_nu(Tnutau);
    /* Genuine neutrino chemical potential: raises the neutrino energy density
     * (each flavour by Tnu^4 (xi^2/4 + xi^4/(8pi^2)); antineutrino carries
     * -xi). Mirrors primat Plasma.rho_nu. It also shifts the n<->p weak rates
     * (handled in weak_rates.c via the FD_nu3 integrand). NOT a spectral
     * distortion (that's rho_nu_SD just below). */
    if (cfg->munuOverTnu != 0.0) {
        double xi = cfg->munuOverTnu;
        rho_3nu += cpr_rho_nu_chempot_excess(Tnue, xi)
                 + cpr_rho_nu_chempot_excess(Tnumu, xi)
                 + cpr_rho_nu_chempot_excess(Tnutau, xi);
    }
    /* Analytic y/gray-type spectral-distortion extra energy density
     * (Python's self.rho_nu_SD term, AnalyticDistortion-only -- the
     * NEVO-table distortion needs no such correction, see
     * cpr_nu_rho_nu_SD's doc comment). Energy-weighted mean flavour
     * temperature, mirroring background.py's Tnu_avg. */
    if (bg->nh.has_analytic_distortion) {
        double Tnu_avg = pow((pow(Tnue, 4.0) + pow(Tnumu, 4.0) + pow(Tnutau, 4.0)) / 3.0, 0.25);
        rho_3nu += cpr_nu_rho_nu_SD(&bg->nh, Tnu_avg);
    }
    double rho_tot = rho_pl + rho_3nu + cpr_plasma_rho_nu_extra(thermo, Tg);

    if (bg->has_lcdm) {
        /* `a` is always supplied exactly by the caller now (Branch E: the
         * combined a(T)/t(T) ODE below always carries x=ln(a*T) in its own
         * state, so a=exp(x-lnT) is recovered analytically at every RHS
         * evaluation -- no "not yet built" a(T) to bootstrap around, unlike
         * the old sequential two-ODE scheme this replaced). */
        rho_tot += bg->rhocdm_a3 / (a * a * a) + bg->rholambda;
    }
    if (bg->has_ede) {
        rho_tot += 2.0 * bg->rhocEDEac / (1.0 + pow(bg->TcEDE / Tg, bg->EDE_exponent));
    }
    /* Analytic mu/y-type spectral-distortion extra energy density
     * (Python's self.rho_nu_SD term) is out of scope (CPLAN.md S0) and
     * always inactive here -- see neutrino_history.h's top comment. */

    return cpr_MeV_to_secm1() * sqrt(rho_tot * 8.0 * M_PI / (3.0 * cpr_config_Mpl(cfg) * cpr_config_Mpl(cfg)));
}

/* ---------------------------------------------------------------------
 * StandardBackground: a(T)/t(a) ODEs + weak rates.
 * ------------------------------------------------------------------- */

/* d(ln a)/d(ln T) = -(3 sbar + T dsbar/dT) / (N_NEVO + 3 sbar), the EM
 * entropy-conservation ODE driving a(T_gamma) (Phys. Rep. S2; see
 * background.py's _setup_background_and_cosmo docstring, "minimal mode").
 * Used directly only by combined_bg_rhs below now (Branch E); kept as a
 * free-standing helper rather than inlined since its formula/derivation
 * comment is referenced from there. */
static double dlnadlnT_value(CPRBackground *bg, double T)
{
    double s, ds_dT;
    cpr_plasma_spl_and_dspl_dT(bg->plasma, T, &s, &ds_dT);
    double sb = s / (T * T * T);
    double dsbdT = ds_dT / (T * T * T) - 3.0 * s / (T * T * T * T);
    double N = cpr_nu_N_NEVO_of_Tg(&bg->nh, T);
    return -(3.0 * sb + T * dsbdT) / (N + 3.0 * sb);
}

/* `T_of_a_smooth` is a *local-only* cubic spline over the same
 * (a_sol_asc, T_sol_asc) nodes `cpr_bg_T_of_a` itself interpolates linearly
 * (background.h's docstring: T_of_a is "always a linear interpolant",
 * matching Python's interp1d default `kind='linear'` -- that choice is
 * kept for every *public* query). Using the public, piecewise-linear
 * cpr_bg_T_of_a as this ODE's RHS, however, feeds the 5th-order adaptive
 * RK45 stepper a function with a curvature kink at every one of the
 * ~O(sampling_temperature_per_decade * decades) grid nodes; profiling
 * (a temporary instrumented build counting accepted/rejected steps) showed
 * a ~65% step-rejection rate *uniformly spread across the whole
 * integration range* -- the signature of a stepper repeatedly hitting a
 * kink, not of one genuinely stiff region -- making this single ODE ~40x
 * more expensive (in accepted+rejected RHS evaluations) than the
 * similarly-sized a(T) ODE just above, and the dominant cost (60%+) of
 * cpr_bg_init_standard. Swapping in a not-a-knot cubic spline (smooth
 * second derivative) for *this RHS evaluation only* removes the kinks the
 * stepper was fighting, without touching `cpr_bg_T_of_a`'s own linear
 * behaviour or this ODE's accuracy contract (BG_ODE_RTOL/ATOL, the
 * solution itself, are unchanged -- only how cheaply the same tolerance is
 * reached).
 *
 * Branch E note: this dtdlna_rhs/T_of_a_smooth pair is now used ONLY for
 * cfg->external_scale_factor=True, where a(T) is already a closed-form
 * algebraic function of T (no entropy ODE to combine with -- see
 * setup_background_and_cosmo's external_scale_factor branch) so there is
 * no "first ODE" to fold dt/d(ln a) into; the combined 2D ODE below
 * (combined_bg_rhs) replaces this pair for the default (minimal,
 * !external_scale_factor) mode, where both a(T) and t(T) genuinely come
 * from ODEs and can share one integration. */
typedef struct { CPRBackground *bg; const CPRCubicSpline *T_of_a_smooth; } DtDlnaCtx;

/* dt/d(ln a) = 1/H(a), with T(a) read from the just-built a(T) inverse
 * (Phys. Rep. background time integration). */
static int dtdlna_rhs(double lna, const double *y, double *ydot, void *ctx_)
{
    (void)y;
    DtDlnaCtx *c = ctx_;
    double a = exp(lna);
    double Tg = cpr_cubic_spline_eval(c->T_of_a_smooth, a);
    double Tnue = cpr_nu_Tnue_of_Tg(&c->bg->nh, Tg);
    double Tnumu = cpr_nu_Tnumu_of_Tg(&c->bg->nh, Tg);
    double Tnutau = cpr_nu_Tnutau_of_Tg(&c->bg->nh, Tg);
    ydot[0] = 1.0 / cpr_bg_Hubble(c->bg, Tg, Tnue, Tnumu, Tnutau, a);
    return 0;
}

/* ---------------------------------------------------------------------
 * Branch E: combined a(T)/t(T) 2D ODE (minimal mode, !external_scale_factor).
 *
 * State y[0] = x = ln(a*T), y[1] = trel (a *relative*, uncalibrated cosmic
 * time -- see "Boundary condition" below), both integrated over the SAME
 * independent variable lnT -- instead of the old two sequential ODEs
 * (d(ln a)/d(ln T) over lnT, then dt/d(ln a) over lna) bridged by the
 * not-a-knot cubic spline T_of_a_smooth above. This eliminates that spline
 * build/free and the second cpr_ode_rk45 call entirely: `a` is recovered
 * analytically at every RHS evaluation as a = exp(y[0] - lnT) (no lookup,
 * no spline), so H(T,a) -- needed by the t-component RHS -- is always
 * evaluated exactly (see cpr_bg_Hubble's explicit `a` parameter).
 *
 * Variable choice for y[0]: x = ln(a*T), rather than the simpler
 * y[0] = ln(a), is used because a*T is O(1)-ish throughout the integration
 * (entropy conservation keeps a*T close to its asymptotic z0 value,
 * drifting only by a relative O(10%) across the e+e- annihilation/
 * neutrino-decoupling era), whereas ln(a) alone spans a huge dynamic range.
 * This keeps x's own RK45 error control (ode_rk.c's per-component scale =
 * atol + rtol*|y_i|, NOT a single shared scale across components) as
 * meaningful at the fixed BG_ODE_ATOL used here as the original
 * dlnadlnT_rhs's own y=ln(a) state.
 *
 * Variable choice for y[1]: deliberately raw cosmic time, NOT tau=ln(t).
 * The natural candidate boundary condition for t is the standard
 * radiation-domination relation t_ini = 1/(2 H(T_start_cosmo)), valid deep
 * in the early, fully relativistic universe -- but T_start_cosmo's
 * *default* value is not always far enough above the shipped NEVO heating
 * table's support for the algebraic, N=0 closed-form entropy-conservation
 * shortcut used for a_end below to also give an exact a_ini at
 * T_start_cosmo (unlike T_end, which IS safely below the table's low edge,
 * so a_end's shortcut is exact). Anchoring tau=ln(t) at T_start_cosmo via
 * that same shortcut would then carry a small systematic error at exactly
 * the boundary condition.
 *
 * Using *raw* t instead sidesteps this: dt/d(ln T) = dt/d(ln a) *
 * d(ln a)/d(ln T) = (1/H) * dlna_dlnT has NO dependence on the current
 * value of t itself (unlike d(ln t)/d(ln T), which divides by t), so this
 * component's ODE is exactly LINEAR in t: any two solutions differing only
 * by their initial condition differ by the SAME additive constant at every
 * lnT. This means y[1] can be integrated from an arbitrary placeholder (0
 * here) anchored at T_end -- the SAME point x is exactly anchored at
 * (a_end, algebraic, N_NEVO(T_end) == 0) -- alongside x in one single ODE
 * call, and the genuinely correct, absolute t(T) recovered AFTERWARDS by a
 * trivial O(1) additive shift: once the pass is solved, a_ini =
 * a(T_start_cosmo) is read off EXACTLY from the just-solved x trajectory
 * (no algebraic approximation needed at T_start_cosmo any more), giving an
 * exact t_ini = 1/(2 H(T_start_cosmo, a_ini)); the shift
 * C = t_ini - trel(T_start_cosmo) then makes t(lnT) = trel(lnT) + C match
 * the original code's t(a) ODE bit-for-bit in its boundary condition (same
 * t_ini formula, same exact a_ini), while still requiring only ONE
 * cpr_ode_rk45 call. See setup_background_and_cosmo for where C is
 * computed and applied.
 *
 * (The raw-vs-log choice for y[1] has no adverse conditioning effect on the
 * RK45 step-size controller despite t spanning many decades: ode_rk.c's
 * error scale is per-component, scale_i = atol + rtol*|y_i|, so each
 * component is normalised by its OWN magnitude independently -- there is
 * no single shared scale across x and trel for a magnitude mismatch to
 * degrade.)
 * ------------------------------------------------------------------- */
typedef struct { CPRBackground *bg; } CombinedBgCtx;

static int combined_bg_rhs(double lnT, const double *y, double *ydot, void *ctx_)
{
    CombinedBgCtx *c = ctx_;
    CPRBackground *bg = c->bg;
    double T = exp(lnT);

    /* Same entropy-conservation RHS as the original dlnadlnT_rhs (a-
     * independent -- only T enters, via the EM plasma's reduced entropy
     * sbar=s/T^3 and the NEVO heating function N). */
    double dlna_dlnT = dlnadlnT_value(bg, T);

    /* x = ln(a*T) => dx/d(ln T) = d(ln a)/d(ln T) + 1. */
    ydot[0] = dlna_dlnT + 1.0;

    /* a recovered analytically from the state -- the key simplification
     * that lets this RHS evaluate H(T,a) exactly without a T(a) lookup. */
    double a = exp(y[0] - lnT);

    double Tnue   = cpr_nu_Tnue_of_Tg(&bg->nh, T);
    double Tnumu  = cpr_nu_Tnumu_of_Tg(&bg->nh, T);
    double Tnutau = cpr_nu_Tnutau_of_Tg(&bg->nh, T);
    double H = cpr_bg_Hubble(bg, T, Tnue, Tnumu, Tnutau, a);

    /* trel: dt/d(ln T) = dt/d(ln a) * d(ln a)/d(ln T) = (1/H) * dlna_dlnT
     * (chain rule; dt/d(ln a) = 1/H is the original dtdlna_rhs's RHS). No
     * t-dependence on the right side -- see the variable-choice comment
     * above for why this is exactly what makes the post-hoc shift valid. */
    ydot[1] = dlna_dlnT / H;
    return 0;
}

/* StandardBackground._setup_background_and_cosmo + _setup_derived_cosmo +
 * _setup_weak_rates, folded into one function (no separate eager "derived
 * cosmo" step is needed in C -- cpr_bg_Omeganuh2_relnu/nrnu read
 * bg->Tg_vec/Tnu_vec directly at call time, exactly mirroring Python's
 * closures over the same arrays). */
static int setup_background_and_cosmo(CPRBackground *bg, char **errmsg)
{
    const CPRConfig *cfg = bg->cfg;
    const CPRPlasma *thermo = bg->plasma;

    double Tstartcosmo = cpr_config_T_start_cosmo(cfg) / cpr_MeV_to_Kelvin(); /* [MeV] */
    double Tend = cpr_config_T_end(cfg) / cpr_MeV_to_Kelvin();                 /* [MeV] */

    int n_T_pts = n_points_per_decade(cfg->sampling_temperature_per_decade, Tend, Tstartcosmo);
    bg->n_Tsol = (size_t)n_T_pts;
    bg->lnT_sol = malloc(bg->n_Tsol * sizeof(double));
    bg->lna_sol = malloc(bg->n_Tsol * sizeof(double));
    double *T_sol = malloc(bg->n_Tsol * sizeof(double));
    for (size_t i = 0; i < bg->n_Tsol; i++) {
        double frac = (bg->n_Tsol == 1) ? 0.0 : (double)i / (double)(bg->n_Tsol - 1);
        bg->lnT_sol[i] = log(Tend) + frac * (log(Tstartcosmo) - log(Tend));
        T_sol[i] = exp(bg->lnT_sol[i]);
    }

    /* Boundary value a(Tend) = zend/Tend from algebraic entropy
     * conservation (no ODE needed for this single point): z0 is the CMB
     * photon temperature [MeV] today, sbar(T)=spl(T)/T^3 the EM plasma's
     * reduced entropy density (Phys. Rep. Eq. 21/24/30), s0bar its T->inf
     * limit (cpr_s0bar). */
    double z0 = g_const.T0CMB / cpr_MeV_to_Kelvin();
    double s_end, ds_dT_end;
    cpr_plasma_spl_and_dspl_dT(thermo, Tend, &s_end, &ds_dT_end);
    double sbar_end = s_end / (Tend * Tend * Tend);
    double zend = z0 / pow(sbar_end / cpr_s0bar(), 1.0 / 3.0);
    double a_end = zend / Tend;

    /* Fixed (not cfg->numerical_precision-derived) tolerance for the
     * background ODE(s) below -- see BG_ODE_RTOL's docstring above for the
     * empirical justification. Tolerance, NOT matched to Python's nominal
     * 0.1*numerical_precision: RK45 (explicit, Dormand-Prince) and Python's
     * LSODA achieve materially different *actual* accuracy at the same
     * *nominal* rtol for this ODE -- confirmed empirically by running the
     * full small/large+amax=8 BBN solve (nuclear_network.c) with this
     * tolerance at the Python-nominal value vs. progressively tighter
     * ones: at 0.1*1e-7=1e-8 the resulting YP(BBN)/D-H/Yn were off by
     * -0.14%/-1.8%/-3.5% from CLAUDE.md's reference numbers (BBN
     * abundances are exponentially sensitive to T(t)/a(T) near freeze-out,
     * so this small a(T) error is greatly amplified downstream); at
     * BG_ODE_RTOL/BG_ODE_ATOL below the same comparison is within
     * 0.002%/0.001%/0.005% -- inside CLAUDE.md's stated +-1e-5 (YP) and
     * +-3e-9 (D/H) bounds. Decoupled from cfg->numerical_precision (rather
     * than e.g. dividing it by a fixed factor) because this ODE is low-
     * dimensional, smooth, and cheap regardless of tolerance -- there is no
     * performance reason to ever loosen it, even for a fast/rough run; a
     * user wanting an even higher-precision *reference* run already has
     * other knobs for that (see CLAUDE.md's "Validation before
     * committing" reference-run setup). */
    CPRRKOpts bg_ode_opts = cpr_ode_rk_default_opts();
    bg_ode_opts.rtol = BG_ODE_RTOL;
    bg_ode_opts.atol = BG_ODE_ATOL;

    /* Combined-path state, populated and kept alive (across the a_sol_asc/
     * T_sol_asc build below) only in the default (!external_scale_factor)
     * branch -- see combined_bg_rhs's docstring above for why a single 2D
     * ODE replaces the old sequential a(T)+t(a) pair. Reused further down
     * to fill bg->t_vec via dense_path_eval, instead of a second ODE solve. */
    CPRDensePath combined_path;
    int have_combined_path = 0;

    bg->external_scale_factor = cfg->external_scale_factor;
    if (bg->external_scale_factor) {
        bg->K_ext = a_end / cpr_nu_x_of_Tg(&bg->nh, Tend);
        /* bg->lna_sol is left unallocated-but-unused here: cpr_bg_a_of_T
         * checks bg->external_scale_factor first and never reads it in this
         * mode. No ODE needed for a(T) -- only t(T) is solved below, via
         * the old dtdlna_rhs/T_of_a_smooth pair (no entropy ODE exists here
         * to combine it with). */
    } else {
        /* Boundary condition for the combined 2D ODE: x is anchored at
         * T_end exactly as the old dlnadlnT_rhs ODE was (x_ini = ln(a_end)
         * + ln(T_end), a_end the algebraic entropy-conservation value just
         * computed above); y[1] (trel) starts from an arbitrary placeholder
         * at the SAME point -- see combined_bg_rhs's "Variable choice for
         * y[1]" docstring for why T_end (not T_start_cosmo) must be the
         * shared anchor for the default config, and why an uncalibrated
         * trel can still be corrected into an exact t(T) after the fact. */
        CombinedBgCtx ctx = { bg };
        dense_path_init(&combined_path, 2);
        CPRRKOpts opts = bg_ode_opts;
        opts.dense_cb = dense_path_push;
        opts.dense_ctx = &combined_path;
        double y2[2] = { log(a_end) + log(Tend), 0.0 };
        char *err = NULL;
        /* Single combined call replaces the old two cpr_ode_rk45 calls
         * (a(T) over lnT, then t(a) over lna) plus the intermediate
         * not-a-knot cubic spline bridging them -- see combined_bg_rhs's
         * top comment. Integrated ASCENDING (T_end -> T_start_cosmo). */
        int rc = cpr_ode_rk45(combined_bg_rhs, &ctx, log(Tend), log(Tstartcosmo), y2, 2,
                               opts, NULL, NULL, &err);
        if (rc) {
            dense_path_free(&combined_path); free(T_sol);
            *errmsg = err;
            return 1;
        }
        have_combined_path = 1;
        for (size_t i = 0; i < bg->n_Tsol; i++) {
            double out2[2];
            dense_path_eval(&combined_path, bg->lnT_sol[i], out2);
            bg->lna_sol[i] = out2[0] - bg->lnT_sol[i]; /* a = exp(x - lnT) */
        }
    }

    /* a_grid = a_of_T(T_sol): descending in `a` since T_sol ascends and a
     * decreases monotonically with T (the universe expands as T drops).
     * Build the ascending-in-a pair (a_sol_asc, T_sol_asc) by reversing,
     * for cpr_bg_T_of_a's cpr_interp_linear call (requires ascending x). */
    double *a_grid = malloc(bg->n_Tsol * sizeof(double));
    for (size_t i = 0; i < bg->n_Tsol; i++) a_grid[i] = cpr_bg_a_of_T(bg, T_sol[i]);
    bg->a_sol_asc = malloc(bg->n_Tsol * sizeof(double));
    bg->T_sol_asc = malloc(bg->n_Tsol * sizeof(double));
    for (size_t i = 0; i < bg->n_Tsol; i++) {
        bg->a_sol_asc[i] = a_grid[bg->n_Tsol - 1 - i];
        bg->T_sol_asc[i] = T_sol[bg->n_Tsol - 1 - i];
    }
    free(a_grid);

    double a_ini = cpr_bg_a_of_T(bg, Tstartcosmo);
    double a_fin = cpr_bg_a_of_T(bg, Tend); /* == a_end by construction */

    bg->n_bg = bg->n_Tsol; /* same grid density for the t(a) sampling below */
    double *lna_samp = malloc(bg->n_bg * sizeof(double));
    for (size_t i = 0; i < bg->n_bg; i++) {
        double frac = (bg->n_bg == 1) ? 0.0 : (double)i / (double)(bg->n_bg - 1);
        lna_samp[i] = log(a_ini) + frac * (log(a_fin) - log(a_ini));
    }

    /* t(T)/t(a): cfg->external_scale_factor still needs its own ODE here
     * (a(T) there is a closed-form algebraic function, not an ODE solution,
     * so there is no entropy-conservation "first ODE" to fold dt/d(ln a)
     * into -- see DtDlnaCtx's docstring above). The default (!external_
     * scale_factor) branch instead reuses `combined_path` (already solved
     * above, alongside a(T)) via dense_path_eval plus a constant shift --
     * no second ODE call (see combined_bg_rhs's "Variable choice for
     * y[1]"). */
    CPRDensePath tpath;
    int have_tpath = 0;
    double combined_t_shift = 0.0;
    if (bg->external_scale_factor) {
        double Tnue_s = cpr_nu_Tnue_of_Tg(&bg->nh, Tstartcosmo);
        double Tnumu_s = cpr_nu_Tnumu_of_Tg(&bg->nh, Tstartcosmo);
        double Tnutau_s = cpr_nu_Tnutau_of_Tg(&bg->nh, Tstartcosmo);
        double t_ini = 1.0 / (2.0 * cpr_bg_Hubble(bg, Tstartcosmo, Tnue_s, Tnumu_s, Tnutau_s, a_ini));

        /* See DtDlnaCtx's docstring above: a smooth spline over the same
         * nodes cpr_bg_T_of_a interpolates linearly, used only as this
         * ODE's RHS. */
        CPRCubicSpline T_of_a_smooth;
        char *spl_err = NULL;
        if (cpr_cubic_spline_fit_notaknot(bg->a_sol_asc, bg->T_sol_asc, bg->n_Tsol,
                                           &T_of_a_smooth, &spl_err)) {
            free(T_sol); free(lna_samp);
            *errmsg = spl_err;
            return 1;
        }
        DtDlnaCtx tctx = { bg, &T_of_a_smooth };
        dense_path_init(&tpath, 1);
        CPRRKOpts topts = bg_ode_opts;
        topts.dense_cb = dense_path_push;
        topts.dense_ctx = &tpath;
        double yt[1] = { t_ini };
        char *terr = NULL;
        int trc = cpr_ode_rk45(dtdlna_rhs, &tctx, log(a_ini), log(a_fin), yt, 1, topts,
                                NULL, NULL, &terr);
        cpr_cubic_spline_free(&T_of_a_smooth);
        if (trc) {
            dense_path_free(&tpath); free(T_sol); free(lna_samp);
            *errmsg = terr;
            return 1;
        }
        have_tpath = 1;
    } else {
        /* Calibrate combined_path's uncalibrated trel into an absolute t:
         * a_ini = a(T_start_cosmo) now comes EXACTLY from the just-solved
         * x trajectory (no algebraic approximation needed, unlike the
         * abandoned tau-anchored-at-T_start_cosmo attempt -- see
         * combined_bg_rhs's docstring), so t_ini = 1/(2 H(T_start_cosmo,
         * a_ini)) is the same exact radiation-domination boundary value the
         * old t(a) ODE used. trel's own RHS has no t-dependence (shown in
         * combined_bg_rhs), so shifting the whole trel(lnT) curve by the
         * constant C = t_ini - trel(T_start_cosmo) gives the true t(lnT)
         * at every point, not just at T_start_cosmo. */
        double Tnue_s = cpr_nu_Tnue_of_Tg(&bg->nh, Tstartcosmo);
        double Tnumu_s = cpr_nu_Tnumu_of_Tg(&bg->nh, Tstartcosmo);
        double Tnutau_s = cpr_nu_Tnutau_of_Tg(&bg->nh, Tstartcosmo);
        double t_ini = 1.0 / (2.0 * cpr_bg_Hubble(bg, Tstartcosmo, Tnue_s, Tnumu_s, Tnutau_s, a_ini));
        double out2[2];
        dense_path_eval(&combined_path, log(Tstartcosmo), out2);
        combined_t_shift = t_ini - out2[1];
    }

    bg->t_vec = malloc(bg->n_bg * sizeof(double));
    bg->a_vec = malloc(bg->n_bg * sizeof(double));
    bg->Tg_vec = malloc(bg->n_bg * sizeof(double));
    bg->Tnue_vec = malloc(bg->n_bg * sizeof(double));
    bg->Tnumu_vec = malloc(bg->n_bg * sizeof(double));
    bg->Tnutau_vec = malloc(bg->n_bg * sizeof(double));
    bg->Tnu_vec = malloc(bg->n_bg * sizeof(double));
    for (size_t i = 0; i < bg->n_bg; i++) {
        double a = exp(lna_samp[i]);
        bg->a_vec[i] = a;
        double Tg = cpr_bg_T_of_a(bg, a);
        bg->Tg_vec[i] = Tg;
        /* t(a): external mode reads the dedicated tpath at this lna_samp
         * point directly (its own natural independent variable); the
         * default mode instead reads the (shift-calibrated) trel off the
         * combined a(T)/t(T) path at the corresponding lnT = log(Tg) (its
         * natural independent variable) -- no extra ODE solve, just one
         * more dense-output lookup on the path already built above, plus
         * the constant shift computed once outside this loop. */
        if (have_tpath) {
            double out1[1];
            dense_path_eval(&tpath, lna_samp[i], out1);
            bg->t_vec[i] = out1[0];
        } else {
            double out2[2];
            dense_path_eval(&combined_path, log(Tg), out2);
            bg->t_vec[i] = out2[1] + combined_t_shift;
        }
        bg->Tnue_vec[i] = cpr_nu_Tnue_of_Tg(&bg->nh, Tg);
        bg->Tnumu_vec[i] = cpr_nu_Tnumu_of_Tg(&bg->nh, Tg);
        bg->Tnutau_vec[i] = cpr_nu_Tnutau_of_Tg(&bg->nh, Tg);
        double e4 = (pow(bg->Tnue_vec[i], 4.0) + pow(bg->Tnumu_vec[i], 4.0)
                     + pow(bg->Tnutau_vec[i], 4.0)) / 3.0;
        bg->Tnu_vec[i] = pow(e4, 0.25);
    }
    if (have_tpath) dense_path_free(&tpath);
    if (have_combined_path) dense_path_free(&combined_path);
    free(T_sol);
    free(lna_samp);

    /* Tg_vec is time-ascending => T-descending; build the T-ascending
     * reverse for cpr_bg_t_of_T (see background.h's field comment). */
    bg->Tg_asc = malloc(bg->n_bg * sizeof(double));
    bg->t_by_Tg_asc = malloc(bg->n_bg * sizeof(double));
    for (size_t i = 0; i < bg->n_bg; i++) {
        bg->Tg_asc[i] = bg->Tg_vec[bg->n_bg - 1 - i];
        bg->t_by_Tg_asc[i] = bg->t_vec[bg->n_bg - 1 - i];
    }

    bg->has_scale_factor = 1;
    bg->has_heating_table = cfg->incomplete_decoupling;

    return 0;
}

static int setup_weak_rates_standard(CPRBackground *bg, char **errmsg)
{
    const CPRConfig *cfg = bg->cfg;
    if (cpr_weak_rates_init(&bg->wr, bg->Tg_vec, bg->Tnue_vec, bg->n_bg, cfg, &bg->nh, errmsg))
        return 1;

    if (cfg->tau_n_normalization) {
        bg->norm_weak_rates = 1.0 / cfg->tau_n;
    } else {
        double Fn = cpr_compute_fn(cfg);
        double GFtilde2 = (g_const.GF * g_const.Vud) * (g_const.GF * g_const.Vud)
                            * (1.0 + 3.0 * g_const.gA * g_const.gA) / (2.0 * M_PI * M_PI * M_PI);
        bg->norm_weak_rates = cpr_MeV_to_secm1() * (GFtilde2 * pow(g_const.me, 5.0)) * Fn;
    }
    return 0;
}

int cpr_bg_init_standard(CPRBackground *bg, const CPRConfig *cfg, const CPRPlasma *plasma,
                          char **errmsg)
{
    memset(bg, 0, sizeof(*bg));
    bg->kind = CPR_BG_STANDARD;
    bg->cfg = cfg;
    bg->plasma = plasma;

    setup_lcdm(bg);
    setup_ede(bg);

    if (cpr_neutrino_history_init(&bg->nh, cfg, plasma, errmsg)) return 1;
    bg->nh_owned = 1;

    cpr_log(cfg, "bg", "Solving cosmological background a(t,T) ...");
    clock_t _t_bg0 = clock();
    if (setup_background_and_cosmo(bg, errmsg)) return 1;
    cpr_log(cfg, "bg", "Background a(t,T) ready in %.2f s",
             (double)(clock() - _t_bg0) / CLOCKS_PER_SEC);

    if (setup_weak_rates_standard(bg, errmsg)) return 1;

    return 0;
}

/* ---------------------------------------------------------------------
 * CustomBackground.
 * ------------------------------------------------------------------- */

/* Minimal tab-or-comma-delimited, named-header table reader for the
 * custom_background file format (T/t/a required columns, extra columns
 * silently ignored, rows in any order). table_io.c's cpr_table_read is
 * deliberately not reused here: its contract is a fixed, pre-known column
 * count with no header-name lookup or column subsetting, whereas this
 * format's defining feature (mirrors np.genfromtxt(..., names=True)) is
 * exactly the opposite -- look up T/t/a by name among an arbitrary,
 * unordered set of named columns. */
static int read_custom_table(const char *filename, double **T_out, double **t_out,
                               double **a_out, size_t *n_out, char **errmsg)
{
    FILE *f = fopen(filename, "r");
    if (!f) {
        char buf[4200];
        snprintf(buf, sizeof(buf), "custom_background file '%s' not found", filename);
        *errmsg = strdup(buf);
        return 1;
    }

    char header[4096];
    if (!fgets(header, sizeof(header), f)) {
        fclose(f);
        *errmsg = strdup("custom_background file is empty");
        return 1;
    }
    char delim = (strchr(header, '\t') != NULL) ? '\t' : ',';

    /* Tokenise the header to find the (0-based) column index of T, t, a. */
    char hdr_copy[4096];
    strncpy(hdr_copy, header, sizeof(hdr_copy) - 1);
    hdr_copy[sizeof(hdr_copy) - 1] = '\0';
    int idx_T = -1, idx_t = -1, idx_a = -1, n_cols = 0;
    /* strtok_r, not strtok: mc.c's worker threads may each call
     * cpr_bg_init_custom concurrently, and strtok's cursor is a single
     * static buffer shared process-wide (see config.c's load_nuclides for
     * the same fix and the observed symptom). */
    char *strtok_state = NULL;
    char *tok = strtok_r(hdr_copy, delim == '\t' ? "\t\r\n" : ",\r\n", &strtok_state);
    while (tok) {
        while (*tok == ' ') tok++;
        size_t L = strlen(tok);
        while (L > 0 && tok[L - 1] == ' ') tok[--L] = '\0';
        if (strcmp(tok, "T") == 0) idx_T = n_cols;
        else if (strcmp(tok, "t") == 0) idx_t = n_cols;
        else if (strcmp(tok, "a") == 0) idx_a = n_cols;
        n_cols++;
        tok = strtok_r(NULL, delim == '\t' ? "\t\r\n" : ",\r\n", &strtok_state);
    }
    if (idx_T < 0 || idx_t < 0 || idx_a < 0) {
        fclose(f);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "custom_background file is missing required column(s): %s%s%s",
                 idx_T < 0 ? "T " : "", idx_t < 0 ? "t " : "", idx_a < 0 ? "a " : "");
        *errmsg = strdup(buf);
        return 1;
    }

    size_t cap = 64, n = 0;
    double *T = malloc(cap * sizeof(double));
    double *t = malloc(cap * sizeof(double));
    double *a = malloc(cap * sizeof(double));
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        double Tv = NAN, tv = NAN, av = NAN;
        int col = 0;
        char *row_state = NULL;
        char *t2 = strtok_r(line, delim == '\t' ? "\t\r\n" : ",\r\n", &row_state);
        while (t2) {
            if (col == idx_T) Tv = strtod(t2, NULL);
            else if (col == idx_t) tv = strtod(t2, NULL);
            else if (col == idx_a) av = strtod(t2, NULL);
            col++;
            t2 = strtok_r(NULL, delim == '\t' ? "\t\r\n" : ",\r\n", &row_state);
        }
        if (n == cap) {
            cap *= 2;
            T = realloc(T, cap * sizeof(double));
            t = realloc(t, cap * sizeof(double));
            a = realloc(a, cap * sizeof(double));
        }
        T[n] = Tv; t[n] = tv; a[n] = av;
        n++;
    }
    fclose(f);

    for (size_t i = 0; i < n; i++) {
        if (!(T[i] > 0.0) || !(t[i] > 0.0) || !(a[i] > 0.0)) {
            free(T); free(t); free(a);
            *errmsg = strdup("custom_background file: columns T, t, a must all be strictly positive");
            return 1;
        }
    }

    *T_out = T; *t_out = t; *a_out = a; *n_out = n;
    return 0;
}

/* Sorts the parallel (T, t, a) triple ascending by `key` (one of the three
 * arrays, e.g. `t` for the t-ascending ordering, modified in place along
 * with its two siblings) via insertion sort on an index permutation --
 * avoids any aliasing ambiguity from sorting an array by itself in place
 * (the custom-background table is never large enough, by construction, to
 * need anything better than insertion sort). */
static void sort_by(const double *key, double *T, double *t, double *a, size_t n)
{
    size_t *idx = malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) idx[i] = i;
    for (size_t i = 1; i < n; i++) {
        size_t cur = idx[i];
        size_t j = i;
        while (j > 0 && key[idx[j - 1]] > key[cur]) { idx[j] = idx[j - 1]; j--; }
        idx[j] = cur;
    }
    double *T2 = malloc(n * sizeof(double)), *t2 = malloc(n * sizeof(double)), *a2 = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) { T2[i] = T[idx[i]]; t2[i] = t[idx[i]]; a2[i] = a[idx[i]]; }
    memcpy(T, T2, n * sizeof(double));
    memcpy(t, t2, n * sizeof(double));
    memcpy(a, a2, n * sizeof(double));
    free(idx); free(T2); free(t2); free(a2);
}

int cpr_bg_init_custom(CPRBackground *bg, const CPRConfig *cfg, const CPRPlasma *plasma,
                        const char *filename, char **errmsg)
{
    memset(bg, 0, sizeof(*bg));
    bg->kind = CPR_BG_CUSTOM;
    bg->cfg = cfg;
    bg->plasma = plasma;

    double *T_raw, *t_raw, *a_raw;
    size_t n;
    if (read_custom_table(filename, &T_raw, &t_raw, &a_raw, &n, errmsg)) return 1;
    bg->n_custom = n;

    bg->t_asc = malloc(n * sizeof(double)); bg->T_by_t = malloc(n * sizeof(double)); bg->a_by_t = malloc(n * sizeof(double));
    bg->T_asc = malloc(n * sizeof(double)); bg->t_by_T = malloc(n * sizeof(double)); bg->a_by_T = malloc(n * sizeof(double));
    bg->a_sort = malloc(n * sizeof(double)); bg->T_by_a = malloc(n * sizeof(double)); bg->t_by_a = malloc(n * sizeof(double));

    memcpy(bg->t_asc, t_raw, n * sizeof(double));
    memcpy(bg->T_by_t, T_raw, n * sizeof(double));
    memcpy(bg->a_by_t, a_raw, n * sizeof(double));
    sort_by(bg->t_asc, bg->T_by_t, bg->t_asc, bg->a_by_t, n); /* sort by t (key==t_asc itself) */

    memcpy(bg->T_asc, T_raw, n * sizeof(double));
    memcpy(bg->t_by_T, t_raw, n * sizeof(double));
    memcpy(bg->a_by_T, a_raw, n * sizeof(double));
    sort_by(bg->T_asc, bg->T_asc, bg->t_by_T, bg->a_by_T, n);

    memcpy(bg->a_sort, bg->a_by_t, n * sizeof(double));
    memcpy(bg->T_by_a, bg->T_by_t, n * sizeof(double));
    memcpy(bg->t_by_a, bg->t_asc, n * sizeof(double));
    sort_by(bg->a_sort, bg->T_by_a, bg->t_by_a, bg->a_sort, n);

    free(T_raw); free(t_raw); free(a_raw);
    bg->has_scale_factor = 1;

    /* Instantaneous-decoupling neutrino history: cpr_neutrino_history_init
     * dispatches on cfg->incomplete_decoupling, which custom_background
     * configs always have False (enforced by cpr_config_validate, mirroring
     * PRIMATConfig.__init__'s warning/forced-False for this combination) --
     * so this always takes the CPR_NU_INSTANTANEOUS branch, exactly as
     * Python's CustomBackground._setup_neutrino_history directly
     * instantiates InstantaneousDecoupling. */
    if (cpr_neutrino_history_init(&bg->nh, cfg, plasma, errmsg)) return 1;
    bg->nh_owned = 1;

    double T_lo = bg->T_asc[0], T_hi = bg->T_asc[n - 1];
    int n_T_pts = n_points_per_decade(cfg->sampling_temperature_per_decade, T_lo, T_hi);
    bg->n_bg = (size_t)n_T_pts;
    bg->Tg_vec = malloc(bg->n_bg * sizeof(double));
    bg->Tnue_vec = malloc(bg->n_bg * sizeof(double));
    for (size_t i = 0; i < bg->n_bg; i++) {
        double frac = (bg->n_bg == 1) ? 0.0 : (double)i / (double)(bg->n_bg - 1);
        bg->Tg_vec[i] = T_lo + frac * (T_hi - T_lo);
        bg->Tnue_vec[i] = cpr_nu_Tnue_of_Tg(&bg->nh, bg->Tg_vec[i]);
    }

    if (cpr_weak_rates_init(&bg->wr, bg->Tg_vec, bg->Tnue_vec, bg->n_bg, cfg, &bg->nh, errmsg))
        return 1;
    if (cfg->tau_n_normalization) {
        bg->norm_weak_rates = 1.0 / cfg->tau_n;
    } else {
        double Fn = cpr_compute_fn(cfg);
        double GFtilde2 = (g_const.GF * g_const.Vud) * (g_const.GF * g_const.Vud)
                            * (1.0 + 3.0 * g_const.gA * g_const.gA) / (2.0 * M_PI * M_PI * M_PI);
        bg->norm_weak_rates = cpr_MeV_to_secm1() * (GFtilde2 * pow(g_const.me, 5.0)) * Fn;
    }
    return 0;
}

void cpr_background_free(CPRBackground *bg)
{
    if (bg->nh_owned) cpr_neutrino_history_free(&bg->nh);
    free(bg->t_vec); free(bg->a_vec); free(bg->Tg_vec); free(bg->Tnue_vec); free(bg->Tnumu_vec);
    free(bg->Tnutau_vec); free(bg->Tnu_vec); free(bg->Tg_asc); free(bg->t_by_Tg_asc);
    free(bg->lnT_sol); free(bg->lna_sol); free(bg->a_sol_asc); free(bg->T_sol_asc);
    free(bg->t_asc); free(bg->T_by_t); free(bg->a_by_t);
    free(bg->T_asc); free(bg->t_by_T); free(bg->a_by_T);
    free(bg->a_sort); free(bg->T_by_a); free(bg->t_by_a);
    cpr_weak_rates_free(&bg->wr);
    memset(bg, 0, sizeof(*bg));
}

/* ---------------------------------------------------------------------
 * Query interface.
 * ------------------------------------------------------------------- */

double cpr_bg_T_of_t(const CPRBackground *bg, double t)
{
    if (bg->kind == CPR_BG_CUSTOM)
        return cpr_interp_linear(bg->t_asc, bg->T_by_t, bg->n_custom, t, CPR_EXTRAP_LINEAR);
    return cpr_interp_linear(bg->t_vec, bg->Tg_vec, bg->n_bg, t, CPR_EXTRAP_LINEAR);
}

double cpr_bg_t_of_T(const CPRBackground *bg, double T)
{
    if (bg->kind == CPR_BG_CUSTOM)
        return cpr_interp_linear(bg->T_asc, bg->t_by_T, bg->n_custom, T, CPR_EXTRAP_LINEAR);
    return cpr_interp_linear(bg->Tg_asc, bg->t_by_Tg_asc, bg->n_bg, T, CPR_EXTRAP_LINEAR);
}

double cpr_bg_a_of_T(const CPRBackground *bg, double T)
{
    if (bg->kind == CPR_BG_CUSTOM)
        return cpr_interp_linear(bg->T_asc, bg->a_by_T, bg->n_custom, T, CPR_EXTRAP_LINEAR);
    if (bg->external_scale_factor) return bg->K_ext * cpr_nu_x_of_Tg(&bg->nh, T);
    return exp(cpr_interp_linear(bg->lnT_sol, bg->lna_sol, bg->n_Tsol, log(T), CPR_EXTRAP_LINEAR));
}

double cpr_bg_T_of_a(const CPRBackground *bg, double a)
{
    if (bg->kind == CPR_BG_CUSTOM)
        return cpr_interp_linear(bg->a_sort, bg->T_by_a, bg->n_custom, a, CPR_EXTRAP_LINEAR);
    return cpr_interp_linear(bg->a_sol_asc, bg->T_sol_asc, bg->n_Tsol, a, CPR_EXTRAP_LINEAR);
}

double cpr_bg_a_of_t(const CPRBackground *bg, double t)
{
    if (bg->kind == CPR_BG_CUSTOM)
        return cpr_interp_linear(bg->t_asc, bg->a_by_t, bg->n_custom, t, CPR_EXTRAP_LINEAR);
    /* StandardBackground uses CONSTANT extrapolation here (Python:
     * fill_value=(a_arr[0], a_arr[-1])), unlike every other Standard
     * interpolant above (LINEAR) -- see background.py's
     * _setup_background_and_cosmo, Step 5. */
    return cpr_interp_linear(bg->t_vec, bg->a_vec, bg->n_bg, t, CPR_EXTRAP_CONSTANT);
}

double cpr_bg_t_of_a(const CPRBackground *bg, double a)
{
    if (bg->kind == CPR_BG_CUSTOM)
        return cpr_interp_linear(bg->a_sort, bg->t_by_a, bg->n_custom, a, CPR_EXTRAP_LINEAR);
    return cpr_interp_linear(bg->a_vec, bg->t_vec, bg->n_bg, a, CPR_EXTRAP_CONSTANT);
}

int cpr_bg_Tnu_of_t(const CPRBackground *bg, double t, double *Tnue, double *Tnumu,
                     double *Tnutau)
{
    if (bg->kind != CPR_BG_STANDARD)
        return 0;
    *Tnue   = cpr_interp_linear(bg->t_vec, bg->Tnue_vec,   bg->n_bg, t, CPR_EXTRAP_LINEAR);
    *Tnumu  = cpr_interp_linear(bg->t_vec, bg->Tnumu_vec,  bg->n_bg, t, CPR_EXTRAP_LINEAR);
    *Tnutau = cpr_interp_linear(bg->t_vec, bg->Tnutau_vec, bg->n_bg, t, CPR_EXTRAP_LINEAR);
    return 1;
}

double cpr_bg_rhoB_BBN(const CPRBackground *bg, double t)
{
    const CPRConfig *cfg = bg->cfg;
    double n0B = cpr_n0CMB() * cfg->eta0b;
    double a = cpr_bg_a_of_t(bg, t);
    return g_const.ma * n0B * cpr_MeV4_to_gcmm3() / (a * a * a);
}

double cpr_bg_weak_nTOp_frwrd(const CPRBackground *bg, double T_K)
{
    return bg->norm_weak_rates * clamp_raw_weak_rate(cpr_weak_rate_nTOp(&bg->wr, T_K));
}
double cpr_bg_weak_nTOp_bkwrd(const CPRBackground *bg, double T_K)
{
    return bg->norm_weak_rates * clamp_raw_weak_rate(cpr_weak_rate_pTOn(&bg->wr, T_K));
}

double cpr_bg_N_eff(const CPRBackground *bg, double Tg, double rho_nu_tot)
{
    (void)bg; /* generic formula, identical for both kinds -- see background.h */
    return rho_nu_tot / cpr_rho_g(Tg) / ((7.0 / 8.0) * pow(4.0 / 11.0, 4.0 / 3.0));
}

int cpr_bg_rho_nu_total_final(const CPRBackground *bg, double *Tg_final, double *rho_nu_tot_final)
{
    const CPRPlasma *thermo = bg->plasma;

    if (bg->kind == CPR_BG_STANDARD) {
        size_t i = bg->n_bg - 1;
        *Tg_final = bg->Tg_vec[i];
        *rho_nu_tot_final = cpr_rho_nu(bg->Tnue_vec[i]) + cpr_rho_nu(bg->Tnumu_vec[i])
                             + cpr_rho_nu(bg->Tnutau_vec[i])
                             + cpr_plasma_rho_nu_extra(thermo, bg->Tg_vec[i]);
        /* Genuine chemical-potential energy excess (see cpr_bg_hubble); must be
         * included here too so Neff reflects it. The CustomBackground branch
         * below derives rho_nu from rho_tot - rho_plasma, so it already picks
         * this up through cpr_bg_hubble. */
        if (bg->cfg->munuOverTnu != 0.0) {
            double xi = bg->cfg->munuOverTnu;
            *rho_nu_tot_final += cpr_rho_nu_chempot_excess(bg->Tnue_vec[i], xi)
                               + cpr_rho_nu_chempot_excess(bg->Tnumu_vec[i], xi)
                               + cpr_rho_nu_chempot_excess(bg->Tnutau_vec[i], xi);
        }
        /* Analytic y/gray distortion's extra energy density (see
         * cpr_bg_hubble). bg->Tnu_vec[i] is exactly the energy-weighted
         * average flavour temperature, mirroring Python's self.Tnu_vec. */
        if (bg->nh.has_analytic_distortion) {
            *rho_nu_tot_final += cpr_nu_rho_nu_SD(&bg->nh, bg->Tnu_vec[i]);
        }
        return 0;
    }

    /* CustomBackground: estimate H at the final table point from a
     * power-law fit ln(a) = p*ln(t) + q over the last N_fit points
     * (a ~ t^p, p->1/2 in radiation domination), then invert the
     * Friedmann equation H^2 = 8 pi G/3 rho_tot for rho_tot, and isolate
     * rho_nu = rho_tot - rho_plasma(Tg_final). See background.py's
     * CustomBackground.rho_nu_total_final docstring for why the power-law
     * fit (not a one-sided finite difference) is used. */
    size_t n = bg->n_custom;
    size_t N_fit = n / 2 < 50 ? n / 2 : 50;
    if (N_fit < 2) N_fit = 2;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t k = n - N_fit; k < n; k++) {
        double lx = log(bg->t_asc[k]), ly = log(bg->a_by_t[k]);
        sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly;
    }
    double Nf = (double)N_fit;
    double p_slope = (Nf * sxy - sx * sy) / (Nf * sxx - sx * sx);
    double H_final = p_slope / bg->t_asc[n - 1];
    double Tg_f = bg->T_by_t[n - 1];

    double H_MeV = H_final / cpr_MeV_to_secm1();
    double Mpl = cpr_config_Mpl(bg->cfg);
    double rho_tot = 3.0 * Mpl * Mpl / (8.0 * M_PI) * H_MeV * H_MeV;

    double rho_plasma = cpr_rho_g(Tg_f) + cpr_plasma_rho_e(thermo, Tg_f)
                         - bg_PQEDofT(thermo, Tg_f) + Tg_f * bg_dPQEDdT(thermo, Tg_f);
    double rho_nu_tot = rho_tot - rho_plasma;

    fprintf(stderr,
            "[custom_background] Neff from Friedmann H^2=8piG/3*rho_tot at Tg = %.4e MeV: "
            "H = %.6e s^-1, rho_tot = %.6e MeV^4, rho_plasma = %.6e MeV^4, rho_nu = %.6e MeV^4\n",
            Tg_f, H_final, rho_tot, rho_plasma, rho_nu_tot);

    *Tg_final = Tg_f;
    *rho_nu_tot_final = rho_nu_tot;
    return 0;
}

int cpr_bg_Omeganuh2_relnu(const CPRBackground *bg, double *out)
{
    if (bg->kind != CPR_BG_STANDARD) return 1;
    size_t i = bg->n_bg - 1;
    double Tnu0 = bg->Tnu_vec[i] / bg->Tg_vec[i] * g_const.T0CMB / cpr_MeV_to_Kelvin();
    *out = (7.0 * M_PI * M_PI / 120.0 * pow(Tnu0, 4.0)) / cpr_config_rhocOverh2(bg->cfg);
    return 0;
}

int cpr_bg_Omeganuh2_nrnu(const CPRBackground *bg, double *out)
{
    if (bg->kind != CPR_BG_STANDARD) return 1;
    size_t i = bg->n_bg - 1;
    double Tnu0 = bg->Tnu_vec[i] / bg->Tg_vec[i] * g_const.T0CMB / cpr_MeV_to_Kelvin();
    *out = (1.5 * ZETA3 / (M_PI * M_PI) * pow(Tnu0, 3.0)) / cpr_config_rhocOverh2(bg->cfg);
    return 0;
}

/* ===========================================================================
 * Background time-evolution TSV writer (mirrors
 * Python's StandardBackground.write_time_evolution / time_evolution_text).
 * ===========================================================================
 */

/* mkdir -p equivalent (copied from nuclear_network.c). */
static void bg_mkdir_p(const char *path)
{
    char buf[4300];
    snprintf(buf, sizeof(buf), "%s", path);
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(buf, 0755);
            *p = '/';
        }
    }
    mkdir(buf, 0755);
}

int cpr_bg_write_time_evolution(const CPRBackground *bg, const char *path, int n_points, char **errmsg)
{
    if (bg->kind != CPR_BG_STANDARD) {
        *errmsg = strdup("cpr_bg_write_time_evolution only supported for CPR_BG_STANDARD");
        return 1;
    }

    const CPRConfig *cfg = bg->cfg;
    const CPRPlasma *thermo = bg->plasma;
    int has_nheating = bg->has_heating_table;

    /* Build the time grid: log-spaced from t_lo to t_hi */
    double T_start_cosmo_K = cpr_config_T_start_cosmo(cfg);
    double T_end_K = cpr_config_T_end(cfg);
    double T_start_cosmo = T_start_cosmo_K / cpr_MeV_to_Kelvin();
    double T_end = T_end_K / cpr_MeV_to_Kelvin();
    double t_lo = cpr_bg_t_of_T(bg, T_start_cosmo);
    double t_hi = cpr_bg_t_of_T(bg, T_end);

    if (t_lo <= 0.0 || t_hi <= 0.0 || t_lo >= t_hi) {
        char buf[4400];
        snprintf(buf, sizeof(buf), "cpr_bg_write_time_evolution: invalid time range [%.3e, %.3e] s", t_lo, t_hi);
        *errmsg = strdup(buf);
        return 1;
    }

    /* Allocate time grid */
    double *t_out = malloc((size_t)n_points * sizeof(double));
    if (!t_out) {
        *errmsg = strdup("cpr_bg_write_time_evolution: out of memory for t_out");
        return 1;
    }

    /* Log-spaced grid */
    double log_t_lo = log10(t_lo);
    double log_t_hi = log10(t_hi);
    double dlogt = (n_points > 1) ? (log_t_hi - log_t_lo) / (n_points - 1) : 0.0;
    for (int i = 0; i < n_points; i++) {
        t_out[i] = pow(10.0, log_t_lo + i * dlogt);
    }

    /* Allocate storage for all columns */
    /* Columns: T, t, a, H, Tnue, Tnumu, Tnutau, [Nheating], rho_plasma, rho_nu_tot, [rho_extra], rho_tot */
    size_t n_cols = 10;
    if (has_nheating) n_cols++;
    /* rho_extra column is added if there are any extra energy-density contributions (LCDM/EDE) */
    int has_extra = bg->has_lcdm || bg->has_ede;
    if (has_extra) n_cols++;
    double *data = calloc(n_points * n_cols, sizeof(double));
    if (!data) {
        free(t_out);
        *errmsg = strdup("cpr_bg_write_time_evolution: out of memory for data");
        return 1;
    }

    /* Fill data for each time point */
    for (int i = 0; i < n_points; i++) {
        double t = t_out[i];
        double T = cpr_bg_T_of_t(bg, t);
        double a = cpr_bg_a_of_t(bg, t);

        /* T and t */
        data[i * n_cols + 0] = T;
        data[i * n_cols + 1] = t;

        if (bg->has_scale_factor) {
            double Tnue = 0.0, Tnumu = 0.0, Tnutau = 0.0;
            /* For StandardBackground, this should always return 1 */
            if (cpr_bg_Tnu_of_t(bg, t, &Tnue, &Tnumu, &Tnutau)) {
                double H = cpr_bg_Hubble(bg, T, Tnue, Tnumu, Tnutau, a);

                data[i * n_cols + 2] = a;
                data[i * n_cols + 3] = H;
                data[i * n_cols + 4] = Tnue;
                data[i * n_cols + 5] = Tnumu;
                data[i * n_cols + 6] = Tnutau;

                size_t col = 7;
                if (has_nheating) {
                    data[i * n_cols + col] = cpr_nu_N_NEVO_of_Tg(&bg->nh, T);
                    col++;
                }

                /* Energy densities */
                double PQEDofT = bg_PQEDofT(thermo, T);
                double dPQEDdT = bg_dPQEDdT(thermo, T);
                double rho_plasma = cpr_rho_g(T) + cpr_plasma_rho_e(thermo, T)
                                 - PQEDofT + T * dPQEDdT;
                double rho_nu_tot = cpr_rho_nu(Tnue) + cpr_rho_nu(Tnumu) + cpr_rho_nu(Tnutau)
                                   + cpr_plasma_rho_nu_extra(thermo, T);

                /* Spectral-distortion contribution */
                if (bg->nh.has_analytic_distortion) {
                    double Tnu_avg = pow((pow(Tnue, 4.0) + pow(Tnumu, 4.0) + pow(Tnutau, 4.0)) / 3.0, 0.25);
                    rho_nu_tot += cpr_nu_rho_nu_SD(&bg->nh, Tnu_avg);
                }
                /* Chemical potential contribution */
                if (cfg->munuOverTnu != 0.0) {
                    double xi = cfg->munuOverTnu;
                    rho_nu_tot += cpr_rho_nu_chempot_excess(Tnue, xi);
                    rho_nu_tot += cpr_rho_nu_chempot_excess(Tnumu, xi);
                    rho_nu_tot += cpr_rho_nu_chempot_excess(Tnutau, xi);
                }

                data[i * n_cols + col] = rho_plasma;
                col++;
                data[i * n_cols + col] = rho_nu_tot;
                col++;

                /* Extra energy-density contributions (LCDM + EDE) */
                double rho_extra = 0.0;
                if (bg->has_lcdm) {
                    rho_extra += bg->rhocdm_a3 / (a * a * a) + bg->rholambda;
                }
                if (bg->has_ede) {
                    rho_extra += 2.0 * bg->rhocEDEac / (1.0 + pow(bg->TcEDE / T, bg->EDE_exponent));
                }
                if (has_extra) {
                    data[i * n_cols + col] = rho_extra;
                    col++;
                }

                double rho_tot = rho_plasma + rho_nu_tot + rho_extra;
                data[i * n_cols + col] = rho_tot;
            }
        }
    }

    /* Prepare directory */
    char abspath[4300];
    snprintf(abspath, sizeof(abspath), "%s", path);
    char *slash = strrchr(abspath, '/');
    if (slash) {
        *slash = '\0';
        bg_mkdir_p(abspath);
        *slash = '/';
    }

    /* Open file */
    FILE *f = fopen(path, "w");
    if (!f) {
        char buf[4400];
        snprintf(buf, sizeof(buf), "cpr_bg_write_time_evolution: cannot open %s", path);
        free(t_out);
        free(data);
        *errmsg = strdup(buf);
        return 1;
    }

    /* Write header */
    fprintf(f, "T [MeV]\tt [s]\ta [1]\tH [s^-1]\tTnue [MeV]\tTnumu [MeV]\tTnutau [MeV]");
    if (has_nheating) {
        fprintf(f, "\tNheating [1]");
    }
    fprintf(f, "\trho_plasma [MeV^4]\trho_nu_tot [MeV^4]");
    if (has_extra) {
        fprintf(f, "\trho_extra [MeV^4]");
    }
    fprintf(f, "\trho_tot [MeV^4]\n");

    /* Write data rows */
    for (int i = 0; i < n_points; i++) {
        for (size_t j = 0; j < n_cols; j++) {
            if (j > 0) fprintf(f, "\t");
            fprintf(f, "%.10e", data[i * n_cols + j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
    free(t_out);
    free(data);

    printf("[output] Background time-evolution data (%d rows) written to %s\n", n_points, path);
    return 0;
}
