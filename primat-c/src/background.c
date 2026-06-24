/* background.c -- see cprimat/background.h.
 *
 * Reference: Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095),
 * cited below as "Phys. Rep.".
 */
#include "cprimat/background.h"
#include "cprimat/constants.h"
#include "cprimat/spline.h"
#include "cprimat/ode_rk.h"
#include "cprimat/log.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/* Fixed (not cfg->numerical_precision-derived) tolerance for the two
 * background ODEs below (a(T) entropy conservation, t(a) Hubble
 * integration) -- see the first call site's comment for the empirical
 * justification (closes a ~1-3% end-to-end BBN-abundance gap traced to
 * RK45-vs-LSODA discretization at the Python-nominal tolerance). */
#define BG_ODE_RTOL 1.0e-15
#define BG_ODE_ATOL 1.0e-15

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
 * Growable (t, y) path recorder used by the two background ODEs below
 * (a(T) entropy-conservation, t(a) Hubble integration). ode_rk.h's
 * cpr_ode_rk45 has no t_eval (it returns only the final state); Python's
 * solve_ivp(..., t_eval=...) effectively does the same thing this does --
 * dense-output interpolation of the accepted-step solution onto a
 * prescribed grid -- so recording every accepted step here and
 * interpolating onto the desired grid afterwards (see bg_path_eval) is a
 * faithful (if differently-stepped) match, not an approximation of a
 * fundamentally different scheme. RK45 in place of LSODA for these two
 * smooth, non-stiff ODEs is exactly what ode_rk.h's own header comment
 * anticipates.
 * ------------------------------------------------------------------- */
typedef struct {
    double *t, *y;
    size_t n, cap;
} CPRPath;

static void path_init(CPRPath *p) { p->t = NULL; p->y = NULL; p->n = 0; p->cap = 0; }

static void path_push(CPRPath *p, double t, double y)
{
    if (p->n == p->cap) {
        p->cap = p->cap ? p->cap * 2 : 64;
        p->t = realloc(p->t, p->cap * sizeof(double));
        p->y = realloc(p->y, p->cap * sizeof(double));
    }
    p->t[p->n] = t;
    p->y[p->n] = y;
    p->n++;
}

static void path_step_cb(double t, const double *y, size_t n, void *ctx)
{
    (void)n;
    path_push((CPRPath *)ctx, t, y[0]);
}

static void path_free(CPRPath *p) { free(p->t); free(p->y); p->n = p->cap = 0; }

/* Evaluates the recorded path's piecewise-linear interpolant at `tq`
 * (`t` ascending, since both ODEs below integrate forward from t0<t1). */
static double path_eval(const CPRPath *p, double tq)
{
    return cpr_interp_linear(p->t, p->y, p->n, tq, CPR_EXTRAP_LINEAR);
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
    bg->lcdm_use_exact = 0; /* radiation-domination approx until a_of_T is built */
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
                      double Tnutau)
{
    const CPRConfig *cfg = bg->cfg;
    const CPRPlasma *thermo = bg->plasma;

    double rho_pl = cpr_rho_g(Tg) + cpr_plasma_rho_e(thermo, Tg)
                     - bg_PQEDofT(thermo, Tg) + Tg * bg_dPQEDdT(thermo, Tg);
    double rho_3nu = cpr_rho_nu(Tnue) + cpr_rho_nu(Tnumu) + cpr_rho_nu(Tnutau);
    /* Genuine neutrino chemical potential: raises the neutrino energy density
     * (each flavour by Tnu^4 (xi^2/4 + xi^4/(8pi^2)); antineutrino carries
     * -xi). Mirrors pyprimat Plasma.rho_nu. It also shifts the n<->p weak rates
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
        double a_cdm = bg->lcdm_use_exact
                        ? cpr_bg_a_of_T(bg, Tg)
                        : (g_const.T0CMB / cpr_MeV_to_Kelvin()) / Tg; /* radiation-domination bootstrap */
        rho_tot += bg->rhocdm_a3 / (a_cdm * a_cdm * a_cdm) + bg->rholambda;
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

typedef struct { CPRBackground *bg; } DlnaDlnTCtx;

/* d(ln a)/d(ln T) = -(3 sbar + T dsbar/dT) / (N_NEVO + 3 sbar), the EM
 * entropy-conservation ODE driving a(T_gamma) (Phys. Rep. S2; see
 * background.py's _setup_background_and_cosmo docstring, "minimal mode"). */
static int dlnadlnT_rhs(double lnT, const double *y, double *ydot, void *ctx_)
{
    (void)y;
    DlnaDlnTCtx *c = ctx_;
    double T = exp(lnT);
    double s, ds_dT;
    cpr_plasma_spl_and_dspl_dT(c->bg->plasma, T, &s, &ds_dT);
    double sb = s / (T * T * T);
    double dsbdT = ds_dT / (T * T * T) - 3.0 * s / (T * T * T * T);
    double N = cpr_nu_N_NEVO_of_Tg(&c->bg->nh, T);
    ydot[0] = -(3.0 * sb + T * dsbdT) / (N + 3.0 * sb);
    return 0;
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
 * reached). */
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
    ydot[0] = 1.0 / cpr_bg_Hubble(c->bg, Tg, Tnue, Tnumu, Tnutau);
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

    bg->external_scale_factor = cfg->external_scale_factor;
    if (bg->external_scale_factor) {
        bg->K_ext = a_end / cpr_nu_x_of_Tg(&bg->nh, Tend);
    } else {
        DlnaDlnTCtx ctx = { bg };
        CPRPath path;
        path_init(&path);
        double y[1] = { log(a_end) };
        CPRRKOpts opts = cpr_ode_rk_default_opts();
        /* Tolerance, NOT matched to Python's nominal 0.1*numerical_precision
         * (see BG_ODE_RTOL's docstring above): RK45 (explicit, Dormand-Prince)
         * and Python's LSODA achieve materially different *actual* accuracy
         * at the same *nominal* rtol for this ODE -- confirmed empirically by
         * running the full small/large+amax=8 BBN solve (nuclear_network.c)
         * with this tolerance at the Python-nominal value vs. progressively
         * tighter ones: at 0.1*1e-7=1e-8 the resulting YP(BBN)/D-H/Yn were off
         * by -0.14%/-1.8%/-3.5% from CLAUDE.md's reference numbers (BBN
         * abundances are exponentially sensitive to T(t)/a(T) near freeze-out,
         * so this small a(T) error is greatly amplified downstream); at
         * BG_ODE_RTOL/BG_ODE_ATOL below the same comparison is within
         * 0.002%/0.001%/0.005% -- inside CLAUDE.md's stated +-1e-5 (YP) and
         * +-3e-9 (D/H) bounds. Retightening ode_bdf.c (a *different*,
         * unrelated solver -- see the project memory note on that earlier,
         * abandoned hypothesis) made no measurable difference; this a(T)/t(a)
         * background accuracy was the actual bottleneck all along. Decoupled
         * from cfg->numerical_precision (rather than e.g. dividing it by a
         * fixed factor) because this ODE is 1-dimensional, smooth, and cheap
         * regardless of tolerance -- there is no performance reason to ever
         * loosen it, even for a fast/rough run; a user wanting an even higher-
         * precision *reference* run already has other knobs for that (see
         * CLAUDE.md's "Validation before committing" reference-run setup). */
        opts.rtol = BG_ODE_RTOL;
        opts.atol = BG_ODE_ATOL;
        char *err = NULL;
        int rc = cpr_ode_rk45(dlnadlnT_rhs, &ctx, log(Tend), log(Tstartcosmo), y, 1, opts,
                               path_step_cb, &path, &err);
        if (rc) {
            path_free(&path); free(T_sol);
            *errmsg = err;
            return 1;
        }
        for (size_t i = 0; i < bg->n_Tsol; i++)
            bg->lna_sol[i] = path_eval(&path, bg->lnT_sol[i]);
        path_free(&path);
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

    double Tnue_s = cpr_nu_Tnue_of_Tg(&bg->nh, Tstartcosmo);
    double Tnumu_s = cpr_nu_Tnumu_of_Tg(&bg->nh, Tstartcosmo);
    double Tnutau_s = cpr_nu_Tnutau_of_Tg(&bg->nh, Tstartcosmo);
    double t_ini = 1.0 / (2.0 * cpr_bg_Hubble(bg, Tstartcosmo, Tnue_s, Tnumu_s, Tnutau_s));

    bg->n_bg = bg->n_Tsol; /* same grid density for the t(a) integration */
    double *lna_samp = malloc(bg->n_bg * sizeof(double));
    for (size_t i = 0; i < bg->n_bg; i++) {
        double frac = (bg->n_bg == 1) ? 0.0 : (double)i / (double)(bg->n_bg - 1);
        lna_samp[i] = log(a_ini) + frac * (log(a_fin) - log(a_ini));
    }

    /* See DtDlnaCtx's docstring above: a smooth spline over the same nodes
     * cpr_bg_T_of_a interpolates linearly, used only as this ODE's RHS. */
    CPRCubicSpline T_of_a_smooth;
    char *spl_err = NULL;
    if (cpr_cubic_spline_fit_notaknot(bg->a_sol_asc, bg->T_sol_asc, bg->n_Tsol,
                                       &T_of_a_smooth, &spl_err)) {
        free(T_sol); free(lna_samp);
        *errmsg = spl_err;
        return 1;
    }
    DtDlnaCtx tctx = { bg, &T_of_a_smooth };
    CPRPath tpath;
    path_init(&tpath);
    double yt[1] = { t_ini };
    CPRRKOpts topts = cpr_ode_rk_default_opts();
    /* Same fixed-tolerance rationale as the a(T) ODE above (BG_ODE_RTOL's
     * docstring) -- this t(a) integration is the other half of the
     * empirically-confirmed bottleneck. */
    topts.rtol = BG_ODE_RTOL;
    topts.atol = BG_ODE_ATOL;
    char *terr = NULL;
    int trc = cpr_ode_rk45(dtdlna_rhs, &tctx, log(a_ini), log(a_fin), yt, 1, topts,
                            path_step_cb, &tpath, &terr);
    cpr_cubic_spline_free(&T_of_a_smooth);
    if (trc) {
        path_free(&tpath); free(T_sol); free(lna_samp);
        *errmsg = terr;
        return 1;
    }

    bg->t_vec = malloc(bg->n_bg * sizeof(double));
    bg->a_vec = malloc(bg->n_bg * sizeof(double));
    bg->Tg_vec = malloc(bg->n_bg * sizeof(double));
    bg->Tnue_vec = malloc(bg->n_bg * sizeof(double));
    bg->Tnumu_vec = malloc(bg->n_bg * sizeof(double));
    bg->Tnutau_vec = malloc(bg->n_bg * sizeof(double));
    bg->Tnu_vec = malloc(bg->n_bg * sizeof(double));
    for (size_t i = 0; i < bg->n_bg; i++) {
        bg->t_vec[i] = path_eval(&tpath, lna_samp[i]);
        double a = exp(lna_samp[i]);
        bg->a_vec[i] = a;
        bg->Tg_vec[i] = cpr_bg_T_of_a(bg, a);
        bg->Tnue_vec[i] = cpr_nu_Tnue_of_Tg(&bg->nh, bg->Tg_vec[i]);
        bg->Tnumu_vec[i] = cpr_nu_Tnumu_of_Tg(&bg->nh, bg->Tg_vec[i]);
        bg->Tnutau_vec[i] = cpr_nu_Tnutau_of_Tg(&bg->nh, bg->Tg_vec[i]);
        double e4 = (pow(bg->Tnue_vec[i], 4.0) + pow(bg->Tnumu_vec[i], 4.0)
                     + pow(bg->Tnutau_vec[i], 4.0)) / 3.0;
        bg->Tnu_vec[i] = pow(e4, 0.25);
    }
    path_free(&tpath);
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
    bg->lcdm_use_exact = 1; /* _replace_LCDM_with_exact: a_of_T is now ready */

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
     * PyPRConfig.__init__'s warning/forced-False for this combination) --
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
