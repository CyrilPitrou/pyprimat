/* weak_rates.c -- see cprimat/weak_rates.h.
 *
 * Direct port of pyprimat/weak_rates/{integrands,corrections,api}.py.
 * Every correction term below is evaluated scalar-at-a-time (one T value),
 * looped over the rate-table grid in cpr_weak_rates_init -- the Python
 * source vectorises over T with numpy for speed, which has no equivalent
 * benefit in C; the formulas are otherwise identical term-for-term.
 */
#include "cprimat/weak_rates.h"
#include "cprimat/constants.h"
#include "cprimat/cache.h"
#include "cprimat/table_io.h"
#include "cprimat/spline.h"
#include "cprimat/quad.h"

#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const double EXP_CUTOFF = 3.0e2; /* integrands.exp_cutoff */

/* Below this temperature the CCRTh thermal correction is clamped to exactly
 * 0 (see L_CCRTh_compute's docstring for the physics/numerics reasons). The
 * nTOp_thermal_<hash>.txt cache grid is built down to this fixed floor
 * rather than down to cfg.T_end (mirrors corrections.py's _T_CCRTH_MIN):
 * the integral is never actually evaluated below it regardless of
 * cfg.T_end_MeV, so letting the grid depend on T_end_MeV only caused
 * spurious cache misses for runs that changed T_end_MeV alone. */
#define CCRTH_T_MIN (pow(10.0, 8.2))  /* [K] */

/* ------------------------------------------------------------------------
 * FD_* integrand kernels (integrands.py). Scalar ports: the Python np.where
 * tail-cutoff guards become plain if/else, which is equivalent (and avoids
 * computing the discarded branch at all, so no np.minimum clamp is needed).
 * ------------------------------------------------------------------------ */

static double FD_nu3(double E, double phi, double x)
{
    double arg = x * E - phi;
    if (arg >= EXP_CUTOFF) return 0.0;
    return 1.0 / (exp(arg) + 1.0);
}

static double FD2(double E, double x)
{
    double arg = x * E;
    if (arg >= EXP_CUTOFF) return 0.0;
    return 1.0 / (exp(arg) + 1.0);
}

static double FD_nu_e2p0(double E, double phi, double x)
{
    double arg = x * E - phi;
    if (arg >= EXP_CUTOFF) return 0.0;
    return E * E / (exp(arg) + 1.0);
}

static double FD_nu_e3p0(double E, double phi, double x)
{
    double arg = x * E - phi;
    if (arg >= EXP_CUTOFF) return 0.0;
    return E * E * E / (exp(arg) + 1.0);
}

static double FD_nu_e4p2(double E, double phi, double x)
{
    double Ex = E * x;
    if (!(2.0 * phi < EXP_CUTOFF && Ex + phi < EXP_CUTOFF && 2.0 * Ex < EXP_CUTOFF))
        return 0.0;
    double eEx = exp(Ex), ephi = exp(phi);
    return E * E * ephi * ((24.0 - Ex * (Ex + 8.0)) * eEx * ephi
                            + eEx * eEx * (Ex - 6.0) * (Ex - 2.0)
                            + 12.0 * ephi * ephi)
           / pow(eEx + ephi, 3.0);
}

static double FD_nu_e2p2(double E, double phi, double x)
{
    double Ex = E * x;
    if (!(3.0 * phi < EXP_CUTOFF && 2.0 * Ex + phi < EXP_CUTOFF && Ex < EXP_CUTOFF))
        return 0.0;
    double eEx = exp(Ex), ephi = exp(phi);
    return ((Ex * (Ex - 4.0) + 2.0) * eEx * eEx * ephi
            + (4.0 - Ex * (Ex + 4.0)) * eEx * ephi * ephi
            + 2.0 * ephi * ephi * ephi)
           / pow(eEx + ephi, 3.0);
}

static double FD_nu_e4p1(double E, double phi, double x)
{
    double Ex = E * x;
    if (!(phi < EXP_CUTOFF && Ex < EXP_CUTOFF)) return 0.0;
    double eEx = exp(Ex), ephi = exp(phi);
    return ephi * E * E * E * (4.0 * ephi + eEx * (4.0 - Ex)) / ((eEx + ephi) * (eEx + ephi));
}

static double FD_nu_e2p1(double E, double phi, double x)
{
    double Ex = E * x;
    if (!(phi < EXP_CUTOFF && Ex < EXP_CUTOFF)) return 0.0;
    double eEx = exp(Ex), ephi = exp(phi);
    return ephi * E * (2.0 * ephi + eEx * (2.0 - Ex)) / ((eEx + ephi) * (eEx + ephi));
}

static double FD_nu_e3p1(double E, double phi, double x)
{
    double Ex = E * x;
    if (!(phi < EXP_CUTOFF && Ex < EXP_CUTOFF)) return 0.0;
    double eEx = exp(Ex), ephi = exp(phi);
    return ephi * E * E * (3.0 * ephi + eEx * (3.0 - Ex)) / ((eEx + ephi) * (eEx + ephi));
}

static double FD_nu_e3p2(double E, double phi, double x)
{
    double Ex = E * x;
    if (!(2.0 * phi < EXP_CUTOFF && Ex + phi < EXP_CUTOFF && 2.0 * Ex < EXP_CUTOFF))
        return 0.0;
    double eEx = exp(Ex), ephi = exp(phi);
    return E * ephi
           * ((12.0 - Ex * (Ex + 6.0)) * eEx * ephi
              + eEx * eEx * (Ex * (Ex - 6.0) + 6.0)
              + 6.0 * ephi * ephi)
           / pow(eEx + ephi, 3.0);
}

/* ------------------------------------------------------------------------
 * Real dilogarithm Li2(y) for y in [0,1] (scipy.special.spence(x) ==
 * Li2(1-x) for real x in (0,1), verified numerically against mpmath; see
 * the design notes in CPLAN.md's weak_rates section). Series for y<=0.5,
 * Euler reflection Li2(y) = pi^2/6 - ln(y)ln(1-y) - Li2(1-y) otherwise.
 * ------------------------------------------------------------------------ */

static double li2_series(double y)
{
    double sum = 0.0, term = y;
    for (int k = 1; k <= 200; k++) {
        sum += term / (k * (double)k);
        term *= y;
    }
    return sum;
}

static double li2(double y)
{
    if (y <= 0.5) return li2_series(y);
    double pi = 3.14159265358979323846;
    return pi * pi / 6.0 - log(y) * log(1.0 - y) - li2_series(1.0 - y);
}

/* scipy.special.spence(x) for real x in (0,1). */
static double spence_real(double x) { return li2(1.0 - x); }

/* ------------------------------------------------------------------------
 * Complex log-Gamma function (Lanczos approximation, g=7, n=9 coefficients
 * -- the standard published set, e.g. Numerical Recipes / Boost), needed
 * for |Gamma(1+Gamma_rel + i*alpha/b)|^2 in FermiCoulomb. We work in log
 * space (returning ln Gamma(z) rather than Gamma(z) itself) because
 * FermiCoulomb's near-threshold limit (b -> 0, i.e. the Coulomb parameter
 * y = alpha/b -> infinity) is a textbook case of two individually
 * divergent factors -- exp(pi*y) (-> +inf) and |Gamma(1+iy)|^2 ~ 2*pi*y*
 * exp(-pi*y) (-> 0) -- whose *product* is finite (the standard Sommerfeld/
 * Gamow factor, ~2*pi*y for y->infinity). Computing exp(pi*y) and |Gamma|^2
 * separately in double precision overflows/underflows to inf/0 long before
 * they would cancel (observed numerically for b ~< 1e-4, i.e. far inside
 * the b-range actually swept by the rate integrals near e -> 1); combining
 * them as exp(pi*y + 2*Re(ln Gamma(z))) keeps the exponent itself bounded
 * (~ln(2*pi*y), not ~pi*y) and is exact for any b > 0, including b -> 0.
 * Re(z) is always close to 1 here (>0.5), so the reflection branch is
 * included for robustness but never actually exercised by FermiCoulomb's
 * call site.
 * ------------------------------------------------------------------------ */

static double complex clgamma(double complex z)
{
    static const double g = 7.0;
    static const double p[9] = {
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    };
    if (creal(z) < 0.5) {
        const double pi = 3.14159265358979323846;
        return clog(pi) - clog(csin(pi * z)) - clgamma(1.0 - z);
    }
    z -= 1.0;
    double complex x = p[0];
    for (int i = 1; i < 9; i++) x += p[i] / (z + (double)i);
    double complex t = z + g + 0.5;
    return 0.5 * clog(2.0 * 3.14159265358979323846) + (z + 0.5) * clog(t) - t + clog(x);
}

/* ------------------------------------------------------------------------
 * FermiCoulomb / RadCorrResum / ComputeFn
 * ------------------------------------------------------------------------ */

double cpr_fermi_coulomb(double b, const CPRConfig *cfg)
{
    (void)cfg;
    double me = g_const.me * g_const.MeV;
    double Gamma = sqrt(1.0 - g_const.alphaem * g_const.alphaem) - 1.0;
    double gamma1 = 1.0 + Gamma;
    double gamma2 = 3.0 + 2.0 * Gamma;
    double Fn_Comp = g_const.hbar * g_const.clight / me;
    /* b -> 0 (e -> 1, beta-decay threshold) is a removable singularity:
     * b*FermiCoulomb(b) tends to a finite Sommerfeld/Gamow limit (the
     * caller's "e*b*FermiCoulomb(b)*..." rate-integral factor needs that
     * limit at the e=1 endpoint, which cpr_quad_adaptive evaluates
     * literally). Clamping b away from exactly 0 avoids 0/0 in y=alpha/b
     * while changing the (already huge, ~1e10-scale) y by a negligible
     * relative amount -- well below double precision's reach either way. */
    if (b < 1e-12) b = 1e-12;
    double y = g_const.alphaem / b;

    double complex lg = clgamma(gamma1 + y * I);
    /* exp(pi*y) * |Gamma(gamma1+iy)|^2, computed jointly (see comment above
     * clgamma) so the b->0 (y->infinity) limit stays a finite double. */
    double sommerfeld = exp(M_PI * y + 2.0 * creal(lg));

    return (1.0 + Gamma / 2.0)
           * 4.0 * pow((2.0 * g_const.radproton * b) / Fn_Comp, 2.0 * Gamma)
           / (tgamma(gamma2) * tgamma(gamma2))
           / pow(1.0 - b * b, Gamma)
           * sommerfeld;
}

double cpr_rad_corr_resum(double b, double y, double en, const CPRConfig *cfg)
{
    (void)cfg;
    const double mA = 1.2e3 * g_const.MeV;
    const double Agndecay = -0.34;
    const double Cndecay = 0.891;
    const double deltand = -0.00043;
    const double Lndecay = 1.02094;
    const double Sndecay = 1.02248;
    const double NLLndecay = -0.0001;

    double me = g_const.me * g_const.MeV;
    double mn = g_const.mn * g_const.MeV;
    double mp = g_const.mp * g_const.MeV;
    double Q = mn - mp;

    double b_safe = (b == 0.0) ? 1.0 : b;
    double Rd = (b == 0.0) ? 1.0 : (atanh(b_safe) / b_safe);

    /* y = E_nu/me -> 0 at the neutron-decay kinematic edge (E_e -> Q) is
     * another removable/integrable singularity: log(2*y) diverges but the
     * divergence is logarithmic (integrable), and ComputeFn's Fn_rad
     * integral (fn_rad_int) evaluates literally at that edge (e = Q/me),
     * same reasoning as the b->0 floor above. */
    if (y < 1e-12) y = 1e-12;

    double Sirlin = 3.0 * log(mp / me) - 3.0 / 4.0
                    + 4.0 * (Rd - 1.0) * (y / (3.0 * en) - 3.0 / 2.0 + log(2.0 * y))
                    + Rd * (2.0 * (1.0 + b * b) + y * y / (6.0 * en * en) - 4.0 * b * Rd)
                    - (4.0 / b) * spence_real(1.0 - (2.0 * b) / (1.0 + b));

    return (1.0 + g_const.alphaem / (2.0 * M_PI) * (Sirlin - 3.0 * log(mp / (2.0 * Q))))
           * (Lndecay + (g_const.alphaem / M_PI) * Cndecay
              + g_const.alphaem / (2.0 * M_PI) * deltand * 2.0 * M_PI / g_const.alphaem)
           * (Sndecay + 1.0 / (134.0 * 2.0 * M_PI) * (log(mp / mA) + Agndecay)
              + NLLndecay);
}

/* Adaptive-quad integrand contexts for ComputeFn. */
typedef struct { double Q_over_me; } FnBornCtx;
static double fn_born_int(double E, void *ctx_)
{
    FnBornCtx *ctx = ctx_;
    if (-1.0 >= E || E >= 1.0)
        return E * (E - ctx->Q_over_me) * (E - ctx->Q_over_me) * sqrt(E * E - 1.0);
    return 0.0;
}

typedef struct { const CPRConfig *cfg; double Q_over_me; } FnRadCtx;
/* Floors b away from exactly 0: cpr_quad_adaptive evaluates its integrand
 * literally at the e=1/pe=0 endpoint (Born threshold), where b=0 is a
 * removable singularity for b*FermiCoulomb(b) (see the comment above
 * clgamma). cpr_fermi_coulomb already floors its *internal* b the same
 * way; flooring it here too, before using it in the outer "b * F(b)"
 * product, keeps the two consistent so the product reaches its true
 * finite limit instead of evaluating to 0 * (huge) = 0. */
static double bfloor(double b) { return (b < 1e-12) ? 1e-12 : b; }

static double fn_rad_int(double e, void *ctx_)
{
    FnRadCtx *ctx = ctx_;
    double b = bfloor(sqrt(e * e - 1.0) / e);
    double q = ctx->Q_over_me;
    double F = cpr_fermi_coulomb(b, ctx->cfg);
    double R = cpr_rad_corr_resum(b, q - e, e, ctx->cfg);
    return e * (e - q) * (e - q) * e * b * F * R;
}

typedef struct {
    double me, mn, mp, Q, gA, deltakappa;
} ChiFMnDecCtx;
static double chi_fm_n_dec(const ChiFMnDecCtx *c, double en, double pe)
{
    double gA = c->gA, dk = c->deltakappa;
    double f1n = ((1.0 + gA) * (1.0 + gA) + 2.0 * dk * gA) / (1.0 + 3.0 * gA * gA);
    double f2n = ((1.0 - gA) * (1.0 - gA) - 2.0 * dk * gA) / (1.0 + 3.0 * gA * gA);
    double f3n = (gA * gA - 1.0) / (1.0 + 3.0 * gA * gA);
    double mnOme = c->mp / c->me; /* M_sgnq(+1) = mp/me, see header docstring */
    double d = en - c->Q / c->me;
    return f1n * d * d * (pe * pe / (mnOme * en))
           - f2n / mnOme * d * d * d
           + (f1n + f2n + f3n) / (2.0 * mnOme) * (4.0 * d * d * d + 2.0 * d * pe * pe)
           + f3n / mnOme * d * d * pe * pe / en;
}

typedef struct { ChiFMnDecCtx chi; } FnFMNoCCRCtx;
static double fn_fm_noccr_int(double pe, void *ctx_)
{
    FnFMNoCCRCtx *ctx = ctx_;
    double en = sqrt(pe * pe + 1.0);
    return pe * pe * chi_fm_n_dec(&ctx->chi, en, pe);
}

typedef struct { ChiFMnDecCtx chi; const CPRConfig *cfg; double Q_over_me; } FnFMCtx;
static double fn_fm_int(double pe, void *ctx_)
{
    FnFMCtx *ctx = ctx_;
    double en = sqrt(pe * pe + 1.0);
    double b = bfloor(pe / en);
    return pe * pe
           * chi_fm_n_dec(&ctx->chi, en, pe)
           * cpr_rad_corr_resum(b, fabs(en - ctx->Q_over_me), en, ctx->cfg)
           * cpr_fermi_coulomb(b, ctx->cfg);
}

double cpr_compute_fn(const CPRConfig *cfg)
{
    double me = g_const.me * g_const.MeV;
    double mn = g_const.mn * g_const.MeV;
    double mp = g_const.mp * g_const.MeV;
    double Q = mn - mp;
    double Q_over_me = Q / me;

    FnBornCtx born_ctx = { Q_over_me };
    double Fn_Born = cpr_quad_adaptive(fn_born_int, &born_ctx, 1.0, Q_over_me, 1e-12, 40, NULL);

    double gA = g_const.gA, deltakappa = cpr_deltakappa();

    if (!cfg->radiative_corrections) {
        if (!cfg->finite_mass_corrections) return Fn_Born;
        ChiFMnDecCtx chi = { me, mn, mp, Q, gA, deltakappa };
        FnFMNoCCRCtx ctx = { chi };
        double pmax = sqrt(Q_over_me * Q_over_me - 1.0);
        double Fn_FM_NoCCR = cpr_quad_adaptive(fn_fm_noccr_int, &ctx, 0.0, pmax, 1e-12, 40, NULL);
        return Fn_Born + Fn_FM_NoCCR;
    }

    FnRadCtx rad_ctx = { cfg, Q_over_me };
    double Fn_rad = cpr_quad_adaptive(fn_rad_int, &rad_ctx, 1.0, Q_over_me, 1e-12, 40, NULL);

    if (!cfg->finite_mass_corrections) return Fn_rad;

    ChiFMnDecCtx chi = { me, mn, mp, Q, gA, deltakappa };
    FnFMCtx fm_ctx = { chi, cfg, Q_over_me };
    double pmax = sqrt(Q_over_me * Q_over_me - 1.0);
    double Fn_FM = cpr_quad_adaptive(fn_fm_int, &fm_ctx, 0.0, pmax, 1e-12, 40, NULL);
    return Fn_rad + Fn_FM;
}

/* ------------------------------------------------------------------------
 * Gauss-Legendre rate-integral grid (_N_GL = 160, _quad_grid).
 * ------------------------------------------------------------------------ */

#define N_GL 160
static double GL_NODES[N_GL], GL_WEIGHTS[N_GL];
static int gl_ready = 0;
static void ensure_gl(void)
{
    if (!gl_ready) { cpr_gauss_legendre(N_GL, GL_NODES, GL_WEIGHTS); gl_ready = 1; }
}

/* Shared per-call context, mirrors _RateContext (minus my_dir/T_nuOverT,
 * which the caller resolves once per T value instead of via an interp1d
 * closure -- see cpr_weak_rates_init). */
typedef struct {
    const CPRConfig *cfg;
    const CPRNeutrinoHistory *nh;
    double me, mn, mp, Q, xi_nu, gA, deltakappa;
} RateCtx;

/* chi_+/-(E) (Phys. Rep. Eq. 81). */
static double chi_func(const RateCtx *ctx, double E, double x, double znu, double sgnq)
{
    double enu = E - sgnq * (ctx->Q / ctx->me);
    return FD_nu3(enu, sgnq * ctx->xi_nu, znu) * FD2(-E, x) * enu * enu;
}

/* FermiCoulomb(b) if the produced lepton is the electron (sgnq*sgnE>0), else 1. */
static double fermi_stat(const RateCtx *ctx, double sgnq, double sgnE, double b)
{
    return (sgnq * sgnE > 0.0) ? cpr_fermi_coulomb(b, ctx->cfg) : 1.0;
}

/* chi_FM (Fokker-Planck finite-mass correction to chi_+/-, Phys. Rep. SIII.G). */
static double chi_func_fm_v(const RateCtx *ctx, double en, double pe, double x,
                             double znu, double sgnq)
{
    double me = ctx->me, mn = ctx->mn, mp = ctx->mp, Q = ctx->Q;
    double gA = ctx->gA, dk = ctx->deltakappa;
    double M_sgnq = (mp + mn - sgnq * Q) / (2.0 * me);
    double f_1 = ((1.0 + sgnq * gA) * (1.0 + sgnq * gA) + 2.0 * dk * sgnq * gA) / (1.0 + 3.0 * gA * gA);
    double f_2 = ((1.0 - sgnq * gA) * (1.0 - sgnq * gA) - 2.0 * dk * sgnq * gA) / (1.0 + 3.0 * gA * gA);
    double f_3 = (gA * gA - 1.0) / (1.0 + 3.0 * gA * gA);
    double enu = en - sgnq * Q / me;
    double FD2_en = FD2(-en, x);

    double term =
        f_1 * FD_nu_e2p0(enu, 0.0, znu) * FD2_en * (pe * pe / (M_sgnq * en))
        + f_2 * FD_nu_e3p0(enu, 0.0, znu) * FD2_en * (-1.0 / M_sgnq)
        + (f_1 + f_2 + f_3) / (2.0 * x * M_sgnq)
          * (FD_nu_e4p2(enu, 0.0, znu) * FD2_en + FD_nu_e2p2(enu, 0.0, znu) * FD2_en * pe * pe)
        + (f_1 + f_2 + f_3) / (2.0 * M_sgnq)
          * (FD_nu_e4p1(enu, 0.0, znu) * FD2_en + FD_nu_e2p1(enu, 0.0, znu) * FD2_en * pe * pe)
        - (f_1 + f_2) / (x * M_sgnq)
          * (FD_nu_e3p1(enu, 0.0, znu) * FD2_en + FD_nu_e2p1(enu, 0.0, znu) * FD2_en * pe * pe / (-en))
        - f_3 * 3.0 / (x * M_sgnq) * FD_nu_e2p0(enu, 0.0, znu) * FD2_en
        + f_3 / (3.0 * M_sgnq) * FD_nu_e3p1(enu, 0.0, znu) * FD2_en * pe * pe / en
        + f_3 * 2.0 / (2.0 * x * 3.0 * M_sgnq) * FD_nu_e3p2(enu, 0.0, znu) * FD2_en * pe * pe / en
        - (f_1 + f_2 + f_3) * 3.0 / (2.0 * x) * (1.0 - pow(mn / mp, sgnq))
          * (FD_nu_e2p1(enu, 0.0, znu) * FD2_en);
    return term;
}

/* Per-panel GL node/weight at index i in [0, N_GL), mapped from [-1,1] to [lo,hi]. */
static void panel_point(int i, double lo, double hi, double *p, double *w)
{
    double half = 0.5 * (hi - lo);
    *p = lo + half * (GL_NODES[i] + 1.0);
    *w = half * GL_WEIGHTS[i];
}

/* Evaluates one of the correction terms (Born/CCR/FMCCR/FMNoCCR/SD/SD_CCR)
 * at a single photon temperature T [K], summing the two GL panels
 * (_quad_grid's "panel A"=[0,p_edge], "panel B"=[p_edge,p_max(T)]).
 * `kind` selects which integrand (mirrors _L_BORN/_L_CCR/_L_FMCCR/
 * _L_FMNoCCR/_L_SD/_L_SD_CCR). */
typedef enum { L_BORN, L_CCR, L_FMCCR, L_FMNOCCR, L_SD, L_SD_CCR } LKind;

double cpr_weak_rate_nTOp(const CPRWeakRates *wr, double T_K)
{
    double v = cpr_interp_quadratic_local(wr->T, wr->frwrd, wr->n, T_K);
    /* wr->T_th's lowest knot is CCRTH_T_MIN, not T_end: below that floor the
     * correction is pinned to 0 here rather than left to
     * cpr_interp_quadratic_local's quadratic extrapolation, which is
     * unconstrained there (mirrors corrections.py's _clamp_below_floor). */
    if (wr->has_thermal && T_K >= CCRTH_T_MIN)
        v += cpr_interp_quadratic_local(wr->T_th, wr->Lnth, wr->n_th, T_K);
    return v;
}

double cpr_weak_rate_pTOn(const CPRWeakRates *wr, double T_K)
{
    double v = cpr_interp_quadratic_local(wr->T, wr->bkwrd, wr->n, T_K);
    if (wr->has_thermal && T_K >= CCRTH_T_MIN)
        v += cpr_interp_quadratic_local(wr->T_th, wr->Lpth, wr->n_th, T_K);
    return v;
}

void cpr_weak_rates_free(CPRWeakRates *wr)
{
    free(wr->T); free(wr->frwrd); free(wr->bkwrd);
    free(wr->T_th); free(wr->Lnth); free(wr->Lpth);
    memset(wr, 0, sizeof(*wr));
}

static int n_points_per_decade(double per_decade, double T_lo, double T_hi)
{
    double decades = log10(T_hi / T_lo);
    int n = (int)lround(per_decade * decades);
    return n < 2 ? 2 : n;
}

/* T_nu(T_gamma)/T_gamma interpolant built from the background arrays
 * (mirrors _build_rate_context's T_nuOverT, interp1d kind='linear'). */
typedef struct { double *Tg_K, *ratio; size_t n; } TNuOverTCtx;

static double tnu_over_t(const TNuOverTCtx *c, double Tg_K)
{
    return cpr_interp_linear(c->Tg_K, c->ratio, c->n, Tg_K, CPR_EXTRAP_LINEAR);
}

/* ---- _quad_grid + correction-term evaluation at a single T (replaces the
 * placeholder eval_L stub above; kept as one function body for locality). */
static double eval_term(const RateCtx *ctx, LKind kind, double T_K, double sgnq,
                          const TNuOverTCtx *tnu_ctx)
{
    ensure_gl();
    double me = ctx->me, Q = ctx->Q;
    double x = me / (g_const.kB * T_K);
    double Tnu_K = T_K * tnu_over_t(tnu_ctx, T_K);
    double xnu = me / (g_const.kB * Tnu_K);

    double pmax = fmax(7.0, 30.0 / x);
    double p_edge = sqrt((Q / me) * (Q / me) - 1.0);

    double sum = 0.0;
    for (int panel = 0; panel < 2; panel++) {
        double lo = (panel == 0) ? 0.0 : p_edge;
        double hi = (panel == 0) ? p_edge : pmax;
        for (int i = 0; i < N_GL; i++) {
            double p, w;
            panel_point(i, lo, hi, &p, &w);
            double E = sqrt(p * p + 1.0);
            double integ;
            switch (kind) {
            case L_BORN:
                integ = p * p * (chi_func(ctx, E, x, xnu, sgnq) + chi_func(ctx, -E, x, xnu, sgnq));
                break;
            case L_CCR: {
                double b = p / E;
                integ = p * p * (chi_func(ctx, E, x, xnu, sgnq)
                                  * cpr_rad_corr_resum(b, fabs(sgnq * Q / me - E), E, ctx->cfg)
                                  * fermi_stat(ctx, sgnq, 1.0, b)
                                  + chi_func(ctx, -E, x, xnu, sgnq)
                                    * cpr_rad_corr_resum(b, fabs(sgnq * Q / me + E), E, ctx->cfg)
                                    * fermi_stat(ctx, sgnq, -1.0, b));
                break;
            }
            case L_FMCCR: {
                double b = p / E;
                integ = p * p * (chi_func_fm_v(ctx, E, p, x, xnu, sgnq)
                                  * cpr_rad_corr_resum(b, fabs(sgnq * Q / me - E), E, ctx->cfg)
                                  * fermi_stat(ctx, sgnq, 1.0, b)
                                  + chi_func_fm_v(ctx, -E, p, x, xnu, sgnq)
                                    * cpr_rad_corr_resum(b, fabs(sgnq * Q / me + E), E, ctx->cfg)
                                    * fermi_stat(ctx, sgnq, -1.0, b));
                break;
            }
            case L_FMNOCCR:
                integ = p * p * (chi_func_fm_v(ctx, E, p, x, xnu, sgnq)
                                  + chi_func_fm_v(ctx, -E, p, x, xnu, sgnq));
                break;
            case L_SD: {
                double enu_p = E - sgnq * (Q / me);
                double enu_m = -E - sgnq * (Q / me);
                double dchi_p = cpr_nu_dFDneu(ctx->nh, enu_p, x, xnu, sgnq) * FD2(-E, x) * enu_p * enu_p;
                double dchi_m = cpr_nu_dFDneu(ctx->nh, enu_m, x, xnu, sgnq) * FD2(E, x) * enu_m * enu_m;
                integ = p * p * (dchi_p + dchi_m);
                break;
            }
            case L_SD_CCR: {
                double b = p / E;
                double enu_p = E - sgnq * (Q / me);
                double enu_m = -E - sgnq * (Q / me);
                double dchi_p = cpr_nu_dFDneu(ctx->nh, enu_p, x, xnu, sgnq) * FD2(-E, x) * enu_p * enu_p;
                double dchi_m = cpr_nu_dFDneu(ctx->nh, enu_m, x, xnu, sgnq) * FD2(E, x) * enu_m * enu_m;
                integ = p * p * (dchi_p
                                  * cpr_rad_corr_resum(b, fabs(sgnq * Q / me - E), E, ctx->cfg)
                                  * fermi_stat(ctx, sgnq, 1.0, b)
                                  + dchi_m
                                    * cpr_rad_corr_resum(b, fabs(sgnq * Q / me + E), E, ctx->cfg)
                                    * fermi_stat(ctx, sgnq, -1.0, b));
                break;
            }
            default:
                integ = 0.0;
            }
            sum += w * integ;
        }
    }
    return sum;
}

/* Sums the active correction terms (mirrors _correction_terms) at one T,
 * for one direction (sgnq = +1: n->p, -1: p->n). Does NOT include CCRTh
 * (handled separately via the thermal cache, see cpr_weak_rates_init). */
static double nonthermal_rate_term(const RateCtx *ctx, double T_K, double sgnq,
                                     const TNuOverTCtx *tnu_ctx)
{
    const CPRConfig *cfg = ctx->cfg;
    double total = 0.0;
    if (cfg->radiative_corrections) {
        total += eval_term(ctx, L_CCR, T_K, sgnq, tnu_ctx);
        if (cfg->finite_mass_corrections)
            total += eval_term(ctx, L_FMCCR, T_K, sgnq, tnu_ctx);
    } else {
        total += eval_term(ctx, L_BORN, T_K, sgnq, tnu_ctx);
        if (cfg->finite_mass_corrections)
            total += eval_term(ctx, L_FMNOCCR, T_K, sgnq, tnu_ctx);
    }
    if (cfg->spectral_distortions) {
        if (cfg->radiative_corrections)
            total += eval_term(ctx, L_SD_CCR, T_K, sgnq, tnu_ctx);
        else
            total += eval_term(ctx, L_SD, T_K, sgnq, tnu_ctx);
    }
    return total;
}

/* ------------------------------------------------------------------------
 * Thermal radiative correction (CCRTh, Brown & Sawyer 2001), from scratch.
 *
 * Port of corrections._L_CCRTh_interpolants's compute-from-scratch branch.
 * Python uses vegas (Monte-Carlo importance sampling) when available, else
 * falls back to scipy.integrate.dblquad/quad (deterministic nested adaptive
 * quadrature) -- weak_rates.h's design note anticipated replacing both with
 * "a deterministic 2D adaptive quadrature"; this ports that dblquad/quad
 * fallback path exactly (same integrands, same rectangular domains), via
 * cpr_quad_adaptive nested inside itself for the 2D integrals. Every helper
 * below is scalar-at-a-time (the Python originals are numpy-vectorised over
 * a Monte-Carlo/quadrature batch; the formulas are otherwise identical
 * term-for-term, including the Fp/Fm asymmetry quirk in
 * IPENCCRDiffBremsstrahlung's res2 subtraction -- ported faithfully rather
 * than "fixed", since the goal is to reproduce Python's numbers).
 * ------------------------------------------------------------------------ */

/* Wrapper around cpr_quad_adaptive for the CCRTh sub-integrals below.
 *
 * scipy.integrate.quad/dblquad (the Python fallback this code mirrors) stop
 * on whichever of two criteria is *loosest*: a relative one (epsrel,
 * supplied explicitly as cfg.epsrel_thermal in the Python source) or an
 * absolute one (epsabs, left at scipy's own default of ~1.49e-8 -- never
 * overridden in corrections.py). Using a small fixed absolute tolerance
 * here (`epsrel * 1e-6`, since all these sub-integrands are O(1) or
 * smaller in the natural units of L_CCRTh, see the docstring above)
 * reproduces that floor and, on its own, is enough for cpr_quad_adaptive's
 * per-leaf Richardson-extrapolated Simpson refinement (quad.c) to converge
 * correctly -- verified directly against Python at 1e-10.
 *
 * But a single top-level cpr_quad_adaptive call over the *full* [a,b] can
 * still go wrong regardless of how tight that tolerance is, for a reason
 * unrelated to tolerance: several of these sub-integrands are
 * sign-changing and sharply localised within a domain that, at low T, is
 * much wider than the feature itself (e.g. IPENCCRT(E,k) here is
 * concentrated within a few units of E=1 inside an [1.001, E_max] domain
 * tens of units wide at T~2e8 K). If the feature happens to fall entirely
 * between the handful of sample points the *first* couple of recursion
 * levels place across the full domain, the "refined" and "whole" Simpson
 * estimates can both come out small *and similar* by coincidence (both
 * having missed it the same way), satisfying the Richardson stopping test
 * at depth 1 and never triggering the deeper bisection that would
 * eventually have landed a sample inside the feature. (Verified
 * numerically: this corrupted L_thermal_2d's true-photon term by 4 orders
 * of magnitude at T=2e8 K, where IPENCCRT changes sign within E in [1,3]
 * out of a domain extending to E_max=10, even with an absolute tol as
 * tight as 1e-8 on the *top-level* call.)
 *
 * The fix is the standard one for this failure mode: pre-partition [a,b]
 * into a fixed number of equal coarse panels before ever invoking the
 * recursive refinement, so the first level of sampling has a guaranteed
 * floor resolution across the whole domain regardless of where a narrow
 * feature sits. n_panels+1 evenly-spaced guaranteed sample points,
 * combined with max_depth additional recursion levels *within* each
 * panel, resolve any feature wider than roughly (b-a)/n_panels with the
 * original per-leaf accuracy intact, at n_panels times the function-
 * evaluation cost of a single top-level call.
 *
 * That multiplier matters here because two call sites below
 * (th2d_E_integrand and c23_outer_integrand) are themselves the per-sample
 * integrand of an *outer* quad_adaptive_relative call: paneling both
 * levels multiplies the panel counts together (n_panels_outer *
 * n_panels_inner), which made an earlier across-the-board choice of 32
 * panels at every call site cost tens of thousands of leaf evaluations
 * per (T, sgnq) point -- correct, but turning a single from-scratch
 * thermal-table build into a multi-minute run. Only the *outer* integral
 * of each nested pair is where the false-convergence failure above was
 * actually diagnosed (a feature narrow compared to the full domain); the
 * corresponding *inner* integral at fixed outer-sample (k at fixed E for
 * th2d_E_integrand; e1me2 at fixed e1pe2 for c23_outer_integrand) is a
 * single smooth, non-cancelling peak that cpr_quad_adaptive's own
 * recursive refinement already resolves correctly on its own (no panels
 * needed there, n_panels=1 is just a plain adaptive call). The 1D drivers
 * (L_thermal_1, L_thermal_2d's outer E, L_thermal_2_3's outer e1pe2) keep
 * a modest n_panels=8 -- enough margin over the few-unit feature widths
 * diagnosed above without the unnecessary 32x cost. */
static double quad_adaptive_relative_n(CPRQuadFunc f, void *ctx, double a, double b,
                                         double epsrel, int max_depth, int n_panels)
{
    double tol = epsrel * 1.0e-6;
    double h = (b - a) / n_panels;
    double sum = 0.0;
    for (int i = 0; i < n_panels; i++) {
        double lo = a + i * h, hi = (i == n_panels - 1) ? b : lo + h;
        sum += cpr_quad_adaptive(f, ctx, lo, hi, tol / n_panels, max_depth, NULL);
    }
    return sum;
}

static double quad_adaptive_relative(CPRQuadFunc f, void *ctx, double a, double b,
                                       double epsrel, int max_depth)
{
    return quad_adaptive_relative_n(f, ctx, a, b, epsrel, max_depth, 8);
}

static double th_A(double E, double k)
{
    double pE = sqrt(E * E - 1.0);
    return (2.0 * E * E + k * k) * log((E + pE) / (E - pE)) - 4.0 * pE * E;
}

static double th_B(double E)
{
    double pE = sqrt(E * E - 1.0);
    return 2.0 * E * log((E + pE) / (E - pE)) - 4.0 * pE;
}

/* Bose-Einstein occupation BE(E/kT); zeroed beyond EXP_CUTOFF (the photon
 * mode is then exponentially absent, matching IPENCCRT's BE(x*k) factor). */
static double th_BE(double EkBT)
{
    if (fabs(EkBT) >= EXP_CUTOFF) return 0.0;
    return 1.0 / (exp(EkBT) - 1.0);
}

/* Chitilde(en) = FD(znu*(en-sgnq*q) - sgnq*xi_nu) * (en-sgnq*q)^2, zeroed
 * (not clamped to the saturated FD value) outside |arg|<EXP_CUTOFF -- this
 * matches corrections.py's local Chitilde_vec exactly (including its
 * asymmetric zeroing of the Pauli-unblocked tail, not just the suppressed
 * one); see this function's module docstring there for context. */
static double th_chitilde(const RateCtx *ctx, double en, double znuval, double sgnq)
{
    double q = ctx->Q / ctx->me;
    double d = en - sgnq * q;
    double arg = znuval * d - sgnq * ctx->xi_nu;
    if (fabs(arg) >= EXP_CUTOFF) return 0.0;
    return (1.0 / (exp(arg) + 1.0)) * d * d;
}

/* d/dE of FD2(E,x) = 1/(exp(E*x)+1); zeroed beyond EXP_CUTOFF (derivative of
 * the saturated tail is 0 there anyway). */
static double th_d_fd2(double en, double xval)
{
    double arg = en * xval;
    if (fabs(arg) >= EXP_CUTOFF) return 0.0;
    double e = exp(arg);
    return -xval * e / ((e + 1.0) * (e + 1.0));
}

/* IPENCCRT: thermal "true photon absorption/emission" sub-integrand
 * (corrections.py's IPENCCRT), integrated over (E,k) in
 * [1.001,E_max]x[0.001,k_max]. */
static double ipen_ccrt(const RateCtx *ctx, double E, double k, double x, double znu, double sgnq)
{
    double pE = sqrt(E * E - 1.0);
    double term1 = th_A(E, k)
        * (FD2(-E, x) * fermi_stat(ctx, sgnq, 1.0, pE / E)
             * (th_chitilde(ctx, E - k, znu, sgnq) + th_chitilde(ctx, E + k, znu, sgnq)
                - 2.0 * th_chitilde(ctx, E, znu, sgnq))
           + FD2(E, x) * fermi_stat(ctx, sgnq, -1.0, pE / E)
             * (th_chitilde(ctx, -E + k, znu, sgnq) + th_chitilde(ctx, -E - k, znu, sgnq)
                - 2.0 * th_chitilde(ctx, -E, znu, sgnq)));
    double term2 = k * th_B(E)
        * (FD2(-E, x) * fermi_stat(ctx, sgnq, 1.0, pE / E)
             * (th_chitilde(ctx, E - k, znu, sgnq) - th_chitilde(ctx, E + k, znu, sgnq))
           + FD2(E, x) * fermi_stat(ctx, sgnq, -1.0, pE / E)
             * (th_chitilde(ctx, -E + k, znu, sgnq) - th_chitilde(ctx, -E - k, znu, sgnq)));
    return g_const.alphaem / (2.0 * M_PI) * (th_BE(x * k) / k) * (term1 - term2);
}

/* IPENCCRDiffBremsstrahlung: thermal differential-bremsstrahlung
 * sub-integrand, same (E,k) domain as ipen_ccrt. The infrared pole at
 * k -> 0 is meant to cancel against the explicit soft-photon subtraction
 * below (active only when |k| < |E -+ sgnq*q|); see the CCRTh clamp's
 * docstring at this term's driver (L_CCRTh_compute) for the residual that
 * survives this cancellation below ~10^8.2 K. */
static double ipen_ccr_diff_brems(const RateCtx *ctx, double E, double k, double x,
                                    double znu, double sgnq)
{
    double q = ctx->Q / ctx->me;
    double pE = sqrt(E * E - 1.0);
    double logterm = log((E + pE) / (E - pE));
    double base = (2.0 * E * E + k * k) * logterm - 4.0 * pE * E;
    double kshift = k * (2.0 * E * logterm - 4.0 * pE);
    double Fp = base + kshift;
    double Fm = base - kshift;

    double res1_fac = FD2(-E, x) * fermi_stat(ctx, sgnq, 1.0, pE / E);
    double res1 = Fp * th_chitilde(ctx, E + k, znu, sgnq);
    if (fabs(k) < fabs(E - sgnq * q))
        res1 -= Fp * FD2(E - sgnq * q, znu) * pow(fabs(E - sgnq * q) - k, 2.0);
    res1 *= res1_fac;

    double res2_fac = FD2(E, x) * fermi_stat(ctx, sgnq, -1.0, pE / E);
    double res2 = Fm * th_chitilde(ctx, -E + k, znu, sgnq);
    /* Python's own res2 subtraction re-uses Fp (not Fm) here -- kept as-is. */
    if (fabs(k) < fabs(E + sgnq * q))
        res2 -= Fp * FD2(-E - sgnq * q, znu) * pow(fabs(E + sgnq * q) - k, 2.0);
    res2 *= res2_fac;

    return g_const.alphaem / (2.0 * M_PI * k) * (res1 + res2);
}

/* C1dE: 1D thermal sub-integrand (corrections.py's C1dE), reusing chi_func
 * (identical formula to the Born chi_+/-(E)). C1dE(E) ~ E/pE * chi_sum(E)
 * diverges (as 1/sqrt(E-1)) at the integral's own lower bound E=1 -- an
 * integrable but *not* removable singularity (unlike the b->0 Sommerfeld
 * limit above): scipy.integrate.quad's open Gauss-Kronrod rule never
 * samples that exact endpoint and so never notices, but cpr_quad_adaptive's
 * Simpson rule does, and flooring pE there would just inject one
 * arbitrarily-large spurious endpoint sample into the Simpson estimate
 * (verified numerically: this previously inflated L_thermal_1 by 6-8
 * orders of magnitude). The fix actually used is below, in
 * c1_integrand_p: substitute the integration variable E -> pE = sqrt(E^2-1)
 * (dE = (pE/E) dpE), which makes the pE in the denominator cancel exactly
 * against the Jacobian, leaving a smooth (singularity-free) integrand. */
static double th_c1de_over_p_jacobian(const RateCtx *ctx, double E, double x, double znu, double sgnq)
{
    /* C1dE(E) * dE/dpE, with the E/pE * pE/E = E^0 cancellation already
     * applied algebraically (see the comment above): */
    return -(g_const.alphaem * M_PI) / (3.0 * x * x)
           * (chi_func(ctx, E, x, znu, sgnq) + chi_func(ctx, -E, x, znu, sgnq));
}

/* C2dE1dE2: 2D thermal sub-integrand (corrections.py's C2dE1dE2), with the
 * Python index_limits mask (restricting the physical (e1,e2) support inside
 * a generously-sized (e1pe2,e1me2) rectangle) becoming a direct early
 * return 0 outside that support. */
static double th_c2de1de2(const RateCtx *ctx, double e1, double e2, double x,
                            double znu, double sgnq)
{
    double e1me2 = e1 - e2;
    double e1pe2 = e1 + e2;
    double min_e1pe2 = 2.0 + fabs(e1me2);
    double max_e1pe2 = 2.0 + fmax(10.0, 15.0 / x) + fabs(e1me2);
    if (!((e1pe2 - min_e1pe2) > 0.0 && (max_e1pe2 - e1pe2) > 0.0))
        return 0.0;

    double p1 = sqrt(e1 * e1 - 1.0), p2 = sqrt(e2 * e2 - 1.0);
    double L_fac = log((e1 * e2 + p1 * p2 + 1.0) / (e1 * e2 - p1 * p2 + 1.0));
    double chi_sum = chi_func(ctx, e1, x, znu, sgnq) + chi_func(ctx, -e1, x, znu, sgnq);
    double ratio = (p1 + p2) / (p1 - p2);
    double log_ratio2 = log(ratio * ratio);
    double dfd2_e2 = th_d_fd2(e2, x);
    double fd2_e2 = FD2(e2, x);

    double term =
        -(1.0 / 4.0) * log_ratio2 * log_ratio2
            * (dfd2_e2 * p2 / p1 * e1 * e1 / e2 * (e1 + e2)
               + fd2_e2 * e1 * e1 / (p1 * p2) * (e2 + e1 / (e2 * e2)))
        + log_ratio2
            * (dfd2_e2 * (p2 * p2 * e1 / e2 * (1.0 / (p1 * p1) + 2.0) - e1 * e1 * p2 / p1 * L_fac)
               + fd2_e2 * (e1 / (p1 * p1 * e2 * e2) * (e2 * e2 + 2.0 * p1 * p1 + 1.0)
                            - (e1 * e1 + e2 * e2) / (e1 + e2)
                            - (e1 * e1 * e2) / (p1 * p2) * L_fac))
        - fd2_e2 * (4.0 * e1 * p2 / p1 + 2.0 * e2 * L_fac);

    return g_const.alphaem / (2.0 * M_PI) * chi_sum * term;
}

/* ---- Driver integrals (one nested-quadrature evaluation per (T,sgnq)). ---- */

typedef struct { const RateCtx *ctx; double E, x, znu, sgnq; } ThInnerCtx;
static double truephoton_k_integrand(double k, void *ctx_)
{
    ThInnerCtx *c = ctx_;
    return ipen_ccrt(c->ctx, c->E, k, c->x, c->znu, c->sgnq);
}
static double brems_k_integrand(double k, void *ctx_)
{
    ThInnerCtx *c = ctx_;
    return ipen_ccr_diff_brems(c->ctx, c->E, k, c->x, c->znu, c->sgnq);
}

typedef struct { const RateCtx *ctx; double x, znu, sgnq, k_max; int is_brems; } Th2DCtx;
static double th2d_E_integrand(double E, void *ctx_)
{
    Th2DCtx *c = ctx_;
    ThInnerCtx ic = { c->ctx, E, c->x, c->znu, c->sgnq };
    CPRQuadFunc f = c->is_brems ? brems_k_integrand : truephoton_k_integrand;
    return quad_adaptive_relative_n(f, &ic, 0.001, c->k_max, c->ctx->cfg->epsrel_thermal, 15, 1);
}

/* _L_ThermalTruePhoton (is_brems=0) / _L_ThermalDiffBremsstrahlung (is_brems=1):
 * both integrate over the same rectangular (E,k) domain, [1.001,E_max] x
 * [0.001,k_max] with E_max=k_max=max(10, 20/x). */
static double L_thermal_2d(const RateCtx *ctx, double x, double znu, double sgnq, int is_brems)
{
    double E_max = fmax(10.0, 20.0 / x);
    double k_max = E_max;
    Th2DCtx c = { ctx, x, znu, sgnq, k_max, is_brems };
    return quad_adaptive_relative(th2d_E_integrand, &c, 1.001, E_max, ctx->cfg->epsrel_thermal, 15);
}

typedef struct { const RateCtx *ctx; double x, znu, sgnq; } C1Ctx;
static double c1_integrand_p(double p, void *ctx_)
{
    C1Ctx *c = ctx_;
    double E = sqrt(p * p + 1.0);
    return th_c1de_over_p_jacobian(c->ctx, E, c->x, c->znu, c->sgnq);
}

/* _L_Thermal_1: 1D integral, E in [1, max(25, 150*kB*T/me)] i.e.
 * p=sqrt(E^2-1) in [0, sqrt(hi^2-1)] (see th_c1de_over_p_jacobian's comment
 * for why integrating over p rather than E). Python hardcodes epsrel=1e-2
 * here (not cfg.epsrel_thermal) -- kept as-is. */
static double L_thermal_1(const RateCtx *ctx, double T_K, double x, double znu, double sgnq)
{
    double hi = fmax(25.0, 150.0 * (g_const.kB * T_K) / ctx->me);
    double p_hi = sqrt(hi * hi - 1.0);
    C1Ctx c = { ctx, x, znu, sgnq };
    return quad_adaptive_relative(c1_integrand_p, &c, 0.0, p_hi, 1.0e-2, 15);
}

typedef struct { const RateCtx *ctx; double x, znu, sgnq; double e1pe2; } C23InnerCtx;
static double c23_inner_integrand(double e1me2, void *ctx_)
{
    C23InnerCtx *c = ctx_;
    double e1 = (c->e1pe2 + e1me2) / 2.0, e2 = (c->e1pe2 - e1me2) / 2.0;
    return 0.5 * th_c2de1de2(c->ctx, e1, e2, c->x, c->znu, c->sgnq);
}
typedef struct { const RateCtx *ctx; double x, znu, sgnq, me2_lo, me2_hi; } C23OuterCtx;
static double c23_outer_integrand(double e1pe2, void *ctx_)
{
    C23OuterCtx *c = ctx_;
    C23InnerCtx ic = { c->ctx, c->x, c->znu, c->sgnq, e1pe2 };
    return quad_adaptive_relative_n(c23_inner_integrand, &ic, c->me2_lo, c->me2_hi,
                                      c->ctx->cfg->epsrel_thermal, 15, 1);
}

/* _L_Thermal_2_3: sum of two 2D integrals (e1me2 < 0 and e1me2 > 0 halves),
 * sharing the same (e1pe2) outer rectangle bound -- see this file's CPLAN
 * derivation comment (in the corresponding Python docstring) for why both
 * branches reduce to the same outer limits despite the differing min/max
 * passed to dblquad in the Python source. */
static double L_thermal_2_3(const RateCtx *ctx, double x, double znu, double sgnq)
{
    double half = fmax(10.0, 15.0 / x);
    double lims_lo = 2.002, lims_hi = 2.0 + half;

    C23OuterCtx oc_neg = { ctx, x, znu, sgnq, -half, -0.001 };
    double res2 = quad_adaptive_relative(c23_outer_integrand, &oc_neg, lims_lo, lims_hi,
                                           ctx->cfg->epsrel_thermal, 15);

    C23OuterCtx oc_pos = { ctx, x, znu, sgnq, 0.001, half };
    double res3 = quad_adaptive_relative(c23_outer_integrand, &oc_pos, lims_lo, lims_hi,
                                           ctx->cfg->epsrel_thermal, 15);

    return res2 + res3;
}

/* L_CCRTh_compute: the full thermal correction at one (T,sgnq). See
 * corrections.py's _L_CCRTh_compute docstring for the physics/numerics
 * rationale of the T < 10^8.2 K clamp (an uncancelled IR residual in the
 * bremsstrahlung term that would otherwise spuriously pull the low-T rate
 * away from the free neutron-decay value). */
static double L_CCRTh_compute(const RateCtx *ctx, double T_K, double sgnq,
                                const TNuOverTCtx *tnu_ctx)
{
    if (T_K < CCRTH_T_MIN) return 0.0;
    double x = ctx->me / (g_const.kB * T_K);
    double Tnu_K = T_K * tnu_over_t(tnu_ctx, T_K);
    double znu = ctx->me / (g_const.kB * Tnu_K);
    return L_thermal_2d(ctx, x, znu, sgnq, 0)
         + L_thermal_2d(ctx, x, znu, sgnq, 1)
         + L_thermal_1(ctx, T_K, x, znu, sgnq)
         + L_thermal_2_3(ctx, x, znu, sgnq);
}

/* ------------------------------------------------------------------------
 * Public entry point.
 * ------------------------------------------------------------------------ */

int cpr_weak_rates_init(CPRWeakRates *wr, const double *Tg_MeV, const double *Tnu_MeV,
                         size_t n_bg, const CPRConfig *cfg, const CPRNeutrinoHistory *nh,
                         char **errmsg)
{
    memset(wr, 0, sizeof(*wr));
    cpr_constants_init();

    /* T_nu(T_gamma)/T_gamma interpolant, ascending in Tg [K] (mirrors
     * _build_rate_context's T_nuOverT). Input arrays may come in either
     * order from the caller's background grid; sort ascending if needed. */
    double *Tg_K = malloc(n_bg * sizeof(double));
    double *ratio = malloc(n_bg * sizeof(double));
    for (size_t i = 0; i < n_bg; i++) {
        Tg_K[i] = Tg_MeV[i] * cpr_MeV_to_Kelvin();
        ratio[i] = Tnu_MeV[i] / Tg_MeV[i];
    }
    if (n_bg >= 2 && Tg_K[0] > Tg_K[n_bg - 1]) {
        for (size_t i = 0; i < n_bg / 2; i++) {
            double t = Tg_K[i]; Tg_K[i] = Tg_K[n_bg - 1 - i]; Tg_K[n_bg - 1 - i] = t;
            double r = ratio[i]; ratio[i] = ratio[n_bg - 1 - i]; ratio[n_bg - 1 - i] = r;
        }
    }
    TNuOverTCtx tnu_ctx = { Tg_K, ratio, n_bg };

    double me = g_const.me * g_const.MeV;
    double mn = g_const.mn * g_const.MeV;
    double mp = g_const.mp * g_const.MeV;
    RateCtx ctx = { cfg, nh, me, mn, mp, mn - mp, cfg->munuOverTnu, g_const.gA, cpr_deltakappa() };

    double T_end = cpr_config_T_end(cfg);
    double T_start = cpr_T_start(); /* fixed 10 MeV era boundary, NOT T_start_cosmo */

    char nd[4200];
    snprintf(nd, sizeof(nd), "%s/rates/weak/", cfg->data_dir);

    CPRFPField fp_fields[24];
    size_t n_fp = cpr_weak_rate_fingerprint(cfg, fp_fields);
    char *fp_hash = cpr_fingerprint_hash(fp_fields, n_fp);
    char path[4300];
    snprintf(path, sizeof(path), "%snTOp_%s.txt", nd, fp_hash);

    int have_cache = cfg->weak_rate_cache != 0;
    if (have_cache) {
        FILE *f = fopen(path, "r");
        if (f) fclose(f); else have_cache = 0;
    }

    if (have_cache) {
        CPRTable tab;
        if (cpr_table_read(path, 3, &tab, errmsg)) {
            free(fp_hash); free(Tg_K); free(ratio);
            return 1;
        }
        wr->n = tab.n_rows;
        wr->T = malloc(wr->n * sizeof(double));
        wr->frwrd = malloc(wr->n * sizeof(double));
        wr->bkwrd = malloc(wr->n * sizeof(double));
        memcpy(wr->T, tab.cols[0], wr->n * sizeof(double));
        memcpy(wr->frwrd, tab.cols[1], wr->n * sizeof(double));
        memcpy(wr->bkwrd, tab.cols[2], wr->n * sizeof(double));
        cpr_table_free(&tab);
    } else {
        int n_pts = n_points_per_decade(cfg->sampling_nTOp_per_decade, T_end, T_start);
        wr->n = (size_t)n_pts;
        wr->T = malloc(wr->n * sizeof(double));
        wr->frwrd = malloc(wr->n * sizeof(double));
        wr->bkwrd = malloc(wr->n * sizeof(double));

        double logTlo = log10(T_end), logThi = log10(T_start);
        for (size_t i = 0; i < wr->n; i++) {
            double frac = (wr->n == 1) ? 0.0 : (double)i / (double)(wr->n - 1);
            wr->T[i] = pow(10.0, logTlo + frac * (logThi - logTlo));
        }

        double Fn = cpr_compute_fn(cfg);
        for (size_t i = 0; i < wr->n; i++) {
            double f = nonthermal_rate_term(&ctx, wr->T[i], +1.0, &tnu_ctx);
            double b = nonthermal_rate_term(&ctx, wr->T[i], -1.0, &tnu_ctx);
            wr->frwrd[i] = (f < 1e-28) ? 0.0 : f / Fn;
            wr->bkwrd[i] = (b < 1e-28) ? 0.0 : b / Fn;
        }

        if (cfg->save_nTOp) {
            double *cols[3] = { wr->T, wr->frwrd, wr->bkwrd };
            /* Best-effort: mkdir -p rates/weak/ then write; a failure to
             * persist the cache is not fatal (matches Python's os.makedirs
             * exist_ok=True followed by an unconditional write -- here we
             * simply don't treat a write error as fatal either). */
            char mkdir_cmd[4300];
            snprintf(mkdir_cmd, sizeof(mkdir_cmd), "%s", nd);
            /* create directory tree without relying on a shell call */
            for (char *p = mkdir_cmd + 1; *p; p++) {
                if (*p == '/') {
                    *p = '\0';
                    mkdir(mkdir_cmd, 0755);
                    *p = '/';
                }
            }
            mkdir(mkdir_cmd, 0755);
            cpr_cache_write(path, fp_fields, n_fp,
                             "T[K] Gamma_nTOp[1/tau_n] Gamma_pTOn[1/tau_n]",
                             cols, 3, wr->n);
        }
    }
    free(fp_hash);

    /* ---- Thermal correction (CCRTh): load from cache if present, else
     * compute from scratch via L_CCRTh_compute (Phase 3b). ---- */
    wr->has_thermal = 0;
    if (cfg->thermal_corrections) {
        CPRFPField th_fields[8];
        size_t n_th_fp = cpr_thermal_fingerprint(cfg, th_fields);
        char *th_hash = cpr_fingerprint_hash(th_fields, n_th_fp);
        char th_path[4300];
        snprintf(th_path, sizeof(th_path), "%snTOp_thermal_%s.txt", nd, th_hash);
        free(th_hash);

        FILE *f = fopen(th_path, "r");
        if (f) {
            fclose(f);
            CPRTable tab;
            if (cpr_table_read(th_path, 3, &tab, errmsg)) {
                free(Tg_K); free(ratio);
                cpr_weak_rates_free(wr);
                return 1;
            }
            wr->n_th = tab.n_rows;
            wr->T_th = malloc(wr->n_th * sizeof(double));
            wr->Lnth = malloc(wr->n_th * sizeof(double));
            wr->Lpth = malloc(wr->n_th * sizeof(double));
            memcpy(wr->T_th, tab.cols[0], wr->n_th * sizeof(double));
            memcpy(wr->Lnth, tab.cols[1], wr->n_th * sizeof(double));
            memcpy(wr->Lpth, tab.cols[2], wr->n_th * sizeof(double));
            cpr_table_free(&tab);
        } else {
            /* No matching cache file: compute CCRTh from scratch (deterministic
             * nested adaptive quadrature, see L_CCRTh_compute above) -- this is
             * the multi-minute-class computation the Python docstring warns
             * about, hence the same "may take a while" notice. */
            fprintf(stderr,
                    "[weak]   Re-evaluating n <--> p thermal corrections. "
                    "This may take a while ...\n");
            int n_th_pts = n_points_per_decade(cfg->sampling_nTOp_thermal_per_decade,
                                                 CCRTH_T_MIN, T_start);
            wr->n_th = (size_t)n_th_pts;
            wr->T_th = malloc(wr->n_th * sizeof(double));
            wr->Lnth = malloc(wr->n_th * sizeof(double));
            wr->Lpth = malloc(wr->n_th * sizeof(double));

            double logTlo = log10(CCRTH_T_MIN), logThi = log10(T_start);
            for (size_t i = 0; i < wr->n_th; i++) {
                double frac = (wr->n_th == 1) ? 0.0 : (double)i / (double)(wr->n_th - 1);
                wr->T_th[i] = pow(10.0, logTlo + frac * (logThi - logTlo));
            }
            for (size_t i = 0; i < wr->n_th; i++) {
                wr->Lnth[i] = L_CCRTh_compute(&ctx, wr->T_th[i], +1.0, &tnu_ctx);
                wr->Lpth[i] = L_CCRTh_compute(&ctx, wr->T_th[i], -1.0, &tnu_ctx);
            }

            if (cfg->save_nTOp_thermal) {
                double *th_cols[3] = { wr->T_th, wr->Lnth, wr->Lpth };
                for (char *p = nd + 1; *p; p++) {
                    if (*p == '/') { *p = '\0'; mkdir(nd, 0755); *p = '/'; }
                }
                mkdir(nd, 0755);
                cpr_cache_write(th_path, th_fields, n_th_fp,
                                 "T[K] L_nTOpCCRTh L_pTOnCCRTh", th_cols, 3, wr->n_th);
            }
        }

        /* The cached CCRTh table is in the same raw units as ComputeWeakRates's
         * pre-normalisation sum (see _thermal_correction_interpolants: it
         * divides by Fn before adding to the non-thermal rate). */
        double Fn = cpr_compute_fn(cfg);
        for (size_t i = 0; i < wr->n_th; i++) {
            wr->Lnth[i] /= Fn;
            wr->Lpth[i] /= Fn;
        }
        wr->has_thermal = 1;
    }

    free(Tg_K); free(ratio);
    return 0;
}
