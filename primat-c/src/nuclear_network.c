/* nuclear_network.c -- see cprimat/nuclear_network.h.
 *
 * Reference: Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095),
 * cited below as "Phys. Rep.".
 */
#include "cprimat/nuclear_network.h"
#include "cprimat/constants.h"
#include "cprimat/network_builder.h"
#include "cprimat/ode_rk.h"
#include "cprimat/ode_bdf.h"
#include "cprimat/log.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

/* Riemann zeta(3) (Apery's constant) -- see constants.c's identical
 * literal; duplicated here (not exposed via constants.h) since it is
 * needed only by the Saha (YA) equilibrium formula below. */
#define ZETA3 1.2020569031595942854

static const CPRNuclide *find_nuclide(const CPRConfig *cfg, const char *name)
{
    for (size_t i = 0; i < cfg->nuclides.n; i++)
        if (strcmp(cfg->nuclides.items[i].name, name) == 0) return &cfg->nuclides.items[i];
    return NULL;
}

/* Saha (Nuclear Statistical Equilibrium) mass-fraction abundance of
 * nuclide `name`, in equilibrium with free neutrons/protons at
 * temperature T_K [Kelvin] and baryon-to-photon ratio eta_b (port of
 * solve()'s local YA closure). Phys. Rep. SS V.A:
 *
 *   Y_A = g_A zeta(3)^(A-1) pi^((1-A)/2) 2^((3A-5)/2)
 *         x (M_A / mn^N mp^Z)^(3/2)
 *         x (kB T)^(3(A-1)/2) eta_b^(A-1)
 *         x Yn^N Yp^Z exp(B_A / kB T)
 *
 * where A=N+Z, g_A=2J+1 (spin degeneracy), B_A the binding energy, all
 * masses/energies carried in erg (the "natural units" convention shared
 * with every other module here -- g_const.MeV/keV are the MeV/keV ->
 * erg conversion factors). Used only at T = T_weak (the MT-era seed,
 * cpr_nuclear_network_solve), so eta_b is the single value eta_b_weak
 * evaluated once there -- no eta_b(T) interpolant is needed. */
static double saha_YA(const CPRConfig *cfg, double eta_b, const char *name,
                       double Yn, double Yp, double T_K)
{
    const CPRNuclide *nuc = find_nuclide(cfg, name);
    const CPRNuclide *nuc_n = find_nuclide(cfg, "n");
    const CPRNuclide *nuc_p = find_nuclide(cfg, "p");
    double A = (double)(nuc->N + nuc->Z);
    double Z = (double)nuc->Z;
    double N = A - Z;

    double Mass = A * g_const.ma * g_const.MeV
                  + g_const.keV * nuc->mass_excess_keV
                  - Z * g_const.me * g_const.MeV;
    double BindE = N * nuc_n->mass_excess_keV + Z * nuc_p->mass_excess_keV
                   - nuc->mass_excess_keV;
    /* (M_A / mn^N mp^Z)^(3/2): ratio of nuclear to free-nucleon masses. */
    double NormYA = pow(Mass / (pow(g_const.mn * g_const.MeV, A - Z)
                                  * pow(g_const.mp * g_const.MeV, Z)),
                          1.5);

    return (2.0 * nuc->spin + 1.0)
           * pow(ZETA3, A - 1.0) * pow(M_PI, (1.0 - A) / 2.0)
           * pow(2.0, (3.0 * A - 5.0) / 2.0)
           * NormYA
           * pow(g_const.kB * T_K, 1.5 * (A - 1.0))
           * pow(eta_b, A - 1.0)
           * pow(Yp, Z) * pow(Yn, N)
           * exp(BindE * g_const.keV / (g_const.kB * T_K));
}

/* ---- Growable per-era (t, Y) recorder, fed by the ODE integrators'
 * step_cb hook; seeded with the initial point (the integrators only
 * report *accepted steps after* t0, mirroring solve_ivp.t/.y which
 * include the initial condition as their first row). ---- */
typedef struct {
    double *t;
    double *Y;   /* row-major (cap x n_sp) */
    size_t n_sp, n, cap;
} CPRRecorder;

static void recorder_init(CPRRecorder *r, size_t n_sp)
{
    r->n_sp = n_sp; r->n = 0; r->cap = 64;
    r->t = malloc(r->cap * sizeof(double));
    r->Y = malloc(r->cap * n_sp * sizeof(double));
}

static void recorder_push(CPRRecorder *r, double t, const double *y)
{
    if (r->n == r->cap) {
        r->cap *= 2;
        r->t = realloc(r->t, r->cap * sizeof(double));
        r->Y = realloc(r->Y, r->cap * r->n_sp * sizeof(double));
    }
    r->t[r->n] = t;
    memcpy(&r->Y[r->n * r->n_sp], y, r->n_sp * sizeof(double));
    r->n++;
}

static void recorder_cb(double t, const double *y, size_t n, void *ctx)
{
    (void)n;
    CPRRecorder *r = ctx;
    if (getenv("CPR_NN_DEBUG") && (r->n % 2000 == 0))
        fprintf(stderr, "[nn debug] step=%zu t=%.6e\n", r->n, t);
    recorder_push(r, t, y);
}

static void recorder_free(CPRRecorder *r)
{
    free(r->t); free(r->Y);
}

/* Scatters one era's local abundance row (in_row, named by in_names) into
 * the wider canonical-column row out_row (named by nn->abundance_names);
 * columns absent from in_names are left at out_row's existing value
 * (caller pre-zeros each row, mirroring _embed's np.zeros base). O(n_in *
 * n_out) name matching is fine here: both are at most ~60 (the `large`
 * network). */
static void embed_row(double *out_row, char (*out_names)[16], size_t n_out,
                       const double *in_row, char (*in_names)[16], size_t n_in)
{
    for (size_t j = 0; j < n_in; j++)
        for (size_t k = 0; k < n_out; k++)
            if (strcmp(in_names[j], out_names[k]) == 0) { out_row[k] = in_row[j]; break; }
}

/* ---- ODE right-hand-side / Jacobian glue: each era's CPRODEFunc/
 * CPRODEJacFunc closes over the background + (for MT/LT) compiled rate
 * kernels it needs, matching solve()'s local Y_prime_HT/MT/LT closures. */

typedef struct { CPRBackground *bg; } HTCtx;

static int ht_rhs(double t, const double *Y, double *dY, void *ctx)
{
    HTCtx *c = ctx;
    double T_K = cpr_bg_T_of_t(c->bg, t) * cpr_MeV_to_Kelvin();
    double f = cpr_bg_weak_nTOp_frwrd(c->bg, T_K);
    double b = cpr_bg_weak_nTOp_bkwrd(c->bg, T_K);
    dY[0] = b * Y[1] - f * Y[0];
    dY[1] = f * Y[0] - b * Y[1];
    return 0;
}

typedef struct { CPRBackground *bg; CPRNuclearRates *nucl; } MTLTCtx;

static int mt_rhs(double t, const double *Y, double *dY, void *ctx)
{
    MTLTCtx *c = ctx;
    double rho = cpr_bg_rhoB_BBN(c->bg, t);
    double T_K = cpr_bg_T_of_t(c->bg, t) * cpr_MeV_to_Kelvin();
    double f = cpr_bg_weak_nTOp_frwrd(c->bg, T_K), b = cpr_bg_weak_nTOp_bkwrd(c->bg, T_K);
    cpr_nuclear_rates_rhs_mt(c->nucl, Y, T_K, rho, f, b, dY);
    return 0;
}

static int mt_jac(double t, const double *Y, double *J, void *ctx)
{
    MTLTCtx *c = ctx;
    double rho = cpr_bg_rhoB_BBN(c->bg, t);
    double T_K = cpr_bg_T_of_t(c->bg, t) * cpr_MeV_to_Kelvin();
    double f = cpr_bg_weak_nTOp_frwrd(c->bg, T_K), b = cpr_bg_weak_nTOp_bkwrd(c->bg, T_K);
    cpr_nuclear_rates_jac_mt(c->nucl, Y, T_K, rho, f, b, J);
    return 0;
}

static int lt_rhs(double t, const double *Y, double *dY, void *ctx)
{
    MTLTCtx *c = ctx;
    double rho = cpr_bg_rhoB_BBN(c->bg, t);
    double T_K = cpr_bg_T_of_t(c->bg, t) * cpr_MeV_to_Kelvin();
    double f = cpr_bg_weak_nTOp_frwrd(c->bg, T_K), b = cpr_bg_weak_nTOp_bkwrd(c->bg, T_K);
    cpr_nuclear_rates_rhs_lt(c->nucl, Y, T_K, rho, f, b, dY);
    return 0;
}

static int lt_jac(double t, const double *Y, double *J, void *ctx)
{
    MTLTCtx *c = ctx;
    double rho = cpr_bg_rhoB_BBN(c->bg, t);
    double T_K = cpr_bg_T_of_t(c->bg, t) * cpr_MeV_to_Kelvin();
    double f = cpr_bg_weak_nTOp_frwrd(c->bg, T_K), b = cpr_bg_weak_nTOp_bkwrd(c->bg, T_K);
    cpr_nuclear_rates_jac_lt(c->nucl, Y, T_K, rho, f, b, J);
    return 0;
}

static double *find_in(double *raw_vals, char (*raw_names)[16], size_t n_raw, const char *name)
{
    for (size_t i = 0; i < n_raw; i++)
        if (strcmp(raw_names[i], name) == 0) return &raw_vals[i];
    return NULL;
}

int cpr_nuclear_network_solve(CPRNuclearNetwork *nn, const CPRConfig *cfg,
                                CPRNuclearRates *nucl, CPRBackground *background,
                                char **errmsg)
{
    memset(nn, 0, sizeof(*nn));
    nn->cfg = cfg; nn->background = background; nn->nucl = nucl;

    /* Refresh nuclear rates with the current rate-variation parameters
     * (mirrors solve()'s nucl.apply_variations(cfg) call at the top). */
    cpr_nuclear_rates_apply_variations(nucl, cfg);

    /* ---- Temperature era boundaries [s]. cpr_T_start/T_weak/T_nucl are
     * *fixed* era boundaries in Kelvin (10/1/0.11 MeV respectively,
     * independent of cfg -- see constants.h), unlike T_end which is the
     * user-configurable cfg->T_end_MeV. ---- */
    double T_start_K = cpr_T_start(), T_weak_K = cpr_T_weak(), T_nucl_K = cpr_T_nucl();
    double T_end_K = cpr_config_T_end(cfg);
    double t_start = cpr_bg_t_of_T(background, T_start_K / cpr_MeV_to_Kelvin());
    double t_weak  = cpr_bg_t_of_T(background, T_weak_K  / cpr_MeV_to_Kelvin());
    double t_nucl  = cpr_bg_t_of_T(background, T_nucl_K  / cpr_MeV_to_Kelvin());
    double t_end   = cpr_bg_t_of_T(background, T_end_K   / cpr_MeV_to_Kelvin());
    nn->t_end = t_end;

    /* ---- Baryon-to-photon ratio at T_weak, for the MT-era Saha seed. ---- */
    double nB_weak = cpr_bg_rhoB_BBN(background, t_weak) / (g_const.ma * cpr_MeV4_to_gcmm3());
    double ngamma_weak = (2.0 * ZETA3 / (M_PI * M_PI)) * pow(T_weak_K / cpr_MeV_to_Kelvin(), 3.0);
    double eta_b_weak = nB_weak / ngamma_weak;

    /* ------------------------------------------------------------------
     * HT era: n <-> p only, non-stiff RK45.
     * ------------------------------------------------------------------ */
    double f0 = cpr_bg_weak_nTOp_frwrd(background, T_start_K);
    double b0 = cpr_bg_weak_nTOp_bkwrd(background, T_start_K);
    double Y_ht[2] = { b0 / (b0 + f0), 0.0 };
    Y_ht[1] = 1.0 - Y_ht[0];

    CPRRecorder rec_ht; recorder_init(&rec_ht, 2);
    recorder_push(&rec_ht, t_start, Y_ht);
    HTCtx ht_ctx = { background };
    CPRRKOpts rk_opts = cpr_ode_rk_default_opts();
    rk_opts.rtol = cfg->numerical_precision; rk_opts.atol = 1.0e-10;
    cpr_log(cfg, "nucl", "Solving neutron decoupling at high temperature era");
    clock_t _t_ht0 = clock();
    if (cpr_ode_rk45(ht_rhs, &ht_ctx, t_start, t_weak, Y_ht, 2, rk_opts,
                      recorder_cb, &rec_ht, errmsg)) {
        recorder_free(&rec_ht);
        return 1;
    }
    cpr_log(cfg, "nucl", "[HT] Finished in %.2f s",
             (double)(clock() - _t_ht0) / CLOCKS_PER_SEC);
    double Yn_HT_f = Y_ht[0], Yp_HT_f = Y_ht[1];

    /* ------------------------------------------------------------------
     * MT era: fixed 18-reaction subset, stiff BDF with analytic Jacobian.
     * ------------------------------------------------------------------ */
    char (*mt_names)[16] = nucl->mt_net.species;
    size_t n_mt = nucl->mt_net.n_species;
    double *Yi_MT = malloc(n_mt * sizeof(double));
    for (size_t i = 0; i < n_mt; i++) {
        if (strcmp(mt_names[i], "n") == 0) Yi_MT[i] = Yn_HT_f;
        else if (strcmp(mt_names[i], "p") == 0) Yi_MT[i] = Yp_HT_f;
        else Yi_MT[i] = saha_YA(cfg, eta_b_weak, mt_names[i], Yn_HT_f, Yp_HT_f, T_weak_K);
    }

    CPRRecorder rec_mt; recorder_init(&rec_mt, n_mt);
    recorder_push(&rec_mt, t_weak, Yi_MT);
    MTLTCtx mt_ctx = { background, nucl };
    CPRBDFOpts bdf_opts = cpr_ode_bdf_default_opts();
    bdf_opts.rtol = cfg->numerical_precision; bdf_opts.atol = 1.0e-15;
    cpr_log(cfg, "nucl", "Solving nuclear network at mid temperature era");
    clock_t _t_mt0 = clock();
    if (cpr_ode_bdf(mt_rhs, mt_jac, &mt_ctx, t_weak, t_nucl, Yi_MT, n_mt, bdf_opts,
                     recorder_cb, &rec_mt, errmsg)) {
        free(Yi_MT); recorder_free(&rec_ht); recorder_free(&rec_mt);
        return 1;
    }
    cpr_log(cfg, "nucl", "[MT] Finished (%s network, %zu nuclides) in %.2f s",
             cfg->network, n_mt, (double)(clock() - _t_mt0) / CLOCKS_PER_SEC);

    /* ------------------------------------------------------------------
     * LT era: the chosen network (small/large, optionally amax-restricted),
     * stiff BDF with analytic Jacobian.
     * ------------------------------------------------------------------ */
    char (*lt_names)[16] = nucl->lt_net.species;
    size_t n_lt = nucl->lt_net.n_species;
    double *Yi_LT = malloc(n_lt * sizeof(double));
    for (size_t i = 0; i < n_lt; i++) {
        double *v = find_in(Yi_MT, mt_names, n_mt, lt_names[i]);
        Yi_LT[i] = v ? *v : 0.0;
    }

    CPRRecorder rec_lt; recorder_init(&rec_lt, n_lt);
    recorder_push(&rec_lt, t_nucl, Yi_LT);
    MTLTCtx lt_ctx = { background, nucl };
    CPRBDFOpts bdf_opts_lt = cpr_ode_bdf_default_opts();
    bdf_opts_lt.rtol = 10.0 * cfg->numerical_precision;
    bdf_opts_lt.atol = cpr_config_is_large(cfg) ? cfg->atol_large_LT : 1.0e-20;
    cpr_log(cfg, "nucl", "Solving nuclear network at low temperature era");
    clock_t _t_lt0 = clock();
    if (cpr_ode_bdf(lt_rhs, lt_jac, &lt_ctx, t_nucl, t_end, Yi_LT, n_lt, bdf_opts_lt,
                     recorder_cb, &rec_lt, errmsg)) {
        free(Yi_MT); free(Yi_LT);
        recorder_free(&rec_ht); recorder_free(&rec_mt); recorder_free(&rec_lt);
        return 1;
    }
    cpr_log(cfg, "nucl", "[LT] Finished (%s network, %zu nuclides) in %.2f s",
             cfg->network, n_lt, (double)(clock() - _t_lt0) / CLOCKS_PER_SEC);

    /* ---- Final abundances: the LT species list is the canonical name
     * list for any network (mirrors solve()'s self.abundance_names = species_L). ---- */
    nn->n_species = n_lt;
    nn->abundance_names = malloc(n_lt * sizeof(*nn->abundance_names));
    memcpy(nn->abundance_names, lt_names, n_lt * sizeof(*nn->abundance_names));
    nn->Y_final = malloc(n_lt * sizeof(double));
    memcpy(nn->Y_final, Yi_LT, n_lt * sizeof(double));

    /* ---- Concatenated HT+MT+LT history, embedding each era's narrower
     * vector into the common n_lt columns by species name (mirrors
     * solve()'s _embed/Y_of_t construction; DT-era extension not ported,
     * see this module's header comment). MT/LT recorders' first row
     * duplicates the previous era's last time point exactly (both eras
     * are seeded at the boundary time), so it is dropped here exactly as
     * Python's sol_MT.t[1:]/sol_LT.t[1:] does. ---- */
    char ht_names[2][16] = { "n", "p" };
    nn->n_t = rec_ht.n + (rec_mt.n - 1) + (rec_lt.n - 1);
    nn->t_hist = malloc(nn->n_t * sizeof(double));
    nn->Y_hist = calloc(nn->n_t * n_lt, sizeof(double));
    size_t row = 0;
    for (size_t i = 0; i < rec_ht.n; i++, row++) {
        nn->t_hist[row] = rec_ht.t[i];
        embed_row(&nn->Y_hist[row * n_lt], nn->abundance_names, n_lt,
                  &rec_ht.Y[i * 2], ht_names, 2);
    }
    for (size_t i = 1; i < rec_mt.n; i++, row++) {
        nn->t_hist[row] = rec_mt.t[i];
        embed_row(&nn->Y_hist[row * n_lt], nn->abundance_names, n_lt,
                  &rec_mt.Y[i * n_mt], mt_names, n_mt);
    }
    for (size_t i = 1; i < rec_lt.n; i++, row++) {
        nn->t_hist[row] = rec_lt.t[i];
        embed_row(&nn->Y_hist[row * n_lt], nn->abundance_names, n_lt,
                  &rec_lt.Y[i * n_lt], lt_names, n_lt);
    }
    nn->t_start = nn->t_hist[0];

    free(Yi_MT); free(Yi_LT);
    recorder_free(&rec_ht); recorder_free(&rec_mt); recorder_free(&rec_lt);

    if (cfg->output_time_evolution)
        cpr_nuclear_network_write_time_evolution(nn, cfg->output_n_points, errmsg);
    if (cfg->output_final_result)
        cpr_nuclear_network_write_final_result(nn, errmsg);

    return 0;
}

void cpr_nuclear_network_free(CPRNuclearNetwork *nn)
{
    free(nn->abundance_names); free(nn->Y_final);
    free(nn->t_hist); free(nn->Y_hist);
    memset(nn, 0, sizeof(*nn));
}

double cpr_nuclear_network_get(const CPRNuclearNetwork *nn, const char *name)
{
    for (size_t i = 0; i < nn->n_species; i++)
        if (strcmp(nn->abundance_names[i], name) == 0) return nn->Y_final[i];
    return 0.0;
}

double cpr_nuclear_network_Y_of_t(const CPRNuclearNetwork *nn, const char *name, double t)
{
    size_t col = nn->n_species;
    for (size_t i = 0; i < nn->n_species; i++)
        if (strcmp(nn->abundance_names[i], name) == 0) { col = i; break; }
    if (col == nn->n_species) return 0.0; /* not tracked */

    if (t <= nn->t_start) return 0.0;                       /* before HT start */
    if (t >= nn->t_hist[nn->n_t - 1]) return nn->Y_final[col]; /* past LT end */

    /* Binary search the bracketing segment, then linear-interpolate --
     * mirrors interp1d's default (linear, here with the explicit
     * fill_value=(0, Y[-1]) clamps applied above instead of bounds_error). */
    size_t lo = 0, hi = nn->n_t - 1;
    while (hi - lo > 1) {
        size_t mid = (lo + hi) / 2;
        if (nn->t_hist[mid] <= t) lo = mid; else hi = mid;
    }
    double t0 = nn->t_hist[lo], t1 = nn->t_hist[hi];
    double y0 = nn->Y_hist[lo * nn->n_species + col], y1 = nn->Y_hist[hi * nn->n_species + col];
    double frac = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0;
    return y0 + frac * (y1 - y0);
}

/* Creates every directory component of `path` in turn, including `path`
 * itself (mkdir -p equivalent without a shell call) -- mirrors
 * os.makedirs(exist_ok=True). The intermediate-component loop alone
 * (walking only embedded '/' characters) never creates the final,
 * slash-free component -- e.g. mkdir_p("results") used to silently do
 * nothing, so a fresh checkout's first write into the (not-yet-existing)
 * "results/" directory failed with ENOENT; the explicit mkdir() below the
 * loop covers that last, most common case. */
static void mkdir_p(const char *path)
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

int cpr_nuclear_network_write_final_result(const CPRNuclearNetwork *nn, char **errmsg)
{
    /* Resolve relative to the current working directory (matching
     * os.path.abspath's behaviour -- Python resolves against cwd, not
     * data_dir). */
    char path[4300];
    snprintf(path, sizeof(path), "%s", nn->cfg->output_final_file);

    char dir[4300];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) { *slash = '\0'; mkdir_p(dir); }

    FILE *f = fopen(path, "w");
    if (!f) {
        char buf[4400];
        snprintf(buf, sizeof(buf), "cpr_nuclear_network_write_final_result: cannot open %s", path);
        *errmsg = strdup(buf);
        return 1;
    }
    fprintf(f, "# %-12sY\n", "nuclide");
    for (size_t i = 0; i < nn->n_species; i++)
        fprintf(f, "%-14s%.6e\n", nn->abundance_names[i], nn->Y_final[i]);
    fclose(f);
    printf("[output] Final abundances (%zu nuclides) written to %s\n", nn->n_species, path);
    return 0;
}

void cpr_nuclear_network_sample_time_evolution(const CPRNuclearNetwork *nn, int n_points,
                                                  double *t_out, double *T_out, double *a_out,
                                                  double *Tnue_out, double *Tnumu_out,
                                                  double *Tnutau_out, double *Y_out)
{
    const CPRConfig *cfg = nn->cfg;
    CPRBackground *bg = nn->background;

    double t_cosmo = cpr_bg_t_of_T(bg, cfg->T_start_cosmo_MeV);
    double t_end = nn->t_end;
    size_t n = (size_t)n_points;
    double logTlo = log10(t_cosmo), logThi = log10(t_end);

    for (size_t i = 0; i < n; i++) {
        double frac = (n == 1) ? 0.0 : (double)i / (double)(n - 1);
        double t = pow(10.0, logTlo + frac * (logThi - logTlo));
        t_out[i] = t;
        T_out[i] = cpr_bg_T_of_t(bg, t);
        a_out[i] = bg->has_scale_factor ? cpr_bg_a_of_t(bg, t) : NAN;

        double Tnue, Tnumu, Tnutau;
        if (cpr_bg_Tnu_of_t(bg, t, &Tnue, &Tnumu, &Tnutau)) {
            Tnue_out[i] = Tnue; Tnumu_out[i] = Tnumu; Tnutau_out[i] = Tnutau;
        } else {
            Tnue_out[i] = Tnumu_out[i] = Tnutau_out[i] = NAN;
        }

        for (size_t s = 0; s < nn->n_species; s++)
            Y_out[i * nn->n_species + s] = cpr_nuclear_network_Y_of_t(nn, nn->abundance_names[s], t);
    }
}

int cpr_nuclear_network_write_time_evolution(const CPRNuclearNetwork *nn, int n_points,
                                                char **errmsg)
{
    /* cfg->output_file == NULL/"" is the in-memory-only escape hatch
     * (mirrors Python's NuclearNetwork._write_time_evolution skipping disk
     * I/O when cfg.output_file is None, e.g. primat-gui/run_bbn's
     * in-memory-only callers via CPRResults's evol_* arrays, populated by
     * cpr_assemble_results regardless of this flag). */
    if (!nn->cfg->output_file || !nn->cfg->output_file[0])
        return 0;

    size_t n = (size_t)n_points;
    double *t_out = malloc(n * sizeof(double));
    double *T_out = malloc(n * sizeof(double));
    double *a_out = malloc(n * sizeof(double));
    double *Tnue_out = malloc(n * sizeof(double));
    double *Tnumu_out = malloc(n * sizeof(double));
    double *Tnutau_out = malloc(n * sizeof(double));
    double *Y_out = malloc(n * nn->n_species * sizeof(double));
    cpr_nuclear_network_sample_time_evolution(nn, n_points, t_out, T_out, a_out,
                                                Tnue_out, Tnumu_out, Tnutau_out, Y_out);

    const char *rel = nn->cfg->output_file;
    char path[4300];
    snprintf(path, sizeof(path), "%s", rel);
    char dir[4300];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) { *slash = '\0'; mkdir_p(dir); }

    FILE *f = fopen(path, "w");
    if (!f) {
        free(t_out); free(T_out); free(a_out);
        free(Tnue_out); free(Tnumu_out); free(Tnutau_out); free(Y_out);
        char buf[4400];
        snprintf(buf, sizeof(buf), "cpr_nuclear_network_write_time_evolution: cannot open %s", path);
        *errmsg = strdup(buf);
        return 1;
    }

    /* Unified schema (PRIMAT.md S7.2), header-compatible with
     * primat.evolution.dump_evolution/load_evolution: no leading "#",
     * tab-separated, t_s/a/T_*_MeV core block then one Y_<nuclide> column
     * per tracked species. Per-reaction flux columns (cfg->output_rates_time_evolution,
     * network="small" only) are not ported -- see this module's header
     * top comment. */
    fprintf(f, "t_s\ta\tT_gamma_MeV\tT_nue_MeV\tT_numu_MeV\tT_nutau_MeV");
    for (size_t s = 0; s < nn->n_species; s++) fprintf(f, "\tY_%s", nn->abundance_names[s]);
    fprintf(f, "\n");

    for (size_t i = 0; i < n; i++) {
        fprintf(f, "%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e",
                t_out[i], a_out[i], T_out[i], Tnue_out[i], Tnumu_out[i], Tnutau_out[i]);
        for (size_t s = 0; s < nn->n_species; s++)
            fprintf(f, "\t%.8e", Y_out[i * nn->n_species + s]);
        fprintf(f, "\n");
    }
    fclose(f);
    free(t_out); free(T_out); free(a_out);
    free(Tnue_out); free(Tnumu_out); free(Tnutau_out); free(Y_out);
    printf("[output] Time-evolution data (%zu rows) written to %s\n", n, path);
    return 0;
}
