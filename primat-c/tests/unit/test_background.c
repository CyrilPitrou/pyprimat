/* test_background.c -- checks cpr_bg_init_standard/cpr_bg_init_custom
 * (Phase 5: background.c) against reference values from a live
 * primat.background.StandardBackground/CustomBackground run with the
 * default PRIMATConfig() (Omegabh2=0.022425, incomplete_decoupling=
 * spectral_distortions=QED_corrections=True, T_start_cosmo_MeV=40,
 * T_end_MeV=1e-3).
 *
 * Tolerances: a(T)/t(T)/Hubble/rhoB go through this port's own RK45
 * integration of the same two ODEs Python solves with LSODA (see
 * background.c's setup_background_and_cosmo docstring) -- a different
 * accepted-step sequence than LSODA's, so a ~1e-2 relative tolerance
 * (matching the precedent already established for the background ODEs
 * elsewhere in this test suite, e.g. test_neutrino_history.c) is used
 * rather than the ~1e-6 a closed-form/table check would warrant. Neff,
 * Omeganuh2_*, and every CustomBackground check are either closed-form or
 * pure table interpolation (no ODE), so those use a much tighter
 * tolerance.
 */
#include "background.h"
#include "constants.h"
#include "config.h"
#include "plasma.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

static int close_rel(double a, double b, double rtol)
{
    return fabs(a - b) <= rtol * fabs(b) + 1e-300;
}

static void test_standard(void)
{
    cpr_constants_init();
    char *err = NULL;
    CPRConfig cfg;
    CHECK(cpr_config_init_defaults(&cfg, "../primat/data", &err) == 0, "config init succeeds");

    CPRPlasma pl;
    CHECK(cpr_plasma_init(&pl, &cfg, &err) == 0, "plasma init succeeds");

    CPRBackground bg;
    CHECK(cpr_bg_init_standard(&bg, &cfg, &pl, &err) == 0, "cpr_bg_init_standard succeeds");
    CHECK(bg.has_scale_factor, "has_scale_factor is set");

    /* Reference: live StandardBackground(PyPRConfig(), Plasma(cfg)). */
    CHECK(close_rel(cpr_bg_a_of_T(&bg, 1.0), 1.688507212004612e-10, 1e-2),
          "a_of_T(1 MeV) matches Python");
    CHECK(close_rel(cpr_bg_a_of_T(&bg, 0.01), 2.348654180710699e-08, 1e-6),
          "a_of_T(0.01 MeV) matches Python (algebraic a_end boundary, no ODE)");
    CHECK(close_rel(cpr_bg_t_of_T(&bg, 1.0), 0.7450169786456344, 1e-2),
          "t_of_T(1 MeV) matches Python");
    CHECK(close_rel(cpr_bg_t_of_T(&bg, 0.1), 118.87342073822313, 1e-2),
          "t_of_T(0.1 MeV) matches Python");

    /* Self-consistency: a_of_t(t_of_T(T)) == a_of_T(T) (both routes
     * through the same underlying solution). */
    /* 1e-4 (not tighter): a_of_T/t_of_T go through the lnT_sol/lna_sol
     * grid, while a_of_t/t_of_T(Tg_asc) go through the separate t_vec/
     * a_vec grid built from the t(a) ODE solution -- two independent
     * piecewise-linear interpolations of the same underlying continuous
     * solution, each with its own ~1e-5-1e-4 interpolation error, so a
     * small residual mismatch between the two routes is expected. */
    double t01 = cpr_bg_t_of_T(&bg, 0.1);
    CHECK(close_rel(cpr_bg_a_of_t(&bg, t01), cpr_bg_a_of_T(&bg, 0.1), 1e-4),
          "a_of_t(t_of_T(T)) == a_of_T(T) self-consistency");

    double Tg_f, rho_nu_f;
    CHECK(cpr_bg_rho_nu_total_final(&bg, &Tg_f, &rho_nu_f) == 0, "rho_nu_total_final succeeds");
    CHECK(close_rel(rho_nu_f, 4.548634316093674e-13, 1e-6), "rho_nu_total_final matches Python");
    CHECK(close_rel(cpr_bg_N_eff(&bg, Tg_f, rho_nu_f), 3.0439772985579183, 1e-6),
          "Neff matches Python (== CLAUDE.md's documented default Neff)");

    CHECK(close_rel(cpr_bg_weak_nTOp_frwrd(&bg, 1.0e9), 0.0011509718118067201, 1e-2),
          "weak_nTOp_frwrd(1e9 K) matches Python");
    CHECK(close_rel(cpr_bg_weak_nTOp_bkwrd(&bg, 1.0e9), 7.970623324614784e-11, 1e-2),
          "weak_nTOp_bkwrd(1e9 K) matches Python");

    CHECK(close_rel(cpr_bg_rhoB_BBN(&bg, cpr_bg_t_of_T(&bg, 0.1)), 3.9506409068149353e-05, 1e-2),
          "rhoB_BBN(t_of_T(0.1 MeV)) matches Python");

    double relnu, nrnu;
    CHECK(cpr_bg_Omeganuh2_relnu(&bg, &relnu) == 0 && close_rel(relnu, 5.698637318527226e-06, 1e-6),
          "Omeganuh2_relnu matches Python");
    CHECK(cpr_bg_Omeganuh2_nrnu(&bg, &nrnu) == 0 && close_rel(nrnu, 10747.714869154908, 1e-6),
          "Omeganuh2_nrnu matches Python");

    cpr_background_free(&bg);
    cpr_plasma_free(&pl);
    cpr_config_free(&cfg);
}

/* Early Dark Energy: cfg.fEDE=0.3/zcEDE=4e9/wnEDE=3 is an unphysically
 * large/early EDE chosen purely to make the contribution visible at BBN
 * temperatures for this test (a realistic EDE resolving the Hubble
 * tension decays away well before BBN, see test_background.c's earlier
 * fEDE=0.05/zcEDE=3000 smoke check in the session that produced this file
 * -- that combination left every BBN-era quantity unchanged to all
 * printed digits, which is *correct* physics, not a useful regression
 * check). Reference: live StandardBackground with these EDE parameters. */
static void test_ede(void)
{
    char *err = NULL;
    CPRConfig cfg;
    cpr_config_init_defaults(&cfg, "../primat/data", &err);
    cfg.fEDE = 0.3;
    cfg.zcEDE = 4.0e9;
    cfg.wnEDE = 3.0;

    CPRPlasma pl;
    cpr_plasma_init(&pl, &cfg, &err);
    CPRBackground bg;
    CHECK(cpr_bg_init_standard(&bg, &cfg, &pl, &err) == 0, "cpr_bg_init_standard succeeds with EDE active");

    double Tnu = cpr_nu_Tnue_of_Tg(&bg.nh, 1.0);
    CHECK(close_rel(cpr_bg_Hubble(&bg, 1.0, Tnu, Tnu, Tnu), 0.7166027791506387, 1e-2),
          "Hubble(1 MeV) with EDE active matches Python");
    CHECK(close_rel(cpr_bg_t_of_T(&bg, 0.1), 118.818566451818, 1e-2),
          "t_of_T(0.1 MeV) with EDE active matches Python");

    cpr_background_free(&bg);
    cpr_plasma_free(&pl);
    cpr_config_free(&cfg);
}

/* CustomBackground: table written by a live StandardBackground run (300
 * log-spaced (T,t,a) rows), reloaded through both Python's
 * CustomBackground and this port's cpr_bg_init_custom -- pure table
 * interpolation (no ODE), so the tolerance is tight throughout. The
 * fixture file is generated by this test itself (not checked into the
 * repo) so the test has no external data dependency beyond the default
 * StandardBackground run already validated above. */
static void test_custom(const char *path)
{
    char *err = NULL;
    CPRConfig cfg;
    cpr_config_init_defaults(&cfg, "../primat/data", &err);
    cfg.incomplete_decoupling = 0;
    cfg.spectral_distortions = 0;

    CPRPlasma pl;
    cpr_plasma_init(&pl, &cfg, &err);
    CPRBackground bg;
    int rc = cpr_bg_init_custom(&bg, &cfg, &pl, path, &err);
    if (rc) { printf("FAIL cpr_bg_init_custom: %s\n", err); failures++; return; }
    CHECK(rc == 0, "cpr_bg_init_custom succeeds");
    CHECK(bg.has_scale_factor, "has_scale_factor is set (custom)");

    /* Reference: live CustomBackground(cfg, Plasma(cfg), path) loading
     * the same file. */
    CHECK(close_rel(cpr_bg_a_of_T(&bg, 1.0), 1.689075377319047e-10, 1e-6),
          "custom: a_of_T(1 MeV) matches Python");
    CHECK(close_rel(cpr_bg_t_of_T(&bg, 0.1), 118.91600371817064, 1e-6),
          "custom: t_of_T(0.1 MeV) matches Python");
    CHECK(close_rel(cpr_bg_a_of_t(&bg, 100.0), 2.010582569163517e-09, 1e-6),
          "custom: a_of_t(100 s) matches Python");
    CHECK(close_rel(cpr_bg_weak_nTOp_frwrd(&bg, 1.0e9), 0.0011508476841614137, 1e-2),
          "custom: weak_nTOp_frwrd(1e9 K) matches Python");
    CHECK(close_rel(cpr_bg_rhoB_BBN(&bg, 100.0), 5.15134315378115e-05, 1e-6),
          "custom: rhoB_BBN(100 s) matches Python");

    double Tg_f, rho_nu_f;
    CHECK(cpr_bg_rho_nu_total_final(&bg, &Tg_f, &rho_nu_f) == 0,
          "custom: rho_nu_total_final (Friedmann finite-difference estimate) succeeds");
    CHECK(close_rel(rho_nu_f, 4.555412976102112e-13, 1e-4),
          "custom: rho_nu_total_final matches Python");
    CHECK(close_rel(cpr_bg_N_eff(&bg, Tg_f, rho_nu_f), 3.048513624352968, 1e-4),
          "custom: Neff (Friedmann estimate) matches Python");

    /* CustomBackground does not track a separate relic-neutrino
     * calculation (mirrors Python's CustomBackground, which does not
     * override Omeganuh2_relnu/nrnu and so inherits Background's `None`
     * default) -- these must report "not available". */
    double dummy;
    CHECK(cpr_bg_Omeganuh2_relnu(&bg, &dummy) != 0, "custom: Omeganuh2_relnu correctly unavailable");
    CHECK(cpr_bg_Omeganuh2_nrnu(&bg, &dummy) != 0, "custom: Omeganuh2_nrnu correctly unavailable");

    cpr_background_free(&bg);
    cpr_plasma_free(&pl);
    cpr_config_free(&cfg);
}

int main(void)
{
    test_standard();
    test_ede();
    test_custom("tests/fixtures/custom_bg_reference.tsv");

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
