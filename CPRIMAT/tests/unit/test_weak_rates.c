/* test_weak_rates.c -- checks the n<->p weak-rate cache-hit path
 * (cpr_weak_rates_init/cpr_weak_rate_nTOp/cpr_weak_rate_pTOn) against
 * reference values from a live pyprimat.weak_rates.api.RecomputeWeakRates
 * run with the default PyPRConfig() (radiative_corrections=
 * finite_mass_corrections=thermal_corrections=spectral_distortions=
 * weak_rate_cache=save_nTOp=tau_n_normalization=True, T_end_MeV=1e-3,
 * T_start_cosmo_MeV=40.0). The default config's weak-rate fingerprint hash
 * is "2218248995f018af" (rates/weak/nTOp_2218248995f018af.txt, confirmed
 * present) and its thermal fingerprint hash is "f2f067e2842add12"
 * (rates/weak/nTOp_thermal_f2f067e2842add12.txt, confirmed present), so
 * this run is a cache hit on both tables -- the from-scratch Gauss-Legendre
 * integration path (cache miss) is exercised by neither this test nor the
 * current C port.
 *
 * The from-scratch thermal computation (Phase 3b, L_CCRTh_compute in
 * weak_rates.c) IS ported, but is intentionally not exercised here via
 * cpr_weak_rates_init: that path builds a full T-grid up to a *fixed*
 * 10 MeV boundary (cpr_T_start(), independent of any PyPRConfig field),
 * where the (E,k) integration domain genuinely widens to E_max~390 (see
 * L_thermal_2d's docstring) -- a real physics-driven cost that Python's
 * own dblquad-based implementation pays too (multi-minute per from-scratch
 * table, hence this file's "may take a while" stderr notice), not a defect
 * in this port. Phase 3b's correctness is instead checked directly against
 * a faithful Python re-implementation of corrections.py's dblquad/quad
 * formulas (bypassing cpr_weak_rates_init's full-table cost) by
 * test_weak_rates_thermal.c, which calls L_CCRTh_compute itself at a few
 * individual T points cheap enough for routine `make test`. */
#include "cprimat/weak_rates.h"
#include "cprimat/neutrino_history.h"
#include "cprimat/plasma.h"
#include "cprimat/constants.h"

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

int main(void)
{
    cpr_constants_init();

    CPRConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.data_dir, "../pyprimat", sizeof(cfg.data_dir) - 1);
    cfg.QED_corrections = 1;
    cfg.n_electron_table = 2000;
    cfg.T_start_cosmo_MeV = 40.0;
    cfg.T_end_MeV = 1.0e-3;
    cfg.incomplete_decoupling = 1;
    cfg.spectral_distortions = 1;
    cfg.analytic_distortions = 0;
    cfg.nevo_file_prefix = "NEVOPRIMAT";
    cfg.radiative_corrections = 1;
    cfg.finite_mass_corrections = 1;
    cfg.thermal_corrections = 1;
    cfg.weak_rate_cache = 1;
    cfg.save_nTOp = 1;
    cfg.sampling_nTOp_per_decade = 80;
    cfg.sampling_nTOp_thermal_per_decade = 20;
    cfg.tau_n_normalization = 1;
    cfg.munuOverTnu = 0.0;

    char *err = NULL;
    CPRPlasma pl;
    if (cpr_plasma_init(&pl, &cfg, &err)) { printf("FAIL plasma init: %s\n", err); return 1; }

    CPRNeutrinoHistory nh;
    if (cpr_neutrino_history_init(&nh, &cfg, &pl, &err)) {
        printf("FAIL cpr_neutrino_history_init: %s\n", err);
        return 1;
    }

    /* Dummy Tg/Tnu background arrays: only feed the T_nu(T_gamma)/T_gamma
     * interpolant used on a cache *miss* (not exercised here, see header
     * comment), so their exact values are immaterial -- two points are
     * enough to satisfy the n_bg>=2 assumption used when sorting them. */
    double Tg_MeV[2] = {1.0e-3, 40.0};
    double Tnu_MeV[2] = {1.0e-3 * 0.7138, 40.0 * 0.7138};

    CPRWeakRates wr;
    int rc = cpr_weak_rates_init(&wr, Tg_MeV, Tnu_MeV, 2, &cfg, &nh, &err);
    if (rc) { printf("FAIL cpr_weak_rates_init: %s\n", err); return 1; }
    CHECK(rc == 0, "cpr_weak_rates_init succeeds (cache hit expected on both tables)");
    CHECK(wr.has_thermal, "thermal_corrections=True and a matching cache file loads has_thermal=1");

    /* Reference values: live RecomputeWeakRates([Tg_vec,Tnu_vec], cfg) at
     * default config, sgnq=+1 (n->p, "frwrd") / -1 (p->n, "bkwrd"); both
     * already in units of 1/tau_n, matching cpr_weak_rate_nTOp/pTOn.
     *
     * Tolerances: at low T the rate is dominated by the cache-loaded
     * nonthermal table (no thermal contribution there), interpolated by
     * cpr_interp_quadratic_local -- a local 3-point stand-in for scipy's
     * global quadratic B-spline (interp1d(kind='quadratic'); see the
     * trade-off documented in spline.h). At higher T the rate spans many
     * orders of magnitude across the table and that interpolation
     * difference, while still small, is no longer negligible at the 1e-6
     * level claimed for smoother tables -- empirically ~1e-4 relative
     * across this table's full T range (verified against the same Python
     * reference at the 11 T-points spanning the whole table, not just
     * the ones checked below); 3e-3 is a comfortable margin above that
     * while still catching any real (orders-of-magnitude, sign, or
     * missing-term) regression. */
    CHECK(close_rel(cpr_weak_rate_nTOp(&wr, 1.16e7), 0.9999983396745395, 1e-6),
          "Gamma_nTOp(1.16e7 K) matches Python");
    CHECK(fabs(cpr_weak_rate_pTOn(&wr, 1.16e7)) < 1e-40,
          "Gamma_pTOn(1.16e7 K) matches Python (tiny, below the table's resolution)");

    CHECK(close_rel(cpr_weak_rate_nTOp(&wr, 5.0e7), 0.9999928879193875, 1e-6),
          "Gamma_nTOp(5e7 K) matches Python");

    CHECK(close_rel(cpr_weak_rate_nTOp(&wr, 1.0e8), 0.9999864100639516, 1e-6),
          "Gamma_nTOp(1e8 K) matches Python");

    CHECK(close_rel(cpr_weak_rate_nTOp(&wr, 1.0e9), 1.011013639491023, 3e-3),
          "Gamma_nTOp(1e9 K) matches Python");
    CHECK(close_rel(cpr_weak_rate_pTOn(&wr, 1.0e9), 7.001395528341625e-08, 1e-2),
          "Gamma_pTOn(1e9 K) matches Python");

    CHECK(close_rel(cpr_weak_rate_nTOp(&wr, 1.0e10), 687.8999904548307, 3e-3),
          "Gamma_nTOp(1e10 K) matches Python");
    CHECK(close_rel(cpr_weak_rate_pTOn(&wr, 1.0e10), 153.2892887765402, 3e-3),
          "Gamma_pTOn(1e10 K) matches Python");

    CHECK(close_rel(cpr_weak_rate_nTOp(&wr, 1.1e11), 56761817.45307914, 3e-3),
          "Gamma_nTOp(1.1e11 K) matches Python");
    CHECK(close_rel(cpr_weak_rate_pTOn(&wr, 1.1e11), 49642897.75408083, 3e-3),
          "Gamma_pTOn(1.1e11 K) matches Python");

    cpr_weak_rates_free(&wr);
    cpr_neutrino_history_free(&nh);
    cpr_plasma_free(&pl);

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
