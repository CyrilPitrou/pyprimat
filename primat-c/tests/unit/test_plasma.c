/* test_plasma.c -- checks the SM plasma thermodynamics port against
 * reference values from a live pyprimat.plasma.Plasma instance (default
 * config: QED_corrections=True, n_electron_table=2000,
 * T_start_cosmo_MeV=40.0, DeltaNeff=0), run with `data_dir="../primat/data"`
 * so this test also exercises the real on-disk QED_*.txt and
 * electron_thermo_cache.txt files -- in particular, the electron-thermo
 * cache file shipped in the repo was produced by Python with exactly this
 * fingerprint (format_version=1, n_electron_table=2000,
 * T_start_cosmo_MeV=40.0), so a correct fingerprint hash port
 * (cache.c, already validated in test_cache.c) plus a correct
 * cpr_table_read of its 5 columns should make this a cache *hit*,
 * skipping computation entirely and loading byte-identical numbers --
 * the strongest possible cross-language compatibility check available
 * for this module.
 */
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
    strncpy(cfg.data_dir, "../primat/data", sizeof(cfg.data_dir) - 1);
    cfg.QED_corrections = 1;
    cfg.recompute_qed_corrections = 0;
    cfg.recompute_electron_thermo = 0;
    cfg.n_electron_table = 2000;
    cfg.T_start_cosmo_MeV = 40.0;
    cfg.DeltaNeff = 0.0;

    char *err = NULL;
    CPRPlasma pl;
    int rc = cpr_plasma_init(&pl, &cfg, &err);
    if (rc) { printf("FAIL cpr_plasma_init: %s\n", err); return 1; }
    CHECK(rc == 0, "cpr_plasma_init succeeds (cache hit expected for electron thermo)");

    /* Reference values from a live Python run (pyprimat.plasma.Plasma,
     * cfg = PyPRConfig() defaults with data_dir pointed at pyprimat/). */
    CHECK(close_rel(cpr_plasma_rho_e(&pl, 1.0), 1.1288162231372227, 1e-5), "rho_e(1.0) matches Python");
    CHECK(close_rel(cpr_plasma_p_e(&pl, 1.0), 0.3637834176879932, 1e-5), "p_e(1.0) matches Python");
    CHECK(close_rel(cpr_plasma_drho_e_dT(&pl, 1.0), 4.561478427783182, 1e-5), "drho_e_dT(1.0) matches Python");
    CHECK(close_rel(cpr_plasma_dp_e_dT(&pl, 1.0), 1.4925996408518007, 1e-5), "dp_e_dT(1.0) matches Python");

    CHECK(close_rel(cpr_plasma_spl(&pl, 1.0), 2.3647498262552173, 1e-6), "spl(1.0) matches Python");
    CHECK(close_rel(cpr_plasma_rho_SM(&pl, 1.0, 1.0, 1.0), 3.5100225014336224, 1e-6), "rho_SM(1,1,1) matches Python");
    CHECK(close_rel(cpr_plasma_p_SM(&pl, 1.0, 1.0, 1.0), 1.1576350184091115, 1e-6), "p_SM(1,1,1) matches Python");
    CHECK(close_rel(cpr_plasma_T_nu_decoupling(&pl, 0.01), 0.007137658555036082, 1e-5),
          "T_nu_decoupling(0.01) matches Python");

    double s, ds_dT;
    cpr_plasma_spl_and_dspl_dT(&pl, 1.0, &s, &ds_dT);
    CHECK(close_rel(s, 2.3647498262552173, 1e-6), "spl_and_dspl_dT s-component matches Python");
    CHECK(close_rel(ds_dT, 7.176936759574728, 1e-4), "spl_and_dspl_dT ds_dT-component matches Python");
    CHECK(fabs(cpr_plasma_dspl_dT(&pl, 1.0) - ds_dT) < 1e-12, "dspl_dT matches spl_and_dspl_dT's ds_dT");

    /* Boltzmann-suppressed low-T limit: exactly 0 below me/30. */
    CHECK(cpr_plasma_rho_e(&pl, g_const.me / 100.0) == 0.0, "rho_e below me/30 is exactly 0");
    CHECK(cpr_plasma_p_e(&pl, g_const.me / 100.0) == 0.0, "p_e below me/30 is exactly 0");

    /* DeltaNeff=0 means rho_nu_extra contributes nothing to rho_SM. */
    CHECK(cpr_plasma_rho_nu_extra(&pl, 1.0) == 0.0, "rho_nu_extra is 0 when DeltaNeff=0");

    cpr_plasma_free(&pl);

    /* Pure cfg-independent functions, checked against their closed forms
     * directly (Phys. Rep. Eq. 25b/26b). */
    CHECK(close_rel(cpr_rho_g(1.0), 2.0 * M_PI * M_PI / 30.0, 1e-12), "rho_g(1.0) matches pi^2/15");
    CHECK(close_rel(cpr_drho_g_dT(2.0), 4.0 * cpr_rho_g(2.0) / 2.0, 1e-12), "drho_g_dT(2.0) matches 4 rho_g/T");
    CHECK(close_rel(cpr_rho_nu(1.0), 7.0 / 4.0 * M_PI * M_PI / 30.0, 1e-12), "rho_nu(1.0) matches 7/8 x 2 x pi^2/30");

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
