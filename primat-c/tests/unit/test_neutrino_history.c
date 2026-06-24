/* test_neutrino_history.c -- checks the neutrino-sector background port
 * against reference values from a live pyprimat.neutrino_history run
 * (make_neutrino_history with default cfg = NEVOTable +
 * spectral_distortions=True, and incomplete_decoupling=False +
 * spectral_distortions=False = InstantaneousDecoupling -- the
 * AnalyticDistortion decorator is out of scope, see neutrino_history.h).
 */
#include "cprimat/neutrino_history.h"
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
    strncpy(cfg.data_dir, "../primat", sizeof(cfg.data_dir) - 1);
    cfg.QED_corrections = 1;
    cfg.n_electron_table = 2000;
    cfg.T_start_cosmo_MeV = 40.0;
    cfg.incomplete_decoupling = 1;
    cfg.spectral_distortions = 1;
    cfg.analytic_distortions = 0;
    cfg.nevo_file_prefix = "NEVOPRIMAT";

    char *err = NULL;
    CPRPlasma pl;
    if (cpr_plasma_init(&pl, &cfg, &err)) { printf("FAIL plasma init: %s\n", err); return 1; }

    CPRNeutrinoHistory nh;
    int rc = cpr_neutrino_history_init(&nh, &cfg, &pl, &err);
    if (rc) { printf("FAIL cpr_neutrino_history_init: %s\n", err); return 1; }
    CHECK(nh.kind == CPR_NU_NEVO_TABLE, "incomplete_decoupling selects CPR_NU_NEVO_TABLE");
    CHECK(nh.has_distortion, "spectral_distortions=True, analytic=False builds the NEVO distortion table");

    CHECK(close_rel(cpr_nu_Tnue_of_Tg(&nh, 1.0), 0.9959782963400438, 1e-7), "Tnue(1.0) matches Python");
    CHECK(close_rel(cpr_nu_Tnumu_of_Tg(&nh, 1.0), 0.9958183497351032, 1e-7), "Tnumu(1.0) matches Python");
    CHECK(close_rel(cpr_nu_Tnutau_of_Tg(&nh, 1.0), 0.9958128729691089, 1e-7), "Tnutau(1.0) matches Python");
    CHECK(close_rel(cpr_nu_N_NEVO_of_Tg(&nh, 1.0), 0.0038652664267106005, 1e-6), "N(1.0) matches Python");
    CHECK(close_rel(cpr_nu_x_of_Tg(&nh, 1.0), 0.513572224877158, 1e-7), "x_of_Tg(1.0) matches Python");

    CHECK(close_rel(cpr_nu_Tnue_of_Tg(&nh, 0.05), 0.03589191744637729, 1e-6), "Tnue(0.05) matches Python");
    CHECK(close_rel(cpr_nu_N_NEVO_of_Tg(&nh, 0.05), -2.9820043203872707e-06, 1e-5), "N(0.05) matches Python");
    CHECK(close_rel(cpr_nu_x_of_Tg(&nh, 0.05), 14.262052006482783, 1e-6), "x_of_Tg(0.05) matches Python");

    /* Tg=50 is above the table's range (radiation-domination extrapolation
     * for x_of_Tg/Tnue, exact 0 -- not edge-clamped -- for N). */
    CHECK(close_rel(cpr_nu_Tnue_of_Tg(&nh, 50.0), 49.99999999997857, 1e-9), "Tnue(50) matches Python (above table)");
    CHECK(cpr_nu_N_NEVO_of_Tg(&nh, 50.0) == 0.0, "N(50) is exactly 0 above the table");
    CHECK(close_rel(cpr_nu_x_of_Tg(&nh, 50.0), 0.01021999786938253, 1e-6), "x_of_Tg(50) matches Python (above table)");

    double Tnue1 = cpr_nu_Tnue_of_Tg(&nh, 1.0);
    double x_arg = g_const.me / 1.0, znu_arg = g_const.me / Tnue1;
    CHECK(close_rel(cpr_nu_dFDneu(&nh, 2.0, x_arg, znu_arg, 1.0), -0.00019089897949153833, 1e-6),
          "dFDneu(en=2,sgnq=+1) matches Python");
    CHECK(close_rel(cpr_nu_dFDneu(&nh, -2.0, x_arg, znu_arg, 1.0), 0.00019089897949153833, 1e-6),
          "dFDneu(en=-2,sgnq=+1) is the negated value (Pauli-blocking branch)");
    CHECK(close_rel(cpr_nu_dFDneu(&nh, 2.0, x_arg, znu_arg, -1.0), -0.00019089897949153833, 1e-6),
          "dFDneu(en=2,sgnq=-1) matches Python (NEVO distortion is sgnq-independent)");

    cpr_neutrino_history_free(&nh);
    cpr_plasma_free(&pl);

    /* InstantaneousDecoupling: incomplete_decoupling=False, spectral_distortions
     * must be False too (PyPRConfig forbids the full-NEVO distortion outside
     * incomplete_decoupling mode -- the analytic decorator that would be legal
     * here is out of scope). */
    CPRConfig cfg2;
    memset(&cfg2, 0, sizeof(cfg2));
    strncpy(cfg2.data_dir, "../primat", sizeof(cfg2.data_dir) - 1);
    cfg2.QED_corrections = 1;
    cfg2.n_electron_table = 2000;
    cfg2.T_start_cosmo_MeV = 40.0;
    cfg2.incomplete_decoupling = 0;
    cfg2.spectral_distortions = 0;

    CPRPlasma pl2;
    if (cpr_plasma_init(&pl2, &cfg2, &err)) { printf("FAIL plasma2 init: %s\n", err); return 1; }
    CPRNeutrinoHistory nh2;
    rc = cpr_neutrino_history_init(&nh2, &cfg2, &pl2, &err);
    CHECK(rc == 0, "cpr_neutrino_history_init succeeds for InstantaneousDecoupling");
    CHECK(nh2.kind == CPR_NU_INSTANTANEOUS, "incomplete_decoupling=False selects CPR_NU_INSTANTANEOUS");
    CHECK(close_rel(cpr_nu_Tnue_of_Tg(&nh2, 1.0), 0.9941462369183307, 1e-7), "InstantaneousDecoupling Tnue(1.0) matches Python");
    CHECK(close_rel(cpr_nu_Tnue_of_Tg(&nh2, 0.01), 0.007143388657256258, 1e-7), "InstantaneousDecoupling Tnue(0.01) matches Python");
    CHECK(cpr_nu_N_NEVO_of_Tg(&nh2, 1.0) == 0.0, "InstantaneousDecoupling N(Tg) is identically 0");
    CHECK(cpr_nu_dFDneu(&nh2, 2.0, 1.0, 1.0, 1.0) == 0.0, "InstantaneousDecoupling dFDneu is identically 0 (no distortion table)");

    cpr_neutrino_history_free(&nh2);
    cpr_plasma_free(&pl2);

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
