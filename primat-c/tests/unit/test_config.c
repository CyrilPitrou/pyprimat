/* test_config.c -- checks cpr_config_init_defaults/cpr_config_set_by_name's
 * generic override dispatch, the C equivalent of
 * ../../tests/test_config.py's PyPRConfig checks: sane defaults, that a
 * named override actually lands on the right typed field (bool/int/double/
 * string), that an unknown key is reported as an error (mirrors Python's
 * "unknown key" warning -- the C port makes it a hard error via *errmsg
 * rather than a warning, since cli.c/ini.c are the ones that decide to
 * downgrade it), and that p_<rxn>/delta_<rxn> keys route into the
 * corresponding CPRRxnMap instead of the fixed-field table.
 */
#include "config.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

int main(void)
{
    char *err = NULL;
    CPRConfig cfg;
    if (cpr_config_init_defaults(&cfg, "../primat/data", &err)) {
        printf("FAIL cpr_config_init_defaults: %s\n", err);
        return 1;
    }

    /* ---- Defaults sanity (mirrors test_config.py's test_default_construction) ---- */
    CHECK(strcmp(cfg.network, "small") == 0, "default network is 'small'");
    CHECK(cfg.numerical_precision == 1e-7, "default numerical_precision is 1e-7");
    CHECK(cfg.incomplete_decoupling == 1, "default incomplete_decoupling is True");
    CHECK(cfg.QED_corrections == 1, "default QED_corrections is True");

    /* ---- Overrides land on the right typed field, one per kind ---- */
    char *set_err = NULL;
    CPRParam p_bool = { .type = CPR_BOOL, .v.b = 0 };
    CHECK(cpr_config_set_by_name(&cfg, "QED_corrections", p_bool, &set_err) == 0,
          "set_by_name accepts a bool override");
    CHECK(cfg.QED_corrections == 0, "QED_corrections overridden to False");

    CPRParam p_int = { .type = CPR_INT, .v.i = 1234 };
    CHECK(cpr_config_set_by_name(&cfg, "n_electron_table", p_int, &set_err) == 0,
          "set_by_name accepts an int override");
    CHECK(cfg.n_electron_table == 1234, "n_electron_table overridden to 1234");

    CPRParam p_double = { .type = CPR_DOUBLE, .v.d = 1e-9 };
    CHECK(cpr_config_set_by_name(&cfg, "numerical_precision", p_double, &set_err) == 0,
          "set_by_name accepts a double override");
    CHECK(cfg.numerical_precision == 1e-9, "numerical_precision overridden to 1e-9");

    CPRParam p_str = { .type = CPR_STRING, .v.s = "large" };
    CHECK(cpr_config_set_by_name(&cfg, "network", p_str, &set_err) == 0,
          "set_by_name accepts a string override");
    CHECK(strcmp(cfg.network, "large") == 0, "network overridden to 'large'");

    /* An int is a valid CPR_DOUBLE-field value too (numeric widening,
     * mirroring Python's duck-typed int/float interchangeability). */
    CPRParam p_int_for_double = { .type = CPR_INT, .v.i = 2 };
    CHECK(cpr_config_set_by_name(&cfg, "numerical_precision", p_int_for_double, &set_err) == 0,
          "set_by_name widens an int to a double field");
    CHECK(cfg.numerical_precision == 2.0, "numerical_precision widened to 2.0");

    /* A string value on a bool field is a type mismatch -> error, not a
     * silent truthy coercion. */
    CPRParam p_str_for_bool = { .type = CPR_STRING, .v.s = "true" };
    free(set_err); set_err = NULL;
    int rc = cpr_config_set_by_name(&cfg, "QED_corrections", p_str_for_bool, &set_err);
    CHECK(rc != 0, "set_by_name rejects a string value for a bool field");
    CHECK(set_err != NULL, "type-mismatch error message is set");
    free(set_err); set_err = NULL;

    /* ---- Unknown key is an error (mirrors test_unknown_key_warns) ---- */
    CPRParam p_anything = { .type = CPR_INT, .v.i = 1 };
    rc = cpr_config_set_by_name(&cfg, "not_a_real_parameter", p_anything, &set_err);
    CHECK(rc != 0, "unknown key is reported as an error");
    CHECK(set_err != NULL && strstr(set_err, "not_a_real_parameter") != NULL,
          "unknown-key error message names the bad key");
    free(set_err); set_err = NULL;

    /* ---- p_<rxn>/delta_<rxn> route into the reaction-rate maps, not the
     * fixed-field table (mirrors test_p_rxn_typo_warns's sibling,
     * test_p_rxn_valid_reaction_does_not_warn) ---- */
    CPRParam p_rate = { .type = CPR_DOUBLE, .v.d = 0.5 };
    CHECK(cpr_config_set_by_name(&cfg, "p_n_p__d_g", p_rate, &set_err) == 0,
          "p_<rxn> key is accepted");
    CHECK(cpr_rxnmap_get(&cfg.p_rxn, "n_p__d_g") == 0.5,
          "p_n_p__d_g landed in cfg.p_rxn under the bare reaction name");
    CHECK(cpr_rxnmap_get(&cfg.p_rxn, "some_other_reaction") == 0.0,
          "cpr_rxnmap_get defaults to 0.0 for an unset reaction");

    CPRParam delta_rate = { .type = CPR_DOUBLE, .v.d = -0.25 };
    CHECK(cpr_config_set_by_name(&cfg, "delta_n_p__d_g", delta_rate, &set_err) == 0,
          "delta_<rxn> key is accepted");
    CHECK(cpr_rxnmap_get(&cfg.delta_rxn, "n_p__d_g") == -0.25,
          "delta_n_p__d_g landed in cfg.delta_rxn, independent of cfg.p_rxn");

    /* ---- Omegabh2 setter recomputes eta0b (dedicated branch in
     * cpr_config_set_by_name, not the generic field table) ---- */
    double eta0b_before = cfg.eta0b;
    CPRParam p_ombh2 = { .type = CPR_DOUBLE, .v.d = 0.02 };
    CHECK(cpr_config_set_by_name(&cfg, "Omegabh2", p_ombh2, &set_err) == 0,
          "Omegabh2 override is accepted");
    CHECK(cpr_config_get_Omegabh2(&cfg) == 0.02, "Omegabh2 getter reflects the override");
    CHECK(cfg.eta0b != eta0b_before, "eta0b was recomputed after the Omegabh2 override");

    cpr_config_free(&cfg);

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
