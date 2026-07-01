/* test_api.c -- direct checks of api.c's cprimat_run/cpr_results_get_quantity,
 * independent of primat/_primat_c/_wrapper.c and Python's
 * tests/test_backend_parity.py (which only exercise cprimat_run indirectly,
 * through the compiled Python extension). Covers three things the parity
 * tests don't reach directly at the C level:
 *   - the error path (bad network name) frees any partial state cleanly
 *     and reports *errmsg (see also test_memory_stress.c's leak-oriented
 *     version of the same check);
 *   - cfg->output_time_evolution populates CPRResults's evol_* in-memory
 *     arrays with a sane shape, matching what
 *     primat/_primat_c/_wrapper.c relies on to avoid any disk I/O;
 *   - a custom_network override (GUI "Customise Reactions", CPRCustomNetwork)
 *     actually changes cprimat_run's result relative to the unmodified
 *     network, and cpr_results_get_quantity resolves both fixed observable
 *     names and per-nuclide abundance names.
 */
#include "api.h"
#include "config.h"
#include "constants.h"
#include "network_data.h"

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
    cpr_constants_init();

    /* ---- Error path: unknown network name ---- */
    {
        char *err = NULL;
        CPRConfig cfg;
        if (cpr_config_init_defaults(&cfg, "../primat/data", &err)) {
            printf("FAIL cpr_config_init_defaults: %s\n", err);
            return 1;
        }
        cpr_config_set_by_name(&cfg, "network",
                                (CPRParam){ CPR_STRING, .v.s = "this_network_does_not_exist" }, &err);

        CPRResults res;
        char *run_err = NULL;
        int rc = cprimat_run(&cfg, NULL, &res, &run_err);
        CHECK(rc != 0, "cprimat_run rejects an unknown network name");
        CHECK(run_err != NULL, "error path sets *errmsg");
        free(run_err);
        cpr_config_free(&cfg);
    }

    /* ---- output_time_evolution populates CPRResults's evol_* arrays ---- */
    {
        char *err = NULL;
        CPRConfig cfg;
        if (cpr_config_init_defaults(&cfg, "../primat/data", &err)) {
            printf("FAIL cpr_config_init_defaults: %s\n", err);
            return 1;
        }
        cpr_config_set_by_name(&cfg, "output_time_evolution", (CPRParam){ CPR_BOOL, .v.b = 1 }, &err);

        CPRResults res;
        int rc = cprimat_run(&cfg, NULL, &res, &err);
        CHECK(rc == 0, "cprimat_run succeeds with output_time_evolution=True");
        if (rc == 0) {
            CHECK(res.has_evolution, "has_evolution is set");
            CHECK(res.n_evolution > 1, "n_evolution has more than one sampled row");
            CHECK(res.evol_t != NULL && res.evol_a != NULL && res.evol_T_gamma != NULL,
                  "evol_t/evol_a/evol_T_gamma arrays are populated");
            CHECK(res.evol_Y != NULL, "evol_Y (row-major t x nuclide) is populated");
            /* Monotonically increasing time and decreasing temperature, the
             * minimal shape sanity check that this is a real time series
             * and not e.g. an uninitialised/zeroed buffer. */
            CHECK(res.evol_t[res.n_evolution - 1] > res.evol_t[0],
                  "evol_t increases from first to last sample");
            CHECK(res.evol_T_gamma[res.n_evolution - 1] < res.evol_T_gamma[0],
                  "evol_T_gamma decreases from first to last sample (Universe cools)");
            /* Every evol_Y entry for the final row must sum close to the
             * same value as Y_final (both are the same physical state). */
            double sum_last_row = 0.0;
            size_t base = (res.n_evolution - 1) * res.n_nuclides;
            for (size_t i = 0; i < res.n_nuclides; i++)
                sum_last_row += res.evol_Y[base + i];
            double sum_Y_final = 0.0;
            for (size_t i = 0; i < res.n_nuclides; i++)
                sum_Y_final += res.Y_final[i];
            CHECK(fabs(sum_last_row - sum_Y_final) < 1e-6,
                  "evol_Y's last row matches Y_final's total abundance");
            cprimat_results_free(&res);
        } else {
            free(err);
        }
        cpr_config_free(&cfg);
    }

    /* ---- custom_network changes the result, and get_quantity resolves
     * both fixed fields and per-nuclide names ---- */
    {
        char *err = NULL;
        CPRConfig cfg;
        if (cpr_config_init_defaults(&cfg, "../primat/data", &err)) {
            printf("FAIL cpr_config_init_defaults: %s\n", err);
            return 1;
        }

        CPRResults res_default;
        CHECK(cprimat_run(&cfg, NULL, &res_default, &err) == 0,
              "baseline small-network run succeeds");

        /* Remove the Li7(p,a)a reaction: the dominant Li7-depletion channel,
         * so its absence must visibly raise Li7oH relative to the baseline
         * (same reaction the GUI custom-network test drives, see
         * test_network_data_phase4.c). */
        char removed_names[1][64];
        snprintf(removed_names[0], sizeof(removed_names[0]), "Li7_p__a_a");
        CPRCustomNetwork custom = { .removed = removed_names, .n_removed = 1,
                                     .tables = NULL, .n_tables = 0 };

        CPRResults res_custom;
        int rc = cprimat_run(&cfg, &custom, &res_custom, &err);
        CHECK(rc == 0, "custom_network run succeeds");
        if (rc == 0) {
            CHECK(res_custom.Li7oH > res_default.Li7oH,
                  "removing Li7(p,a)a raises Li7oH relative to the baseline run");

            int found = 0;
            double doh = cpr_results_get_quantity(&res_custom, "DoH", &found);
            CHECK(found && doh == res_custom.DoH, "get_quantity resolves a fixed field (DoH)");

            found = 0;
            double y_he4 = cpr_results_get_quantity(&res_custom, "He4", &found);
            CHECK(found && y_he4 > 0.0, "get_quantity resolves a per-nuclide name (He4)");

            found = 1;
            cpr_results_get_quantity(&res_custom, "not_a_real_quantity", &found);
            CHECK(found == 0, "get_quantity reports not-found for an unknown name");

            cprimat_results_free(&res_custom);
        } else {
            free(err);
        }
        cprimat_results_free(&res_default);
        cpr_config_free(&cfg);
    }

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
