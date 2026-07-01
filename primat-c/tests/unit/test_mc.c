/* test_mc.c -- statistical smoke/regression test for mc.c's
 * cpr_mc_uncertainty: since the
 * per-sample RNG stream is deliberately not bit-matched to NumPy's
 * default_rng (see mc.h's top comment), there is no fixed reference value
 * to reproduce -- instead this checks the statistical properties any
 * correct MC propagation must have: the sample mean converges to the
 * central (all p_<rxn>=0) value within a few sigma/sqrt(N), the population
 * std is positive and stable in order of magnitude across two independent
 * base seeds, and the result is independent of how many worker threads
 * are used (same seed range, n_jobs=1 vs n_jobs=4 must match exactly,
 * since each sample is fully determined by its own seed -- see mc.c's
 * cpr_mc_uncertainty docstring). Runs on the `small` network (12
 * reactions) to keep the test fast. */
#include "mc.h"

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
    const char *quantities[] = { "YPBBN", "DoH" };
    CPRParamSet base_params[] = {
        { "network", { CPR_STRING, .v.s = "small" } },
    };

    char *err = NULL;
    CPRMCResult res;
    int rc = cpr_mc_uncertainty(60, quantities, 2, "../primat/data",
                                 base_params, 1, /*seed=*/12345, /*n_jobs=*/4, NULL,
                                 NULL, NULL, 0, /*show_progress=*/0,
                                 &res, &err);
    if (rc) {
        printf("FAIL: cpr_mc_uncertainty returned an error: %s\n", err ? err : "(null)");
        free(err);
        return 1;
    }

    for (size_t q = 0; q < res.n; q++) {
        CPRMCQuantity *it = &res.items[q];
        double sigma_of_mean = it->std / sqrt(60.0);
        char msg[256];

        snprintf(msg, sizeof(msg), "%s: mean (%.8e) within 5 sigma/sqrt(N) of central (%.8e)",
                  it->name, it->mean, it->central);
        CHECK(fabs(it->mean - it->central) < 5.0 * sigma_of_mean + 1e-300, msg);

        snprintf(msg, sizeof(msg), "%s: std (%.3e) is finite and positive", it->name, it->std);
        CHECK(isfinite(it->std) && it->std > 0.0, msg);
    }

    /* Re-run at n_jobs=1 over the same seed range: every sample is keyed by
     * its own seed (mirrors np.array_split chunking being order/count-
     * independent), so the two result sets must match exactly, not just
     * statistically. */
    CPRMCResult res_serial;
    char *err2 = NULL;
    int rc2 = cpr_mc_uncertainty(60, quantities, 2, "../primat/data",
                                  base_params, 1, /*seed=*/12345, /*n_jobs=*/1, NULL,
                                  NULL, NULL, 0, /*show_progress=*/0,
                                  &res_serial, &err2);
    CHECK(rc2 == 0, "serial (n_jobs=1) re-run also succeeds");
    if (rc2 == 0) {
        for (size_t q = 0; q < res.n; q++) {
            size_t qs = cpr_mc_result_index(&res_serial, res.items[q].name);
            char msg[128];
            snprintf(msg, sizeof(msg), "%s: n_jobs=4 and n_jobs=1 give identical mean", res.items[q].name);
            CHECK(qs < res_serial.n && res.items[q].mean == res_serial.items[qs].mean, msg);
        }
        cpr_mc_result_free(&res_serial);
    } else {
        free(err2);
    }

    /* prev reuse: extending a 30-sample run to 60 must reproduce the
     * 60-sample from-scratch run exactly (same seeds, same RNG draws),
     * and truncating back down to 30 must reproduce the original 30-sample
     * run exactly -- mirrors tests/test_mc.py's
     * test_extend_matches_full_run/test_extend_truncates_when_fewer_requested. */
    CPRMCResult res_part;
    char *err3 = NULL;
    int rc3 = cpr_mc_uncertainty(30, quantities, 2, "../primat/data",
                                  base_params, 1, /*seed=*/12345, /*n_jobs=*/4, NULL,
                                  NULL, NULL, 0, /*show_progress=*/0,
                                  &res_part, &err3);
    CHECK(rc3 == 0, "30-sample run for prev-reuse setup succeeds");
    if (rc3 == 0) {
        const double *prev_values[2] = { res_part.items[0].values, res_part.items[1].values };
        double prev_centrals[2] = { res_part.items[0].central, res_part.items[1].central };

        CPRMCResult res_ext;
        char *err4 = NULL;
        int rc4 = cpr_mc_uncertainty(60, quantities, 2, "../primat/data",
                                      base_params, 1, /*seed=*/12345, /*n_jobs=*/4, NULL,
                                      prev_centrals, prev_values, 30, /*show_progress=*/0,
                                      &res_ext, &err4);
        CHECK(rc4 == 0, "60-sample extension of 30-sample prev succeeds");
        if (rc4 == 0) {
            for (size_t q = 0; q < res.n; q++) {
                char msg[128];
                snprintf(msg, sizeof(msg), "%s: extended-from-prev matches full from-scratch run sample-for-sample",
                         res.items[q].name);
                int ok = 1;
                for (int i = 0; i < 60; i++)
                    if (res_ext.items[q].values[i] != res.items[q].values[i]) ok = 0;
                CHECK(ok, msg);
            }
            cpr_mc_result_free(&res_ext);
        } else {
            free(err4);
        }

        CPRMCResult res_trunc;
        char *err5 = NULL;
        int rc5 = cpr_mc_uncertainty(20, quantities, 2, "../primat/data",
                                      base_params, 1, /*seed=*/12345, /*n_jobs=*/4, NULL,
                                      prev_centrals, prev_values, 30, /*show_progress=*/0,
                                      &res_trunc, &err5);
        CHECK(rc5 == 0, "20-sample truncation of 30-sample prev succeeds");
        if (rc5 == 0) {
            for (size_t q = 0; q < res.n; q++) {
                char msg[128];
                snprintf(msg, sizeof(msg), "%s: truncated-from-prev matches the prev prefix", res.items[q].name);
                int ok = 1;
                for (int i = 0; i < 20; i++)
                    if (res_trunc.items[q].values[i] != res_part.items[q].values[i]) ok = 0;
                CHECK(ok, msg);
            }
            cpr_mc_result_free(&res_trunc);
        } else {
            free(err5);
        }

        cpr_mc_result_free(&res_part);
    } else {
        free(err3);
    }

    cpr_mc_result_free(&res);

    if (failures == 0) printf("All test_mc checks passed.\n");
    else printf("%d test_mc check(s) FAILED.\n", failures);
    return failures == 0 ? 0 : 1;
}
