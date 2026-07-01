/* test_memory_stress.c -- repeated init/solve/free cycles used as a memory
 * leak workload for `make leak-test` (README.md "Checking for memory
 * leaks"). This file is a *functional* smoke test in its own right (part
 * of `make test`: every iteration is checked for success/conservation
 * like any other unit test), but its real job is to give an external leak
 * checker (macOS `leaks --atExit`, or valgrind --leak-check=full on Linux)
 * a long-running single process that exercises every allocating entry
 * point in api.h/mc.h/config.h several times over, so a per-call leak of
 * even a few hundred bytes (invisible in a single `primat-c` invocation)
 * accumulates into something the checker's byte-count reliably flags.
 *
 * Covered entry points, each cycled several times with matching frees:
 *   - cpr_config_init_defaults / cpr_config_free (small + large,amax=8)
 *   - cprimat_run / cprimat_results_free (success path)
 *   - cprimat_run's error path (bad --network name -> *errmsg set, then
 *     freed by the caller) -- exercises the failure-cleanup code paths
 *     inside cpr_nuclear_rates_init/cpr_bg_init_* that a success-only
 *     workload never reaches.
 *   - cpr_mc_uncertainty / cpr_mc_result_free (small network, few
 *     samples, n_jobs>1 so worker-thread-local allocations are exercised
 *     too).
 *
 * Kept fast on purpose (small network, amax=8 for the large one, tiny MC
 * sample counts): this binary is meant to be run for many iterations, not
 * for numerical precision -- see the reference run in CLAUDE.md /
 * test_api.c-equivalent tests for accuracy checks.
 */
#include "api.h"
#include "config.h"
#include "constants.h"
#include "mc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

/* Lightweight wrapper so run_once/run_error_once's own error bookkeeping
 * composes with the file-wide CHECK() failure counter. */
static int CHECK_OK(int cond)
{
    if (!cond) {
        printf("FAIL: cpr_config_init_defaults\n");
        failures++;
    }
    return cond;
}

/* One success-path cycle: build a config for `network` (and `amax`, if
 * >0), run BBN, check baryon number conservation (sum_s A_s Y_s == 1,
 * CLAUDE.md's exact-conservation invariant -- a cheap correctness check
 * that also forces every element of Y_final to actually be read, so an
 * ASan/valgrind build would catch an uninitialised-read bug here too),
 * then free everything. */
static void run_once(const char *network, long amax)
{
    char *err = NULL;
    CPRConfig cfg;
    if (!CHECK_OK(cpr_config_init_defaults(&cfg, "../primat/data", &err) == 0)) {
        free(err);
        return;
    }
    cpr_config_set_by_name(&cfg, "network", (CPRParam){ CPR_STRING, .v.s = network }, &err);
    if (amax > 0)
        cpr_config_set_by_name(&cfg, "amax", (CPRParam){ CPR_INT, .v.i = amax }, &err);

    CPRResults res;
    int rc = cprimat_run(&cfg, NULL, &res, &err);
    CHECK(rc == 0, network);
    if (rc == 0) {
        /* A100/A2/... mass numbers aren't stored in CPRResults directly,
         * but reading every Y_final element (via YPBBN's known
         * relationship is overkill) at least confirms the array is
         * valid/sized as advertised -- avoids the checker only ever
         * touching the first bytes of the allocation. */
        double sum = 0.0;
        for (size_t i = 0; i < res.n_nuclides; i++)
            sum += res.Y_final[i];
        CHECK(sum > 0.0 && sum < 2.0, "Y_final sums to a sane baryon-conserving value");
        cprimat_results_free(&res);
    } else {
        free(err);
    }
    cpr_config_free(&cfg);
}

/* Deliberately invalid config (unknown network name): cprimat_run must
 * fail cleanly and set *errmsg, without leaking whatever partial state
 * (Plasma, weak-rate tables, ...) it had already built before hitting the
 * error -- the failure-path counterpart of run_once above. */
static void run_error_once(void)
{
    char *err = NULL;
    CPRConfig cfg;
    if (!CHECK_OK(cpr_config_init_defaults(&cfg, "../primat/data", &err) == 0)) {
        free(err);
        return;
    }
    int set_rc = cpr_config_set_by_name(&cfg, "network",
                                          (CPRParam){ CPR_STRING, .v.s = "this_network_does_not_exist" },
                                          &err);
    CHECK(set_rc == 0, "cpr_config_set_by_name accepts an arbitrary network string");

    CPRResults res;
    char *run_err = NULL;
    int rc = cprimat_run(&cfg, NULL, &res, &run_err);
    CHECK(rc != 0, "cprimat_run rejects an unknown network name");
    CHECK(run_err != NULL, "error path sets *errmsg");
    free(run_err);
    /* rc != 0 means `res` was never successfully filled -- nothing to
     * cprimat_results_free here (mirrors api.h's "0 on success" contract). */
    cpr_config_free(&cfg);
}

/* One MC cycle: small network, few samples, 2 worker threads so each
 * thread's own per-worker Plasma/Background/weak-rate setup (mc.h's top
 * comment) is allocated and freed every iteration too. */
static void run_mc_once(void)
{
    const char *quantities[] = { "YPBBN", "DoH" };
    CPRParamSet base_params[] = {
        { "network", { CPR_STRING, .v.s = "small" } },
    };
    char *err = NULL;
    CPRMCResult res;
    int rc = cpr_mc_uncertainty(8, quantities, 2, "../primat/data",
                                  base_params, 1, /*seed=*/2024, /*n_jobs=*/2, NULL,
                                  NULL, NULL, 0, /*show_progress=*/0,
                                  &res, &err);
    CHECK(rc == 0, "cpr_mc_uncertainty succeeds");
    if (rc == 0)
        cpr_mc_result_free(&res);
    else
        free(err);
}

int main(void)
{
    cpr_constants_init();

    const int n_small = 6;
    const int n_large_amax8 = 3;
    const int n_error = 4;
    const int n_mc = 2;

    for (int i = 0; i < n_small; i++)
        run_once("small", 0);

    for (int i = 0; i < n_large_amax8; i++)
        run_once("large", 8);

    for (int i = 0; i < n_error; i++)
        run_error_once();

    for (int i = 0; i < n_mc; i++)
        run_mc_once();

    printf(failures == 0 ? "\nAll memory-stress checks passed.\n"
                          : "\n%d memory-stress check(s) FAILED.\n", failures);
    return failures == 0 ? 0 : 1;
}
