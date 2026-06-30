/* test_cache.c -- verifies the fingerprint hash matches Python bit-for-bit.
 *
 * Ground truth was captured directly from the running Python code:
 *   python3 -c "
 *     from primat.config import PRIMATConfig
 *     from primat.weak_rates.cache import _weak_rate_fingerprint, _thermal_fingerprint
 *     from primat.cache_utils import fingerprint_hash
 *     cfg = PyPRConfig()
 *     print(fingerprint_hash(_weak_rate_fingerprint(cfg)))
 *     print(fingerprint_hash(_thermal_fingerprint(cfg)))
 *   "
 * -> 2218248995f018af
 * -> 0eccbdd5dbb5dd93
 * (default PyPRConfig(): incomplete_decoupling/QED_corrections/
 * spectral_distortions/radiative_corrections/finite_mass_corrections all
 * True, analytic_distortions False, every numeric default unchanged.)
 */
#include "cache.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void expect_str_eq(const char *what, const char *got, const char *want)
{
    if (strcmp(got, want) != 0) {
        printf("FAIL %s: got %s, want %s\n", what, got, want);
        failures++;
    } else {
        printf("ok   %s: %s\n", what, got);
    }
}

int main(void)
{
    char *errmsg = NULL;
    CPRConfig cfg;
    if (cpr_config_init_defaults(&cfg, "../primat/data", &errmsg)) {
        printf("FAIL cpr_config_init_defaults: %s\n", errmsg);
        return 1;
    }

    CPRFPField wfields[20];
    size_t nw = cpr_weak_rate_fingerprint(&cfg, wfields);
    char *wjson = cpr_fingerprint_json(wfields, nw);
    char *whash = cpr_sha256_hex16(wjson);
    printf("weak json: %s\n", wjson);
    expect_str_eq("weak_rate_fingerprint hash", whash, "2218248995f018af");
    free(wjson);
    free(whash);

    CPRFPField tfields[10];
    size_t nt = cpr_thermal_fingerprint(&cfg, tfields);
    char *tjson = cpr_fingerprint_json(tfields, nt);
    char *thash = cpr_sha256_hex16(tjson);
    printf("thermal json: %s\n", tjson);
    expect_str_eq("thermal_fingerprint hash", thash, "0eccbdd5dbb5dd93");
    free(tjson);
    free(thash);

    /* Round-trip: read back the hash header of an existing Python-written
     * cache file and confirm cpr_cache_read_fingerprint_hash parses it.
     * Uses the default-config hash (2218248995f018af, also relied on by
     * test_weak_rates.c) rather than a one-off file, since that one is
     * load-bearing for another test and so less likely to silently
     * disappear from a future "refresh shipped weak-rate caches"-style
     * regeneration on the Python side. */
    char *read_hash = cpr_cache_read_fingerprint_hash(
        "../primat/data/weak/nTOp_2218248995f018af.txt");
    if (!read_hash) {
        printf("FAIL reading existing cache file header\n");
        failures++;
    } else {
        expect_str_eq("read existing cache file's own hash header",
                       read_hash, "2218248995f018af");
        free(read_hash);
    }

    /* Write+read round-trip with our own writer. */
    double col0[3] = {1.0, 2.0, 3.0};
    double col1[3] = {0.1, 0.2, 0.3};
    double *cols[2] = {col0, col1};
    const char *out_path = "/tmp/cprimat_test_cache.txt";
    if (cpr_cache_write(out_path, wfields, nw, "T[K] rate", cols, 2, 3, NULL)) {
        printf("FAIL cpr_cache_write\n");
        failures++;
    } else {
        char *rt_hash = cpr_cache_read_fingerprint_hash(out_path);
        expect_str_eq("write-then-read-back hash", rt_hash, "2218248995f018af");
        free(rt_hash);
    }

    cpr_config_free(&cfg);

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
