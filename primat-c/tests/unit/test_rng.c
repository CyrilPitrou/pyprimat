/* test_rng.c -- statistical sanity checks for the xoshiro256** / Box-Muller
 * RNG: not bit-matched to anything, just checked for range, determinism
 * given a seed, and mean/std convergence of the normal sampler. */
#include "cprimat/rng.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

int main(void)
{
    CPRRng rng1, rng2;
    cpr_rng_seed(&rng1, 42);
    cpr_rng_seed(&rng2, 42);

    int deterministic = 1;
    for (int i = 0; i < 100; i++)
        if (cpr_rng_next(&rng1) != cpr_rng_next(&rng2)) { deterministic = 0; break; }
    CHECK(deterministic, "same seed produces same sequence");

    cpr_rng_seed(&rng1, 1);
    cpr_rng_seed(&rng2, 2);
    CHECK(cpr_rng_next(&rng1) != cpr_rng_next(&rng2), "different seeds produce different sequences");

    cpr_rng_seed(&rng1, 12345);
    int in_range = 1;
    for (int i = 0; i < 100000; i++) {
        double u = cpr_rng_uniform(&rng1);
        if (u < 0.0 || u >= 1.0) { in_range = 0; break; }
    }
    CHECK(in_range, "uniform samples stay in [0, 1)");

    cpr_rng_seed(&rng1, 777);
    size_t N = 200000;
    double sum = 0.0, sumsq = 0.0;
    for (size_t i = 0; i < N; i++) {
        double z = cpr_rng_normal(&rng1);
        sum += z;
        sumsq += z * z;
    }
    double mean = sum / (double)N;
    double var = sumsq / (double)N - mean * mean;
    /* mean ~ N(0, 1/N); for N=2e5 the std error of the mean is ~0.0022,
     * so a 0.05 tolerance is generously loose to avoid flakiness. */
    CHECK(fabs(mean) < 0.05, "normal sampler mean is close to 0");
    CHECK(fabs(var - 1.0) < 0.05, "normal sampler variance is close to 1");

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
