/* rng.h -- xoshiro256** PRNG + Box-Muller normal sampling (CPLAN.md S3.4).
 *
 * Used by mc_uncertainty's per-reaction rate draws and the tau_n draw.
 * Deliberately NOT required to reproduce NumPy's default_rng bit-for-bit:
 * mc_uncertainty is a statistical estimate, validated by mean/std
 * convergence, not a reference value compared term-by-term against Python.
 */
#ifndef CPRIMAT_RNG_H
#define CPRIMAT_RNG_H

#include <stdint.h>

typedef struct {
    uint64_t s[4];
} CPRRng;

/* Seeds the generator via SplitMix64 (the standard way to expand a single
 * 64-bit seed into xoshiro256**'s 256 bits of state without ever landing on
 * the forbidden all-zero state). */
void cpr_rng_seed(CPRRng *rng, uint64_t seed);

/* Returns a uniform random uint64 (the generator's native output). */
uint64_t cpr_rng_next(CPRRng *rng);

/* Returns a uniform double in [0, 1). */
double cpr_rng_uniform(CPRRng *rng);

/* Returns a standard-normal (mean 0, variance 1) sample via the
 * Box-Muller transform. Each call draws two fresh uniforms and returns one
 * of the two resulting normal deviates (the other is discarded) -- simpler
 * and stateless, at the cost of one wasted uniform pair per call; mc's
 * draw counts are small enough that this is not a measurable cost. */
double cpr_rng_normal(CPRRng *rng);

#endif /* CPRIMAT_RNG_H */
