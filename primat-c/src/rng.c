/* rng.c -- see cprimat/rng.h. xoshiro256** (public domain, D. Blackman &
 * S. Vigna, https://prng.di.unimi.it/) seeded via SplitMix64. */
#include "rng.h"

#include <math.h>

static uint64_t rotl(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

void cpr_rng_seed(CPRRng *rng, uint64_t seed)
{
    /* SplitMix64: expands the single 64-bit seed into 4 well-mixed 64-bit
     * words, avoiding the all-zero state xoshiro256** cannot recover from. */
    uint64_t z = seed;
    for (int i = 0; i < 4; i++) {
        z += 0x9E3779B97F4A7C15ULL;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        x = x ^ (x >> 31);
        rng->s[i] = x;
    }
}

uint64_t cpr_rng_next(CPRRng *rng)
{
    uint64_t *s = rng->s;
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    return result;
}

double cpr_rng_uniform(CPRRng *rng)
{
    /* Top 53 bits -> exact double in [0, 1) (the standard "53-bit"
     * technique, matching the precision of a double's mantissa). */
    return (double)(cpr_rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

double cpr_rng_normal(CPRRng *rng)
{
    double u1, u2;
    do { u1 = cpr_rng_uniform(rng); } while (u1 <= 0.0); /* avoid log(0) */
    u2 = cpr_rng_uniform(rng);
    double r = sqrt(-2.0 * log(u1));
    return r * cos(2.0 * M_PI * u2);
}
