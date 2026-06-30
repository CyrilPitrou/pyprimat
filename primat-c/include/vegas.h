/* vegas.h -- 2D VEGAS adaptive-importance-sampling Monte Carlo integration
 * (Lepage 1978), replacing the deterministic nested-quadrature path
 * previously used for weak_rates.c's thermal (CCRTh) sub-integrals.
 *
 * Ports the *algorithm* used by Python's `vegas` package (corrections.py's
 * `_L_ThermalTruePhoton`/`_L_ThermalDiffBremsstrahlung`/`_L_Thermal_2_3`),
 * not its bit-for-bit RNG stream: this integrator uses cprimat's own
 * xoshiro256** RNG (rng.h), seeded deterministically from the caller so
 * results are reproducible run-to-run in C, unlike Python's unseeded
 * `vegas.Integrator`. Cross-backend agreement is only expected to the
 * ~1e-3 relative Monte-Carlo noise floor both sides already accept for
 * this term (see weak_rates.c's CCRTh docstring), not bit-for-bit.
 */
#ifndef CPRIMAT_VEGAS_H
#define CPRIMAT_VEGAS_H

#include <stdint.h>

typedef double (*CPRVegasFunc)(const double x[2], void *ctx);

typedef struct {
    double mean;   /* inverse-variance-weighted estimate of the integral */
    double sigma;  /* combined standard error of that estimate */
} CPRVegasResult;

/* Integrates f over the rectangle [lo[0],hi[0]] x [lo[1],hi[1]].
 *
 * Two phases, each of n_eval samples per iteration:
 *  - n_itn_warmup iterations: adapt the per-dimension importance-sampling
 *    grid only: rebuild it after every iteration from that iteration's
 *    accumulated f^2-weighted histogram (Lepage's damped refinement, see
 *    vegas.c), discard the estimates themselves.
 *  - n_itn_measure further iterations: keep adapting the grid the same way,
 *    but now also accumulate each iteration's (mean, variance) into a
 *    running inverse-variance-weighted combination, returned as the result.
 * This mirrors corrections.py's "call the integrator once to let it adapt,
 * call it again and keep the second result" idiom -- total evaluation
 * budget is (n_itn_warmup + n_itn_measure) * n_eval calls to f, matching
 * Python's 2 * cfg.vegas_n_itn * cfg.vegas_n_eval when both phase lengths
 * are set to cfg.vegas_n_itn (the call sites in weak_rates.c do this).
 *
 * seed: deterministically derived per call site (see weak_rates.c) so a
 * given (T, sgnq, integrand) combination always reproduces the same value.
 */
CPRVegasResult cpr_vegas_integrate(CPRVegasFunc f, void *ctx,
                                    const double lo[2], const double hi[2],
                                    int n_eval, int n_itn_warmup,
                                    int n_itn_measure, uint64_t seed);

#endif /* CPRIMAT_VEGAS_H */
