/* mc.h -- threaded Monte-Carlo nuclear-rate/tau_n uncertainty propagation
 * (port of pyprimat/main.py's `mc_uncertainty`/`_mc_run_batch`,
 * CPLAN.md S9).
 *
 * Each of `num_mc` samples draws every active nuclear-rate offset
 * `p_<rxn>` (one per reaction in the chosen LT-era network) independently
 * from N(0,1), plus `tau_n ~ N(cfg.tau_n, cfg.std_tau_n)`, and re-solves
 * the full BBN network. The expensive, p_*-independent part of the setup
 * (Plasma, the cosmological Background, the n<->p weak rates) is built
 * once *per worker thread* and reused across every sample that thread
 * draws -- mirroring `_mc_run_batch`'s "expensive background computed
 * once, reused across samples" design -- so only the cheap part (redraw
 * rates, re-solve the nuclear network) repeats per sample.
 *
 * Unlike Python's `joblib`-process-based parallelism (which needs
 * pickling and a `default_rng(seed)` per sample to stay order-independent
 * across chunking), CPRIMAT uses POSIX `pthread`s sharing one address
 * space: each worker still draws its own `CPRRng` seeded from `seed + i`
 * per sample `i` (rng.h's xoshiro256**, *not* bit-identical to NumPy's
 * `default_rng` -- see rng.h's top comment), so results are deterministic
 * per (seed, sample index) and independent of how many threads are used,
 * but are not expected to reproduce Python's MC sample values term for
 * term (only matched statistically, mean/std convergence -- CPLAN.md
 * S11's "Additional smoke/regression tests").
 */
#ifndef CPRIMAT_MC_H
#define CPRIMAT_MC_H

#include "cprimat/config.h"
#include <stddef.h>

typedef struct {
    char name[40];   /* quantity name, e.g. "YPBBN", "DoH", "H2" (cpr_results_get_quantity) */
    double central;  /* value at nominal rates (all p_<rxn>=0, tau_n=cfg.tau_n) */
    double mean;      /* mean of `values` */
    double std;       /* population standard deviation of `values` */
    double *values;   /* length num_mc, owned; sample i drawn from seed+i */
} CPRMCQuantity;

typedef struct {
    CPRMCQuantity *items;
    size_t n;
} CPRMCResult;

/* Runs the full MC propagation. `rates_dir` + `base_params`/`n_base_params`
 * (key/value overrides, applied in order via cpr_config_set_by_name --
 * mirrors cli.c's --set handling and Python's `params` dict) build the
 * base config each worker thread re-derives its own CPRConfig from.
 * `quantities`/`n_quantities` name the result-dict keys or nuclide names
 * to collect (cpr_results_get_quantity); an unknown name is an error.
 * `seed` is the base RNG seed (sample i uses seed+i); `n_jobs` is the
 * number of worker threads (<=0 means "use all detected cores").
 *
 * Fills `out` (zeroed first; caller must cpr_mc_result_free). Returns 0 on
 * success, nonzero with *errmsg set (caller frees) on any config/init/
 * solve failure (the first one encountered, possibly from any worker --
 * remaining workers are still joined before returning). */
int cpr_mc_uncertainty(int num_mc, const char * const *quantities, size_t n_quantities,
                        const char *rates_dir,
                        const CPRParamSet *base_params, size_t n_base_params,
                        int seed, int n_jobs,
                        CPRMCResult *out, char **errmsg);

void cpr_mc_result_free(CPRMCResult *out);

/* Convenience accessor: index of `name` within `out->items`, or
 * `out->n` if not found (mirrors MCResult.__getitem__'s dict lookup, as a
 * linear scan since n_quantities is always small). */
size_t cpr_mc_result_index(const CPRMCResult *out, const char *name);

#endif /* CPRIMAT_MC_H */
