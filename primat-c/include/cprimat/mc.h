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
#include "cprimat/network_data.h"
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
 * number of worker threads (<=0 means "use all detected cores"). `custom`
 * (may be NULL) is the GUI "Customise Reactions" override, forwarded
 * verbatim to every worker's own cpr_nuclear_rates_init and to the central
 * cprimat_run -- read-only and shared across threads (no copy needed since
 * it is never mutated after cpr_mc_uncertainty's caller builds it).
 *
 * `prev_centrals`/`prev_values`/`n_prev` are the incremental-reuse
 * counterpart of Python's `mc_uncertainty(..., prev=...)` (PyPRIMAT's
 * `main.py`): pass `n_prev > 0` to reuse `n_prev` already-computed samples
 * instead of recomputing them. `prev_centrals` (length `n_quantities`,
 * parallel to `quantities`) supplies each quantity's central value (so it
 * is not recomputed); `prev_values[q]` (length `n_prev`) supplies
 * quantity q's first `n_prev` sample values, for sample indices
 * `seed .. seed+n_prev-1` -- the caller is responsible for verifying that
 * `seed`/`base_params`/`custom`/`quantities` are unchanged from the call
 * that produced these values (this function does not check; mirrors
 * Python's `mc_uncertainty` doing that check itself, but here the check is
 * pushed to the caller -- see `primat/backend.py`'s `run_mc`). Only
 * `min(n_prev, num_mc)` samples are actually reused: extra `prev_values`
 * beyond `num_mc` are ignored (truncation, nothing solved), and any
 * shortfall (`n_prev < num_mc`) is filled by solving samples
 * `seed+n_prev .. seed+num_mc-1`. Pass `n_prev=0` (with `prev_centrals`/
 * `prev_values` NULL) for an ordinary from-scratch run.
 *
 * Fills `out` (zeroed first; caller must cpr_mc_result_free). Returns 0 on
 * success, nonzero with *errmsg set (caller frees) on any config/init/
 * solve failure (the first one encountered, possibly from any worker --
 * remaining workers are still joined before returning). */
int cpr_mc_uncertainty(int num_mc, const char * const *quantities, size_t n_quantities,
                        const char *rates_dir,
                        const CPRParamSet *base_params, size_t n_base_params,
                        int seed, int n_jobs, const CPRCustomNetwork *custom,
                        const double *prev_centrals, const double * const *prev_values,
                        size_t n_prev,
                        CPRMCResult *out, char **errmsg);

void cpr_mc_result_free(CPRMCResult *out);

/* Convenience accessor: index of `name` within `out->items`, or
 * `out->n` if not found (mirrors MCResult.__getitem__'s dict lookup, as a
 * linear scan since n_quantities is always small). */
size_t cpr_mc_result_index(const CPRMCResult *out, const char *name);

#endif /* CPRIMAT_MC_H */
