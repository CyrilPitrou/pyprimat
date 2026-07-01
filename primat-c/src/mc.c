/* mc.c -- see mc.h. Threaded port of primat/main.py's
 * mc_uncertainty/_mc_run_batch/_mc_collect_samples.
 */
#include "mc.h"
#include "api.h"
#include "plasma.h"
#include "background.h"
#include "network_data.h"
#include "nuclear_network.h"
#include "rng.h"

#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Cooperative-cancellation flag for cpr_mc_request_cancel (mc.h): reset to
 * 0 at the start of every cpr_mc_uncertainty call, set to 1 by an external
 * thread (e.g. Python's SIGINT poll loop in _wrapper.c) to ask worker_main's
 * per-sample loop to stop early. sig_atomic_t is not require here (no
 * signal handler touches it directly) but its single-word, no-torn-write
 * guarantee is exactly what an unsynchronised cross-thread flag needs. */
static volatile sig_atomic_t g_cpr_mc_cancel = 0;

void cpr_mc_request_cancel(void)
{
    g_cpr_mc_cancel = 1;
}

/* Progress reporting: a shared counter (mutex-protected for cross-thread
 * visibility) incremented by each worker after each successfully solved sample,
 * read every 250 ms by a dedicated progress thread that prints a \r-updated
 * line to stderr.  Only active when show_progress=1 in cpr_mc_uncertainty. */
struct CPRProgressCtx {
    pthread_mutex_t mu;
    int n_done;   /* total samples completed so far (starts at n_prev_eff) */
    int total;    /* num_mc: the target count shown to the user */
    volatile int running; /* set to 0 by cpr_mc_uncertainty to stop the thread */
};

static void *progress_thread_fn(void *arg)
{
    struct CPRProgressCtx *ctx = arg;
    /* Print the initial state immediately (before the first sleep) so the
     * user sees "0/N (0%)" right away without waiting 250 ms. */
    pthread_mutex_lock(&ctx->mu);
    int last_done = ctx->n_done;
    pthread_mutex_unlock(&ctx->mu);
    int pct = ctx->total > 0 ? (int)(100L * last_done / ctx->total) : 0;
    fprintf(stderr, "\r[MC] %d/%d samples (%3d%%)", last_done, ctx->total, pct);
    fflush(stderr);

    while (ctx->running) {
        usleep(250000); /* 250 ms between updates */
        pthread_mutex_lock(&ctx->mu);
        int done = ctx->n_done;
        pthread_mutex_unlock(&ctx->mu);
        if (done != last_done) {
            pct = ctx->total > 0 ? (int)(100L * done / ctx->total) : 0;
            fprintf(stderr, "\r[MC] %d/%d samples (%3d%%)",
                    done, ctx->total, pct);
            fflush(stderr);
            last_done = done;
        }
    }
    /* Do NOT print the final "100%" line here: the caller (cpr_mc_uncertainty)
     * prints it after joining this thread, ensuring exactly one final line. */
    return NULL;
}

/* One worker's share of the work: a contiguous slice [seed_lo, seed_hi) of
 * sample seeds, writing its results directly into the shared `out->items`
 * value arrays at the matching sample index (seed - base_seed) -- safe
 * without locking since each worker's index range is disjoint. */
typedef struct {
    const char *data_dir;
    const CPRParamSet *base_params;
    size_t n_base_params;
    int seed_lo, seed_hi; /* this worker's seed range [lo, hi) */
    int base_seed;         /* out->items[q]->values index = seed - base_seed */
    const CPRCustomNetwork *custom; /* GUI override, shared read-only across workers */
    const char * const *quantities;
    size_t n_quantities;
    CPRMCResult *out;             /* shared; each worker only writes its own index range */
    struct CPRProgressCtx *prog;  /* shared progress counter; NULL if show_progress=0 */
    char *errmsg;           /* this worker's first error, if any (NULL otherwise) */
} CPRMCWorker;

/* Builds one worker's own CPRConfig + Plasma + CPRNuclearRates +
 * CPRBackground -- the part of PyPR's setup that does *not* depend on the
 * sampled p_<rxn>/tau_n values, built once per worker and reused across
 * every sample in its seed range (mirrors _mc_run_batch's docstring). */
static int worker_setup(const CPRMCWorker *w, CPRConfig *cfg, CPRPlasma *pl,
                         CPRNuclearRates *nr, CPRBackground *bg, char **errmsg)
{
    if (cpr_config_init_defaults(cfg, w->data_dir, errmsg)) return 1;
    for (size_t i = 0; i < w->n_base_params; i++) {
        char *set_err = NULL;
        if (cpr_config_set_by_name(cfg, w->base_params[i].key, w->base_params[i].value, &set_err)) {
            *errmsg = set_err;
            cpr_config_free(cfg);
            return 1;
        }
        free(set_err);
    }
    if (cpr_config_validate(cfg, errmsg)) { cpr_config_free(cfg); return 1; }
    cfg->show_progress = 0; /* suppress per-sample phase markers; only the central solve and MC counter are shown */

    if (cpr_plasma_init(pl, cfg, errmsg)) { cpr_config_free(cfg); return 1; }
    if (cpr_nuclear_rates_init(nr, cfg, w->custom, errmsg)) {
        cpr_plasma_free(pl); cpr_config_free(cfg);
        return 1;
    }
    int bg_rc = cfg->custom_background
        ? cpr_bg_init_custom(bg, cfg, pl, cfg->custom_background, errmsg)
        : cpr_bg_init_standard(bg, cfg, pl, errmsg);
    if (bg_rc) {
        cpr_nuclear_rates_free(nr); cpr_plasma_free(pl); cpr_config_free(cfg);
        return 1;
    }
    return 0;
}

/* One MC sample at the already-built (cfg, pl, nr, bg): redraw every
 * active thermonuclear rate offset p_<rxn> ~ N(0,1) plus
 * tau_n ~ N(cfg.tau_n, cfg.std_tau_n) from a single per-sample CPRRng
 * seeded with `seed` (mirrors _mc_run_batch's `default_rng(seed)`, so the
 * result for a given seed is independent of how seeds are chunked across
 * threads), re-applies the rate variations, re-solves the nuclear
 * network, and writes each requested quantity into
 * `out->items[q].values[seed - base_seed]`. Returns 0 on success, nonzero
 * with *errmsg set (caller frees) on a solve failure. */
static int run_one_sample(int seed, int base_seed, CPRConfig *cfg, CPRNuclearRates *nr,
                           CPRBackground *bg, double tau_n_central, double norm_times_tau_n,
                           const char * const *quantities, size_t n_quantities,
                           CPRMCResult *out, char **errmsg)
{
    CPRRng rng;
    cpr_rng_seed(&rng, (uint64_t)seed);

    /* names[0] is the prepended weak n__p entry (network_data.h), not a
     * thermonuclear reaction read from the network file -- excluded here
     * exactly as Python's load_reaction_names (reads the file directly,
     * never sees the weak entry) excludes it from rate_keys. */
    for (size_t i = 1; i < nr->lt_net.n_reac; i++)
        cpr_rxnmap_set(&cfg->p_rxn, nr->lt_net.names[i], cpr_rng_normal(&rng));

    /* One further standard-normal draw, after the rate offsets (so the RNG
     * stream order does not depend on the reaction count) -- mirrors
     * _mc_run_batch's tau_n perturbation. NormWeakRates = 1/tau_n when
     * tau_n_normalization is set, so scaling by tau_n_central and dividing
     * by tau_n_sample updates the normalisation without recomputing the
     * weak-rate tables themselves. */
    double tau_n_sample = tau_n_central + cfg->std_tau_n * cpr_rng_normal(&rng);
    if (cfg->tau_n_normalization) {
        cfg->tau_n = tau_n_sample;
        bg->norm_weak_rates = norm_times_tau_n / tau_n_sample;
    }

    cpr_nuclear_rates_apply_variations(nr, cfg);

    CPRNuclearNetwork nn;
    if (cpr_nuclear_network_solve(&nn, cfg, nr, bg, errmsg))
        return 1;

    CPRResults results;
    cpr_assemble_results(&results, cfg, &nn, bg);
    size_t idx = (size_t)(seed - base_seed);
    for (size_t q = 0; q < n_quantities; q++) {
        int found;
        double v = cpr_results_get_quantity(&results, quantities[q], &found);
        out->items[q].values[idx] = found ? v : NAN;
    }
    cprimat_results_free(&results);
    cpr_nuclear_network_free(&nn);
    return 0;
}

static void *worker_main(void *arg)
{
    CPRMCWorker *w = arg;
    CPRConfig cfg;
    CPRPlasma pl;
    CPRNuclearRates nr;
    CPRBackground bg;
    char *err = NULL;
    if (worker_setup(w, &cfg, &pl, &nr, &bg, &err)) {
        w->errmsg = err;
        return NULL;
    }

    double tau_n_central = cfg.tau_n;
    double norm_times_tau_n = bg.norm_weak_rates * tau_n_central;

    for (int seed = w->seed_lo; seed < w->seed_hi; seed++) {
        if (g_cpr_mc_cancel) break;
        char *serr = NULL;
        if (run_one_sample(seed, w->base_seed, &cfg, &nr, &bg, tau_n_central,
                            norm_times_tau_n, w->quantities, w->n_quantities, w->out, &serr)) {
            w->errmsg = serr;
            break;
        }
        /* Increment the shared progress counter after each successful sample
         * so the progress thread's display stays current. */
        if (w->prog) {
            pthread_mutex_lock(&w->prog->mu);
            w->prog->n_done++;
            pthread_mutex_unlock(&w->prog->mu);
        }
    }

    cpr_background_free(&bg);
    cpr_nuclear_rates_free(&nr);
    cpr_plasma_free(&pl);
    cpr_config_free(&cfg);
    return NULL;
}

int cpr_mc_uncertainty(int num_mc, const char * const *quantities, size_t n_quantities,
                        const char *data_dir,
                        const CPRParamSet *base_params, size_t n_base_params,
                        int seed, int n_jobs, const CPRCustomNetwork *custom,
                        const double *prev_centrals, const double * const *prev_values,
                        size_t n_prev,
                        int show_progress,
                        CPRMCResult *out, char **errmsg)
{
    memset(out, 0, sizeof(*out));
    g_cpr_mc_cancel = 0; /* fresh run: ignore any stale cancel request */

    /* Reuse guard: n_prev_eff samples (capped at num_mc, mirroring
     * mc_uncertainty's `min(len(prev), num_mc)`) are taken verbatim from
     * prev_values/prev_centrals instead of recomputing the central value or
     * solving those samples. */
    size_t n_prev_eff = (prev_values != NULL && n_prev > 0 && num_mc > 0)
        ? (n_prev < (size_t)num_mc ? n_prev : (size_t)num_mc)
        : 0;

    out->n = n_quantities;
    out->items = calloc(n_quantities, sizeof(CPRMCQuantity));
    for (size_t q = 0; q < n_quantities; q++) {
        snprintf(out->items[q].name, sizeof(out->items[q].name), "%s", quantities[q]);
        out->items[q].values = malloc((size_t)num_mc * sizeof(double));
    }

    if (n_prev_eff > 0) {
        /* Central values and the first n_prev_eff samples come straight
         * from the caller-supplied prev arrays -- no cprimat_run needed. */
        for (size_t q = 0; q < n_quantities; q++) {
            out->items[q].central = prev_centrals[q];
            memcpy(out->items[q].values, prev_values[q], n_prev_eff * sizeof(double));
        }
    } else {
        /* Central value (all p_<rxn>=0, tau_n=cfg.tau_n): one ordinary
         * cprimat_run, exactly mirroring mc_uncertainty's `central_inst`. */
        CPRConfig central_cfg;
        if (cpr_config_init_defaults(&central_cfg, data_dir, errmsg)) {
            cpr_mc_result_free(out);
            return 1;
        }
        for (size_t i = 0; i < n_base_params; i++) {
            char *set_err = NULL;
            if (cpr_config_set_by_name(&central_cfg, base_params[i].key, base_params[i].value, &set_err)) {
                *errmsg = set_err;
                cpr_config_free(&central_cfg);
                cpr_mc_result_free(out);
                return 1;
            }
            free(set_err);
        }
        if (cpr_config_validate(&central_cfg, errmsg)) {
            cpr_config_free(&central_cfg);
            cpr_mc_result_free(out);
            return 1;
        }
        CPRResults central_results;
        if (cprimat_run(&central_cfg, custom, &central_results, errmsg)) {
            cpr_config_free(&central_cfg);
            cpr_mc_result_free(out);
            return 1;
        }
        for (size_t q = 0; q < n_quantities; q++) {
            int found;
            out->items[q].central = cpr_results_get_quantity(&central_results, quantities[q], &found);
            if (!found) {
                *errmsg = strdup("cpr_mc_uncertainty: unknown quantity name");
                cprimat_results_free(&central_results);
                cpr_config_free(&central_cfg);
                cpr_mc_result_free(out);
                return 1;
            }
        }
        cprimat_results_free(&central_results);
        cpr_config_free(&central_cfg);
    }

    if ((size_t)num_mc <= n_prev_eff) return 0;

    /* Only the samples beyond the reused prefix need solving: seeds
     * [seed+n_prev_eff, seed+num_mc), written into out->items[q].values at
     * the same [n_prev_eff, num_mc) index range (mirrors
     * mc_uncertainty's `new_seeds = range(n_prev, num_mc)`). */
    int solve_seed_lo = seed + (int)n_prev_eff;
    int n_to_solve = num_mc - (int)n_prev_eff;

    if (n_jobs <= 0) {
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        n_jobs = (ncpu > 0) ? (int)ncpu : 1;
    }
    if (n_jobs > n_to_solve) n_jobs = n_to_solve;

    /* Optional progress thread: reads the shared counter every 250 ms and
     * prints a \r-updated "N/total (XX%)" line to stderr.  The counter starts
     * at n_prev_eff (already-reused samples) and workers increment it after
     * each successfully solved sample, so the display reflects true progress
     * even when n_prev_eff > 0. */
    struct CPRProgressCtx prog_ctx;
    pthread_t prog_thr;
    int prog_active = (show_progress && n_to_solve > 0);
    if (prog_active) {
        memset(&prog_ctx, 0, sizeof(prog_ctx));
        pthread_mutex_init(&prog_ctx.mu, NULL);
        prog_ctx.n_done  = (int)n_prev_eff;
        prog_ctx.total   = num_mc;
        prog_ctx.running = 1;
        /* Banner line: total workload at a glance.  The thread itself prints
         * the initial \r counter, so no duplicate initial line here. */
        fprintf(stderr, "[MC] Running %d sample%s...\n",
                num_mc, num_mc != 1 ? "s" : "");
        fflush(stderr);
        pthread_create(&prog_thr, NULL, progress_thread_fn, &prog_ctx);
    }

    pthread_t *threads = malloc((size_t)n_jobs * sizeof(pthread_t));
    CPRMCWorker *workers = calloc((size_t)n_jobs, sizeof(CPRMCWorker));

    /* Split [solve_seed_lo, solve_seed_lo+n_to_solve) into n_jobs contiguous
     * chunks (mirrors _mc_collect_samples' np.array_split): chunk sizes
     * differ by at most 1, and since each sample is fully determined by its
     * own seed (not by which chunk it landed in), the result is identical
     * regardless of n_jobs -- only the wall-clock parallelism changes.
     * `base_seed` stays `seed` (not `solve_seed_lo`) so run_one_sample's
     * `idx = seed - base_seed` lands at the correct absolute sample index. */
    int base = n_to_solve / n_jobs, rem = n_to_solve % n_jobs;
    int cursor = solve_seed_lo;
    for (int j = 0; j < n_jobs; j++) {
        int chunk = base + (j < rem ? 1 : 0);
        workers[j] = (CPRMCWorker){
            .data_dir = data_dir, .base_params = base_params, .n_base_params = n_base_params,
            .seed_lo = cursor, .seed_hi = cursor + chunk, .base_seed = seed, .custom = custom,
            .quantities = quantities, .n_quantities = n_quantities, .out = out,
            .prog = prog_active ? &prog_ctx : NULL, .errmsg = NULL,
        };
        cursor += chunk;
        pthread_create(&threads[j], NULL, worker_main, &workers[j]);
    }

    char *first_err = NULL;
    for (int j = 0; j < n_jobs; j++) {
        pthread_join(threads[j], NULL);
        if (workers[j].errmsg && !first_err) first_err = workers[j].errmsg;
        else free(workers[j].errmsg);
    }
    free(threads);
    free(workers);

    /* Stop and join the progress thread. On an ordinary completion, print the
     * single final "100%" line here -- after the thread exits -- so there is
     * exactly one terminating line regardless of what intermediate updates
     * the thread printed while workers were running. A cancelled run instead
     * reports how far it actually got (workers broke out of their per-sample
     * loop early, so prog_ctx.n_done < num_mc). */
    int cancelled = g_cpr_mc_cancel != 0;
    if (prog_active) {
        int n_done_at_stop = prog_ctx.n_done;
        prog_ctx.running = 0;
        pthread_join(prog_thr, NULL);
        pthread_mutex_destroy(&prog_ctx.mu);
        if (cancelled)
            fprintf(stderr, "\r[MC] cancelled after %d/%d samples\n", n_done_at_stop, num_mc);
        else
            fprintf(stderr, "\r[MC] %d/%d samples (100%%)\n", num_mc, num_mc);
        fflush(stderr);
    }

    if (cancelled) {
        free(first_err);
        cpr_mc_result_free(out);
        return CPR_MC_CANCELLED;
    }

    if (first_err) {
        *errmsg = first_err;
        cpr_mc_result_free(out);
        return 1;
    }

    /* Mean/std (population std, matching np.std's default ddof=0). */
    for (size_t q = 0; q < n_quantities; q++) {
        double sum = 0.0;
        for (int i = 0; i < num_mc; i++) sum += out->items[q].values[i];
        double mean = sum / (double)num_mc;
        double var = 0.0;
        for (int i = 0; i < num_mc; i++) {
            double d = out->items[q].values[i] - mean;
            var += d * d;
        }
        out->items[q].mean = mean;
        out->items[q].std = sqrt(var / (double)num_mc);
    }
    return 0;
}

void cpr_mc_result_free(CPRMCResult *out)
{
    for (size_t q = 0; q < out->n; q++) free(out->items[q].values);
    free(out->items);
    out->items = NULL;
    out->n = 0;
}

size_t cpr_mc_result_index(const CPRMCResult *out, const char *name)
{
    for (size_t q = 0; q < out->n; q++)
        if (strcmp(out->items[q].name, name) == 0) return q;
    return out->n;
}
