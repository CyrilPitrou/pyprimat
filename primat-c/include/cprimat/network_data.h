/* network_data.h -- loaders for the network/reaction list files (Phase 0),
 * plus (Phase 4, CPLAN.md S13) the physics layer that turns them into a
 * solver-ready CPRNetworkDef: load_network, NetworkDefinition::fill_buffer/
 * apply_variations, and UpdateNuclearRates, on top of network_builder.h's
 * generic stoichiometry-driven RHS/Jacobian.
 *
 * custom_network (the GUI's "Customise Reactions" per-reaction rate-table
 * override -- removed/replaced/added) *is* ported: see CPRCustomNetwork
 * below and cpr_load_network's `custom` parameter. A minimal port of
 * Python's reaction_stoichiometry tokeniser (parse_reaction_name in
 * network_data.c) derives an "added" reaction's stoichiometry from its
 * "a_b__c_d" name; it only supports the "spaced" syntax (the one
 * _RATE_SYNTAX_ and every GUI-generated name use), not the legacy
 * "compact" ("abTOcd") syntax.
 *
 * Still deliberately not ported (CPLAN.md S0/S4: gui/ and plotting.py are
 * out of scope, and these are GUI-only call paths): to_filename/
 * reaction_category/group_reactions_by_category (display/lookup helpers
 * with no effect on a run's numerics), and the files provenance list on
 * NetworkDefinition (rate-table path per reaction, used only by the GUI
 * table -- not needed for the verbose console listing). The sources list
 * (ref= labels) is now ported: CPRNetworkDef.sources, populated by
 * cpr_load_network and printed by cprimat_run's verbose print_reactions.
 * compute_detailed_balance_coefficients is ported in full (used
 * both for cfg->decay_reverse_rates and for an "added" reaction's reverse
 * rate, mirroring Python's _inject_custom_reactions).
 */
#ifndef CPRIMAT_NETWORK_DATA_H
#define CPRIMAT_NETWORK_DATA_H

#include "cprimat/config.h"
#include "cprimat/network_builder.h"

#include <stddef.h>

/* One line of a data/nuclear/networks/<name>.txt file: either
 * "<rxn_name>, <rate_table_filename>" (a tabulated reaction; `rxn_name` is
 * also the lookup key used against detailed_balance.csv/
 * reactions_large.csv and the tables/<rxn_name>/ directory, and
 * `table_file` is the specific candidate table within it -- the
 * "_primat.txt"/"_parthenope3.0.txt" naming distinguishes sibling
 * alternates, see CLAUDE.md's "Adding a new reaction" section), or a bare
 * "<rxn_name>" with no comma -- a T9-independent analytic decay/electron-
 * capture reaction looked up by name in decays.txt instead of a rate-table
 * file (`table_file[0] == '\0'` then; e.g. "B12__C12_Bm" in large.txt). */
typedef struct {
    char name[64];
    char table_file[128]; /* empty string => look up `name` in decays.txt */
} CPRNetworkEntry;

typedef struct {
    CPRNetworkEntry *entries;
    size_t n;
} CPRNetworkList;

/* Loads a data/nuclear/networks/<name>.txt file. Rejects (nonzero return,
 * *errmsg set) a network file that lists the same reaction name twice --
 * mirrors load_network's ValueError on duplicate entries (CLAUDE.md
 * "Adding a new reaction", step 5). */
int cpr_load_network_list(const char *path, CPRNetworkList *out, char **errmsg);
void cpr_network_list_free(CPRNetworkList *list);

/* One row of data/nuclear/tables/decays.txt: a T9-independent beta-decay/
 * electron-capture rate (see CLAUDE.md "Adding a new reaction", the
 * decays.txt exception). `ref` is the trailing free-text citation. */
typedef struct {
    char name[64];
    double halflife_s;
    double rate_s_inv;
    double uncertainty;
    char ref[128];
} CPRDecayEntry;

typedef struct {
    CPRDecayEntry *entries;
    size_t n;
} CPRDecayTable;

int cpr_load_decays(const char *path, CPRDecayTable *out, char **errmsg);
void cpr_decay_table_free(CPRDecayTable *t);

/* One row of data/nuclear/data/detailed_balance.csv: the alpha/beta/gamma
 * detailed-balance coefficients and Q-value [keV] used to derive each
 * reaction's reverse rate (Phys. Rep. detailed-balance formula -- ported
 * with the physics in Phase 4, not here). */
typedef struct {
    char reaction[64];
    double Q_keV, alpha, beta, gamma;
} CPRDetailedBalanceEntry;

typedef struct {
    CPRDetailedBalanceEntry *entries;
    size_t n;
} CPRDetailedBalanceTable;

int cpr_load_detailed_balance(const char *path, CPRDetailedBalanceTable *out,
                                char **errmsg);
void cpr_detailed_balance_free(CPRDetailedBalanceTable *t);

/* One row of data/nuclear/data/reactions_large.csv: reactants/products as
 * raw "+"-joined strings (e.g. "B10+He3" / "C11+H2") -- left untokenised
 * here; reaction_stoichiometry's "TO"-splitting logic is Phase 4 physics. */
typedef struct {
    char name[64];
    char reactants[64];
    char products[64];
    char source[32];
    char ref[128];
} CPRReactionEntry;

typedef struct {
    CPRReactionEntry *entries;
    size_t n;
} CPRReactionTable;

int cpr_load_reactions_large(const char *path, CPRReactionTable *out,
                               char **errmsg);
void cpr_reaction_table_free(CPRReactionTable *t);

/* ========================================================================
 * Phase 4 physics layer.
 * ========================================================================
 */

/* One GUI-uploaded/edited rate table (port of Python's custom_tables dict
 * value: a verbatim 2/3-column "T9 rate [err]" text, already parsed into
 * arrays here). `name` is looked up against the network's selected
 * reaction names: if already present in reactions_large.csv this overrides
 * that reaction's forward rate ("replaced"); otherwise the reaction is
 * brand-new ("added") and its stoichiometry is derived from `name` itself
 * (see parse_reaction_name in network_data.c). `err` may be all-zero when
 * the uploaded table had no third column (mirrors Python's
 * np.zeros_like(rate) fallback). */
typedef struct {
    char name[64];
    double *T9, *rate, *err;
    size_t n;
} CPRCustomTable;

/* The GUI "Customise Reactions" override (port of the custom_network dict
 * passed to UpdateNuclearRates.__init__): `removed` names are dropped from
 * the selected network outright; `tables` (replaced + added, merged exactly
 * as Python merges custom_network["replaced"]/["added"] into one
 * custom_tables dict) supplies forward-rate overrides/new reactions. The
 * GUI's `"filenames"` key is display-only and has no C-side counterpart. */
typedef struct {
    char (*removed)[64];
    size_t n_removed;
    CPRCustomTable *tables;
    size_t n_tables;
} CPRCustomNetwork;

/* Reverse-rate coefficients (alpha, beta, gamma) of backward(T9) =
 * alpha * T9^beta * exp(gamma/T9) * forward(T9), derived from nuclide
 * masses/spins/mass-excesses alone (port of compute_detailed_balance_coefficients,
 * a direct port of PRIMAT's GatherInfoReac/Qreaction/PowerT9/
 * FactorInverseReaction). Reproduces the catalog's tabulated values to
 * <0.5% (see network_data.py's docstring); used here only as a from-
 * scratch fallback for decay reverse rates when cfg->decay_reverse_rates
 * is set (default False -- the catalog's own detailed_balance.csv values
 * are used otherwise, see cpr_load_network).
 *
 * reactants/products: nuclide names (cfg->nuclides keys), each repeated
 * once per unit of multiplicity (e.g. d+d -> ["H2","H2"]), lengths
 * n_react/n_prod. Returns 0 on success, nonzero with *errmsg set (caller
 * frees) if a name is not found in cfg->nuclides. */
int cpr_compute_detailed_balance_coefficients(const char * const *reactants, size_t n_react,
                                                const char * const *products, size_t n_prod,
                                                const CPRConfig *cfg,
                                                double *alpha, double *beta, double *gamma,
                                                char **errmsg);

/* A fully assembled reaction network for one solver era (port of
 * NetworkDefinition). `network`/`names`/`weak_flags`/`lepton_dZ` are
 * aligned, length n_reac, with index 0 always the prepended weak n__p
 * conversion, which (mirroring Python's NetworkDefinition exactly) has no
 * row of its own in fwd/abg/bwd_cap -- those are row-major
 * ((n_reac-1) x ...), one row per *thermonuclear* reaction (names[1..]):
 * fwd* rows are length n_grid (the master T9 grid `grid`), abg rows are
 * length 3 (alpha, beta, gamma). `bwd_cap` (length n_reac-1) is the
 * reverse-rate cap from cpr_reverse_rate_cap. `buf` (length 2*n_reac)
 * is cpr_network_fill_buffer's output/cache, valid when `cache_valid`
 * (matching __post_init__'s single-slot (T_t, clamp) memo, see
 * NetworkDefinition.fill_buffer's docstring for why). */
typedef struct {
    char (*species)[16];
    long *N, *Z;
    size_t n_species;

    CPRReaction *network;     /* length n_reac, species indices into species[] */
    char (*names)[64];        /* length n_reac, names[0] == "n__p" */
    char (*sources)[64];      /* length n_reac; ref= label from rate-table header ("" for n__p, decay ref for decays) */
    int *weak_flags;          /* length n_reac, 1 iff lepton_dZ != 0 */
    long *lepton_dZ;          /* length n_reac */
    size_t n_reac;

    double *grid;             /* master T9 grid, length n_grid, ascending */
    size_t n_grid;

    double *fwd;              /* active forward rates (mutated by apply_variations) */
    double *fwd_median;
    double *fwd_expsigma;
    double *abg;
    double *bwd_cap;

    double *buf;               /* length 2*n_reac: fill_buffer's r[2i]/r[2i+1] */
    double cache_T_t;
    int cache_clamp;
    int cache_valid;
} CPRNetworkDef;

/* Builds the selected network from its text reaction list -- the master
 * entry point (port of load_network). `era` is "MT" (intersect with the
 * fixed historical ORDER_MT/full-small-list order -- always integrated
 * even for `network="large"`, since the full network is too stiff there,
 * see CLAUDE.md "Key configuration flags") or "LT" (the full selected
 * list). `reaction_names`/`n_reaction_names` mirror load_network's
 * `reaction_names` override parameter: pass NULL/0 to read
 * cfg->network's own file (data/nuclear/networks/<network>.txt, or
 * small.txt's 12 reactions for network="small" -- both are real on-disk
 * files here, unlike Python's hardcoded ORDER_SMALL, since CPRIMAT's
 * data/ tree already ships small.txt with identical content).
 *
 * `custom` (may be NULL) applies the GUI's "Customise Reactions" override
 * (see CPRCustomNetwork above): removed names are dropped from the
 * resolved bare-name list before the amax/era filters run, and added/
 * replaced names get their forward rate (and, for an added reaction, its
 * stoichiometry/reverse-rate coefficients) from `custom->tables` instead of
 * the shipped catalog/tables tree.
 *
 * Returns 0 on success (caller must cpr_network_def_free), nonzero with
 * *errmsg set (caller frees) otherwise. */
int cpr_load_network(const CPRConfig *cfg, const char *era,
                      const char * const *reaction_names, size_t n_reaction_names,
                      const CPRCustomNetwork *custom,
                      CPRNetworkDef *out, char **errmsg);
void cpr_network_def_free(CPRNetworkDef *net);

/* Updates net->fwd by applying p_<name>/delta_<name> rate variations
 * from cfg (port of NetworkDefinition.apply_variations):
 *   fwd = fwd_median * (exp(p * log(expsigma)) + delta)
 * With p=0 and delta=0 (the defaults) fwd reverts to fwd_median.
 * delta is a direct fractional additive shift (0.1 → +10%); it always
 * applies when nonzero regardless of cfg->rescale_nuclear_rates (that
 * flag is kept for backward compat but no longer gates delta).
 * Skips names[0] (n__p), handled by the separate weak-rate cache. */
void cpr_network_apply_variations(CPRNetworkDef *net, const CPRConfig *cfg);

/* Fills and returns net->buf, the forward/backward rate buffer at photon
 * temperature T_t_K (port of NetworkDefinition.fill_buffer). `nTOp_frwrd`/
 * `nTOp_bkwrd` are the n<->p weak rate already evaluated at T_t_K (e.g.
 * via cpr_weak_rate_nTOp/pTOn, weak_rates.h) -- unlike Python, which
 * passes the *callables* and evaluates them internally, the caller
 * evaluates them once and passes the scalar in; this is equivalent
 * because cpr_weak_rate_nTOp/pTOn are themselves cheap quadratic-
 * interpolation lookups (no benefit to caching the call itself), while
 * the rest of fill_buffer's work (the n_reac-row rate-table interpolation
 * below) is what __post_init__'s (T_t, clamp) memo actually exists to
 * skip on repeated same-T_t Newton-corrector calls -- preserved here via
 * net->cache_T_t/cache_clamp/cache_valid exactly as in Python.
 *
 * Forward rates are linearly interpolated in net->fwd; backward rates
 * from detailed balance (bwd = alpha*T9^beta*exp(min(gamma/T9,EXP_CAP))*fwd,
 * floored at 0, and -- when `clamp` -- capped at net->bwd_cap). Returns a
 * pointer to net->buf (length 2*net->n_reac): r[2i] forward, r[2i+1]
 * backward for reaction i; valid until the next call. */
const double *cpr_network_fill_buffer(CPRNetworkDef *net, double T_t_K,
                                        double nTOp_frwrd, double nTOp_bkwrd, int clamp);

/* Builds era networks and temperature-dependent rate buffers for both the
 * MT and LT solver eras of cfg->network (port of UpdateNuclearRates),
 * compiling each via network_builder.h and verifying N/Z conservation
 * immediately (cpr_check_conservation) -- a violation means the reaction
 * list is physically inconsistent and this fails rather than returning a
 * network that would integrate nonsense. */
typedef struct {
    CPRNetworkDef mt_net, lt_net;
    CPRCompiledNetwork mt_compiled, lt_compiled;
} CPRNuclearRates;

int cpr_nuclear_rates_init(CPRNuclearRates *nr, const CPRConfig *cfg,
                             const CPRCustomNetwork *custom, char **errmsg);
void cpr_nuclear_rates_free(CPRNuclearRates *nr);
void cpr_nuclear_rates_apply_variations(CPRNuclearRates *nr, const CPRConfig *cfg);

/* MT/LT RHS and analytic Jacobian (port of UpdateNuclearRates.rhsMT/
 * JacobianMT/rhsLT/JacobianLT). `clamp` is always 0 for MT (mirroring
 * Python's fill_buffer(..., clamp=False) there) and 1 for LT. `Y` has
 * length nr->mt_net.n_species / nr->lt_net.n_species respectively; `J`
 * (Jacobian only) is row-major (n_species x n_species). */
void cpr_nuclear_rates_rhs_mt(CPRNuclearRates *nr, const double *Y, double T_t_K,
                                double rhoBBN, double nTOp_frwrd, double nTOp_bkwrd,
                                double *dY);
void cpr_nuclear_rates_jac_mt(CPRNuclearRates *nr, const double *Y, double T_t_K,
                                double rhoBBN, double nTOp_frwrd, double nTOp_bkwrd,
                                double *J);
void cpr_nuclear_rates_rhs_lt(CPRNuclearRates *nr, const double *Y, double T_t_K,
                                double rhoBBN, double nTOp_frwrd, double nTOp_bkwrd,
                                double *dY);
void cpr_nuclear_rates_jac_lt(CPRNuclearRates *nr, const double *Y, double T_t_K,
                                double rhoBBN, double nTOp_frwrd, double nTOp_bkwrd,
                                double *J);

#endif /* CPRIMAT_NETWORK_DATA_H */
