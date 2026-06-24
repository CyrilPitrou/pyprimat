/* config.h -- CPRIMAT run-time configuration (port of pyprimat/config.py).
 *
 * Unlike Python's dynamically-typed PyPRConfig, CPRConfig is a single plain
 * struct with one typed field per DEFAULT_PARAMS entry: C has no convenient
 * dynamic-attribute story, and a struct is both simpler to read and faster
 * to access than threading every physics formula through a generic
 * key/value lookup. The *external* interface (ini file, CLI flags, --set)
 * still goes through a generic tagged-union CPRParam (see cpr_parse_literal
 * / cpr_config_set_by_name below) exactly as CPLAN.md S6 describes -- that
 * union is the parsing/dispatch boundary, not the storage representation.
 *
 * Optional ("None"-able) Python values are represented as:
 *   - string-typed param, no value     -> NULL char* (nevo_file, ...)
 *   - amax (int-or-None)               -> -1 sentinel (Python requires a
 *                                          positive int when set, so -1 is
 *                                          unambiguous)
 */
#ifndef CPRIMAT_CONFIG_H
#define CPRIMAT_CONFIG_H

#include <stddef.h>

/* ---- Generic tagged-union value, used only at the parsing/CLI/ini
 * boundary (cpr_parse_literal, cpr_config_set_by_name). ---- */
typedef enum { CPR_NONE, CPR_BOOL, CPR_INT, CPR_DOUBLE, CPR_STRING } CPRType;

typedef struct {
    CPRType type;
    union {
        int b;          /* CPR_BOOL: 0/1 */
        long i;         /* CPR_INT */
        double d;       /* CPR_DOUBLE */
        const char *s;  /* CPR_STRING; not owned -- caller-managed lifetime */
    } v;
} CPRParam;

/* A single named (key, value) pair -- the unit ini/cli parsing produces. */
typedef struct {
    const char *key;   /* not owned */
    CPRParam value;
} CPRParamSet;

/* Parses one literal token the same way pyprimat.cli's --set escape hatch
 * does (ast.literal_eval-equivalent): try int, then float, then
 * true/false/none (case-insensitive), else fall back to the literal string
 * (quotes, if any, are stripped). `s` must outlive the returned CPRParam
 * when the result is CPR_STRING (no copy is made). */
CPRParam cpr_parse_literal(const char *s);

/* Small open dictionary for p_<rxn> / NP_delta_<rxn>, mirroring
 * PyPRConfig.p_rxn / NP_delta_rxn. Linear-scan array: the reaction count is
 * at most ~430 (the "large" network), so a hash table buys nothing here. */
typedef struct {
    char name[40];
    double value;
} CPRRxnEntry;

typedef struct {
    CPRRxnEntry *entries;
    size_t n, cap;
} CPRRxnMap;

double cpr_rxnmap_get(const CPRRxnMap *map, const char *name); /* 0.0 default */
void cpr_rxnmap_set(CPRRxnMap *map, const char *name, double value);
void cpr_rxnmap_free(CPRRxnMap *map);

/* One nuclide row from rates/nuclear/data/nuclides.csv. */
typedef struct {
    char name[16];
    int N, Z;
    double mass_excess_keV;
    double spin;
} CPRNuclide;

typedef struct {
    CPRNuclide *items;
    size_t n;
} CPRNuclideTable;

/* ------------------------------------------------------------------------
 * CPRConfig: every DEFAULT_PARAMS entry as a typed field, grouped exactly
 * as in config.py's DEFAULT_PARAMS dict (comments there explain each flag
 * in physics terms; not repeated here -- see config.py).
 * ------------------------------------------------------------------------ */
typedef struct {
    /* ---- general behaviour and numerical settings ---- */
    int verbose;
    int debug;
    double numerical_precision;
    int numba_installed; /* unused in C (no JIT path); kept for CLI/ini parity */

    /* ---- neutrino decoupling ---- */
    int incomplete_decoupling;

    /* ---- electromagnetic plasma ---- */
    int QED_corrections;
    int n_electron_table;
    int recompute_electron_thermo;
    int recompute_qed_corrections;

    /* ---- spectral distortions ----
     * analytic_distortions / y_SZ / y_gray are NOT ported (CPLAN.md S0:
     * analytic y/gray-type distortions out of scope for v1). The fields are
     * kept here, fixed at their Python defaults (False/0/0), purely so
     * cpr_config_set_by_name() can accept and validate them by name without
     * every caller special-casing "unknown key" for a name that *does* exist
     * in DEFAULT_PARAMS -- setting any of them to a non-default value is
     * rejected by cpr_config_validate(). (There is deliberately no mu-type /
     * delta_xi_nu distortion: a neutrino chemical potential is munuOverTnu,
     * which IS ported -- it shifts the weak rates and, via
     * cpr_rho_nu_chempot_excess, the neutrino energy density / Neff.) */
    int spectral_distortions;
    int analytic_distortions;
    double y_SZ;
    double y_gray;

    /* ---- custom NEVO tables (NULL = unset / use shipped default) ---- */
    char *nevo_file;
    char *nevo_spectral_file;
    char *nevo_grid_file;
    char *nevo_file_prefix; /* never NULL; defaults to "NEVOPRIMAT" */

    /* ---- background mode ---- */
    int external_scale_factor;
    char *custom_background; /* NULL = not set */

    /* ---- fundamental constants (overridable) ---- */
    double GN;

    /* ---- background thermodynamics ---- */
    double T_start_cosmo_MeV;
    double T_end_MeV;
    int sampling_temperature_per_decade;

    /* ---- n <-> p weak rates ---- */
    int radiative_corrections;
    int finite_mass_corrections;
    int thermal_corrections;
    int weak_rate_cache;
    int save_nTOp;
    int sampling_nTOp_per_decade;
    int save_nTOp_thermal;
    int sampling_nTOp_thermal_per_decade;
    int tau_n_normalization;
    double tau_n;
    double std_tau_n;
    int vegas_n_eval;     /* unused: C side uses deterministic quadrature only (S3.3/S7b) */
    int vegas_n_itn;      /* unused, ditto */
    double epsrel_thermal;

    /* ---- output options ---- */
    int output_time_evolution;
    int output_rates_time_evolution;
    int output_n_points;
    char *output_file;
    int output_final_result;
    char *output_final_file;
    int output_background_evolution;
    char *output_background_file;

    /* ---- nuclear network ---- */
    char *rate_interp_order; /* "linear" | "quadratic" | "cubic" */
    int rate_grid_npts;
    double rate_grid_T9_min;
    double rate_grid_T9_max;
    char *network;
    int amax; /* -1 = None (no filter); else positive int */
    double atol_large_LT;
    int rescale_nuclear_rates;
    int nuclear_qed_corrections;

    /* ---- cosmological inputs ---- */
    double Omegabh2_; /* backing field; use cpr_config_set_Omegabh2() to set
                          (mirrors the Python @property that recomputes
                          eta0b on assignment) */
    double Omegach2;
    double h;
    double DeltaNeff;
    double munuOverTnu;

    /* ---- decay-era options (decay_era execution itself is out of scope,
     * CPLAN.md S0; the flags are kept so cpr_config_set_by_name() round-
     * trips every DEFAULT_PARAMS key, same rationale as analytic_distortions
     * above) ---- */
    int decay_reverse_rates;
    int decay_era;
    double t_decay_end;
    int decay_n_points;
    int output_decay_evolution;
    char *output_decay_file;

    /* ---- Early Dark Energy ---- */
    double fEDE;
    double zcEDE;
    double wnEDE;

    /* ------------------------------------------------------------------
     * Derived / non-DEFAULT_PARAMS state
     * ------------------------------------------------------------------ */
    double Omegabh2_to_eta0b;
    double eta0b;

    CPRRxnMap p_rxn;
    CPRRxnMap NP_delta_rxn;

    CPRNuclideTable nuclides;

    char data_dir[4096]; /* directory containing rates/ (mirrors PyPRConfig.data_dir) */
} CPRConfig;

/* True iff cfg->network == "small" / "large" (mirrors is_small/is_large). */
int cpr_config_is_small(const CPRConfig *cfg);
int cpr_config_is_large(const CPRConfig *cfg);

/* Derived constants depending on overridable params (mirrors the Python
 * @property of the same name). */
double cpr_config_Mpl(const CPRConfig *cfg);
double cpr_config_rhocOverh2(const CPRConfig *cfg);
double cpr_config_T_start_cosmo(const CPRConfig *cfg); /* [K] */
double cpr_config_T_end(const CPRConfig *cfg);         /* [K] */

/* Fills `cfg` with every DEFAULT_PARAMS value (string fields strdup'd so
 * the whole struct can later be freed uniformly by cpr_config_free).
 * `rates_dir` is the directory containing rates/ (passed in rather than
 * derived from argv[0], since CPRIMAT supports --rates-dir / the
 * CPRIMAT_RATES_DIR env var ahead of the executable-relative default --
 * see cli.c). Loads nuclides.csv from `rates_dir`. Returns 0 on success,
 * nonzero (with *errmsg set, caller frees) if nuclides.csv is missing or
 * malformed. */
int cpr_config_init_defaults(CPRConfig *cfg, const char *rates_dir, char **errmsg);

/* Sets cfg->Omegabh2_ and recomputes Omegabh2_to_eta0b/eta0b (the C
 * equivalent of the Python Omegabh2 property setter). */
void cpr_config_set_Omegabh2(CPRConfig *cfg, double value);
double cpr_config_get_Omegabh2(const CPRConfig *cfg);

/* Routes one (name, value) pair into the matching typed field, exactly like
 * PyPRConfig.__setattr__: a name with prefix "p_" or "NP_delta_" goes into
 * the corresponding CPRRxnMap (value coerced to double); any other name
 * must match a DEFAULT_PARAMS key (looked up via the internal field table
 * in config.c) or this returns nonzero (unknown key -- caller decides
 * whether that is a warning or an error; cli.c/ini.c warn, mirroring
 * Python's `warnings.warn`).
 *
 * Type mismatches (e.g. a string value for a double field) also return
 * nonzero with *errmsg set (caller frees); booleans accept CPR_BOOL or
 * CPR_INT (0/1, mirroring Python's duck-typed bool/int interchangeability
 * in DEFAULT_PARAMS); numeric fields accept CPR_INT for CPR_DOUBLE (widened). */
int cpr_config_set_by_name(CPRConfig *cfg, const char *name, CPRParam value,
                            char **errmsg);

/* Validates flag-combination invariants (mirrors the `raise ValueError`
 * blocks in PyPRConfig.__init__, except the ones that require modules not
 * yet ported -- see config.c's top-of-function comment for the current
 * list). Returns 0 if valid, nonzero with *errmsg set (caller frees)
 * otherwise. Call once after all overrides (ini/cli/--set) are applied. */
int cpr_config_validate(CPRConfig *cfg, char **errmsg);

/* Frees every strdup'd string field, the nuclide table, and the two
 * CPRRxnMap dictionaries. Does not free `cfg` itself. */
void cpr_config_free(CPRConfig *cfg);

#endif /* CPRIMAT_CONFIG_H */
