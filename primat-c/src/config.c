#include "cprimat/config.h"
#include "cprimat/constants.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ===========================================================================
 * Literal parsing (--set KEY=VALUE / ini values), mirroring the
 * ast.literal_eval-equivalent used by pyprimat/cli.py.
 * ===========================================================================
 */
CPRParam cpr_parse_literal(const char *s)
{
    CPRParam p;
    char *end;

    while (isspace((unsigned char)*s)) s++;
    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) len--;

    /* Quoted string literal: strip matching quotes, return as-is. */
    if (len >= 2 && ((s[0] == '"' && s[len - 1] == '"') ||
                      (s[0] == '\'' && s[len - 1] == '\''))) {
        static char buf[1024];
        size_t n = len - 2 < sizeof(buf) - 1 ? len - 2 : sizeof(buf) - 1;
        memcpy(buf, s + 1, n);
        buf[n] = '\0';
        p.type = CPR_STRING;
        p.v.s = buf;
        return p;
    }

    if (strncasecmp(s, "none", len) == 0 && len == 4) {
        p.type = CPR_NONE;
        return p;
    }
    if (strncasecmp(s, "true", len) == 0 && len == 4) {
        p.type = CPR_BOOL;
        p.v.b = 1;
        return p;
    }
    if (strncasecmp(s, "false", len) == 0 && len == 5) {
        p.type = CPR_BOOL;
        p.v.b = 0;
        return p;
    }

    /* Try integer (must consume the whole trimmed token). */
    {
        char tmp[256];
        size_t n = len < sizeof(tmp) - 1 ? len : sizeof(tmp) - 1;
        memcpy(tmp, s, n);
        tmp[n] = '\0';
        long iv = strtol(tmp, &end, 10);
        if (end == tmp + n && n > 0) {
            p.type = CPR_INT;
            p.v.i = iv;
            return p;
        }
        double dv = strtod(tmp, &end);
        if (end == tmp + n && n > 0) {
            p.type = CPR_DOUBLE;
            p.v.d = dv;
            return p;
        }
    }

    /* Fall back to literal (unquoted) string. */
    {
        static char buf[1024];
        size_t n = len < sizeof(buf) - 1 ? len : sizeof(buf) - 1;
        memcpy(buf, s, n);
        buf[n] = '\0';
        p.type = CPR_STRING;
        p.v.s = buf;
        return p;
    }
}

/* ===========================================================================
 * CPRRxnMap: p_<rxn> / NP_delta_<rxn> dictionary.
 * ===========================================================================
 */
double cpr_rxnmap_get(const CPRRxnMap *map, const char *name)
{
    for (size_t i = 0; i < map->n; i++)
        if (strcmp(map->entries[i].name, name) == 0)
            return map->entries[i].value;
    return 0.0;
}

void cpr_rxnmap_set(CPRRxnMap *map, const char *name, double value)
{
    for (size_t i = 0; i < map->n; i++) {
        if (strcmp(map->entries[i].name, name) == 0) {
            map->entries[i].value = value;
            return;
        }
    }
    if (map->n == map->cap) {
        map->cap = map->cap ? map->cap * 2 : 64;
        map->entries = realloc(map->entries, map->cap * sizeof(CPRRxnEntry));
    }
    strncpy(map->entries[map->n].name, name, sizeof(map->entries[map->n].name) - 1);
    map->entries[map->n].name[sizeof(map->entries[map->n].name) - 1] = '\0';
    map->entries[map->n].value = value;
    map->n++;
}

void cpr_rxnmap_free(CPRRxnMap *map)
{
    free(map->entries);
    map->entries = NULL;
    map->n = map->cap = 0;
}

/* ===========================================================================
 * nuclides.csv loader (mirrors PyPRConfig._load_nuclide_data).
 * ===========================================================================
 */
static int load_nuclides(CPRConfig *cfg, char **errmsg)
{
    char path[4200];
    snprintf(path, sizeof(path), "%s/csv/nuclides.csv", cfg->data_dir);

    FILE *f = fopen(path, "r");
    if (!f) {
        *errmsg = strdup("nuclides.csv not found (data_dir misconfigured?)");
        return 1;
    }

    char line[512];
    /* header: name,N,Z,A,Q,mass_excess_keV,spin -- locate columns by name so
     * a reordering of nuclides.csv doesn't silently break this loader. */
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        *errmsg = strdup("nuclides.csv is empty");
        return 1;
    }
    int col_name = -1, col_N = -1, col_Z = -1, col_mex = -1, col_spin = -1;
    {
        char hdr[512];
        strncpy(hdr, line, sizeof(hdr) - 1);
        hdr[sizeof(hdr) - 1] = '\0';
        int idx = 0;
        char *strtok_state = NULL;
        /* strtok_r, not strtok: mc.c's worker threads each call
         * cpr_config_init_defaults concurrently, and strtok keeps its
         * cursor in a single static buffer shared by every caller in the
         * process -- under threading that corrupts other threads'
         * in-progress parses (observed as spurious "header missing"/
         * dropped-row failures). strtok_r's state lives on this thread's
         * own stack instead. */
        for (char *tok = strtok_r(hdr, ",\r\n", &strtok_state); tok;
             tok = strtok_r(NULL, ",\r\n", &strtok_state), idx++) {
            if (strcmp(tok, "name") == 0) col_name = idx;
            else if (strcmp(tok, "N") == 0) col_N = idx;
            else if (strcmp(tok, "Z") == 0) col_Z = idx;
            else if (strcmp(tok, "mass_excess_keV") == 0) col_mex = idx;
            else if (strcmp(tok, "spin") == 0) col_spin = idx;
        }
    }
    if (col_name < 0 || col_N < 0 || col_Z < 0 || col_mex < 0 || col_spin < 0) {
        fclose(f);
        *errmsg = strdup("nuclides.csv header missing one of name,N,Z,mass_excess_keV,spin");
        return 1;
    }

    size_t cap = 64, n = 0;
    CPRNuclide *items = malloc(cap * sizeof(CPRNuclide));
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\0' || line[0] == '\n') continue;
        char row[512];
        strncpy(row, line, sizeof(row) - 1);
        row[sizeof(row) - 1] = '\0';
        char *fields[16] = {0};
        int nf = 0;
        char *strtok_state = NULL;
        for (char *tok = strtok_r(row, ",\r\n", &strtok_state); tok && nf < 16;
             tok = strtok_r(NULL, ",\r\n", &strtok_state))
            fields[nf++] = tok;
        if (nf <= col_name || nf <= col_N || nf <= col_Z || nf <= col_mex || nf <= col_spin)
            continue;

        if (n == cap) {
            cap *= 2;
            items = realloc(items, cap * sizeof(CPRNuclide));
        }
        CPRNuclide *nuc = &items[n];
        strncpy(nuc->name, fields[col_name], sizeof(nuc->name) - 1);
        nuc->name[sizeof(nuc->name) - 1] = '\0';
        nuc->N = atoi(fields[col_N]);
        nuc->Z = atoi(fields[col_Z]);
        nuc->mass_excess_keV = atof(fields[col_mex]);
        nuc->spin = atof(fields[col_spin]);
        n++;
    }
    fclose(f);

    cfg->nuclides.items = items;
    cfg->nuclides.n = n;
    return 0;
}

/* ===========================================================================
 * Defaults + field table for generic name-based dispatch.
 * ===========================================================================
 */
typedef enum { F_BOOL, F_INT, F_INT_OR_NONE, F_DOUBLE, F_STRING } FieldKind;

typedef struct {
    const char *name;
    FieldKind kind;
    size_t offset;
} FieldDesc;

#define FLD(field, kind) { #field, kind, offsetof(CPRConfig, field) }

static const FieldDesc FIELD_TABLE[] = {
    FLD(verbose, F_BOOL),
    FLD(debug, F_BOOL),
    FLD(numerical_precision, F_DOUBLE),
    FLD(numba_installed, F_BOOL),
    FLD(incomplete_decoupling, F_BOOL),
    FLD(QED_corrections, F_BOOL),
    FLD(n_electron_table, F_INT),
    FLD(recompute_electron_thermo, F_BOOL),
    FLD(recompute_qed_corrections, F_BOOL),
    FLD(spectral_distortions, F_BOOL),
    FLD(analytic_distortions, F_BOOL),
    FLD(y_SZ, F_DOUBLE),
    FLD(y_gray, F_DOUBLE),
    FLD(nevo_file, F_STRING),
    FLD(nevo_spectral_file, F_STRING),
    FLD(nevo_grid_file, F_STRING),
    FLD(nevo_file_prefix, F_STRING),
    FLD(external_scale_factor, F_BOOL),
    FLD(custom_background, F_STRING),
    FLD(GN, F_DOUBLE),
    FLD(T_start_cosmo_MeV, F_DOUBLE),
    FLD(T_end_MeV, F_DOUBLE),
    FLD(sampling_temperature_per_decade, F_INT),
    FLD(radiative_corrections, F_BOOL),
    FLD(finite_mass_corrections, F_BOOL),
    FLD(thermal_corrections, F_BOOL),
    FLD(weak_rate_cache, F_BOOL),
    FLD(save_nTOp, F_BOOL),
    FLD(sampling_nTOp_per_decade, F_INT),
    FLD(save_nTOp_thermal, F_BOOL),
    FLD(sampling_nTOp_thermal_per_decade, F_INT),
    FLD(tau_n_normalization, F_BOOL),
    FLD(tau_n, F_DOUBLE),
    FLD(std_tau_n, F_DOUBLE),
    FLD(vegas_n_eval, F_INT),
    FLD(vegas_n_itn, F_INT),
    FLD(epsrel_thermal, F_DOUBLE),
    FLD(output_time_evolution, F_BOOL),
    FLD(output_rates_time_evolution, F_BOOL),
    FLD(output_n_points, F_INT),
    FLD(output_file, F_STRING),
    FLD(output_final_result, F_BOOL),
    FLD(output_final_file, F_STRING),
    FLD(output_background_evolution, F_BOOL),
    FLD(output_background_file, F_STRING),
    FLD(output_mc_samples, F_BOOL),
    FLD(output_mc_file, F_STRING),
    FLD(rate_interp_order, F_STRING),
    FLD(rate_grid_npts, F_INT),
    FLD(rate_grid_T9_min, F_DOUBLE),
    FLD(rate_grid_T9_max, F_DOUBLE),
    FLD(network, F_STRING),
    FLD(amax, F_INT_OR_NONE),
    FLD(atol_large_LT, F_DOUBLE),
    FLD(rescale_nuclear_rates, F_BOOL),
    FLD(nuclear_qed_corrections, F_BOOL),
    FLD(rates_dir, F_STRING),
    FLD(user_rates_dir, F_STRING),
    FLD(Omegach2, F_DOUBLE),
    FLD(h, F_DOUBLE),
    FLD(DeltaNeff, F_DOUBLE),
    FLD(munuOverTnu, F_DOUBLE),
    FLD(decay_reverse_rates, F_BOOL),
    FLD(decay_era, F_BOOL),
    FLD(t_decay_end, F_DOUBLE),
    FLD(decay_n_points, F_INT),
    FLD(output_decay_evolution, F_BOOL),
    FLD(output_decay_file, F_STRING),
    FLD(fEDE, F_DOUBLE),
    FLD(zcEDE, F_DOUBLE),
    FLD(wnEDE, F_DOUBLE),
    /* Omegabh2 deliberately absent: routed to cpr_config_set_Omegabh2()
     * by cpr_config_set_by_name() below, mirroring the Python @property. */
};
#define FIELD_TABLE_N (sizeof(FIELD_TABLE) / sizeof(FIELD_TABLE[0]))

static char *cpr_strdup(const char *s) { return s ? strdup(s) : NULL; }

int cpr_config_init_defaults(CPRConfig *cfg, const char *data_dir, char **errmsg)
{
    memset(cfg, 0, sizeof(*cfg));
    cpr_constants_init();

    strncpy(cfg->data_dir, data_dir, sizeof(cfg->data_dir) - 1);

    cfg->verbose = 0;
    cfg->debug = 0;
    cfg->numerical_precision = 1.e-7;
    cfg->numba_installed = 1;

    cfg->incomplete_decoupling = 1;

    cfg->QED_corrections = 1;
    cfg->n_electron_table = 2000;
    cfg->recompute_electron_thermo = 0;
    cfg->recompute_qed_corrections = 0;

    cfg->spectral_distortions = 1;
    cfg->analytic_distortions = 0;
    cfg->y_SZ = 0.;
    cfg->y_gray = 0.;

    cfg->nevo_file = NULL;
    cfg->nevo_spectral_file = NULL;
    cfg->nevo_grid_file = NULL;
    cfg->nevo_file_prefix = cpr_strdup("NEVOPRIMAT");

    cfg->external_scale_factor = 0;
    cfg->custom_background = NULL;

    cfg->GN = 6.70883e-45;

    cfg->T_start_cosmo_MeV = 40.0;
    cfg->T_end_MeV = 1.e-3;
    cfg->sampling_temperature_per_decade = 600;

    cfg->radiative_corrections = 1;
    cfg->finite_mass_corrections = 1;
    cfg->thermal_corrections = 1;
    cfg->weak_rate_cache = 1;
    cfg->save_nTOp = 1;
    cfg->sampling_nTOp_per_decade = 80;
    cfg->save_nTOp_thermal = 1;
    cfg->sampling_nTOp_thermal_per_decade = 20;
    cfg->tau_n_normalization = 1;
    cfg->tau_n = 878.4;
    cfg->std_tau_n = 0.5;
    cfg->vegas_n_eval = 20000;
    cfg->vegas_n_itn = 20;
    cfg->epsrel_thermal = 1.e-2;

    cfg->output_time_evolution = 0;
    cfg->output_rates_time_evolution = 0;
    cfg->output_n_points = 500;
    cfg->output_file = cpr_strdup("results/output_tables.tsv");
    cfg->output_final_result = 0;
    cfg->output_final_file = cpr_strdup("results/output_final.dat");
    cfg->output_background_evolution = 0;
    cfg->output_background_file = cpr_strdup("results/output_background.tsv");
    cfg->output_mc_samples = 0;
    cfg->output_mc_file = cpr_strdup("results/output_mc_samples.tsv");

    cfg->rate_interp_order = cpr_strdup("linear");
    cfg->rate_grid_npts = 1000;
    cfg->rate_grid_T9_min = 1.0e-3;
    cfg->rate_grid_T9_max = 10.0;
    cfg->network = cpr_strdup("small");
    cfg->amax = -1; /* None */
    cfg->atol_large_LT = 1.e-26;
    cfg->rescale_nuclear_rates = 0;
    cfg->nuclear_qed_corrections = 1;
    cfg->rates_dir = NULL;
    cfg->user_rates_dir = NULL;

    cfg->Omegabh2_ = 0.022425;
    cfg->Omegach2 = 0.11933;
    cfg->h = 0.6766;
    cfg->DeltaNeff = 0.;
    cfg->munuOverTnu = 0.;

    cfg->decay_reverse_rates = 0;
    cfg->decay_era = 0;
    cfg->t_decay_end = 3.156e16;
    cfg->decay_n_points = 200;
    cfg->output_decay_evolution = 0;
    cfg->output_decay_file = cpr_strdup("results/output_decay_evolution.tsv");

    cfg->fEDE = 0.;
    cfg->zcEDE = 1.e8;
    cfg->wnEDE = 1.;

    if (load_nuclides(cfg, errmsg))
        return 1;

    /* Omegabh2_to_eta0b / eta0b depend on Omegabh2_, set just above. */
    cpr_config_set_Omegabh2(cfg, cfg->Omegabh2_);
    return 0;
}

int cpr_config_is_small(const CPRConfig *cfg) { return strcmp(cfg->network, "small") == 0; }
int cpr_config_is_large(const CPRConfig *cfg) { return strcmp(cfg->network, "large") == 0; }

double cpr_config_Mpl(const CPRConfig *cfg) { return 1. / sqrt(cfg->GN); }

double cpr_config_rhocOverh2(const CPRConfig *cfg)
{
    double H = cpr_HubbleOverh();
    return 3. / (8. * M_PI * cfg->GN) * H * H;
}

double cpr_config_T_start_cosmo(const CPRConfig *cfg)
{
    return cfg->T_start_cosmo_MeV * cpr_MeV_to_Kelvin();
}

double cpr_config_T_end(const CPRConfig *cfg)
{
    return cfg->T_end_MeV * cpr_MeV_to_Kelvin();
}

static int path_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

void cpr_config_resolve_rates_path(const CPRConfig *cfg, const char *relpath,
                                    char *out, size_t outsize)
{
    char candidate[4200];

    if (cfg->rates_dir) {
        snprintf(candidate, sizeof(candidate), "%s/%s", cfg->rates_dir, relpath);
        if (path_exists(candidate)) {
            snprintf(out, outsize, "%s", candidate);
            return;
        }
    }
    if (cfg->user_rates_dir) {
        snprintf(candidate, sizeof(candidate), "%s/%s", cfg->user_rates_dir, relpath);
        if (path_exists(candidate)) {
            snprintf(out, outsize, "%s", candidate);
            return;
        }
    }
    /* Shipped default, always tried last (and returned even if missing, so
     * the caller's "file not found" error points at the expected default
     * location -- mirrors PyPRConfig.resolve_rates_path). cfg->data_dir is
     * the data folder itself (e.g. .../primat/data), not its parent. */
    snprintf(out, outsize, "%s/%s", cfg->data_dir, relpath);
}

void cpr_config_set_Omegabh2(CPRConfig *cfg, double value)
{
    cfg->Omegabh2_ = value;
    /* Omegabh2_to_eta0b = (rhocOverh2 / n0CMB) / (ma / maOvermB); eta0b =
     * Omegabh2_to_eta0b * Omegabh2 (Phys. Rep. baryon-to-photon ratio). */
    cfg->Omegabh2_to_eta0b = (cpr_config_rhocOverh2(cfg) / cpr_n0CMB())
                             / (g_const.ma / cpr_maOvermB());
    cfg->eta0b = cfg->Omegabh2_to_eta0b * cfg->Omegabh2_;
}

double cpr_config_get_Omegabh2(const CPRConfig *cfg) { return cfg->Omegabh2_; }

int cpr_config_set_by_name(CPRConfig *cfg, const char *name, CPRParam value,
                            char **errmsg)
{
    if (strncmp(name, "p_", 2) == 0) {
        double d = value.type == CPR_DOUBLE ? value.v.d
                 : value.type == CPR_INT ? (double)value.v.i
                 : value.type == CPR_BOOL ? (double)value.v.b : 0.0;
        cpr_rxnmap_set(&cfg->p_rxn, name + 2, d);
        return 0;
    }
    if (strncmp(name, "NP_delta_", 9) == 0) {
        double d = value.type == CPR_DOUBLE ? value.v.d
                 : value.type == CPR_INT ? (double)value.v.i
                 : value.type == CPR_BOOL ? (double)value.v.b : 0.0;
        cpr_rxnmap_set(&cfg->NP_delta_rxn, name + 9, d);
        return 0;
    }
    if (strcmp(name, "Omegabh2") == 0) {
        double d = value.type == CPR_DOUBLE ? value.v.d
                 : value.type == CPR_INT ? (double)value.v.i : NAN;
        if (isnan(d)) {
            *errmsg = strdup("Omegabh2 requires a numeric value");
            return 1;
        }
        cpr_config_set_Omegabh2(cfg, d);
        return 0;
    }

    for (size_t i = 0; i < FIELD_TABLE_N; i++) {
        if (strcmp(FIELD_TABLE[i].name, name) != 0)
            continue;
        void *field = (char *)cfg + FIELD_TABLE[i].offset;
        switch (FIELD_TABLE[i].kind) {
        case F_BOOL:
            if (value.type != CPR_BOOL && value.type != CPR_INT) {
                *errmsg = malloc(128);
                snprintf(*errmsg, 128, "%s expects a bool", name);
                return 1;
            }
            *(int *)field = value.type == CPR_BOOL ? value.v.b : (int)value.v.i;
            return 0;
        case F_INT:
            if (value.type != CPR_INT && value.type != CPR_BOOL) {
                *errmsg = malloc(128);
                snprintf(*errmsg, 128, "%s expects an int", name);
                return 1;
            }
            *(int *)field = value.type == CPR_INT ? (int)value.v.i : value.v.b;
            return 0;
        case F_INT_OR_NONE:
            if (value.type == CPR_NONE) {
                *(int *)field = -1;
                return 0;
            }
            if (value.type != CPR_INT) {
                *errmsg = malloc(128);
                snprintf(*errmsg, 128, "%s expects an int or None", name);
                return 1;
            }
            *(int *)field = (int)value.v.i;
            return 0;
        case F_DOUBLE:
            if (value.type == CPR_DOUBLE) *(double *)field = value.v.d;
            else if (value.type == CPR_INT) *(double *)field = (double)value.v.i;
            else {
                *errmsg = malloc(128);
                snprintf(*errmsg, 128, "%s expects a number", name);
                return 1;
            }
            return 0;
        case F_STRING:
            free(*(char **)field);
            if (value.type == CPR_NONE) {
                *(char **)field = NULL;
            } else if (value.type == CPR_STRING) {
                *(char **)field = strdup(value.v.s);
            } else {
                *errmsg = malloc(128);
                snprintf(*errmsg, 128, "%s expects a string or None", name);
                return 1;
            }
            return 0;
        }
    }

    *errmsg = malloc(256);
    snprintf(*errmsg, 256, "unknown parameter key: %s", name);
    return 1;
}

int cpr_config_validate(CPRConfig *cfg, char **errmsg)
{
    /* custom_background: force instantaneous decoupling / no spectral
     * distortions, mirroring PyPRConfig.__init__ (warnings.warn there
     * becomes a silent forcing here; CPRIMAT's CLI layer can print a note
     * if it cares -- the physics invariant is what matters at this layer). */
    if (cfg->custom_background != NULL) {
        if (cfg->external_scale_factor) {
            *errmsg = strdup("custom_background and external_scale_factor are mutually exclusive");
            return 1;
        }
        cfg->incomplete_decoupling = 0;
        cfg->spectral_distortions = 0;
    }

    /* NOTE: the network-file-existence check (PyPRConfig.__init__'s
     * `network must be 'small' or name an existing file...`) and the
     * p_<rxn>/NP_delta_<rxn> typo check against the configured network's
     * reaction list both require network_data.c (Phase 4, not yet ported)
     * to enumerate valid reaction names -- deferred to
     * cpr_network_validate() once that module exists, called from
     * cprimat_run() after this function. */

    if (cfg->amax != -1 && cfg->amax < 1) {
        *errmsg = strdup("amax must be None (-1) or a positive integer");
        return 1;
    }

    if (cfg->external_scale_factor && !cfg->incomplete_decoupling) {
        *errmsg = strdup("external_scale_factor=True requires incomplete_decoupling=True");
        return 1;
    }

    /* Validate spectral-distortion flag combination (mirrors
     * PRIMATConfig.__init__'s equivalent block in config.py). */
    if (cfg->spectral_distortions) {
        if (cfg->analytic_distortions) {
            if (cfg->incomplete_decoupling) {
                *errmsg = strdup(
                    "spectral_distortions=True with analytic_distortions=True "
                    "requires instantaneous decoupling (incomplete_decoupling=False).");
                return 1;
            }
        } else {
            if (!cfg->incomplete_decoupling) {
                *errmsg = strdup(
                    "spectral_distortions=True with analytic_distortions=False "
                    "requires incomplete_decoupling=True (the full NEVO spectrum "
                    "file is only available in the non-instantaneous decoupling mode).");
                return 1;
            }
        }
    }

    /* NEVO override existence/shape checks: deferred to neutrino_history.c
     * (Phase 3a), which owns resolve_nevo_path() and the CSV column counts. */

    return 0;
}

void cpr_config_free(CPRConfig *cfg)
{
    for (size_t i = 0; i < FIELD_TABLE_N; i++) {
        if (FIELD_TABLE[i].kind == F_STRING) {
            void *field = (char *)cfg + FIELD_TABLE[i].offset;
            free(*(char **)field);
            *(char **)field = NULL;
        }
    }
    cpr_rxnmap_free(&cfg->p_rxn);
    cpr_rxnmap_free(&cfg->NP_delta_rxn);
    free(cfg->nuclides.items);
    cfg->nuclides.items = NULL;
    cfg->nuclides.n = 0;
}
