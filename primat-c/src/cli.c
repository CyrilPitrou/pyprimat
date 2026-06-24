/* cli.c -- see cprimat/cli.h. */
#include "cprimat/cli.h"
#include "cprimat/api.h"
#include "cprimat/cache.h"
#include "cprimat/config.h"
#include "cprimat/ini.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Matches "nTOp_*.txt" (the weak-rate cache naming convention; thermal
 * caches are "nTOp_thermal_*.txt" and match the same glob, exactly as
 * pyprimat.cache_utils.list_weak_cache_files globs both with one pattern). */
static int is_weak_cache_name(const char *name)
{
    return strncmp(name, "nTOp_", 5) == 0
        && strlen(name) > 4 && strcmp(name + strlen(name) - 4, ".txt") == 0;
}

static int list_or_clear_weak_cache(const char *data_dir, int clear)
{
    char dir_path[4096];
    snprintf(dir_path, sizeof(dir_path), "%s/rates/weak", data_dir);

    DIR *d = opendir(dir_path);
    if (!d) {
        fprintf(stderr, "cannot open cache directory '%s'\n", dir_path);
        return 0;
    }
    int n = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!is_weak_cache_name(ent->d_name))
            continue;
        n++;
        if (clear) {
            char file_path[4352];
            snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, ent->d_name);
            remove(file_path);
        }
    }
    closedir(d);
    return n;
}

static void usage(const char *prog)
{
    printf("usage: %s [--Omegabh2 VALUE] [--DeltaNeff VALUE] [--network NAME]\n"
           "          [--amax A] [--numerical_precision RTOL] [--verbose]\n"
           "          [--cache-info] [--cache-clear] [--ini PATH]\n"
           "          [--rates-dir PATH] [--set KEY=VALUE ...]\n", prog);
}

int cpr_cli_main(int argc, char **argv)
{
    const char *rates_dir = getenv("CPRIMAT_RATES_DIR");
    if (!rates_dir) rates_dir = "..";
    const char *ini_path = NULL;
    int cache_info = 0, cache_clear = 0;

    /* --rates-dir and --ini must be known before cpr_config_init_defaults
     * runs (the former picks the data directory; the latter is applied
     * after defaults), so scan for them first; everything else is applied
     * in a second pass, in the same precedence order as cli.py: defaults,
     * then .ini, then named flags, then --set (later wins). */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rates-dir") == 0 && i + 1 < argc) {
            rates_dir = argv[++i];
        } else if (strcmp(argv[i], "--ini") == 0 && i + 1 < argc) {
            ini_path = argv[++i];
        } else if (strcmp(argv[i], "--cache-info") == 0) {
            cache_info = 1;
        } else if (strcmp(argv[i], "--cache-clear") == 0) {
            cache_clear = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

    CPRConfig cfg;
    char *err = NULL;
    if (cpr_config_init_defaults(&cfg, rates_dir, &err)) {
        fprintf(stderr, "error: %s\n", err);
        free(err);
        return 1;
    }

    if (cache_info || cache_clear) {
        int n = list_or_clear_weak_cache(cfg.data_dir, cache_clear);
        if (cache_clear)
            printf("Removed %d cached weak-rate file(s) from %s/rates/weak/.\n", n, cfg.data_dir);
        else
            printf("%d cached weak-rate file(s) in %s/rates/weak/.\n", n, cfg.data_dir);
        cpr_config_free(&cfg);
        return 0;
    }

    if (ini_path) {
        if (cpr_ini_load(&cfg, ini_path, &err)) {
            fprintf(stderr, "error: %s\n", err);
            free(err);
            cpr_config_free(&cfg);
            return 1;
        }
    }

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        CPRParam p;
        const char *key = NULL;
        int has_val = (i + 1 < argc);

        if (strcmp(a, "--rates-dir") == 0 || strcmp(a, "--ini") == 0
            || strcmp(a, "--cache-info") == 0 || strcmp(a, "--cache-clear") == 0
            || strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            if (strcmp(a, "--rates-dir") == 0 || strcmp(a, "--ini") == 0) i++;
            continue;
        } else if (strcmp(a, "--Omegabh2") == 0 && has_val) {
            cpr_config_set_Omegabh2(&cfg, atof(argv[++i]));
            continue;
        } else if (strcmp(a, "--DeltaNeff") == 0 && has_val) {
            key = "DeltaNeff"; p = cpr_parse_literal(argv[++i]);
        } else if (strcmp(a, "--network") == 0 && has_val) {
            key = "network"; p = (CPRParam){CPR_STRING, .v.s = argv[++i]};
        } else if (strcmp(a, "--amax") == 0 && has_val) {
            key = "amax"; p = cpr_parse_literal(argv[++i]);
        } else if (strcmp(a, "--numerical_precision") == 0 && has_val) {
            key = "numerical_precision"; p = cpr_parse_literal(argv[++i]);
        } else if (strcmp(a, "--verbose") == 0) {
            key = "verbose"; p = (CPRParam){CPR_BOOL, .v.b = 1};
        } else if (strcmp(a, "--json") == 0) {
            continue; /* no solver/JSON output yet in Phase 0 */
        } else if (strcmp(a, "--set") == 0 && has_val) {
            const char *entry = argv[++i];
            const char *eq = strchr(entry, '=');
            if (!eq) {
                fprintf(stderr, "--set %s: expected KEY=VALUE\n", entry);
                cpr_config_free(&cfg);
                return 2;
            }
            char name[256];
            size_t klen = (size_t)(eq - entry);
            if (klen >= sizeof(name)) klen = sizeof(name) - 1;
            memcpy(name, entry, klen);
            name[klen] = '\0';
            key = name;
            p = cpr_parse_literal(eq + 1);
            char *set_err = NULL;
            if (cpr_config_set_by_name(&cfg, name, p, &set_err)) {
                fprintf(stderr, "--set %s: %s\n", entry, set_err ? set_err : "error");
                free(set_err);
            }
            continue;
        } else {
            fprintf(stderr, "unrecognized argument: %s\n", a);
            usage(argv[0]);
            cpr_config_free(&cfg);
            return 2;
        }

        char *set_err = NULL;
        if (cpr_config_set_by_name(&cfg, key, p, &set_err)) {
            fprintf(stderr, "--%s: %s\n", key, set_err ? set_err : "error");
            free(set_err);
        }
    }

    if (cpr_config_validate(&cfg, &err)) {
        fprintf(stderr, "error: %s\n", err);
        free(err);
        cpr_config_free(&cfg);
        return 1;
    }

    CPRResults results;
    if (cprimat_run(&cfg, NULL, &results, &err)) {
        fprintf(stderr, "error: %s\n", err);
        free(err);
        cpr_config_free(&cfg);
        return 1;
    }

    /* Mirrors cli.py's main() plain-text report (the --json dump there has
     * no C analogue yet; CPRResults is a fixed struct, not a generic
     * key/value set, so there is nothing to introspect into JSON). */
    if (results.has_Neff)
        printf("Neff       = %.8f\n", results.Neff);
    printf("YP (BBN)   = %.8f\n", results.YPBBN);
    printf("YP (CMB)   = %.8f\n", results.YPCMB);
    printf("D/H        = %.7e\n", results.DoH);
    printf("He3/H      = %.7e\n", results.He3oH);
    printf("He3/He4    = %.7e\n", results.He3oHe4);
    printf("Li7/H      = %.6e\n", results.Li7oH);
    if (results.has_Li6oLi7)
        printf("Li6/Li7    = %.6e\n", results.Li6oLi7);
    if (results.has_YCNO)
        printf("CNO (mass) = %.6e\n", results.YCNO);

    cprimat_results_free(&results);
    cpr_config_free(&cfg);
    return 0;
}
