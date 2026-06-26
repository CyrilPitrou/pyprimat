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
#include <sys/stat.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

/* Matches "nTOp_*.txt" (the weak-rate cache naming convention; thermal
 * caches are "nTOp_thermal_*.txt" and match the same glob, exactly as
 * pyprimat.cache_utils.list_weak_cache_files globs both with one pattern). */
static int is_weak_cache_name(const char *name)
{
    return strncmp(name, "nTOp_", 5) == 0
        && strlen(name) > 4 && strcmp(name + strlen(name) - 4, ".txt") == 0;
}

static void print_credits(void)
{
    fputs("primat is developed by Cyril Pitrou (https://www2.iap.fr/users/pitrou/) "
          "with features related to neutrino physics written by Julien Froustey.\n\n",
          stdout);
    fputs("The story started in the 1980s with BBN codes written by Elisabeth "
          "Vangioni and Alain Coc which eventually lead to 'ezbbn', a large "
          "nuclear network FORTRAN code whose nuclear rates tables were maintained "
          "by Alain Coc. PRIMAT, initially a Mathematica code, was based on "
          "'ezbbn' with improved neutrino physics. It is now translated into a "
          "python code, but it also relies on a C backend to improve its "
          "performance.\n\n",
          stdout);
    fputs("For notebooks, examples and documentation, download the source code "
          "(https://github.com/CyrilPitrou/primat).\n\n",
          stdout);
    fputs("Please cite the publication (https://arxiv.org/abs/1801.08023) if "
          "you use it.\n",
          stdout);
}

static int list_or_clear_weak_cache(const char *data_dir, int clear)
{
    char dir_path[4096];
    snprintf(dir_path, sizeof(dir_path), "%s/weak", data_dir);

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
           "          [--cache-info] [--cache-clear] [--credits] [--ini PATH]\n"
           "          [--data-dir PATH] [--custom-nuclear-dir PATH]\n"
           "          [--set KEY=VALUE ...]\n", prog);
}

static int path_is_dir(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && (st.st_mode & S_IFDIR);
}

/* Best-effort absolute path to the running executable's own directory, so
 * the default data dir can be anchored to where `cprimat` itself lives
 * rather than to the caller's CWD (the old ".." default silently broke
 * whenever invoked from anywhere other than primat-c/, e.g. from
 * primat-c/build/ or the repo root -- see FOLDER.md). Returns 0 and fills
 * `out` on success, nonzero if the platform call fails (caller falls back
 * to a CWD-relative guess). */
static int executable_dir(char *out, size_t outsize)
{
    char exe_path[4096];
#if defined(__APPLE__)
    uint32_t sz = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &sz) != 0) return 1;
#elif defined(__linux__)
    ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (n <= 0) return 1;
    exe_path[n] = '\0';
#else
    return 1;
#endif
    char *slash = strrchr(exe_path, '/');
    if (!slash) return 1;
    *slash = '\0';
    snprintf(out, outsize, "%s", exe_path);
    return 0;
}

/* Resolves the default data dir: CPRIMAT_DATA_DIR env var wins outright;
 * otherwise try "<exe_dir>/../primat/data" (works for `cprimat` run from
 * primat-c/, primat-c/build/, or any installed location with that sibling
 * layout); otherwise fall back to the legacy CWD-relative "../primat/data"
 * guess (works only when invoked with CWD == primat-c/). Does not require
 * the resolved directory to exist -- cpr_config_init_defaults reports a
 * clear "nuclides.csv not found" error downstream if it's wrong, and the
 * user can always override with --data-dir. */
static const char *default_data_dir(char *buf, size_t bufsize)
{
    const char *env = getenv("CPRIMAT_DATA_DIR");
    if (env) return env;

    char exe_dir[4096];
    if (executable_dir(exe_dir, sizeof(exe_dir)) == 0) {
        /* The binary normally lives in primat-c/build/cprimat, so the
         * sibling primat/ package is two levels up; also try one level up
         * in case cprimat was copied/symlinked directly into primat-c/. */
        snprintf(buf, bufsize, "%s/../../primat/data", exe_dir);
        if (path_is_dir(buf)) return buf;
        snprintf(buf, bufsize, "%s/../primat/data", exe_dir);
        if (path_is_dir(buf)) return buf;
    }
    snprintf(buf, bufsize, "../primat/data");
    return buf;
}

int cpr_cli_main(int argc, char **argv)
{
    char data_dir_buf[4096];
    const char *data_dir = default_data_dir(data_dir_buf, sizeof(data_dir_buf));
    const char *custom_nuclear_dir = NULL;
    const char *ini_path = NULL;
    int cache_info = 0, cache_clear = 0, credits = 0;

    /* --data-dir, --custom-nuclear-dir and --ini must be known before
     * cpr_config_init_defaults runs (the first picks the data directory;
     * the others are applied after defaults), so scan for them first;
     * everything else is applied in a second pass, in the same precedence
     * order as cli.py: defaults, then .ini, then named flags, then --set
     * (later wins). */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--custom-nuclear-dir") == 0 && i + 1 < argc) {
            custom_nuclear_dir = argv[++i];
        } else if (strcmp(argv[i], "--ini") == 0 && i + 1 < argc) {
            ini_path = argv[++i];
        } else if (strcmp(argv[i], "--cache-info") == 0) {
            cache_info = 1;
        } else if (strcmp(argv[i], "--cache-clear") == 0) {
            cache_clear = 1;
        } else if (strcmp(argv[i], "--credits") == 0) {
            credits = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

    if (credits) {
        print_credits();
        return 0;
    }

    CPRConfig cfg;
    char *err = NULL;
    if (cpr_config_init_defaults(&cfg, data_dir, &err)) {
        fprintf(stderr, "error: %s\n", err);
        free(err);
        return 1;
    }

    if (custom_nuclear_dir) {
        if (!path_is_dir(custom_nuclear_dir)) {
            fprintf(stderr, "--custom-nuclear-dir: '%s' is not a directory\n", custom_nuclear_dir);
            cpr_config_free(&cfg);
            return 2;
        }
        free(cfg.user_rates_dir);
        cfg.user_rates_dir = strdup(custom_nuclear_dir);
    }

    if (cache_info || cache_clear) {
        int n = list_or_clear_weak_cache(cfg.data_dir, cache_clear);
        if (cache_clear)
            printf("Removed %d cached weak-rate file(s) from %s/weak/.\n", n, cfg.data_dir);
        else
            printf("%d cached weak-rate file(s) in %s/weak/.\n", n, cfg.data_dir);
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

        if (strcmp(a, "--data-dir") == 0 || strcmp(a, "--custom-nuclear-dir") == 0
            || strcmp(a, "--ini") == 0
            || strcmp(a, "--cache-info") == 0 || strcmp(a, "--cache-clear") == 0
            || strcmp(a, "--credits") == 0
            || strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            if (strcmp(a, "--data-dir") == 0 || strcmp(a, "--custom-nuclear-dir") == 0
                || strcmp(a, "--ini") == 0) i++;
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
