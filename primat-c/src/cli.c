/* cli.c -- see cli.h. */
#include "cli.h"
#include "api.h"
#include "cache.h"
#include "config.h"
#include "ini.h"
#include "mc.h"

#include <dirent.h>
#include <math.h>
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
 * primat.cache_utils.list_weak_cache_files globs both with one pattern). */
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

/* Boolean PRIMATConfig flags that accept --flag / --no-flag pairs on the CLI,
 * mirroring Python cli.py's BooleanOptionalAction loop. */
static const char * const bool_flags[] = {
    "QED_corrections",
    "nuclear_qed_corrections",
    "radiative_corrections",
    "finite_mass_corrections",
    "thermal_corrections",
    "spectral_distortions",
    "output_time_evolution",
    "output_final_result",
    "output_background_evolution",
    "output_mc_samples",
    NULL
};

static void usage(const char *prog)
{
    printf("usage: %s [-h] [--credits] [--version]\n"
           "          [--Omegabh2 VALUE] [--DeltaNeff VALUE] [--network NAME]\n"
           "          [--amax A] [--numerical_precision RTOL] [--munuOverTnu XI]\n"
           "          [--output_file FILE] [--output_final_file FILE]\n"
           "          [--output_background_file FILE] [--output_mc_file FILE]\n"
           "          [--QED_corrections | --no-QED_corrections]\n"
           "          [--nuclear_qed_corrections | --no-nuclear_qed_corrections]\n"
           "          [--radiative_corrections | --no-radiative_corrections]\n"
           "          [--finite_mass_corrections | --no-finite_mass_corrections]\n"
           "          [--thermal_corrections | --no-thermal_corrections]\n"
           "          [--spectral_distortions | --no-spectral_distortions]\n"
           "          [--output_time_evolution | --no-output_time_evolution]\n"
           "          [--output_final_result | --no-output_final_result]\n"
           "          [--output_background_evolution | --no-output_background_evolution]\n"
           "          [--output_mc_samples | --no-output_mc_samples]\n"
           "          [--mc N] [--mc-seed SEED]\n"
           "          [--json] [--verbose] [--cache-info] [--cache-clear]\n"
           "          [--ini PATH] [--data_dir PATH] [--user_nuclear_dir PATH]\n"
           "          [--set KEY=VALUE ...]\n\n"
           "Run a Big Bang Nucleosynthesis computation with primat-c and print the\n"
           "resulting Neff/abundances.\n\n"
           "options:\n"
           "  -h, --help            Show this help message and exit.\n"
           "  --credits             Print the project credits and exit.\n"
           "  --version             Print the primat-c version and exit.\n"
           "  --Omegabh2 VALUE      Baryon density Omega_b h^2 (default: 0.022425).\n"
           "  --DeltaNeff VALUE     Extra relativistic degrees of freedom on top of\n"
           "                        the SM neutrino sector (default: 0).\n"
           "  --network NAME        Nuclear reaction network used in the LT era\n"
           "                        (default: small). Built-in choices are 'small',\n"
           "                        'small_parthenope' and 'large', but any name for\n"
           "                        which data/nuclear/networks/<NAME>.txt exists is\n"
           "                        accepted.\n"
           "  --amax A              Drop reactions involving any nuclide with mass\n"
           "                        number > A (positive integer); applies to any\n"
           "                        --network. E.g. --network large --amax 8\n"
           "                        reproduces the old 'medium' network's 68 reactions.\n"
           "  --numerical_precision RTOL\n"
           "                        Relative tolerance passed to the ODE solver\n"
           "                        (default: 1e-7).\n"
           "  --munuOverTnu XI      Reduced neutrino chemical potential mu/T, same\n"
           "                        for all flavours (default: 0).\n"
           "  --output_file FILE    Write the full time-evolution TSV to FILE when\n"
           "                        --output_time_evolution is enabled.\n"
           "  --output_final_file FILE\n"
           "                        Write the final-abundance table to FILE when\n"
           "                        --output_final_result is enabled.\n"
           "  --output_background_file FILE\n"
           "                        Write the background time-evolution TSV to FILE\n"
           "                        when --output_background_evolution is enabled.\n"
           "  --output_mc_file FILE\n"
           "                        Write Monte-Carlo samples to FILE when --mc is\n"
           "                        used and --output_mc_samples is enabled.\n"
           "  --QED_corrections, --no-QED_corrections\n"
           "                        QED interaction corrections to the EM plasma\n"
           "                        equation of state. (default: True).\n"
           "  --nuclear_qed_corrections, --no-nuclear_qed_corrections\n"
           "                        QED corrections to radiative-capture nuclear\n"
           "                        reaction rates (Pitrou & Pospelov 2020).\n"
           "                        (default: True).\n"
           "  --radiative_corrections, --no-radiative_corrections\n"
           "                        Coulomb + T=0 resummed radiative corrections to\n"
           "                        n<->p (CCR); if False, use Born approximation.\n"
           "                        (default: True).\n"
           "  --finite_mass_corrections, --no-finite_mass_corrections\n"
           "                        Finite-nucleon-mass (Fokker-Planck) correction\n"
           "                        to n<->p. (default: True).\n"
           "  --thermal_corrections, --no-thermal_corrections\n"
           "                        Finite-temperature radiative corrections to\n"
           "                        n<->p (CCRTh; Brown & Sawyer 2001). (default: True).\n"
           "  --spectral_distortions, --no-spectral_distortions\n"
           "                        Correct n<->p rates for non-Fermi-Dirac neutrino\n"
           "                        distributions. (default: True).\n"
           "  --output_time_evolution, --no-output_time_evolution\n"
           "                        Write the full time-evolution series (in-memory\n"
           "                        always; to disk if output_file is set).\n"
           "                        (default: False).\n"
           "  --output_final_result, --no-output_final_result\n"
           "                        Write the final results dict to output_final_file.\n"
           "                        (default: False).\n"
           "  --output_background_evolution, --no-output_background_evolution\n"
           "                        Write the cosmological background time series to\n"
           "                        disk. (default: False).\n"
           "  --output_mc_samples, --no-output_mc_samples\n"
           "                        Write --mc samples to output_mc_file.\n"
           "                        (default: False).\n"
           "  --mc N                Run an N-sample Monte-Carlo nuclear-rate/tau_n\n"
           "                        uncertainty propagation and print each observable\n"
           "                        as 'value +/- sigma'. Uses all available CPU cores.\n"
           "  --mc-seed SEED        Base RNG seed for --mc (default: 0); sample i\n"
           "                        uses seed+i.\n"
           "  --json                Print the full results dict as JSON instead of a\n"
           "                        short summary.\n"
           "  --verbose             Enable internal progress messages (timings,\n"
           "                        cache hits, ...).\n"
           "  --cache-info          Print the number of cached n<->p weak-rate files\n"
           "                        and exit, without running a solve.\n"
           "  --cache-clear         Delete every cached n<->p weak-rate file and exit,\n"
           "                        without running a solve. The cache is always\n"
           "                        safely regenerable.\n"
           "  --ini PATH            Load parameters from an INI file (applied after\n"
           "                        defaults, before named flags and --set).\n"
           "  --data_dir PATH       Replace the entire data tree (NEVO/, weak/,\n"
           "                        plasma/, nuclear/, csv/) with PATH.\n"
           "                        Default: auto-detected from the executable location\n"
           "                        or CPRIMAT_DATA_DIR environment variable.\n"
           "  --user_nuclear_dir PATH\n"
           "                        Additive overlay for nuclear networks and rate\n"
           "                        tables only (primat/data/nuclear/ equivalent).\n"
           "                        Checked before the default tree; shipped networks\n"
           "                        remain accessible even when this is set.\n"
           "  --set KEY=VALUE       Set any CPRConfig parameter (including\n"
           "                        p_<reaction>/delta_<reaction> rate variations),\n"
           "                        e.g. --set T_end_MeV=1e-4. Repeatable; later\n"
           "                        values win.\n",
           prog);
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
 * primat-c/build/ or the repo root). Returns 0 and fills
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
 * user can always override with --data_dir. */
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

/* ---- Collected CLI overrides, built during the second parse pass and
 * forwarded verbatim to cpr_mc_uncertainty as base_params so MC workers
 * start from exactly the same configuration the main run used. ---- */
#define MAX_CLI_PARAMS 256

typedef struct {
    CPRParamSet items[MAX_CLI_PARAMS];
    size_t n;
    /* Storage for key strings built from argv that don't point to argv
     * directly (only --set currently needs this via the name[] buffer). */
    char key_store[MAX_CLI_PARAMS][256];
} CLIParams;

static void cli_params_add(CLIParams *cp, const char *key, CPRParam val)
{
    if (cp->n >= MAX_CLI_PARAMS) return;
    cp->items[cp->n].key   = key;
    cp->items[cp->n].value = val;
    cp->n++;
}

/* Apply param to cfg and record it in cp for MC reuse. */
static void apply_param(CPRConfig *cfg, CLIParams *cp,
                        const char *key, CPRParam val, const char *flag_label)
{
    char *set_err = NULL;
    if (cpr_config_set_by_name(cfg, key, val, &set_err)) {
        fprintf(stderr, "%s: %s\n", flag_label, set_err ? set_err : "error");
        free(set_err);
        return;
    }
    cli_params_add(cp, key, val);
}

/* ---- JSON output ---- */

/* Prints a JSON-safe string (escaping backslash and double-quote). */
static void print_json_str(const char *s)
{
    putchar('"');
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') putchar('\\');
        putchar(*s);
    }
    putchar('"');
}

static void print_json(const CPRResults *results, const CPRMCResult *mc)
{
    printf("{\n");
    const char *sep = "";

    if (results->has_Neff) {
        printf("%s  \"Neff\": %.10g", sep, results->Neff); sep = ",\n";
    }
    printf("%s  \"YPBBN\": %.10g", sep, results->YPBBN);   sep = ",\n";
    printf("%s  \"YPCMB\": %.10g", sep, results->YPCMB);   sep = ",\n";
    printf("%s  \"DoH\": %.10g",   sep, results->DoH);      sep = ",\n";
    printf("%s  \"He3oH\": %.10g", sep, results->He3oH);    sep = ",\n";
    printf("%s  \"He3oHe4\": %.10g", sep, results->He3oHe4); sep = ",\n";
    printf("%s  \"Li7oH\": %.10g", sep, results->Li7oH);    sep = ",\n";
    if (results->has_Li6oLi7) {
        printf("%s  \"Li6oLi7\": %.10g", sep, results->Li6oLi7); sep = ",\n";
    }
    if (results->has_YCNO) {
        printf("%s  \"YCNO\": %.10g", sep, results->YCNO); sep = ",\n";
    }
    if (results->has_Omeganurel) {
        printf("%s  \"Omeganurel\": %.10g", sep, results->Omeganurel); sep = ",\n";
    }
    if (results->has_OneOverOmeganunr) {
        printf("%s  \"OneOverOmeganunr\": %.10g", sep, results->OneOverOmeganunr); sep = ",\n";
    }

    /* Per-nuclide final abundances. */
    if (results->n_nuclides > 0) {
        printf("%s  \"Y_final\": {", sep); sep = ",\n";
        for (size_t i = 0; i < results->n_nuclides; i++) {
            printf("%s\n    ", i > 0 ? "," : "");
            print_json_str(results->nuclide_names[i]);
            printf(": %.10g", results->Y_final[i]);
        }
        printf("\n  }");
    }

    /* MC summary (central/mean/std per quantity; not the full sample array). */
    if (mc && mc->n > 0) {
        printf("%s  \"mc\": {", sep);
        for (size_t i = 0; i < mc->n; i++) {
            const CPRMCQuantity *q = &mc->items[i];
            printf("%s\n    ", i > 0 ? "," : "");
            print_json_str(q->name);
            printf(": {\"central\": %.10g, \"mean\": %.10g, \"std\": %.10g}",
                   q->central, q->mean, q->std);
        }
        printf("\n  }");
    }

    printf("\n}\n");
}

/* ---- Plain-text report (mirrors cli.py's default output) ---- */

static void print_plain(const CPRConfig *cfg, const CPRResults *results,
                        const CPRMCResult *mc, int mc_n)
{
    const char *sep = "────────────────────────────────────────────────────";
    char header[80];
    snprintf(header, sizeof(header), "PRIMAT results at T = %g MeV", cfg->T_end_MeV);
    printf("%s\n", sep);
    int left_pad = (52 - (int)strlen(header)) / 2;
    if (left_pad < 0) left_pad = 0;
    printf("%*s%s\n", left_pad, "", header);
    printf("%s\n", sep);

/* Helper: if mc has this quantity, append " +/- std"; else nothing. */
#define MC_STD(name, fmt) do { \
    if (mc) { \
        size_t idx = cpr_mc_result_index(mc, name); \
        if (idx < mc->n) printf(" +/- " fmt, mc->items[idx].std); \
    } \
} while (0)

    if (results->has_Neff) {
        printf("Neff       = %.8f", results->Neff);
        MC_STD("Neff", "%.8f");
        putchar('\n');
    }
    printf("YP (BBN)   = %.8f", results->YPBBN);
    MC_STD("YPBBN", "%.8f"); putchar('\n');
    printf("YP (CMB)   = %.8f", results->YPCMB);
    MC_STD("YPCMB", "%.8f"); putchar('\n');
    printf("D/H        = %.7e", results->DoH);
    MC_STD("DoH", "%.7e"); putchar('\n');
    printf("He3/H      = %.7e", results->He3oH);
    MC_STD("He3oH", "%.7e"); putchar('\n');
    printf("He3/He4    = %.7e", results->He3oHe4);
    MC_STD("He3oHe4", "%.7e"); putchar('\n');
    printf("Li7/H      = %.6e", results->Li7oH);
    MC_STD("Li7oH", "%.6e"); putchar('\n');
    if (results->has_Li6oLi7) {
        printf("Li6/Li7    = %.6e", results->Li6oLi7);
        MC_STD("Li6oLi7", "%.6e"); putchar('\n');
    }
    if (results->has_YCNO) {
        printf("CNO (mass) = %.6e", results->YCNO);
        MC_STD("YCNO", "%.6e"); putchar('\n');
    }
#undef MC_STD

    if (mc) printf("--- Monte-Carlo: %d samples ---\n", mc_n);
}

int cpr_cli_main(int argc, char **argv)
{
    char data_dir_buf[4096];
    const char *data_dir = default_data_dir(data_dir_buf, sizeof(data_dir_buf));
    const char *custom_nuclear_dir = NULL;
    const char *ini_path = NULL;
    int cache_info = 0, cache_clear = 0, credits = 0, version = 0;
    int do_json = 0;
    int mc_n = 0, mc_seed = 0;

    /* --data_dir, --user_nuclear_dir and --ini must be known before
     * cpr_config_init_defaults runs (the first picks the data directory;
     * the others are applied after defaults), so scan for them first;
     * everything else is applied in a second pass, in the same precedence
     * order as cli.py: defaults, then .ini, then named flags, then --set
     * (later wins). */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data_dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--user_nuclear_dir") == 0 && i + 1 < argc) {
            custom_nuclear_dir = argv[++i];
        } else if (strcmp(argv[i], "--ini") == 0 && i + 1 < argc) {
            ini_path = argv[++i];
        } else if (strcmp(argv[i], "--cache-info") == 0) {
            cache_info = 1;
        } else if (strcmp(argv[i], "--cache-clear") == 0) {
            cache_clear = 1;
        } else if (strcmp(argv[i], "--credits") == 0) {
            credits = 1;
        } else if (strcmp(argv[i], "--version") == 0) {
            version = 1;
        } else if (strcmp(argv[i], "--json") == 0) {
            do_json = 1;
        } else if (strcmp(argv[i], "--mc") == 0 && i + 1 < argc) {
            mc_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mc-seed") == 0 && i + 1 < argc) {
            mc_seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

    if (version) {
        printf("primat-c %s\n", CPRIMAT_VERSION);
        return 0;
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
            fprintf(stderr, "--user_nuclear_dir: '%s' is not a directory\n", custom_nuclear_dir);
            cpr_config_free(&cfg);
            return 2;
        }
        free(cfg.user_nuclear_dir);
        cfg.user_nuclear_dir = strdup(custom_nuclear_dir);
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

    /* Collect all user-supplied overrides here; forwarded to MC workers. */
    CLIParams cp;
    memset(&cp, 0, sizeof(cp));

    /* Record user_nuclear_dir in base_params so MC workers inherit it. */
    if (custom_nuclear_dir)
        cli_params_add(&cp, "user_nuclear_dir",
                       (CPRParam){CPR_STRING, .v.s = custom_nuclear_dir});

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has_val = (i + 1 < argc);

        /* Already handled in first pass or not a solver param. */
        if (strcmp(a, "--data_dir") == 0 || strcmp(a, "--user_nuclear_dir") == 0
            || strcmp(a, "--ini") == 0) { i++; continue; }
        if (strcmp(a, "--cache-info") == 0 || strcmp(a, "--cache-clear") == 0
            || strcmp(a, "--credits") == 0 || strcmp(a, "--version") == 0
            || strcmp(a, "--json") == 0
            || strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) continue;
        if (strcmp(a, "--mc") == 0 || strcmp(a, "--mc-seed") == 0) { i++; continue; }

        /* ---- Simple scalar flags (string or numeric) ---- */
        if (strcmp(a, "--Omegabh2") == 0 && has_val) {
            CPRParam p = {CPR_DOUBLE, .v.d = atof(argv[++i])};
            apply_param(&cfg, &cp, "Omegabh2", p, "--Omegabh2");
        } else if (strcmp(a, "--DeltaNeff") == 0 && has_val) {
            CPRParam p = cpr_parse_literal(argv[++i]);
            apply_param(&cfg, &cp, "DeltaNeff", p, "--DeltaNeff");
        } else if (strcmp(a, "--network") == 0 && has_val) {
            CPRParam p = {CPR_STRING, .v.s = argv[++i]};
            apply_param(&cfg, &cp, "network", p, "--network");
        } else if (strcmp(a, "--amax") == 0 && has_val) {
            CPRParam p = cpr_parse_literal(argv[++i]);
            apply_param(&cfg, &cp, "amax", p, "--amax");
        } else if (strcmp(a, "--numerical_precision") == 0 && has_val) {
            CPRParam p = cpr_parse_literal(argv[++i]);
            apply_param(&cfg, &cp, "numerical_precision", p, "--numerical_precision");
        } else if (strcmp(a, "--munuOverTnu") == 0 && has_val) {
            CPRParam p = cpr_parse_literal(argv[++i]);
            apply_param(&cfg, &cp, "munuOverTnu", p, "--munuOverTnu");
        } else if (strcmp(a, "--verbose") == 0) {
            CPRParam p = {CPR_BOOL, .v.b = 1};
            apply_param(&cfg, &cp, "verbose", p, "--verbose");

        /* ---- Output file paths ---- */
        } else if (strcmp(a, "--output_file") == 0 && has_val) {
            CPRParam p = {CPR_STRING, .v.s = argv[++i]};
            apply_param(&cfg, &cp, "output_file", p, "--output_file");
        } else if (strcmp(a, "--output_final_file") == 0 && has_val) {
            CPRParam p = {CPR_STRING, .v.s = argv[++i]};
            apply_param(&cfg, &cp, "output_final_file", p, "--output_final_file");
        } else if (strcmp(a, "--output_background_file") == 0 && has_val) {
            CPRParam p = {CPR_STRING, .v.s = argv[++i]};
            apply_param(&cfg, &cp, "output_background_file", p, "--output_background_file");
        } else if (strcmp(a, "--output_mc_file") == 0 && has_val) {
            CPRParam p = {CPR_STRING, .v.s = argv[++i]};
            apply_param(&cfg, &cp, "output_mc_file", p, "--output_mc_file");

        /* ---- Boolean --flag / --no-flag pairs ---- */
        } else if (strncmp(a, "--", 2) == 0) {
            /* Check --no-<flag> first (longer prefix), then --<flag>. */
            int matched = 0;
            for (int fi = 0; bool_flags[fi]; fi++) {
                char pos_flag[64], neg_flag[70];
                snprintf(pos_flag, sizeof(pos_flag), "--%s", bool_flags[fi]);
                snprintf(neg_flag, sizeof(neg_flag), "--no-%s", bool_flags[fi]);
                if (strcmp(a, neg_flag) == 0) {
                    CPRParam p = {CPR_BOOL, .v.b = 0};
                    apply_param(&cfg, &cp, bool_flags[fi], p, neg_flag);
                    matched = 1; break;
                } else if (strcmp(a, pos_flag) == 0) {
                    CPRParam p = {CPR_BOOL, .v.b = 1};
                    apply_param(&cfg, &cp, bool_flags[fi], p, pos_flag);
                    matched = 1; break;
                }
            }
            if (!matched) {
                /* ---- --set KEY=VALUE ---- */
                if (strcmp(a, "--set") == 0 && has_val) {
                    const char *entry = argv[++i];
                    const char *eq = strchr(entry, '=');
                    if (!eq) {
                        fprintf(stderr, "--set %s: expected KEY=VALUE\n", entry);
                        cpr_config_free(&cfg);
                        return 2;
                    }
                    if (cp.n >= MAX_CLI_PARAMS) {
                        fprintf(stderr, "--set: too many parameters (max %d)\n", MAX_CLI_PARAMS);
                        cpr_config_free(&cfg);
                        return 2;
                    }
                    size_t klen = (size_t)(eq - entry);
                    if (klen >= sizeof(cp.key_store[0])) klen = sizeof(cp.key_store[0]) - 1;
                    memcpy(cp.key_store[cp.n], entry, klen);
                    cp.key_store[cp.n][klen] = '\0';
                    const char *key = cp.key_store[cp.n];
                    CPRParam p = cpr_parse_literal(eq + 1);
                    apply_param(&cfg, &cp, key, p, entry);
                } else {
                    fprintf(stderr, "unrecognized argument: %s\n", a);
                    usage(argv[0]);
                    cpr_config_free(&cfg);
                    return 2;
                }
            }
        } else {
            fprintf(stderr, "unrecognized argument: %s\n", a);
            usage(argv[0]);
            cpr_config_free(&cfg);
            return 2;
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

    /* ---- Optional Monte-Carlo uncertainty propagation ---- */
    CPRMCResult mc_result;
    memset(&mc_result, 0, sizeof(mc_result));
    CPRMCResult *mc = NULL;

    if (mc_n > 0) {
        /* Standard observable set, mirroring _DEFAULT_MC_OBSERVABLES in
         * primat/backend.py; cpr_mc_uncertainty silently skips any quantity
         * whose name is not in CPRResults (e.g. "Li6oLi7" when the small
         * network is used and has_Li6oLi7 == 0). We filter them here instead
         * by checking the results struct so unknown-quantity errors are avoided. */
        const char *all_quantities[] = {
            "Neff", "YPBBN", "YPCMB", "DoH", "He3oH", "He3oHe4", "Li7oH",
            "Li6oLi7", "YCNO"
        };
        const char *quantities[9];
        size_t n_q = 0;
        for (size_t qi = 0; qi < sizeof(all_quantities)/sizeof(all_quantities[0]); qi++) {
            int found = 0;
            cpr_results_get_quantity(&results, all_quantities[qi], &found);
            if (found) quantities[n_q++] = all_quantities[qi];
        }

        if (cpr_mc_uncertainty(mc_n, quantities, n_q,
                               data_dir,
                               cp.items, cp.n,
                               mc_seed, -1 /* all cores */, NULL,
                               NULL, NULL, 0, /*show_progress=*/1,
                               &mc_result, &err)) {
            fprintf(stderr, "MC error: %s\n", err);
            free(err);
            /* Non-fatal: print the main result without MC. */
        } else {
            mc = &mc_result;
        }
    }

    /* ---- Output ---- */
    if (do_json) {
        print_json(&results, mc);
    } else {
        print_plain(&cfg, &results, mc, mc_n);
    }

    if (mc) cpr_mc_result_free(&mc_result);
    cprimat_results_free(&results);
    cpr_config_free(&cfg);
    return 0;
}
