/* test_network_data_phase4.c -- checks the Phase 4 physics layer
 * (cpr_load_network / cpr_nuclear_rates_init) on the real "small" and
 * "large, amax=8" networks: species/reaction counts against CLAUDE.md's
 * documented numbers, N/Z conservation (via cpr_check_conservation,
 * exercised inside cpr_nuclear_rates_init itself -- this test fails loudly
 * if that ever throws), and a basic sanity check that rhsLT/JacobianLT
 * produce finite, non-degenerate output at a representative (Y, T) point. */
#include "cprimat/network_data.h"
#include "cprimat/constants.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

static long find_species(CPRNetworkDef *net, const char *name)
{
    for (size_t i = 0; i < net->n_species; i++)
        if (strcmp(net->species[i], name) == 0) return (long)i;
    return -1;
}

int main(void)
{
    cpr_constants_init();
    char *err = NULL;

    CPRConfig cfg;
    if (cpr_config_init_defaults(&cfg, "../pyprimat", &err)) {
        printf("FAIL config init: %s\n", err);
        return 1;
    }

    /* ---- "small" network (cfg->network == "small" by default). ---- */
    CPRNuclearRates small;
    if (cpr_nuclear_rates_init(&small, &cfg, &err)) {
        printf("FAIL cpr_nuclear_rates_init(small): %s\n", err);
        return 1;
    }
    CHECK(small.lt_net.n_reac == 13, "small LT network: n__p + 12 thermonuclear reactions");
    CHECK(small.mt_net.n_reac == 13, "small MT network: same 13 (MT==LT for network=\"small\")");
    CHECK(small.lt_net.n_species == 8, "small network: 8 species (n,p,H2,H3,He3,He4,Li7,Be7)");
    CHECK(strcmp(small.lt_net.names[0], "n__p") == 0, "small: names[0] == \"n__p\"");

    /* Representative (Y, T) RHS/Jacobian evaluation: T9=0.5 (T_t in K),
     * rho an arbitrary O(1) baryon-density value, Y mostly protons with a
     * trace of everything else (a physically plausible mid-BBN point). */
    {
        size_t ns = small.lt_net.n_species;
        double *Y = malloc(ns * sizeof(double));
        for (size_t i = 0; i < ns; i++) Y[i] = 1e-10;
        long ip = find_species(&small.lt_net, "p"), in = find_species(&small.lt_net, "n");
        Y[ip] = 0.75; Y[in] = 0.25;
        double *dY = malloc(ns * sizeof(double));
        double *J = malloc(ns * ns * sizeof(double));
        double T_t_K = 0.5e9; /* T9 = 0.5 */
        cpr_nuclear_rates_rhs_lt(&small, Y, T_t_K, 1.0, 0.5, 0.01, dY);
        cpr_nuclear_rates_jac_lt(&small, Y, T_t_K, 1.0, 0.5, 0.01, J);
        int all_finite = 1;
        for (size_t i = 0; i < ns; i++) if (!isfinite(dY[i])) all_finite = 0;
        for (size_t i = 0; i < ns * ns; i++) if (!isfinite(J[i])) all_finite = 0;
        CHECK(all_finite, "small: rhsLT/JacobianLT produce finite output");

        /* Baryon-number conservation of the RHS itself: sum_s A_s*dY_s/dt
         * should vanish to numerical precision (each reaction's net
         * stoichiometry conserves A by construction, see
         * cpr_check_conservation -- this is an independent end-to-end
         * check that the compiled kernel matches that invariant). */
        double dA = 0.0;
        for (size_t i = 0; i < ns; i++) dA += (double)(small.lt_net.N[i] + small.lt_net.Z[i]) * dY[i];
        CHECK(fabs(dA) < 1e-8, "small: rhsLT conserves baryon number (sum A_s dY_s/dt = 0)");

        /* Finite-difference cross-check of the analytic Jacobian against
         * the RHS, at this same point. */
        double h = 1e-7;
        int fd_ok = 1;
        for (size_t u = 0; u < ns; u++) {
            double *Yp = malloc(ns * sizeof(double)), *Ym = malloc(ns * sizeof(double));
            memcpy(Yp, Y, ns * sizeof(double)); memcpy(Ym, Y, ns * sizeof(double));
            Yp[u] += h; Ym[u] -= h;
            double *dYp = malloc(ns * sizeof(double)), *dYm = malloc(ns * sizeof(double));
            cpr_nuclear_rates_rhs_lt(&small, Yp, T_t_K, 1.0, 0.5, 0.01, dYp);
            cpr_nuclear_rates_rhs_lt(&small, Ym, T_t_K, 1.0, 0.5, 0.01, dYm);
            for (size_t s = 0; s < ns; s++) {
                double fd = (dYp[s] - dYm[s]) / (2.0 * h);
                if (fabs(fd - J[s * ns + u]) > 1e-4 * (fabs(fd) + 1.0)) fd_ok = 0;
            }
            free(Yp); free(Ym); free(dYp); free(dYm);
        }
        CHECK(fd_ok, "small: analytic JacobianLT matches finite-difference rhsLT derivative");
        free(Y); free(dY); free(J);
    }
    cpr_nuclear_rates_free(&small);

    /* ---- "large, amax=8" network: must reproduce the old "medium"
     * network's exact 68-reaction equivalent (CLAUDE.md). ---- */
    free(cfg.network);
    cfg.network = strdup("large");
    cfg.amax = 8;
    CPRNuclearRates med;
    if (cpr_nuclear_rates_init(&med, &cfg, &err)) {
        printf("FAIL cpr_nuclear_rates_init(large, amax=8): %s\n", err);
        return 1;
    }
    CHECK(med.lt_net.n_reac == 68, "large+amax8 LT network: n__p + 67 reactions (CLAUDE.md)");
    cpr_nuclear_rates_free(&med);

    /* ---- full "large" network: just needs to load + conserve N/Z. ---- */
    cfg.amax = -1;
    CPRNuclearRates large;
    if (cpr_nuclear_rates_init(&large, &cfg, &err)) {
        printf("FAIL cpr_nuclear_rates_init(large): %s\n", err);
        return 1;
    }
    CHECK(large.lt_net.n_reac == 429, "full large LT network: n__p + 428 reactions (CLAUDE.md)");
    cpr_nuclear_rates_free(&large);

    cpr_config_free(&cfg);

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
