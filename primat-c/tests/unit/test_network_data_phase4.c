/* test_network_data_phase4.c -- checks the Phase 4 physics layer
 * (cpr_load_network / cpr_nuclear_rates_init) on the real "small" and
 * "large, amax=8" networks: species/reaction counts against CLAUDE.md's
 * documented numbers, N/Z conservation (via cpr_check_conservation,
 * exercised inside cpr_nuclear_rates_init itself -- this test fails loudly
 * if that ever throws), and a basic sanity check that rhsLT/JacobianLT
 * produce finite, non-degenerate output at a representative (Y, T) point. */
#include "network_data.h"
#include "constants.h"

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
    if (cpr_config_init_defaults(&cfg, "../primat/data", &err)) {
        printf("FAIL config init: %s\n", err);
        return 1;
    }

    /* ---- "small" network (cfg->network == "small" by default). ---- */
    CPRNuclearRates small;
    if (cpr_nuclear_rates_init(&small, &cfg, NULL, &err)) {
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
    if (cpr_nuclear_rates_init(&med, &cfg, NULL, &err)) {
        printf("FAIL cpr_nuclear_rates_init(large, amax=8): %s\n", err);
        return 1;
    }
    CHECK(med.lt_net.n_reac == 68, "large+amax8 LT network: n__p + 67 reactions (CLAUDE.md)");
    cpr_nuclear_rates_free(&med);

    /* ---- full "large" network: just needs to load + conserve N/Z. ---- */
    cfg.amax = -1;
    CPRNuclearRates large;
    if (cpr_nuclear_rates_init(&large, &cfg, NULL, &err)) {
        printf("FAIL cpr_nuclear_rates_init(large): %s\n", err);
        return 1;
    }
    CHECK(large.lt_net.n_reac == 429, "full large LT network: n__p + 428 reactions (CLAUDE.md)");
    cpr_nuclear_rates_free(&large);

    cpr_config_free(&cfg);

    /* ---- custom_network (the GUI "Customise Reactions" override,
     * CPRCustomNetwork): removed/replaced/added on the "small" network. ---- */
    {
        char *cerr = NULL;
        CPRConfig ccfg;
        if (cpr_config_init_defaults(&ccfg, "../primat/data", &cerr)) {
            printf("FAIL custom_network config init: %s\n", cerr);
            return 1;
        }
        /* Disable the nuclear-QED rescale (default on, see CLAUDE.md): it
         * post-multiplies fwd_median for every reaction it knows about --
         * including d_p__He3_g, a radiative capture -- after the
         * custom_network injection loop (step 10, network_data.c), so
         * leaving it on would make the injected rate=10*T9 unrecoverable
         * by a simple value check below. */
        ccfg.nuclear_qed_corrections = 0;

        /* "replaced": override d_p__He3_g's forward rate with a known
         * synthetic table (rate = 10*T9, on a grid wide/dense enough for
         * cpr_resample_rate_table's cubic notaknot fit, see spline.c). */
        double T9[6]  = { 0.001, 0.01, 0.1, 1.0, 5.0, 10.0 };
        double rep_rate[6], rep_err[6];
        for (int i = 0; i < 6; i++) { rep_rate[i] = 10.0 * T9[i]; rep_err[i] = 0.0; }

        /* "added": an off-catalog reaction (d_d__He4_g; the catalog only
         * has the aliased "d_d__a_g" -- see reactions_large.csv -- so this
         * exercises parse_reaction_name's tokeniser, not find_reaction_entry). */
        double add_rate[6], add_err[6];
        for (int i = 0; i < 6; i++) { add_rate[i] = 1.0 + T9[i]; add_err[i] = 0.0; }

        CPRCustomTable tables[2];
        snprintf(tables[0].name, sizeof(tables[0].name), "d_p__He3_g");
        tables[0].T9 = T9; tables[0].rate = rep_rate; tables[0].err = rep_err; tables[0].n = 6;
        snprintf(tables[1].name, sizeof(tables[1].name), "d_d__He4_g");
        tables[1].T9 = T9; tables[1].rate = add_rate; tables[1].err = add_err; tables[1].n = 6;

        char removed_names[1][64];
        snprintf(removed_names[0], sizeof(removed_names[0]), "Li7_p__a_a");

        CPRCustomNetwork custom = {
            .removed = removed_names, .n_removed = 1,
            .tables = tables, .n_tables = 2,
        };

        CPRNuclearRates cnr;
        if (cpr_nuclear_rates_init(&cnr, &ccfg, &custom, &cerr)) {
            printf("FAIL cpr_nuclear_rates_init(custom_network): %s\n", cerr);
            return 1;
        }
        CHECK(cnr.lt_net.n_reac == 13,
              "custom_network: small -1 removed +1 added == still 13 (n__p + 12)");
        long irem = -1;
        for (size_t i = 0; i < cnr.lt_net.n_reac; i++)
            if (strcmp(cnr.lt_net.names[i], "Li7_p__a_a") == 0) irem = (long)i;
        CHECK(irem < 0, "custom_network: removed reaction 'Li7_p__a_a' is absent");

        long iadd = -1;
        for (size_t i = 0; i < cnr.lt_net.n_reac; i++)
            if (strcmp(cnr.lt_net.names[i], "d_d__He4_g") == 0) iadd = (long)i;
        CHECK(iadd >= 0, "custom_network: added reaction 'd_d__He4_g' is present");
        if (iadd >= 0) {
            /* Stoichiometry: 2 d -> He4 (+g, not a tracked species). */
            const CPRReaction *rx = &cnr.lt_net.network[iadd];
            int n_d = 0, n_he4 = 0;
            for (size_t s = 0; s < rx->reactants.n; s++)
                if (strcmp(cnr.lt_net.species[rx->reactants.species_idx[s]], "H2") == 0)
                    n_d += (int)rx->reactants.mult[s];
            for (size_t s = 0; s < rx->products.n; s++)
                if (strcmp(cnr.lt_net.species[rx->products.species_idx[s]], "He4") == 0)
                    n_he4 += (int)rx->products.mult[s];
            CHECK(n_d == 2 && n_he4 == 1,
                  "custom_network: added reaction stoichiometry is 2*H2 -> He4 (+g)");
        }

        long irep = -1;
        for (size_t i = 0; i < cnr.lt_net.n_reac; i++)
            if (strcmp(cnr.lt_net.names[i], "d_p__He3_g") == 0) irep = (long)i;
        CHECK(irep >= 0, "custom_network: replaced reaction 'd_p__He3_g' still present");
        if (irep >= 0) {
            /* The resampled forward rate at the grid point nearest T9=1.0
             * should match the injected rate=10*T9 there, not the shipped
             * table's value (fwd_median is row-major (n_reac-1) x n_grid,
             * one row per *thermonuclear* reaction -- names[1..], so row
             * index is irep-1, see network_data.h's CPRNetworkDef docstring). */
            size_t i1 = 0;
            double best = 1e300;
            for (size_t g = 0; g < cnr.lt_net.n_grid; g++) {
                double d = fabs(cnr.lt_net.grid[g] - 1.0);
                if (d < best) { best = d; i1 = g; }
            }
            double got = cnr.lt_net.fwd_median[(size_t)(irep - 1) * cnr.lt_net.n_grid + i1];
            double want = 10.0 * cnr.lt_net.grid[i1];
            CHECK(fabs(got - want) < 1e-3 * fabs(want),
                  "custom_network: replaced reaction's resampled rate matches the injected table");
        }

        cpr_nuclear_rates_free(&cnr);
        cpr_config_free(&ccfg);
    }

    /* ---- user_nuclear_dir overlay (mirrors PRIMATConfig.user_nuclear_dir /
     * test_config.py): cpr_config_resolve_rates_path checks user_nuclear_dir
     * before the shipped default; and a user_nuclear_dir-supplied network file
     * is loadable end-to-end through cpr_nuclear_rates_init exactly like a
     * shipped one. ---- */
    {
        char *oerr = NULL;
        CPRConfig ocfg;
        if (cpr_config_init_defaults(&ocfg, "../primat/data", &oerr)) {
            printf("FAIL overlay config init: %s\n", oerr);
            return 1;
        }

        char path[4300];

        /* No overlay set: resolves to the shipped default. */
        cpr_config_resolve_rates_path(&ocfg, "nuclear/networks/small.txt", path, sizeof(path));
        CHECK(strstr(path, "../primat/data/nuclear/networks/small.txt") != NULL,
              "resolve_rates_path falls back to the shipped default when no overlay is set");

        /* Build a throwaway user_nuclear_dir containing only a custom
         * 2-reaction network file (referencing two shipped rate tables by
         * name, so this stays an *additive* overlay, not a full takeover --
         * mirrors CLAUDE.md's "true additive overlay" note). Overlay roots
         * behave like primat/data/nuclear, so networks/ lives directly under
         * the overlay directory. */
        system("rm -rf build/test_user_nuclear_dir && mkdir -p build/test_user_nuclear_dir/networks");
        FILE *nf = fopen("build/test_user_nuclear_dir/networks/overlaynet.txt", "w");
        CHECK(nf != NULL, "overlay: created temp user_nuclear_dir network file");
        if (nf) {
            fprintf(nf, "n_p__d_g\nd_d__He3_n\n");
            fclose(nf);
        }

        free(ocfg.user_nuclear_dir);
        ocfg.user_nuclear_dir = strdup("build/test_user_nuclear_dir");

        cpr_config_resolve_rates_path(&ocfg, "nuclear/networks/overlaynet.txt", path, sizeof(path));
        CHECK(strstr(path, "test_user_nuclear_dir/networks/overlaynet.txt") != NULL,
              "resolve_rates_path prefers user_nuclear_dir when the file exists there");
        /* A name only present in the shipped tree still resolves there,
         * since user_nuclear_dir is additive, not a full takeover. */
        cpr_config_resolve_rates_path(&ocfg, "nuclear/networks/small.txt", path, sizeof(path));
        CHECK(strstr(path, "../primat/data/nuclear/networks/small.txt") != NULL,
              "resolve_rates_path still finds shipped files not present in user_nuclear_dir");

        free(ocfg.network);
        ocfg.network = strdup("overlaynet");
        CPRNuclearRates overlay_net;
        if (cpr_nuclear_rates_init(&overlay_net, &ocfg, NULL, &oerr)) {
            printf("FAIL cpr_nuclear_rates_init(overlaynet): %s\n", oerr);
            return 1;
        }
        CHECK(overlay_net.lt_net.n_reac == 3,
              "overlay: user_nuclear_dir-supplied network loads end-to-end (n__p + 2 reactions)");
        cpr_nuclear_rates_free(&overlay_net);

        cpr_config_free(&ocfg);
        system("rm -rf build/test_user_nuclear_dir");
    }

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
