/* api.c -- see cprimat/api.h. Port of pyprimat/main.py's PyPR.__init__ +
 * PyPR.solve().
 */
#include "cprimat/api.h"
#include "cprimat/constants.h"
#include "cprimat/plasma.h"
#include "cprimat/background.h"
#include "cprimat/network_data.h"
#include "cprimat/nuclear_network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Mirrors main.py's local `_ratio` helper: 0/0 -> nan (nothing produced,
 * nothing expected), x/0 -> inf (something produced, nothing to divide by;
 * e.g. a custom network stripping out a denominator nuclide). */
static double ratio(double num, double den)
{
    if (den != 0.0) return num / den;
    return (num == 0.0) ? NAN : INFINITY;
}

/* N+Z (mass number) of nuclide `name` per cfg->nuclides.csv, or 0 if not
 * found (mirrors a Y_final.get(name, 0.0) miss contributing nothing). */
static int nuclide_A(const CPRConfig *cfg, const char *name)
{
    for (size_t j = 0; j < cfg->nuclides.n; j++)
        if (strcmp(cfg->nuclides.items[j].name, name) == 0)
            return cfg->nuclides.items[j].N + cfg->nuclides.items[j].Z;
    return 0;
}

/* Builds the "BBN observables" dict (main.py's `solve()` result, see
 * api.h's top comment) from an already-solved `nn` and its driving
 * `cfg`/`bg`. Factored out of cprimat_run so mc.c's per-sample MC loop
 * can reuse the same assembly logic without re-running cprimat_run's own
 * Plasma/CPRNuclearRates/CPRBackground setup (which mc.c builds once per
 * worker thread and reuses across samples -- see mc.h's top comment).
 * `nn` is read-only here; the caller still owns and frees it. */
void cpr_assemble_results(CPRResults *results, const CPRConfig *cfg,
                           const CPRNuclearNetwork *nn, const CPRBackground *bg)
{
    memset(results, 0, sizeof(*results));

    /* ---- Light-element ratios (main.py's `results` dict, unconditional
     * entries). YPBBN = 4*Y_He4 (BBN-convention helium mass fraction);
     * YPCMB converts it to the CMB-convention n_He/(n_He+n_H) mass
     * fraction via the He4/H atomic-mass ratios. ---- */
    double Yp  = cpr_nuclear_network_get(nn, "p");
    double Yd  = cpr_nuclear_network_get(nn, "H2");
    double Yt  = cpr_nuclear_network_get(nn, "H3");
    double YHe3 = cpr_nuclear_network_get(nn, "He3");
    double Ya  = cpr_nuclear_network_get(nn, "He4");
    double YLi7 = cpr_nuclear_network_get(nn, "Li7");
    double YBe7 = cpr_nuclear_network_get(nn, "Be7");

    results->YPBBN = 4.0 * Ya;
    results->YPCMB = ((g_const.He4Overma / 4.0) * results->YPBBN)
        / ((g_const.He4Overma / 4.0) * results->YPBBN
           + g_const.HOverma * (1.0 - results->YPBBN));
    results->DoH     = ratio(Yd, Yp);
    results->He3oH   = ratio(Yt + YHe3, Yp);
    results->He3oHe4 = ratio(Yt + YHe3, Ya);
    results->Li7oH   = ratio(YLi7 + YBe7, Yp);

    /* Li6/Li7 (observable ratio after Be7->Li7 decay): large-network only,
     * guarded on Y(Li6) > 0 exactly as main.py does. */
    double YLi6 = cpr_nuclear_network_get(nn, "Li6");
    if (YLi6 > 0.0) {
        results->has_Li6oLi7 = 1;
        results->Li6oLi7 = YLi6 / (YLi7 + YBe7);
    }

    /* YCNO (mass fraction): sum_i A_i Y_i over all tracked C/N/O isotopes
     * (large network only); guarded on >0 exactly as main.py does. */
    double cno = 0.0;
    for (size_t i = 0; i < nn->n_species; i++) {
        const char *s = nn->abundance_names[i];
        if (strlen(s) >= 2 && (s[0] == 'C' || s[0] == 'N' || s[0] == 'O')) {
            int all_digits = 1;
            for (const char *p = s + 1; *p; p++)
                if (*p < '0' || *p > '9') { all_digits = 0; break; }
            if (all_digits)
                cno += nuclide_A(cfg, s) * cpr_nuclear_network_get(nn, s);
        }
    }
    if (cno > 0.0) {
        results->has_YCNO = 1;
        results->YCNO = cno;
    }

    /* Neutrino sector: Neff/Omeganurel/OneOverOmeganunr, only when the
     * background tracks a neutrino sector (mirrors main.py's `is not None`
     * guards on rho_nu_total_final/Omeganuh2_relnu/_nrnu). */
    double Tg_f, rho_nu_tot_f;
    if (cpr_bg_rho_nu_total_final(bg, &Tg_f, &rho_nu_tot_f) == 0) {
        results->has_Neff = 1;
        results->Neff = cpr_bg_N_eff(bg, Tg_f, rho_nu_tot_f);
    }
    double relnu;
    if (cpr_bg_Omeganuh2_relnu(bg, &relnu) == 0) {
        results->has_Omeganurel = 1;
        results->Omeganurel = relnu * 1e6;
    }
    double nrnu;
    if (cpr_bg_Omeganuh2_nrnu(bg, &nrnu) == 0) {
        results->has_OneOverOmeganunr = 1;
        results->OneOverOmeganunr = 1.0 / (nrnu * 1e-6);
    }

    /* ---- Per-nuclide final abundances (owned copy: `nn` outlives this
     * call only as long as the caller keeps it alive). ---- */
    results->n_nuclides = nn->n_species;
    results->nuclide_names = malloc(nn->n_species * sizeof(*results->nuclide_names));
    results->Y_final = malloc(nn->n_species * sizeof(double));
    for (size_t i = 0; i < nn->n_species; i++) {
        memcpy(results->nuclide_names[i], nn->abundance_names[i], 16);
        results->Y_final[i] = nn->Y_final[i];
    }
}

int cprimat_run(const CPRConfig *cfg, CPRResults *results, char **errmsg)
{
    CPRPlasma pl;
    if (cpr_plasma_init(&pl, cfg, errmsg))
        return 1;

    CPRNuclearRates nr;
    if (cpr_nuclear_rates_init(&nr, cfg, errmsg)) {
        cpr_plasma_free(&pl);
        return 1;
    }
    /* Mirrors NuclearNetwork.solve()'s own nucl.apply_variations(cfg) call
     * (p_<rxn>/NP_delta_<rxn> rate-variation knobs); cpr_nuclear_network_solve's
     * docstring requires this be done by the caller before passing `nr` in. */
    cpr_nuclear_rates_apply_variations(&nr, cfg);

    CPRBackground bg;
    int bg_rc = cfg->custom_background
        ? cpr_bg_init_custom(&bg, cfg, &pl, cfg->custom_background, errmsg)
        : cpr_bg_init_standard(&bg, cfg, &pl, errmsg);
    if (bg_rc) {
        cpr_nuclear_rates_free(&nr);
        cpr_plasma_free(&pl);
        return 1;
    }

    CPRNuclearNetwork nn;
    if (cpr_nuclear_network_solve(&nn, cfg, &nr, &bg, errmsg)) {
        cpr_background_free(&bg);
        cpr_nuclear_rates_free(&nr);
        cpr_plasma_free(&pl);
        return 1;
    }

    cpr_assemble_results(results, cfg, &nn, &bg);

    /* Output files (output_final.dat / time-evolution TSV) are already
     * written by cpr_nuclear_network_solve itself, gated on
     * cfg->output_final_result/output_time_evolution -- nothing more to do
     * here (cfg->output_background_evolution is not yet honoured, see
     * api.h's top comment). */
    cpr_nuclear_network_free(&nn);
    cpr_background_free(&bg);
    cpr_nuclear_rates_free(&nr);
    cpr_plasma_free(&pl);
    return 0;
}

void cprimat_results_free(CPRResults *results)
{
    free(results->nuclide_names);
    free(results->Y_final);
    results->nuclide_names = NULL;
    results->Y_final = NULL;
    results->n_nuclides = 0;
}

double cpr_results_get_quantity(const CPRResults *r, const char *name, int *found)
{
    *found = 1;
    if (strcmp(name, "YPCMB") == 0) return r->YPCMB;
    if (strcmp(name, "YPBBN") == 0) return r->YPBBN;
    if (strcmp(name, "DoH") == 0) return r->DoH;
    if (strcmp(name, "He3oH") == 0) return r->He3oH;
    if (strcmp(name, "He3oHe4") == 0) return r->He3oHe4;
    if (strcmp(name, "Li7oH") == 0) return r->Li7oH;
    if (strcmp(name, "Li6oLi7") == 0 && r->has_Li6oLi7) return r->Li6oLi7;
    if (strcmp(name, "YCNO") == 0 && r->has_YCNO) return r->YCNO;
    if (strcmp(name, "Neff") == 0 && r->has_Neff) return r->Neff;
    if (strcmp(name, "Omeganurel") == 0 && r->has_Omeganurel) return r->Omeganurel;
    if (strcmp(name, "OneOverOmeganunr") == 0 && r->has_OneOverOmeganunr) return r->OneOverOmeganunr;
    for (size_t i = 0; i < r->n_nuclides; i++)
        if (strcmp(r->nuclide_names[i], name) == 0) return r->Y_final[i];
    *found = 0;
    return 0.0;
}
