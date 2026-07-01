/* api.h -- the thin top-level wrapper (port of primat/main.py's PRIMAT
 * class).
 *
 * `cprimat_run` is the single entry point mirroring `PRIMAT(params).solve()`:
 * it owns the full init sequence (plasma -> nuclear rates -> background ->
 * nuclear network) and assembles the same "BBN observables" dict that
 * `PRIMAT.solve()` returns, plus the per-nuclide final abundances (`Y_final`
 * in Python). Unlike Python's dynamically-keyed dict, each optional
 * observable here is a `has_*` flag next to its value, set only when the
 * corresponding Python dict key would have been present (e.g. `Neff`/
 * `Omeganurel`/`OneOverOmeganunr` are CPR_BG_STANDARD-only, `Li6oLi7`/
 * `YCNO` are large-network-only).
 *
 * `custom_network` (the GUI "Customise Reactions" override) *is* supported,
 * via `cprimat_run`'s optional `custom` parameter -- see network_data.h's
 * CPRCustomNetwork.
 * `cfg.output_time_evolution`/`output_final_file` *are* honoured (delegated
 * to nuclear_network.h's existing writers for the disk side; `cfg.output_time_evolution`
 * also populates `CPRResults`'s `evol_*` in-memory arrays directly --
 * so `primat/_primat_c/_wrapper.c` can hand the same
 * `EvolutionResult` shape back to Python with no disk I/O).
 *
 * Reference: Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095).
 */
#ifndef CPRIMAT_API_H
#define CPRIMAT_API_H

#include "config.h"
#include "background.h"
#include "network_data.h"
#include "nuclear_network.h"
#include <stddef.h>

typedef struct {
    /* ---- Light-element ratios (always present; mirrors PyPR.solve()'s
     * unconditional dict entries). _ratio's "0/0 -> nan, x/0 -> inf"
     * convention (main.py) is reproduced exactly. ---- */
    double YPCMB, YPBBN, DoH, He3oH, He3oHe4, Li7oH;

    /* ---- Large-network-only (set iff the corresponding nuclide is
     * tracked with Y>0 at the final state -- mirrors main.py's
     * `if finL.get("Li6", 0.0) > 0` / `if cno > 0` guards). ---- */
    int has_Li6oLi7;
    double Li6oLi7;
    int has_YCNO;
    double YCNO;

    /* ---- Neutrino sector (CPR_BG_STANDARD only; CPR_BG_CUSTOM's
     * cpr_bg_rho_nu_total_final/Omeganuh2_* still return a value in this
     * port -- see background.h -- but mirror Python's "only added if not
     * None" semantics via these flags for forward parity). ---- */
    int has_Neff;
    double Neff;
    int has_Omeganurel;
    double Omeganurel;
    int has_OneOverOmeganunr;
    double OneOverOmeganunr;

    /* ---- Per-nuclide final mass-fraction abundances Y (mirrors
     * PRIMAT.nuclear.Y_final / get_quantity's nuclide-name fallback).
     * Owned; freed by cprimat_results_free. ---- */
    char (*nuclide_names)[16];
    double *Y_final;
    size_t n_nuclides;

    /* ---- Unified time-evolution arrays, populated
     * iff cfg->output_time_evolution. Mirrors Python's in-memory
     * EvolutionResult so primat/_primat_c/_wrapper.c can hand the same
     * shape back to primat/backend.py with no disk I/O. evol_Y is
     * n_evolution * n_nuclides, row-major, in nuclide_names column order
     * (reuses the field above -- same species list as Y_final). Owned;
     * freed by cprimat_results_free. has_evolution=0 means not populated
     * (n_evolution/evol_* are then 0/NULL). ---- */
    int has_evolution;
    size_t n_evolution;
    double *evol_t, *evol_a, *evol_T_gamma, *evol_Tnue, *evol_Tnumu, *evol_Tnutau;
    double *evol_Y;
} CPRResults;

/* Runs one full PyPR(params).solve()-equivalent BBN computation: builds
 * Plasma -> CPRNuclearRates -> CPRBackground (standard or custom, per
 * cfg->custom_background) -> CPRNuclearNetwork, integrates HT->MT->LT,
 * and fills `results` (zeroed first). Honours cfg->output_final_file
 * (always), cfg->output_time_evolution (if set), and
 * cfg->output_background_evolution (if set) the same way as the Python
 * backend. `custom` (may be NULL) is forwarded verbatim to
 * cpr_nuclear_rates_init -- the GUI "Customise Reactions" override.
 *
 * Returns 0 on success (caller must cprimat_results_free), nonzero with
 * *errmsg set (caller frees) on any init/integration failure -- mirrors
 * PyPR's constructor or solve() raising. */
int cprimat_run(const CPRConfig *cfg, const CPRCustomNetwork *custom,
                  CPRResults *results, char **errmsg);

/* Factored out of cprimat_run so mc.c's per-sample MC loop can reuse the
 * exact same observable-assembly logic against an already-solved `nn`
 * (and the worker's already-built `bg`), without repeating the expensive
 * Plasma/CPRNuclearRates/CPRBackground setup per sample -- see mc.h's top
 * comment. Zeroes `results` first; both `nn`/`bg` are read-only and still
 * owned by the caller. */
void cpr_assemble_results(CPRResults *results, const CPRConfig *cfg,
                           const CPRNuclearNetwork *nn, const CPRBackground *bg);

void cprimat_results_free(CPRResults *results);

/* Returns a scalar quantity by name (mirrors PyPR.get_quantity): first
 * checks the fixed result fields above (by name, e.g. "YPBBN"/"DoH"/
 * "Neff"/...), then falls back to a per-nuclide final abundance lookup in
 * `nuclide_names`/`Y_final` (e.g. "H2"/"He4"/"Li7"). Sets *found = 0 (and
 * returns 0.0) if `name` matches neither -- mirrors get_quantity's
 * ValueError, but as a status flag instead of an exception since C has no
 * exception mechanism; callers needing the "unknown quantity" error mirror
 * cli.c-style error formatting on their own. */
double cpr_results_get_quantity(const CPRResults *results, const char *name, int *found);

#endif /* CPRIMAT_API_H */
