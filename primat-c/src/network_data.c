/* network_data.c -- see cprimat/network_data.h. */
#include "network_data.h"

#include "constants.h"
#include "spline.h"
#include "table_io.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *trim(char *s)
{
    while (isspace((unsigned char)*s)) s++;
    if (*s == '\0') return s;
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) *end-- = '\0';
    return s;
}

/* -------------------------------------------------------------------- */
/* Network list files: "name, table_file.txt" per line. */

int cpr_load_network_list(const char *path, CPRNetworkList *out, char **errmsg)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        char buf[4352];
        snprintf(buf, sizeof(buf), "cannot open network list '%s'", path);
        *errmsg = strdup(buf);
        return 1;
    }

    out->entries = NULL;
    out->n = 0;
    size_t cap = 0;

    char line[512];
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char *s = trim(line);
        if (*s == '\0' || *s == '#')
            continue;

        char *comma = strchr(s, ',');
        char *name, *table_file;
        if (comma) {
            *comma = '\0';
            name = trim(s);
            table_file = trim(comma + 1);
        } else {
            /* Bare reaction name: a decays.txt-resolved analytic decay
             * (see CPRNetworkEntry's doc comment). */
            name = trim(s);
            table_file = "";
        }

        for (size_t i = 0; i < out->n; i++) {
            if (strcmp(out->entries[i].name, name) == 0) {
                char buf[4352];
                snprintf(buf, sizeof(buf),
                          "%s:%d: duplicate reaction entry '%s' (likely copy-paste mistake)",
                          path, lineno, name);
                *errmsg = strdup(buf);
                fclose(f);
                cpr_network_list_free(out);
                return 1;
            }
        }

        if (out->n == cap) {
            cap = cap ? cap * 2 : 64;
            out->entries = realloc(out->entries, cap * sizeof(CPRNetworkEntry));
        }
        CPRNetworkEntry *e = &out->entries[out->n++];
        snprintf(e->name, sizeof(e->name), "%s", name);
        snprintf(e->table_file, sizeof(e->table_file), "%s", table_file);
    }

    fclose(f);
    return 0;
}

void cpr_network_list_free(CPRNetworkList *list)
{
    free(list->entries);
    list->entries = NULL;
    list->n = 0;
}

/* -------------------------------------------------------------------- */
/* decays.txt: "name  halflife_s  rate_s^-1  uncertainty  free-text ref". */

int cpr_load_decays(const char *path, CPRDecayTable *out, char **errmsg)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        char buf[4352];
        snprintf(buf, sizeof(buf), "cannot open decays file '%s'", path);
        *errmsg = strdup(buf);
        return 1;
    }

    out->entries = NULL;
    out->n = 0;
    size_t cap = 0;

    char line[512];
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char *s = trim(line);
        if (*s == '\0' || *s == '#')
            continue;

        char name[64];
        double halflife, rate, unc;
        int consumed = 0;
        if (sscanf(s, "%63s %lf %lf %lf%n", name, &halflife, &rate, &unc, &consumed) != 4) {
            char buf[4352];
            snprintf(buf, sizeof(buf), "%s:%d: cannot parse decay row '%s'", path, lineno, s);
            *errmsg = strdup(buf);
            fclose(f);
            cpr_decay_table_free(out);
            return 1;
        }
        char *ref = trim(s + consumed);

        if (out->n == cap) {
            cap = cap ? cap * 2 : 64;
            out->entries = realloc(out->entries, cap * sizeof(CPRDecayEntry));
        }
        CPRDecayEntry *e = &out->entries[out->n++];
        snprintf(e->name, sizeof(e->name), "%s", name);
        e->halflife_s = halflife;
        e->rate_s_inv = rate;
        e->uncertainty = unc;
        snprintf(e->ref, sizeof(e->ref), "%s", ref);
    }

    fclose(f);
    return 0;
}

void cpr_decay_table_free(CPRDecayTable *t)
{
    free(t->entries);
    t->entries = NULL;
    t->n = 0;
}

/* -------------------------------------------------------------------- */
/* Small CSV helper shared by detailed_balance.csv / reactions_large.csv:
 * both are simple comma-separated, no quoting, fixed field count, one
 * header line. */

static int csv_split(char *line, char **fields, int n_expected)
{
    int n = 0;
    char *p = line;
    while (n < n_expected) {
        char *comma = strchr(p, ',');
        fields[n++] = p;
        if (!comma) {
            if (n != n_expected) return n;
            break;
        }
        *comma = '\0';
        p = comma + 1;
    }
    return n;
}

/* Extract the ``ref=`` label from the first ``#`` header line of a rate-table
 * .txt file (mirrors network_data.py's _reaction_source_from_lines).  The
 * shipped tables have a header such as:
 *   # n + p > d + g   [n_p__d_g]   ref=Ando et al. 2006
 * Writes the text after ``ref=`` (trailing whitespace/newline stripped) into
 * buf[0..bufsize-1].  Falls back to "?" if the file cannot be opened or
 * contains no ``ref=`` token in its ``#`` header lines. */
static void read_reaction_source(const char *path, char *buf, size_t bufsize)
{
    FILE *f = fopen(path, "r");
    if (!f) { snprintf(buf, bufsize, "?"); return; }
    char line[512];
    const char *found = NULL;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '#') break;           /* stop at first data line */
        const char *p = strstr(line, "ref=");
        if (p) { found = p + 4; break; }
    }
    fclose(f);
    if (!found) { snprintf(buf, bufsize, "?"); return; }
    snprintf(buf, bufsize, "%s", found);
    /* Strip trailing whitespace / newline. */
    size_t n = strlen(buf);
    while (n > 0 && ((unsigned char)buf[n - 1] <= ' ')) buf[--n] = '\0';
}

int cpr_load_detailed_balance(const char *path, CPRDetailedBalanceTable *out,
                                char **errmsg)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        char buf[4352];
        snprintf(buf, sizeof(buf), "cannot open detailed_balance file '%s'", path);
        *errmsg = strdup(buf);
        return 1;
    }

    out->entries = NULL;
    out->n = 0;
    size_t cap = 0;

    char line[512];
    int lineno = 0;
    int have_header = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char *s = trim(line);
        if (*s == '\0') continue;
        if (!have_header) { have_header = 1; continue; } /* "reaction,Q_keV,alpha,beta,gamma" */

        char *fields[5];
        if (csv_split(s, fields, 5) != 5) {
            char buf[4352];
            snprintf(buf, sizeof(buf), "%s:%d: expected 5 CSV fields, got '%s'", path, lineno, s);
            *errmsg = strdup(buf);
            fclose(f);
            cpr_detailed_balance_free(out);
            return 1;
        }

        if (out->n == cap) {
            cap = cap ? cap * 2 : 128;
            out->entries = realloc(out->entries, cap * sizeof(CPRDetailedBalanceEntry));
        }
        CPRDetailedBalanceEntry *e = &out->entries[out->n++];
        snprintf(e->reaction, sizeof(e->reaction), "%s", fields[0]);
        e->Q_keV = atof(fields[1]);
        e->alpha = atof(fields[2]);
        e->beta  = atof(fields[3]);
        e->gamma = atof(fields[4]);
    }

    fclose(f);
    return 0;
}

void cpr_detailed_balance_free(CPRDetailedBalanceTable *t)
{
    free(t->entries);
    t->entries = NULL;
    t->n = 0;
}

int cpr_load_reactions_large(const char *path, CPRReactionTable *out, char **errmsg)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        char buf[4352];
        snprintf(buf, sizeof(buf), "cannot open reactions_large file '%s'", path);
        *errmsg = strdup(buf);
        return 1;
    }

    out->entries = NULL;
    out->n = 0;
    size_t cap = 0;

    char line[512];
    int lineno = 0;
    int have_header = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char *s = trim(line);
        if (*s == '\0') continue;
        if (!have_header) { have_header = 1; continue; } /* "name,reactants,products,source,ref" */

        char *fields[5];
        if (csv_split(s, fields, 5) != 5) {
            char buf[4352];
            snprintf(buf, sizeof(buf), "%s:%d: expected 5 CSV fields, got '%s'", path, lineno, s);
            *errmsg = strdup(buf);
            fclose(f);
            cpr_reaction_table_free(out);
            return 1;
        }

        if (out->n == cap) {
            cap = cap ? cap * 2 : 128;
            out->entries = realloc(out->entries, cap * sizeof(CPRReactionEntry));
        }
        CPRReactionEntry *e = &out->entries[out->n++];
        snprintf(e->name, sizeof(e->name), "%s", fields[0]);
        snprintf(e->reactants, sizeof(e->reactants), "%s", fields[1]);
        snprintf(e->products, sizeof(e->products), "%s", fields[2]);
        snprintf(e->source, sizeof(e->source), "%s", fields[3]);
        snprintf(e->ref, sizeof(e->ref), "%s", fields[4]);
    }

    fclose(f);
    return 0;
}

void cpr_reaction_table_free(CPRReactionTable *t)
{
    free(t->entries);
    t->entries = NULL;
    t->n = 0;
}

/* ========================================================================
 * Phase 4 physics layer.
 * ========================================================================
 */

/* Max exponent passed to exp() in the detailed-balance reverse-rate
 * formula bwd = alpha*T9^beta*exp(gamma/T9)*fwd -- mirrors network_data.py's
 * _EXP_CAP (e^600 ~ 1e260, already inf at ~709 in double precision). */
static const double CPR_EXP_CAP = 600.0;
/* Forward-rate floor below which the reverse rate is forced to zero
 * (mirrors network_data.py's _FLOOR; just above the smallest denormal). */
static const double CPR_REVERSE_FLOOR = 1.0001e-35;

/* Historical MT-era reaction order (network_data.py's ORDER_MT, minus its
 * leading "n__p" -- that entry is always excluded from era selection by
 * construction, see cpr_select_era_reactions below, so omitting it here
 * avoids an unreachable array slot). MT always integrates this fixed
 * 17-reaction subset (intersected with whatever the selected network
 * actually has), even for network="large", because the full network is
 * too stiff to integrate before the deuterium bottleneck opens. */
static const char *CPR_ORDER_MT[] = {
    "Be7_d__a_a_p", "Be7_n__Li7_p", "Be7_n__a_a", "He3_a__Be7_g",
    "He3_d__a_p", "He3_n__t_p", "Li6_p__Be7_g", "Li7_p__a_a", "Li7_p__a_a_g",
    "d_a__Li6_g", "d_d__He3_n", "d_d__t_p", "d_p__He3_g", "n_p__d_g",
    "t_a__Li7_g", "t_d__a_n", "t_p__a_g",
};
#define CPR_N_ORDER_MT (sizeof(CPR_ORDER_MT) / sizeof(CPR_ORDER_MT[0]))

/* Stable light-nuclide species orders (network_data.py's SPECIES_SMALL/
 * SPECIES_MD): light species first, in this fixed physically-meaningful
 * order, then any remaining active nuclide in nuclides.csv's own order
 * (see cpr_species_order). */
static const char *CPR_SPECIES_SMALL[] = {"n", "p", "H2", "H3", "He3", "He4", "Li7", "Be7"};
#define CPR_N_SPECIES_SMALL (sizeof(CPR_SPECIES_SMALL) / sizeof(CPR_SPECIES_SMALL[0]))
static const char *CPR_SPECIES_MD[] = {"n", "p", "H2", "H3", "He3", "He4", "Li7", "Be7",
                                         "He6", "Li8", "Li6", "B8"};
#define CPR_N_SPECIES_MD (sizeof(CPR_SPECIES_MD) / sizeof(CPR_SPECIES_MD[0]))

/* Electric charge of each lepton bookkeeping token (A=0, not in the ODE
 * state vector). Bm = beta- = electron (Z=-1); Bp = beta+ = positron (Z=+1). */
static long lepton_Z(const char *tok)
{
    if (strcmp(tok, "Bm") == 0) return -1;
    if (strcmp(tok, "Bp") == 0) return +1;
    return 0;
}

static const CPRNuclide *find_nuclide(const CPRConfig *cfg, const char *name)
{
    for (size_t i = 0; i < cfg->nuclides.n; i++)
        if (strcmp(cfg->nuclides.items[i].name, name) == 0) return &cfg->nuclides.items[i];
    return NULL;
}

static const CPRReactionEntry *find_reaction_entry(const CPRReactionTable *rxn, const char *name)
{
    for (size_t i = 0; i < rxn->n; i++)
        if (strcmp(rxn->entries[i].name, name) == 0) return &rxn->entries[i];
    return NULL;
}

static const CPRDetailedBalanceEntry *find_db_entry(const CPRDetailedBalanceTable *db, const char *name)
{
    for (size_t i = 0; i < db->n; i++)
        if (strcmp(db->entries[i].reaction, name) == 0) return &db->entries[i];
    return NULL;
}

static const CPRDecayEntry *find_decay_entry(const CPRDecayTable *decays, const char *name)
{
    for (size_t i = 0; i < decays->n; i++)
        if (strcmp(decays->entries[i].name, name) == 0) return &decays->entries[i];
    return NULL;
}

/* Up to this many distinct nuclide species on one side of a parsed
 * "+"-joined reactant/product field (network_data.py's _side_counts) --
 * the heaviest large-network reactions have at most 4. */
#define CPR_SIDE_MAX 8

typedef struct {
    char names[CPR_SIDE_MAX][16];
    long counts[CPR_SIDE_MAX];
    size_t n;
} CPRSideCounts;

/* Parses one CSV side such as "Li7+p" into nuclide multiplicities,
 * dropping photons ('g') and accumulating the net lepton charge of any
 * Bm/Bp token into *lepton_dZ (port of network_data.py's _side_counts). */
static void side_counts(const char *field, CPRSideCounts *out, long *lepton_dZ)
{
    out->n = 0;
    *lepton_dZ = 0;
    char buf[64];
    snprintf(buf, sizeof(buf), "%s", field);
    char *save = NULL;
    char *tok = strtok_r(buf, "+", &save);
    while (tok) {
        long lz = lepton_Z(tok);
        if (lz != 0) {
            *lepton_dZ += lz;
        } else if (strcmp(tok, "g") != 0) {
            size_t i;
            for (i = 0; i < out->n; i++)
                if (strcmp(out->names[i], tok) == 0) break;
            if (i == out->n) {
                snprintf(out->names[out->n], sizeof(out->names[out->n]), "%s", tok);
                out->counts[out->n] = 0;
                out->n++;
            }
            out->counts[i] += 1;
        }
        tok = strtok_r(NULL, "+", &save);
    }
}

/* Reverse-rate coefficients from nuclide data alone (network_data.py's
 * compute_detailed_balance_coefficients; PRIMAT's GatherInfoReac/Qreaction/
 * PowerT9/FactorInverseReaction). See network_data.h's docstring for when
 * this is actually invoked (only cfg->decay_reverse_rates=True, default
 * False).
 *
 * Physics: equilibrium of the forward/backward Saha factors gives
 *   gamma = -Q/(kB*1e9 K)   (Q = energy released; gamma<0 for Q>0)
 *   beta  = 1.5*(n_react - n_prod)   (thermal-wavelength powers)
 *   alpha = [prod_i g_i*(M_i*kB/2pi)^-1.5 / m_i!] / [same for products] * units
 * with g=2J+1 the spin degeneracy, M the nuclear mass, m! the symmetry
 * factor for identical particles, and `units` the dimensional constant
 * making alpha dimensionless. */
int cpr_compute_detailed_balance_coefficients(const char * const *reactants, size_t n_react,
                                                const char * const *products, size_t n_prod,
                                                const CPRConfig *cfg,
                                                double *alpha, double *beta, double *gamma,
                                                char **errmsg)
{
    const double keV = g_const.keV, kB = g_const.kB;
    const double ma_e = g_const.ma * g_const.MeV; /* atomic mass unit [erg] */
    const double me_e = g_const.me * g_const.MeV; /* electron mass [erg] */

    /* Nuclear (not atomic) rest mass energy [erg]: A*m_u + excess - Z*m_e. */
    double mass_cache[16];
    const char *names[16];
    size_t n_names = 0;
    for (size_t k = 0; k < n_react; k++) names[n_names++] = reactants[k];
    for (size_t k = 0; k < n_prod; k++) names[n_names++] = products[k];
    for (size_t i = 0; i < n_names; i++) {
        const CPRNuclide *nuc = find_nuclide(cfg, names[i]);
        if (!nuc) {
            char buf[128];
            snprintf(buf, sizeof(buf), "compute_detailed_balance_coefficients: "
                      "unknown nuclide '%s'", names[i]);
            *errmsg = strdup(buf);
            return 1;
        }
        mass_cache[i] = (nuc->N + nuc->Z) * ma_e + nuc->mass_excess_keV * keV - nuc->Z * me_e;
    }
    /* binding(s) = N*ExcessMass(n) + Z*ExcessMass(p) - ExcessMass(s), in
     * keV (a plain number, units restored by *keV below); ExcessMass(n)
     * and ExcessMass(p) are the n/p rows themselves. */
    const CPRNuclide *nuc_n = find_nuclide(cfg, "n"), *nuc_p = find_nuclide(cfg, "p");
    double EX_n = nuc_n->mass_excess_keV, EX_p = nuc_p->mass_excess_keV;

    double n_in = (double)n_react, n_out = (double)n_prod;
    *beta = 1.5 * (n_in - n_out);

    double Q = 0.0; /* erg */
    for (size_t k = 0; k < n_prod; k++) {
        const CPRNuclide *nuc = find_nuclide(cfg, products[k]);
        Q += keV * (nuc->N * EX_n + nuc->Z * EX_p - nuc->mass_excess_keV);
    }
    for (size_t k = 0; k < n_react; k++) {
        const CPRNuclide *nuc = find_nuclide(cfg, reactants[k]);
        Q -= keV * (nuc->N * EX_n + nuc->Z * EX_p - nuc->mass_excess_keV);
    }
    *gamma = -Q / (kB * 1.0e9);

    /* quantum_factor(side) = prod over distinct species of
     * [(2J+1) * (M*kB*1e9/2pi)^1.5]^m / m! -- the spin-weighted quantum-
     * concentration factor, with the m! symmetry factor for m identical
     * particles. Distinct-species grouping mirrors Python's
     * collections.Counter(side).items(). */
    double qf_react = 1.0, qf_prod = 1.0;
    for (size_t side = 0; side < 2; side++) {
        const char * const *names_s = side == 0 ? reactants : products;
        size_t n_s = side == 0 ? n_react : n_prod;
        double *qf = side == 0 ? &qf_react : &qf_prod;
        size_t base = side == 0 ? 0 : n_react;
        char seen[16][16];
        size_t n_seen = 0;
        for (size_t k = 0; k < n_s; k++) {
            int dup = 0;
            for (size_t s = 0; s < n_seen; s++) if (strcmp(seen[s], names_s[k]) == 0) { dup = 1; break; }
            if (dup) continue;
            snprintf(seen[n_seen++], sizeof(seen[0]), "%s", names_s[k]);
            long m = 0;
            for (size_t k2 = 0; k2 < n_s; k2++) if (strcmp(names_s[k2], names_s[k]) == 0) m++;
            size_t idx = base + k;
            const CPRNuclide *nuc = find_nuclide(cfg, names_s[k]);
            double term = (2.0 * nuc->spin + 1.0) * pow(2.0 * M_PI / mass_cache[idx] / (kB * 1.0e9), -1.5);
            long mfact = 1;
            for (long f = 2; f <= m; f++) mfact *= f;
            *qf *= pow(term, (double)m) / (double)mfact;
        }
    }
    /* Dimensional constant per net particle that renders alpha
     * dimensionless (PRIMAT's FactorInverseReaction "Units"). */
    double units = pow((ma_e / (g_const.clight * g_const.clight)) / pow(g_const.hbar * g_const.clight, 3.0),
                        n_in - n_out);
    *alpha = qf_react / qf_prod * units;
    return 0;
}

/* QED correction factor for the five radiative-capture reactions affected
 * by pair-production in the final-state photon (Pitrou & Pospelov 2020;
 * port of network_data.py's _qed_nuclear_rescale -- see its docstring for
 * the full physics derivation). factor[] (length n_grid) is filled in
 * place; returns 1 if `name` is one of the five affected reactions, 0
 * (factor left untouched) otherwise. */
static int qed_nuclear_rescale(const char *name, const double *grid, size_t n_grid, double *factor)
{
    /* Fine-structure constant (CODATA 2018) and electron mass [MeV] (PDG) --
     * literal constants matching network_data.py's local ALPHA/ME_MEV
     * exactly (not g_const.alphaem/g_const.me, which are PDG values from a
     * possibly different edition; kept verbatim per CLAUDE.md's "port
     * faithfully" convention since this factor must reproduce Python's
     * numbers bit-for-bit). */
    const double ALPHA = 1.0 / 137.035999084;
    const double ME_MEV = 0.51099895;

    if (strcmp(name, "n_p__d_g") == 0) {
        /* Polynomial fit to the Gamow-peak/electric-dipole QED correction,
         * capped at its T9->0 (pair-threshold) limit. */
        const double T9_ZERO_LIMIT = 1.0009003934476768;
        for (size_t i = 0; i < n_grid; i++) {
            double T9 = grid[i];
            double poly = 1.0003328617393168
                + 0.00010013475534938917 * T9
                + 0.00004089993260910648 * T9 * T9
                - 0.000011824673537229535 * T9 * T9 * T9
                + 1.0522377796855455e-6 * T9 * T9 * T9 * T9;
            factor[i] = fmin(poly, T9_ZERO_LIMIT);
        }
        return 1;
    }

    /* Reactant (Z,A) pairs and Q-values [MeV] for the four Kroll electric-
     * dipole reactions (mass excesses from NUBASE2020, matching
     * nuclides.csv: n:8071.318, p:7288.971, H2:13135.723, H3:14949.811,
     * He3:14931.219, He4:2424.916, Li7:14907.105, Be7:15769.000 [keV]). */
    long Z1, A1, Z2, A2; double Q;
    if (strcmp(name, "d_p__He3_g") == 0) { Z1=1; A1=2; Z2=1; A2=1; Q=(13135.723+7288.971-14931.219)*1e-3; }
    else if (strcmp(name, "t_p__a_g") == 0) { Z1=1; A1=3; Z2=1; A2=1; Q=(14949.811+7288.971-2424.916)*1e-3; }
    else if (strcmp(name, "t_a__Li7_g") == 0) { Z1=1; A1=3; Z2=2; A2=4; Q=(14949.811+2424.916-14907.105)*1e-3; }
    else if (strcmp(name, "He3_a__Be7_g") == 0) { Z1=2; A1=3; Z2=2; A2=4; Q=(14931.219+2424.916-15769.000)*1e-3; }
    else return 0;

    const double pi = M_PI;
    for (size_t i = 0; i < n_grid; i++) {
        double T9 = grid[i];
        /* Most-likely kinetic energy at the Gamow peak [MeV] (Landau &
         * Lifshitz, via Pitrou & Pospelov 2020). */
        double EML = 0.1220 * pow((double)Z1, 2.0 / 3.0) * pow((double)Z2, 2.0 / 3.0)
                     * pow((double)(A1 * A2) / (double)(A1 + A2), 1.0 / 3.0) * pow(T9, 2.0 / 3.0);
        double Ea = (EML + Q) / ME_MEV; /* available energy in units of m_e */
        factor[i] = 1.0
            - (10.0 * ALPHA) / (9.0 * pi)
            + (2.0 * ALPHA * log(64.0)) / (9.0 * pi)
            - (ALPHA * log(4.0 / (Ea * Ea))) / (3.0 * pi)
            + (3.0 * ALPHA * (1.0 + log(4.0 * Ea * Ea))) / (4.0 * pi * Ea * Ea * Ea * Ea);
    }
    return 1;
}

/* Orders active nuclides with light species first (network_data.py's
 * _species_order): SPECIES_MD if any of its 4 heavier-than-SPECIES_SMALL
 * members (He6/Li8/Li6/B8) is active, else SPECIES_SMALL; then every
 * remaining active nuclide in cfg->nuclides' own (nuclides.csv) order.
 * `active` (length n_active) need not be sorted; `out` (caller-allocated,
 * length >= n_active) receives the ordered species names. Returns the
 * number of species written (== n_active). */
static int active_contains(char active[][16], size_t n_active, const char *name)
{
    for (size_t i = 0; i < n_active; i++) if (strcmp(active[i], name) == 0) return 1;
    return 0;
}

static size_t species_order(const CPRConfig *cfg, char active[][16], size_t n_active,
                              char out[][16])
{
    int use_md = 0;
    for (size_t i = 8; i < CPR_N_SPECIES_MD; i++)
        if (active_contains(active, n_active, CPR_SPECIES_MD[i])) { use_md = 1; break; }
    const char **base = use_md ? CPR_SPECIES_MD : CPR_SPECIES_SMALL;
    size_t n_base = use_md ? CPR_N_SPECIES_MD : CPR_N_SPECIES_SMALL;

    size_t n_out = 0;
    for (size_t i = 0; i < n_base; i++)
        if (active_contains(active, n_active, base[i])) snprintf(out[n_out++], 16, "%s", base[i]);
    for (size_t i = 0; i < cfg->nuclides.n; i++) {
        const char *name = cfg->nuclides.items[i].name;
        if (!active_contains(active, n_active, name)) continue;
        int already = 0;
        for (size_t k = 0; k < n_out; k++) if (strcmp(out[k], name) == 0) { already = 1; break; }
        if (!already) snprintf(out[n_out++], 16, "%s", name);
    }
    return n_out;
}

/* Caps each reaction's reverse rate at its value at T9 ~= T_nucl/1e9
 * (network_data.py's _reverse_rate_cap -- see its docstring: below T_nucl
 * the exp(gamma/T9) factor of exothermic reverse rates can grow by orders
 * of magnitude, an "exothermic blow-up" that would prevent BDF convergence
 * for heavy large-network nuclides; pinning the cap at T_nucl preserves
 * detailed balance near BBN onset and removes the low-T divergence). */
static double reverse_rate_cap_one(const double *grid, size_t n_grid, double alpha,
                                     double beta, double gamma, const double *fwd_row,
                                     const CPRConfig *cfg)
{
    double target = cpr_T_nucl() / 1.0e9; /* T9 at T_nucl (cpr_T_nucl() is in K) */
    /* searchsorted(grid, target): index of first grid value >= target. */
    size_t j = 0;
    while (j < n_grid && grid[j] < target) j++;
    if (j >= n_grid) j = n_grid - 1;
    (void)cfg;
    double T9c = grid[j];
    return alpha * pow(T9c, beta) * exp(fmin(gamma / T9c, CPR_EXP_CAP)) * fwd_row[j];
}

/* Generous fixed caps for cpr_load_network's local working arrays: the
 * `large` network has 428 reactions over ~59 species; both margins below
 * leave ample room (and amax/era filtering only ever shrinks these
 * counts). */
#define CPR_MAX_REACTIONS 600
#define CPR_MAX_LOCAL_SPECIES 128

static int add_to_set(char set[][16], size_t *n, const char *name)
{
    for (size_t i = 0; i < *n; i++) if (strcmp(set[i], name) == 0) return 0;
    snprintf(set[*n], 16, "%s", name);
    (*n)++;
    return 1;
}

static long species_index(char (*species)[16], size_t n_species, const char *name)
{
    for (size_t i = 0; i < n_species; i++) if (strcmp(species[i], name) == 0) return (long)i;
    return -1;
}

static int custom_is_removed(const CPRCustomNetwork *custom, const char *name)
{
    if (!custom) return 0;
    for (size_t i = 0; i < custom->n_removed; i++)
        if (strcmp(custom->removed[i], name) == 0) return 1;
    return 0;
}

static const CPRCustomTable *custom_find_table(const CPRCustomNetwork *custom, const char *name)
{
    if (!custom) return NULL;
    for (size_t i = 0; i < custom->n_tables; i++)
        if (strcmp(custom->tables[i].name, name) == 0) return &custom->tables[i];
    return NULL;
}

/* Maps PRIMAT's single-letter aliases (d/t/a) to nuclide names; any other
 * token (a full nuclide name, "g", "Bm", "Bp") passes through unchanged
 * (port of network_data.py's _ALIAS). */
static const char *alias_nuclide(const char *tok)
{
    if (strcmp(tok, "d") == 0) return "H2";
    if (strcmp(tok, "t") == 0) return "H3";
    if (strcmp(tok, "a") == 0) return "He4";
    return tok;
}

/* Builds a "+"-joined reactants/products field (reactions_large.csv's own
 * format, consumed by side_counts) from one "_"-joined side of a
 * "spaced"-syntax reaction name, e.g. "d_d" -> "H2+H2". */
static void build_side_field(const char *side, char *out, size_t outsize)
{
    out[0] = '\0';
    char buf[64];
    snprintf(buf, sizeof(buf), "%s", side);
    char *save = NULL;
    int first = 1;
    for (char *tok = strtok_r(buf, "_", &save); tok; tok = strtok_r(NULL, "_", &save)) {
        if (!first) strncat(out, "+", outsize - strlen(out) - 1);
        strncat(out, alias_nuclide(tok), outsize - strlen(out) - 1);
        first = 0;
    }
}

/* Derives an "added" reaction's reactants/products fields from its
 * "a_b__c_d" name (port of reaction_stoichiometry's "TO"-split fallback
 * path, restricted to the "spaced" syntax used by every GUI-generated name
 * -- see network_data.h's docstring). Returns 0 on success, nonzero with
 * *errmsg set if the name has no "__" separator. */
static int parse_reaction_name(const char *name, char *rfield, size_t rsize,
                                 char *pfield, size_t psize, char **errmsg)
{
    const char *sep = strstr(name, "__");
    if (!sep) {
        char buf[160];
        snprintf(buf, sizeof(buf), "cannot add reaction '%s': no '__' separator "
                  "(only the \"a_b__c_d\" syntax is supported)", name);
        *errmsg = strdup(buf);
        return 1;
    }
    char rside[64];
    size_t rlen = (size_t)(sep - name);
    if (rlen >= sizeof(rside)) rlen = sizeof(rside) - 1;
    memcpy(rside, name, rlen);
    rside[rlen] = '\0';
    build_side_field(rside, rfield, rsize);
    build_side_field(sep + 2, pfield, psize);
    return 0;
}

/* Extends `rxn_map`/`db` with any custom->tables entry whose name is absent
 * from the shipped catalog -- a brand-new "added" reaction (port of
 * network_data.py's _inject_custom_reactions). A name already present in
 * `rxn_map` ("replaced") is untouched here: only its forward rate is
 * overridden later, in cpr_load_network's per-reaction loop. Reverse-rate
 * coefficients are derived via cpr_compute_detailed_balance_coefficients
 * only for a purely nuclear addition (net lepton charge == 0); a weak
 * addition, or one whose nuclide data is incomplete, is left without a
 * `db` entry, defaulting to abg=(0,0,0) (forward-only) downstream --
 * mirroring Python's try/except-Exception: pass exactly. */
static int inject_custom_reactions(const CPRCustomNetwork *custom, CPRReactionTable *rxn_map,
                                     CPRDetailedBalanceTable *db, const CPRConfig *cfg,
                                     char **errmsg)
{
    if (!custom) return 0;
    for (size_t t = 0; t < custom->n_tables; t++) {
        const char *name = custom->tables[t].name;
        if (find_reaction_entry(rxn_map, name)) continue; /* "replaced": already catalogued */

        char rfield[64], pfield[64];
        if (parse_reaction_name(name, rfield, sizeof(rfield), pfield, sizeof(pfield), errmsg))
            return 1;

        rxn_map->entries = realloc(rxn_map->entries, (rxn_map->n + 1) * sizeof(CPRReactionEntry));
        CPRReactionEntry *e = &rxn_map->entries[rxn_map->n++];
        snprintf(e->name, sizeof(e->name), "%s", name);
        snprintf(e->reactants, sizeof(e->reactants), "%s", rfield);
        snprintf(e->products, sizeof(e->products), "%s", pfield);
        snprintf(e->source, sizeof(e->source), "custom");
        e->ref[0] = '\0';

        CPRSideCounts rs, ps; long ldz_r, ldz_p;
        side_counts(rfield, &rs, &ldz_r);
        side_counts(pfield, &ps, &ldz_p);
        if (ldz_p - ldz_r != 0) continue; /* weak addition: no reverse rate */

        const char *reactants[CPR_SIDE_MAX]; size_t nr = 0;
        const char *products[CPR_SIDE_MAX]; size_t np = 0;
        for (size_t k = 0; k < rs.n; k++)
            for (long m = 0; m < rs.counts[k]; m++) reactants[nr++] = rs.names[k];
        for (size_t k = 0; k < ps.n; k++)
            for (long m = 0; m < ps.counts[k]; m++) products[np++] = ps.names[k];
        double a, b, g; char *dberr = NULL;
        if (cpr_compute_detailed_balance_coefficients(reactants, nr, products, np, cfg,
                                                         &a, &b, &g, &dberr) == 0) {
            db->entries = realloc(db->entries, (db->n + 1) * sizeof(CPRDetailedBalanceEntry));
            CPRDetailedBalanceEntry *de = &db->entries[db->n++];
            snprintf(de->reaction, sizeof(de->reaction), "%s", name);
            de->Q_keV = 0.0; de->alpha = a; de->beta = b; de->gamma = g;
        } else {
            free(dberr); /* missing spin/mass data: forward-only fallback */
        }
    }
    return 0;
}

int cpr_load_network(const CPRConfig *cfg, const char *era,
                      const char * const *reaction_names, size_t n_reaction_names,
                      const CPRCustomNetwork *custom,
                      CPRNetworkDef *out, char **errmsg)
{
    memset(out, 0, sizeof(*out));
    char era_upper[4];
    snprintf(era_upper, sizeof(era_upper), "%s", era);
    for (char *p = era_upper; *p; p++) *p = (char)toupper((unsigned char)*p);
    int is_mt = (strcmp(era_upper, "MT") == 0);
    if (!is_mt && strcmp(era_upper, "LT") != 0) {
        *errmsg = strdup("cpr_load_network: era must be \"MT\" or \"LT\"");
        return 1;
    }

    /* ---- 1. Resolve the bare reaction-name list + table-file map. ---- */
    char bare_names[CPR_MAX_REACTIONS][64];
    char table_files[CPR_MAX_REACTIONS][128];
    size_t n_bare = 0;
    CPRNetworkList file_list = {0};
    int have_file_list = 0;
    if (reaction_names) {
        if (n_reaction_names > CPR_MAX_REACTIONS) {
            *errmsg = strdup("cpr_load_network: too many reactions (raise CPR_MAX_REACTIONS)");
            return 1;
        }
        for (size_t i = 0; i < n_reaction_names; i++) {
            /* Mirrors _parse_network_entries: an entry may itself carry a
             * ", filename" override even when passed directly. */
            char entry[192];
            snprintf(entry, sizeof(entry), "%s", reaction_names[i]);
            char *comma = strchr(entry, ',');
            if (comma) {
                *comma = '\0';
                char *name = trim(entry), *file = trim(comma + 1);
                snprintf(bare_names[n_bare], 64, "%s", name);
                snprintf(table_files[n_bare], 128, "%s", file);
            } else {
                char *name = trim(entry);
                snprintf(bare_names[n_bare], 64, "%s", name);
                snprintf(table_files[n_bare], 128, "%s_primat.txt", name);
            }
            n_bare++;
        }
    } else {
        char path[4300];
        char relpath[300];
        snprintf(relpath, sizeof(relpath), "nuclear/networks/%s.txt", cfg->network);
        cpr_config_resolve_rates_path(cfg, relpath, path, sizeof(path));
        if (cpr_load_network_list(path, &file_list, errmsg)) return 1;
        have_file_list = 1;
        if (file_list.n > CPR_MAX_REACTIONS) {
            *errmsg = strdup("cpr_load_network: too many reactions (raise CPR_MAX_REACTIONS)");
            cpr_network_list_free(&file_list);
            return 1;
        }
        for (size_t i = 0; i < file_list.n; i++) {
            snprintf(bare_names[n_bare], 64, "%s", file_list.entries[i].name);
            if (file_list.entries[i].table_file[0])
                snprintf(table_files[n_bare], 128, "%s", file_list.entries[i].table_file);
            else
                table_files[n_bare][0] = '\0'; /* decay: looked up in decays.txt, no table */
            n_bare++;
        }
        cpr_network_list_free(&file_list);
    }
    (void)have_file_list;

    /* ---- 1b. Drop custom->removed names (GUI "Customise Reactions" toggle-
     * off), mirroring UpdateNuclearRates.__init__'s `removed` set filter
     * applied before load_network is even called. ---- */
    if (custom && custom->n_removed) {
        size_t kept = 0;
        for (size_t i = 0; i < n_bare; i++) {
            if (custom_is_removed(custom, bare_names[i])) continue;
            if (kept != i) {
                snprintf(bare_names[kept], 64, "%s", bare_names[i]);
                snprintf(table_files[kept], 128, "%s", table_files[i]);
            }
            kept++;
        }
        n_bare = kept;
    }

    /* ---- 2. Load the reaction catalog (reactions_large.csv, detailed_balance.csv). ---- */
    char base_dir[4200], tables_dir[4200];
    snprintf(base_dir, sizeof(base_dir), "%s/csv", cfg->data_dir);
    snprintf(tables_dir, sizeof(tables_dir), "%s/nuclear/tables", cfg->data_dir);
    char rxn_path[4300];
    snprintf(rxn_path, sizeof(rxn_path), "%s/reactions_large.csv", base_dir);
    CPRReactionTable rxn_map;
    if (cpr_load_reactions_large(rxn_path, &rxn_map, errmsg)) return 1;
    char db_path[4300];
    snprintf(db_path, sizeof(db_path), "%s/detailed_balance.csv", base_dir);
    CPRDetailedBalanceTable db;
    if (cpr_load_detailed_balance(db_path, &db, errmsg)) { cpr_reaction_table_free(&rxn_map); return 1; }

    /* ---- 2b. Inject custom->tables' "added" reactions into rxn_map/db, then
     * append any not already selected to bare_names -- both must happen
     * before the amax filter (step 3) so an added reaction is amax-filtered
     * exactly like a shipped one (mirrors UpdateNuclearRates.__init__
     * appending added_names to self._selected_names before calling
     * load_network). ---- */
    if (inject_custom_reactions(custom, &rxn_map, &db, cfg, errmsg)) {
        cpr_reaction_table_free(&rxn_map); cpr_detailed_balance_free(&db);
        return 1;
    }
    if (custom) {
        for (size_t t = 0; t < custom->n_tables; t++) {
            const char *name = custom->tables[t].name;
            int already = 0;
            for (size_t i = 0; i < n_bare; i++)
                if (strcmp(bare_names[i], name) == 0) { already = 1; break; }
            if (already) continue;
            if (n_bare >= CPR_MAX_REACTIONS) {
                *errmsg = strdup("cpr_load_network: too many reactions (raise CPR_MAX_REACTIONS)");
                cpr_reaction_table_free(&rxn_map); cpr_detailed_balance_free(&db);
                return 1;
            }
            snprintf(bare_names[n_bare], 64, "%s", name);
            table_files[n_bare][0] = '\0'; /* unused: custom_find_table always wins in step 9 */
            n_bare++;
        }
    }

    /* ---- 3. amax filter (any positive cfg->amax; -1 = None/disabled). ---- */
    char filtered[CPR_MAX_REACTIONS][64];
    size_t n_filtered = 0;
    for (size_t i = 0; i < n_bare; i++) {
        if (cfg->amax < 0) { snprintf(filtered[n_filtered++], 64, "%s", bare_names[i]); continue; }
        const CPRReactionEntry *e = find_reaction_entry(&rxn_map, bare_names[i]);
        if (!e) {
            char buf[160];
            snprintf(buf, sizeof(buf), "reaction '%s' is not present in reactions_large.csv", bare_names[i]);
            *errmsg = strdup(buf);
            cpr_reaction_table_free(&rxn_map); cpr_detailed_balance_free(&db);
            return 1;
        }
        CPRSideCounts react, prod; long ldz;
        side_counts(e->reactants, &react, &ldz);
        side_counts(e->products, &prod, &ldz);
        long max_A = 0;
        for (size_t k = 0; k < react.n; k++) {
            const CPRNuclide *nuc = find_nuclide(cfg, react.names[k]);
            if (nuc && nuc->N + nuc->Z > max_A) max_A = nuc->N + nuc->Z;
        }
        for (size_t k = 0; k < prod.n; k++) {
            const CPRNuclide *nuc = find_nuclide(cfg, prod.names[k]);
            if (nuc && nuc->N + nuc->Z > max_A) max_A = nuc->N + nuc->Z;
        }
        if (max_A <= cfg->amax) snprintf(filtered[n_filtered++], 64, "%s", bare_names[i]);
    }

    /* ---- 4. Era selection. ---- */
    char selected[CPR_MAX_REACTIONS][64];
    size_t n_selected = 0;
    if (is_mt) {
        /* Intersect with the fixed historical MT order, for every network
         * including "small": for the shipped (uncustomised) small.txt this
         * intersection reproduces `filtered` unchanged (all 12 of its
         * thermonuclear reactions are in CPR_ORDER_MT already), but doing
         * the intersection unconditionally -- rather than special-casing
         * "small" to skip it -- also correctly drops a custom-added
         * reaction not in ORDER_MT from the MT era, mirroring Python's
         * _select_era_reactions (which filters against ORDER_SMALL for
         * "small", a subset of ORDER_MT in the same relative order). */
        for (size_t k = 0; k < CPR_N_ORDER_MT; k++)
            for (size_t i = 0; i < n_filtered; i++)
                if (strcmp(filtered[i], CPR_ORDER_MT[k]) == 0) {
                    snprintf(selected[n_selected++], 64, "%s", CPR_ORDER_MT[k]);
                    break;
                }
    } else {
        for (size_t i = 0; i < n_filtered; i++)
            snprintf(selected[n_selected++], 64, "%s", filtered[i]);
    }

    /* ---- 5. Parse each selected reaction's stoichiometry; collect active species. ---- */
    char active[CPR_MAX_LOCAL_SPECIES][16];
    size_t n_active = 0;
    add_to_set(active, &n_active, "n");
    add_to_set(active, &n_active, "p");

    CPRSideCounts *react_sides = malloc(n_selected * sizeof(CPRSideCounts));
    CPRSideCounts *prod_sides = malloc(n_selected * sizeof(CPRSideCounts));
    long *net_lepton_dZ = malloc(n_selected * sizeof(long));
    int *is_weak = malloc(n_selected * sizeof(int));
    char (*sel_table_file)[128] = malloc(n_selected * sizeof(*sel_table_file));

    for (size_t i = 0; i < n_selected; i++) {
        const CPRReactionEntry *e = find_reaction_entry(&rxn_map, selected[i]);
        if (!e) {
            char buf[160];
            snprintf(buf, sizeof(buf), "reaction '%s' is not present in reactions_large.csv", selected[i]);
            *errmsg = strdup(buf);
            free(react_sides); free(prod_sides); free(net_lepton_dZ); free(is_weak); free(sel_table_file);
            cpr_reaction_table_free(&rxn_map); cpr_detailed_balance_free(&db);
            return 1;
        }
        long ldz_r, ldz_p;
        side_counts(e->reactants, &react_sides[i], &ldz_r);
        side_counts(e->products, &prod_sides[i], &ldz_p);
        net_lepton_dZ[i] = ldz_p - ldz_r;
        is_weak[i] = (net_lepton_dZ[i] != 0);
        /* Default filename, same fallback as step 1 (an explicit
         * comma-supplied filename from the network-list/override always
         * wins via this lookup against `bare_names`/`table_files`). */
        const char *deffile = NULL;
        for (size_t b = 0; b < n_bare; b++)
            if (strcmp(bare_names[b], selected[i]) == 0) { deffile = table_files[b]; break; }
        if (deffile && deffile[0]) snprintf(sel_table_file[i], 128, "%s", deffile);
        else snprintf(sel_table_file[i], 128, "%s_primat.txt", selected[i]);

        for (size_t k = 0; k < react_sides[i].n; k++) add_to_set(active, &n_active, react_sides[i].names[k]);
        for (size_t k = 0; k < prod_sides[i].n; k++) add_to_set(active, &n_active, prod_sides[i].names[k]);
    }

    /* ---- 6. MT-era species extension (network_data.py's _extend_mt_species). ---- */
    if (is_mt && strcmp(cfg->network, "small") != 0) {
        char file_nuclides[CPR_MAX_LOCAL_SPECIES][16];
        size_t n_fn = 0;
        add_to_set(file_nuclides, &n_fn, "n");
        add_to_set(file_nuclides, &n_fn, "p");
        for (size_t i = 0; i < n_filtered; i++) {
            const CPRReactionEntry *e = find_reaction_entry(&rxn_map, filtered[i]);
            if (!e) continue;
            CPRSideCounts r, p; long ldz;
            side_counts(e->reactants, &r, &ldz); side_counts(e->products, &p, &ldz);
            for (size_t k = 0; k < r.n; k++) add_to_set(file_nuclides, &n_fn, r.names[k]);
            for (size_t k = 0; k < p.n; k++) add_to_set(file_nuclides, &n_fn, p.names[k]);
        }
        for (size_t k = 0; k < CPR_N_SPECIES_MD; k++) {
            const char *s = CPR_SPECIES_MD[k];
            if (!active_contains(file_nuclides, n_fn, s)) continue;
            const CPRNuclide *nuc = find_nuclide(cfg, s);
            if (cfg->amax >= 0 && nuc && nuc->N + nuc->Z > cfg->amax) continue;
            add_to_set(active, &n_active, s);
        }
    }

    /* ---- 7. Species ordering, N/Z arrays. ---- */
    char ordered[CPR_MAX_LOCAL_SPECIES][16];
    size_t n_species = species_order(cfg, active, n_active, ordered);
    out->species = malloc(n_species * sizeof(*out->species));
    out->N = malloc(n_species * sizeof(long));
    out->Z = malloc(n_species * sizeof(long));
    out->n_species = n_species;
    for (size_t i = 0; i < n_species; i++) {
        snprintf(out->species[i], 16, "%s", ordered[i]);
        const CPRNuclide *nuc = find_nuclide(cfg, ordered[i]);
        out->N[i] = nuc ? nuc->N : 0;
        out->Z[i] = nuc ? nuc->Z : 0;
    }
    /* ---- 8. Master T9 grid. ---- */
    size_t n_grid = (size_t)cfg->rate_grid_npts;
    out->n_grid = n_grid;
    out->grid = malloc(n_grid * sizeof(double));
    {
        double lo = log10(cfg->rate_grid_T9_min), hi = log10(cfg->rate_grid_T9_max);
        for (size_t i = 0; i < n_grid; i++) {
            double frac = (n_grid == 1) ? 0.0 : (double)i / (double)(n_grid - 1);
            out->grid[i] = pow(10.0, lo + frac * (hi - lo));
        }
    }

    /* ---- 9. Build per-reaction stoichiometry, names, rate tables. ---- */
    size_t n_reac = 1 + n_selected; /* index 0 = n__p */
    out->n_reac = n_reac;
    out->names = malloc(n_reac * sizeof(*out->names));
    out->sources = calloc(n_reac, sizeof(*out->sources)); /* populated per reaction below */
    out->network = malloc(n_reac * sizeof(CPRReaction));
    out->weak_flags = malloc(n_reac * sizeof(int));
    out->lepton_dZ = malloc(n_reac * sizeof(long));

    snprintf(out->names[0], 64, "n__p");
    /* n__p source: rates come from the background weak-rate computation, not a
     * rate-table file -- leave blank, matching Python's NetworkDefinition.sources[0]. */
    out->sources[0][0] = '\0';
    memset(&out->network[0], 0, sizeof(CPRReaction));
    out->network[0].reactants.n = 1;
    out->network[0].reactants.species_idx[0] = species_index(out->species, n_species, "n");
    out->network[0].reactants.mult[0] = 1;
    out->network[0].products.n = 1;
    out->network[0].products.species_idx[0] = species_index(out->species, n_species, "p");
    out->network[0].products.mult[0] = 1;
    out->weak_flags[0] = 1;
    out->lepton_dZ[0] = -1; /* n -> p + e- (Bm emitted): balances nuclear dZ=+1 */

    out->fwd = malloc(n_selected * n_grid * sizeof(double));
    out->fwd_median = malloc(n_selected * n_grid * sizeof(double));
    out->fwd_expsigma = malloc(n_selected * n_grid * sizeof(double));
    out->abg = malloc(n_selected * 3 * sizeof(double));
    out->bwd_cap = malloc(n_selected * sizeof(double));

    CPRDecayTable decays = {0};
    int have_decays = 0;
    int load_fail = 0;
    char load_errbuf[256] = {0};

    for (size_t i = 0; i < n_selected && !load_fail; i++) {
        size_t ridx = i + 1;
        snprintf(out->names[ridx], 64, "%s", selected[i]);
        out->weak_flags[ridx] = is_weak[i];
        out->lepton_dZ[ridx] = net_lepton_dZ[i];

        CPRReaction *rx = &out->network[ridx];
        memset(rx, 0, sizeof(*rx));
        rx->reactants.n = react_sides[i].n;
        for (size_t k = 0; k < react_sides[i].n; k++) {
            rx->reactants.species_idx[k] = species_index(out->species, n_species, react_sides[i].names[k]);
            rx->reactants.mult[k] = react_sides[i].counts[k];
        }
        rx->products.n = prod_sides[i].n;
        for (size_t k = 0; k < prod_sides[i].n; k++) {
            rx->products.species_idx[k] = species_index(out->species, n_species, prod_sides[i].names[k]);
            rx->products.mult[k] = prod_sides[i].counts[k];
        }

        double *fwd_row = &out->fwd_median[i * n_grid];
        double *err_row = &out->fwd_expsigma[i * n_grid];

        const CPRCustomTable *ct = custom_find_table(custom, selected[i]);
        if (ct) {
            /* GUI override ("replaced" or "added") wins even for a weak/
             * decay reaction -- it is resampled like any other table, not
             * broadcast as a constant (mirrors network_data.py's
             * _build_rate_tables: the custom_tables branch is checked
             * before the is_weak/decay branch). */
            if (cpr_resample_rate_table(ct->T9, ct->rate, ct->n, out->grid, fwd_row, n_grid, errmsg) ||
                cpr_resample_rate_table(ct->T9, ct->err, ct->n, out->grid, err_row, n_grid, errmsg)) {
                load_fail = 1; break;
            }
            snprintf(out->sources[ridx], 64, "custom");
        } else if (is_weak[i]) {
            /* Radioactive decay: T9-independent rate from decays.txt,
             * broadcast onto the master grid (no rate table to resample). */
            if (!have_decays) {
                char decays_path[4300];
                snprintf(decays_path, sizeof(decays_path), "%s/decays.txt", tables_dir);
                if (cpr_load_decays(decays_path, &decays, errmsg)) { load_fail = 1; break; }
                have_decays = 1;
            }
            const CPRDecayEntry *de = find_decay_entry(&decays, selected[i]);
            if (!de) {
                snprintf(load_errbuf, sizeof(load_errbuf),
                          "decay reaction '%s' has no entry in decays.txt", selected[i]);
                load_fail = 1; break;
            }
            for (size_t g = 0; g < n_grid; g++) { fwd_row[g] = de->rate_s_inv; err_row[g] = de->uncertainty; }
            snprintf(out->sources[ridx], 64, "%s", de->ref[0] ? de->ref : "?");
        } else {
            char table_path[4500];
            char table_relpath[600];
            snprintf(table_relpath, sizeof(table_relpath), "nuclear/tables/%s/%s",
                     selected[i], sel_table_file[i]);
            cpr_config_resolve_rates_path(cfg, table_relpath, table_path, sizeof(table_path));
            CPRTable tab;
            if (cpr_table_read(table_path, 3, &tab, errmsg)) { load_fail = 1; break; }
            if (cpr_resample_rate_table(tab.cols[0], tab.cols[1], tab.n_rows, out->grid, fwd_row, n_grid, errmsg) ||
                cpr_resample_rate_table(tab.cols[0], tab.cols[2], tab.n_rows, out->grid, err_row, n_grid, errmsg)) {
                cpr_table_free(&tab); load_fail = 1; break;
            }
            cpr_table_free(&tab);
            read_reaction_source(table_path, out->sources[ridx], sizeof(*out->sources));
        }

        /* Detailed-balance (alpha, beta, gamma): catalog value if present,
         * else (decays only) from-scratch via compute_detailed_balance_
         * coefficients when cfg->decay_reverse_rates, else (0,0,0) --
         * decays are irreversible at BBN temperatures by default. */
        double *abg_row = &out->abg[i * 3];
        const CPRDetailedBalanceEntry *dbe = find_db_entry(&db, selected[i]);
        if (dbe) {
            abg_row[0] = dbe->alpha; abg_row[1] = dbe->beta; abg_row[2] = dbe->gamma;
        } else if (is_weak[i] && cfg->decay_reverse_rates) {
            const char *react_names[CPR_SIDE_MAX]; long react_n[CPR_SIDE_MAX];
            const char *prod_names[CPR_SIDE_MAX]; long prod_n[CPR_SIDE_MAX];
            size_t nr = 0, np = 0;
            for (size_t k = 0; k < react_sides[i].n; k++)
                for (long m = 0; m < react_sides[i].counts[k]; m++) react_names[nr++] = react_sides[i].names[k];
            for (size_t k = 0; k < prod_sides[i].n; k++)
                for (long m = 0; m < prod_sides[i].counts[k]; m++) prod_names[np++] = prod_sides[i].names[k];
            (void)react_n; (void)prod_n;
            double a, b, g; char *dberr = NULL;
            if (cpr_compute_detailed_balance_coefficients(react_names, nr, prod_names, np, cfg,
                                                             &a, &b, &g, &dberr) == 0 && g < 0.0) {
                abg_row[0] = a; abg_row[1] = b; abg_row[2] = g;
            } else {
                free(dberr);
                abg_row[0] = abg_row[1] = abg_row[2] = 0.0;
            }
        } else {
            abg_row[0] = abg_row[1] = abg_row[2] = 0.0;
        }
    }

    free(react_sides); free(prod_sides); free(net_lepton_dZ); free(is_weak); free(sel_table_file);
    cpr_decay_table_free(&decays);
    cpr_reaction_table_free(&rxn_map); cpr_detailed_balance_free(&db);

    if (load_fail) {
        if (load_errbuf[0] && !*errmsg) *errmsg = strdup(load_errbuf);
        cpr_network_def_free(out);
        return 1;
    }

    /* ---- 10. Nuclear QED rescale, reverse-rate cap. ---- */
    if (cfg->nuclear_qed_corrections) {
        double factor[CPR_MAX_REACTIONS]; /* reused per-reaction; sized for grid below */
        (void)factor;
        double *fac = malloc(n_grid * sizeof(double));
        for (size_t i = 0; i < n_selected; i++) {
            if (qed_nuclear_rescale(selected[i], out->grid, n_grid, fac)) {
                double *row = &out->fwd_median[i * n_grid];
                for (size_t g = 0; g < n_grid; g++) row[g] *= fac[g];
            }
        }
        free(fac);
    }
    memcpy(out->fwd, out->fwd_median, n_selected * n_grid * sizeof(double));
    for (size_t i = 0; i < n_selected; i++) {
        out->bwd_cap[i] = reverse_rate_cap_one(out->grid, n_grid, out->abg[i * 3], out->abg[i * 3 + 1],
                                                 out->abg[i * 3 + 2], &out->fwd[i * n_grid], cfg);
    }

    out->buf = malloc(2 * n_reac * sizeof(double));
    out->cache_valid = 0;
    return 0;
}

void cpr_network_def_free(CPRNetworkDef *net)
{
    free(net->species); free(net->N); free(net->Z);
    free(net->network); free(net->names); free(net->sources); free(net->weak_flags); free(net->lepton_dZ);
    free(net->grid);
    free(net->fwd); free(net->fwd_median); free(net->fwd_expsigma); free(net->abg); free(net->bwd_cap);
    free(net->buf);
    memset(net, 0, sizeof(*net));
}

void cpr_network_apply_variations(CPRNetworkDef *net, const CPRConfig *cfg)
{
    size_t n_grid = net->n_grid;
    for (size_t i = 1; i < net->n_reac; i++) { /* skip names[0] == n__p */
        size_t row = i - 1;
        double p = cpr_rxnmap_get(&cfg->p_rxn, net->names[i]);
        double delta = cpr_rxnmap_get(&cfg->delta_rxn, net->names[i]);
        double *fwd_row = &net->fwd[row * n_grid];
        const double *median_row = &net->fwd_median[row * n_grid];
        /* variation = exp(p * log(sigma)) + delta; baseline p=0,delta=0 → 1.0.
         * delta is a direct fractional additive shift (delta=0.1 → +10%).
         * cfg->rescale_nuclear_rates is kept for backward compat but no longer
         * gates delta; any nonzero delta always applies. */
        if (p == 0.0 && delta == 0.0) {
            memcpy(fwd_row, median_row, n_grid * sizeof(double));
        } else {
            const double *sigma_row = &net->fwd_expsigma[row * n_grid];
            double cap = cfg->mc_rate_rescale_cap; /* 0.0 = no cap */
            for (size_t g = 0; g < n_grid; g++) {
                double variation = exp(p * log(sigma_row[g])) + delta;
                /* Clamp to [1/cap, cap] when a cap is set (cap > 0), to prevent
                 * unphysically extreme rescalings for poorly-constrained rates. */
                if (cap > 0.0) {
                    if (variation > cap) variation = cap;
                    else if (variation < 1.0 / cap) variation = 1.0 / cap;
                }
                fwd_row[g] = median_row[g] * variation;
            }
        }
    }
    net->cache_valid = 0; /* the active fwd table changed underneath any cached buf */
}

const double *cpr_network_fill_buffer(CPRNetworkDef *net, double T_t_K,
                                        double nTOp_frwrd, double nTOp_bkwrd, int clamp)
{
    if (net->cache_valid && T_t_K == net->cache_T_t && clamp == net->cache_clamp)
        return net->buf;

    double *r = net->buf;
    r[0] = nTOp_frwrd;
    r[1] = nTOp_bkwrd;

    double T9 = T_t_K * 1.0e-9;
    const double *g = net->grid;
    size_t n_grid = net->n_grid;
    /* searchsorted(g, T9) - 1, clamped to [0, n_grid-2] (mirrors fill_buffer's
     * own clamp exactly). */
    size_t i = 0;
    while (i < n_grid && g[i] <= T9) i++;
    long ii = (long)i - 1;
    if (ii < 0) ii = 0;
    else if (ii > (long)n_grid - 2) ii = (long)n_grid - 2;
    double w = (T9 - g[ii]) / (g[ii + 1] - g[ii]);

    size_t n_thermo = net->n_reac - 1;
    for (size_t k = 0; k < n_thermo; k++) {
        double fwd = net->fwd[k * n_grid + (size_t)ii] * (1.0 - w) + net->fwd[k * n_grid + (size_t)ii + 1] * w;
        double alpha = net->abg[k * 3], beta = net->abg[k * 3 + 1], gamma = net->abg[k * 3 + 2];
        double bwd = alpha * pow(T9, beta) * exp(fmin(gamma / T9, CPR_EXP_CAP)) * fwd;
        if (fwd <= CPR_REVERSE_FLOOR) bwd = 0.0;
        if (bwd < 0.0) bwd = 0.0; /* reverse rate is physically non-negative */
        if (clamp && bwd > net->bwd_cap[k]) bwd = net->bwd_cap[k];
        r[2 + 2 * k] = fwd;
        r[3 + 2 * k] = bwd;
    }

    net->cache_T_t = T_t_K;
    net->cache_clamp = clamp;
    net->cache_valid = 1;
    return r;
}

int cpr_nuclear_rates_init(CPRNuclearRates *nr, const CPRConfig *cfg,
                             const CPRCustomNetwork *custom, char **errmsg)
{
    memset(nr, 0, sizeof(*nr));
    if (cpr_load_network(cfg, "MT", NULL, 0, custom, &nr->mt_net, errmsg)) return 1;
    if (cpr_load_network(cfg, "LT", NULL, 0, custom, &nr->lt_net, errmsg)) {
        cpr_network_def_free(&nr->mt_net);
        return 1;
    }
    cpr_network_apply_variations(&nr->mt_net, cfg);
    cpr_network_apply_variations(&nr->lt_net, cfg);

    cpr_compile_network(nr->mt_net.network, nr->mt_net.n_reac, nr->mt_net.n_species, &nr->mt_compiled);
    cpr_compile_network(nr->lt_net.network, nr->lt_net.n_reac, nr->lt_net.n_species, &nr->lt_compiled);

    size_t mt_n_weak = 0, lt_n_weak = 0;
    size_t *mt_weak_idx = malloc(nr->mt_net.n_reac * sizeof(size_t));
    size_t *lt_weak_idx = malloc(nr->lt_net.n_reac * sizeof(size_t));
    for (size_t i = 0; i < nr->mt_net.n_reac; i++) if (nr->mt_net.weak_flags[i]) mt_weak_idx[mt_n_weak++] = i;
    for (size_t i = 0; i < nr->lt_net.n_reac; i++) if (nr->lt_net.weak_flags[i]) lt_weak_idx[lt_n_weak++] = i;

    int rc = cpr_check_conservation(&nr->mt_compiled, nr->mt_net.N, nr->mt_net.Z,
                                      mt_weak_idx, mt_n_weak, nr->mt_net.lepton_dZ, errmsg);
    if (!rc)
        rc = cpr_check_conservation(&nr->lt_compiled, nr->lt_net.N, nr->lt_net.Z,
                                      lt_weak_idx, lt_n_weak, nr->lt_net.lepton_dZ, errmsg);
    free(mt_weak_idx); free(lt_weak_idx);
    if (rc) { cpr_nuclear_rates_free(nr); return 1; }
    return 0;
}

void cpr_nuclear_rates_free(CPRNuclearRates *nr)
{
    cpr_network_def_free(&nr->mt_net);
    cpr_network_def_free(&nr->lt_net);
    cpr_compiled_network_free(&nr->mt_compiled);
    cpr_compiled_network_free(&nr->lt_compiled);
}

void cpr_nuclear_rates_apply_variations(CPRNuclearRates *nr, const CPRConfig *cfg)
{
    cpr_network_apply_variations(&nr->mt_net, cfg);
    cpr_network_apply_variations(&nr->lt_net, cfg);
}

void cpr_nuclear_rates_rhs_mt(CPRNuclearRates *nr, const double *Y, double T_t_K,
                                double rhoBBN, double nTOp_frwrd, double nTOp_bkwrd, double *dY)
{
    const double *r = cpr_network_fill_buffer(&nr->mt_net, T_t_K, nTOp_frwrd, nTOp_bkwrd, 0);
    cpr_network_rhs(&nr->mt_compiled, Y, rhoBBN, r, dY);
}

void cpr_nuclear_rates_jac_mt(CPRNuclearRates *nr, const double *Y, double T_t_K,
                                double rhoBBN, double nTOp_frwrd, double nTOp_bkwrd, double *J)
{
    const double *r = cpr_network_fill_buffer(&nr->mt_net, T_t_K, nTOp_frwrd, nTOp_bkwrd, 0);
    cpr_network_jacobian(&nr->mt_compiled, Y, rhoBBN, r, J);
}

void cpr_nuclear_rates_rhs_lt(CPRNuclearRates *nr, const double *Y, double T_t_K,
                                double rhoBBN, double nTOp_frwrd, double nTOp_bkwrd, double *dY)
{
    const double *r = cpr_network_fill_buffer(&nr->lt_net, T_t_K, nTOp_frwrd, nTOp_bkwrd, 1);
    cpr_network_rhs(&nr->lt_compiled, Y, rhoBBN, r, dY);
}

void cpr_nuclear_rates_jac_lt(CPRNuclearRates *nr, const double *Y, double T_t_K,
                                double rhoBBN, double nTOp_frwrd, double nTOp_bkwrd, double *J)
{
    const double *r = cpr_network_fill_buffer(&nr->lt_net, T_t_K, nTOp_frwrd, nTOp_bkwrd, 1);
    cpr_network_jacobian(&nr->lt_compiled, Y, rhoBBN, r, J);
}
