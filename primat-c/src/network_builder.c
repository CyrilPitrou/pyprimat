/* network_builder.c -- see cprimat/network_builder.h. */
#include "cprimat/network_builder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long factorial_l(long n)
{
    long f = 1;
    for (long k = 2; k <= n; k++) f *= k;
    return f;
}

/* Index helper for the row-major (n_rx x width) padded arrays. */
static inline size_t at(size_t row, size_t col, size_t width) { return row * width + col; }

void cpr_compile_network(const CPRReaction *reactions, size_t n_rx, size_t n_sp,
                           CPRCompiledNetwork *out)
{
    memset(out, 0, sizeof(*out));
    out->n_rx = n_rx;
    out->n_sp = n_sp;

    /* First pass: derive the per-reaction net-update and jacobian-variable
     * sets (sized at most 2*CPR_STOICH_MAX_TERMS each, since they are built
     * from the union/merge of the two CPR_STOICH_MAX_TERMS-capped sides),
     * and the row widths needed to pad everything. */
    size_t MR = 1, MP = 1, MA = 1, MV = 1;
    long  (*af_idx_tmp)[2 * CPR_STOICH_MAX_TERMS] = malloc(n_rx ? n_rx * sizeof(*af_idx_tmp) : sizeof(*af_idx_tmp));
    double(*af_co_tmp)[2 * CPR_STOICH_MAX_TERMS]  = malloc(n_rx ? n_rx * sizeof(*af_co_tmp)  : sizeof(*af_co_tmp));
    size_t *af_len_tmp = calloc(n_rx ? n_rx : 1, sizeof(size_t));
    long  (*vr_idx_tmp)[2 * CPR_STOICH_MAX_TERMS] = malloc(n_rx ? n_rx * sizeof(*vr_idx_tmp) : sizeof(*vr_idx_tmp));
    size_t *vr_len_tmp = calloc(n_rx ? n_rx : 1, sizeof(size_t));

    for (size_t i = 0; i < n_rx; i++) {
        const CPRReaction *rx = &reactions[i];

        /* Net stoichiometric change per species: +product, -reactant
         * (mirrors compile_network's `net` dict, built incrementally so a
         * species appearing on both sides nets out, e.g. a catalyst). */
        long net_idx[2 * CPR_STOICH_MAX_TERMS];
        long net_co[2 * CPR_STOICH_MAX_TERMS];
        size_t n_net = 0;
        for (size_t k = 0; k < rx->reactants.n; k++) {
            long s = rx->reactants.species_idx[k];
            long found = -1;
            for (size_t m = 0; m < n_net; m++) if (net_idx[m] == s) { found = (long)m; break; }
            if (found < 0) { net_idx[n_net] = s; net_co[n_net] = -rx->reactants.mult[k]; n_net++; }
            else net_co[found] -= rx->reactants.mult[k];
        }
        for (size_t k = 0; k < rx->products.n; k++) {
            long s = rx->products.species_idx[k];
            long found = -1;
            for (size_t m = 0; m < n_net; m++) if (net_idx[m] == s) { found = (long)m; break; }
            if (found < 0) { net_idx[n_net] = s; net_co[n_net] = rx->products.mult[k]; n_net++; }
            else net_co[found] += rx->products.mult[k];
        }
        size_t n_af = 0;
        for (size_t m = 0; m < n_net; m++) {
            if (net_co[m] != 0) {
                af_idx_tmp[i][n_af] = net_idx[m];
                af_co_tmp[i][n_af] = (double)net_co[m];
                n_af++;
            }
        }
        af_len_tmp[i] = n_af;

        /* vr_idx: union(reactant species, product species), sorted ascending
         * (mirrors `sorted(set(react) | set(prod))`) -- the only columns of
         * the Jacobian this reaction can contribute to. */
        size_t n_vr = 0;
        for (size_t k = 0; k < rx->reactants.n; k++) {
            long s = rx->reactants.species_idx[k];
            int dup = 0;
            for (size_t m = 0; m < n_vr; m++) if (vr_idx_tmp[i][m] == s) { dup = 1; break; }
            if (!dup) vr_idx_tmp[i][n_vr++] = s;
        }
        for (size_t k = 0; k < rx->products.n; k++) {
            long s = rx->products.species_idx[k];
            int dup = 0;
            for (size_t m = 0; m < n_vr; m++) if (vr_idx_tmp[i][m] == s) { dup = 1; break; }
            if (!dup) vr_idx_tmp[i][n_vr++] = s;
        }
        /* insertion sort (n_vr is tiny: at most 2*CPR_STOICH_MAX_TERMS) */
        for (size_t a = 1; a < n_vr; a++) {
            long v = vr_idx_tmp[i][a];
            size_t b = a;
            while (b > 0 && vr_idx_tmp[i][b - 1] > v) { vr_idx_tmp[i][b] = vr_idx_tmp[i][b - 1]; b--; }
            vr_idx_tmp[i][b] = v;
        }
        vr_len_tmp[i] = n_vr;

        if (rx->reactants.n > MR) MR = rx->reactants.n;
        if (rx->products.n > MP) MP = rx->products.n;
        if (n_af > MA) MA = n_af;
        if (n_vr > MV) MV = n_vr;
    }

    out->MR = MR; out->MP = MP; out->MA = MA; out->MV = MV;
    size_t alloc_rx = n_rx ? n_rx : 1; /* avoid zero-size malloc edge cases */

    out->ri_idx = calloc(alloc_rx * MR, sizeof(long));
    out->ri_pow = calloc(alloc_rx * MR, sizeof(long));
    out->ri_len = calloc(alloc_rx, sizeof(size_t));
    out->pi_idx = calloc(alloc_rx * MP, sizeof(long));
    out->pi_pow = calloc(alloc_rx * MP, sizeof(long));
    out->pi_len = calloc(alloc_rx, sizeof(size_t));
    out->af_idx = calloc(alloc_rx * MA, sizeof(long));
    out->af_co  = calloc(alloc_rx * MA, sizeof(double));
    out->af_len = calloc(alloc_rx, sizeof(size_t));
    out->vr_idx = calloc(alloc_rx * MV, sizeof(long));
    out->vr_len = calloc(alloc_rx, sizeof(size_t));
    out->Rm1 = calloc(alloc_rx, sizeof(double));
    out->Pm1 = calloc(alloc_rx, sizeof(double));
    out->invsr = calloc(alloc_rx, sizeof(double));
    out->invsp = calloc(alloc_rx, sizeof(double));

    for (size_t i = 0; i < n_rx; i++) {
        const CPRReaction *rx = &reactions[i];
        size_t R = 0, P = 0;
        long sr = 1, sp = 1;
        for (size_t k = 0; k < rx->reactants.n; k++) {
            out->ri_idx[at(i, k, MR)] = rx->reactants.species_idx[k];
            out->ri_pow[at(i, k, MR)] = rx->reactants.mult[k];
            R += (size_t)rx->reactants.mult[k];
            sr *= factorial_l(rx->reactants.mult[k]);
        }
        out->ri_len[i] = rx->reactants.n;
        for (size_t k = 0; k < rx->products.n; k++) {
            out->pi_idx[at(i, k, MP)] = rx->products.species_idx[k];
            out->pi_pow[at(i, k, MP)] = rx->products.mult[k];
            P += (size_t)rx->products.mult[k];
            sp *= factorial_l(rx->products.mult[k]);
        }
        out->pi_len[i] = rx->products.n;

        out->Rm1[i] = (double)R - 1.0;
        out->Pm1[i] = (double)P - 1.0;
        out->invsr[i] = 1.0 / (double)sr;
        out->invsp[i] = 1.0 / (double)sp;

        for (size_t k = 0; k < af_len_tmp[i]; k++) {
            out->af_idx[at(i, k, MA)] = af_idx_tmp[i][k];
            out->af_co[at(i, k, MA)] = af_co_tmp[i][k];
        }
        out->af_len[i] = af_len_tmp[i];
        for (size_t k = 0; k < vr_len_tmp[i]; k++)
            out->vr_idx[at(i, k, MV)] = vr_idx_tmp[i][k];
        out->vr_len[i] = vr_len_tmp[i];
    }

    free(af_idx_tmp); free(af_co_tmp); free(af_len_tmp);
    free(vr_idx_tmp); free(vr_len_tmp);
}

void cpr_compiled_network_free(CPRCompiledNetwork *cn)
{
    free(cn->ri_idx); free(cn->ri_pow); free(cn->ri_len);
    free(cn->pi_idx); free(cn->pi_pow); free(cn->pi_len);
    free(cn->af_idx); free(cn->af_co); free(cn->af_len);
    free(cn->vr_idx); free(cn->vr_len);
    free(cn->Rm1); free(cn->Pm1); free(cn->invsr); free(cn->invsp);
    memset(cn, 0, sizeof(*cn));
}

void cpr_network_rhs(const CPRCompiledNetwork *cn, const double *Y, double rho,
                       const double *r, double *dY)
{
    memset(dY, 0, cn->n_sp * sizeof(double));
    for (size_t i = 0; i < cn->n_rx; i++) {
        /* Forward flux: prefactor (rate * rho^(R-1) / sym_reactants) times
         * the reactant monomial prod_s Y_s^(c_s^react). */
        double Ff = r[2 * i] * pow(rho, cn->Rm1[i]) * cn->invsr[i];
        for (size_t k = 0; k < cn->ri_len[i]; k++)
            Ff *= pow(Y[cn->ri_idx[at(i, k, cn->MR)]], (double)cn->ri_pow[at(i, k, cn->MR)]);
        double Fb = r[2 * i + 1] * pow(rho, cn->Pm1[i]) * cn->invsp[i];
        for (size_t k = 0; k < cn->pi_len[i]; k++)
            Fb *= pow(Y[cn->pi_idx[at(i, k, cn->MP)]], (double)cn->pi_pow[at(i, k, cn->MP)]);
        double net = Ff - Fb;
        for (size_t k = 0; k < cn->af_len[i]; k++)
            dY[cn->af_idx[at(i, k, cn->MA)]] += cn->af_co[at(i, k, cn->MA)] * net;
    }
}

void cpr_network_jacobian(const CPRCompiledNetwork *cn, const double *Y, double rho,
                            const double *r, double *J)
{
    memset(J, 0, cn->n_sp * cn->n_sp * sizeof(double));
    for (size_t i = 0; i < cn->n_rx; i++) {
        double cf = r[2 * i] * pow(rho, cn->Rm1[i]) * cn->invsr[i];
        double cb = r[2 * i + 1] * pow(rho, cn->Pm1[i]) * cn->invsp[i];
        for (size_t vj = 0; vj < cn->vr_len[i]; vj++) {
            long u = cn->vr_idx[at(i, vj, cn->MV)];

            /* d(reactant monomial)/dY[u]: power rule, p*Y[u]^(p-1) times
             * every *other* reactant's power (0 if u is not a reactant). */
            long pu = 0;
            for (size_t k = 0; k < cn->ri_len[i]; k++)
                if (cn->ri_idx[at(i, k, cn->MR)] == u) pu = cn->ri_pow[at(i, k, cn->MR)];
            double dmr = 0.0;
            if (pu != 0) {
                dmr = (double)pu * pow(Y[u], (double)(pu - 1));
                for (size_t k = 0; k < cn->ri_len[i]; k++) {
                    long s = cn->ri_idx[at(i, k, cn->MR)];
                    if (s != u) dmr *= pow(Y[s], (double)cn->ri_pow[at(i, k, cn->MR)]);
                }
            }

            long pu2 = 0;
            for (size_t k = 0; k < cn->pi_len[i]; k++)
                if (cn->pi_idx[at(i, k, cn->MP)] == u) pu2 = cn->pi_pow[at(i, k, cn->MP)];
            double dmp = 0.0;
            if (pu2 != 0) {
                dmp = (double)pu2 * pow(Y[u], (double)(pu2 - 1));
                for (size_t k = 0; k < cn->pi_len[i]; k++) {
                    long s = cn->pi_idx[at(i, k, cn->MP)];
                    if (s != u) dmp *= pow(Y[s], (double)cn->pi_pow[at(i, k, cn->MP)]);
                }
            }

            double dnet = cf * dmr - cb * dmp;
            if (dnet == 0.0) continue;
            for (size_t k = 0; k < cn->af_len[i]; k++) {
                long s = cn->af_idx[at(i, k, cn->MA)];
                J[s * (long)cn->n_sp + u] += cn->af_co[at(i, k, cn->MA)] * dnet;
            }
        }
    }
}

int cpr_check_conservation(const CPRCompiledNetwork *cn, const long *N, const long *Z,
                            const size_t *weak_indices, size_t n_weak,
                            const long *lepton_dZ, char **errmsg)
{
    char buf[2048];
    size_t off = 0;
    int n_bad = 0;
    for (size_t i = 0; i < cn->n_rx; i++) {
        long dN = 0, dZ_nuc = 0;
        for (size_t k = 0; k < cn->af_len[i]; k++) {
            long s = cn->af_idx[at(i, k, cn->MA)];
            long c = (long)llround(cn->af_co[at(i, k, cn->MA)]);
            dN += c * N[s];
            dZ_nuc += c * Z[s];
        }
        long ldZ = lepton_dZ ? lepton_dZ[i] : 0;
        long dZ_total = dZ_nuc + ldZ;
        long dA = dN + dZ_nuc; /* leptons carry A=0 */

        int is_weak = 0;
        for (size_t w = 0; w < n_weak; w++) if (weak_indices[w] == i) { is_weak = 1; break; }

        int bad;
        if (is_weak) {
            bad = lepton_dZ ? (dA != 0 || dZ_total != 0) : (dA != 0);
        } else {
            bad = (dN != 0 || dZ_nuc != 0);
        }
        if (bad && n_bad < 5) {
            int n = snprintf(buf + off, sizeof(buf) - off,
                              "%s(%zu: dN=%ld, dZ=%ld)", n_bad ? ", " : "", i, dN, dZ_nuc);
            if (n > 0) off += (size_t)n;
        }
        if (bad) n_bad++;
    }
    if (n_bad) {
        char full[2200];
        snprintf(full, sizeof(full), "network violates N/Z conservation in %d reaction(s) "
                  "(index, dN, dZ): %s%s", n_bad, buf, n_bad > 5 ? "..." : "");
        *errmsg = strdup(full);
        return 1;
    }
    return 0;
}
