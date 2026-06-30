/* network_builder.h -- stoichiometry-driven nuclear-network RHS/Jacobian
 * (port of primat/network_builder.py: compile_network, the rhs/jacobian
 * kernels, check_conservation).
 *
 * A BBN run integrates one ODE per nuclide, coupled through hundreds of
 * reactions. Rather than hand-writing the algebra of every reaction, a
 * network is described *abstractly* as a list of reactions, each a pair of
 * {species_index: multiplicity} sides, and compiled once (cpr_compile_network)
 * into flat arrays that two small kernels (cpr_network_rhs/cpr_network_jacobian)
 * evaluate for *any* network -- the same code path serves the 8-species
 * "small" network and the ~59-species "large" one.
 *
 * Mass-action kinetics: each reaction i has reactants with multiplicities
 * c_s^react and products with c_s^prod. The net flux is
 *
 *   F_forward  = r[2i]   * rho^(R-1) / sym(reactants) * prod_s Y_s^(c_s^react)
 *   F_backward = r[2i+1] * rho^(P-1) / sym(products)  * prod_s Y_s^(c_s^prod)
 *
 * and every nuclide's equation accumulates its net stoichiometric share:
 *
 *   dY_s/dt += (c_s^prod - c_s^react) * (F_forward - F_backward).
 *
 * R = sum_s c_s^react, P = sum_s c_s^prod (rho^(R-1)/rho^(P-1) convert the
 * rate-per-baryon-density convention into a rate-per-Y); sym(side) =
 * prod_s (c_s!) is the symmetry factor avoiding double counting identical
 * reactants/products (e.g. d+d carries 1/2!). `r` is the flat
 * forward/backward rate buffer filled by cpr_network_fill_buffer
 * (network_data.h/.c).
 */
#ifndef CPRIMAT_NETWORK_BUILDER_H
#define CPRIMAT_NETWORK_BUILDER_H

#include <stddef.h>

/* One side of one reaction: up to CPR_STOICH_MAX_TERMS distinct species,
 * each with an integer multiplicity. No primat reaction (small or large)
 * has more than 4 distinct species on either side; 8 is a generous margin. */
#define CPR_STOICH_MAX_TERMS 8

typedef struct {
    long species_idx[CPR_STOICH_MAX_TERMS]; /* index into the abundance vector Y */
    long mult[CPR_STOICH_MAX_TERMS];
    size_t n;
} CPRStoichSide;

typedef struct {
    CPRStoichSide reactants, products;
} CPRReaction;

/* Flat-array encoding of a network's *topology* (who reacts with whom and
 * with what stoichiometry), built once per network at setup and read --
 * never modified -- by the kernels on every solver step. Ragged
 * per-reaction data (reactions range from 1-body decays to 3-body
 * reactions) is flattened into row-major (n_rx x width) arrays padded with
 * zeros, plus a companion `*_len` array giving the valid column count per
 * row -- a fixed-shape, allocation-free-per-step layout for the hot-path
 * kernels.
 *
 * Worked example -- "d + d -> He4 + g" with species order [n, p, d, He4]
 * (photons untracked), as reaction row i:
 *   ri_idx[i] = [2],    ri_pow[i] = [2], ri_len[i] = 1   (two deuterons)
 *   pi_idx[i] = [3],    pi_pow[i] = [1], pi_len[i] = 1   (one He4)
 *   af_idx[i] = [2, 3], af_co[i] = [-2, +1], af_len[i] = 2  (d: -2, He4: +1)
 *   vr_idx[i] = [2, 3], vr_len[i] = 2   (flux depends on d, He4)
 *   Rm1[i] = 1, Pm1[i] = 0, invsr[i] = 1/2! = 0.5, invsp[i] = 1
 */
typedef struct {
    size_t n_rx, n_sp;
    size_t MR, MP, MA, MV; /* row widths: max reactant/product/net-update/jacobian-var terms */

    long *ri_idx, *ri_pow; size_t *ri_len; /* reactant monomial, (n_rx x MR) row-major */
    long *pi_idx, *pi_pow; size_t *pi_len; /* product monomial,  (n_rx x MP) row-major */
    long *af_idx;   double *af_co; size_t *af_len; /* net update: dY[af_idx]+=af_co*flux, (n_rx x MA) */
    long *vr_idx;   size_t *vr_len;        /* union(reactants,products): jacobian columns, (n_rx x MV) */

    double *Rm1, *Pm1;     /* R-1, P-1 (density powers), length n_rx */
    double *invsr, *invsp; /* 1/sym(reactants), 1/sym(products), length n_rx */
} CPRCompiledNetwork;

/* Compiles `reactions` (n_rx entries, each a CPRReaction with species
 * indices in [0, n_sp)) into a CPRCompiledNetwork. Run once per network at
 * setup. Caller must cpr_compiled_network_free the result. */
void cpr_compile_network(const CPRReaction *reactions, size_t n_rx, size_t n_sp,
                           CPRCompiledNetwork *out);
void cpr_compiled_network_free(CPRCompiledNetwork *cn);

/* dY/dt of the whole network at one (Y, rho, r). `r` is the flat
 * forward/backward rate buffer (r[2i] forward, r[2i+1] backward for
 * reaction i; length 2*cn->n_rx). `dY` (length cn->n_sp) is overwritten
 * (not accumulated). This is the inner loop called on every solver
 * evaluation -- see this header's module docstring for the mass-action
 * formula it implements. */
void cpr_network_rhs(const CPRCompiledNetwork *cn, const double *Y, double rho,
                       const double *r, double *dY);

/* Analytic Jacobian J[s,u] = d(dY_s/dt)/dY_u, row-major (cn->n_sp x
 * cn->n_sp), overwritten (not accumulated). Derivation: F_i = cf*M_react(Y)
 * - cb*M_prod(Y) with cf/cb the Y-independent forward/backward prefactors
 * and M_side = prod_s Y_s^(c_s) a monomial; differentiating a monomial by
 * the power rule, d/dY[u] prod_s Y_s^(c_s) = p * Y[u]^(p-1) * prod_{s!=u}
 * Y_s^(c_s) where p is u's power in that monomial (0 if absent). Only the
 * species in vr_idx[i] (union of both monomials) are differentiated, which
 * is what keeps this affordable for the ~59-species large network. */
void cpr_network_jacobian(const CPRCompiledNetwork *cn, const double *Y, double rho,
                            const double *r, double *J);

/* Verifies the assembled stoichiometry conserves baryon number and electric
 * charge -- an exact integer check (no floating point, no test
 * abundances): per reaction, the net coefficients (+products-reactants) are
 * dotted with each species' N and Z.
 *
 *   A = N+Z must be conserved by every reaction (nucleons are conserved).
 *   Z (electric charge) must be conserved by every reaction, including weak
 *     ones -- the n<->p weak rate emits an electron (A=0, Z=-1) accounted
 *     for via lepton_dZ rather than appearing in the ODE species.
 *   N must be conserved by every *nuclear* (strong/EM) reaction; weak
 *     processes convert n<->p (N changes by +-1) and are exempt -- listed
 *     in weak_indices (length n_weak, indices into [0, cn->n_rx)).
 *
 * lepton_dZ (length cn->n_rx, may be NULL to fall back to the legacy A-only
 * check for weak reactions): net electric charge carried by
 * emitted/absorbed leptons per reaction, e.g. -1 for n__p (Bm/electron
 * emitted, balancing the nuclear dZ=+1 from n->p).
 *
 * Returns 0 if the network conserves N/Z, nonzero with *errmsg set (caller
 * frees, naming the violating reactions) otherwise -- a violation means the
 * reaction list is physically inconsistent and should never be integrated. */
int cpr_check_conservation(const CPRCompiledNetwork *cn, const long *N, const long *Z,
                            const size_t *weak_indices, size_t n_weak,
                            const long *lepton_dZ, char **errmsg);

#endif /* CPRIMAT_NETWORK_BUILDER_H */
