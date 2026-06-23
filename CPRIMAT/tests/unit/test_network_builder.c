/* test_network_builder.c -- checks cpr_compile_network/cpr_network_rhs/
 * cpr_network_jacobian/cpr_check_conservation against hand-derived values,
 * using the same d+d->He4+g worked example as network_builder.h's docstring,
 * plus a separate small conservation-only network. */
#include "cprimat/network_builder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

static int close(double a, double b) { return fabs(a - b) < 1e-9 * fabs(b) + 1e-300; }

int main(void)
{
    /* Species order [n, p, d, He4] = indices [0, 1, 2, 3]. One reaction:
     * d + d -> He4 + g (photon untracked), matching network_builder.h's
     * worked example exactly. */
    CPRReaction rx;
    memset(&rx, 0, sizeof(rx));
    rx.reactants.n = 1; rx.reactants.species_idx[0] = 2; rx.reactants.mult[0] = 2; /* 2 d */
    rx.products.n = 1;  rx.products.species_idx[0] = 3;  rx.products.mult[0] = 1;  /* 1 He4 */

    CPRCompiledNetwork cn;
    cpr_compile_network(&rx, 1, 4, &cn);

    CHECK(cn.n_rx == 1 && cn.n_sp == 4, "compiled network has 1 reaction, 4 species");
    CHECK(close(cn.Rm1[0], 1.0), "Rm1 = R-1 = 1 (two reactants)");
    CHECK(close(cn.Pm1[0], 0.0), "Pm1 = P-1 = 0 (one product)");
    CHECK(close(cn.invsr[0], 0.5), "invsr = 1/2! (two identical deuterons)");
    CHECK(close(cn.invsp[0], 1.0), "invsp = 1/1! (one He4)");
    CHECK(cn.af_len[0] == 2, "net update touches 2 species (d, He4)");
    CHECK(cn.vr_len[0] == 2, "jacobian variables: d and He4");

    /* RHS at Y = [0, 0, 2, 0], rho=1, r=[1,0] (forward rate 1, backward 0):
     * Ff = 1 * rho^1 * 0.5 * Y[d]^2 = 0.5*4 = 2; dY[d] = -2*Ff = -4;
     * dY[He4] = +1*Ff = 2. */
    double Y[4] = {0.0, 0.0, 2.0, 0.0};
    double r[2] = {1.0, 0.0};
    double dY[4];
    cpr_network_rhs(&cn, Y, 1.0, r, dY);
    CHECK(close(dY[0], 0.0) && close(dY[1], 0.0), "rhs: n, p unaffected");
    CHECK(close(dY[2], -4.0), "rhs: dY[d] = -4 (two d consumed per flux unit)");
    CHECK(close(dY[3], 2.0), "rhs: dY[He4] = +2");

    /* Jacobian at the same point: dFf/dY[d] = cf*2*Y[d] = 1*0.5*2*2 = 2;
     * J[d,d] = -2*2 = -4; J[He4,d] = +1*2 = 2; all other entries 0
     * (Fb=0 identically since r[1]=0, so no He4-column contributions). */
    double J[16];
    cpr_network_jacobian(&cn, Y, 1.0, r, J);
    CHECK(close(J[2 * 4 + 2], -4.0), "jacobian: d(dY[d])/dY[d] = -4");
    CHECK(close(J[3 * 4 + 2], 2.0), "jacobian: d(dY[He4])/dY[d] = +2");
    CHECK(close(J[0 * 4 + 0], 0.0) && close(J[2 * 4 + 3], 0.0),
          "jacobian: unrelated entries are 0");

    /* Finite-difference cross-check of the full Jacobian against the
     * analytic one, at a generic (non-degenerate) abundance point. */
    double Y2[4] = {0.3, 0.5, 0.7, 0.2};
    double r2[2] = {3.0, 1.5};
    double Janalytic[16];
    cpr_network_jacobian(&cn, Y2, 2.0, r2, Janalytic);
    double h = 1e-6;
    int fd_ok = 1;
    for (int u = 0; u < 4; u++) {
        double Yp[4], Ym[4], dYp[4], dYm[4];
        memcpy(Yp, Y2, sizeof(Yp)); memcpy(Ym, Y2, sizeof(Ym));
        Yp[u] += h; Ym[u] -= h;
        cpr_network_rhs(&cn, Yp, 2.0, r2, dYp);
        cpr_network_rhs(&cn, Ym, 2.0, r2, dYm);
        for (int s = 0; s < 4; s++) {
            double fd = (dYp[s] - dYm[s]) / (2.0 * h);
            if (fabs(fd - Janalytic[s * 4 + u]) > 1e-5 * (fabs(fd) + 1.0)) fd_ok = 0;
        }
    }
    CHECK(fd_ok, "analytic jacobian matches finite-difference rhs derivative");

    cpr_compiled_network_free(&cn);

    /* ---- check_conservation ---- */
    /* Nuclear reaction n + p -> d (NOT the weak n<->p bookkeeping entry):
     * N=[1,0,1], Z=[0,1,1] for species [n, p, d]; conserves both N and Z. */
    CPRReaction np;
    memset(&np, 0, sizeof(np));
    np.reactants.n = 2;
    np.reactants.species_idx[0] = 0; np.reactants.mult[0] = 1; /* n */
    np.reactants.species_idx[1] = 1; np.reactants.mult[1] = 1; /* p */
    np.products.n = 1; np.products.species_idx[0] = 2; np.products.mult[0] = 1; /* d */
    CPRCompiledNetwork cn2;
    cpr_compile_network(&np, 1, 3, &cn2);
    long N3[3] = {1, 0, 1}, Z3[3] = {0, 1, 1};
    char *err = NULL;
    CHECK(cpr_check_conservation(&cn2, N3, Z3, NULL, 0, NULL, &err) == 0,
          "n+p->d conserves N and Z (nuclear reaction)");
    cpr_compiled_network_free(&cn2);

    /* Weak n__p bookkeeping reaction: n -> p (N changes by -1, exempted via
     * weak_indices), with lepton_dZ=-1 (emitted electron) balancing the
     * nuclear dZ=+1 from n(Z=0)->p(Z=1). */
    CPRReaction weak;
    memset(&weak, 0, sizeof(weak));
    weak.reactants.n = 1; weak.reactants.species_idx[0] = 0; weak.reactants.mult[0] = 1;
    weak.products.n = 1;  weak.products.species_idx[0] = 1;  weak.products.mult[0] = 1;
    CPRCompiledNetwork cn3;
    cpr_compile_network(&weak, 1, 2, &cn3);
    long N2[2] = {1, 0}, Z2[2] = {0, 1};
    size_t weak_idx[1] = {0};
    long lepton_dZ[1] = {-1};
    CHECK(cpr_check_conservation(&cn3, N2, Z2, weak_idx, 1, lepton_dZ, &err) == 0,
          "n__p conserves A and total charge once lepton_dZ is accounted for");

    /* Same reaction but NOT listed as weak: must be flagged, since N is not
     * conserved by a "nuclear" reaction. */
    int rc = cpr_check_conservation(&cn3, N2, Z2, NULL, 0, NULL, &err);
    CHECK(rc != 0, "n->p without weak_indices is correctly flagged as N-violating");
    if (rc != 0) free(err);
    cpr_compiled_network_free(&cn3);

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
