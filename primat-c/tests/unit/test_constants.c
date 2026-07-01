/* test_constants.c -- pins constants.c's *derived* electroweak/unit-
 * conversion values against an independent hand-computation, mirroring
 * ../../tests/test_constants.py's checks on the Python side
 * (primat.constants.CONST). alphaem/GF/mZ are primary PDG inputs (set
 * verbatim in both languages); sW2 (sin^2(theta_W)) and the effective
 * electron/muon couplings derived from it go through a formula that a
 * stray factor of 2 or a swapped GF/mZ would break silently -- it would
 * just shift every weak rate that uses sW2 by a few percent, not raise
 * any error. Pinning the formula here, independently of constants.c's own
 * implementation, catches that class of bug immediately instead of as an
 * unexplained drift in Neff/YP much later.
 */
#include "constants.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

static int close_rel(double a, double b, double rtol)
{
    return fabs(a - b) <= rtol * fabs(b) + 1e-300;
}

int main(void)
{
    cpr_constants_init();

    /* sin^2(theta_W) = 1/2 * (1 - sqrt(1 - 2*sqrt(2)*pi*alphaem/(GF*mZ^2))),
     * the on-shell relation -- see test_constants.py's docstring. */
    double expected_sW2 = 0.5 * (1.0 - sqrt(1.0 - 2.0 * sqrt(2.0) * M_PI
                                  * g_const.alphaem / (g_const.GF * g_const.mZ * g_const.mZ)));
    CHECK(close_rel(cpr_sW2(), expected_sW2, 1e-12), "sW2 matches on-shell relation");
    /* Sanity check against the well-known PDG ballpark (~0.223 in MSbar;
     * the on-shell scheme used here is close but not identical). */
    CHECK(cpr_sW2() > 0.20 && cpr_sW2() < 0.24, "sW2 is in the expected ballpark");

    CHECK(close_rel(cpr_geL(), 0.5 + cpr_sW2(), 1e-12), "geL == 0.5 + sW2");
    CHECK(close_rel(cpr_geR(), cpr_sW2(), 1e-12), "geR == sW2");
    CHECK(close_rel(cpr_gmuL(), -0.5 + cpr_sW2(), 1e-12), "gmuL == -0.5 + sW2");
    CHECK(close_rel(cpr_gmuR(), cpr_sW2(), 1e-12), "gmuR == sW2");

    /* T_weak/T_nucl are MeV_to_Kelvin scaled by their defining MeV values. */
    CHECK(close_rel(cpr_T_weak(), 1.0 * cpr_MeV_to_Kelvin(), 1e-12),
          "T_weak == 1.0 MeV in Kelvin");
    CHECK(close_rel(cpr_T_nucl(), 0.11 * cpr_MeV_to_Kelvin(), 1e-12),
          "T_nucl == 0.11 MeV in Kelvin");

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
