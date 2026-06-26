#include "cprimat/constants.h"
#include <math.h>

/* Riemann zeta(3) (Apery's constant), needed by cpr_n0CMB() below. Python
 * gets this from scipy.special.zeta(3); libm has no zeta function, so the
 * literal (17 significant digits, well beyond double precision) is the
 * simplest faithful port. */
#define ZETA3 1.2020569031595942854
#ifndef M_PI
#  define M_PI  3.141592653589793238462643383279502884
#endif

CPRConstants g_const;

void cpr_constants_init(void)
{
    g_const.Kelvin = 1.;
    g_const.second = 1.;
    g_const.cm     = 1.;
    g_const.gram   = 1.;

    g_const.kB     = 1.380649e-16;
    g_const.clight = 2.99792458e+10;
    g_const.hbar   = 6.62607015 / (2. * M_PI) * 1e-27;
    g_const.Mpc    = 3.08567758149e+24;
    g_const.MeV    = 1.602176634e-6;
    g_const.keV    = 1.602176634e-9;

    g_const.alphaem = 1. / 137.035999084;
    g_const.GF      = 1.1663787e-5 * 1.e-6;
    g_const.mZ      = 91.1876e3;

    g_const.me = 0.51099895;
    g_const.mn = 939.56542052;
    g_const.mp = 938.27208816;

    g_const.T0CMB = 2.7255;
    g_const.Neff_SM = 3.044;

    g_const.gA        = 1.2756;
    g_const.kappa_p   = 2.79284734463 - 1.;
    g_const.kappa_n   = -1.91304273;
    g_const.Vud       = 0.9738;
    g_const.radproton = 0.8409e-13;

    g_const.ma        = 931.494061;
    g_const.He4Overma = 4.0026032541;
    g_const.HOverma   = 1.00782503223;
}

double cpr_erg(void)
{
    return g_const.gram * g_const.cm * g_const.cm / g_const.second;
}

double cpr_MeV_to_Kelvin(void) { return g_const.MeV / g_const.kB; }
double cpr_MeV_to_secm1(void)  { return g_const.MeV / g_const.hbar; }
double cpr_MeV_to_g(void)      { return g_const.MeV / (g_const.clight * g_const.clight); }
double cpr_MeV_to_cmm1(void)   { return g_const.MeV / (g_const.hbar * g_const.clight); }

double cpr_MeV4_to_gcmm3(void)
{
    double cmm1 = cpr_MeV_to_cmm1();
    return cpr_MeV_to_g() * cmm1 * cmm1 * cmm1;
}

double cpr_T_start(void) { return 10.0 * cpr_MeV_to_Kelvin(); }
double cpr_T_weak(void)  { return 1.0 * cpr_MeV_to_Kelvin(); }
double cpr_T_nucl(void)  { return 0.11 * cpr_MeV_to_Kelvin(); }

double cpr_sW2(void)
{
    /* On-shell relation: sin^2(theta_W) from GF, mZ, alphaem. */
    return 0.5 * (1. - sqrt(1. - 2. * sqrt(2.) * M_PI * g_const.alphaem
                             / (g_const.GF * g_const.mZ * g_const.mZ)));
}

double cpr_geL(void) { return 0.5 + cpr_sW2(); }
double cpr_geR(void) { return cpr_sW2(); }
double cpr_gmuL(void) { return -0.5 + cpr_sW2(); }
double cpr_gmuR(void) { return cpr_sW2(); }
double cpr_deltakappa(void) { return g_const.kappa_p - g_const.kappa_n; }

double cpr_s0bar(void)
{
    /* Relativistic boson gas, g=2 (photon): s_gamma = (4 pi^2/45) T^3
     * (Phys. Rep. Eq. 24). */
    return 4. * M_PI * M_PI / 45.;
}

double cpr_s0CMB(void)
{
    double t = g_const.T0CMB / cpr_MeV_to_Kelvin();
    return cpr_s0bar() * t * t * t;
}

double cpr_n0CMB(void)
{
    /* n_gamma = (2 zeta(3)/pi^2) T^3 for a bosonic gas with g=2 (photon). */
    double t = g_const.T0CMB / cpr_MeV_to_Kelvin();
    return (2. * ZETA3) / (M_PI * M_PI) * t * t * t;
}

double cpr_mB(void)
{
    /* Mean baryon mass [MeV] for a 24.7% He4 mass-fraction mixture with H. */
    const double percentHe = 24.7 / 100.;
    return ((1. - percentHe) * g_const.HOverma
            + percentHe * g_const.He4Overma / 4.) * g_const.ma;
}

double cpr_maOvermB(void) { return g_const.ma / cpr_mB(); }

double cpr_HubbleOverh(void)
{
    /* 100 km/s/Mpc converted to natural (MeV) units via the cm/s/Mpc chain. */
    return (100. * (1.e+5 * g_const.cm * cpr_MeV_to_cmm1()))
           / (g_const.second * cpr_MeV_to_secm1())
           / (g_const.Mpc * cpr_MeV_to_cmm1());
}
