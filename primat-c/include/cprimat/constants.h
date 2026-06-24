/* constants.h -- frozen physical constants and unit-conversion factors.
 *
 * Direct port of pyprimat/constants.py's `Constants` dataclass: the base
 * fields are literal PDG values (verbatim, no computation differs from
 * Python), and the derived quantities are implemented as functions of
 * `g_const` rather than dataclass `@property`s, since C has no lazy
 * per-instance property mechanism and these are pure functions of fixed
 * constants anyway (cheap to recompute, never needs caching).
 *
 * Units convention (unchanged from Python): natural units throughout
 * (Kelvin = second = cm = gram = 1); the `cpr_MeV_to_*` functions convert
 * *to* CGS only where needed.
 */
#ifndef CPRIMAT_CONSTANTS_H
#define CPRIMAT_CONSTANTS_H

typedef struct {
    /* ---- CGS base units (dimensionless by convention: natural units) ---- */
    double Kelvin, second, cm, gram;

    /* ---- Fundamental constants (PDG) ---- */
    double kB;      /* Boltzmann constant [erg/K] */
    double clight;  /* speed of light [cm/s] */
    double hbar;    /* Planck constant [erg s] */
    double Mpc;     /* megaparsec [cm] */
    double MeV;     /* 1 MeV [erg] */
    double keV;     /* 1 keV [erg] */

    /* ---- Electroweak sector (PDG) ---- */
    double alphaem; /* fine-structure constant */
    double GF;      /* Fermi constant [MeV^-2] */
    double mZ;      /* Z boson mass [MeV] */

    /* ---- Fermion masses [MeV] (PDG) ---- */
    double me, mn, mp;

    /* ---- CMB ---- */
    double T0CMB;   /* photon temperature today [K] */

    /* ---- Standard-model effective neutrino number ---- */
    double Neff_SM; /* 3 instantaneous-decoupling flavours + NEVO/QED heating corrections */

    /* ---- Weak-rate nuclear-structure constants (PDG) ---- */
    double gA;        /* nucleon axial coupling */
    double kappa_p;    /* proton anomalous magnetic moment */
    double kappa_n;    /* neutron anomalous magnetic moment */
    double Vud;        /* CKM matrix element |V_ud| */
    double radproton;  /* proton charge radius [cm] */

    /* ---- Atomic masses ---- */
    double ma;         /* 1 unified atomic mass unit [MeV] */
    double He4Overma;  /* M(He4) / u */
    double HOverma;    /* M(H) / u */
} CPRConstants;

/* The single frozen instance, populated at startup by cpr_constants_init(). */
extern CPRConstants g_const;

/* Fills g_const with the literal values above. Idempotent; call once at
 * program start (or let any module needing g_const call it -- cheap). */
void cpr_constants_init(void);

/* ---- Derived quantities (pure functions of g_const) ---- */
double cpr_erg(void);
double cpr_MeV_to_Kelvin(void);
double cpr_MeV_to_secm1(void);
double cpr_MeV_to_g(void);
double cpr_MeV_to_cmm1(void);
double cpr_MeV4_to_gcmm3(void);

/* ---- Fixed temperature eras [K] ---- */
double cpr_T_start(void);  /* 10 MeV */
double cpr_T_weak(void);   /* 1 MeV */
double cpr_T_nucl(void);   /* 0.11 MeV */

/* ---- Electroweak mixing angle and effective couplings ---- */
double cpr_sW2(void);
double cpr_geL(void);
double cpr_geR(void);
double cpr_gmuL(void);
double cpr_gmuR(void);
double cpr_deltakappa(void);

/* ---- High-T plasma entropy/number-density normalisations ---- */
double cpr_s0bar(void);
double cpr_s0CMB(void);
double cpr_n0CMB(void);

/* ---- Mean baryon mass (H + He4 mixture) ---- */
double cpr_mB(void);
double cpr_maOvermB(void);

/* ---- Hubble constant in natural units, per unit h ---- */
double cpr_HubbleOverh(void);

#endif /* CPRIMAT_CONSTANTS_H */
