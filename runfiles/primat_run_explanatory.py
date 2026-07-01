# -*- coding: utf-8 -*-
"""
primat_run_explanatory.py
==========================
Minimal, heavily-commented template for a standalone BBN run. Copy this file
and uncomment/edit whichever options you need; every option shown below is at
its default value, so running this file unmodified reproduces the standard
run (see CLAUDE.md's "Validation before committing" table for the expected
YPBBN/D-H reference values).

Run from the repo root so that the shipped ``data/`` data resolve correctly:

    python runfiles/primat_run_explanatory.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primat.backend import run_bbn  # , run_mc, dump_mc_samples (see bottom of file)

cfg = dict(
    # Every key below is shown at its DEFAULT_PARAMS default (see
    # primat/config.py for the authoritative, more detailed comments this
    # file summarises); uncomment and edit whichever you need to override.
    # All 73 DEFAULT_PARAMS keys are listed, grouped exactly as in config.py.

    # ---- general behaviour and numerical settings -------------------------
    # verbose=False,                  # print primat's own progress messages
    # debug=False,                    # print extra debug messages
    # numerical_precision=1e-7,       # rtol for all solve_ivp calls
    # numba_installed=True,           # re-checked at runtime; enables JIT kernels if available

    # ---- neutrino decoupling -----------------------------------------------
    # incomplete_decoupling=True,     # True: NEVO non-instantaneous decoupling table; False: instantaneous decoupling

    # ---- electromagnetic plasma --------------------------------------------
    # QED_corrections=True,           # QED corrections to the EM plasma equation of state
    # n_electron_table=2000,          # grid points for the electron-thermo tables
    # recompute_electron_thermo=False,  # force recomputation of the electron-thermo cache
    # recompute_qed_corrections=False,  # force recomputation (and overwrite) of data/plasma/QED_*.txt

    # ---- spectral distortions ----------------------------------------------
    # spectral_distortions=True,      # n<->p rate corrections from non-Fermi-Dirac neutrino spectra
    # analytic_distortions=False,     # True: analytic y_SZ/y_gray distortion instead of the NEVO spectral table
    # y_SZ=0.0,                       # amplitude of the y-type (Compton/SZ-like) distortion
    # y_gray=0.0,                     # amplitude of the gray-type (temperature-rescaling) distortion

    # ---- custom NEVO tables (None = shipped default) -----------------------
    # nevo_file=None,                 # override the 6/7-column thermo table
    # nevo_spectral_file=None,        # override the 86-column spectral-distortion table
    # nevo_grid_file=None,            # override the y-grid for nevo_spectral_file
    # nevo_file_prefix="NEVOPRIMAT",  # base filename for the default NEVO thermo/spectral tables

    # ---- data directory override and nuclear overlay ----------------------
    # data_dir=None,          # replace the entire primat/data/ tree (NEVO/, weak/, plasma/, nuclear/, csv/)
    # user_nuclear_dir=None,  # additive overlay for nuclear networks & rate tables only (primat/data/nuclear/ equivalent)

    # ---- background mode ----------------------------------------------------
    # external_scale_factor=False,    # read a(T_gamma) directly from the NEVO table's x column
    # custom_background=None,        # path to a user-supplied background file (T, t, a columns)

    # ---- fundamental constants (overridable for sensitivity studies) -------
    # GN=6.674299257609439e-11,       # Newton's constant, SI [m^3 kg^-1 s^-2]

    # ---- background thermodynamics ------------------------------------------
    # T_start_cosmo_MeV=40.0,         # starting temperature [MeV]
    # T_end_MeV=1e-3,                 # end temperature for nuclear integration [MeV]
    # sampling_temperature_per_decade=600,  # points per decade of T for the background a(T)/t(T) grid

    # ---- n <-> p weak rates --------------------------------------------------
    # radiative_corrections=True,     # Coulomb + T=0 resummed radiative corrections (CCR); False: Born approximation
    # finite_mass_corrections=True,   # Fokker-Planck finite-nucleon-mass correction
    # thermal_corrections=True,       # finite-temperature radiative corrections (CCRTh)
    # weak_rate_cache=True,           # if False, never load the weak-rate cache (always recompute)
    # save_nTOp=True,                 # save computed n<->p rates to data/weak/
    # sampling_nTOp_per_decade=80,    # points per decade of T in the n<->p rate grid
    # save_nTOp_thermal=True,         # save computed thermal n<->p rates to data/weak/
    # sampling_nTOp_thermal_per_decade=20,  # points per decade of T for the thermal-correction table
    # tau_n_normalization=True,       # normalise weak rates using the neutron lifetime tau_n
    # tau_n=878.4,                    # neutron lifetime [s]
    # std_tau_n=0.5,                  # 1-sigma uncertainty on tau_n [s] (used for MC sampling)
    # vegas_n_eval=20000,             # vegas: evaluations per iteration (thermal-correction integral)
    # vegas_n_itn=20,                 # vegas: number of iterations
    # epsrel_thermal=1e-2,            # dblquad fallback relative tolerance

    # ---- output options -------------------------------------------------------
    # output_time_evolution=False,    # write the unified time-evolution TSV (PRIMAT.md S7.2);
    #                                  # forces the python backend unless using run_mc/run_bbn's C support (see force_backend below)
    # output_rates_time_evolution=False,  # include per-reaction nuclear-rate columns in the time-evolution TSV
    # output_n_points=500,            # number of points in the time-evolution TSV
    # output_file="results/output_tables.tsv",     # path for output_time_evolution
    # output_final_result=False,      # write a two-column (nuclide, Y) final-abundances file
    # output_final_file="results/output_final.dat",  # path for output_final_result
    # output_background_evolution=False,  # write the cosmological background's own time-evolution TSV
    # output_background_file="results/output_background.tsv",  # path for output_background_evolution
    # output_mc_samples=False,        # write every MC sample (run_mc/mc_uncertainty) to a TSV, one column per quantity
    # output_mc_file="results/output_mc_samples.tsv",  # path for output_mc_samples

    # ---- nuclear network --------------------------------------------------
    # rate_interp_order="linear",     # nuclear rate table interpolation: "linear" / "quadratic" / "cubic"
    # rate_grid_npts=1000,            # points in the master T9 grid used to resample every rate table
    # rate_grid_T9_min=1e-3,          # minimum T9 [GK] of the master rate grid
    # rate_grid_T9_max=10.0,          # maximum T9 [GK] of the master rate grid
    # network="small",                # "small" / "small_parthenope" / "large" / custom network filename
    # amax=None,                      # filter any network to reactions with A <= amax
    # atol_large_LT=1e-26,            # solve_ivp absolute tolerance for the large-network LT era
    # rescale_nuclear_rates=False,    # vary all nuclear rates by a uniform factor (sensitivity studies)
    # mc_rate_rescale_cap=30,         # clamp MC rate variation factor to [1/cap, cap]; None disables the cap
    # nuclear_qed_corrections=True,   # QED correction to select radiative-capture rates (Pitrou & Pospelov 2020)

    # ---- cosmological inputs ------------------------------------------------
    # Omegabh2=0.022425,              # baryon density Omega_b h^2 (Planck 2018 default)
    # Omegach2=0.11933,               # cold dark matter density Omega_c h^2 (Planck 2018)
    # h=0.6766,                       # reduced Hubble constant h = H_0 / (100 km/s/Mpc) (Planck 2018)
    # DeltaNeff=0.0,                  # extra relativistic species beyond SM neutrinos
    # munuOverTnu=0.0,                # reduced neutrino chemical potential xi = mu/T

    # ---- decay-era options --------------------------------------------------
    # decay_reverse_rates=False,      # compute detailed-balance reverse rates for radioactive decays
    # decay_era=False,                # run a 4th "Decay Time" era after LT (network="large" only)
    # t_decay_end=3.156e16,           # DT era duration [s] (default: 1 Gyr)
    # decay_n_points=200,             # log-spaced output points in the DT era
    # output_decay_evolution=False,   # write a TSV of the DT-era abundance time evolution
    # output_decay_file="results/output_decay_evolution.tsv",  # path for output_decay_evolution

    # ---- Early Dark Energy --------------------------------------------------
    # fEDE=0.0,                       # EDE fraction at peak; 0 = disabled
    # zcEDE=1e8,                      # redshift of EDE peak
    # wnEDE=1.0,                      # EDE equation-of-state parameter
)
# force_backend: None/"auto" (default: C extension if built, else pure Python),
# "c", or "python" -- see primat/backend.py's module docstring for exactly
# which features (extra_rho/custom_network/background=, output_time_evolution)
# always fall back to "python" regardless of this setting.
result = run_bbn(cfg, force_backend="auto")
print("Neff  =", result.get("Neff"))
print("YPBBN =", result["YPBBN"])
print("D/H   =", result["DoH"])

# Monte-Carlo nuclear-rate/tau_n uncertainty propagation (uncomment to run):
# the same dispatch story as run_bbn -- C backend when available, else
# pure-Python joblib (see primat/backend.py's module docstring for the RNG
# caveat: C and Python samples are statistically, not bit-for-bit, equal).
#
# from primat.backend import run_mc, dump_mc_samples
# mc = run_mc(50, ['YPBBN', 'DoH'], params=cfg, force_backend="auto")
# print("YPBBN =", mc['YPBBN'].mean, "+/-", mc['YPBBN'].std)
# with open("results/output_mc_samples.tsv", "w") as f:
#     f.write(dump_mc_samples(mc))
