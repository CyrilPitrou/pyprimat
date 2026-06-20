# -*- coding: utf-8 -*-
"""
weak_rates.cache — fingerprinted on-disk cache for the n<->p weak-rate tables
===============================================================================

Fingerprint dicts and the log-spaced grid-density helper shared by
weak_rates.api (ComputeWeakRates/InterpolateWeakRates/RecomputeWeakRates) and
weak_rates.corrections (the CCRTh thermal-correction cache).  See
weak_rates.api for the cache-file layout and invalidation policy.
"""

import numpy as np

__all__ = ['WEAK_RATE_FORMAT_VERSION', '_WEAK_RATE_BG_FIELDS', '_THERMAL_BG_FIELDS',
           'n_points_per_decade', '_thermal_fingerprint', '_weak_rate_fingerprint']

# ---------------------------------------------------------------------------
# Fingerprinted cache for the n<->p weak-rate tables
# ---------------------------------------------------------------------------
# Bump this whenever a code change alters the *numerical content* of the
# cached files for a fixed configuration (new physics term, changed formula,
# different file layout, ...).  Bumping it invalidates every existing cache
# file regardless of its fingerprint.
#
# v1: forward and backward rates stored together in nTOp_<hash>.txt (hash in
# filename, rates in units of 1/tau_n, clamped below 1e-28 to zero).
# Fingerprints simplified: thermal uses only the T range, incomplete-decoupling
# flag, and NEVO file selection; weak-rate drops sampling_temperature_per_decade,
# nevo_grid_file, external_scale_factor, thermal_corrections and
# thermal_fingerprint_hash.  tau_n_flag renamed to tau_n_normalization.
# v2: sampling_nTOp/sampling_nTOp_thermal (total grid points) replaced by
# sampling_nTOp_per_decade/sampling_nTOp_thermal_per_decade (points per decade
# of T), so the grid density now stays fixed when T_end_MeV changes the span.
# v3: the n<->p rate table nTOp_<hash>.txt now stores ONLY the non-thermal
# rate (Born + finite-mass + CCR + spectral-distortion); the finite-temperature
# radiative correction (CCRTh) is kept in its own nTOp_thermal_<hash>.txt and
# recombined at point of use (RecomputeWeakRates), matching the fingerprint
# which never included thermal_corrections.  The CCRTh table content also
# changed: the n->p direction is now clamped to 0 below ~10^8.2 K (see
# _L_CCRTh_compute) to remove a spurious infrared-divergent bremsstrahlung
# residual.  Both changes invalidate every v2 cache file.
WEAK_RATE_FORMAT_VERSION = 1

# Config fields entering the weak-rate fingerprint (nTOp_<hash>.txt).
# DeltaNeff is deliberately NOT listed: it only shifts the time-temperature
# relation Tg(t) and does not affect the rate integrand at fixed Tg (in decoupling approximation).
# In principle if we consider a DeltaNeff with incomplete decoupling we must also consider the associated NEVO file.
# We need to review the interplay between NEVO and PyPRIMAT.
# Note  that spectral distortions and incomplete decoupling effects are expected to have a small effect on weak rates.
_WEAK_RATE_BG_FIELDS = [
    "radiative_corrections",
    "finite_mass_corrections",
    "munuOverTnu",
    "QED_corrections",
    "incomplete_decoupling",
    "spectral_distortions",
    "analytic_distortions",
    "delta_xi_nu",
    "y_SZ",
    "y_gray",
    "T_start_cosmo_MeV",
    "T_end_MeV",
    "sampling_nTOp_per_decade",
    "nevo_file",
    "nevo_spectral_file",
    "nevo_file_prefix",
]

# Config fields entering the thermal-correction fingerprint
# (nTOp_thermal_<hash>.txt).  Only the temperature range and sampling, the neutrino
# decoupling mode (with or without QED corrections), and the NEVO thermo/spectral table selection matter for
# the double-integral over (E, k) that defines the finite-temperature
# radiative correction.
# When improving the interpolay with NEVO this could be improved. 
_THERMAL_BG_FIELDS = [
    "T_end_MeV",
    "T_start_cosmo_MeV",
    "sampling_nTOp_thermal_per_decade",
    "QED_corrections",
    "incomplete_decoupling",
    "nevo_file",
    "nevo_file_prefix"
]


def n_points_per_decade(per_decade, T_lo, T_hi):
    """Number of log-spaced grid points spanning [T_lo, T_hi] at a fixed
    density of ``per_decade`` points per decade of T.

    Used so that ``sampling_nTOp_per_decade``/``sampling_nTOp_thermal_per_decade``
    keep a constant grid resolution even if ``T_end_MeV`` (and hence the
    number of decades spanned) changes, unlike the old total-point-count
    parametrisation.

    Args:
        per_decade: float, desired points per decade of T.
        T_lo, T_hi: float, grid endpoints [K], T_hi > T_lo.

    Returns:
        int, number of points (at least 2).
    """
    decades = np.log10(T_hi / T_lo)
    return max(2, int(round(per_decade * decades)))


def _thermal_fingerprint(cfg):
    """Fingerprint dict for the thermal radiative-correction cache file
    ``nTOp_thermal_<hash>.txt``.

    Only the fields that actually affect the finite-temperature double
    integral (Brown & Sawyer 2001) are included: the temperature integration
    range, the neutrino-to-photon temperature ratio T_ν(T_γ) (fixed by the
    NEVO table selection), and the thermal-correction grid density.

    Args:
        cfg: PyPRConfig instance.

    Returns:
        dict, JSON-serialisable.
    """
    fp = {"format_version": WEAK_RATE_FORMAT_VERSION,
          "sampling_nTOp_thermal_per_decade": cfg.sampling_nTOp_thermal_per_decade}
    for key in _THERMAL_BG_FIELDS:
        fp[key] = getattr(cfg, key)
    return fp


def _weak_rate_fingerprint(cfg):
    """Fingerprint dict for the n<->p weak-rate cache file ``nTOp_<hash>.txt``.

    ``cfg.tau_n_normalization``/``cfg.tau_n`` are deliberately excluded: the
    stored rates are in units of 1/τ_n (Fn already applied inside
    :func:`ComputeWeakRates`), so only 1/tau_n needs multiplying after
    loading — the cached values themselves are tau_n-independent.

    The thermal-correction cache has its own hash-named file and is not
    folded in here: the two caches are independent, and ``thermal_corrections``
    itself does not affect the stored non-thermal rates.

    Args:
        cfg: PyPRConfig instance.

    Returns:
        dict, JSON-serialisable; pass to :func:`fingerprint_hash` for the hash.
    """
    fp = {"format_version":          WEAK_RATE_FORMAT_VERSION,
          "sampling_nTOp_per_decade": cfg.sampling_nTOp_per_decade,
          "radiative_corrections":   cfg.radiative_corrections,
          "finite_mass_corrections": cfg.finite_mass_corrections}
    for key in _WEAK_RATE_BG_FIELDS:
        fp[key] = getattr(cfg, key)
    return fp

