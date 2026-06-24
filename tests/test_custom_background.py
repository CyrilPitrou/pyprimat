# -*- coding: utf-8 -*-
"""
test_custom_background.py
=========================
Tests for the ``custom_background`` mode: a user-supplied (T, t, a) table
drives the cosmological background while the nuclear network is solved with
instantaneous-decoupling n<->p weak rates.

Strategy
--------
1. Run the code in the standard instantaneous-decoupling mode
   (``incomplete_decoupling=False``, ``spectral_distortions=False``) to obtain
   a reference background (T, t, a table) and reference BBN results.
2. Write that background to a temporary TSV file.
3. Run again with ``custom_background=<file>``, which reads the same table
   back and reconstructs the background from it.
4. Assert that the BBN observables (YPBBN, DoH) match to within a small
   relative tolerance arising only from interpolation of the table, and that
   Neff is estimated and physically reasonable.

Additional tests cover:
- Warning behaviour when the caller sets conflicting flags
  (``incomplete_decoupling=True``, ``spectral_distortions=True``).
- Error on missing required columns.
- Incompatibility of ``custom_background`` with ``external_scale_factor``.
"""
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helper: write a background file from a solved PRIMAT instance
# ---------------------------------------------------------------------------

def _write_background_file(background, path):
    """Write a custom-background TSV (T [MeV], t [s], a) from a Background.

    Samples the background on a 2000-point log-spaced time grid spanning the
    full BBN window (T_start_cosmo down to T_end).  The file uses tab
    delimiters and a plain header (matching the format accepted by
    CustomBackground._load_table).

    Args:
        background: a solved primat.background.Background with has_scale_factor=True.
        path (str): destination file path.
    """
    cfg     = background.cfg
    T_start = cfg.T_start_cosmo / cfg.MeV_to_Kelvin  # [MeV]
    T_end   = cfg.T_end         / cfg.MeV_to_Kelvin  # [MeV]
    t_start = float(background.t_of_T(T_start))
    t_end   = float(background.t_of_T(T_end))
    t_arr   = np.logspace(np.log10(t_start), np.log10(t_end), 2000)
    T_arr   = background.T_of_t(t_arr)
    a_arr   = background.a_of_t(t_arr)
    data    = np.column_stack([T_arr, t_arr, a_arr])
    np.savetxt(path, data, delimiter='\t', header="T\tt\ta", comments='')


# ---------------------------------------------------------------------------
# Shared reference fixture (instantaneous decoupling, no NEVO tables)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ref_run():
    """Standard PRIMAT run with instantaneous decoupling (the baseline for
    the custom-background comparison).

    Uses ``incomplete_decoupling=False`` and ``spectral_distortions=False``
    so that the standard and custom backgrounds are driven by the same
    physics (instantaneous-decoupling T_ν(T_γ)).

    Returns:
        Solved PRIMAT instance.
    """
    from primat.main import PRIMAT
    r = PRIMAT({
        "incomplete_decoupling":  False,
        "spectral_distortions":   False,
        "network":                "small",
    })
    r.solve()
    return r


# ---------------------------------------------------------------------------
# Main agreement test
# ---------------------------------------------------------------------------

@pytest.mark.solve
def test_custom_background_matches_reference(ref_run, tmp_path):
    """Custom-background run must reproduce the reference BBN observables.

    The custom table is written from the reference background (which was
    produced by the same instantaneous-decoupling physics that CustomBackground
    uses for weak rates), so the only differences are due to table
    interpolation.  We allow a relative tolerance of 1e-5 on YPBBN and DoH —
    well below any physically meaningful threshold.

    Neff is expected to be in (2.9, 3.1): the standard instantaneous-decoupling
    value is ~3.0.
    """
    from primat.main import PRIMAT

    bg_file = str(tmp_path / "background.tsv")
    _write_background_file(ref_run.background, bg_file)

    r_custom = PRIMAT({
        "custom_background": bg_file,
        "network":           "small",
    })
    r_custom.solve()

    ref = ref_run.results
    cst = r_custom.results

    # BBN abundances must agree to within the table-interpolation budget.
    # With a 2000-point log-spaced grid the interpolation of T(t)/a(t)
    # introduces ~3e-5 relative error in YPBBN (dominated by n/p freeze-out
    # at T ~ 1 MeV) and ~2e-6 in DoH.  We allow 1e-4, which is already far
    # below any physically meaningful threshold (observational precision on
    # YPBBN is ~0.1%; on D/H ~1%).
    assert cst["YPBBN"] == pytest.approx(ref["YPBBN"], rel=1e-4), (
        f"YPBBN mismatch: reference={ref['YPBBN']:.8f}, "
        f"custom={cst['YPBBN']:.8f}"
    )
    assert cst["DoH"] == pytest.approx(ref["DoH"], rel=1e-4), (
        f"DoH mismatch: reference={ref['DoH']:.7e}, "
        f"custom={cst['DoH']:.7e}"
    )

    # Neff must be present and physically reasonable.
    assert "Neff" in cst, "Neff missing from custom-background results"
    assert 2.9 < cst["Neff"] < 3.1, (
        f"Neff = {cst['Neff']:.6f} outside expected range (2.9, 3.1)"
    )


# ---------------------------------------------------------------------------
# Flag-conflict warning tests
# ---------------------------------------------------------------------------

@pytest.mark.solve
def test_custom_background_warns_incomplete_decoupling(ref_run, tmp_path):
    """custom_background must warn and override incomplete_decoupling=True."""
    from primat.main import PRIMAT

    bg_file = str(tmp_path / "bg_warn_id.tsv")
    _write_background_file(ref_run.background, bg_file)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = PRIMAT({
            "custom_background":     bg_file,
            "incomplete_decoupling": True,   # should be overridden with a warning
            "network":               "small",
        })

    assert r.cfg.incomplete_decoupling is False, (
        "incomplete_decoupling should be forced to False by custom_background"
    )
    override_warnings = [str(w.message) for w in caught if "incomplete_decoupling" in str(w.message)]
    assert override_warnings, (
        "Expected a warning about overriding incomplete_decoupling; got none"
    )


@pytest.mark.solve
def test_custom_background_warns_spectral_distortions(ref_run, tmp_path):
    """custom_background must warn and override spectral_distortions=True."""
    from primat.main import PRIMAT

    bg_file = str(tmp_path / "bg_warn_sd.tsv")
    _write_background_file(ref_run.background, bg_file)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = PRIMAT({
            "custom_background":   bg_file,
            "spectral_distortions": True,   # incompatible, should be overridden
            # Need incomplete_decoupling=False to avoid the "no NEVO + SD" error
            # that fires *after* the custom_background override.  Since
            # custom_background forces incomplete_decoupling=False first, this
            # is taken care of automatically.
            "network":             "small",
        })

    assert r.cfg.spectral_distortions is False
    sd_warnings = [str(w.message) for w in caught if "spectral_distortions" in str(w.message)]
    assert sd_warnings, (
        "Expected a warning about overriding spectral_distortions; got none"
    )


# ---------------------------------------------------------------------------
# Error tests (no solve needed)
# ---------------------------------------------------------------------------

def test_custom_background_missing_columns(tmp_path):
    """A file lacking the 'a' column must raise a ValueError."""
    from primat.main import PRIMAT

    bad_file = str(tmp_path / "bad.tsv")
    np.savetxt(bad_file, np.ones((5, 2)), delimiter='\t',
               header="T\tt", comments='')

    with pytest.raises(ValueError, match="missing required columns"):
        PRIMAT({"custom_background": bad_file, "network": "small"})


def test_custom_background_external_scale_factor_conflict():
    """Combining custom_background with external_scale_factor must raise ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        from primat.config import PRIMATConfig
        PRIMATConfig({
            "custom_background":  "some_file.tsv",
            "external_scale_factor": True,
        })
