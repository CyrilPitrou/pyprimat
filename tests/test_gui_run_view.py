"""
Direct parity test for ``primat.gui.run_view.GuiRun``.

``GuiRun`` is the backend-agnostic stand-in the GUI hands to
``primat/gui/panels.py`` in place of a solved ``primat.main.PRIMAT``
instance (see its module docstring): built from a plain
``primat.backend.run_bbn`` result dict plus a solve-free
``PRIMATConfig``/``UpdateNuclearRates`` pair, exactly as
``primat/gui/app.py``'s ``_solve`` constructs it. None of the existing GUI
tests (``tests/test_gui.py``, ``tests/test_gui_custom_network.py``)
instantiate ``GuiRun`` directly or compare it against a real ``PRIMAT`` --
they only exercise it indirectly through Streamlit's ``AppTest`` harness (and
are skipped entirely when the optional ``gui`` extra is not installed). This
module needs neither streamlit nor plotly: ``GuiRun``/``PRIMATConfig``/
``UpdateNuclearRates``/``backend.run_bbn`` are all core ``primat`` pieces.

A mismatch here (e.g. ``abundance_names`` ordering, ``get_quantity``
resolving a different value, ``__getitem__``/``T_of_t`` disagreeing with the
real interpolators) is exactly the class of bug the GUI panel regressions
documented in ``test_gui_custom_network.py`` describe: something rendering
subtly wrong data without raising.
"""
import pytest

pytestmark = pytest.mark.slow

_PARAMS = {"network": "small"}


def _build_gui_run():
    """Mirror primat/gui/app.py's _solve(): run_bbn with
    output_time_evolution=True, output_file=None (in-memory only), plus a
    solve-free PRIMATConfig/UpdateNuclearRates pair for cfg/nucl."""
    from primat import backend
    from primat.config import PRIMATConfig
    from primat.network_data import UpdateNuclearRates
    from primat.gui.run_view import GuiRun

    # force_backend="python": solved_small (tests/conftest.py) is a plain
    # primat.main.PRIMAT instance, which always uses the pure-Python
    # Background/NuclearNetwork implementation, never the C backend. Using
    # the default "auto" here would silently pick the C backend
    # (primat._primat_c, when built) and compare across backends instead of
    # checking GuiRun's own interpolator/getter logic -- the known
    # ~7e-4-relative C-vs-Python D/H gap (CLAUDE.md) would then dominate
    # this test's diffs instead of any real GuiRun bug.
    full_params = dict(_PARAMS, output_time_evolution=True, output_file=None)
    result = backend.run_bbn(full_params, force_backend="python")
    cfg = PRIMATConfig(_PARAMS)
    nucl = UpdateNuclearRates(cfg)
    return GuiRun(result, cfg, nucl)


def test_gui_run_A_Z_N_match_a_real_primat_instance(solved_small):
    """A/Z/N are built the same way (cfg.Nuclides) in both GuiRun.__init__
    and PRIMAT.__init__ -- must be identical dicts."""
    run = _build_gui_run()
    assert run.A == solved_small.A
    assert run.Z == solved_small.Z
    assert run.N == solved_small.N


def test_gui_run_abundance_names_matches_real_primat_instance(solved_small):
    run = _build_gui_run()
    assert set(run.abundance_names) == set(solved_small.abundance_names)


def test_gui_run_get_quantity_matches_real_primat_instance(solved_small):
    run = _build_gui_run()
    for key in ("DoH", "YPBBN", "He3oH", "Li7oH"):
        assert run.get_quantity(key) == pytest.approx(
            solved_small.get_quantity(key), rel=1e-9)
    # Per-nuclide final-abundance fallback (not a fixed result-dict key).
    assert run.get_quantity("He4") == pytest.approx(
        solved_small.get_quantity("He4"), rel=1e-9)


def test_gui_run_get_quantity_raises_on_unknown_name():
    run = _build_gui_run()
    with pytest.raises(ValueError):
        run.get_quantity("not_a_real_quantity")


def test_gui_run_getitem_interpolator_matches_real_primat_instance(solved_small):
    """GuiRun.__getitem__ is built from the discrete evolution arrays
    (Y_interpolator), while PRIMAT's is a live ODE-solution interpolator --
    they must still agree numerically at any shared time."""
    run = _build_gui_run()
    t_mid = 0.5 * (run.evolution.t[0] + run.evolution.t[-1])
    assert run["He4"](t_mid) == pytest.approx(solved_small["He4"](t_mid), rel=1e-3)


def test_gui_run_T_of_t_matches_real_primat_instance(solved_small):
    run = _build_gui_run()
    t_mid = 0.5 * (run.evolution.t[0] + run.evolution.t[-1])
    assert run.T_of_t(t_mid) == pytest.approx(solved_small.T_of_t(t_mid), rel=1e-3)
