"""
Tests for the optional Streamlit GUI (``pyprimat.gui``, see ``GUI.md``).

The GUI is the third way (alongside a Python script and the ``pyprimat``
console-script CLI, ``tests/test_cli.py``) of driving the same
"params dict -> PyPR -> results dict" contract.  These tests:

* check that the optional ``gui`` extra (streamlit/plotly) does not leak
  into the mandatory import of ``pyprimat`` (``test_wheel_smoke.py`` already
  covers that the package *data* ships correctly; this covers that the
  *import graph* stays separate);
* drive ``pyprimat/gui/app.py`` end-to-end with Streamlit's ``AppTest``
  harness (no browser needed) and check that a default small-network run
  reproduces the exact values pinned in ``test_cli.py`` -- i.e. the GUI
  calls ``PyPR`` identically to the CLI;
* check that an invalid flag combination (caught by ``PyPRConfig``) is
  surfaced as a clean ``st.error`` rather than a traceback (GUI.md
  verification step 5).

Each ``AppTest`` run that clicks "Run BBN" performs one full small-network
solve (~1.2 s, like ``test_cli.py``), so this module is marked
``slow``/``solve``. All tests are skipped if the optional ``gui`` extra
(``pip install ".[gui]"``) is not installed.
"""
import pytest

st = pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest

from pyprimat.gui import params_form

pytestmark = [pytest.mark.slow, pytest.mark.solve, pytest.mark.gui]

APP_PATH = "pyprimat/gui/app.py"


# ---------------------------------------------------------------------------
# Packaging: the core package must not require the gui extra
# ---------------------------------------------------------------------------

def test_pyprimat_import_does_not_pull_in_gui():
    """``import pyprimat`` must not import ``pyprimat.gui`` (or streamlit).

    ``pyprimat.gui`` is shipped inside the package (GUI.md "Packaging") so
    that ``pip install ".[gui]"`` provides the ``pyprimat-gui`` console
    script, but ``streamlit``/``plotly`` are optional: a plain
    ``pip install pyprimat`` (no extra) must still let
    ``from pyprimat import PyPR`` work. Guard against ``pyprimat/__init__.py``
    ever growing an eager ``from . import gui`` or similar.
    """
    # pyprimat (and its gui subpackage) are already imported by the time this
    # test module runs -- check the *module source* instead of re-importing,
    # which is robust regardless of import order within the test session.
    import pyprimat
    assert "pyprimat.gui" not in getattr(pyprimat, "__all__", [])
    with open(pyprimat.__file__) as f:
        source = f.read()
    assert "gui" not in source, (
        "pyprimat/__init__.py must not reference the gui subpackage, so "
        "`import pyprimat` keeps working without the optional gui extra"
    )


# ---------------------------------------------------------------------------
# Parameter form helpers
# ---------------------------------------------------------------------------

def test_form_metadata_covers_amax_default():
    """`amax` (the one DEFAULT_PARAMS key whose default is ``None``) must be
    handled by the curated/conditional logic, not the generic type-based
    widget chooser (which has no sensible widget for ``None``)."""
    assert "amax" in params_form._FORM_METADATA
    assert "amax" in params_form._CONDITIONAL


def test_available_networks_includes_small_and_large():
    networks = params_form._available_networks()
    assert "small" in networks
    assert "large" in networks
    assert "medium" in networks


# ---------------------------------------------------------------------------
# End-to-end: AppTest drives the Streamlit script without a browser
# ---------------------------------------------------------------------------

def _run_bbn(at):
    """Click the sidebar "Run BBN" button and let the app rerun."""
    at.sidebar.button[0].click()
    at.run(timeout=120)
    return at


def test_app_loads_without_error():
    """The app renders (sidebar form + placeholder message) with no run yet."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception
    assert at.sidebar.button[0].label == "Run BBN"
    # Before any run, the main area shows the "set parameters" placeholder.
    assert any("Run BBN" in info.value for info in at.info)


def test_default_run_matches_cli_reference():
    """Default (small-network) GUI run reproduces test_cli.py's pinned values.

    Both the GUI (`pyprimat.gui.app._solve`) and the `pyprimat` console
    script (`pyprimat.cli.main`) call `PyPR(params=params).PyPRresults()`
    with the same defaults, so they must agree to full precision -- this is
    the GUI.md verification step 3 ("reference run parity"), pinned to the
    same values as `test_cli.py::test_cli_default_summary` /
    `test_cli_json_matches_default_summary`.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    _run_bbn(at)
    assert not at.exception

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["N_eff"] == "3.04397730"
    assert metrics["Y_P (BBN)"] == "0.24691081"
    assert metrics["D / H"] == "2.4365492e-05"

    # Per-nuclide final-abundance table (render_results_panel).
    df = at.dataframe[0].value
    by_nuclide = df.set_index("nuclide")["Y"]
    assert by_nuclide["p"] == pytest.approx(7.530290e-01, rel=1e-5)
    assert by_nuclide["He4"] == pytest.approx(0.24691081 / 4., rel=1e-5)


def test_evolution_panel_renders_with_default_selection():
    """The abundance-evolution panel's nuclide multiselect defaults to the
    'light elements' preset and renders without error alongside the results
    panel (both tabs' bodies execute on every run, see GUI.md §4)."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    _run_bbn(at)
    assert not at.exception

    [evolution_select] = [
        ms for ms in at.multiselect if ms.label == "Nuclides to plot"
    ]
    # Default preset (params_form._LIGHT_NUCLIDES intersected with the
    # small-network's 8 tracked nuclides).
    assert set(evolution_select.value) == {
        "n", "p", "H2", "H3", "He3", "He4", "Li7", "Be7",
    }


# ---------------------------------------------------------------------------
# Conditional widgets and error surfacing
# ---------------------------------------------------------------------------

def test_amax_widget_only_shown_for_large_network():
    """`amax` (GUI.md §2 "Network") only appears once `network='large'`."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)

    def has_amax_checkbox():
        return any(c.key == "amax_enabled" for c in at.sidebar.checkbox)

    assert not has_amax_checkbox()  # default network is "small"

    [network_select] = [s for s in at.sidebar.selectbox if s.key == "network"]
    network_select.set_value("large")
    at.run(timeout=60)
    assert has_amax_checkbox()


def test_invalid_flag_combination_surfaces_as_error_not_traceback():
    """spectral_distortions=True + incomplete_decoupling=False (with the
    default analytic_distortions=False) is rejected by `PyPRConfig.__init__`
    (config.py); the GUI must show `st.error`, not crash (GUI.md
    verification step 5)."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)

    def toggle(key):
        [t] = [t for t in at.sidebar.toggle if t.key == key]
        return t

    toggle("spectral_distortions").set_value(True)
    toggle("incomplete_decoupling").set_value(False)
    at.run(timeout=60)
    _run_bbn(at)

    assert not at.exception
    assert len(at.error) == 1
    assert "incomplete_decoupling" in at.error[0].value
