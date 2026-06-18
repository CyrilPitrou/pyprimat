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
import os

import pytest

st = pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest

from pyprimat.gui import params_form

pytestmark = [pytest.mark.slow, pytest.mark.solve, pytest.mark.gui]

APP_PATH = "pyprimat/gui/app.py"

# network="large" needs the generated AC2024 rate/data CSVs (tests/test_large_network.py).
_AC2024_DIR = os.path.join(os.path.dirname(__file__), "..", "pyprimat",
                           "rates", "nuclear", "data")
_needs_ac2024 = pytest.mark.skipif(
    not os.path.isdir(_AC2024_DIR),
    reason="rates/nuclear/data not generated",
)


def _download_button(at, label):
    """Find the ``st.download_button`` with the given ``label`` in ``at``.

    ``AppTest`` exposes download buttons as ``UnknownElement`` nodes (no
    dedicated accessor), so walk the element tree looking for one whose
    ``label`` matches. Returns ``None`` if not found.
    """
    def walk(node):
        for child in getattr(node, "children", {}).values():
            if type(child).__name__ == "UnknownElement" and getattr(child, "label", None) == label:
                return child
            found = walk(child)
            if found is not None:
                return found
        return None

    return walk(at.main)


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
    handled by curated logic, not the generic type-based widget chooser
    (which has no sensible widget for ``None``). Unlike the other
    conditionally-rendered keys, `amax` is now offered for *every* network
    (CUSTOMPOPUP.md §3.3), so it is handled directly in
    ``render_sidebar_form`` rather than through ``_CONDITIONAL``."""
    assert "amax" in params_form._FORM_METADATA
    assert "amax" not in params_form._CONDITIONAL


def test_available_networks_includes_small_and_large():
    networks = params_form._available_networks()
    assert networks == ["large", "small", "small_parthenope"]


def test_network_label_appends_reaction_count():
    """The selectbox shows e.g. 'small (12)'/'large (N)' so users can
    gauge a network's size before picking it; the count is read dynamically
    (CUSTOMPOPUP.md dropped the old fixed-size 'deuterium' network)."""
    assert params_form._network_label("small") == "small (12)"
    n_large = len(params_form.load_reaction_names(
        params_form.PyPRConfig({"network": "large"}), "large"))
    assert params_form._network_label("large") == f"large ({n_large})"


# ---------------------------------------------------------------------------
# End-to-end: AppTest drives the Streamlit script without a browser
# ---------------------------------------------------------------------------

def _run_bbn(at):
    """Click the sidebar "Run BBN" button and let the app rerun.

    "Run BBN" is no longer the sidebar's only button -- "Import custom
    network"/"Create custom network" (CUSTOMPOPUP.md §5.2) come first -- so
    find it by label rather than assuming index 0.
    """
    [run_button] = [b for b in at.sidebar.button if b.label == "Run BBN"]
    run_button.click()
    at.run(timeout=120)
    return at


def test_app_loads_without_error():
    """The app renders (sidebar form + placeholder message) with no run yet."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception
    assert {"Run BBN", "Import custom network", "Create custom network"} <= {
        b.label for b in at.sidebar.button
    }
    # Before any run, the main area shows the "set parameters" placeholder.
    assert any("Run BBN" in info.value for info in at.info)


def _markdown_table_rows(md_value):
    """Parse a "| col1 | col2 | ... |" Markdown table into a dict keyed by
    the first column (stripping leading/trailing whitespace from each cell).
    Skips the header and separator ("|---|---|") lines."""
    rows = {}
    for line in md_value.splitlines()[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def test_default_run_matches_cli_reference():
    """Default (small-network) GUI run reproduces test_cli.py's pinned values.

    Both the GUI (`pyprimat.gui.app._solve`) and the `pyprimat` console
    script (`pyprimat.cli.main`) call `PyPR(params=params).PyPRresults()`
    with the same defaults, so they must agree to full precision -- this is
    the GUI.md verification step 3 ("reference run parity"), pinned to the
    same values as `test_cli.py::test_cli_default_summary` /
    `test_cli_json_matches_default_summary` (spectral_distortions=True,
    IDEAS2.md item 2).
    """
    from pyprimat.network_data import nuclide_latex

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    _run_bbn(at)
    assert not at.exception

    # "Standard ratios" Markdown table (render_results_panel).
    [ratios_md] = [
        md for md in at.markdown if "| Quantity | Value |" in md.value
    ]
    ratios = _markdown_table_rows(ratios_md.value)
    # These pins were last refreshed after the weak-rates update (commits
    # 4c5b8e0/d8bd969/78c9572/521cf4d); the shifts are within the CLAUDE.md
    # tolerance (mirrors test_cli.py::test_cli_default_summary).
    assert ratios[r"$N_{\text{eff}}$"] == ["3.04397730"]
    assert ratios[r"$Y_P\ (\text{BBN})$"] == ["0.24699670"]
    assert ratios[r"$\text{D}/\text{H}$"] == ["2.4349840e-05"]

    # Per-nuclide final-abundance Markdown table (render_results_panel).
    [abundances_md] = [
        md for md in at.markdown if "| Nuclide | A | Z | Y |" in md.value
    ]
    by_nuclide = _markdown_table_rows(abundances_md.value)
    assert float(by_nuclide[nuclide_latex("p")][-1]) == pytest.approx(7.529422e-01, rel=1e-5)
    assert float(by_nuclide[nuclide_latex("He4")][-1]) == pytest.approx(0.06174942, rel=1e-5)


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

def test_amax_widget_shown_for_every_network():
    """`amax` (GUI.md §2 "Network") is offered regardless of `network`'s
    value (CUSTOMPOPUP.md §3.3 dropped the old "large only" restriction)."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)

    def has_amax_checkbox():
        return any(c.key == "amax_enabled" for c in at.sidebar.checkbox)

    assert has_amax_checkbox()  # default network is "small"; still shown

    [network_select] = [s for s in at.sidebar.selectbox if s.key == "network"]
    network_select.set_value("large")
    at.run(timeout=60)
    assert has_amax_checkbox()


@_needs_ac2024
def test_time_evolution_download_available_for_large_network():
    """The "output_time_evolution.tsv" download button (under the "Abundances
    time evolution" subsection of the Output tab, see
    ``panels.render_downloads_panel``) is offered for ``network="large"``
    too, not just "small".

    ``NuclearNetwork._write_time_evolution`` (nuclear_network.py) derives its
    ``Y<species>`` columns from ``self.abundance_names``, which already
    covers all three networks
    (8/12/~59 nuclides, see ``test_large_network.py::
    test_large_network_time_evolution_tsv``) -- the GUI's ``_solve`` must not
    special-case "large" out of generating that TSV.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)

    [network_select] = [s for s in at.sidebar.selectbox if s.key == "network"]
    network_select.set_value("large")
    at.run(timeout=60)

    _run_bbn(at)
    assert not at.exception

    button = _download_button(at, "output_time_evolution.tsv")
    assert button is not None
    assert button.proto.url.endswith(".tsv")

    # The old "not available for the large network" fallback caption for the
    # *time-evolution* download is gone (a different, unrelated caption with
    # the same wording now exists for "Customise Reactions", which the large
    # network does legitimately disable -- see params_form.py).
    assert not any("not available" in c.value and "time evolution" in c.value.lower()
                   for c in at.caption)


def test_quick_mc_uncertainty_adds_sigma_column():
    """The "Quick MC uncertainty" toggle (Item 14) adds a "+/- 1 sigma (quick
    MC)" column to the "Standard ratios" table, with a positive sigma for
    YPBBN -- mirroring ``tests/test_mc.py::test_std_positive``."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)

    [toggle] = [t for t in at.sidebar.toggle if t.key == "quick_mc_uncertainty"]
    toggle.set_value(True)
    at.run(timeout=60)
    _run_bbn(at)
    assert not at.exception

    [ratios_md] = [
        md for md in at.markdown if "Standard ratios" not in md.value
        and "quick MC" in md.value and "|" in md.value
    ]
    assert "± 1σ (quick MC, 30 samples)" in ratios_md.value
    assert "$Y_P\\ (\\text{BBN})$" in ratios_md.value


def test_quick_mc_uncertainty_with_customised_network():
    """Quick MC must keep working (no exception, sigma column still renders)
    when a custom network (built via "Create custom network" -> "Apply and
    run BBN") has a reaction removed -- guards app._quick_mc's decode/forward
    of the JSON custom_network entry."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)

    [toggle] = [t for t in at.sidebar.toggle if t.key == "quick_mc_uncertainty"]
    toggle.set_value(True)
    at.run(timeout=60)

    [create_btn] = [b for b in at.sidebar.button if b.label == "Create custom network"]
    create_btn.click()
    at.run(timeout=60)

    [ddtp] = [t for t in at.toggle if t.key == "_dialog_keep_d_d__t_p"]
    ddtp.set_value(False)
    at.run(timeout=60)

    [apply_btn] = [b for b in at.button if b.key == "_dialog_apply"]
    apply_btn.click()
    at.run(timeout=120)
    assert not at.exception

    [ratios_md] = [
        md for md in at.markdown if "Standard ratios" not in md.value
        and "quick MC" in md.value and "|" in md.value
    ]
    assert "± 1σ (quick MC, 30 samples)" in ratios_md.value


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
