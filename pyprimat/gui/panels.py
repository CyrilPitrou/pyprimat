# -*- coding: utf-8 -*-
"""
pyprimat.gui.panels
====================

The two result panels of the PyPRIMAT GUI (GUI.md §4-5):

* :func:`render_results_panel` -- the standard BBN ratios (Neff, Yp, D/H,
  He3/He4, He3/H, Li7/H) plus a per-nuclide table of final abundances.
* :func:`render_evolution_panel` -- an interactive ``A_i Y_i(t)`` plot with
  per-nuclide selection, paralleling ``notebooks/AbundanceEvolution.ipynb``.
* :func:`final_abundances_text` -- the ``output_final.dat``-format text for
  the download button rendered by ``pyprimat.gui.app`` below the two panels
  (alongside the time-evolution download).

All three take an already-solved ``pyprimat.PyPR`` instance (see
``pyprimat.gui.app``, which calls ``run.solve()`` once and caches the
result).
"""
import re

import numpy as np
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st

from pyprimat.nuclear import nuclide_latex


# ---------------------------------------------------------------------------
# Final abundances + standard ratios panel
# ---------------------------------------------------------------------------

# Display precision per CLAUDE.md "Reporting numerical results": these flags'
# effect on Neff is at the 1e-2..1e-3 level, so a handful of decimals are
# needed to distinguish e.g. incomplete_decoupling / QED_corrections runs.
_RATIO_FORMAT = {
    "Neff":    ".8f",
    "YPBBN":   ".8f",
    "YPCMB":   ".8f",
    "DoH":     ".7e",
    "He3oH":   ".7e",
    "He3oHe4": ".6e",
    "Li7oH":   ".6e",
}

# LaTeX labels (rendered by st.markdown's KaTeX support) for the "Standard
# ratios" table below.
_RATIO_LABELS = {
    "Neff":    r"$N_{\text{eff}}$",
    "YPBBN":   r"$Y_P\ (\text{BBN})$",
    "YPCMB":   r"$Y_P\ (\text{CMB})$",
    "DoH":     r"$\text{D}/\text{H}$",
    "He3oH":   r"$({}^{3}\text{He}+\text{T})/\text{H}$",
    "He3oHe4": r"${}^{3}\text{He}/{}^{4}\text{He}$",
    "Li7oH":   r"$({}^{7}\text{Li}+{}^{7}\text{Be})/\text{H}$",
}


def render_results_panel(run):
    """Render the final-abundances + standard-ratios panel.

    Parameters
    ----------
    run : pyprimat.PyPR
        An already-solved ``PyPR`` instance (``run.solve()`` must have been
        called, e.g. by ``pyprimat.gui.app._solve``).

    Layout
    ------
    1. A vertical, two-column table (Markdown, with LaTeX-rendered labels) of
       the 7 headline observables from ``run.PyPRresults()`` (the 9-key
       results dict, ``main.py:751-761``; ``Omeganurel``/``OneOverOmeganunr``
       are omitted here as niche neutrino-energy-density quantities),
       formatted to the precision required by ``CLAUDE.md``.
    2. A table of every tracked nuclide (``run.abundance_names``), with the
       nuclide name in standard isotope LaTeX notation (``nuclide_latex``),
       its mass number ``A``, charge ``Z``, and final mass-fraction abundance
       ``Y`` (``run.get_quantity(name)``).

    The ``output_final.dat``-format download for this table is provided
    separately by :func:`final_abundances_text`, rendered by
    ``pyprimat.gui.app`` alongside the time-evolution download.
    """
    results = run.PyPRresults()

    st.subheader("Standard ratios")
    lines = ["| Quantity | Value |", "|---|---|"]
    lines += [
        f"| {_RATIO_LABELS[key]} | {format(results[key], fmt)} |"
        for key, fmt in _RATIO_FORMAT.items()
    ]
    st.markdown("\n".join(lines))

    st.subheader("Final abundances")
    lines = ["| Nuclide | A | Z | Y |", "|---|---|---|---|"]
    lines += [
        f"| {nuclide_latex(name)} | {run.A[name]} | {run.Z[name]} | {run.get_quantity(name):.6e} |"
        for name in run.abundance_names
    ]
    st.markdown("\n".join(lines))


def final_abundances_text(run):
    """Return the ``output_final.dat``-format text for every tracked nuclide.

    Same two-column ``nuclide  Y`` format as ``PyPR._write_final_result``
    (``output_final_result=True``), built from the in-memory results so that
    flag is not needed just to export this table. ``Y`` is the final
    mass-fraction abundance of every nuclide in ``run.abundance_names`` (8 /
    12 / ~59 for the small / medium / large network).
    """
    lines = [f"# {'nuclide':<12}Y"]
    lines += [
        f"{name:<14}{run.get_quantity(name):.6e}" for name in run.abundance_names
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Abundance time-evolution panel
# ---------------------------------------------------------------------------

# Default nuclide selection and time grid, matching
# notebooks/AbundanceEvolution.ipynb (species_small/medium lists and
# `t = np.logspace(0, 5, 500)`, i.e. 1 s to 1e5 s -- the range over which all
# three networks have completed nucleosynthesis and the Y_i(t) interpolators
# remain well defined).
_LIGHT_NUCLIDES = ["n", "p", "H2", "H3", "He3", "He4", "Li6", "Li7", "Be7"]
_T_GRID = np.logspace(0, 5, 500)  # cosmic time [s]

# Plotly (as embedded by Streamlit) does not load MathJax, so the
# "$...$" LaTeX from nuclide_latex() is shown as literal text in chart
# legends/axis titles rather than being typeset. Use Unicode super/subscripts
# there instead -- e.g. "He3" -> "³He", matching nuclide_latex's "{}^{3}He".
_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _nuclide_unicode(name):
    """Unicode isotope label for Plotly legends/axes (e.g. 'He3' -> '³He')."""
    m = re.match(r"^([A-Z][a-z]?)(\d+)$", name)
    if not m:
        return name
    symbol, mass_number = m.groups()
    return mass_number.translate(_SUPERSCRIPT_DIGITS) + symbol


def render_evolution_panel(run):
    """Render the interactive ``A_i Y_i(t)`` abundance-evolution panel.

    Parameters
    ----------
    run : pyprimat.PyPR
        An already-solved ``PyPR`` instance.

    Notes
    -----
    Mirrors ``notebooks/AbundanceEvolution.ipynb``: for each selected nuclide
    ``name``, plots ``run.A[name] * run[name](t)`` (the mass fraction
    weighted by mass number, i.e. the per-baryon abundance) on a log-log
    Plotly figure. ``run[name]`` is the ``Y(t)`` interpolator from
    ``PyPR.__getitem__`` (``main.py:913``), which is built once at solve time
    -- so re-rendering with a different nuclide selection or x-axis choice is
    just a re-evaluation of cached interpolators, not a re-solve.

    The ``output_time_evolution``-format download for this data is provided
    separately (``pyprimat.gui.app``'s ``_solve``), rendered alongside the
    final-abundances download.
    """
    names = run.abundance_names
    light_default = [n for n in _LIGHT_NUCLIDES if n in names]

    if "evolution_selection" not in st.session_state:
        st.session_state["evolution_selection"] = light_default

    preset_cols = st.columns([1, 1, 1, 3])
    if preset_cols[0].button("Light elements"):
        st.session_state["evolution_selection"] = light_default
    if preset_cols[1].button("All"):
        st.session_state["evolution_selection"] = list(names)
    if preset_cols[2].button("Clear"):
        st.session_state["evolution_selection"] = []

    selection = st.multiselect(
        "Nuclides to plot", options=names, key="evolution_selection",
    )

    use_temperature = st.radio(
        "X axis",
        ["Cosmic time t [s]", "Photon temperature T_γ [MeV]"],
        horizontal=True,
    ) == "Photon temperature T_γ [MeV]"

    fig = go.Figure()
    if selection:
        # Sample a qualitative colour per nuclide from the 'turbo' colormap,
        # matching the cm.turbo palette used for the ~59-nuclide large
        # network in AbundanceEvolution.ipynb.
        n = len(selection)
        positions = [i / (n - 1) if n > 1 else 0.0 for i in range(n)]
        colors = pcolors.sample_colorscale("turbo", positions)

        x_vals = run.T_of_t(_T_GRID) if use_temperature else _T_GRID
        for name, color in zip(selection, colors):
            y_vals = run.A[name] * run[name](_T_GRID)
            mask = y_vals > 0  # log-y axis cannot show zero/negative values
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=x_vals[mask], y=y_vals[mask],
                mode="lines", name=_nuclide_unicode(name), line=dict(color=color),
            ))

    x_title = "Photon temperature T_γ [MeV]" if use_temperature else "Cosmic time t [s]"
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Aᵢ Yᵢ  (per-baryon abundance)",
        xaxis_type="log",
        yaxis_type="log",
        legend_title="Nuclide",
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    if use_temperature:
        # T_gamma decreases monotonically with cosmic time, so the natural
        # data order runs from high to low T; reverse the axis so time still
        # flows left-to-right, matching the temperature ticks in
        # AbundanceEvolution.ipynb's add_temperature_axis helper.
        fig.update_xaxes(autorange="reversed")

    st.plotly_chart(fig, width="stretch")

    if not selection:
        st.caption("Select one or more nuclides above to plot their abundance evolution.")
