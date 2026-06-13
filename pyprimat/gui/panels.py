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
import html
import os
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
    "He3oHe4": r"$({}^{3}\text{He}+\text{T})/{}^{4}\text{He}$",
    "Li7oH":   r"$({}^{7}\text{Li}+{}^{7}\text{Be})/\text{H}$",
}


def render_results_panel(run, mc=None):
    """Render the final-abundances + standard-ratios panel.

    Parameters
    ----------
    run : pyprimat.PyPR
        An already-solved ``PyPR`` instance (``run.solve()`` must have been
        called, e.g. by ``pyprimat.gui.app._solve``).
    mc : pyprimat.main.MCResult or None, optional
        Result of a 30-sample :func:`pyprimat.main.mc_uncertainty` call over
        the same parameters (``pyprimat.gui.app._quick_mc``), or ``None`` if
        the "Quick MC uncertainty" toggle is off. When given, an extra
        "+/- 1 sigma (quick MC)" column is added to the "Standard ratios"
        table below, using ``mc[key].std`` formatted to the same precision as
        the central value.

    Layout
    ------
    1. A vertical table (Markdown, with LaTeX-rendered labels) of the 7
       headline observables from ``run.PyPRresults()`` (the 9-key results
       dict, ``main.py:751-761``; ``Omeganurel``/``OneOverOmeganunr`` are
       omitted here as niche neutrino-energy-density quantities), formatted to
       the precision required by ``CLAUDE.md``, plus an optional MC-uncertainty
       column (see ``mc`` above).
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
    if mc is None:
        lines = ["| Quantity | Value |", "|---|---|"]
        lines += [
            f"| {_RATIO_LABELS[key]} | {format(results[key], fmt)} |"
            for key, fmt in _RATIO_FORMAT.items()
        ]
    else:
        lines = ["| Quantity | Value | ± 1σ (quick MC, 30 samples) |", "|---|---|---|"]
        lines += [
            f"| {_RATIO_LABELS[key]} | {format(results[key], fmt)} "
            f"| {format(mc[key].std, fmt)} |"
            for key, fmt in _RATIO_FORMAT.items()
        ]
    st.markdown("\n".join(lines))
    if mc is not None:
        st.caption(
            "30-sample Monte Carlo over nuclear-rate and neutron-lifetime "
            "uncertainties -- a quick, noisy estimate, not a "
            "publication-quality error bar."
        )

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
# Reactions panel
# ---------------------------------------------------------------------------

def _equation_unicode(equation):
    """Render a plain ``a + b <-> c + d`` equation with Unicode nuclide symbols.

    Each whitespace-separated token that is a nuclide name (everything except
    the ``+`` and ``<->`` separators) is passed through :func:`_nuclide_unicode`,
    so ``"H2 + H2 <-> He3 + n"`` becomes ``"²H + ²H ↔ ³He + n"``.

    Unlike the LaTeX form used elsewhere, this produces plain text suitable for
    a raw HTML ``<table>`` cell: Streamlit's KaTeX support only typesets
    ``$...$`` inside Markdown, *not* inside HTML injected via
    ``unsafe_allow_html``, so the reactions table uses Unicode super/subscripts
    instead (matching the Plotly legends in :func:`render_evolution_panel`).
    """
    out = []
    for tok in equation.split():
        if tok == "+":
            out.append("+")
        elif tok == "<->":
            out.append("↔")
        else:
            out.append(_nuclide_unicode(tok))
    return " ".join(out)


def render_reactions_panel(run):
    """Render the table of loaded reactions and their data sources.

    Lists every reaction integrated by the chosen network's LT solver (the full
    selected set; the MT era uses only a fixed 18-reaction subset), as produced
    by :meth:`pyprimat.nuclear.UpdateNuclearRates.describe_reactions`. Columns:

    * **Reaction** -- the readable ``a + b <-> c + d`` form with Unicode isotope
      symbols (e.g. ``²H + ²H ↔ ³He + n``);
    * **Source** -- the ``ref=`` provenance from the rate table's header line
      (e.g. ``And06``), or ``weak n<->p`` for the tabulated ``nTOp`` weak rate.
    * **File** -- the rate table's filename (``rates/nuclear/tables/<name>.txt``),
      or ``--`` for the weak ``nTOp`` entry (its rates are supplied at solve time
      and have no on-disk table).

    Rendering uses a plain HTML ``<table>`` (via ``unsafe_allow_html``) rather
    than ``st.columns`` so the columns size to their content -- giving clean
    vertical/horizontal grid lines with no large trailing whitespace.  Because
    Streamlit only typesets ``$...$`` KaTeX inside *Markdown* (not inside
    injected HTML), the equations use Unicode super/subscripts
    (:func:`_equation_unicode`).  The rate tables themselves are downloadable
    from the **Downloads** tab (:func:`render_downloads_panel`).

    Parameters
    ----------
    run : pyprimat.PyPR
        An already-solved ``PyPR`` instance; ``run.nucl`` carries the compiled
        networks.
    """
    reactions = run.nucl.describe_reactions()
    st.subheader(f"Reactions ({len(reactions)} in the {run.cfg.network} network)")
    st.caption(
        "Full reaction set of the low-temperature solver. The MT era uses a "
        "fixed 18-reaction subset of these. Sources are the `ref=` labels from "
        "each rate table header; download the rate tables from the Downloads tab."
    )

    # Content-sized HTML table with collapsed borders -> crisp grid lines and no
    # proportional-column whitespace.  ``html.escape`` guards the few sources
    # that contain "&" (e.g. "CF88&MF89  (analytic, PRIMAT-main.m)").
    css = (
        "<style>"
        "table.pyprimat-rxn{border-collapse:collapse;margin:0.25rem 0 0.75rem;}"
        "table.pyprimat-rxn th,table.pyprimat-rxn td"
        "{border:1px solid rgba(128,128,128,0.5);padding:4px 12px;text-align:left;}"
        "table.pyprimat-rxn th{font-weight:600;}"
        "table.pyprimat-rxn td.rxn-eq{white-space:nowrap;}"
        "</style>"
    )
    rows = [
        "<tr>"
        f"<td class='rxn-eq'>{html.escape(_equation_unicode(equation))}</td>"
        f"<td>{html.escape(source)}</td>"
        f"<td>{html.escape(os.path.basename(file)) if file else '--'}</td>"
        "</tr>"
        for name, equation, source, file in reactions
    ]
    table = (
        "<table class='pyprimat-rxn'>"
        "<thead><tr><th>Reaction</th><th>Source</th><th>File</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    st.markdown(css + table, unsafe_allow_html=True)


def render_downloads_panel(run, time_evolution_tsv):
    """Render the Downloads tab: standard output files + per-reaction rate tables.

    Collects every file a user might want to export from a completed run in one
    place (rather than scattering download buttons under the result panels):

    * **output_final.dat** -- the final abundances in the ``output_final.dat``
      text format (:func:`final_abundances_text`).
    * **output_time_evolution.tsv** -- the full ``A_i Y_i(t)`` time series in the
      ``output_time_evolution`` format, produced once at solve time by
      ``pyprimat.gui.app._solve`` and passed in as ``time_evolution_tsv``.
    * **Reaction rate tables** -- the ``rates/nuclear/tables/<name>.txt`` rate
      table for any reaction in the loaded network.  An in-table download link is
      not possible (Streamlit's HTML sanitiser strips ``data:`` hrefs and
      browsers block ``file://`` ones), and the large network has ~433
      reactions, so a single "pick a reaction → download" selectbox is used
      rather than one button per reaction.

    Parameters
    ----------
    run : pyprimat.PyPR
        An already-solved ``PyPR`` instance.
    time_evolution_tsv : str
        Contents of the time-evolution TSV (see ``app._solve``).
    """
    st.subheader("Downloads")

    # 1. Standard BBN output files, side by side.
    st.markdown("**Output files**")
    out_cols = st.columns(2)
    out_cols[0].download_button(
        "Final abundances (output_final.dat)",
        data=final_abundances_text(run),
        file_name="output_final.dat",
        mime="text/plain",
        width="stretch",
        key="dl_final",
    )
    out_cols[1].download_button(
        "Time evolution (output_time_evolution.tsv)",
        data=time_evolution_tsv,
        file_name="output_time_evolution.tsv",
        mime="text/tab-separated-values",
        width="stretch",
        key="dl_evolution",
    )

    # 2. Per-reaction rate tables.  Only reactions with a rate table on disk are
    # offered (everything but the weak nTOp pseudo-reaction, whose rates are
    # supplied at solve time).
    st.markdown("**Reaction rate tables**")
    downloadable = {
        f"{name}  ({os.path.basename(file)})": file
        for name, equation, source, file in run.nucl.describe_reactions()
        if file is not None
    }
    if not downloadable:
        st.caption("This network has no downloadable rate tables.")
        return
    choice = st.selectbox(
        "Rate table", list(downloadable), key="ratefile_choice"
    )
    path = downloadable[choice]
    basename = os.path.basename(path)
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        st.warning(f"Rate table `{basename}` is unavailable.")
        return
    st.download_button(
        label=f"Download {basename}",
        data=data,
        file_name=basename,
        mime="text/plain",
        key="ratefile_download",
    )


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
