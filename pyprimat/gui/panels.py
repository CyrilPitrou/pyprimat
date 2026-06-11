# -*- coding: utf-8 -*-
"""
pyprimat.gui.panels
====================

The two result panels of the PyPRIMAT GUI (GUI.md §4-5):

* :func:`render_results_panel` -- the standard BBN ratios (Neff, Yp, D/H,
  He3/He4, He3/H, Li7/H) plus a per-nuclide table of final abundances.
* :func:`render_evolution_panel` -- an interactive ``A_i Y_i(t)`` plot with
  per-nuclide selection, paralleling ``notebooks/AbundanceEvolution.ipynb``.

Both functions take an already-solved ``pyprimat.PyPR`` instance (see
``pyprimat.gui.app``, which calls ``run.solve()`` once and caches the
result).
"""
import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st


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

_RATIO_LABELS = {
    "Neff":    "N_eff",
    "YPBBN":   "Y_P (BBN)",
    "YPCMB":   "Y_P (CMB)",
    "DoH":     "D / H",
    "He3oH":   "(³He+T) / H",
    "He3oHe4": "³He / ⁴He",
    "Li7oH":   "(⁷Li+⁷Be) / H",
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
    1. A row of ``st.metric`` cards for the 7 headline observables from
       ``run.PyPRresults()`` (the 9-key results dict, ``main.py:751-761``;
       ``Omeganurel``/``OneOverOmeganunr`` are omitted here as niche
       neutrino-energy-density quantities), formatted to the precision
       required by ``CLAUDE.md``.
    2. A sortable table of every tracked nuclide (``run.abundance_names``)
       with its mass number ``A``, charge ``Z``, and final mass-fraction
       abundance ``Y`` (``run.get_quantity(name)``).
    3. A download button producing the same two-column ``nuclide  Y`` text
       format as ``PyPR._write_final_result`` (``output_final_result=True``),
       without requiring that flag or any disk write.
    """
    results = run.PyPRresults()

    st.subheader("Standard ratios")
    cols = st.columns(len(_RATIO_FORMAT))
    for col, (key, fmt) in zip(cols, _RATIO_FORMAT.items()):
        col.metric(_RATIO_LABELS[key], format(results[key], fmt))

    st.subheader("Final abundances")
    names = run.abundance_names
    rows = [
        {
            "nuclide": name,
            "A": run.A[name],
            "Z": run.Z[name],
            "Y": run.get_quantity(name),
        }
        for name in names
    ]
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Y": st.column_config.NumberColumn("Y", format="%.6e"),
        },
    )

    # Same two-column format as PyPR._write_final_result's output_final_file,
    # produced from the in-memory results so output_final_result=True is not
    # needed just to inspect/export this table.
    text_lines = [f"# {'nuclide':<12}Y"]
    text_lines += [f"{row['nuclide']:<14}{row['Y']:.6e}" for row in rows]
    st.download_button(
        "Download final abundances (output_final.dat format)",
        data="\n".join(text_lines) + "\n",
        file_name="output_final.dat",
        mime="text/plain",
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
                mode="lines", name=name, line=dict(color=color),
            ))

    x_title = "Photon temperature T_γ [MeV]" if use_temperature else "Cosmic time t [s]"
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="A_i · Y_i  (per-baryon abundance)",
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
