# -*- coding: utf-8 -*-
"""
primat.gui.panels
====================

The two result panels of the primat GUI:

* :func:`render_results_panel` -- the standard BBN ratios (Neff, Yp, D/H,
  He3/He4, He3/H, Li7/H) plus a per-nuclide table of final abundances.
* :func:`render_evolution_panel` -- an interactive ``A_i Y_i(t)`` plot with
  per-nuclide selection, paralleling ``notebooks/AbundanceEvolution.ipynb``.
* :func:`final_abundances_text` -- the ``output_final.dat``-format text for
  the download button rendered by ``primat.gui.app`` below the two panels
  (alongside the time-evolution download).

All three take an already-solved ``primat.PRIMAT`` instance (see
``primat.gui.app``, which calls ``run.solve()`` once and caches the
result).
"""
import html
import os
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from primat.constants import CONST
from primat.evolution import dump_evolution
from primat.network_data import nuclide_latex
from primat.plotting import abundance_evolution_curves
from primat.gui import custom_rates
from primat.gui.session_keys import SessionKeys


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
    "Li6oLi7": ".6e",
    "YCNO":    ".6e",
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
    "Li6oLi7": r"${}^{6}\text{Li}/({}^{7}\text{Li}+{}^{7}\text{Be})$",
    "YCNO":    r"$\text{CNO (mass)}$",
}


def render_results_panel(run, mc=None):
    """Render the final-abundances + standard-ratios panel.

    Parameters
    ----------
    run : primat.PRIMAT
        An already-solved ``PRIMAT`` instance (``run.solve()`` must have been
        called, e.g. by ``primat.gui.app._solve``).
    mc : primat.main.MCResult or None, optional
        Result of a quick :func:`primat.main.mc_uncertainty` call over
        the same parameters (``primat.gui.app._quick_mc``), or ``None`` if
        the "Quick MC uncertainty" toggle is off. When given, an extra
        "+/- 1 sigma (quick MC)" column is added to the "Standard ratios"
        table below, using ``mc[key].std`` formatted to the same precision as
        the central value.  The sample count shown in the header/caption is read
        back from the result (``len(mc[key].values)``) so it always matches the
        "MC samples" value the user chose.

    Layout
    ------
    1. A vertical table (Markdown, with LaTeX-rendered labels) of the 7
       headline observables from ``run.primat_results()`` (the 9-key results
       dict, ``main.py:751-761``; ``Omeganurel``/``OneOverOmeganunr`` are
       omitted here as niche neutrino-energy-density quantities), formatted to
       the precision required by ``CLAUDE.md``, plus an optional MC-uncertainty
       column (see ``mc`` above).
    2. A table of every tracked nuclide (``run.abundance_names``), with the
       nuclide name in standard isotope LaTeX notation (``nuclide_latex``),
       its mass number ``A``, charge ``Z``, and final mass-fraction abundance
       ``Y`` (``run.get_quantity(name)``).

    The ``output_final.txt``-format download for this table is provided
    separately by :func:`final_abundances_text`, rendered by
    ``primat.gui.app`` alongside the time-evolution download.
    """
    results = run.primat_results()

    st.subheader("Standard ratios")
    if mc is None:
        lines = ["| Quantity | Value |", "|---|---|"]
        lines += [
            f"| {_RATIO_LABELS[key]} | {format(results[key], fmt)} |"
            for key, fmt in _RATIO_FORMAT.items()
            if key in results
        ]
    else:
        # Sample count is read back from the result so the header matches the
        # "MC samples" value the user picked (the GUI lets it vary up to 100).
        n_mc = len(next(iter(mc._data.values())).values)
        lines = [f"| Quantity | Value | ± 1σ (quick MC, {n_mc} samples) |",
                 "|---|---|---|"]
        lines += [
            f"| {_RATIO_LABELS[key]} | {format(results[key], fmt)} "
            f"| {format(mc[key].std, fmt)} |"
            for key, fmt in _RATIO_FORMAT.items()
            if key in results
        ]
    st.markdown("\n".join(lines))
    if mc is not None:
        st.caption(
            f"{n_mc}-sample Monte Carlo over nuclear-rate and neutron-lifetime "
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


def final_abundances_text(run, mc=None):
    """Return the ``output_final.txt``-format text for every tracked nuclide.

    Same two-/three-column format as ``PRIMAT._write_final_result``
    (``output_final_result=True``), via :func:`primat.backend.dump_final_with_sigma`,
    built from the in-memory results so that flag is not needed just to
    export this table. ``Y`` is the final mass-fraction abundance of every
    nuclide in ``run.abundance_names`` (8 / ~59 for the large network, fewer
    with an amax cutoff).

    Parameters
    ----------
    mc : primat.main.MCResult or None, optional
        The cached "quick MC" result (see ``primat.gui.app._quick_mc``); when
        given and a nuclide name is one of its quantities, a third
        ``sigma_N<n>`` column is added for that nuclide (quick MC only covers
        the standard ratios, not nuclide Y's, so in practice this column is
        currently always empty for nuclides -- kept generic so it picks up
        nuclide sigmas automatically if quick MC's quantity set ever grows).
    """
    from primat.backend import dump_final_with_sigma

    names = run.abundance_names
    Y = {name: run.get_quantity(name) for name in names}
    if mc is None:
        return dump_final_with_sigma(names, Y)
    mc_names = set(mc.quantity_names())
    sigma = {name: mc[name].std for name in names if name in mc_names}
    num_mc = len(next(iter(mc._data.values())).values)
    if not sigma:
        return dump_final_with_sigma(names, Y)
    # Nuclides outside quick MC's quantity set get a 0.0 placeholder sigma
    # rather than being dropped, so every row keeps the same column count.
    sigma = {name: sigma.get(name, 0.0) for name in names}
    return dump_final_with_sigma(names, Y, sigma=sigma, num_mc=num_mc)


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
    by :meth:`primat.network_data.UpdateNuclearRates.describe_reactions`. Columns:

    * **Reaction** -- the readable ``a + b <-> c + d`` form with Unicode isotope
      symbols (e.g. ``²H + ²H ↔ ³He + n``);
    * **Source** -- the ``ref=`` provenance from the rate table's header line
      (e.g. ``And06``), or ``weak n<->p`` for the tabulated ``n__p`` weak rate.
    * **File** -- the rate table's filename (``data/nuclear/tables/<name>.txt``),
      or ``--`` for the weak ``n__p`` entry (its rates are supplied at solve time
      and have no on-disk table).

    Rendering uses a plain HTML ``<table>`` (via ``unsafe_allow_html``) rather
    than ``st.columns`` so the columns size to their content -- giving clean
    vertical/horizontal grid lines with no large trailing whitespace.  Because
    Streamlit only typesets ``$...$`` KaTeX inside *Markdown* (not inside
    injected HTML), the equations use Unicode super/subscripts
    (:func:`_equation_unicode`).  The rate tables themselves are downloadable
    just above this list (:func:`_render_reaction_downloads`).

    Parameters
    ----------
    run : primat.PRIMAT
        An already-solved ``PRIMAT`` instance; ``run.nucl`` carries the compiled
        networks.
    """
    reactions = run.nucl.describe_reactions()

    _render_reaction_downloads(run)

    st.subheader(f"{len(reactions)} reactions")
    st.caption(
        "Full reaction set of the low-temperature solver. The MT era uses a "
        "fixed 18-reaction subset of these."
    )

    # Content-sized HTML table with collapsed borders -> crisp grid lines and no
    # proportional-column whitespace.  ``html.escape`` guards the few sources
    # that contain "&" (e.g. "CF88&MF89  (analytic, PRIMAT-main.m)").
    css = (
        "<style>"
        "table.primat-rxn{border-collapse:collapse;margin:0.25rem 0 0.75rem;}"
        "table.primat-rxn th,table.primat-rxn td"
        "{border:1px solid rgba(128,128,128,0.5);padding:4px 12px;text-align:left;}"
        "table.primat-rxn th{font-weight:600;}"
        "table.primat-rxn td.rxn-eq{white-space:nowrap;}"
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
        "<table class='primat-rxn'>"
        "<thead><tr><th>Reaction</th><th>Source</th><th>File</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    st.markdown(css + table, unsafe_allow_html=True)


def _render_reaction_downloads(run):
    """Render the "Export this network" download section.

    Placed at the top of the Reactions tab (:func:`render_reactions_panel`),
    *before* the (potentially long, e.g. ~430-row for the large network)
    reaction list, so it is visible without scrolling.

    Always available -- not just for an actual customisation (a run with no
    "Create/modify network" override gets an empty
    ``{"removed": [], "replaced": {}, "added": {}}``, exporting the plain
    shipped network's reactions/tables verbatim) -- so a user can always grab
    a self-contained, re-importable snapshot of whatever just ran, even an
    unmodified ``small``/``large``. ``custom_rates.export_zip`` itself doesn't
    care whether anything was actually changed.

    Parameters
    ----------
    run : primat.PRIMAT
        An already-solved ``PRIMAT`` instance.
    """
    custom_network = st.session_state.get(SessionKeys.run_custom_network_dict) or {
        "removed": [], "replaced": {}, "added": {},
    }
    active = st.session_state.get(SessionKeys.active_custom_network)
    title = active["title"] if active else (run.cfg.network if run.cfg.network != "large"
                                            else "large")
    st.markdown("**Export this network**")
    kept_names = [name for name, equation, source, file
                  in run.nucl.describe_reactions()
                  if name not in ("n__p", "n__p")]
    try:
        zip_bytes = custom_rates.export_zip(run.cfg, custom_network, kept_names,
                                            network_filename=title)
    except Exception as exc:
        st.warning(f"Could not build the network export: {exc}")
    else:
        st.download_button(
            f"Download network (zip)",
            data=zip_bytes,
            file_name=f"{title}.zip",
            mime="application/zip",
            key="dl_custom_network",
            help=f"networks/{title}.txt + tables/<name>/<filename> for every "
                 "reaction in this run, re-importable from the sidebar's "
                 "\"Import custom network\" button -- even for an unmodified "
                 "network, this is a self-contained snapshot of exactly the "
                 "reactions/tables used.",
        )


def weak_rates_text(cfg, background):
    """Return a TSV string (T[K], Gamma_nTOp[1/s], Gamma_pTOn[1/s]) for n↔p rates.

    Evaluates the normalised forward and backward weak rates on a 500-point
    log-spaced grid from ``cfg.T_end`` to ``cfg.T_start_cosmo`` (both in
    Kelvin), covering the full BBN temperature range.

    Parameters
    ----------
    cfg : primat.config.PRIMATConfig
    background : primat.background.StandardBackground or CustomBackground
        A Python background object (e.g. ``primat.gui.app._build_background``'s
        result) -- built separately from whichever backend actually solved
        the BBN network, since only the Python background exposes
        ``weak_nTOp_frwrd``/``weak_nTOp_bkwrd``.

    Returns
    -------
    str
        Tab-separated text with one header line and 500 data rows.
    """
    T_K = np.logspace(np.log10(cfg.T_end), np.log10(cfg.T_start_cosmo), 500)
    frwrd = background.weak_nTOp_frwrd(T_K)
    bkwrd = background.weak_nTOp_bkwrd(T_K)
    lines = ["T[K]\tGamma_nTOp[1/s]\tGamma_pTOn[1/s]"]
    for t, f, b in zip(T_K, frwrd, bkwrd):
        lines.append(f"{t:.6e}\t{f:.6e}\t{b:.6e}")
    return "\n".join(lines)


def render_downloads_panel(run, mc=None, background=None):
    """Render the Output tab: the standard, network-independent output files.

    Collects every file a user might want to export from a completed run in one
    place (rather than scattering download buttons under the result panels):

    * **output_final.txt** -- the final abundances in the ``output_final.dat``
      text format (:func:`final_abundances_text`).
    * **output_time_evolution.tsv** -- the full ``A_i Y_i(t)`` time series, in
      the unified schema (``primat.evolution``, ``PRIMAT.md`` S7.2), built
      lazily here via :func:`primat.evolution.dump_evolution` on
      ``run.evolution`` -- no disk I/O happens until this download button is
      actually clicked (``PRIMAT.md`` S7.5). Populated on either backend.
    * **output_background.tsv**, **nTOp_total.tsv** (weak rates) -- built from
      ``background``, a separately-constructed Python
      ``StandardBackground``/``CustomBackground`` (``primat.gui.app``'s
      ``_build_background``), so these are available regardless of which
      backend actually solved the BBN network (the C backend has no
      equivalent of either file -- see ``CLAUDE.md``).
    * **decays.txt** (large network only) -- the consolidated beta-decay /
      electron-capture rate table used by the large network
      (``data/nuclear/tables/decays.txt``).

    The per-reaction "Custom network" export and "Reaction rate tables"
    downloads instead live at the top of the Reactions tab
    (:func:`_render_reaction_downloads`), visible immediately above the
    (potentially long) reaction list rather than tucked away in this tab.

    * **output_mc_samples.tsv** -- every quick-MC sample drawn (one column per
      quantity, one row per sample), via :func:`primat.backend.dump_mc_samples`
      -- only offered when ``mc`` is given, i.e. a quick MC was actually run.

    Parameters
    ----------
    run : primat.PRIMAT or primat.gui.run_view.GuiRun
        An already-solved run, with ``output_time_evolution=True`` so that
        ``run.evolution`` is populated.
    mc : primat.main.MCResult or None, optional
        The cached "quick MC" result (``primat.gui.app._quick_mc``), or
        ``None`` if no quick MC has been run for this result -- in which case
        no MC-samples download button is shown, and ``final_abundances_text``
        falls back to its plain two-column form.
    background : primat.background.StandardBackground or CustomBackground or None, optional
        A separately-built Python background (``primat.gui.app._build_background``),
        used only for the ``output_background.tsv``/``nTOp_total.tsv``
        downloads. ``None`` (e.g. if building it failed) skips those two
        downloads with an explanatory note.
    """
    # Each file gets its own subsection title directly above its download
    # button (rather than a blanket "Output"/"Output files" header), stacked
    # one below another.
    def _file_download(title, label, data, file_name, mime, key, help=None):
        st.markdown(f"**{title}**")
        st.download_button(
            label, data=data, file_name=file_name, mime=mime,
            key=key, help=help,
        )

    _file_download(
        "Final abundances", "output_final.txt",
        data=final_abundances_text(run, mc=mc), file_name="output_final.txt",
        mime="text/plain", key="dl_final",
    )
    if mc is not None:
        from primat.backend import dump_mc_samples

        n_mc = len(next(iter(mc._data.values())).values)
        _file_download(
            "Quick MC samples", "output_mc_samples.tsv",
            data=dump_mc_samples(mc), file_name="output_mc_samples.tsv",
            mime="text/tab-separated-values", key="dl_mc_samples",
            help=f"{n_mc} quick-MC samples, one column per quantity.",
        )
    _file_download(
        "Abundances time evolution", "output_time_evolution.tsv",
        data=dump_evolution(run.evolution), file_name="output_time_evolution.tsv",
        mime="text/tab-separated-values", key="dl_evolution",
    )
    if background is not None:
        _file_download(
            "Background evolution", "output_background.tsv",
            data=background.time_evolution_text(run.cfg.output_n_points),
            file_name="output_background.tsv",
            mime="text/tab-separated-values", key="dl_background",
        )
        _file_download(
            "Weak rates", "nTOp_total.tsv",
            data=weak_rates_text(run.cfg, background), file_name="nTOp_total.tsv",
            mime="text/tab-separated-values", key="dl_weak_rates",
            help="Total normalised n↔p weak rates: T[K], Γ_{n→p}[s⁻¹], Γ_{p→n}[s⁻¹].",
        )
    else:
        st.caption(
            "*(Background evolution / weak-rates downloads are unavailable -- "
            "see the error above.)*"
        )
    if run.cfg.network == "large":
        decays_path = os.path.join(
            run.cfg._resolved_data_dir, "nuclear", "tables", "decays.txt"
        )
        try:
            with open(decays_path, "rb") as fh:
                decays_data = fh.read()
            _file_download(
                "Decay rates", "decays.txt",
                data=decays_data, file_name="decays.txt",
                mime="text/plain", key="dl_decays",
            )
        except OSError:
            st.warning("`decays.txt` is unavailable.")


# ---------------------------------------------------------------------------
# Abundance time-evolution panel
# ---------------------------------------------------------------------------

# Default nuclide selection and time grid, matching
# notebooks/AbundanceEvolution.ipynb (species_small/large lists and
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


# Matplotlib line styles (as returned by primat.plotting.nuclide_styles) ->
# Plotly ``line.dash`` values, so the GUI uses the *same* per-isotope dashing
# as the notebooks.  Named styles map to Plotly's named dashes; the custom
# (offset, on-off-tuple) patterns map to Plotly's "Npx,Mpx,..." dash strings.
_NAMED_DASH = {
    "solid":   "solid",
    "dashed":  "dash",
    "dashdot": "dashdot",
    "dotted":  "dot",
}


def _plotly_dash(linestyle):
    """Translate a matplotlib linestyle into a Plotly ``line.dash`` value.

    Parameters
    ----------
    linestyle : str or tuple
        Either a matplotlib named style ("solid"/"dashed"/"dashdot"/"dotted")
        or an explicit ``(offset, (on, off, ...))`` dash tuple (the finer
        patterns :data:`primat.plotting.LINESTYLES` uses for elements with
        many isotopes).

    Returns
    -------
    str
        A Plotly dash specification: a named dash for the four standard styles,
        or a comma-separated pixel pattern (e.g. ``"5px,1px"``) for tuples.
    """
    if isinstance(linestyle, str):
        return _NAMED_DASH.get(linestyle, "solid")
    # linestyle is (offset, (on, off, on, off, ...)); Plotly ignores the offset
    # and takes the on/off lengths as a "Npx,Mpx,..." string.
    _offset, pattern = linestyle
    return ",".join(f"{int(v)}px" for v in pattern)


def render_evolution_panel(run):
    """Render the interactive ``A_i Y_i(t)`` abundance-evolution panel.

    Parameters
    ----------
    run : primat.PRIMAT
        An already-solved ``PRIMAT`` instance.

    Notes
    -----
    Mirrors ``notebooks/AbundanceEvolution.ipynb``: for each selected nuclide
    ``name``, plots ``A_i Y_i(t)`` (the mass fraction weighted by mass number,
    i.e. the per-baryon abundance) on a log-log Plotly figure, using
    :func:`primat.plotting.abundance_evolution_curves` -- the same
    backend-agnostic curve-computation helper the notebooks use, built on
    ``run.evolution`` (an ``EvolutionResult``, populated on either backend)
    rather than any live continuous-time interpolator.

    The ``output_time_evolution``-format download for this data is provided
    separately (``primat.gui.app``'s ``_solve``), rendered alongside the
    final-abundances download.
    """
    names = run.abundance_names
    light_default = [n for n in _LIGHT_NUCLIDES if n in names]
    default_selection = list(names)

    if "evolution_selection" not in st.session_state:
        st.session_state["evolution_selection"] = default_selection

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
        ["Cosmic time t [s]", "Photon temperature T_γ [K]"],
        horizontal=True,
    ) == "Photon temperature T_γ [K]"

    t_grid = _T_GRID

    # Per-nuclide (t, A_i Y_i, color, linestyle, label) curves, computed by the
    # same backend-agnostic helper the notebooks use (primat.plotting), so
    # this works whether `run` was solved by the C or Python backend.
    curves = abundance_evolution_curves(run.evolution, run.A, names, t_grid)

    fig = go.Figure()
    for name in selection:
        t_masked, y_masked, color, linestyle, _label = curves[name]
        if t_masked.size == 0:
            continue
        # T_of_t returns MeV; convert to Kelvin for the x axis.
        x_vals = run.T_of_t(t_masked) * CONST.MeV_to_Kelvin if use_temperature else t_masked
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_masked,
            mode="lines", name=_nuclide_unicode(name),
            line=dict(color=color, dash=_plotly_dash(linestyle)),
        ))

    # Plain unicode, not "$...$" LaTeX: st.plotly_chart doesn't load the
    # MathJax script LaTeX rendering needs, so a "$...$" title would just
    # show the literal dollar-quoted source instead of rendering -- matches
    # the T_γ notation already used in the radio label above.
    x_title = "T_γ [K]" if use_temperature else "Cosmic time t [s]"
    # Y-axis floor: large network has heavy isotopes with abundances as low as
    # ~1e-45, so we clip the range at 1e-50 to keep them visible; for the
    # light small/amax-restricted networks 1e-36 is sufficient and avoids blank space.
    y_floor = -50 if run.cfg.network == "large" else -36
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Aᵢ Yᵢ  (per-baryon abundance)",
        xaxis_type="log",
        yaxis_type="log",
        yaxis_range=[y_floor, 0],
        legend_title="Nuclide",
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    # "power" forces ticks like 10^3 instead of plotly's default mix of plain
    # digits ("1000") and exponents -- the user wants powers of 10 explicit
    # everywhere on this log axis.
    fig.update_xaxes(exponentformat="power")
    fig.update_yaxes(exponentformat="power")
    if use_temperature:
        # T_gamma decreases monotonically with cosmic time, so the natural
        # data order runs from high to low T; reverse the axis so time still
        # flows left-to-right, matching the temperature ticks in
        # AbundanceEvolution.ipynb's add_temperature_axis helper.
        fig.update_xaxes(autorange="reversed")

    st.plotly_chart(fig, width="stretch")

    if not selection:
        st.caption("Select one or more nuclides above to plot their abundance evolution.")
