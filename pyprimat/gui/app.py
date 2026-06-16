# -*- coding: utf-8 -*-
"""
pyprimat.gui.app
=================

Streamlit entry point for the PyPRIMAT GUI (GUI.md §3, §6).

Run with::

    pyprimat-gui                          # after `pip install ".[gui]"`
    streamlit run pyprimat/gui/app.py     # from a source checkout

Layout: a sidebar parameter form (built by
:func:`pyprimat.gui.params_form.render_sidebar_form`) plus a "Run BBN"
button; the main area shows the two result panels from
:mod:`pyprimat.gui.panels` once a run has completed. The "params dict ->
PyPR -> results" contract is identical to ``runfiles/PyPRIMAT_run.py`` and
the ``pyprimat`` console script (``pyprimat/cli.py``).
"""
import os
import tempfile
import time

import streamlit as st

from pyprimat import PyPR
from pyprimat.gui import panels
from pyprimat.gui.params_form import render_sidebar_form
from pyprimat.main import mc_uncertainty
from pyprimat.gui.panels import _RATIO_FORMAT


st.set_page_config(
    page_title="PyPRIMAT",
    page_icon="⚛️",  # atom symbol
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _solve(params_items):
    """Build a ``PyPR`` instance for ``params`` and solve the network.

    Parameters
    ----------
    params_items : tuple of (str, value) pairs
        A hashable, order-independent-by-construction encoding of a
        ``params`` dict (see ``app.main``, which sorts the items before
        calling this function), required because ``st.cache_resource`` keys
        its cache on the function arguments.

    Returns
    -------
    (pyprimat.PyPR, str, str)
        The solved instance -- ``run.PyPRresults()``, ``run.abundance_names``,
        and ``run[name](t)`` are all ready to use without triggering further
        computation -- together with the contents of the nuclear time-evolution
        TSV (``output_time_evolution`` format, see
        ``nuclear_network.py:NuclearNetwork._write_time_evolution``) and of the
        background time-evolution TSV (``output_background_evolution`` format,
        see ``background.py:Background.write_time_evolution``), both as strings.
        ``_write_time_evolution`` derives its ``Y<species>`` columns from
        ``self.abundance_names``, so this works the same way for all three
        networks (8 / 12 / ~59 nuclide columns).

    Notes
    -----
    Constructing ``PyPR`` (loading rate tables, computing the cosmological
    background and weak rates) and then ``solve()``-ing the HT/MT/LT network
    are both potentially expensive (seconds for the default config, much
    longer at reference precision or for the large network). Caching on the
    exact parameter set means re-running with unchanged parameters --
    e.g. just toggling which nuclides are plotted in the evolution panel --
    is instant.  ``cache_resource`` (rather than ``cache_data``) is required
    because ``PyPR`` instances hold live SciPy interpolators that are not
    picklable.

    Both TSV files are produced by pointing the respective output path at a
    temporary file, which is read back into memory and removed immediately --
    so the cached result carries the data itself rather than a path that a
    later, differently-parametrised solve could overwrite.
    """
    params = dict(params_items)

    fd_evo, tmp_evo = tempfile.mkstemp(suffix=".tsv", prefix="pyprimat_evolution_")
    os.close(fd_evo)
    fd_bg, tmp_bg = tempfile.mkstemp(suffix=".tsv", prefix="pyprimat_background_")
    os.close(fd_bg)
    try:
        run = PyPR(params=dict(params,
                               output_time_evolution=True,
                               output_file=tmp_evo,
                               output_background_evolution=True,
                               output_background_file=tmp_bg))
        run.solve()
        with open(tmp_evo) as f:
            time_evolution_tsv = f.read()
        with open(tmp_bg) as f:
            background_tsv = f.read()
    finally:
        os.remove(tmp_evo)
        os.remove(tmp_bg)
    return run, time_evolution_tsv, background_tsv


def _quick_mc(params_items, num_mc):
    """Run a quick :func:`pyprimat.main.mc_uncertainty` for the standard ratios.

    Parameters
    ----------
    params_items : tuple of (str, value) pairs
        Same hashable encoding of ``params`` as :func:`_solve`.
    num_mc : int
        Number of Monte Carlo samples requested (the GUI caps this at 100).

    Returns
    -------
    pyprimat.main.MCResult
        Indexed by the 7 ``_RATIO_FORMAT`` keys (Neff, YPBBN, YPCMB, DoH,
        He3oH, He3oHe4, Li7oH); each entry has ``.mean`` and ``.std``.

    Notes
    -----
    A few dozen samples is deliberately small (a handful of seconds rather than
    minutes) and gives only a *quick, noisy* estimate of the uncertainty --
    see the "Quick MC uncertainty" toggle's help text in
    ``params_form.render_sidebar_form``.

    **Incremental reuse.** The previous result (for the *same* parameters) is
    kept in ``st.session_state`` and passed to :func:`mc_uncertainty` as
    ``prev``.  Because sample ``i`` is fully determined by ``seed + i``, raising
    the sample count only solves the additional samples (e.g. 30 -> 50 runs 20
    new ones); lowering it just truncates the stored samples without solving
    anything.  We deliberately do *not* use ``st.cache_resource`` here so that a
    larger request can extend the smaller cached one instead of being a plain
    cache miss that recomputes everything.
    """
    cache = st.session_state.get("_quick_mc_cache")
    # Reuse the cached MCResult as a starting point only when it was computed
    # for exactly these parameters; mc_uncertainty itself re-checks seed and
    # quantities before trusting ``prev``.
    prev = cache[1] if (cache is not None and cache[0] == params_items) else None
    mc = mc_uncertainty(num_mc, list(_RATIO_FORMAT),
                        params=dict(params_items), seed=0, prev=prev)
    st.session_state["_quick_mc_cache"] = (params_items, mc)
    return mc


def main():
    st.title("⚛️ PyPRIMAT")
    st.caption(
        "Big Bang Nucleosynthesis abundances — interactive front end for "
        "`pyprimat.PyPR`"
    )

    params, quick_mc, mc_samples = render_sidebar_form()
    run_clicked = st.sidebar.button("Run BBN", type="primary", width="stretch")
    _render_footer()

    if run_clicked:
        # Snapshot the current form state; subsequent reruns (e.g. from
        # ticking a nuclide checkbox) reuse this snapshot via
        # st.session_state rather than re-triggering a solve with whatever
        # the sidebar currently shows.
        st.session_state["params"] = dict(params)
        st.session_state["quick_mc"] = quick_mc
        st.session_state["mc_samples"] = mc_samples

    stored_params = st.session_state.get("params")
    if stored_params is None:
        st.info("Set parameters in the sidebar, then click **Run BBN**.")
        return

    # st.cache_resource requires hashable arguments; a sorted tuple of items
    # is both hashable and order-independent (so key ordering in the params
    # dict never causes a spurious cache miss).
    params_items = tuple(sorted(stored_params.items()))

    try:
        with st.spinner("Solving the BBN network…"):
            t0 = time.time()
            run, time_evolution_tsv, background_tsv = _solve(params_items)
            elapsed = time.time() - t0
    except Exception as exc:
        # PyPRConfig validates e.g. `amax`/`network` and the
        # spectral_distortions/incomplete_decoupling/analytic_distortions
        # flag combination, raising ValueError on bad input -- surface that
        # (and any other failure) as a clean message rather than a traceback.
        st.error(f"PyPRIMAT run failed: {exc}")
        return

    network = stored_params.get("network", "small")
    omegabh2 = stored_params.get("Omegabh2", 0.022425)
    st.caption(
        f"network = `{network}`, Ωᵇ h² = {omegabh2:g} "
        f"(solved in {elapsed:.2f} s)"
    )

    mc = None
    if st.session_state.get("quick_mc", False):
        num_mc = st.session_state.get("mc_samples", 30)
        with st.spinner(f"Running {num_mc}-sample quick MC uncertainty…"):
            mc = _quick_mc(params_items, num_mc)

    tab_results, tab_evolution, tab_reactions, tab_downloads = st.tabs(
        ["Final abundances", "Abundance evolution", "Reactions", "Downloads"])
    with tab_results:
        panels.render_results_panel(run, mc=mc)
    with tab_evolution:
        panels.render_evolution_panel(run)
    with tab_reactions:
        panels.render_reactions_panel(run)
    with tab_downloads:
        panels.render_downloads_panel(run, time_evolution_tsv, background_tsv)


def _render_footer():
    """Sidebar attribution footer, shown below the parameter form."""
    st.sidebar.caption(
        "PyPRIMAT and this GUI are developed by "
        "[Cyril Pitrou](https://www2.iap.fr/users/pitrou/)."
    )


if __name__ == "__main__":
    main()
