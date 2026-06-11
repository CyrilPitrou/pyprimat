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
import time

import streamlit as st

from pyprimat import PyPR
from pyprimat.gui import panels
from pyprimat.gui.params_form import render_sidebar_form


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
    pyprimat.PyPR
        A solved instance: ``run.PyPRresults()``, ``run.abundance_names``,
        and ``run[name](t)`` are all ready to use without triggering further
        computation.

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
    """
    run = PyPR(params=dict(params_items))
    run.solve()
    return run


def main():
    st.title("⚛️ PyPRIMAT")
    st.caption(
        "Big Bang Nucleosynthesis abundances — interactive front end for "
        "`pyprimat.PyPR`"
    )

    params = render_sidebar_form()
    run_clicked = st.sidebar.button("Run BBN", type="primary", width="stretch")

    if run_clicked:
        # Snapshot the current form state; subsequent reruns (e.g. from
        # ticking a nuclide checkbox) reuse this snapshot via
        # st.session_state rather than re-triggering a solve with whatever
        # the sidebar currently shows.
        st.session_state["params"] = dict(params)

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
            run = _solve(params_items)
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

    tab_results, tab_evolution = st.tabs(["Final abundances", "Abundance evolution"])
    with tab_results:
        panels.render_results_panel(run)
    with tab_evolution:
        panels.render_evolution_panel(run)


if __name__ == "__main__":
    main()
