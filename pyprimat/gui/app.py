# -*- coding: utf-8 -*-
"""
pyprimat.gui.app
=================

Streamlit entry point for the PyPRIMAT GUI.

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
import json
import os
import tempfile
import time
import types

import streamlit as st

from pyprimat import PyPR
from pyprimat.config import PyPRConfig
from pyprimat.gui import panels
from pyprimat.gui.params_form import render_sidebar_form
from pyprimat.main import mc_uncertainty
from pyprimat.gui.panels import _RATIO_FORMAT
from pyprimat.gui.session_keys import SessionKeys
from pyprimat.network_data import UpdateNuclearRates
from pyprimat.weak_rates.cache import thermal_cache_exists


st.set_page_config(
    page_title="PyPRIMAT",
    page_icon="⚛️",  # atom symbol
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _solve(params_items):
    """Build a ``PyPR`` instance for ``params`` and solve the network.

    Shows a "Computing weak rates…"/"Solving the BBN network…" progress
    placeholder (created internally via ``st.empty()``, not passed in --
    Streamlit forbids mutating, inside a cached function, a widget/element
    created outside of it, since that mutation wouldn't replay on a cache
    *hit*) so the user can tell which stage a (potentially many-second) run
    is actually in, rather than staring at a single generic spinner. Cleared
    again once the solve succeeds; left on screen if it raises, as a hint at
    which stage failed.

    Parameters
    ----------
    params_items : tuple of (str, value) pairs
        A hashable, order-independent-by-construction encoding of a
        ``params`` dict (see ``app.main``, which sorts the items before
        calling this function), required because ``st.cache_resource`` keys
        its cache on the function arguments.  May include a ``"custom_network"``
        entry (JSON text built by the "Create custom network"/"Import custom
        network" popups, ``params_form._custom_network_dialog``/
        ``_import_dialog``); it is popped out and decoded before building
        ``PyPR`` (it is not a ``PyPRConfig``/``DEFAULT_PARAMS`` field) but
        stays part of the tuple so a different customisation still produces a
        different cache key.

    Returns
    -------
    (pyprimat.PyPR, str, str, float)
        The solved instance -- ``run.PyPRresults()``, ``run.abundance_names``,
        and ``run[name](t)`` are all ready to use without triggering further
        computation -- together with the contents of the nuclear time-evolution
        TSV (``output_time_evolution`` format, see
        ``nuclear_network.py:NuclearNetwork._write_time_evolution``) and of the
        background time-evolution TSV (``output_background_evolution`` format,
        see ``background.py:Background.write_time_evolution``), both as strings,
        and the wall-clock time (seconds) the actual solve took. The elapsed
        time is measured *inside* this cached function (rather than by the
        caller timing the call) specifically so that a cache *hit* -- a rerun
        with unchanged parameters, which never re-executes this function body
        at all -- still reports the original solve's real duration instead of
        the ~0 s a caller-side timer would measure around an instant cache
        lookup.
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
    t0 = time.time()
    params = dict(params_items)
    custom_network_json = params.pop("custom_network", None)
    custom_network = json.loads(custom_network_json) if custom_network_json else None

    # For the large network, always integrate the "decay-time" (DT) era so the
    # evolution panel can optionally show abundances past the end of BBN, out to
    # the age of the Universe (its "Show radioactive decays" toggle).  The DT
    # integration is a cheap constant-matrix exponentiation (~0.1 s), and the
    # solve()-level guard ignores decay_era for the small network, so we
    # can set it unconditionally for large only.  t_decay_end is set to ~13.8 Gyr
    # (the age of the Universe) unless the user already overrode it.
    AGE_UNIVERSE_S = 13.8e9 * 365.25 * 86400.0   # ≈ 4.35×10^17 s
    decay_extras = {}
    if params.get("network", "small") == "large":
        decay_extras["decay_era"] = True
        if "t_decay_end" not in params:    # respect an explicit user override
            decay_extras["t_decay_end"] = AGE_UNIVERSE_S
        # Extend the integration to lower temperature so heavy-nuclide tails
        # are fully resolved (default T_end=0.01 MeV cuts off too early for
        # some large-network isotopes).
        if "T_end_MeV" not in params:
            decay_extras["T_end_MeV"] = 1e-4

    fd_evo, tmp_evo = tempfile.mkstemp(suffix=".tsv", prefix="pyprimat_evolution_")
    os.close(fd_evo)
    fd_bg, tmp_bg = tempfile.mkstemp(suffix=".tsv", prefix="pyprimat_background_")
    os.close(fd_bg)
    status = st.empty()
    try:
        # PyPR.__init__ itself (before solve()) is where the n<->p weak
        # rates get (re)computed -- see CLAUDE.md's "Execution flow" step 1
        # -- so that's the message to show here, not "Solving...". When
        # thermal_corrections is on AND its fingerprinted cache file isn't
        # already on disk, that step also runs the CCRTh finite-temperature
        # correction's vegas Monte Carlo integration -- by far the slowest
        # part of the whole weak-rate computation -- so it's worth calling
        # out explicitly. A cheap throwaway PyPRConfig (no rate loading) is
        # enough to check thermal_cache_exists; ``PyPR`` below builds its
        # own cfg from the same params regardless. Checking the cache
        # rather than just ``cfg.thermal_corrections`` matters because that
        # flag alone says nothing about whether a slow recompute is
        # actually about to happen -- most of the time it's already cached
        # and this step is fast.
        note = "### Computing weak rates…"
        preview_cfg = PyPRConfig(params)
        if preview_cfg.thermal_corrections and not thermal_cache_exists(preview_cfg):
            note += ("  \n*(includes finite-temperature thermal "
                     "corrections — this can take a while)*")
        status.markdown(note)
        run = PyPR(params=dict(params,
                               output_time_evolution=True,
                               output_file=tmp_evo,
                               output_background_evolution=True,
                               output_background_file=tmp_bg,
                               **decay_extras),
                   custom_network=custom_network)
        status.markdown("### Solving the BBN network…")
        run.solve()
        with open(tmp_evo) as f:
            time_evolution_tsv = f.read()
        with open(tmp_bg) as f:
            background_tsv = f.read()
    finally:
        os.remove(tmp_evo)
        os.remove(tmp_bg)
    status.empty()
    return run, time_evolution_tsv, background_tsv, time.time() - t0


def _quick_mc(params_items, num_mc, run):
    """Run a quick :func:`pyprimat.main.mc_uncertainty` for the standard ratios.

    Parameters
    ----------
    params_items : tuple of (str, value) pairs
        Same hashable encoding of ``params`` as :func:`_solve`.
    num_mc : int
        Number of Monte Carlo samples requested (the GUI caps this at 100).
    run : pyprimat.main.PyPR
        The already-solved reference run (see ``app.main``), used only to
        determine which ``_RATIO_FORMAT`` keys are actually present in
        ``run.results`` -- e.g. ``Li6oLi7`` requires a network producing Li6
        (large) and ``YCNO`` requires CNO species (large), so for the
        default small network neither key exists and requesting them from
        ``get_quantity`` would raise ``ValueError`` (see ``main.py``'s
        conditional ``results["Li6oLi7"] = ...`` / ``results["YCNO"] = ...``).

    Returns
    -------
    pyprimat.main.MCResult
        Indexed by whichever ``_RATIO_FORMAT`` keys are valid for this run's
        network (Neff, YPBBN, YPCMB, DoH, He3oH, He3oHe4, Li7oH, and
        Li6oLi7/YCNO when applicable); each entry has ``.mean`` and ``.std``.

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

    **Customised networks.** ``params_items`` may include a JSON-encoded
    "custom_network" entry (the "Customise Reactions" override, see
    :class:`pyprimat.main.PyPR`'s docstring).  It is decoded here and passed
    through to :func:`mc_uncertainty` as its own ``custom_network`` kwarg
    (and stripped from ``params`` so ``PyPRConfig`` never sees an unknown
    key): removed reactions are excluded from the varied rate set and
    replaced reactions are sampled using their own (possibly custom) error
    column, so quick MC's ± 1σ band reflects the same customisation as the
    central-value run, including any inflated/deflated rate uncertainty
    uploaded for a replaced reaction.
    """
    cache = st.session_state.get("_quick_mc_cache")
    # Reuse the cached MCResult as a starting point only when it was computed
    # for exactly these parameters; mc_uncertainty itself re-checks seed,
    # quantities, params and custom_network before trusting ``prev``.
    prev = cache[1] if (cache is not None and cache[0] == params_items) else None
    cn_json = dict(params_items).get("custom_network")
    custom_network = json.loads(cn_json) if cn_json else None
    mc_params = {k: v for k, v in params_items if k != "custom_network"}
    quantities = [q for q in _RATIO_FORMAT if q in run.results]
    mc = mc_uncertainty(num_mc, quantities,
                        params=mc_params, seed=0, prev=prev,
                        custom_network=custom_network)
    st.session_state["_quick_mc_cache"] = (params_items, mc)
    return mc


@st.cache_resource(show_spinner=False)
def _build_preview(params_items):
    """Build just enough of a ``PyPR`` to back the Reactions summary tab.

    That tab (:func:`pyprimat.gui.panels.render_reactions_panel` and
    :func:`pyprimat.gui.panels._render_reaction_downloads`) only ever reads
    ``run.cfg`` and ``run.nucl`` -- the compiled MT/LT networks and their
    rate tables (see ``CLAUDE.md``'s "Execution flow", step 3). It never
    touches ``run.background``/``run.nuclear``, so building a full ``PyPR``
    here would needlessly also run step 4: constructing ``StandardBackground``,
    which computes the n<->p weak rates. Those depend on most of the
    "Physics" sidebar group (``radiative_corrections``, ``thermal_corrections``,
    ``spectral_distortions``, ``analytic_distortions``, ...), so changing any
    of them busts this function's ``st.cache_resource`` key and forces a full
    weak-rate (re)computation, sometimes several seconds -- on *every* such
    sidebar edit, even though this tab reflects only ``run.nucl`` and has
    nothing to do with weak rates at all. Skipping straight to
    ``PyPRConfig`` + ``UpdateNuclearRates`` (mirroring ``PyPR.__init__``'s
    own steps 1+3, see ``main.py``) avoids that cost entirely; a
    ``types.SimpleNamespace`` stands in for ``PyPR`` since only the two
    attributes above are ever read off it.

    This tab must always reflect whatever the sidebar currently shows --
    even before "Run BBN" is first clicked, and even after the sidebar has
    been changed since the last completed run -- so it cannot simply reuse
    :func:`_solve`'s cached result. Cached the same way as :func:`_solve` so
    repeatedly viewing the same configuration is instant.

    ``params_items`` is the same sorted-items encoding as :func:`_solve`'s
    (``params`` straight from ``render_sidebar_form``, which already embeds
    a JSON ``"custom_network"`` entry when a custom network is active --
    see its "network" branch).
    """
    params = dict(params_items)
    custom_network_json = params.pop("custom_network", None)
    custom_network = json.loads(custom_network_json) if custom_network_json else None
    cfg = PyPRConfig(params)
    nucl = UpdateNuclearRates(cfg, custom_network=custom_network)
    return types.SimpleNamespace(cfg=cfg, nucl=nucl)


def main():
    st.title("⚛️ PyPRIMAT")
    st.caption(
        "Big Bang Nucleosynthesis abundances — interactive front end for "
        "`pyprimat.PyPR`"
    )
    st.markdown(
        "PyPRIMAT computes primordial light-element abundances (D, He3, He4, "
        "Li7, ...) after Big Bang Nucleosynthesis."
    )

    params, quick_mc, mc_samples = render_sidebar_form()
    _render_footer()

    # `params` already carries a JSON "custom_network" entry when one is
    # active (set by render_sidebar_form's "network" branch), so comparing
    # its sorted items against the last-run snapshot (`st.session_state
    # ["params"]`, set below exactly the same way) tells whether the
    # sidebar's *exact* current configuration has already been solved --
    # changing anything (including just re-picking the same custom network)
    # makes this False again until "Run BBN" is clicked anew.
    current_items = tuple(sorted(params.items()))
    stored_params = st.session_state.get(SessionKeys.params)
    params_up_to_date = (stored_params is not None
                 and tuple(sorted(stored_params.items())) == current_items)
    # The stored quick-MC settings only update on a "Run BBN" click (see
    # below), same as `stored_params` -- so if the sidebar's quick-MC
    # toggle/sample-count has moved away from what was last actually run,
    # the button must stay clickable even though `params` itself hasn't
    # changed, or there would be no way to ever trigger the new MC request.
    mc_up_to_date = (not quick_mc) or (
        st.session_state.get(SessionKeys.quick_mc, False)
        and mc_samples == st.session_state.get(SessionKeys.mc_samples)
    )
    up_to_date = params_up_to_date and mc_up_to_date

    # Above the result tabs (not in the sidebar) so it stays visible even
    # when the sidebar is folded; left at its natural (content-sized) width
    # rather than stretched across the full column. "Shaded" (secondary,
    # disabled) once the current configuration is already solved -- nothing
    # left to run until something changes (including the quick-MC request).
    run_clicked = st.button("Run BBN", type=("secondary" if up_to_date else "primary"),
                            disabled=up_to_date)

    if run_clicked:
        # Snapshot the current form state; subsequent reruns (e.g. from
        # ticking a nuclide checkbox) reuse this snapshot via
        # st.session_state rather than re-triggering a solve with whatever
        # the sidebar currently shows.
        st.session_state[SessionKeys.params] = dict(params)
        st.session_state[SessionKeys.quick_mc] = quick_mc
        st.session_state[SessionKeys.mc_samples] = mc_samples
        # Snapshot the active custom network's dict (if any -- set by the
        # "Create custom network"/"Import custom network" popups, see
        # params_form._render_dialog_footer/_import_dialog), so the Reactions
        # tab's export button reflects what was actually run rather than
        # whatever the sidebar shows now.
        active = st.session_state.get(SessionKeys.active_custom_network)
        st.session_state[SessionKeys.run_custom_network_dict] = (
            active["custom_network"] if active else None)
        stored_params = st.session_state[SessionKeys.params]
        params_up_to_date = True

    # The active tab is forced via a "_tabs_gen"-keyed remount (st.tabs's
    # `default=` is otherwise only honoured the very first time a given key
    # is used, same quirk as every other key-tracked widget -- see
    # params_form's session_keys.SessionKeys docstring for the general pattern):
    # whenever "up to date" flips, bump the generation so the new `default`
    # actually takes effect instead of being ignored in favour of whichever
    # tab the user last had open. Gated on `params_up_to_date` alone (not the
    # MC-aware `up_to_date`) -- a stale quick-MC request must not hide the
    # already-solved BBN results tabs.
    if st.session_state.get("_tabs_up_to_date") != params_up_to_date:
        st.session_state["_tabs_up_to_date"] = params_up_to_date
        st.session_state["_tabs_gen"] = st.session_state.get("_tabs_gen", 0) + 1
    default_tab = "Final abundances" if params_up_to_date else "Reactions summary"
    tabs_gen = st.session_state.get("_tabs_gen", 0)

    tab_reactions, tab_results, tab_evolution, tab_downloads = st.tabs(
        ["Reactions summary", "Final abundances", "Abundance evolution",
         "Output tables"],
        default=default_tab, key=f"_main_tabs_{tabs_gen}",
    )

    with tab_reactions:
        # Always built from the *current* sidebar state (not the last
        # completed run), so this tab reflects in-progress edits immediately.
        # Loading every rate table for the large network is not instant, so
        # this gets its own spinner too -- otherwise the tab would sit blank
        # for a moment with no indication anything is happening.
        try:
            with st.spinner("Loading network…"):
                preview = _build_preview(current_items)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Cannot build this network: {exc}")
        else:
            panels.render_reactions_panel(preview)

    if not params_up_to_date:
        not_run_msg = (
            "This tab will appear once **Run BBN** has solved the current "
            "configuration."
        )
        with tab_results:
            st.info(not_run_msg)
        with tab_evolution:
            st.info(not_run_msg)
        with tab_downloads:
            st.info(not_run_msg)
        if st.session_state.get(SessionKeys.params) is None:
            st.info("Set parameters in the sidebar, then click **Run BBN**.")
        return

    stored_params = st.session_state[SessionKeys.params]
    # st.cache_resource requires hashable arguments; a sorted tuple of items
    # is both hashable and order-independent (so key ordering in the params
    # dict never causes a spurious cache miss).
    params_items = tuple(sorted(stored_params.items()))

    # _solve advances its own large-font status placeholder through
    # "Computing weak rates…"/"Solving the BBN network…" as it actually
    # reaches each stage (see its docstring) -- it must create that
    # placeholder itself rather than have it passed in from here: Streamlit
    # forbids mutating, inside a `@st.cache_resource`-decorated function, a
    # widget/element that was created outside of it (the mutation wouldn't
    # replay on a cache hit, leaving the two inconsistent). A plain
    # st.spinner(...) renders its own message at normal body-text size,
    # hence the separate, larger placeholder for the stage text.
    try:
        with st.spinner(""):
            # `elapsed` comes back *from* _solve (timed inside the cached
            # function itself) rather than being measured here around the
            # call -- see _solve's docstring: on the very next rerun after a
            # real solve (the `if run_clicked: st.rerun()` below, needed so
            # the button/tab visuals catch up), this call is an instant cache
            # *hit*, so timing it here would report ~0 s instead of how long
            # the actual solve took.
            run, time_evolution_tsv, background_tsv, elapsed = _solve(params_items)
    except (ValueError, RuntimeError) as exc:
        # PyPRConfig validates e.g. `amax`/`network` and the
        # spectral_distortions/incomplete_decoupling/analytic_distortions
        # flag combination, raising ValueError on bad input; nuclear_network
        # raises RuntimeError for internal-state misuse. Surface those as a
        # clean message rather than a traceback. Other exception types are
        # genuine bugs and should propagate so they show up loudly. Whichever
        # stage message _solve last showed is left on screen above this
        # error -- a useful hint at which stage actually failed.
        st.error(f"PyPRIMAT run failed: {exc}")
        return

    if run_clicked:
        # The "Run BBN" button's own disabled/shaded state (and the active
        # tab's `default=`) were rendered earlier in *this exact* script run,
        # using the pre-click `up_to_date` -- Streamlit's top-down model means
        # they can't retroactively reflect the solve that just succeeded
        # below them. One extra rerun (cheap: _solve is cache_resource'd, so
        # it's an instant cache hit) is the only way to have the button
        # actually look shaded and the tab actually switch to "Final
        # abundances" immediately after this same click, rather than only on
        # the next unrelated interaction.
        st.rerun()

    mc = None
    if st.session_state.get(SessionKeys.quick_mc, False):
        num_mc = st.session_state.get(SessionKeys.mc_samples, 30)
        with st.spinner(f"Running {num_mc}-sample quick MC uncertainty…"):
            mc = _quick_mc(params_items, num_mc, run)

    with tab_results:
        st.caption(f"(solved in {elapsed:.2f} s)")
        panels.render_results_panel(run, mc=mc)
    with tab_evolution:
        panels.render_evolution_panel(run)
    with tab_downloads:
        panels.render_downloads_panel(run, time_evolution_tsv, background_tsv)


def _render_footer():
    """Sidebar attribution footer, shown below the parameter form."""
    st.sidebar.caption(
        "PyPRIMAT is developed by [Cyril Pitrou](https://www2.iap.fr/users/pitrou/) "
        "and Julien Froustey. This GUI is developed by Cyril Pitrou."
    )
    st.sidebar.caption(
        "Download the [source code](https://github.com/CyrilPitrou/pyprimat) "
        "and cite the [publication](https://arxiv.org/abs/1801.08023) if you use it."
    )


if __name__ == "__main__":
    main()
