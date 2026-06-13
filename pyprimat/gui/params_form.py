# -*- coding: utf-8 -*-
"""
pyprimat.gui.params_form
=========================

Builds the sidebar parameter form (GUI.md §2) from
``pyprimat.config.DEFAULT_PARAMS``, the single authoritative dict of every
``PyPRConfig`` key.

Design
------
``DEFAULT_PARAMS`` has ~50 keys, most of which are caching/debug knobs that a
typical user never touches (e.g. ``save_nTOp``, ``recompute_qed_corrections``,
``numba_installed``).  We therefore split the form in two:

* A **curated set** of "headline" flags (``_FORM_METADATA`` below), grouped
  under ``GROUP_ORDER`` and shown as expanded/visible sidebar sections, each
  with a short physics-oriented label and tooltip condensed from the comments
  in ``pyprimat/config.py``.
* A **"Constants" expander** (``_CONSTANTS_METADATA`` below) exposing only
  ``GN`` and ``tau_n`` -- the two fundamental constants users occasionally
  vary for sensitivity studies. Every other ``DEFAULT_PARAMS`` key (caching,
  precision, output, debug knobs) is left at its default and not shown.

In both cases the widget type is derived from the *type of the default
value* (bool -> toggle, int/float -> number_input, str -> selectbox/text
input), and -- mirroring the ``pyprimat`` CLI's "forward only what changed"
convention (``pyprimat/cli.py``) -- :func:`render_sidebar_form` returns a
dict containing only the entries whose value differs from
``DEFAULT_PARAMS``, so unset flags keep relying on ``PyPRConfig``'s own
defaults.
"""
import importlib.resources

import streamlit as st

from pyprimat.config import DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# Curated metadata for the "headline" parameters shown by default.
#
# Each entry maps a DEFAULT_PARAMS key to (group, label, help_text).  Group
# order/visibility is controlled by GROUP_ORDER below; within a group, keys
# are rendered in the (insertion) order they appear here -- this matters for
# `network` (must be set before `amax` is shown/hidden) and for
# `spectral_distortions` (must be set before its sub-options are shown/hidden).
# ---------------------------------------------------------------------------
_FORM_METADATA = {
    # ---- Cosmology ---------------------------------------------------------
    "Omegabh2": (
        "Cosmology", r"$\Omega_b h^2$  (baryon density)",
        "Baryon density parameter; sets the baryon-to-photon ratio ηᵇ "
        "used throughout the network.",
    ),
    "DeltaNeff": (
        "Cosmology", r"$\Delta N_{\text{eff}}$",
        "Extra effective relativistic degrees of freedom on top of the "
        "Standard-Model neutrino sector.",
    ),
    "munuOverTnu": (
        "Cosmology", r"$\xi_\nu = \mu_\nu/T_\nu$",
        "Reduced neutrino chemical potential (same for all three flavours). "
        "Non-zero values are physically consistent only with "
        "incomplete_decoupling=False, since the NEVO table assumes it vanishes.",
    ),

    # ---- Network ------------------------------------------------------------
    "network": (
        "Network", "Reaction network",
        "Nuclear reaction network used in the low-temperature (LT) era: "
        "'small' (12 nuclides), 'medium' (62 reactions), 'large' (~433 "
        "reactions, ~59 nuclides), or 'small_parthenope' for comparison runs. "
        "The HT/MT eras are unaffected (always n<->p / fixed 18-reaction set).",
    ),
    "amax": (
        "Network", "Max mass number A (large only)",
        "With network='large', drop reactions involving any nuclide with mass "
        "number A > amax (must be an integer > 7). Leave unchecked to keep all "
        "~59 nuclides.",
    ),

    # ---- Physics toggles -------------------------------------------------------
    "incomplete_decoupling": (
        "Physics", "Incomplete neutrino decoupling",
        "True (default): non-instantaneous decoupling using the precomputed "
        "NEVO table (ν flavour temperatures differ slightly due to "
        "partial reheating by e+e- annihilation). False: instantaneous "
        "decoupling, Tν/Tγ = (4/11)^(1/3).",
    ),
    "QED_corrections": (
        "Physics", "QED plasma corrections",
        "Include QED interaction corrections to the electromagnetic plasma "
        "equation of state (electron/positron pressure and density).",
    ),
    "nuclear_qed_corrections": (
        "Physics", "Nuclear QED rate corrections",
        "True (default): apply a T9-dependent QED rescaling (Pitrou & "
        "Pospelov 2020) to the forward rates of npTOdg, dpTOHe3g, tpTOag, "
        "taTOLi7g, He3aTOBe7g.",
    ),
    "nTOp_Born_approximation": (
        "Physics", "Born approximation for n↔p",
        "Use the crude Born-approximation n<->p rate (off by a few percent); "
        "for debugging or comparison with simpler codes only.",
    ),
    "spectral_distortions": (
        "Physics", "Spectral distortions",
        "Corrections to n<->p weak rates from deviations of the neutrino "
        "phase-space distribution away from a perfect Fermi-Dirac shape.",
    ),
    "analytic_distortions": (
        "Physics", "→ analytic distortion model",
        "Parameterise the distortion analytically as μ-type "
        "(delta_xi_nu) and/or y-type (y_SZ) instead of reading the full "
        "NEVO spectrum file. Requires incomplete_decoupling=False.",
    ),
    "delta_xi_nu": (
        "Physics", "→ δξν (μ-type distortion)",
        "Shift of the reduced neutrino chemical potential for the μ-type "
        "spectral distortion (applied to all three flavours).",
    ),
    "y_SZ": (
        "Physics", "→ y_SZ (y-type distortion)",
        "Amplitude of the y-type (Sunyaev-Zel'dovich-like) spectral "
        "distortion.",
    ),
}

# Order (and default expanded/collapsed state) of the curated sidebar groups.
GROUP_ORDER = ["Cosmology", "Network", "Physics"]
_EXPANDED_GROUPS = {"Cosmology", "Network"}

# ---------------------------------------------------------------------------
# "Constants" section: the only DEFAULT_PARAMS keys outside the curated
# groups that users commonly want to override (e.g. for sensitivity studies).
# ---------------------------------------------------------------------------
_CONSTANTS_METADATA = {
    "GN": (
        r"$G_N$  (Newton's constant) [MeV⁻²]",
        "Gravitational constant entering the Friedmann equation.",
    ),
    "tau_n": (
        r"$\tau_n$  (neutron lifetime) [s]",
        "Neutron lifetime, used to normalise the n<->p weak rates "
        "(when tau_n_flag=True, the default).",
    ),
}

# Keys whose widget is only shown conditionally on another key's value.
# Maps key -> (controlling_key, required_value).
_CONDITIONAL = {
    "amax": ("network", "large"),
    "analytic_distortions": ("spectral_distortions", True),
    "delta_xi_nu": ("spectral_distortions", True),
    "y_SZ": ("spectral_distortions", True),
}


def _available_networks():
    """Return the selectable values for the ``network`` parameter.

    'small' is PyPRIMAT's built-in default network and needs no file; the
    other choices are discovered from ``pyprimat/rates/nuclear/networks/*.txt``
    (see ``PyPRConfig.__init__``, which validates ``network`` against exactly
    these files for any value other than 'small').
    """
    names = []
    try:
        net_dir = importlib.resources.files("pyprimat") / "rates" / "nuclear" / "networks"
        names = sorted(p.stem for p in net_dir.iterdir() if p.suffix == ".txt")
    except (FileNotFoundError, ModuleNotFoundError, NotADirectoryError):
        pass
    if "small" not in names:
        names.insert(0, "small")
    return names


def _widget_for(key, label, help_text):
    """Render a single widget for ``key``, typed from its default value.

    Returns the widget's current value (which equals ``DEFAULT_PARAMS[key]``
    unless the user has changed it -- Streamlit persists the value across
    reruns via ``key=key`` in ``st.session_state``).
    """
    default = DEFAULT_PARAMS[key]

    if key == "network":
        options = _available_networks()
        index = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=index, help=help_text, key=key)

    if isinstance(default, bool):
        return st.toggle(label, value=default, help=help_text, key=key)

    if isinstance(default, int):
        return int(st.number_input(label, value=default, step=1, help=help_text, key=key))

    if isinstance(default, float):
        # "%.6g" keeps both O(1) values (Omegabh2) and very small/large ones
        # (GN=6.7e-45) readable.
        return st.number_input(
            label, value=default, format="%.6g", help=help_text, key=key,
        )

    # Fallback for string-valued parameters (e.g. output_file paths).
    return st.text_input(label, value=str(default), help=help_text, key=key)


def render_sidebar_form():
    """Render the full parameter form in the Streamlit sidebar.

    Returns
    -------
    params : dict
        Subset of ``DEFAULT_PARAMS`` keys whose value the user changed from
        the default, suitable for ``PyPR(params=this_dict)``. Keys left at
        their default are omitted entirely so ``PyPRConfig`` continues to be
        the single source of truth for defaults (mirrors ``pyprimat.cli``'s
        "forward only what changed" behaviour).
    quick_mc : bool
        Whether the "Quick MC uncertainty" toggle is enabled. This is a
        GUI-only flag (not a ``PyPRConfig``/``DEFAULT_PARAMS`` key), so it is
        returned separately rather than folded into ``params``.
    mc_samples : int
        Number of Monte Carlo samples to draw for the quick uncertainty
        estimate (only meaningful when ``quick_mc`` is True).  Capped at 100 in
        the widget because each sample is a full nuclear-network solve and more
        than that is too slow for an interactive "quick" estimate.  Also a
        GUI-only value, returned separately.
    """
    params = {}

    st.sidebar.header("Parameters")

    # ---- Curated groups -----------------------------------------------------
    by_group = {g: [] for g in GROUP_ORDER}
    for key, (group, label, help_text) in _FORM_METADATA.items():
        by_group[group].append((key, label, help_text))

    for group in GROUP_ORDER:
        with st.sidebar.expander(group, expanded=(group in _EXPANDED_GROUPS)):
            for key, label, help_text in by_group[group]:
                if key in _CONDITIONAL:
                    ctrl_key, required = _CONDITIONAL[key]
                    current = params.get(ctrl_key, DEFAULT_PARAMS[ctrl_key])
                    if current != required:
                        continue
                    if key == "amax":
                        # `amax` defaults to None, so it needs an explicit
                        # enable/disable checkbox rather than a bare number
                        # input (which could never represent "no filter").
                        enabled = st.checkbox(
                            "Limit max mass number", value=False,
                            help=help_text, key="amax_enabled",
                        )
                        if enabled:
                            params["amax"] = int(st.number_input(
                                label, min_value=8, value=20, step=1,
                                help=help_text, key="amax_value",
                            ))
                        continue

                value = _widget_for(key, label, help_text)
                if value != DEFAULT_PARAMS[key]:
                    params[key] = value

    # ---- Constants: GN and tau_n only ----------------------------------------
    with st.sidebar.expander("Constants", expanded=False):
        for key, (label, help_text) in _CONSTANTS_METADATA.items():
            value = _widget_for(key, label, help_text)
            if value != DEFAULT_PARAMS[key]:
                params[key] = value

    # ---- Uncertainty: optional quick MC error bars ---------------------------
    with st.sidebar.expander("Uncertainty", expanded=False):
        quick_mc = st.toggle(
            "Quick MC uncertainty",
            value=False,
            help="After the main run, also run a small Monte Carlo "
                 "(varying every nuclear-rate p_* and the neutron lifetime "
                 "tau_n) and show mean +/- 1 sigma next to each standard "
                 "ratio below. With only a few dozen samples this is a "
                 "*quick, noisy* estimate -- enough to gauge the order of "
                 "magnitude of the uncertainty, not a publication-quality "
                 "error bar.",
            key="quick_mc_uncertainty",
        )
        # Number of MC samples. Capped at 100: each sample is a full network
        # solve, so more than that stops being "quick". Because the samples are
        # seed-deterministic (sample i uses seed+i), raising this value only
        # solves the *additional* samples -- the GUI reuses any already-computed
        # ones via the ``prev`` argument of mc_uncertainty (see app._quick_mc).
        mc_samples = int(st.number_input(
            "MC samples",
            min_value=2, max_value=100, value=30, step=10,
            help="How many Monte Carlo samples to draw (max 100). Increasing "
                 "this reuses the samples already computed and only solves the "
                 "extra ones, so refining from e.g. 30 to 50 runs just 20 new "
                 "solves.",
            key="quick_mc_samples",
            disabled=not quick_mc,
        ))

    return params, quick_mc, mc_samples
