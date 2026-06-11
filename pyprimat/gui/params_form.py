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

* A **curated set** of ~20 "headline" flags (``_FORM_METADATA`` below),
  grouped under ``GROUP_ORDER`` and shown as expanded/visible sidebar
  sections, each with a short physics-oriented label and tooltip condensed
  from the comments in ``pyprimat/config.py``.
* Every *other* key in ``DEFAULT_PARAMS`` is rendered automatically inside an
  "Advanced" expander, so the form always covers the full configuration
  surface even as new flags are added to ``DEFAULT_PARAMS``.

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
        "Cosmology", "Ωᵇ h²  (baryon density)",
        "Baryon density parameter; sets the baryon-to-photon ratio ηᵇ "
        "used throughout the network.",
    ),
    "DeltaNeff": (
        "Cosmology", "ΔN_eff",
        "Extra effective relativistic degrees of freedom on top of the "
        "Standard-Model neutrino sector.",
    ),
    "munuOverTnu": (
        "Cosmology", "ξν = μν / Tν",
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

    # ---- Precision -----------------------------------------------------------
    "numerical_precision": (
        "Precision", "Numerical precision (rtol)",
        "Relative tolerance passed to solve_ivp for the background and all "
        "three network eras (HT/MT/LT). Smaller = slower but more accurate.",
    ),
    "T_start_cosmo_MeV": (
        "Precision", "T_start [MeV]",
        "Starting photon temperature of the cosmological background "
        "integration.",
    ),
    "n_temperature_table": (
        "Precision", "Background table points",
        "Number of grid points in the precomputed background temperature "
        "table (a(T), Hubble rate, etc.).",
    ),
    "sampling_nTOp": (
        "Precision", "n↔p rate grid points",
        "Number of points in the n<->p weak-rate table.",
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
        "Apply a T9-dependent QED rescaling (Pitrou & Pospelov 2020) to the "
        "forward rates of npTOdg, dpTOHe3g, tpTOag, taTOLi7g, He3aTOBe7g.",
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

    # ---- Output ---------------------------------------------------------------
    "output_time_evolution": (
        "Output", "Write time-evolution TSV",
        "Write the full background + abundance time series to "
        "output_file (small/medium networks only).",
    ),
    "output_final_result": (
        "Output", "Write final-abundance file",
        "Write a two-column (nuclide, Y) table of final abundances to "
        "output_final_file (this duplicates the download button below).",
    ),
}

# Order (and default expanded/collapsed state) of the curated sidebar groups.
GROUP_ORDER = ["Cosmology", "Network", "Precision", "Physics", "Output"]
_EXPANDED_GROUPS = {"Cosmology", "Network"}

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
        # (numerical_precision=1e-7, GN=6.7e-45) readable.
        return st.number_input(
            label, value=default, format="%.6g", help=help_text, key=key,
        )

    # Fallback for string-valued parameters (e.g. output_file paths).
    return st.text_input(label, value=str(default), help=help_text, key=key)


def _humanize(key):
    """Turn a DEFAULT_PARAMS key like 'recompute_electron_thermo' into a label."""
    return key.replace("_", " ")


def render_sidebar_form():
    """Render the full parameter form in the Streamlit sidebar.

    Returns
    -------
    dict
        Subset of ``DEFAULT_PARAMS`` keys whose value the user changed from
        the default, suitable for ``PyPR(params=this_dict)``. Keys left at
        their default are omitted entirely so ``PyPRConfig`` continues to be
        the single source of truth for defaults (mirrors ``pyprimat.cli``'s
        "forward only what changed" behaviour).
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

    # ---- Advanced: every remaining DEFAULT_PARAMS key ------------------------
    curated_keys = set(_FORM_METADATA)
    remaining = [k for k in DEFAULT_PARAMS if k not in curated_keys]
    with st.sidebar.expander("Advanced", expanded=False):
        st.caption(
            "Caching, debug, and rate-table knobs. Defaults match "
            "`pyprimat.config.DEFAULT_PARAMS`."
        )
        for key in remaining:
            value = _widget_for(key, _humanize(key), None)
            if value != DEFAULT_PARAMS[key]:
                params[key] = value

    return params
