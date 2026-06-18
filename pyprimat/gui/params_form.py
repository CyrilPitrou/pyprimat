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
import io
import json
import re

import streamlit as st

from pyprimat.config import DEFAULT_PARAMS, PyPRConfig
from pyprimat.network_data import load_network, load_reaction_names
from pyprimat.gui import custom_rates
from pyprimat.gui.panels import _equation_unicode

# Above this reaction count, rendering one checkbox + uploader row per reaction
# is slow enough in Streamlit to warrant a heads-up caption (it still works).
# 'large' (~428 reactions) trips this; small/medium stay well under it.
_CUSTOMISABLE_REACTIONS_WARN = 120


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

    # ---- Nuclear reactions ---------------------------------------------------
    "network": (
        "Nuclear reactions", "Reaction network",
        "Nuclear reaction network used in the low-temperature (LT) era: "
        "'small' (12 nuclides), 'medium' (62 reactions), 'large' (~433 "
        "reactions, ~59 nuclides), or 'small_parthenope' for comparison runs. "
        "The HT/MT eras are unaffected (always n<->p / fixed 18-reaction set).",
    ),
    "amax": (
        "Nuclear reactions", "Max mass number A (large only)",
        "With network='large', drop reactions involving any nuclide with mass "
        "number A > amax (must be an integer >= 7). Leave unchecked to keep all "
        "~59 nuclides.",
    ),
    "nuclear_qed_corrections": (
        "Nuclear reactions", "Nuclear QED rate corrections",
        "True (default): apply a T9-dependent QED rescaling (Pitrou & "
        "Pospelov 2020) to the forward rates of n_p__d_g, d_p__He3_g, t_p__a_g, "
        "t_a__Li7_g, He3_a__Be7_g.",
    ),

    # ---- Plasma physics ------------------------------------------------------
    "QED_corrections": (
        "Plasma physics", "QED plasma corrections",
        "Include QED interaction corrections to the electromagnetic plasma "
        "equation of state (electron/positron pressure and density).",
    ),

    # ---- Weak rates ----------------------------------------------------------
    "incomplete_decoupling": (
        "Weak rates", "Incomplete neutrino decoupling",
        "True (default): non-instantaneous decoupling using the precomputed "
        "NEVO table (ν flavour temperatures differ slightly due to "
        "partial reheating by e+e- annihilation). False: instantaneous "
        "decoupling, Tν/Tγ = (4/11)^(1/3).",
    ),
    "radiative_corrections": (
        "Weak rates", "Radiative corrections (n↔p)",
        "Include T=0 Coulomb + resummed radiative corrections (CCR, Phys. Rep. "
        "Eq. 101; Czarnecki et al. 2004).  When False the crude Born approximation "
        "is used instead.",
    ),
    "finite_mass_corrections": (
        "Weak rates", "Finite-mass corrections (n↔p)",
        "Include the Fokker-Planck finite-nucleon-mass correction to the n↔p rate "
        "(Phys. Rep. §III.G).  Uses FMCCR when radiative_corrections=True, "
        "FMNoCCR otherwise.",
    ),
    "thermal_corrections": (
        "Weak rates", "Thermal radiative corrections (n↔p)",
        "Include finite-temperature radiative corrections to the n↔p rate "
        "(CCRTh; Brown & Sawyer 2001, Phys. Rep. §III.H).",
    ),
    "spectral_distortions": (
        "Weak rates", "Spectral distortions",
        "Corrections to n<->p weak rates from deviations of the neutrino "
        "phase-space distribution away from a perfect Fermi-Dirac shape.",
    ),
    "analytic_distortions": (
        "Weak rates", "→ analytic distortion model",
        "Parameterise the distortion analytically as μ-type "
        "(delta_xi_nu) and/or y-type (y_SZ) instead of reading the full "
        "NEVO spectrum file. Requires incomplete_decoupling=False.",
    ),
    "delta_xi_nu": (
        "Weak rates", "→ δξν (μ-type distortion)",
        "Shift of the reduced neutrino chemical potential for the μ-type "
        "spectral distortion (applied to all three flavours).",
    ),
    "y_SZ": (
        "Weak rates", "→ y_SZ (y-type distortion)",
        "Amplitude of the y-type (Sunyaev-Zel'dovich-like) spectral "
        "distortion.",
    ),
}

# Order (and default expanded/collapsed state) of the curated sidebar groups.
GROUP_ORDER = ["Cosmology", "Nuclear reactions", "Plasma physics", "Weak rates"]
_EXPANDED_GROUPS = {"Cosmology", "Nuclear reactions"}

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
        "(when tau_n_normalization=True, the default).",
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


@st.cache_data(show_spinner=False)
def _reaction_equations(network):
    """Return ``{bare_reaction_name: "a + b <-> c + d"}`` for ``network``.

    Loads the full LT ``NetworkDefinition`` (rate tables and all) just to read
    off :meth:`NetworkDefinition.reaction_equation` -- the same source used by
    the Reactions tab (``pyprimat.gui.panels``) after a run.  Cached per
    ``network`` name (``st.cache_data``) so this cost (reading every rate
    table on disk) is paid once per network selection rather than on every
    sidebar widget interaction/rerun.
    """
    cfg = PyPRConfig({"network": network})
    names = load_reaction_names(cfg, network)
    net = load_network(cfg, era="LT", reaction_names=names)
    # net.names[0] is the prepended weak "n__p", absent from the network text
    # file/``names`` list -- skip it so the mapping is keyed by bare names.
    return {
        name: _equation_unicode(net.reaction_equation(i))
        for i, name in enumerate(net.names) if i > 0
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


def _network_label(network):
    """Return ``"<network> (<n>)"`` for display in the selectbox, ``<n>`` being
    the reaction count.

    ``load_reaction_names`` already special-cases 'small' (no file on disk,
    just the hard-coded :data:`ORDER_SMALL`/``_KEY12_REACTIONS`` list) and
    'large' (filtered by ``amax`` via ``PyPRConfig``), so it is the single
    source of truth for the count -- counting lines in the network's own
    ``.txt`` file would under-count both of those.
    """
    n = len(load_reaction_names(PyPRConfig({"network": network}), network))
    return f"{network} ({n})"


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
        return st.selectbox(label, options, index=index, help=help_text, key=key,
                             format_func=_network_label)

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


def _render_custom_reactions(params):
    """Render the "Customise Reactions" panel, in the Nuclear reactions group.

    Lets the user drop reactions from the selected network and/or substitute a
    rate table for any kept reaction, entirely in-memory (``st.session_state``
    / uploaded-file buffers -- nothing is written to disk, see the module's
    plan/README). Available for every network including ``network="large"``
    (~428 reactions): the large list is rendered too, with a heads-up caption
    that it may take a moment.

    Side effect: when active, sets ``params["custom_network"]`` to a
    JSON-encoded ``{"removed": [...], "replaced": {name: raw_text, ...}}``
    dict (consumed by ``pyprimat.gui.app._solve`` / ``PyPR(custom_network=...)``
    via ``pyprimat.network_data.UpdateNuclearRates``), and stores the decoded
    dict in ``st.session_state["custom_network_dict"]`` for the Reactions
    tab's export button.
    """
    network = params.get("network", DEFAULT_PARAMS["network"])
    cfg = PyPRConfig({"network": network})
    try:
        reaction_entries = load_reaction_names(cfg, network)
    except Exception:
        return
    bare_names = [re.split(r'[, ]+', e, maxsplit=1)[0].strip() for e in reaction_entries]

    def _reset_customisation_state():
        """Drop every uploaded file and removed/kept choice from memory."""
        for key in list(st.session_state):
            if (key.startswith("keep_") or key.startswith("upload_")
                    or key in ("custom_import", "custom_import_target",
                               "_customise_imported_id")):
                del st.session_state[key]
        st.session_state["_customise_replaced"] = {}
        st.session_state["_customise_upload_version"] = {}
        st.session_state["custom_network_dict"] = None

    # Reset per-reaction widget state when the selected network changes, so
    # removed/replaced choices from a previous network don't leak in.
    if st.session_state.get("_customise_network") != network:
        _reset_customisation_state()
        st.session_state["_customise_network"] = network

    customise = st.checkbox(
        "Customise Reactions", value=False, key="customise_reactions",
        help="Drop reactions and/or upload a custom rate table for any kept "
             "reaction in this network. Nothing is written to disk.",
    )
    if not customise:
        # Untoggling forgets everything: uploaded files, removed/kept
        # choices, and the imported-zip tracker, so re-enabling later starts
        # from a clean slate rather than restoring the previous session.
        if st.session_state.get("_customise_was_on"):
            _reset_customisation_state()
        st.session_state["_customise_was_on"] = False
        return
    st.session_state["_customise_was_on"] = True

    # The large network unfolds into ~428 checkbox+uploader rows; warn that this
    # is intentional and may take a moment to render, rather than leaving the
    # user wondering why the sidebar grew so long.
    if len(bare_names) > _CUSTOMISABLE_REACTIONS_WARN:
        st.caption(
            f"This network has {len(bare_names)} reactions; the list below may "
            "take a moment to render."
        )

    replaced_text = st.session_state.setdefault("_customise_replaced", {})

    # ---- Import a previously exported customisation (zip). Per-reaction
    # table replacement is done directly on each reaction's own uploader
    # below, not here. ---------------------------------------------------
    import_file = st.file_uploader(
        "Either import previous customization", type=["zip"], key="custom_import",
        help="Re-apply a customisation previously exported from the "
             "Reactions tab of the results (`networks/custom.txt` + "
             "`tables/*_custom.txt`). "
             "It carries its own removed/kept reaction list and replacement "
             "tables, so it is applied directly -- no extra input needed. "
             "To replace a single reaction's table, use that reaction's own "
             "uploader below instead.",
    )
    if import_file is not None:
        # Auto-applied once per upload (gated on file_id): a zip carries its
        # own reaction selection, so there is no further user input to wait
        # for before applying it.
        last_imported = st.session_state.get("_customise_imported_id")
        if import_file.file_id != last_imported:
            st.session_state["_customise_imported_id"] = import_file.file_id
            try:
                result = custom_rates.import_zip(import_file)
            except Exception as exc:
                st.error(f"Could not import zip: {exc}")
            else:
                kept = set(result["kept"])
                for name in bare_names:
                    st.session_state[f"keep_{name}"] = name in kept
                st.session_state["_customise_replaced"] = dict(result["replaced"])
                replaced_text = st.session_state["_customise_replaced"]
                st.rerun()

    # The per-reaction uploaders are deliberately compact: hide the dropzone's
    # icon, "Drag and drop file here" text and "Limit 200MB per file..."
    # caption -- all noise in a one-row-per-reaction list with KB-sized rate
    # tables -- leaving just the "Browse files" button, and trim the
    # dropzone's own padding so it doesn't render as an oversized white box.
    st.markdown(
        "<style>"
        "[data-testid='stFileUploaderDropzoneInstructions'] {display: none;}"
        "[data-testid='stFileUploaderDropzone'] {padding: 0.25rem; min-height: 0;}"
        "</style>",
        unsafe_allow_html=True,
    )

    equations = _reaction_equations(network)
    # Streamlit ties a file_uploader's value to its widget *key*: deleting
    # st.session_state[key] alone does not clear an already-uploaded file,
    # since the browser-held upload is resubmitted on rerun as long as the
    # key is unchanged. So each reaction's uploader key carries a version
    # counter that the reset button bumps, forcing a fresh (empty) widget
    # instance instead of the stale one.
    upload_versions = st.session_state.setdefault("_customise_upload_version", {})

    st.markdown(
        "Or modify reactions below",
        help="Uploaded rate tables must have three whitespace-separated "
             "columns: temperature [GK], rate, and an uncertainty factor. "
             "Lines that are not data (e.g. comments) should start with `#` "
             "so they are ignored. After running, the customization details "
             "can be downloaded from the Reactions tab.",
    )

    removed = []
    for name in bare_names:
        # Checkbox + reset button share a line; the uploader gets its own
        # full-width line below so it has room to render without squeezing
        # against (or overlapping) the reset button.  The checkbox column is
        # widened (vs. the [4, 1] split used before the equation was added)
        # to fit "name  a + b <-> c + d" without wrapping.
        head_cols = st.columns([7, 1])
        # Flag a reaction whose rate table is no longer the shipped default
        # right on the checkbox label -- this is the only place a table that
        # arrived via a *zip* import (rather than this row's own uploader,
        # which would show its own "uploaded file" chip) is visible at all.
        # Show the equation alone (e.g. "n + p ↔ ²H") rather than the bare
        # PRIMAT name (e.g. "n_p__d_g") -- the equation is self-explanatory and
        # matches the Reactions tab; fall back to the bare name only if no
        # equation could be derived (shouldn't normally happen).
        label = equations.get(name, name)
        if name in replaced_text:
            label += "  *(custom table)*"
        keep = head_cols[0].checkbox(label, value=True, key=f"keep_{name}")
        if not keep:
            removed.append(name)
            replaced_text.pop(name, None)
        has_override = name in replaced_text
        if head_cols[1].button(
            "↺", key=f"clear_{name}", disabled=not has_override,
            help="Remove the uploaded table and revert to the default rate "
                 "table for this reaction." if has_override
                 else "No custom table uploaded for this reaction.",
        ):
            replaced_text.pop(name, None)
            # Bump this reaction's uploader version so it gets a brand-new
            # widget key next run -- the only reliable way to make a
            # file_uploader forget a previously uploaded file (see comment
            # above the upload_versions assignment).
            upload_versions[name] = upload_versions.get(name, 0) + 1
            st.rerun()

        upload = st.file_uploader(
            "rate table", key=f"upload_{name}_{upload_versions.get(name, 0)}",
            disabled=not keep, label_visibility="collapsed",
        )
        if keep and upload is not None:
            raw_bytes = upload.getvalue()
            raw_text = raw_bytes.decode()
            try:
                custom_rates.parse_rate_upload(io.StringIO(raw_text))
            except Exception as exc:
                st.error(f"`{name}`: {exc}")
            else:
                replaced_text[name] = raw_text

    replaced = {n: t for n, t in replaced_text.items() if n not in removed}
    custom_network = {"removed": removed, "replaced": replaced}
    st.session_state["custom_network_dict"] = custom_network
    if removed or replaced:
        params["custom_network"] = json.dumps(custom_network, sort_keys=True)


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
                                label, min_value=7, value=20, step=1,
                                help=help_text, key="amax_value",
                            ))
                        continue

                value = _widget_for(key, label, help_text)
                if value != DEFAULT_PARAMS[key]:
                    params[key] = value

            if group == "Nuclear reactions":
                _render_custom_reactions(params)

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
