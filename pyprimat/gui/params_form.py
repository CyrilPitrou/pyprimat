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

Custom networks
----------------
"Nuclear reactions" carries two buttons, "Import custom network" and "Create
custom network", opening the ``st.dialog`` popups in §5 onward of
CUSTOMPOPUP.md instead of the old inline "Customise Reactions" checkbox list.
See :func:`_custom_network_dialog`/:func:`_import_dialog`.
"""
import importlib.resources
import json
import os
import re

import streamlit as st

from pyprimat.config import DEFAULT_PARAMS, PyPRConfig
from pyprimat.network_data import (
    load_network, load_reaction_names, reaction_category,
    group_reactions_by_category, available_rate_tables, reaction_stoichiometry,
    AMAX_LARGE,
)
from pyprimat.gui import custom_rates
from pyprimat.gui.panels import _equation_unicode


# ---------------------------------------------------------------------------
# Curated metadata for the "headline" parameters shown by default.
#
# Each entry maps a DEFAULT_PARAMS key to (group, label, help_text).  Group
# order/visibility is controlled by GROUP_ORDER below; within a group, keys
# are rendered in the (insertion) order they appear here -- this matters for
# `spectral_distortions` (must be set before its sub-options are shown/hidden)
# and for the "Physics" group's Weak rates / Plasma physics / Nuclear QED
# sub-headings (`_SUBHEADING` below).
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
        "'small' (12 reactions), 'small_parthenope' (12 reactions, Parthenope "
        "3.0 rate tables, for comparison runs), or 'large' (~429 reactions, "
        "~59 nuclides, optionally restricted via 'Limit max mass number' "
        "below). The HT/MT eras are unaffected (always n<->p / fixed "
        "18-reaction set). Manually changing this clears any active custom "
        "network built via \"Create custom network\".",
    ),
    "amax": (
        "Nuclear reactions", "Max mass number A",
        "Drop reactions involving any nuclide with mass number A > amax "
        "(must be an integer >= 2). Applies to any network above. Leave "
        "unchecked to keep every reaction.",
    ),
    "nuclear_qed_corrections": (
        "Physics", "Nuclear QED rate corrections",
        "True (default): apply a T9-dependent QED rescaling (Pitrou & "
        "Pospelov 2020) to the forward rates of n_p__d_g, d_p__He3_g, t_p__a_g, "
        "t_a__Li7_g, He3_a__Be7_g.",
    ),

    # ---- Physics: weak rates ---------------------------------------------------
    "incomplete_decoupling": (
        "Physics", "Incomplete neutrino decoupling",
        "True (default): non-instantaneous decoupling using the precomputed "
        "NEVO table (ν flavour temperatures differ slightly due to "
        "partial reheating by e+e- annihilation). False: instantaneous "
        "decoupling, Tν/Tγ = (4/11)^(1/3).",
    ),
    "radiative_corrections": (
        "Physics", "Radiative corrections (n↔p)",
        "Include T=0 Coulomb + resummed radiative corrections (CCR, Phys. Rep. "
        "Eq. 101; Czarnecki et al. 2004).  When False the crude Born approximation "
        "is used instead.",
    ),
    "finite_mass_corrections": (
        "Physics", "Finite-mass corrections (n↔p)",
        "Include the Fokker-Planck finite-nucleon-mass correction to the n↔p rate "
        "(Phys. Rep. §III.G).  Uses FMCCR when radiative_corrections=True, "
        "FMNoCCR otherwise.",
    ),
    "thermal_corrections": (
        "Physics", "Thermal radiative corrections (n↔p)",
        "Include finite-temperature radiative corrections to the n↔p rate "
        "(CCRTh; Brown & Sawyer 2001, Phys. Rep. §III.H).",
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

    # ---- Physics: plasma physics ----------------------------------------------
    "QED_corrections": (
        "Physics", "QED plasma corrections",
        "Include QED interaction corrections to the electromagnetic plasma "
        "equation of state (electron/positron pressure and density).",
    ),
}

# Order (and default expanded/collapsed state) of the curated sidebar groups.
GROUP_ORDER = ["Cosmology", "Nuclear reactions", "Physics"]
_EXPANDED_GROUPS = {"Cosmology", "Nuclear reactions"}

# Sub-heading printed in the "Physics" group right before the first key of
# each cluster (Weak rates / Plasma physics / Nuclear QED), so the merged
# group doesn't read as an undifferentiated wall of toggles. Keyed by the
# first _FORM_METADATA key of each cluster, in the insertion order above.
_SUBHEADING = {
    "incomplete_decoupling": "Weak rates",
    "QED_corrections": "Plasma physics",
    "nuclear_qed_corrections": "Nuclear QED",
}

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
# `amax` is unconditional (any network), so it is handled directly in
# render_sidebar_form rather than through this table.
_CONDITIONAL = {
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


def _equation_for(name):
    """Best-effort equation string for ``name``, for the popup's reaction rows.

    Looks the name up among the full ``large`` network's reactions first
    (covers every shipped/network-file reaction); falls back to deriving the
    equation directly from :func:`reaction_stoichiometry` for a brand-new
    GUI-added reaction that isn't in any catalog yet.
    """
    equations = _reaction_equations("large")
    if name in equations:
        return equations[name]
    try:
        react, prod = reaction_stoichiometry(name)
    except (ValueError, KeyError):
        return name

    def side(counts):
        return " + ".join(s for s, c in counts.items() for _ in range(int(c)))

    return _equation_unicode(f"{side(react)} <-> {side(prod)}")


def _bare(entry):
    """Strip a "name, filename.txt" network-file entry down to its bare name."""
    return re.split(r'[, ]+', entry, maxsplit=1)[0].strip()


@st.cache_resource(show_spinner=False)
def _cfg():
    """A throwaway ``PyPRConfig`` for helpers that only need ``cfg.data_dir``."""
    return PyPRConfig()


def _available_networks():
    """Return the selectable values for the ``network`` parameter.

    'small' is PyPRIMAT's built-in default network and needs no file; the
    other choices are discovered from ``pyprimat/rates/nuclear/networks/*.txt``
    (see ``PyPRConfig.__init__``, which validates ``network`` against exactly
    these files for any value other than 'small'). When a custom network is
    active (built via "Create custom network" or restored via "Import custom
    network"), its title is appended as a synthetic, display-only entry
    (CUSTOMPOPUP.md §7.2) -- selecting any *other* entry here clears it.
    """
    names = {"small"}
    try:
        net_dir = importlib.resources.files("pyprimat") / "rates" / "nuclear" / "networks"
        names |= {p.stem for p in net_dir.iterdir() if p.suffix == ".txt"}
    except (FileNotFoundError, ModuleNotFoundError, NotADirectoryError):
        pass
    result = sorted(names)
    active = st.session_state.get("_active_custom_network")
    if active and active["title"] not in result:
        result = result + [active["title"]]
    return result


def _network_label(network):
    """Return ``"<network> (<n>)"`` for display in the selectbox, ``<n>`` being
    the reaction count.

    ``load_reaction_names`` already special-cases 'small' (no file on disk,
    just the hard-coded :data:`ORDER_SMALL`/``_KEY12_REACTIONS`` list); for the
    active custom network's synthetic entry, the count comes straight from its
    stored kept-reaction list instead (there is no on-disk file to read).
    """
    active = st.session_state.get("_active_custom_network")
    if active and active["title"] == network:
        return f"{network} ({len(active['kept'])})"
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
        # Only pass `index` on the very first render (before the widget has
        # a session_state entry of its own) -- once it does (including via
        # the "_pending_network_label" mechanism above), passing both raises
        # a Streamlit warning about a value set through two different paths.
        kwargs = {}
        if key not in st.session_state:
            kwargs["index"] = options.index(default) if default in options else 0
        return st.selectbox(label, options, help=help_text, key=key,
                             format_func=_network_label, **kwargs)

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


# ---------------------------------------------------------------------------
# "Create custom network" / "Import custom network" buttons + dialogs
# (CUSTOMPOPUP.md §5-§8)
# ---------------------------------------------------------------------------

def _sanitize_filename(title):
    """Turn a free-text network title into a safe zip/filename stem."""
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', (title or "").strip())
    return cleaned.strip("_") or "custom"


def _category_nuclide_hint(cat):
    """E.g. ``", He3, t"`` for category 3 -- the nuclides newly unlocked there."""
    names = sorted(s for s, (n, z) in PyPRConfig.Nuclides.items()
                   if n + z == cat and s not in ("n", "p"))
    return (", " + ", ".join(names)) if names else ""


def _dialog_superset_entries(dialog_amax):
    """The large network's reaction entries, filtered by ``dialog_amax``.

    This is the full row set the popup renders, regardless of which named
    network is the "Select Network to modify" base (CUSTOMPOPUP.md §6.3):
    reactions in the base network's own list start checked, every other entry
    within the amax band starts unchecked.
    """
    entries = load_reaction_names(_cfg(), "large")
    if dialog_amax is None:
        return entries
    return [e for e in entries if reaction_category(_bare(e)) <= dialog_amax]


def _dialog_base_selection_and_tables(base_network, dialog_amax):
    """``(kept_bare_names, {bare_name: filename})`` for the chosen base network.

    For a *named* network (small/small_parthenope/large), reads its own
    reaction-list file (filtered by ``dialog_amax``) and the "name,
    filename.txt" syntax it may use (e.g. small_parthenope's
    ``*_parthenope3.0.txt`` tables) for the pre-selected rate table.  For a
    previously built/imported *custom* network (CUSTOMPOPUP.md §7.4), reads
    straight from ``st.session_state["_known_custom_networks"]`` instead --
    it is already a concrete, fully-resolved list, no amax filtering needed.
    """
    known = st.session_state.get("_known_custom_networks", {})
    if base_network in known:
        return set(known[base_network]["kept"]), {}

    entries = load_reaction_names(_cfg(), base_network)
    kept = set()
    tables = {}
    for entry in entries:
        parts = re.split(r'[, ]+', entry, maxsplit=1)
        bare = parts[0].strip()
        if dialog_amax is not None and reaction_category(bare) > dialog_amax:
            continue
        kept.add(bare)
        if len(parts) > 1:
            tables[bare] = parts[1].strip()
    return kept, tables


def _reset_dialog_reaction_state(base_network, dialog_amax):
    """(Re-)initialise the dialog's per-reaction state for a (network, amax) pair.

    Mirrors the old ``_customise_network`` guard: whenever the user changes
    "Select Network to modify" or its amax, every per-reaction toggle/table
    choice is rebuilt from scratch for the new base.
    """
    known = st.session_state.get("_known_custom_networks", {})
    keep, table_choice, uploaded = {}, {}, {}
    if base_network in known:
        info = known[base_network]
        for name in info["kept"]:
            keep[name] = True
            raw = info.get("tables", {}).get(name)
            if raw is not None:
                basename = f"{name}_custom.txt"
                uploaded.setdefault(name, {})[basename] = raw
                table_choice[name] = basename
    else:
        base_kept, base_tables = _dialog_base_selection_and_tables(base_network, dialog_amax)
        for name in base_kept:
            keep[name] = True
            if name in base_tables:
                table_choice[name] = base_tables[name]
    st.session_state["_dialog_keep"] = keep
    st.session_state["_dialog_table_choice"] = table_choice
    st.session_state["_dialog_uploaded_tables"] = uploaded
    st.session_state["_dialog_added"] = {}


def _dialog_network_options():
    """Named networks plus any custom network known this session (§7.4)."""
    return _available_networks() + [
        n for n in st.session_state.get("_known_custom_networks", {})
        if n not in _available_networks()
    ]


def _dialog_network_label(name):
    known = st.session_state.get("_known_custom_networks", {})
    if name in known:
        return f"{name} ({len(known[name]['kept'])})"
    return _network_label(name)


def _resolved_table_exists(name):
    """Whether ``name`` (currently toggled "keep") actually has a rate table."""
    if name in st.session_state.get("_dialog_added", {}):
        return True
    if name in st.session_state.get("_dialog_uploaded_tables", {}):
        return True
    return bool(available_rate_tables(name, _cfg()))


def _dialog_kept_names():
    """Every reaction the user has toggled on *and* that resolves to a table.

    A kept reaction with no resolved table (only possible for a brand-new
    "Add new rate" addition whose upload somehow didn't register) is dropped
    here with a warning rather than failing deep inside ``load_network``.
    """
    keep_map = st.session_state.get("_dialog_keep", {})
    kept, missing = [], []
    for name, is_kept in keep_map.items():
        if not is_kept:
            continue
        (kept if _resolved_table_exists(name) else missing).append(name)
    if missing:
        st.warning(
            "Skipping reaction(s) with no rate table: " + ", ".join(sorted(missing))
        )
    return kept


def _dialog_to_custom_network(kept_names):
    """Build the ``{"removed", "replaced", "added"}`` dict from live dialog state."""
    dialog_amax = (st.session_state.get("_dialog_amax_value")
                   if st.session_state.get("_dialog_amax_enabled") else None)
    superset = {_bare(e) for e in _dialog_superset_entries(dialog_amax)}
    kept_set = set(kept_names)
    removed = sorted(superset - kept_set)

    table_choice = st.session_state.get("_dialog_table_choice", {})
    uploaded = st.session_state.get("_dialog_uploaded_tables", {})
    added = dict(st.session_state.get("_dialog_added", {}))

    cfg = _cfg()
    replaced = {}
    for name in kept_names:
        if name in added:
            continue
        choice = table_choice.get(name)
        if choice is None or choice == f"{name}.txt":
            continue   # shipped default, nothing to override
        if choice in uploaded.get(name, {}):
            replaced[name] = uploaded[name][choice]
        else:
            # An on-disk alternate filename (e.g. a "*_parthenope3.0.txt"
            # sibling) -- load_network's custom_tables mechanism only knows
            # raw text, not filenames, so resolve to text here.
            path = os.path.join(cfg.data_dir, "rates", "nuclear", "tables", name, choice)
            try:
                with open(path) as f:
                    replaced[name] = f.read()
            except OSError:
                pass
    return {"removed": removed, "replaced": replaced, "added": added}


def _kept_to_custom_network(kept, replaced):
    """Build the ``{"removed", "replaced", "added"}`` dict from an imported zip.

    ``amax`` is re-derived purely from ``kept`` (CUSTOMPOPUP.md §7.1): the
    heaviest category among ``kept`` reactions, treated as "no filter" once it
    reaches :data:`AMAX_LARGE`.
    """
    entries = load_reaction_names(_cfg(), "large")
    bare_names = {_bare(e) for e in entries}
    implied_amax = max((reaction_category(n) for n in kept), default=AMAX_LARGE)
    if implied_amax >= AMAX_LARGE:
        superset = bare_names
    else:
        superset = {n for n in bare_names if reaction_category(n) <= implied_amax}
    kept_set = set(kept)
    removed = sorted(superset - kept_set)
    added = {n: replaced[n] for n in kept_set - bare_names if n in replaced}
    true_replaced = {n: t for n, t in replaced.items() if n not in added}
    return {"removed": removed, "replaced": true_replaced, "added": added}


def _render_category(cat, names):
    """One foldable mass-number category: select/deselect-all + reaction rows."""
    with st.expander(f"Category {cat} (A <= {cat}{_category_nuclide_hint(cat)})",
                     expanded=False):
        c1, c2 = st.columns(2)
        if c1.button("Select all", key=f"_dialog_selall_{cat}"):
            for n in names:
                st.session_state["_dialog_keep"][n] = True
            st.rerun()
        if c2.button("Deselect all", key=f"_dialog_deselall_{cat}"):
            for n in names:
                st.session_state["_dialog_keep"][n] = False
            st.rerun()
        for name in names:
            _render_reaction_row(name)


def _render_reaction_row(name):
    """One reaction's toggle + equation + rate-table picker + uploader."""
    keep_map = st.session_state["_dialog_keep"]
    default = keep_map.get(name, False)
    equation = _equation_for(name)
    tables = available_rate_tables(name, _cfg()) + [
        b for b in st.session_state["_dialog_uploaded_tables"].get(name, {})
        if b not in available_rate_tables(name, _cfg())
    ]

    cols = st.columns([1, 4, 3, 2])
    keep_map[name] = cols[0].toggle("keep", value=default, key=f"_dialog_keep_{name}",
                                    label_visibility="collapsed")
    cols[1].markdown(equation)
    if len(tables) > 1:
        current = st.session_state["_dialog_table_choice"].get(name, tables[0])
        index = tables.index(current) if current in tables else 0
        choice = cols[2].selectbox(
            "table", tables, key=f"_dialog_table_{name}", index=index,
            label_visibility="collapsed",
        )
        st.session_state["_dialog_table_choice"][name] = choice
    else:
        cols[2].caption(tables[0] if tables else "(no table)")
        if tables:
            st.session_state["_dialog_table_choice"].setdefault(name, tables[0])

    if cols[3].button("Add new rate table", key=f"_dialog_addtable_{name}"):
        st.session_state[f"_dialog_show_uploader_{name}"] = True
    if st.session_state.get(f"_dialog_show_uploader_{name}"):
        up = st.file_uploader("New table", key=f"_dialog_upload_{name}",
                              label_visibility="collapsed")
        if up is not None:
            raw = up.getvalue().decode()
            try:
                custom_rates.parse_rate_upload(raw)
            except Exception as exc:
                st.error(f"`{name}`: {exc}")
            else:
                basename = up.name
                st.session_state["_dialog_uploaded_tables"].setdefault(name, {})[basename] = raw
                st.session_state["_dialog_table_choice"][name] = basename
                st.session_state[f"_dialog_show_uploader_{name}"] = False
                st.rerun()


def _render_add_rate_section(dialog_amax, all_entries):
    """"Add new rate" pop-up: a brand-new reaction not in the current selection.

    Two checks beyond the live stoichiometry/conservation validation already
    done by :func:`custom_rates.validate_new_reaction`: the name must not
    already exist in the current selection, and it must not exceed the
    dialog's active ``amax`` -- checked in that order (cheap/no-upload-needed
    check first) before requiring the rate-table upload.
    """
    st.divider()
    with st.popover("Add new rate", use_container_width=True):
        name = st.text_input("Reaction name", key="_dialog_add_name",
                             placeholder="He3_d__He4_p")
        parsed_ok = False
        if name.strip():
            try:
                eq = custom_rates.validate_new_reaction(name)
            except ValueError as exc:
                st.error(str(exc))
            else:
                parsed_ok = True
                st.success(f"Parsed as: {eq}")
        upload = st.file_uploader("Rate table", key="_dialog_add_table")
        if st.button("Add reaction", key="_dialog_add_submit", disabled=not parsed_ok):
            bare = name.strip()
            existing = {_bare(e) for e in all_entries} | set(st.session_state["_dialog_added"])
            if bare in existing:
                st.error(f"'{bare}' already exists in the current selection.")
                return
            try:
                cat = reaction_category(bare)
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
                return
            if dialog_amax is not None and cat > dialog_amax:
                st.error(
                    f"reaction {bare!r} involves a nuclide with A={cat}, which "
                    f"exceeds the current amax={dialog_amax}.")
                return
            if upload is None:
                st.error("Upload a rate table for the new reaction first.")
                return
            raw = upload.getvalue().decode()
            try:
                custom_rates.parse_rate_upload(raw)
            except Exception as exc:
                st.error(f"Rate table: {exc}")
                return
            st.session_state["_dialog_added"][bare] = raw
            st.session_state["_dialog_keep"][bare] = True
            st.session_state["_dialog_table_choice"][bare] = upload.name
            st.rerun()


def _render_dialog_footer(params, title, base_network, dialog_amax):
    """"Save custom network" (download) + the very large "Apply and run BBN"."""
    st.divider()
    kept_names = _dialog_kept_names()

    if st.button("Save custom network", use_container_width=True, key="_dialog_save"):
        custom_network = _dialog_to_custom_network(kept_names)
        zip_bytes = custom_rates.export_zip(
            _cfg(), custom_network, kept_names, network_filename=_sanitize_filename(title))
        st.session_state["_dialog_zip_bytes"] = zip_bytes
    if st.session_state.get("_dialog_zip_bytes") is not None:
        st.download_button(
            f"Download {title}.zip", data=st.session_state["_dialog_zip_bytes"],
            file_name=f"{_sanitize_filename(title)}.zip", mime="application/zip",
            key="_dialog_download", use_container_width=True,
        )

    if st.button("Apply and run BBN", type="primary", use_container_width=True,
                 key="_dialog_apply"):
        custom_network = _dialog_to_custom_network(kept_names)
        new_params = dict(params)
        new_params["network"] = "large"
        new_params.pop("amax", None)
        new_params["custom_network"] = json.dumps(custom_network, sort_keys=True)

        st.session_state["_active_custom_network"] = {
            "title": title, "kept": kept_names, "custom_network": custom_network,
        }
        st.session_state.setdefault("_known_custom_networks", {})
        st.session_state["_known_custom_networks"][title] = {
            "kept": kept_names,
            "tables": {**custom_network.get("replaced", {}), **custom_network.get("added", {})},
        }
        # Mirror app.py's "Run BBN" click handler (the bottom of app.main()
        # re-solves unconditionally from these session_state keys on every
        # rerun) -- there is no separate trigger mechanism needed.
        st.session_state["params"] = new_params
        st.session_state["quick_mc"] = st.session_state.get("quick_mc_uncertainty", False)
        st.session_state["mc_samples"] = st.session_state.get("quick_mc_samples", 30)
        st.session_state["run_custom_network_dict"] = custom_network
        # The sidebar's "network" selectbox widget was already instantiated
        # earlier in this same script run, so its session_state value cannot
        # be set directly here (Streamlit forbids mutating an
        # already-instantiated widget's key) -- stash it and let
        # render_sidebar_form apply it at the very top of the *next* run.
        st.session_state["_pending_network_label"] = title
        st.session_state["_show_custom_dialog"] = False
        st.session_state["_dialog_zip_bytes"] = None
        st.rerun()


@st.dialog("Create custom network", width="large")
def _custom_network_dialog(params):
    st.session_state.setdefault("_dialog_title", "custom")
    title = st.text_input("Network title", key="_dialog_title")

    st.markdown("**Select Network to modify**")
    col1, col2 = st.columns([2, 1])
    options = _dialog_network_options()
    if st.session_state.get("_dialog_base_network") not in options:
        st.session_state["_dialog_base_network"] = options[0]
    base_network = col1.selectbox(
        "Network", options, key="_dialog_base_network",
        format_func=_dialog_network_label,
    )
    amax_enabled = col2.checkbox("Limit A", key="_dialog_amax_enabled")
    dialog_amax = None
    if amax_enabled:
        dialog_amax = int(col2.number_input(
            "amax", min_value=2, value=st.session_state.get("_dialog_amax_value", 20),
            key="_dialog_amax_value",
        ))

    # Reset per-reaction state if the (base_network, amax) pair changed since
    # the dialog last computed it -- mirrors the old _customise_network guard.
    sig = (base_network, dialog_amax)
    if st.session_state.get("_dialog_signature") != sig:
        _reset_dialog_reaction_state(base_network, dialog_amax)
        st.session_state["_dialog_signature"] = sig

    all_entries = _dialog_superset_entries(dialog_amax)
    bare_all = [_bare(e) for e in all_entries]
    groups = group_reactions_by_category(bare_all + list(st.session_state["_dialog_added"]))

    for cat in sorted(groups):
        _render_category(cat, groups[cat])

    _render_add_rate_section(dialog_amax, all_entries)
    _render_dialog_footer(params, title, base_network, dialog_amax)


@st.dialog("Import custom network")
def _import_dialog():
    fh = st.file_uploader("Custom network zip", type=["zip"], key="_import_dialog_upload")
    if fh is not None:
        try:
            result = custom_rates.import_zip(fh)
        except Exception as exc:
            st.error(f"Could not import zip: {exc}")
            return
        title = result["title"]
        kept = result["kept"]
        st.session_state.setdefault("_known_custom_networks", {})
        st.session_state["_known_custom_networks"][title] = {
            "kept": kept, "tables": dict(result["replaced"]),
        }
        custom_network = _kept_to_custom_network(kept, result["replaced"])
        st.session_state["_active_custom_network"] = {
            "title": title, "kept": kept, "custom_network": custom_network,
        }
        st.session_state["_pending_network_label"] = title
        st.session_state["_show_import_dialog"] = False
        st.rerun()


def _render_custom_network_buttons(params):
    """Two buttons, in the "Nuclear reactions" group, opening the popups above."""
    st.session_state.setdefault("_known_custom_networks", {})
    cols = st.columns(2)
    if cols[0].button("Import custom network", use_container_width=True):
        st.session_state["_show_import_dialog"] = True
    if cols[1].button("Create custom network", use_container_width=True):
        st.session_state["_show_custom_dialog"] = True

    if st.session_state.get("_show_import_dialog"):
        _import_dialog()
    if st.session_state.get("_show_custom_dialog"):
        _custom_network_dialog(params)


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
    # Apply a pending "select this synthetic custom-network entry" request
    # from the previous run (see _render_dialog_footer/_import_dialog) before
    # the "network" widget below is instantiated -- Streamlit forbids setting
    # an already-instantiated widget's session_state value directly.
    pending_network = st.session_state.pop("_pending_network_label", None)
    if pending_network is not None:
        st.session_state["network"] = pending_network

    params = {}

    st.sidebar.header("Parameters")

    # ---- Curated groups -----------------------------------------------------
    by_group = {g: [] for g in GROUP_ORDER}
    for key, (group, label, help_text) in _FORM_METADATA.items():
        by_group[group].append((key, label, help_text))

    for group in GROUP_ORDER:
        with st.sidebar.expander(group, expanded=(group in _EXPANDED_GROUPS)):
            current_heading = None
            for key, label, help_text in by_group[group]:
                if key in _SUBHEADING and _SUBHEADING[key] != current_heading:
                    current_heading = _SUBHEADING[key]
                    st.markdown(f"**{current_heading}**")

                if key == "amax":
                    # `amax` defaults to None, so it needs an explicit
                    # enable/disable checkbox rather than a bare number
                    # input (which could never represent "no filter"). Now
                    # offered for every network, not just "large".
                    enabled = st.checkbox(
                        "Limit max mass number", value=False,
                        help=help_text, key="amax_enabled",
                    )
                    if enabled:
                        params["amax"] = int(st.number_input(
                            label, min_value=2, value=20, step=1,
                            help=help_text, key="amax_value",
                        ))
                    continue

                if key in _CONDITIONAL:
                    ctrl_key, required = _CONDITIONAL[key]
                    current = params.get(ctrl_key, DEFAULT_PARAMS[ctrl_key])
                    if current != required:
                        continue

                if key == "network":
                    value = _widget_for(key, label, help_text)
                    active = st.session_state.get("_active_custom_network")
                    if active and value == active["title"]:
                        # Synthetic entry: the underlying network is always
                        # "large", driven entirely by custom_network (§7.2).
                        params["network"] = "large"
                        params["custom_network"] = json.dumps(
                            active["custom_network"], sort_keys=True)
                    else:
                        if active and value != active["title"]:
                            # Manually picking a different network abandons
                            # the active custom network (§7.2).
                            st.session_state["_active_custom_network"] = None
                        if value != DEFAULT_PARAMS[key]:
                            params[key] = value
                    continue

                value = _widget_for(key, label, help_text)
                if value != DEFAULT_PARAMS[key]:
                    params[key] = value

            if group == "Nuclear reactions":
                _render_custom_network_buttons(params)

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
