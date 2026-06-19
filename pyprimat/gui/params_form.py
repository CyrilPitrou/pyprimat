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
    these files for any value other than 'small'). Every custom network built
    or imported *this session* (``_known_custom_networks``, not just the
    currently-active one) is appended too, as a synthetic, display-only
    entry (CUSTOMPOPUP.md §7.2): picking one directly from this dropdown
    re-activates it, so switching to a real network and back to a previously
    used custom one (e.g. to compare results) works without re-opening the
    popup.
    """
    names = {"small"}
    try:
        net_dir = importlib.resources.files("pyprimat") / "rates" / "nuclear" / "networks"
        names |= {p.stem for p in net_dir.iterdir() if p.suffix == ".txt"}
    except (FileNotFoundError, ModuleNotFoundError, NotADirectoryError):
        pass
    result = sorted(names)
    for title in st.session_state.get("_known_custom_networks", {}):
        if title not in result:
            result.append(title)
    return result


def _network_label(network):
    """Return ``"<network> (<n>)"`` for display in the selectbox, ``<n>`` being
    the reaction count.

    ``load_reaction_names`` already special-cases 'small' (no file on disk,
    just the hard-coded :data:`ORDER_SMALL`/``_KEY12_REACTIONS`` list); for
    any known custom network's synthetic entry, the count comes straight
    from its stored kept-reaction list instead (there is no on-disk file to
    read).
    """
    known = st.session_state.get("_known_custom_networks", {})
    if network in known:
        return f"{network} ({len(known[network]['kept'])})"
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

_RESERVED_NETWORK_NAMES = {"small", "small_parthenope", "large"}


def _sanitize_filename(title):
    """Turn a free-text network title into a safe zip/filename stem."""
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', (title or "").strip())
    return cleaned.strip("_") or "custom"


_NUCLIDE_NAME_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _group_nuclides_by_element(nuclides):
    """One display line per element, isotopes grouped, e.g. ``"He3, He4, He6"``.

    ``n``/``p`` (which don't fit the "<symbol><mass>" naming convention) are
    kept together on their own leading line. Elements are ordered by their
    lightest isotope's mass number, matching the mass-number category order
    used throughout the rest of the dialog.
    """
    groups: dict[str, list[tuple[int, str]]] = {}
    specials = []
    for name in sorted(nuclides):
        if name in ("n", "p"):
            specials.append(name)
            continue
        m = _NUCLIDE_NAME_RE.match(name)
        if not m:
            specials.append(name)
            continue
        symbol, mass = m.group(1), int(m.group(2))
        groups.setdefault(symbol, []).append((mass, name))
    lines = []
    if specials:
        lines.append(", ".join(specials))
    for symbol in sorted(groups, key=lambda s: min(mass for mass, _ in groups[s])):
        isotopes = [n for _, n in sorted(groups[symbol])]
        lines.append(", ".join(isotopes))
    return lines


def _render_evolved_nuclides_section():
    """Foldable summary of every nuclide the currently-kept reactions touch.

    Placed right after the "Decays" category: a quick sanity check of what
    the ``amax``/per-reaction toggles above actually add up to, before
    committing via "Apply and run BBN".
    """
    keep_map = st.session_state["_dialog_keep"]
    kept = [n for n, is_kept in keep_map.items() if is_kept]
    nuclides = {"n", "p"}
    for name in kept:
        try:
            react, prod = reaction_stoichiometry(name)
        except (ValueError, KeyError):
            continue
        nuclides.update(s for s in react if s in _cfg().Nuclides)
        nuclides.update(s for s in prod if s in _cfg().Nuclides)
    with st.expander(f"Show evolved nuclides ({len(nuclides)})", expanded=False):
        for line in _group_nuclides_by_element(nuclides):
            st.markdown(f"- {line}")


def _category_nuclide_hint(cat):
    """E.g. ``", He3, t"`` for category 3 -- the nuclides newly unlocked there."""
    names = sorted(s for s, (n, z) in _cfg().Nuclides.items()
                   if n + z == cat and s not in ("n", "p"))
    return (", " + ", ".join(names)) if names else ""


def _bump_dialog_gen():
    """Force every per-reaction widget below to remount on the next render.

    Streamlit widgets keep their *own* persisted value forever once a given
    key has been used, regardless of any ``value=``/``index=`` passed on a
    later render -- so mutating the backing ``_dialog_keep``/``_dialog_table_choice``
    dicts directly (Select all/Deselect all, or rebuilding them when the base
    network changes) would otherwise have no visible effect, since the widget
    would just keep showing its own stale state. Embedding this counter in
    every per-reaction widget key forces a fresh widget (which *does* read our
    backing dict as its initial value) whenever we bump it.
    """
    st.session_state["_dialog_gen"] = st.session_state.get("_dialog_gen", 0) + 1


@st.cache_resource(show_spinner=False)
def _decay_rates():
    """``{name: (rate_s, f, halflife_s, ref)}`` from ``tables/decays.txt``.

    Backs the popup's dedicated "Decays" category (CUSTOMPOPUP.md follow-up):
    a decay reaction has no per-reaction rate-table folder (its rate is a
    single T9-independent row in the shared ``decays.txt``, see
    :func:`pyprimat.network_data._load_decay_table`), so it must be told apart
    from a genuinely tableless reaction.
    """
    from pyprimat.network_data import _load_decay_table
    tables_dir = os.path.join(_cfg().data_dir, "rates", "nuclear", "tables")
    return _load_decay_table(tables_dir)


def _is_decay(name):
    return name in _decay_rates()


def _current_table_text(name):
    """Best-effort raw text of the table currently selected for ``name``.

    Backs each reaction row's "Show rate table" popup: prefers an uploaded
    override, falls back to reading the chosen on-disk file.
    """
    added = st.session_state.get("_dialog_added", {})
    if name in added:
        return added[name]
    choice = st.session_state.get("_dialog_table_choice", {}).get(name)
    uploaded_for_name = st.session_state.get("_dialog_uploaded_tables", {}).get(name, {})
    if choice in uploaded_for_name:
        return uploaded_for_name[choice]
    if choice:
        path = os.path.join(_cfg().data_dir, "rates", "nuclear", "tables", name, choice)
        try:
            with open(path) as f:
                return f.read()
        except OSError:
            return f"(could not read {choice})"
    return "(no rate table for this reaction)"


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
            # A custom network's own "kept" list was built against *its own*
            # amax (or none at all); re-applying the dialog's current amax
            # here too is what actually shrinks the kept set when the user
            # lowers "Limit A" while a custom network is the base -- without
            # this, every one of its reactions stayed "kept" regardless of
            # amax, while the rows shown above (filtered by amax) and the
            # totals caption below disagreed (e.g. "41/17 kept").
            if dialog_amax is not None and reaction_category(name) > dialog_amax:
                continue
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
    st.session_state["_dialog_decay_override"] = {}
    # Force every reaction-row widget to remount and read the dicts just
    # rebuilt above -- otherwise the previous base network's toggle/table
    # widget states (keyed by reaction name, which can repeat across
    # networks) would silently stick around and the category list would
    # appear unchanged.
    _bump_dialog_gen()


def _dialog_network_options():
    """Named networks plus any custom network known this session (§7.4).

    ``_available_networks()`` already includes every known custom network
    (not just the active one), so this is just an alias kept for the
    dialog's own readability.
    """
    return _available_networks()


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
    if _is_decay(name):
        # Always resolved: decays.txt supplies a shipped default rate even
        # with no GUI override (see _dialog_to_custom_network).
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
    """Build the ``{"removed", "replaced", "added", "filenames"}`` dict from
    live dialog state.

    ``removed`` is computed against the *full, unfiltered* large-network
    reaction list, not the dialog's own ``amax``-filtered view: the dialog's
    "Limit A" only narrows which rows are shown/toggleable in the popup, it
    does not express an intent to keep every reaction above that cutoff.
    Using the filtered view here would silently leave every reaction the
    user never even saw (anything above the dialog's amax) in the resulting
    network -- it is neither in ``kept_names`` nor in a filtered ``removed``.

    ``filenames`` (``{name: filename}``, one entry per ``replaced``/``added``
    reaction that has a real on-disk or uploaded basename -- decay-rate
    overrides have none, they're not file-backed) is an extra, optional key
    threaded through ``UpdateNuclearRates``/``load_network`` purely so the
    Reactions summary tab can show the actual filename instead of a blank
    "File" column for a customised reaction (see ``network_data.py``'s
    ``describe_reactions``).
    """
    superset = {_bare(e) for e in _dialog_superset_entries(None)}
    kept_set = set(kept_names)
    removed = sorted(superset - kept_set)

    table_choice = st.session_state.get("_dialog_table_choice", {})
    uploaded = st.session_state.get("_dialog_uploaded_tables", {})
    added = dict(st.session_state.get("_dialog_added", {}))
    decay_overrides = st.session_state.get("_dialog_decay_override", {})

    cfg = _cfg()
    replaced = {}
    filenames = {}
    for name in kept_names:
        if name in added:
            filenames[name] = table_choice.get(name, f"{name} (uploaded)")
            continue
        if name in decay_overrides:
            replaced[name] = custom_rates.decay_override_table_text(name, decay_overrides[name])
            continue
        choice = table_choice.get(name)
        if choice is None or choice == f"{name}_primat.txt":
            continue   # shipped default, nothing to override
        if choice in uploaded.get(name, {}):
            replaced[name] = uploaded[name][choice]
            filenames[name] = choice
        else:
            # An on-disk alternate filename (e.g. a "*_parthenope3.0.txt"
            # sibling) -- load_network's custom_tables mechanism only knows
            # raw text, not filenames, so resolve to text here.
            path = os.path.join(cfg.data_dir, "rates", "nuclear", "tables", name, choice)
            try:
                with open(path) as f:
                    replaced[name] = f.read()
            except OSError:
                continue
            filenames[name] = choice
    return {"removed": removed, "replaced": replaced, "added": added, "filenames": filenames}


def _request_expander_open(expander_key):
    """Ask the named ``st.expander`` to be open on the *next* render.

    Like the sidebar's "network" selectbox (see ``_pending_network_label``),
    an expander's ``key=``-tracked open/closed state cannot be set directly
    from inside a callback that runs *after* the expander has already been
    instantiated this same script run (Select all/Deselect all's button is
    rendered inside the expander it should keep open) -- so this stashes the
    request for ``render_sidebar_form``/the dialog's top to apply just before
    that expander is (re-)created.
    """
    st.session_state["_dialog_pending_open"] = expander_key


def _kept_count(names, keep_map):
    """Count of ``names`` currently toggled on, reading the *fresh* value.

    On the rerun triggered by clicking a toggle inside this very category,
    Streamlit already has the new value sitting in
    ``st.session_state[widget_key]`` before our script body runs -- but
    ``_dialog_keep[name]`` only gets updated later, as a side effect of
    :func:`_render_reaction_row`/:func:`_render_decay_row` actually drawing
    that row.  A category header's "n/total kept" count is computed *before*
    its rows are drawn (the label has to be known up front to open the
    ``st.expander``), so reading ``_dialog_keep`` there would show the
    previous render's count -- always one click stale, which is exactly the
    "always off by one" bug this guards against.  Checking the widget key
    first (falling back to ``_dialog_keep`` for a name not yet rendered this
    generation) picks up the just-clicked value instead.
    """
    gen = st.session_state["_dialog_gen"]
    return sum(
        1 for n in names
        if st.session_state.get(f"_dialog_keep_{gen}_{n}", keep_map.get(n, False))
    )


def _render_category(cat, names):
    """One foldable mass-number category: select/deselect-all + reaction rows."""
    keep_map = st.session_state["_dialog_keep"]
    n_kept = _kept_count(names, keep_map)
    label = f"Category {cat} (A <= {cat}{_category_nuclide_hint(cat)}) -- {n_kept}/{len(names)} kept"
    expander_key = f"_dialog_expander_cat_{cat}"
    with st.expander(label, key=expander_key):
        # Narrow, content-sized columns + a trailing spacer keep the two
        # buttons right next to each other instead of "Select all" sitting
        # at the far left of a half-width column and "Unselect all" at the
        # far left of the other half (i.e. visually far apart).
        c1, c2, _spacer = st.columns([1, 1, 4])
        if c1.button("Select all", key=f"_dialog_selall_{cat}"):
            for n in names:
                st.session_state["_dialog_keep"][n] = True
            _bump_dialog_gen()
            _request_expander_open(expander_key)
            st.rerun()
        if c2.button("Unselect all", key=f"_dialog_deselall_{cat}"):
            for n in names:
                st.session_state["_dialog_keep"][n] = False
            _bump_dialog_gen()
            _request_expander_open(expander_key)
            st.rerun()
        for name in names:
            _render_reaction_row(name)


def _render_decay_category(names, decay_rates):
    """Dedicated category for analytic Bm/Bp decay reactions.

    These have no per-reaction rate-table folder (their rate is a single
    T9-independent row in the shared ``decays.txt``), so they are pulled out
    of the mass-number categories (where they used to show a misleading
    "(no table)") into their own group with an editable decay-rate field.
    """
    keep_map = st.session_state["_dialog_keep"]
    n_kept = _kept_count(names, keep_map)
    label = f"Decays -- {n_kept}/{len(names)} kept"
    expander_key = "_dialog_expander_decay"
    with st.expander(label, key=expander_key):
        # Narrow, content-sized columns + a trailing spacer keep the two
        # buttons right next to each other instead of "Select all" sitting
        # at the far left of a half-width column and "Unselect all" at the
        # far left of the other half (i.e. visually far apart).
        c1, c2, _spacer = st.columns([1, 1, 4])
        if c1.button("Select all", key="_dialog_selall_decay"):
            for n in names:
                st.session_state["_dialog_keep"][n] = True
            _bump_dialog_gen()
            _request_expander_open(expander_key)
            st.rerun()
        if c2.button("Unselect all", key="_dialog_deselall_decay"):
            for n in names:
                st.session_state["_dialog_keep"][n] = False
            _bump_dialog_gen()
            _request_expander_open(expander_key)
            st.rerun()
        for name in names:
            _render_decay_row(name, decay_rates[name])


def _render_reaction_row(name):
    """One reaction's toggle + equation + rate-table picker + uploader."""
    keep_map = st.session_state["_dialog_keep"]
    gen = st.session_state["_dialog_gen"]
    default = keep_map.get(name, False)
    equation = _equation_for(name)
    disk_tables = available_rate_tables(name, _cfg())
    uploaded_for_name = set(st.session_state["_dialog_uploaded_tables"].get(name, {}))
    if name in st.session_state["_dialog_added"]:
        # A brand-new reaction from "Add new rate" stores its table in
        # _dialog_added (not _dialog_uploaded_tables), keyed by reaction name
        # rather than basename -- surface it here under its chosen basename
        # (_dialog_table_choice[name], set at add time) so the picker doesn't
        # show a misleading "(no table)" for it.
        uploaded_for_name.add(st.session_state["_dialog_table_choice"].get(name, f"{name}.txt"))
    tables = disk_tables + [b for b in uploaded_for_name if b not in disk_tables]

    cols = st.columns([1, 3, 3, 2, 2])
    # `gen` is embedded in every widget key below so that Select all/Deselect
    # all and a base-network change (both of which mutate the backing dicts
    # directly) actually take effect -- see _bump_dialog_gen's docstring.
    keep_map[name] = cols[0].toggle("keep", value=default, key=f"_dialog_keep_{gen}_{name}",
                                    label_visibility="collapsed")
    cols[1].markdown(equation)
    if len(tables) > 1:
        current = st.session_state["_dialog_table_choice"].get(name, tables[0])
        index = tables.index(current) if current in tables else 0
        choice = cols[2].selectbox(
            "table", tables, key=f"_dialog_table_{gen}_{name}", index=index,
            label_visibility="collapsed",
        )
        st.session_state["_dialog_table_choice"][name] = choice
    else:
        cols[2].caption(tables[0] if tables else "(no table)")
        if tables:
            st.session_state["_dialog_table_choice"].setdefault(name, tables[0])

    show_uploader_key = f"_dialog_show_uploader_{name}"
    if cols[3].button("Add new rate table", key=f"_dialog_addtable_{gen}_{name}"):
        # Fold/unfold: a second click on the same reaction folds the upload
        # region back up again, e.g. if the user changed their mind.
        st.session_state[show_uploader_key] = not st.session_state.get(show_uploader_key, False)
    with cols[4].popover("Show rate table", use_container_width=True):
        st.code(_current_table_text(name), language=None)

    if st.session_state.get(show_uploader_key, False):
        # Rendered on its own full-width line below the row (not squeezed
        # into one of the narrow columns above), where the dropzone has room
        # to render normally.
        up = st.file_uploader(f"New rate table for {name}", key=f"_dialog_upload_{name}")
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
                st.session_state[show_uploader_key] = False
                # Force the table-choice selectbox above to remount so it
                # actually shows the new table selected (see
                # _bump_dialog_gen's docstring) instead of keeping whatever
                # it displayed before this upload.
                _bump_dialog_gen()
                st.rerun()


def _render_decay_row(name, info):
    """One decay reaction's toggle + equation + editable decay rate.

    Unlike :func:`_render_reaction_row`, there is no rate-table picker (a
    decay reaction has a single rate, not a per-T9 table) -- instead a
    ``number_input`` pre-filled from ``decays.txt`` lets the user override the
    constant decay rate directly.
    """
    keep_map = st.session_state["_dialog_keep"]
    gen = st.session_state["_dialog_gen"]
    default = keep_map.get(name, False)
    equation = _equation_for(name)
    shipped_rate, _f_unc, halflife_s, ref = info
    overrides = st.session_state["_dialog_decay_override"]
    current_rate = overrides.get(name, shipped_rate)

    cols = st.columns([1, 4, 3])
    keep_map[name] = cols[0].toggle("keep", value=default, key=f"_dialog_keep_{gen}_{name}",
                                    label_visibility="collapsed")
    cols[1].markdown(f"{equation}  (T½={halflife_s:.4e} s, {ref})")
    new_rate = cols[2].number_input(
        "decay rate [1/s]", value=current_rate, format="%.6e",
        key=f"_dialog_decay_rate_{gen}_{name}", label_visibility="collapsed",
        help="Constant (T-independent) decay rate = log(2) / half-life "
             "[s⁻¹]. Editing this overrides the shipped decays.txt value.",
    )
    if new_rate != shipped_rate:
        overrides[name] = new_rate
    else:
        overrides.pop(name, None)


def _render_add_rate_section(dialog_amax, all_entries):
    """"Add new rate": a brand-new reaction not in the current selection.

    Two checks beyond the live stoichiometry/conservation validation already
    done by :func:`custom_rates.validate_new_reaction`: the name must not
    already exist in the current selection, and it must not exceed the
    dialog's active ``amax`` -- checked in that order (cheap/no-upload-needed
    check first) before requiring the rate-table upload.

    Uses a plain toggled container rather than ``st.popover``: a popover's
    open/closed state cannot be set programmatically, so a successful
    "Add reaction" click could not dismiss it; this flag-controlled container
    can be collapsed (and is, on success) like any other widget.
    """
    st.divider()
    if st.button("Add new rate", key="_dialog_add_rate_open_btn"):
        st.session_state["_dialog_add_rate_open"] = True
    if not st.session_state.get("_dialog_add_rate_open"):
        return

    with st.container(border=True):
        st.caption(
            "Add a reaction that need not be in this network (or in "
            "PyPRIMAT's catalog at all). Its stoichiometry is read from the name."
        )
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
        cols = st.columns(2)
        if cols[0].button("Add reaction", key="_dialog_add_submit",
                          disabled=not parsed_ok, use_container_width=True):
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
            st.session_state["_dialog_add_rate_open"] = False
            st.rerun()
        if cols[1].button("Cancel", key="_dialog_add_cancel", use_container_width=True):
            st.session_state["_dialog_add_rate_open"] = False
            st.rerun()


def _render_dialog_footer(params, title, base_network, dialog_amax):
    """Direct "Download network details" zip + "Apply and run BBN", side by side.

    No intermediate "Save" click: the zip is built eagerly from the current
    toggle/table state on every render, so the download button is always
    immediately ready (the same amount of work an explicit "Save" button
    used to gate, just done a render earlier).
    """
    st.divider()
    kept_names = _dialog_kept_names()
    custom_network = _dialog_to_custom_network(kept_names)
    safe_title = _sanitize_filename(title)
    title_reserved = title.strip() in _RESERVED_NETWORK_NAMES

    cols = st.columns(2)
    try:
        zip_bytes = custom_rates.export_zip(
            _cfg(), custom_network, kept_names, network_filename=safe_title)
    except Exception as exc:
        cols[0].error(f"Could not build zip: {exc}")
    else:
        cols[0].download_button(
            f"Download network details ({safe_title}.zip)", data=zip_bytes,
            file_name=f"{safe_title}.zip", mime="application/zip",
            key="_dialog_download", use_container_width=True, disabled=title_reserved,
            help="Save this customisation as a re-importable .zip "
                 "(networks/<title>.txt + tables/<name>/<filename> for "
                 "every kept reaction).",
        )

    if cols[1].button("Apply and run BBN", type="primary", use_container_width=True,
                      key="_dialog_apply", disabled=title_reserved):
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
            "custom_network": custom_network,
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
        st.rerun()


def _on_custom_dialog_dismiss():
    """Clear the "open" flag when the user dismisses the dialog via its own
    "x"/Esc/click-outside (rather than our own buttons).

    Without this, dismissing the dialog that way leaves ``_show_custom_dialog``
    stuck ``True`` (Streamlit gives no other signal for it), so *any* later
    full-script rerun -- e.g. changing an unrelated sidebar number -- would
    see the flag still set and pop the dialog right back open.
    """
    st.session_state["_show_custom_dialog"] = False


def _on_import_dialog_dismiss():
    st.session_state["_show_import_dialog"] = False


@st.dialog("Create custom network", width="large", on_dismiss=_on_custom_dialog_dismiss)
def _custom_network_dialog(params):
    st.session_state.setdefault("_dialog_title", "custom")
    st.session_state.setdefault("_dialog_gen", 0)
    st.session_state.setdefault("_dialog_decay_override", {})

    # Apply a pending "keep this category expander open" request from a
    # Select all/Deselect all click on the *previous* run -- see
    # _request_expander_open's docstring for why this can't be done directly
    # from inside the button's own click handler.
    pending_open = st.session_state.pop("_dialog_pending_open", None)
    if pending_open is not None:
        st.session_state[pending_open] = True

    # Narrow columns + a trailing spacer keep these inputs from stretching
    # across the full "large"-width dialog.
    title_col, _spacer = st.columns([2, 3])
    title = title_col.text_input(
        "Network title", key="_dialog_title",
        help="Name for this custom network -- this is what will appear in "
             "the sidebar's network dropdown (as \"<title> (<N>)\") once "
             "applied, and the filename used for the saved .zip/network "
             "file.",
    )
    title_reserved = title.strip() in _RESERVED_NETWORK_NAMES
    if title_reserved:
        title_col.error(
            f"'{title.strip()}' is a built-in network name and cannot be "
            "used for a custom network -- pick a different title."
        )

    st.markdown("**Select Network to modify**")
    col1, col2, _spacer2 = st.columns([2, 2, 3])
    options = _dialog_network_options()
    # Default to "small"/amax=7 (a deliberately small, fast-to-browse
    # starting point) the first time the dialog opens this session; both are
    # seeded via setdefault *before* their widgets are created below, which is
    # the safe way to set a widget's initial value in Streamlit (setting it
    # *after* creation, e.g. from a button handler, is what _pending_open /
    # _pending_network_label exist to work around elsewhere in this module).
    st.session_state.setdefault("_dialog_base_network", "small" if "small" in options else options[0])
    st.session_state.setdefault("_dialog_amax_enabled", True)
    st.session_state.setdefault("_dialog_amax_value", 7)
    if st.session_state.get("_dialog_base_network") not in options:
        st.session_state["_dialog_base_network"] = options[0]
    base_network = col1.selectbox(
        "Network", options, key="_dialog_base_network",
        format_func=_dialog_network_label, label_visibility="collapsed",
        help="Starting point: selects which reactions below start toggled "
             "on (every reaction is still individually editable below, "
             "regardless of this choice).",
    )
    amax_col, value_col = col2.columns(2)
    amax_enabled = amax_col.checkbox(
        "Limit A", key="_dialog_amax_enabled",
        help="Restrict to reactions with A <= amax, to reduce the network size.",
    )
    dialog_amax = None
    if amax_enabled:
        # `key` already has a session_state entry by now (seeded above via
        # setdefault before this widget's first creation), so passing `value`
        # too would just trigger a Streamlit warning about the value being
        # set through two different paths for no benefit.
        dialog_amax = int(value_col.number_input(
            "amax", min_value=2, key="_dialog_amax_value", label_visibility="collapsed",
        ))

    # Reset per-reaction state if the (base_network, amax) pair changed since
    # the dialog last computed it -- mirrors the old _customise_network guard.
    sig = (base_network, dialog_amax)
    if st.session_state.get("_dialog_signature") != sig:
        _reset_dialog_reaction_state(base_network, dialog_amax)
        st.session_state["_dialog_signature"] = sig

    all_entries = _dialog_superset_entries(dialog_amax)
    bare_all = [_bare(e) for e in all_entries] + list(st.session_state["_dialog_added"])
    decay_rates = _decay_rates()
    decay_names = sorted(n for n in bare_all if n in decay_rates)
    nuclide_names = [n for n in bare_all if n not in decay_rates]
    groups = group_reactions_by_category(nuclide_names)

    for cat in sorted(groups):
        _render_category(cat, groups[cat])
    if decay_names:
        _render_decay_category(decay_names, decay_rates)

    _render_evolved_nuclides_section()

    keep_map = st.session_state["_dialog_keep"]
    total_kept = sum(1 for v in keep_map.values() if v)
    st.caption(f"**{total_kept} / {len(bare_all)} reactions kept**")

    _render_add_rate_section(dialog_amax, all_entries)
    _render_dialog_footer(params, title, base_network, dialog_amax)


@st.dialog("Import custom network", on_dismiss=_on_import_dialog_dismiss)
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
        decay_overrides = result.get("decay_overrides", {})
        custom_network = custom_rates.kept_to_custom_network(
            _cfg(), kept, result["replaced"], decay_overrides=decay_overrides)
        st.session_state.setdefault("_known_custom_networks", {})
        st.session_state["_known_custom_networks"][title] = {
            "kept": kept, "tables": dict(result["replaced"]), "custom_network": custom_network,
            "decay_overrides": dict(decay_overrides),
        }
        st.session_state["_active_custom_network"] = {
            "title": title, "kept": kept, "custom_network": custom_network,
        }
        st.session_state["_pending_network_label"] = title
        st.session_state["_show_import_dialog"] = False
        st.rerun()


def _render_custom_network_buttons(params):
    """Two buttons, stacked, in the "Nuclear reactions" group, opening the
    popups above.

    Each click explicitly clears the *other* dialog's show-flag as a first
    line of defence; ``on_dismiss=`` on both ``@st.dialog``s
    (``_on_custom_dialog_dismiss``/``_on_import_dialog_dismiss``) is the
    second, clearing the right flag when the user dismisses one via its own
    close ("x"), Esc, or click-outside, so the flag never gets stuck True --
    which would otherwise both pop the dialog back open on the next unrelated
    rerun (e.g. editing a sidebar number) and block opening the other dialog
    (Streamlit allows only one at a time).
    """
    st.session_state.setdefault("_known_custom_networks", {})
    if st.button("Import custom network", use_container_width=True,
                key="_btn_import_custom_network"):
        st.session_state["_show_import_dialog"] = True
        st.session_state["_show_custom_dialog"] = False
    if st.button("Create/modify network", use_container_width=True,
                key="_btn_create_custom_network"):
        # Pre-select whichever network is currently active in the sidebar as
        # the dialog's "Network to modify" -- set directly (not via the
        # _pending_*-style workaround) because this assignment happens
        # *before* the dialog (and so its "_dialog_base_network" selectbox)
        # is instantiated in this very script run.
        st.session_state["_dialog_base_network"] = st.session_state.get("network", "small")
        st.session_state["_show_custom_dialog"] = True
        st.session_state["_show_import_dialog"] = False

    if st.session_state.get("_show_import_dialog"):
        _import_dialog()
    elif st.session_state.get("_show_custom_dialog"):
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
                    known = st.session_state.get("_known_custom_networks", {})
                    if value in known:
                        # Synthetic entry (built/imported this session, not
                        # necessarily the currently-*active* one -- picking
                        # any previously-used custom network's name here
                        # re-activates it, e.g. to switch back to it for a
                        # comparison run): the underlying network is always
                        # "large", driven entirely by custom_network (§7.2).
                        custom_network = known[value]["custom_network"]
                        params["network"] = "large"
                        params["custom_network"] = json.dumps(custom_network, sort_keys=True)
                        st.session_state["_active_custom_network"] = {
                            "title": value, "kept": known[value]["kept"],
                            "custom_network": custom_network,
                        }
                    else:
                        active = st.session_state.get("_active_custom_network")
                        if active and value != active["title"]:
                            # Manually picking a real network abandons the
                            # active custom network for display purposes
                            # only -- _known_custom_networks (and so the
                            # ability to switch back to it) is untouched.
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
