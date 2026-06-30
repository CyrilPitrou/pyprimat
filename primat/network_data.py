# -*- coding: utf-8 -*-
"""
Unified nuclear-network construction and use.

This module is the single place where primat turns a named reaction list into
ODE equations.  The lists in ``data/nuclear/networks/{small_parthenope,large}.txt``
(plus the hardcoded ``small``) name the thermonuclear reactions to keep; an
``amax`` cutoff (any positive integer >= 1) further filters any of these by
maximum nuclide mass number, regardless of which named network it is applied to.  The weak ``n <-> p`` conversion is
not stored in those text files because the high-temperature era integrates only
``n`` and ``p`` directly; for MT/LT network solves it is prepended internally as
the first buffer entry.

The physics convention is the usual mass-action BBN form.  A reaction
``r + ... -> p + ...`` contributes a forward flux proportional to the product of
reactant abundances, a backward flux proportional to product abundances and the
detailed-balance coefficient, and each nuclide receives its stoichiometric
coefficient times ``forward - backward``.  ``compile_network`` and
``NetworkKernels`` from :mod:`primat.network_builder` convert that declarative
stoichiometry into fast RHS/Jacobian evaluators.
"""

from __future__ import annotations

import os
import re
import csv
import io
from dataclasses import dataclass
from functools import lru_cache
from math import factorial
from collections import Counter

import numpy as np
from scipy.interpolate import interp1d

from .network_builder import (
    CompiledNetwork,
    NetworkKernels,
    check_conservation,
    compile_network,
)

__all__ = [
    "CompiledNetwork",
    "NetworkDefinition",
    "NetworkKernels",
    "ORDER_SMALL",
    "ORDER_MT",
    "ORDER_LT",
    "SPECIES_MD",
    "SPECIES_SMALL",
    "UpdateNuclearRates",
    "_LinearRate",
    "_REACTIONS_LARGE",
    "_KEY12_REACTIONS",
    "check_conservation",
    "compile_network",
    "load_network",
    "load_reaction_names",
    "available_rate_tables",
    "reaction_category",
    "group_reactions_by_category",
    "AMAX_LARGE",
    "nuclide_latex",
    "phase_network",
    "reaction_stoichiometry",
    "reaction_display_name",
    "to_filename",
    "reaction_species",
    "compute_detailed_balance_coefficients",
]

# Maximum exponent passed to np.exp() when computing reverse rates from the
# detailed-balance formula  bwd = α T9^β exp(γ/T9) × fwd.  At low T9 the
# γ/T9 term grows without bound for exothermic reactions; clamping prevents
# float overflow (e^600 ≈ 10^260, already double-precision infinity at ~709).
_EXP_CAP = 600.0

# Forward-rate threshold below which the reverse rate is forced to zero.
# When fwd ≈ 0 (reaction frozen out), the detailed-balance formula would
# amplify floating-point noise into a spurious huge reverse rate.  The floor
# value is just above the smallest denormalised double (~5e-324) so the
# comparison is safe without relying on subnormal arithmetic.
_FLOOR = 1.0001e-35
_PHOTONS = {"g"}
_LEPTONS = {"Bm", "Bp"}


def _resample_rate_table(T9_src, rate_src, T9_dst):
    """Resample a rate table from its source T9 grid to the master T9 grid.

    Uses log-log cubic interpolation (rates are smooth positive functions of T9
    in log-log space; this reproduces the existing tables to a few parts in 1e5).
    Falls back to semi-log (log T9, linear rate) when any rate value is
    non-positive (e.g. error columns that may contain zeros).

    The master grid is built from cfg.rate_grid_{npts,T9_min,T9_max} in
    load_network.  This makes load_network grid-agnostic: tables generated with
    different grids (e.g. custom networks or --keep-source-grid output from
    convert_ac2024_rates.py) are all normalised to the same common grid so that
    fill_buffer's single searchsorted path remains valid.

    Args:
        T9_src  : 1-D float array, source T9 values [GK].
        rate_src: 1-D float array, rate values on T9_src (same length).
        T9_dst  : 1-D float array, master T9 grid [GK].

    Returns:
        1-D float array of rate values resampled onto T9_dst.
    """
    lx_src, lx_dst = np.log10(T9_src), np.log10(T9_dst)
    if np.all(rate_src > 0):
        f = interp1d(lx_src, np.log10(rate_src), kind="cubic",
                     bounds_error=False, fill_value="extrapolate")
        return 10.0 ** f(lx_dst)
    # Non-positive values (e.g. error column): fall back to linear interpolation
    # of rate vs log10(T9) to avoid taking log of zero.
    f = interp1d(lx_src, rate_src, kind="linear",
                 bounds_error=False, fill_value="extrapolate")
    return f(lx_dst)

# Reaction-naming syntax used throughout primat, both for the canonical
# strings this module builds/parses and for the on-disk file names under
# data/nuclear/{tables,networks} and data/csv/*.csv:
#   "spaced"  (default): "<reactants joined by '_'>__<products joined by '_'>",
#             e.g. "n_p__d_g" for n + p -> d + g.  Every character is a valid
#             Python identifier character, so e.g. cfg.p_n_p__d_g works.
#   "compact" (legacy):  "<reactants><products>" joined by the literal "TO",
#             e.g. "n_p__d_g".  Kept only for parsing backward compatibility
#             (see _tokenise) -- :func:`reaction_stoichiometry` and
#             :func:`to_filename` accept either syntax as input regardless of
#             this flag, auto-detecting which one was used.  This flag only
#             controls the syntax *generated* by :func:`_format_name` (used
#             when this module needs to build a new name string itself, e.g.
#             in :func:`to_filename`'s catalog search).
_RATE_SYNTAX_ = "spaced"

# The two literal forms of the weak n -> p conversion name, accepted
# interchangeably wherever a reaction name is parsed (e.g. by
# reaction_stoichiometry, to_filename, load_network).
_WEAK_NTOP_NAMES = ("n__p", "n__p")

# Historical MT order from PRIMAT.  MT always integrates the intersection of the
# selected network with this list, because activating the full network before
# the deuterium bottleneck opens makes the BDF problem unnecessarily stiff.
ORDER_MT = [
    "n__p", "Be7_d__a_a_p", "Be7_n__Li7_p", "Be7_n__a_a", "He3_a__Be7_g",
    "He3_d__a_p", "He3_n__t_p", "Li6_p__Be7_g", "Li7_p__a_a", "Li7_p__a_a_g",
    "d_a__Li6_g", "d_d__He3_n", "d_d__t_p", "d_p__He3_g", "n_p__d_g",
    "t_a__Li7_g", "t_d__a_n", "t_p__a_g",
]

# Stable light-nuclide orders used when embedding HT/MT/LT solutions into a
# common time-evolution table.  Larger LT networks append their extra nuclides in
# the order supplied by ``nuclides.csv``.
SPECIES_SMALL = ["n", "p", "H2", "H3", "He3", "He4", "Li7", "Be7"]
SPECIES_MD = SPECIES_SMALL + ["He6", "Li8", "Li6", "B8"]

# Special-cased LaTeX forms for the bookkeeping species that are not written as
# "<element symbol><mass number>" (the neutron and the proton, i.e. bare ``n``
# and ``p``, where the implicit mass number 1 is conventionally not shown).
_NUCLIDE_LATEX_SPECIAL = {"n": r"\mathrm{n}", "p": r"\mathrm{p}"}

# Matches every other primat nuclide name: an element symbol (one capital
# letter optionally followed by a lowercase letter, e.g. "He", "B", "Na")
# followed by its mass number (e.g. "He3", "B10", "Na23").
_NUCLIDE_NAME_RE = re.compile(r"^([A-Z][a-z]?)(\d+)$")


def nuclide_latex(name):
    """Return the LaTeX form of a primat nuclide name, e.g. for axis labels.

    primat names nuclides as ``"<element symbol><mass number>"`` (e.g.
    ``"He3"``, ``"B10"``), with the neutron and proton as the bare bookkeeping
    names ``"n"`` and ``"p"``.  This maps such a name to the standard
    isotope notation ``${}^{A}\\mathrm{Sym}$`` (e.g. ``"He3"`` ->
    ``r"${}^{3}\\mathrm{He}$"``), suitable for Matplotlib/Plotly labels and
    Streamlit tables (which both support a LaTeX subset via ``$...$``).

    Parameters
    ----------
    name : str
        A nuclide name as it appears in ``PRIMAT.abundance_names``.

    Returns
    -------
    str
        The LaTeX representation, including the surrounding ``$...$``.

    Examples
    --------
    >>> nuclide_latex("He3")
    '${}^{3}\\\\mathrm{He}$'
    >>> nuclide_latex("n")
    '$\\\\mathrm{n}$'
    """
    special = _NUCLIDE_LATEX_SPECIAL.get(name)
    if special is not None:
        return f"${special}$"
    m = _NUCLIDE_NAME_RE.match(name)
    if not m:
        # Fall back to the plain name for anything that doesn't fit the
        # "<symbol><A>" convention, rather than raising on an unknown format.
        return f"${name}$"
    symbol, mass_number = m.groups()
    return rf"${{}}^{{{mass_number}}}\mathrm{{{symbol}}}$"

# Token aliases used when parsing compact PRIMAT names such as ``d_d__He3_n``.
_ALIAS = {"d": "H2", "t": "H3", "a": "He4"}
# _TOKENS is used for greedy tokenisation. We should include all nuclides we know about.
_TOKENS = [
    "He3", "He4", "He6", "Li6", "Li7", "Li8", "Li9", "Be7", "Be8", "Be9", "Be10", "Be11", "Be12",
    "B8", "B10", "B11", "B12", "B13", "B14", "B15",
    "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16",
    "N12", "N13", "N14", "N15", "N16", "N17",
    "O13", "O14", "O15", "O16", "O17", "O18", "O19", "O20",
    "F17", "F18", "F19", "F20",
    "Ne18", "Ne19", "Ne20", "Ne21", "Ne22", "Ne23",
    "Na20", "Na21", "Na22", "Na23",
    "n", "p", "d", "t", "a", "g", "TO", "Bm", "Bp",
]

# The small network is a named PRIMAT order, not a file-backed option.  The
# first entry is the weak n<->p rate used in MT/LT buffers; the remaining twelve
# entries are the thermonuclear rate tables.
ORDER_SMALL = [
    "n__p", "n_p__d_g", "d_p__He3_g", "d_d__He3_n", "d_d__t_p", "t_p__a_g",
    "t_d__a_n", "t_a__Li7_g", "He3_n__t_p", "He3_d__a_p", "He3_a__Be7_g",
    "Be7_n__Li7_p", "Li7_p__a_a",
]
_KEY12_REACTIONS = ORDER_SMALL[1:]


def _network_dir_from_cwd() -> str:
    """Return the package's network-list directory for import-time defaults.

    ``data/`` lives inside the ``primat`` package (it is shipped as package
    data), so the path is resolved relative to this file — never the current
    working directory — and works for both editable and regular installs.
    """
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "data", "nuclear", "networks")
    )


def load_reaction_names(cfg_or_dir, network: str | None = None) -> list[str]:
    """Read a thermonuclear reaction list from ``data/nuclear/networks``.

    Parameters
    ----------
    cfg_or_dir : PRIMATConfig or str
        Either a configuration object with ``data_dir`` or a direct path to
        the ``networks`` directory.  Accepting both forms lets module constants
        be initialised without constructing a full configuration.
    network : str, optional
        Which list to read.  ``"small"`` is special and returns
        :data:`ORDER_SMALL` without touching the filesystem.  Any other value
        is interpreted as ``<network>.txt`` inside ``data/nuclear/networks``
        unless it already ends in ``.txt``.

    Returns
    -------
    list[str]
        Reaction names in file order, excluding comments and blank lines.

    Example
    -------
    >>> len(load_reaction_names("/repo/data/nuclear/networks", "small"))
    12
    """
    if hasattr(cfg_or_dir, "_resolved_data_dir"):
        if hasattr(cfg_or_dir, "resolve_rates_path"):
            # Honour the user_nuclear_dir overlay (see
            # PRIMATConfig.resolve_rates_path) so a user-supplied network
            # file does not require touching the installed package.
            nets_dir = cfg_or_dir.resolve_rates_path("nuclear", "networks")
        else:
            nets_dir = os.path.join(cfg_or_dir._resolved_data_dir, "nuclear", "networks")
        network = network or cfg_or_dir.network
    else:
        nets_dir = os.fspath(cfg_or_dir)
        network = network or "large"

    if network in (None, "small"):
        return list(_KEY12_REACTIONS)

    filename = network
    if not filename.endswith(".txt"):
        filename += ".txt"

    path = os.path.join(nets_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"network {network!r} was requested, but {path} does not exist"
        )
    with open(path) as f:
        return [
            line.split("#", 1)[0].strip()
            for line in f
            if line.split("#", 1)[0].strip()
        ]


def _bare_entry(entry: str) -> str:
    """Strip a network-file entry's optional ``", filename"``/``", rate"``
    second column, returning just the bare reaction name.

    Every line of ``small.txt``/``large.txt`` (and GUI-exported custom
    network files) is either a bare name or ``"name, filename.txt"`` (or,
    for a decay with an overridden rate, ``"name, rate"``) -- see
    ``load_network``'s pre-parse loop. Callers that only want the reaction
    name itself (e.g. the module-level catalogs below) go through this
    helper rather than re-deriving the split inline.
    """
    return re.split(r'[, ]+', entry, maxsplit=1)[0].strip()


try:
    _REACTIONS_LARGE = [_bare_entry(e)
                        for e in load_reaction_names(_network_dir_from_cwd(), "large")]
except OSError:
    # Importing documentation tooling outside the repository should still work.
    _REACTIONS_LARGE = []

ORDER_LT = ["n__p"] + _REACTIONS_LARGE


def _read_csv(path):
    """Read a small comma-separated PRIMAT data file without adding pandas."""
    with open(path) as f:
        rows = [line.rstrip("\n").split(",") for line in f if line.strip()]
    return rows[0], rows[1:]


def reaction_species(name):
    """Expand a reaction name into explicit reactant and product nuclide lists.

    The reaction *name* (e.g. ``'ddtp'``) is compact; this returns the two sides
    as flat lists that repeat a nuclide once per unit of multiplicity, which is
    the form :func:`compute_detailed_balance_coefficients` expects.

    Example
    -------
    >>> reaction_species('ddtp')          # d + d -> t + p
    (['H2', 'H2'], ['H3', 'p'])

    Returns
    -------
    (reactants, products) : tuple[list[str], list[str]]
        Nuclide keys (PRIMATConfig.Nuclides convention: 'n', 'p', 'H2', 'H3',
        'He3', 'He4', ...), with each species repeated by its multiplicity.
        The photon 'g' is dropped (it carries no nuclear mass/spin).
    """
    react, prod = reaction_stoichiometry(name)

    def expand(multiplicity_dict):
        # {'H2': 2} -> ['H2', 'H2']
        out = []
        for species, count in multiplicity_dict.items():
            out += [species] * count
        return out

    return expand(react), expand(prod)


def compute_detailed_balance_coefficients(reactants, products, cfg):
    r"""Compute the reverse-rate coefficients (alpha, beta, gamma) of a reaction.

    The network stores only the *forward* thermonuclear rate of each reaction.
    The *backward* rate is reconstructed by detailed balance as

        backward(T9) = alpha * T9**beta * exp(gamma / T9) * forward(T9),

    where ``T9`` is the temperature in units of 1e9 K.  This routine derives
    alpha, beta and gamma from nuclide data alone (masses, spins, mass excesses),
    so the network no longer depends on a hand-tabulated table.  It is a direct
    Python port of PRIMAT's ``GatherInfoReac`` (functions ``Qreaction``,
    ``PowerT9`` and ``FactorInverseReaction`` in ``PRIMAT-main.m``) and
    reproduces the published values to < 0.5%.

    Physics
    -------
    For a reaction with reactants ``i`` and products ``j``, equilibrium of the
    forward and backward Saha factors gives

        gamma = -Q / (kB * 1e9 K)         (Q = energy released; gamma < 0 when Q > 0)
        beta  = 3/2 * (n_reactants - n_products)         (thermal-wavelength powers)
        alpha = [prod_i  g_i (M_i kB / 2π)^{3/2} / m_i! ]
                / [prod_j g_j (M_j kB / 2π)^{3/2} / m_j! ] * unit_factor

    with ``g = 2J+1`` the spin degeneracy, ``M`` the nuclear mass, ``m!`` the
    symmetry factor for identical particles, and ``unit_factor`` the
    dimensional constant ``((m_u/c^2)/(ħc)^3)^{n_in-n_out}`` that makes alpha
    dimensionless.  All of (M kB / 2π)^{3/2} is evaluated at T9 = 1 K*1e9; the
    temperature dependence is carried entirely by the ``T9**beta`` term.

    Parameters
    ----------
    reactants, products : list[str]
        Nuclide names with multiplicity (see :func:`reaction_species`).
    cfg : PRIMATConfig
        Supplies the nuclide tables (``Nuclides`` = [N, Z], ``NuclExcessMass``
        in keV, ``NuclSpin``) and the fundamental constants (kB, hbar, clight,
        ma, me, keV, MeV) in the CGS-erg system used throughout primat.

    Returns
    -------
    (alpha, beta, gamma) : tuple[float, float, float]
        Coefficients of ``alpha * T9**beta * exp(gamma/T9)`` (the backward /
        forward rate ratio).

    Example
    -------
    >>> cfg = PRIMATConfig()
    >>> compute_detailed_balance_coefficients(['n', 'p'], ['H2'], cfg)   # n + p -> d (+ gamma)
    (4.71614e+09, 1.5, -25.815)
    """
    keV, kB, MeV = cfg.keV, cfg.kB, cfg.MeV
    ma_e, me_e = cfg.ma * MeV, cfg.me * MeV          # atomic mass unit, electron mass [erg]
    NZ, EX, SP = cfg.Nuclides, cfg.NuclExcessMass, cfg.NuclSpin

    def mass(s):
        # Nuclear (not atomic) rest mass energy [erg]:  A*m_u + excess - Z*m_e.
        # Subtracting the Z electron masses converts the atomic mass excess into
        # the bare nuclear mass.
        n, z = NZ[s]
        return (n + z) * ma_e + EX[s] * keV - z * me_e

    def binding(s):
        # Binding energy in keV (as a plain number, units restored by *keV):
        #   B = N*ExcessMass(n) + Z*ExcessMass(p) - ExcessMass(nuclide).
        # n and p are the reference, so B('n') = B('p') = 0 and B('H2') = 2224.57.
        n, z = NZ[s]
        return n * EX["n"] + z * EX["p"] - EX[s]

    n_in, n_out = len(reactants), len(products)

    # beta: each non-relativistic species contributes a (kB T)^{3/2} thermal
    # factor; the net power of T9 is 3/2 times the change in particle number.
    beta = 1.5 * (n_in - n_out)

    # Q-value (energy released) = (binding of products) - (binding of reactants),
    # converted from keV-number to erg.  gamma = -Q/(kB * 1e9 K) puts the
    # Boltzmann factor exp(-Q/kB T) into the T9 = T/1e9 convention.
    Q = keV * (sum(binding(s) for s in products)
               - sum(binding(s) for s in reactants))   # [erg]
    gamma = -Q / (kB * 1e9)

    def quantum_factor(side):
        # Product over distinct species of  [ (2J+1) * (M kB 1e9 / 2π)^{3/2} ]^m / m!
        # i.e. the spin-weighted quantum-concentration factor for that side,
        # with the m! symmetry factor for m identical particles.
        val = 1.0
        for s, m in Counter(side).items():
            term = (2 * SP[s] + 1) * (2 * np.pi / mass(s) / (kB * 1e9)) ** (-1.5)
            val *= term ** m / factorial(m)
        return val

    # Dimensional constant (per net particle) that renders alpha dimensionless;
    # exactly PRIMAT's FactorInverseReaction "Units".
    units = ((ma_e / cfg.clight**2) / (cfg.hbar * cfg.clight)**3) ** (n_in - n_out)
    alpha = quantum_factor(reactants) / quantum_factor(products) * units
    return alpha, beta, gamma


def _tokenise(name):
    """Split a reaction name into nuclide tokens plus a ``"TO"`` separator.

    Both syntaxes are accepted, auto-detected from the name itself, so callers
    (:func:`reaction_stoichiometry`, :func:`to_filename`) need no syntax
    branching of their own:

    * **spaced** (``"a_b__c_d"``): the reactant/product sides are split on the
      double underscore, each side's tokens on a single underscore, e.g.
      ``"n_p__d_g"`` -> ``["n", "p", "TO", "d", "g"]``.
    * **compact** (legacy, ``"abTOcd"``): tokens are concatenated with no
      separator and greedily matched against :data:`_TOKENS` (which includes
      the literal ``"TO"`` marking the reactant/product split), e.g.
      ``"n_p__d_g"`` -> ``["n", "p", "TO", "d", "g"]``.

    The returned token list is the same in both cases, with a literal
    ``"TO"`` entry standing in for whichever separator was actually used.
    """
    if "__" in name:
        react_part, _, prod_part = name.partition("__")
        out = list(react_part.split("_")) if react_part else []
        out.append("TO")
        out.extend(prod_part.split("_") if prod_part else [])
        return out
    out, i = [], 0
    while i < len(name):
        for tk in _TOKENS:
            if name[i:i + len(tk)] == tk:
                out.append(tk)
                i += len(tk)
                break
        else:
            raise ValueError(f"cannot tokenise reaction name {name!r}")
    return out


def _format_name(react_tokens, prod_tokens, syntax=None):
    """Join reactant/product tokens into a reaction name in the given syntax.

    Parameters
    ----------
    react_tokens, prod_tokens : sequence[str]
        Nuclide tokens (e.g. ``["n", "p"]``, ``["d", "g"]``), in PRIMAT's
        single-letter alias convention (``d``/``t``/``a`` for H2/H3/He4).
    syntax : {"spaced", "compact"}, optional
        Defaults to the module-level :data:`_RATE_SYNTAX_`.

    Example
    -------
    >>> _format_name(["n", "p"], ["d", "g"])
    'n_p__d_g'
    >>> _format_name(["n", "p"], ["d", "g"], syntax="compact")
    'n_p__d_g'
    """
    syntax = syntax or _RATE_SYNTAX_
    if syntax == "spaced":
        return "_".join(react_tokens) + "__" + "_".join(prod_tokens)
    if syntax == "compact":
        return "".join(react_tokens) + "TO" + "".join(prod_tokens)
    raise ValueError(f"_RATE_SYNTAX_ must be 'compact' or 'spaced', got {syntax!r}")


def reaction_display_name(name):
    """Human-readable ``"react1 + react2 > prod1 + prod2"`` form of ``name``.

    Unlike :func:`reaction_stoichiometry` (which maps PRIMAT's single-letter
    aliases ``d``/``t``/``a`` to their canonical species ``H2``/``H3``/``He4``
    for the ODE machinery), this keeps the *literal* tokens from ``name``
    itself -- matching the shipped tables' own header convention (see
    :func:`_reaction_source_from_lines`'s docstring, e.g.
    ``"# n + p > d + g   [n_p__d_g]   ref=And06"``) so a GUI-uploaded table's
    provenance line reads the same way.

    Parameters
    ----------
    name : str
        Reaction name in either the ``"a_b__c_d"`` or compact ``"abTOcd"``
        syntax (see :func:`_tokenise`).

    Returns
    -------
    str

    Example
    -------
    >>> reaction_display_name("He3_n__a_g")
    'He3 + n > a + g'
    """
    tokens = _tokenise(name)
    split = tokens.index("TO")
    react, prod = tokens[:split], tokens[split + 1:]
    return f"{' + '.join(react)} > {' + '.join(prod)}"


def reaction_stoichiometry(name):
    """Return ``(reactants, products)`` as nuclide-multiplicity dictionaries.

    The compact PRIMAT names concatenate nuclide tokens.  For example
    ``d_d__He3_n`` means ``d + d -> He3 + n`` and is returned as
    ``({"H2": 2}, {"He3": 1, "n": 1})``.

    Two ways of locating the reactant/product split are supported:

    * **Catalog path** (``name`` present in ``detailed_balance.csv``): the
      split is fixed by the detailed-balance exponent beta, following the
      PRIMAT convention ``beta = 1.5 * (n_reactants - n_products)``.
    * **Fallback path** (``name`` not in the catalog, e.g. a new reaction
      added directly to a network file): the literal ``"TO"`` token in the
      name marks the split directly, so no beta lookup is needed.  The
      derived stoichiometry is validated for baryon-number and charge
      conservation (mirroring
      ``generate_rates/nuclide_table.conservation_residual``); a mismatch
      raises ``ValueError`` naming the reaction and the imbalance, instead of
      a cryptic ``KeyError``.

    Example
    -------
    >>> reaction_stoichiometry("n_p__d_g")
    ({'n': 1, 'p': 1}, {'H2': 1})
    """
    if name in _WEAK_NTOP_NAMES:
        # The physical reaction is n → p + e⁻(Bm), but the ODE state vector
        # does not track the emitted electron.  The lepton charge bookkeeping
        # lives in NetworkDefinition.lepton_dZ (= -1 for n__p), so the caller
        # of check_conservation can verify dZ = 0 uniformly.  We return only the
        # nuclear species here so that phase_network / compile_network see the
        # correct ODE stoichiometry.
        return {"n": 1}, {"p": 1}

    # We use the detailed_balance.csv to determine the split between reactants
    # and products, avoiding the need for the hardcoded tokens and aliases for
    # most reactions.
    _, _, _, nuc_NZ, db, _ = _reaction_catalog(_default_data_dir())

    def count(seq):
        counts = {}
        for tok in seq:
            key = _ALIAS.get(tok, tok)
            counts[key] = counts.get(key, 0) + 1
        return counts

    if name not in db:
        # Fallback: the "TO" token already marks the reactant/product split
        # in the name itself, so tokenise and split there directly -- no beta
        # lookup needed (that's only available via the catalog path above).
        tokens = _tokenise(name)
        if "TO" not in tokens:
            raise ValueError(
                f"reaction {name!r} is not in detailed_balance.csv and has "
                f"no 'TO' separator, so its stoichiometry cannot be derived")
        split = tokens.index("TO")
        react = count(t for t in tokens[:split] if t != "g")
        prod  = count(t for t in tokens[split + 1:] if t != "g")

        # Validate A (=N+Z) and Z conservation across the reaction, using
        # nuclides.csv for nuclear species and _LEPTON_Z for the Bm/Bp
        # bookkeeping tokens (A=0, Z=∓1).
        def totals(counts):
            A = Z = 0
            for tok, mult in counts.items():
                if tok in _LEPTON_Z:
                    Z += _LEPTON_Z[tok] * mult
                    continue
                n, z = nuc_NZ[tok]
                A += (n + z) * mult
                Z += z * mult
            return A, Z

        Ar, Zr = totals(react)
        Ap, Zp = totals(prod)
        if (Ar, Zr) != (Ap, Zp):
            raise ValueError(
                f"reaction {name!r}: auto-derived stoichiometry does not "
                f"conserve baryon number / charge "
                f"(reactants: A={Ar}, Z={Zr}; products: A={Ap}, Z={Zp})")
        return react, prod

    beta = db[name][1]
    tokens = [t for t in _tokenise(name) if t not in {"g", "TO"}]
    n_react = int(round((len(tokens) + beta / 1.5) / 2))

    return count(tokens[:n_react]), count(tokens[n_react:])


def to_filename(name):
    """Map a bare reaction name to its fully-separated catalog/file name.

    Some historical code (and tests) used names without any explicit
    reactant/product separator, e.g. ``"npdg"``.  This helper finds the
    matching :data:`_RATE_SYNTAX_`-appropriate separated form by trying every
    reactant/product split point against ``detailed_balance.csv``.  Names that
    already carry either separator (``"__"`` or the literal ``"TO"``) are
    returned unchanged.

    Example
    -------
    >>> to_filename("npdg")
    'n_p__d_g'
    """
    if name in _WEAK_NTOP_NAMES:
        return "n__p" if _RATE_SYNTAX_ == "spaced" else "n__p"
    if "__" in name or "TO" in name:
        return name

    _, _, _, _, db, _ = _reaction_catalog(_default_data_dir())

    if name in db: return name

    tokens = _tokenise(name)
    n_nuclide = len([t for t in tokens if t != "g"])

    # Try to find the reaction in db by splitting tokens at different points,
    # checking both syntaxes since the catalog may have been generated under
    # either one.
    for i in range(1, n_nuclide):
        for syntax in ("spaced", "compact"):
            candidate = _format_name(tokens[:i], tokens[i:], syntax=syntax)
            if candidate in db:
                return candidate

    raise ValueError(f"cannot map reaction name {name!r} to a separated filename")


def phase_network(order, species):
    """Resolve reaction names into an index-based stoichiometric network.

    Parameters
    ----------
    order : sequence[str]
        Rate-buffer order.  Reaction ``i`` uses ``r[2*i]`` for the forward rate
        and ``r[2*i+1]`` for the backward rate.
    species : sequence[str]
        Abundance-vector species order.

    Returns
    -------
    list
        ``[(reactants, products), ...]`` where each side is
        ``{species_index: multiplicity}``.

    Example
    -------
    >>> phase_network(["n__p"], ["n", "p"])
    [({0: 1}, {1: 1})]
    """
    idx = {s: i for i, s in enumerate(species)}
    net = []
    for name in order:
        react, prod = reaction_stoichiometry(name)
        # Bm/Bp (emitted electron/positron) are lepton bookkeeping tokens, not
        # part of the ODE state vector (see reaction_stoichiometry's "n__p"
        # special case docstring) -- decay reactions such as "Be7__Li7_Bp" carry
        # one in their stoichiometry dict, so it must be dropped here too,
        # exactly as the production path (_side_counts, used by load_network)
        # already does.
        net.append(({idx[s]: c for s, c in react.items() if s not in _LEPTONS},
                    {idx[s]: c for s, c in prod.items() if s not in _LEPTONS}))
    return net


class _LinearRate:
    """Fast equivalent of ``interp1d(kind='linear', fill_value='extrapolate')``.

    Appends one synthetic knot on each side of the grid whose value is obtained
    by projecting the edge slope, then delegates to ``np.interp`` (a fast C
    loop).  Any query point within ``[xlo, xhi]`` — a range twice as wide as
    the original grid — gets exact linear extrapolation; beyond that the value
    is clamped (which never occurs in practice for BBN T9 grids).

    Parameters
    ----------
    x : 1-D array
        Grid knots (e.g. T9 values), strictly increasing.
    y : 1-D array, same length as *x*
        Tabulated values at the knots.

    Example
    -------
    >>> f = _LinearRate(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]))
    >>> f(1.5)
    15.0
    """

    __slots__ = ("xp", "fp")

    def __init__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        # Extend the grid by one span in each direction, projecting the edge
        # slopes so that np.interp (which clamps at the boundary) gives true
        # linear extrapolation up to one full data-range width beyond the grid.
        span = x[-1] - x[0]
        xlo, xhi = x[0] - span, x[-1] + span
        ylo = y[0]  + (y[1]  - y[0])  / (x[1]  - x[0])  * (xlo - x[0])
        yhi = y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (xhi - x[-1])
        self.xp = np.concatenate(([xlo], x, [xhi]))
        self.fp = np.concatenate(([ylo], y, [yhi]))

    def __call__(self, T):
        """Evaluate at *T* (scalar or array)."""
        return np.interp(T, self.xp, self.fp)


@dataclass
class NetworkDefinition:
    """A fully assembled reaction network for one solver era.

    ``names`` includes the prepended weak ``n__p`` entry when present, while the
    text network files name only thermonuclear reactions.  ``network`` is the
    corresponding index-based stoichiometry used by :func:`compile_network`.

    Example
    -------
    >>> cfg = PRIMATConfig({"network": "large", "amax": 8})
    >>> net = load_network(cfg, era="MT")
    >>> net.names[0], len(net.species)
    ('n__p', 12)
    """

    species: list[str]
    N: np.ndarray
    Z: np.ndarray
    network: list[tuple[dict[int, int], dict[int, int]]]
    weak_indices: set[int]
    names: list[str]
    grid: np.ndarray
    _fwd: np.ndarray
    _fwd_median: np.ndarray
    _expsigma: np.ndarray
    _abg: np.ndarray
    _bwd_cap: np.ndarray
    lepton_dZ: list[int]   # net lepton charge per reaction: e.g. -1 for n__p (Bm emitted)
    # Per-reaction provenance string (aligned with ``names``), read from the
    # first ``#`` header line of each rate table at load time (e.g. "And06" for
    # n_p__d_g).  ``None`` when the network was built without source bookkeeping
    # (e.g. directly from ``reaction_names`` in a test).
    sources: list[str] | None = None
    # Per-reaction rate-table path (aligned with ``names``): the on-disk
    # ``data/nuclear/tables/<name>.txt`` each forward rate was loaded from.
    # ``None`` for entries with no rate table (``n__p``, whose weak rates are
    # supplied at solve time) and ``None`` for the whole list when the network
    # was built without source bookkeeping. Used by the GUI reactions table to
    # offer a download button per reaction.
    files: list[str] | None = None

    def __post_init__(self):
        self.index = {s: i for i, s in enumerate(self.species)}
        self.n_reac = len(self.network)
        self._buf = np.empty(2 * self.n_reac)
        # One-slot memo for fill_buffer: BDF's implicit solver evaluates the
        # RHS and the (separately supplied, analytic) Jacobian at the same
        # fixed T_t across one or more Newton corrector iterations before
        # advancing t (see rhsLT/JacobianLT, rhsMT/JacobianMT below), so
        # consecutive fill_buffer calls very often share T_t exactly. Caching
        # the last (T_t, clamp) pair skips the n<->p weak-rate interpolant
        # evaluation and the rate-table relookup, which profiling
        # (studies/profile_solve.py) showed dominates LT-era solve time.
        self._cache_T_t = None
        self._cache_clamp = None

    def reaction_equation(self, i):
        """Human-readable equation for reaction ``i`` as ``a + b <-> c + d``.

        Built from the index-based stoichiometry ``self.network[i]`` and the
        ``self.species`` name list, so a reaction such as ``d_d__He3_n`` (stored
        as ``({H2: 2}, {He3: 1, n: 1})``) is rendered ``H2 + H2 <-> He3 + n``.
        Stoichiometric multiplicities are expanded into repeated tokens to
        mirror the compact PRIMAT reaction name.  The weak entry ``n__p`` is
        rendered ``n <-> p``.

        Parameters
        ----------
        i : int
            Reaction index into ``self.names`` / ``self.network``.

        Returns
        -------
        str
            The ``reactants <-> products`` equation.
        """
        def side(counts):
            # Expand {species_index: multiplicity} into a flat "a + a + b" list.
            toks = []
            for idx, mult in counts.items():
                toks.extend([self.species[idx]] * mult)
            return " + ".join(toks)

        react, prod = self.network[i]
        return f"{side(react)} <-> {side(prod)}"

    def describe_reactions(self):
        """List every reaction as ``(name, equation, source, file)`` tuples.

        Combines :meth:`reaction_equation` with the per-reaction provenance in
        :attr:`sources` (the ``ref=`` field of each rate table's header line)
        and the rate-table path in :attr:`files`.  Used by the verbose console
        output (see :meth:`UpdateNuclearRates.__init__`) and by the GUI
        reactions table to show, e.g.,
        ``('n_p__d_g', 'n + p <-> H2', 'And06', '.../tables/n_p__d_g.txt')``.

        Returns
        -------
        list[tuple[str, str, str, str | None]]
            One tuple per reaction, in solver/buffer order (``n__p`` first).
            The last element is the rate-table path, or ``None`` for entries
            with no table (e.g. the weak ``n__p`` conversion).
        """
        src = self.sources if self.sources is not None else [""] * len(self.names)
        files = self.files if self.files is not None else [None] * len(self.names)
        return [
            (name, self.reaction_equation(i), src[i], files[i])
            for i, name in enumerate(self.names)
        ]

    def apply_variations(self, cfg):
        """Update the active forward rate tables ``self._fwd`` by applying any
        variation parameters (p_* and delta_*) from the configuration.

        This allows Monte Carlo loops to reuse the same network objects while
        refreshing the rates at the start of each solve.
        """
        NP = cfg.rescale_nuclear_rates
        # Skip names[0] which is always n__p (handled separately in the solver)
        for i, name in enumerate(self.names[1:]):
            p = getattr(cfg, f"p_{name}")
            delta = getattr(cfg, f"delta_{name}")

            if p == 0.0 and (not NP or delta == 0.0):
                # No variation: revert to median
                self._fwd[i] = self._fwd_median[i]
            else:
                # Apply p uncertainty: median * exp(p * log(expsigma))
                variation = np.exp(p * np.log(self._expsigma[i]))
                if NP:
                    variation += delta
                self._fwd[i] = self._fwd_median[i] * variation

    def fill_buffer(self, T_t, nTOp_frwrd, nTOp_bkwrd, clamp=True):
        """Fill the forward/backward rate buffer at photon temperature ``T_t``.

        ``T_t`` is in kelvin.  Forward rates are linearly interpolated in the
        active rate tables.  Backward rates are obtained from detailed balance.
        Clamping is applied if ``clamp=True`` to prevent low-temperature
        reverse-rate blow-up (standard for the large network's LT era).

        Returns the same internal buffer object on a cache hit (``T_t`` and
        ``clamp`` bit-identical to the previous call -- see ``__post_init__``),
        so this is exact, not approximate: callers must treat the returned
        array as read-only and consume it before the next call, exactly as
        they already do (``rhsLT``/``JacobianLT`` etc. use it immediately).
        """
        if T_t == self._cache_T_t and clamp == self._cache_clamp:
            return self._buf

        r = self._buf
        r[0] = nTOp_frwrd(T_t)
        r[1] = nTOp_bkwrd(T_t)

        T9 = T_t * 1e-9
        g = self.grid
        i = int(np.searchsorted(g, T9) - 1)
        if i < 0:
            i = 0
        elif i > g.size - 2:
            i = g.size - 2
        w = (T9 - g[i]) / (g[i + 1] - g[i])

        # Slice copies are intentional: the backward-rate clipping below must
        # not mutate the cached table columns shared by later evaluations.
        fwd = self._fwd[:, i] * (1.0 - w) + self._fwd[:, i + 1] * w
        alpha, beta, gamma = self._abg[:, 0], self._abg[:, 1], self._abg[:, 2]
        bwd = alpha * T9 ** beta * np.exp(np.minimum(gamma / T9, _EXP_CAP)) * fwd
        bwd[fwd <= _FLOOR] = 0.0
        # A reverse rate is physically non-negative.  At low T9 the resampled
        # forward table can carry tiny negative interpolation/extrapolation
        # noise (and the detailed-balance prefactor amplifies it), which would
        # turn the reverse flux into a spurious source/sink.  Floor at 0.
        np.maximum(bwd, 0.0, out=bwd)

        if clamp:
            np.minimum(bwd, self._bwd_cap, out=bwd)

        r[2::2] = fwd
        r[3::2] = bwd
        self._cache_T_t = T_t
        self._cache_clamp = clamp
        return r


def _default_data_dir() -> str:
    """Package-shipped data root (``primat/data/``, containing nuclear/, csv/, etc.).

    Defined here so :func:`reaction_stoichiometry` and :func:`to_filename` can
    reach :func:`_reaction_catalog` without constructing a throwaway
    ``PRIMATConfig`` (which re-reads ``nuclides.csv`` and would create a
    config<->nuclear circular import).  Equivalent to
    ``PRIMATConfig()._pkg_data_dir``.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@lru_cache(maxsize=None)
def _reaction_catalog(data_dir: str):
    """Load nuclide metadata, reaction stoichiometry and detailed balance tables.

    Parameters
    ----------
    data_dir : str
        Data root directory (the ``primat/data/`` directory or a user-supplied
        replacement via ``cfg.data_dir``).  Contains ``csv/``, ``nuclear/``,
        ``plasma/``, ``NEVO/``, ``weak/`` subdirectories.  This is a fixed path
        for a given primat installation, so the result is cached with
        :func:`functools.lru_cache`: the three CSV files under ``csv/`` are
        read at most once per process instead of on every :func:`load_network`,
        :func:`reaction_stoichiometry` or :func:`to_filename` call.
        Equivalent to ``cfg._resolved_data_dir``.
    """
    base = os.path.join(data_dir, "csv")
    tables_dir = os.path.join(data_dir, "nuclear", "tables")
    _, nuc_rows = _read_csv(os.path.join(base, "nuclides.csv"))
    nuc_order = [row[0] for row in nuc_rows]
    nuc_NZ = {row[0]: (int(row[1]), int(row[2])) for row in nuc_rows}

    _, db_rows = _read_csv(os.path.join(base, "detailed_balance.csv"))
    db = {row[0]: (float(row[2]), float(row[3]), float(row[4])) for row in db_rows}

    # Prefer the generated data copy.  The repository-root mirror is accepted so
    # users can inspect or regenerate the catalog without changing loader code.
    rxn_path = os.path.join(base, "reactions_large.csv")
    if not os.path.exists(rxn_path):
        rxn_path = os.path.join(data_dir, "nuclear", "reactions_large.csv")
    _, rxn_rows = _read_csv(rxn_path)
    rxn_map = {row[0]: (row[1], row[2]) for row in rxn_rows}
    return tables_dir, base, nuc_order, nuc_NZ, db, rxn_map


# Electric charge of each lepton bookkeeping token (A=0, not in ODE state vector).
# Bm = β⁻ = electron (Z = -1);  Bp = β⁺ = positron (Z = +1).
_LEPTON_Z = {"Bm": -1, "Bp": +1}


def _side_counts(field):
    """Parse one CSV side such as ``Li7+p`` into nuclide multiplicities.

    Returns:
        counts    : {nuclide_name: multiplicity} (photons and leptons excluded).
        lepton_dZ : net electric charge carried by leptons on this side.
                    +1 per Bm on the product side contributes dZ = -1 (electron
                    emitted); used by ``check_conservation`` to verify dZ = 0
                    uniformly across all reactions, including weak ones.
    """
    counts, lepton_dZ = {}, 0
    for tok in field.split("+"):
        if tok in _LEPTON_Z:
            lepton_dZ += _LEPTON_Z[tok]  # accumulate lepton charge on this side
            continue
        if tok in _PHOTONS:
            continue
        counts[tok] = counts.get(tok, 0) + 1
    return counts, lepton_dZ


def _species_order(nuclides, nuc_order):
    """Order active nuclides with light species first, then CSV order."""
    base = SPECIES_MD if any(s in nuclides for s in SPECIES_MD[8:]) else SPECIES_SMALL
    ordered = [s for s in base if s in nuclides]
    ordered.extend(s for s in nuc_order if s in nuclides and s not in ordered)
    return ordered


def _qed_nuclear_rescale(name, T9_grid):
    """QED correction factor for radiative-capture reactions (Pitrou & Pospelov 2020).

    When ``cfg.nuclear_qed_corrections=True`` this factor is multiplied into the
    forward rate tables at load time.  The five affected reactions all have a real
    photon in the final state; QED corrections arise from pair-production processes
    (γ → e⁺e⁻) that open up when the available energy exceeds 2mₑ.

    For ``n_p__d_g`` the correction is taken from a polynomial fit to a dedicated
    Gamow-peak integration (Pitrou & Pospelov 2020); the fit is capped at its
    T9 → 0 limit 1.0009003934476768.

    For the four remaining reactions the correction is the Kroll electric-dipole
    formula (Kroll & Watson 1954 as applied in Pitrou & Pospelov 2020):

        f_Kroll(Eₐ) = 1 − 10α/(9π) + 2α ln(64)/(9π) − α ln(4/Eₐ²)/(3π)
                      + 3α(1 + ln(4Eₐ²))/(4π Eₐ⁴)

    where Eₐ = (E_ML + Q)/mₑ, E_ML = 0.1220 Z₁^{2/3} Z₂^{2/3} (A₁A₂/(A₁+A₂))^{1/3} T9^{2/3} MeV
    is the most-likely kinetic energy at the Gamow peak, and Q is the reaction Q-value.

    Parameters
    ----------
    name : str
        Reaction identifier (e.g. ``"n_p__d_g"``).
    T9_grid : np.ndarray
        Master temperature grid in GK.

    Returns
    -------
    np.ndarray or None
        Correction factors, shape ``(len(T9_grid),)``, or ``None`` if the
        reaction requires no QED correction.

    Example
    -------
    >>> import numpy as np
    >>> T9 = np.array([0.01, 0.1, 1.0])
    >>> f = _qed_nuclear_rescale("n_p__d_g", T9)
    >>> (f > 1.0).all()
    True
    """
    # Fine structure constant (CODATA 2018)
    ALPHA = 1.0 / 137.035999084
    # Electron mass [MeV] (PDG)
    ME_MEV = 0.51099895

    if name in ("n_p__d_g", "n_p__d_g"):
        # Polynomial fit to the QED correction for n + p → d + γ, derived from
        # the Gamow-peak integration with the electric-dipole model in
        # Pitrou & Pospelov 2020.  The polynomial slightly exceeds its T9=0
        # limit at intermediate temperatures, so we cap at that limit.
        poly = (1.0003328617393168
                + 0.00010013475534938917 * T9_grid
                + 0.00004089993260910648 * T9_grid**2
                - 0.000011824673537229535 * T9_grid**3
                + 1.0522377796855455e-6  * T9_grid**4)
        # Cap at the T9→0 limit (pair-threshold value)
        T9_ZERO_LIMIT = 1.0009003934476768
        return np.minimum(poly, T9_ZERO_LIMIT)

    # Reactant (Z,A) pairs and Q-values [MeV] for the four electric-dipole reactions.
    # Q = mass_excess(reactant1) + mass_excess(reactant2) - mass_excess(product)  [keV→MeV]
    # Mass excesses from NUBASE2020 (same source as nuclides.csv in this repo):
    #   n:8071.318, p:7288.971, H2:13135.723, H3:14949.811,
    #   He3:14931.219, He4:2424.916, Li7:14907.105, Be7:15769.000  [all keV]
    _ED_PARAMS = {
        #           Z1  A1  Z2  A2  Q [MeV]
        "d_p__He3_g":  (1, 2,  1, 1,  (13135.723 + 7288.971 - 14931.219) * 1e-3),
        "t_p__a_g":    (1, 3,  1, 1,  (14949.811 + 7288.971 -  2424.916) * 1e-3),
        "t_a__Li7_g":  (1, 3,  2, 4,  (14949.811 + 2424.916 - 14907.105) * 1e-3),
        "He3_a__Be7_g":(2, 3,  2, 4,  (14931.219 + 2424.916 - 15769.000) * 1e-3),
    }

    if name not in _ED_PARAMS:
        return None

    Z1, A1, Z2, A2, Q = _ED_PARAMS[name]

    # Most-likely kinetic energy at the Gamow peak [MeV]
    # (Landau & Lifshitz via Pitrou & Pospelov 2020)
    EML = 0.1220 * Z1**(2/3) * Z2**(2/3) * ((A1 * A2) / (A1 + A2))**(1/3) * T9_grid**(2/3)

    # Available energy in units of mₑ
    Ea = (EML + Q) / ME_MEV

    # Kroll factor (Pitrou & Pospelov 2020, using PRIMAT-Main.m RescaleElectricDipoleKroll):
    #   f = 1 - 10α/(9π) + 2α ln64/(9π) - α ln(4/Ea²)/(3π)
    #         + 3α(1 + ln(4Ea²))/(4π Ea⁴)
    # The last term diverges as Ea→0 but Ea ≥ Q/mₑ ≥ 3 for all reactions here.
    pi = np.pi
    f = (1.0
         - (10 * ALPHA) / (9 * pi)
         + (2 * ALPHA * np.log(64)) / (9 * pi)
         - (ALPHA * np.log(4.0 / Ea**2)) / (3 * pi)
         + (3 * ALPHA * (1.0 + np.log(4.0 * Ea**2))) / (4 * pi * Ea**4))
    return f


def _reaction_source_from_lines(lines):
    """Extract the data source label from a rate table's first ``#`` line.

    Every rate table under ``data/nuclear/tables/`` starts with a header such
    as ``# n + p > d + g   [n_p__d_g]   ref=And06``.  The ``ref=`` field names the
    experimental/theoretical compilation the rate was taken from (here the
    ``And06`` = Ando et al. 2006 evaluation).  Shared by :func:`_read_reaction_source`
    (an on-disk shipped table) and the GUI's "replaced"/"added" custom tables
    (raw uploaded text, see ``UpdateNuclearRates.__init__``), so a customised
    reaction's provenance is read the same way whenever its raw text actually
    carries one -- e.g. picking an existing alternate shipped table from the
    dropdown, which copies that table's own ``ref=`` header verbatim.

    Parameters
    ----------
    lines : iterable of str
        Lines of the table text, in order.

    Returns
    -------
    str or None
        The text after ``ref=`` on the first ``#`` line (e.g. ``"And06"``).  If
        the header has no ``ref=`` field, the whole comment line (minus the
        leading ``#``) is returned; ``None`` if there is no leading ``#`` line
        at all (e.g. a bare upload with no header).
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "ref=" in line:
                return line.split("ref=", 1)[1].strip()
            # No explicit ref= field: fall back to the bare comment.
            return line.lstrip("#").strip()
        break  # first non-comment line reached without a header
    return None


def _read_reaction_source(table_path):
    """Extract the data source label from an on-disk rate table's header.

    Thin wrapper around :func:`_reaction_source_from_lines` for a shipped
    ``data/nuclear/tables/<name>/<file>.txt`` file. See that function for the
    header format.

    Parameters
    ----------
    table_path : str
        Path to the rate table ``.txt`` file.

    Returns
    -------
    str
        The provenance label, or ``"?"`` if the file cannot be read or has no
        header at all.
    """
    try:
        with open(table_path) as fh:
            source = _reaction_source_from_lines(fh)
    except OSError:
        return "?"
    return source if source is not None else "?"


def _load_decay_table(tables_dir):
    """Parse ``data/nuclear/tables/decays.txt`` into a per-reaction dict.

    Radioactive-decay reactions (Bm/Bp on the products side) have a rate that
    is, by construction, independent of temperature: ``rate_s^-1 =
    ln(2)/halflife_s``.  Rather than a 500-row table repeating the same number
    (as the other analytic reactions get), every decay reaction has a single
    row in ``decays.txt`` with columns ``name  halflife_s  rate_s^-1
    uncertainty  ref`` (see :func:`generate_rates.convert_ac2024_rates.write_decay_file`).
    This is parsed once here; :func:`load_network` then broadcasts
    ``rate_s^-1`` to a constant array on the master T9 grid for each decay
    reaction it selects.

    Parameters
    ----------
    tables_dir : str
        Path to ``data/nuclear/tables`` (where ``decays.txt`` lives).

    Returns
    -------
    dict[str, tuple[float, float, float, str]]
        Maps reaction name -> ``(rate_s, f, halflife_s, ref)``, where
        ``rate_s`` [s⁻¹] is the constant forward rate, ``f`` is the
        multiplicative 1-sigma uncertainty factor, ``halflife_s`` [s] is the
        half-life (kept for provenance/display), and ``ref`` is the
        nuclear-data source label (``"-"`` if absent).
    """
    table = {}
    path = os.path.join(tables_dir, "decays.txt")
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            name, halflife_s, rate_s, f = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            ref = parts[4] if len(parts) > 4 else "?"
            table[name] = (rate_s, f, halflife_s, ref)
    return table


def available_rate_tables(name: str, cfg) -> list[str]:
    """Basenames of every rate-table file inside ``tables/<name>/``.

    Backs the GUI popup's per-reaction "rate table" dropdown
    (CUSTOMPOPUP.md §2.4/§6.4): a reaction with more than one candidate table
    (e.g. the PRIMAT-default ``n_p__d_g_primat.txt`` plus a
    ``n_p__d_g_parthenope3.0.txt`` alternate) lets the user pick which one to
    use. The PRIMAT-default file (``f"{name}_primat.txt"``) is always sorted
    first when present; the rest follow in alphabetical order.

    Parameters
    ----------
    name : str
        Bare reaction name (the folder under ``tables/`` to list).
    cfg : PRIMATConfig
        Used only for ``cfg._resolved_data_dir`` (via ``resolve_rates_path``).

    Returns
    -------
    list[str]
        ``[]`` if ``tables/<name>/`` does not exist yet -- the case for a
        brand-new GUI-added reaction that only has a user-uploaded table,
        never written to disk.
    """
    if hasattr(cfg, "resolve_rates_path"):
        reaction_dir = cfg.resolve_rates_path("nuclear", "tables", name)
    else:
        reaction_dir = os.path.join(cfg._resolved_data_dir, "nuclear", "tables", name)
    if not os.path.isdir(reaction_dir):
        return []
    files = sorted(f for f in os.listdir(reaction_dir) if f.endswith(".txt"))
    default = f"{name}_primat.txt"
    if default in files:
        files.remove(default)
        files.insert(0, default)
    return files


def _parse_network_entries(reaction_names, network_label):
    """Split raw network-file entries into bare names + alternate-table map.

    Each entry is either a bare reaction name (``"n_p__d_g"``) or a
    ``"bare_name, filename.txt"`` pair pointing at a non-default rate table.
    Literal duplicate entries (e.g. the same line twice in a network file)
    are rejected rather than silently dropped, since a duplicate is far more
    likely to be a copy-paste mistake than an intentional no-op.

    Args:
        reaction_names: sequence[str], the raw entries (one per network-file
            line, or the hardcoded ORDER_SMALL/ORDER_MT list).
        network_label: str, used only in the duplicate-entry error message.

    Returns:
        (bare_names, bare_to_file): bare_names is reaction_names with any
        ", filename.txt" suffix stripped; bare_to_file maps each bare name to
        its rate-table filename, defaulting to "<bare>_primat.txt" (the
        PRIMAT-default table written by convert_ac2024_rates.py) when no
        filename was given.
    """
    seen_entries = set()
    for entry in reaction_names:
        if entry in seen_entries:
            raise ValueError(
                f"reaction entry {entry!r} is already present in network "
                f"{network_label!r} (duplicate line in the network file)")
        seen_entries.add(entry)

    bare_to_file = {}
    bare_names = []
    for entry in reaction_names:
        parts = re.split(r'[, ]+', entry, maxsplit=1)
        if len(parts) > 1:
            bare, fname = parts[0].strip(), parts[1].strip()
        else:
            bare = entry.strip()
            fname = bare + "_primat.txt"
        bare_names.append(bare)
        bare_to_file[bare] = fname
    return bare_names, bare_to_file


def _inject_custom_reactions(bare_names, custom_tables, rxn_map, db, cfg):
    """Add brand-new reactions (GUI "Customise Reactions" panel) to the catalog.

    A brand-new reaction is any selected bare name *absent* from the shipped
    catalog (``reactions_large.csv``) but whose forward rate is supplied
    through ``custom_tables``.  Its stoichiometry is derived from the name
    itself (the "a_b__c_d" syntax, via :func:`reaction_stoichiometry`) and its
    reverse-rate (alpha, beta, gamma) coefficients from nuclide data
    (:func:`compute_detailed_balance_coefficients`), then injected into the
    catalog so the rest of :func:`load_network` treats the new reaction
    exactly like a shipped one.

    Args:
        bare_names: list[str], bare reaction names selected for this network.
        custom_tables: dict[str, tuple], see load_network's docstring.
        rxn_map, db: dicts from the lru_cached :func:`_reaction_catalog`.
        cfg: PRIMATConfig instance (passed to the detailed-balance helper).

    Returns:
        (rxn_map, db): the same dicts, copied (not mutated in place) and
        extended with the new reactions, to avoid poisoning the cache shared
        with other networks/calls. Returned unchanged (same objects) when
        there is nothing to inject.
    """
    new_names = [n for n in bare_names if n not in rxn_map and n in custom_tables]
    if not new_names:
        return rxn_map, db

    rxn_map = dict(rxn_map)
    db = dict(db)
    for n in new_names:
        try:
            react_counts, prod_counts = reaction_stoichiometry(n)
        except (ValueError, KeyError) as exc:
            # Unparseable name, unknown nuclide token, or a stoichiometry
            # that does not conserve baryon number/charge: surface a clear
            # message naming the offending reaction.
            raise ValueError(f"cannot add reaction {n!r}: {exc}") from exc
        # Re-serialise to the CSV "A+B" field form used by reactions_large.csv
        # so that _side_counts (applied below to every reaction) handles
        # lepton bookkeeping and multiplicities uniformly.
        rfield = "+".join(s for s, c in react_counts.items() for _ in range(int(c)))
        pfield = "+".join(s for s, c in prod_counts.items() for _ in range(int(c)))
        rxn_map[n] = (rfield, pfield)
        # Give the new reaction a physical reverse rate via detailed balance
        # when it is purely nuclear (no emitted lepton); weak additions are
        # left reverse-rate-free (abg = 0), mirroring the decay default.
        rcounts, dZr = _side_counts(rfield)
        pcounts, dZp = _side_counts(pfield)
        if dZp - dZr == 0:
            reactants = [s for s, c in rcounts.items() for _ in range(int(c))]
            products = [s for s, c in pcounts.items() for _ in range(int(c))]
            try:
                db[n] = compute_detailed_balance_coefficients(reactants, products, cfg)
            except Exception:
                # Missing spin/mass data for one of the nuclides: fall back
                # to a forward-only reaction rather than failing the run.
                pass
    return rxn_map, db


def _apply_amax_filter(bare_names, rxn_map, nuc_NZ, amax):
    """Drop reactions involving a nuclide with mass number A > amax.

    Applies to *any* network (not just "large" -- small/medium-sized networks
    simply have no reaction above the cutoff, so the filter is a no-op for
    them).  Must run *before* the MT/LT era branch in :func:`load_network`,
    not only inside its per-reaction parsing loop: the MT-era intersection
    tests against the *filtered* bare names, so an MT-era solve cannot try to
    run a reaction that the LT era would have dropped for amax.

    Args:
        bare_names: list[str], bare reaction names before filtering.
        rxn_map: dict, bare name -> (reactants_field, products_field).
        nuc_NZ: dict, nuclide name -> (N, Z).
        amax: int or None; None disables the filter (returns bare_names as-is).

    Returns:
        list[str], the filtered bare names (new list; input is not mutated).
    """
    if amax is None:
        return bare_names
    filtered_bare_names = []
    for name in bare_names:
        if name not in rxn_map:
            raise KeyError(f"reaction {name!r} is not present in reactions_large.csv")
        react, _ = _side_counts(rxn_map[name][0])
        prod, _ = _side_counts(rxn_map[name][1])
        all_nuclides = set(react) | set(prod)
        if any(sum(nuc_NZ[s]) > amax for s in all_nuclides if s in nuc_NZ):
            continue
        filtered_bare_names.append(name)
    return filtered_bare_names


def _select_era_reactions(era, cfg, bare_names):
    """Restrict the (already amax-filtered) bare names to the requested era.

    Args:
        era: "MT" or "LT" (case-insensitive).
        cfg: PRIMATConfig instance (only ``cfg.network`` is read).
        bare_names: list[str], amax-filtered bare reaction names.

    Returns:
        (era, selected): era upper-cased; selected is the era-restricted list
        of bare names, in the fixed ORDER_SMALL/ORDER_MT order for "MT", or
        ``bare_names`` unchanged for "LT".
    """
    era = era.upper()
    if era == "MT" and cfg.network == "small":
        allowed = set(bare_names)
        selected = [name for name in ORDER_SMALL
                    if name not in _WEAK_NTOP_NAMES and name in allowed]
    elif era == "MT":
        allowed = set(bare_names)
        selected = [name for name in ORDER_MT
                    if name not in _WEAK_NTOP_NAMES and name in allowed]
    elif era == "LT":
        selected = bare_names
    else:
        raise ValueError(f"era must be 'MT' or 'LT', got {era!r}")
    return era, selected


def _parse_reaction_sides(selected, bare_to_file, rxn_map):
    """Resolve each selected reaction's reactant/product side counts.

    Args:
        selected: list[str], era-restricted bare reaction names.
        bare_to_file: dict, bare name -> rate-table filename.
        rxn_map: dict, bare name -> (reactants_field, products_field).

    Returns:
        (parsed, active_nuclides): parsed is a list of
        ``(name, filename, react, prod, is_weak, net_lepton_dZ)`` tuples
        (react/prod are Counter-like dicts of nuclide -> multiplicity,
        excluding Bm/Bp leptons); active_nuclides is the set of all nuclides
        (plus "n", "p") appearing in any selected reaction.
    """
    parsed = []
    active_nuclides = {"n", "p"}
    for name in selected:
        # Look up the custom filename if provided, otherwise default to the
        # PRIMAT-shipped table's name (see the fallback in
        # _parse_network_entries above).
        filename = bare_to_file.get(name, name + "_primat.txt")

        if name not in rxn_map:
            raise KeyError(f"reaction {name!r} is not present in reactions_large.csv")
        react, lepton_dZ_r = _side_counts(rxn_map[name][0])
        prod, lepton_dZ_p = _side_counts(rxn_map[name][1])
        # Net lepton charge contribution for the conservation check:
        # dZ_lepton = Z_leptons_on_products − Z_leptons_on_reactants.
        # For β⁻ decay (Bm on products): dZ_lepton = -1, balancing the +1 from
        # the nuclear Z change.  See check_conservation and _LEPTON_Z.
        net_lepton_dZ = lepton_dZ_p - lepton_dZ_r
        is_weak = (net_lepton_dZ != 0)

        # `selected` is already amax-filtered, so no per-name amax check is
        # needed here.
        parsed.append((name, filename, react, prod, is_weak, net_lepton_dZ))
        active_nuclides.update(react)
        active_nuclides.update(prod)
    return parsed, active_nuclides


def _extend_mt_species(era, cfg, bare_names, rxn_map, nuc_NZ, amax, active_nuclides):
    """Add the fixed MT-era species set (SPECIES_MD) to active_nuclides.

    The MT era historically carries a fixed set of nuclides through the
    solver even when only a subset of reactions is active. For standard
    networks this is all of SPECIES_MD (12 nuclides). For custom networks we
    only add SPECIES_MD members that actually appear in the file's full
    reaction list (before the MT intersection), so we don't carry columns for
    species that are completely absent from the network. The amax filter
    must apply here too, to keep the MT and LT eras' species lists consistent
    (e.g. amax=7 drops Li8/B8 from the LT network; carrying them through the
    MT era would leave the solver unable to map MT columns onto the LT
    abundance layout by name).

    Args:
        era: "MT" or "LT" (already upper-cased).
        cfg: PRIMATConfig instance (only ``cfg.network`` is read).
        bare_names: list[str], amax-filtered bare reaction names (the full
            network's list, before the MT intersection).
        rxn_map: dict, bare name -> (reactants_field, products_field).
        nuc_NZ: dict, nuclide name -> (N, Z).
        amax: int or None.
        active_nuclides: set[str], mutated in place to add the MT species.

    Returns:
        None (mutates ``active_nuclides`` in place); a no-op unless
        ``era == "MT"`` and ``cfg.network != "small"``.
    """
    if not (era == "MT" and cfg.network != "small"):
        return
    file_nuclides: set[str] = {"n", "p"}
    for bn in bare_names:
        if bn in rxn_map:
            r, _ = _side_counts(rxn_map[bn][0])
            p, _ = _side_counts(rxn_map[bn][1])
            file_nuclides.update(r)
            file_nuclides.update(p)
    active_nuclides.update(
        s for s in SPECIES_MD
        if s in file_nuclides and (amax is None or sum(nuc_NZ[s]) <= amax)
    )


def _build_rate_tables(parsed, idx, custom_tables, tables_dir, grid, cfg, db):
    """Assemble the per-reaction rate tables, stoichiometry and provenance.

    This is the core "fill in the columns" step of :func:`load_network`:
    for each parsed reaction (in order, after the ``n__p`` weak conversion
    prepended at index 0) it resolves the forward-rate table (custom upload /
    decay / on-disk thermonuclear table), resamples it onto the master grid,
    and looks up or computes its detailed-balance (alpha, beta, gamma)
    coefficients.

    Args:
        parsed: list of ``(name, filename, react, prod, is_weak,
            net_lepton_dZ)`` tuples from :func:`_parse_reaction_sides`.
        idx: dict, species name -> index in the solver's abundance vector.
        custom_tables: dict, see load_network's docstring.
        tables_dir: str, path to data/nuclear/tables/.
        grid: np.ndarray, master T9 grid (log-spaced).
        cfg: PRIMATConfig instance.
        db: dict, bare name -> (alpha, beta, gamma) detailed-balance triple.

    Returns:
        (names, network, weak_indices, lepton_dZ_list, sources, files,
         fwd_median, fwd_expsigma, abg): names/network/... are aligned lists
        with index 0 = the n__p weak conversion (prepended here);
        fwd_median/fwd_expsigma are 2D arrays (n_reactions, grid.size); abg is
        (n_reactions, 3).
    """
    # n__p: net lepton dZ = -1 (Bm on products, Z_Bm = -1).
    # This balances the +1 nuclear dZ (n→p converts Z=0 to Z=1), making the
    # reaction electrically neutral to check_conservation.
    names = ["n__p"]
    network = [({idx["n"]: 1}, {idx["p"]: 1})]
    weak_indices = {0}
    lepton_dZ_list = [-1]   # index 0 = n__p (electron emitted: dZ = -1)
    # Provenance label per reaction, read from each table's header (ref= field).
    # n__p has no rate table here (its weak rates are supplied at solve time), so
    # we label it as the tabulated weak n<->p conversion.
    sources = ["weak n<->p"]
    # Rate-table path per reaction (aligned with ``names``).  n__p has no rate
    # table (its weak rates are supplied at solve time), hence ``None``.
    files = [None]
    fwd_median, fwd_expsigma, abg = [], [], []
    # decays.txt is loaded lazily (only the `large` network has Bm/Bp decay
    # reactions; `is_weak` among `parsed` entries is exactly the decay flag,
    # since the only other weak reaction is n__p, handled separately above).
    decay_table = None
    for name, filename, react_names, prod_names, is_weak, net_lepton_dZ in parsed:
        names.append(name)
        if is_weak:
            weak_indices.add(len(names) - 1)
        lepton_dZ_list.append(net_lepton_dZ)

        network.append((
            {idx[s]: c for s, c in react_names.items()},
            {idx[s]: c for s, c in prod_names.items()},
        ))

        if name in custom_tables:
            # GUI-uploaded rate table override: raw (T9, rate, err) arrays on
            # the uploader's own grid, resampled with the exact same
            # log-log cubic interpolation as the on-disk tables so that
            # "custom upload" and "shipped table" are interchangeable inputs
            # to the solver.  Checked *before* the decay branch so that an
            # uploaded table wins even for a weak reaction (a brand-new added
            # weak reaction, or a user-replaced decay table).
            # 5-tuple when built from a GUI custom_network (see
            # UpdateNuclearRates.__init__): filename (or None) and the
            # ref= label read from the raw text's own header (or None if it
            # has none); a direct (non-GUI) caller's raw 3-tuple has neither.
            entry = custom_tables[name]
            T9_src, rate_src, err_src = entry[:3]
            custom_filename = entry[3] if len(entry) > 3 else None
            custom_source = entry[4] if len(entry) > 4 else None
            if is_weak:
                # A decay's override is the synthetic constant-rate table
                # from custom_rates.decay_override_table_text -- every row
                # repeats the same overridden rate, so showing the source
                # label (meaningless for a single T-independent number)
                # would be less informative than just naming it and quoting
                # the value, mirroring the unmodified-decay branch below.
                rate_val = float(np.asarray(rate_src).reshape(-1)[0])
                sources.append(f"Custom decay rate: {rate_val:.6e} s⁻¹")
            elif custom_source is not None:
                # The raw text actually carries its own provenance (e.g. an
                # existing alternate shipped table picked from the dropdown,
                # which copies that table's header verbatim) -- show it
                # rather than the generic fallback below.
                sources.append(custom_source)
            else:
                # Genuinely new/edited content with no header at all: this
                # really is the only case "custom upload" describes.
                sources.append("custom upload")
            files.append(custom_filename)
            fwd_median.append(_resample_rate_table(T9_src, rate_src, grid))
            fwd_expsigma.append(_resample_rate_table(T9_src, err_src, grid))
        elif is_weak:
            # Radioactive decay: constant (T9-independent) rate from
            # decays.txt, broadcast onto the master grid -- no rate table,
            # no resampling (see _load_decay_table's docstring).  The rate
            # itself (not just ref/half-life) is the physically meaningful
            # number to show here, since this row has no per-T9 rate table
            # to point to.
            if decay_table is None:
                decay_table = _load_decay_table(tables_dir)
            rate_s, f, halflife_s, ref = decay_table[name]
            sources.append(f"{rate_s:.6e} s⁻¹  (T1/2={halflife_s:.4g} s, {ref})")
            files.append(os.path.join(tables_dir, "decays.txt"))
            # Convert rate from s^-1 to the natural-unit T9-based convention:
            # the solver calls fill_buffer which uses rates in units of the
            # same natural-unit system as the thermonuclear rates (cm^3/s/mol,
            # etc.), but decay rates are absolute s^-1.  The network RHS
            # multiplies by rhoB^(R-1) where R=1 for a decay, giving rhoB^0=1,
            # so the rate enters the ODE directly as rate_s [s^-1].
            # We store it as rate_s (the units cancel in the ODE for R=1).
            # However, fill_buffer returns the rate in T9-grid form:
            # the rate column is just the constant rate_s repeated on the grid.
            fwd_median.append(np.full(grid.shape, rate_s))
            fwd_expsigma.append(np.full(grid.shape, f))
        else:
            # Per-reaction folder layout: data/nuclear/tables/<name>/<filename>
            # (the PRIMAT default table is <name>/<name>.txt; sibling files in
            # the same folder are alternate candidate tables, e.g. a
            # "_parthenope3.0.txt" variant -- see available_rate_tables()).
            # Resolved per-file (not via the shared tables_dir) so a single
            # user_nuclear_dir entry can override one reaction's table while
            # every other reaction still falls back to the shipped table --
            # an additive overlay, not a directory-wide takeover.
            if hasattr(cfg, "resolve_rates_path"):
                table_path = cfg.resolve_rates_path("nuclear", "tables", name, filename)
            else:
                table_path = os.path.join(tables_dir, name, filename)
            sources.append(_read_reaction_source(table_path))
            files.append(table_path)

            data = np.loadtxt(table_path, unpack=True)
            T9_src = data[0]
            # Resample from the file's own T9 grid to the master grid.  When
            # all tables share the same grid (the common case) this is nearly
            # a no-op.
            fwd_median.append(_resample_rate_table(T9_src, data[1], grid))
            fwd_expsigma.append(_resample_rate_table(T9_src, data[2], grid))

        # Detailed-balance (reverse-rate) coefficients (alpha, beta, gamma).
        # For decays: by default abg = (0,0,0) i.e. no reverse rate (decays
        # are irreversible at BBN temperatures).  When cfg.decay_reverse_rates
        # is True, compute the thermal reverse rate from the nuclide data so
        # that detailed balance is enforced -- relevant only at temperatures
        # comparable to the decay Q-value (keV range), far below BBN.
        if is_weak and name not in db:
            if cfg.decay_reverse_rates:
                # Derive (alpha, beta, gamma) from nuclide masses/spins.
                # react_names / prod_names already exclude Bm/Bp leptons
                # (from _side_counts), so pass them directly.
                # Expand stoichiometry: {'He4': 2} -> ['He4', 'He4'] so that
                # compute_detailed_balance_coefficients gets the correct
                # particle count for beta and the correct Q-value sum.
                _reactants = [s for s, c in react_names.items()
                              for _ in range(int(c))]
                _products  = [s for s, c in prod_names.items()
                              for _ in range(int(c))]
                try:
                    _alpha, _beta, _gamma = compute_detailed_balance_coefficients(
                        _reactants, _products, cfg
                    )
                    # gamma >= 0 would mean Q <= 0 (endothermic decay), which
                    # is unphysical: all beta-decays are exothermic. Guard
                    # against numerical accidents or unhandled edge cases.
                    if _gamma >= 0.0:
                        abg.append([0.0, 0.0, 0.0])
                    else:
                        abg.append([_alpha, _beta, _gamma])
                except Exception:
                    abg.append([0.0, 0.0, 0.0])
            else:
                abg.append([0.0, 0.0, 0.0])
        else:
            abg.append(list(db.get(name, (0.0, 0.0, 0.0))))

    fwd_median = np.asarray(fwd_median)
    fwd_expsigma = np.asarray(fwd_expsigma)
    abg = np.asarray(abg)
    return (names, network, weak_indices, lepton_dZ_list, sources, files,
            fwd_median, fwd_expsigma, abg)


def _apply_nuclear_qed(names, fwd_median, grid, cfg):
    """Rescale radiative-capture forward rates by the QED correction factor.

    The correction (Pitrou & Pospelov 2020) accounts for pair-production in
    the final-state photon.  Multiplying into ``fwd_median`` makes the
    corrected value the new median so that ``p_*``/``delta_*`` rate
    variations apply relative to it.

    Args:
        names: list[str], reaction names (index 0 is "n__p", skipped: the
            weak conversion has no radiative-capture QED correction).
        fwd_median: np.ndarray (n_reactions, grid.size), mutated in place.
        grid: np.ndarray, master T9 grid.
        cfg: PRIMATConfig instance.

    Returns:
        None (mutates ``fwd_median`` in place); a no-op unless
        ``cfg.nuclear_qed_corrections`` is True.
    """
    if not cfg.nuclear_qed_corrections:
        return
    for i, rname in enumerate(names[1:]):   # names[0] is n__p, handled separately
        factor = _qed_nuclear_rescale(rname, grid)
        if factor is not None:
            fwd_median[i] *= factor


def _reverse_rate_cap(grid, abg, fwd, cfg):
    """Cap each reaction's reverse rate at its value at T_nucl.

    Below T_nucl (the nucleosynthesis onset temperature) the ``exp(γ/T9)``
    factor of exothermic reverse rates can grow by many orders of magnitude,
    producing a stiff "blow-up" for heavy nuclides in the large network that
    would prevent BDF convergence. Pinning the cap at T_nucl preserves
    detailed balance near BBN onset and removes the low-T divergence safely
    (see the module docstring for the large-network caveats).

    Args:
        grid: np.ndarray, master T9 grid (log-spaced).
        abg: np.ndarray (n_reactions, 3), (alpha, beta, gamma) per reaction.
        fwd: np.ndarray (n_reactions, grid.size), forward rates on the grid.
        cfg: PRIMATConfig instance (``cfg.T_nucl`` in K).

    Returns:
        np.ndarray (n_reactions,), the reverse-rate cap per reaction.
    """
    j = int(np.searchsorted(grid, cfg.T_nucl / 1.0e9))  # index of T9 ≈ T_nucl/10⁹
    j = min(max(j, 0), grid.size - 1)
    a_, b_, g_ = abg[:, 0], abg[:, 1], abg[:, 2]
    T9c = grid[j]                              # T9 at the capping temperature
    return a_ * T9c ** b_ * np.exp(np.minimum(g_ / T9c, _EXP_CAP)) * fwd[:, j]


def load_network(cfg, subset_file=None, era: str = "LT", reaction_names=None,
                  custom_tables=None):
    """Build the selected network from its text reaction list.

    Parameters
    ----------
    cfg : PRIMATConfig
        Configuration with repository paths and temperature boundaries.
    subset_file : str, optional
        Legacy filename or modern filename.  When omitted, ``cfg.network`` picks
        ``small.txt``, ``medium.txt`` or ``large.txt``.
    era : {"MT", "LT"}
        ``"MT"`` keeps only the intersection with :data:`ORDER_MT`; ``"LT"``
        keeps the full selected list.
    reaction_names : sequence[str], optional
        Direct reaction list, mainly for tests.  Also the mechanism used to
        *remove* a reaction for a GUI "custom network": simply omit its entry
        from the list passed here (no separate removal parameter is needed).
    custom_tables : dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], optional
        Per-reaction rate-table override: ``name -> (T9_src, rate_src, err_src)``
        raw arrays (not yet resampled onto the master grid).  When a reaction's
        bare name is a key of this dict, its forward-rate table is taken from
        these arrays (run through the same :func:`_resample_rate_table` log-log
        cubic interpolation as the on-disk tables) instead of from
        ``data/nuclear/tables/<name>.txt``.  Used by the GUI's "Customise
        Reactions" panel to substitute a user-uploaded rate table without
        writing anything to disk.

    Returns
    -------
    NetworkDefinition
        Species, stoichiometry, rate tables and detailed-balance coefficients in
        the exact order expected by the solver buffer.

    Example
    -------
    >>> net = load_network(PRIMATConfig({"network": "large", "amax": 8}), era="LT")
    >>> len(net.names)  # n__p + 67 thermonuclear reactions (A <= 8)
    68

    Implementation
    --------------
    This function only sequences the named steps below (each independently
    unit-tested): parse the network-file entries, resolve/inject custom rate
    tables, apply the ``amax`` cutoff, intersect with the requested era,
    resolve each reaction's stoichiometry, determine the active species set,
    build the per-reaction rate tables on the master T9 grid, apply nuclear
    QED rescaling, and compute the reverse-rate cap.
    """
    if custom_tables is None:
        custom_tables = {}
    if reaction_names is None:
        reaction_names = load_reaction_names(cfg, subset_file or cfg.network)

    bare_names, bare_to_file = _parse_network_entries(
        reaction_names, subset_file or cfg.network)

    tables_dir, data_dir, nuc_order, nuc_NZ, db, rxn_map = _reaction_catalog(cfg._resolved_data_dir)

    rxn_map, db = _inject_custom_reactions(bare_names, custom_tables, rxn_map, db, cfg)

    amax = cfg.amax
    bare_names = _apply_amax_filter(bare_names, rxn_map, nuc_NZ, amax)

    era, selected = _select_era_reactions(era, cfg, bare_names)

    parsed, active_nuclides = _parse_reaction_sides(selected, bare_to_file, rxn_map)

    _extend_mt_species(era, cfg, bare_names, rxn_map, nuc_NZ, amax, active_nuclides)

    species = _species_order(active_nuclides, nuc_order)
    idx = {s: i for i, s in enumerate(species)}
    N = np.array([nuc_NZ[s][0] for s in species], dtype=int)
    Z = np.array([nuc_NZ[s][1] for s in species], dtype=int)

    # Master T9 grid: all tables are resampled onto this grid at load time so
    # that fill_buffer's single searchsorted path is always valid, regardless of
    # the grid used when generating the rate files (e.g. --keep-source-grid).
    grid = np.logspace(np.log10(cfg.rate_grid_T9_min),
                       np.log10(cfg.rate_grid_T9_max),
                       cfg.rate_grid_npts)

    (names, network, weak_indices, lepton_dZ_list, sources, files,
     fwd_median, fwd_expsigma, abg) = _build_rate_tables(
        parsed, idx, custom_tables, tables_dir, grid, cfg, db)

    _apply_nuclear_qed(names, fwd_median, grid, cfg)

    # Active forward rates (initially median)
    fwd = fwd_median.copy()

    bwd_cap = _reverse_rate_cap(grid, abg, fwd, cfg)

    return NetworkDefinition(species, N, Z, network, weak_indices, names, grid,
                             fwd, fwd_median, fwd_expsigma, abg, bwd_cap,
                             lepton_dZ=lepton_dZ_list, sources=sources,
                             files=files)


def reaction_category(name: str) -> int:
    """Heaviest nuclide's mass number A (=N+Z) among name's reactants/products.

    Drives the GUI popup's mass-number-banded category view
    (CUSTOMPOPUP.md §5/§6): category k contains exactly the reactions that a
    filter of amax=k keeps but amax=k-1 would drop, so categories and the
    amax filter stay consistent by construction. Photon ("g") and lepton
    ("Bm"/"Bp") tokens are excluded from the max (they don't carry a mass
    number).

    Uses :func:`reaction_stoichiometry`, so it works for shipped catalog
    reactions, network-file "TO"-derived reactions, and brand-new GUI-added
    reactions alike (anything reaction_stoichiometry can parse).

    Example
    -------
    >>> reaction_category("n_p__d_g")
    2
    """
    react, prod = reaction_stoichiometry(name)
    _, _, _, nuc_NZ, _, _ = _reaction_catalog(_default_data_dir())
    nuclides = set(react) | set(prod)
    return max(sum(nuc_NZ[s]) for s in nuclides if s in nuc_NZ)


def group_reactions_by_category(names) -> dict:
    """{category_A: [bare_name, ...]}, sorted by category; names keep input order."""
    groups: dict[int, list[str]] = {}
    for name in names:
        groups.setdefault(reaction_category(name), []).append(name)
    return dict(sorted(groups.items()))


# True maximum nuclide mass number reachable in the large network's catalog
# (measured: the heaviest nuclide referenced by any data/nuclear/networks/
# large.txt reaction is Na23, A=23). Used by the GUI popup (CUSTOMPOPUP.md
# §6.2/§7.1) to detect "this kept-reaction list used every reaction up to the
# top of the catalog", i.e. equivalent to "no amax filter" (amax=None).
AMAX_LARGE = 23


class UpdateNuclearRates:
    """Build era networks and temperature-dependent rate buffers.

    This class unifies all networks (small, medium, large) under a single
    architecture: each solver era (MT, LT) has a corresponding
    ``NetworkDefinition`` which manages stoichiometric network compilation and
    fast, vectorised rate-buffer filling.
    """

    def __init__(self, cfg, custom_network=None):
        """
        Parameters
        ----------
        custom_network : dict, optional
            GUI "Customise Reactions" override, with three keys:

            - ``"removed"``: list of bare reaction names to drop entirely from
              ``cfg.network``'s reaction list (e.g. ``["d_d__t_p"]``).
            - ``"replaced"``: dict ``name -> raw_table_text``, where
              ``raw_table_text`` is the verbatim contents of a 2- or 3-column
              uploaded rate file (``T9  rate  [err]``) for a *kept* reaction,
              parsed here with :func:`numpy.loadtxt` on an in-memory buffer (no
              file is written to disk) and fed to :func:`load_network` as
              ``custom_tables`` so it is resampled with the exact same
              log-log cubic interpolation as the shipped tables.
            - ``"added"``: dict ``name -> raw_table_text`` for *brand-new*
              reactions that need not exist in ``cfg.network`` (or even in the
              shipped catalog ``reactions_large.csv``).  The name must follow
              the ``a_b__c_d`` syntax so that :func:`load_network` can derive
              its stoichiometry (and detailed-balance reverse rate) from the
              name and the nuclide tables; its forward rate comes from the
              uploaded table, just like ``"replaced"``.
            - ``"filenames"`` (optional): dict ``name -> filename`` naming the
              on-disk/uploaded basename behind a ``"replaced"``/``"added"``
              entry, display-only (used by :meth:`NetworkDefinition.describe_reactions`'s
              "File" column; absent entries just show no filename).

            ``None`` (default) reproduces the standard, uncustomised network.
            Never applies to the weak ``n__p`` reaction (handled by a separate
            cache, see :mod:`primat.weak_rates`).
        """
        if cfg.verbose:
            print(f"[rates-py] Building {cfg.network!r} network from text lists.")

        removed = set(custom_network.get("removed", [])) if custom_network else set()
        custom_tables = {}
        added_names = []
        if custom_network:
            # "replaced" tables override a kept reaction; "added" tables supply
            # a brand-new reaction.  Both are parsed identically here and handed
            # to load_network via custom_tables; the only difference is that the
            # added names are appended to the selected reaction list below.
            added = custom_network.get("added", {})
            added_names = list(added)
            # Optional, display-only: the GUI's per-reaction filename/upload
            # basename (CUSTOMPOPUP.md), threaded through purely so
            # describe_reactions() can show it in the Reactions summary tab's
            # "File" column instead of leaving it blank for a customised
            # reaction.  Absent for direct (non-GUI) custom_network dicts.
            filenames = custom_network.get("filenames", {})
            for name, raw_text in {**custom_network.get("replaced", {}), **added}.items():
                data = np.loadtxt(io.StringIO(raw_text), unpack=True)
                T9_src, rate_src = data[0], data[1]
                err_src = data[2] if data.shape[0] > 2 else np.zeros_like(rate_src)
                # Read the raw text's own "ref=" header if it has one (e.g. the
                # user picked an existing alternate shipped table from the
                # dropdown, which copies that table's header verbatim -- see
                # custom_rates.export_zip/_match_shipped_file). Only a
                # genuinely uploaded/edited table with no such header falls
                # back to a generic "custom upload" label in
                # _build_rate_tables.
                source = _reaction_source_from_lines(raw_text.splitlines())
                custom_tables[name] = (T9_src, rate_src, err_src, filenames.get(name), source)

        self._selected_names = [n for n in load_reaction_names(cfg, cfg.network)
                                 if re.split(r'[, ]+', n, maxsplit=1)[0].strip() not in removed]
        self._selected_names += added_names
        self._mt_net = load_network(cfg, era="MT", reaction_names=self._selected_names,
                                     custom_tables=custom_tables)
        self._lt_net = load_network(cfg, era="LT", reaction_names=self._selected_names,
                                     custom_tables=custom_tables)

        # Apply initial variations
        self.apply_variations(cfg)

        self._K_MT = self._compile_checked(self._mt_net, cfg)
        self._K_LT = self._compile_checked(self._lt_net, cfg)
        
        self._rhsMT_rbuf = np.empty(2 * len(self._mt_net.names))
        self._jacMT_rbuf = np.empty(2 * len(self._mt_net.names))
        self._rhsLT_rbuf = np.empty(2 * len(self._lt_net.names))
        self._jacLT_rbuf = np.empty(2 * len(self._lt_net.names))

        # Legacy attribute names used by tests and output helpers.
        self._order_MT = self._mt_net.names
        self._order_LT = self._lt_net.names
        self.species_large = self._lt_net.species
        self.large_NZ = {
            s: (int(n), int(z)) for s, n, z in zip(self._lt_net.species,
                                                   self._lt_net.N, self._lt_net.Z)
        }

        if cfg.verbose:
            print(f"[rates-py] MT network: {len(self._mt_net.names)-1} reactions over "
                  f"{len(self._mt_net.species)} nuclides.")
            print(f"[rates-py] MT nuclides: {', '.join(self._mt_net.species)}")
            print(f"[rates-py] LT network: {len(self._lt_net.names)-1} reactions over "
                  f"{len(self._lt_net.species)} nuclides.")
            print(f"[rates-py] LT nuclides: {', '.join(self._lt_net.species)}")
            self.print_reactions()

    def describe_reactions(self):
        """Return the LT (full) network's reactions as
        ``(name, equation, source, file)`` tuples.

        Thin delegate to :meth:`NetworkDefinition.describe_reactions` for the LT
        network (the complete selected reaction set; the MT era only uses a fixed
        18-reaction subset).  The fourth element is the rate-table path (``None``
        for the weak ``n__p`` entry).  Used by the verbose console listing and by
        the GUI reactions table.
        """
        return self._lt_net.describe_reactions()

    def print_reactions(self):
        """Print the loaded LT reactions as ``a + b <-> c + d   [source]``.

        Each line shows the reaction in readable form together with the data
        source taken from the rate table's header (the ``ref=`` field).  Called
        automatically from :meth:`__init__` when ``cfg.verbose`` is set.
        """
        reactions = self.describe_reactions()
        print("-" * 60)
        print(f"Loaded {len(reactions)} reactions (LT network):")
        print("-" * 60)
        # Pad the equation column so the source labels line up in the terminal.
        width = max(len(eq) for _, eq, _, _ in reactions)
        for name, equation, source, _file in reactions:
            print(f"  {equation:<{width}}   [{source}]")

    def apply_variations(self, cfg):
        """Update active forward rate tables in both era networks."""
        self._mt_net.apply_variations(cfg)
        self._lt_net.apply_variations(cfg)

    @staticmethod
    def _compile_checked(net, cfg):
        """Compile a network and fail early if N/Z/A conservation is broken."""
        cnet = compile_network(net.network, len(net.species))
        check_conservation(cnet, net.N, net.Z,
                           weak_indices=net.weak_indices,
                           lepton_dZ=net.lepton_dZ)
        return NetworkKernels(cnet, cfg.numba_installed)

    def rhsMT(self, Y, T_t, rhoBBN, nTOp_frwrd, nTOp_bkwrd):
        """MT RHS for the selected network intersected with :data:`ORDER_MT`."""
        r = self._mt_net.fill_buffer(T_t, nTOp_frwrd, nTOp_bkwrd, clamp=False)
        return self._K_MT.rhs(Y, rhoBBN, r)

    def JacobianMT(self, Y, T_t, rhoBBN, nTOp_frwrd, nTOp_bkwrd):
        """MT analytic Jacobian for the selected network."""
        r = self._mt_net.fill_buffer(T_t, nTOp_frwrd, nTOp_bkwrd, clamp=False)
        return self._K_MT.jacobian(Y, rhoBBN, r)

    def rhsLT(self, Y, T_t, rhoBBN, nTOp_frwrd, nTOp_bkwrd):
        """LT RHS for the full selected network."""
        r = self._lt_net.fill_buffer(T_t, nTOp_frwrd, nTOp_bkwrd, clamp=True)
        return self._K_LT.rhs(Y, rhoBBN, r)

    def JacobianLT(self, Y, T_t, rhoBBN, nTOp_frwrd, nTOp_bkwrd):
        """LT analytic Jacobian for the full selected network."""
        r = self._lt_net.fill_buffer(T_t, nTOp_frwrd, nTOp_bkwrd, clamp=True)
        return self._K_LT.jacobian(Y, rhoBBN, r)


def _make_frwrd(rxn):
    """Create a temporary forward-rate extractor for output logging."""
    def frwrd(self, T):
        # We need a way to extract the active rate for logging in time_evolution.
        # Since we unified everything, we can use the LT network's table.
        # This is only used for output_rates_time_evolution=True.
        T9 = T * 1e-9
        net = self._lt_net
        if rxn not in net.names: return 0.0
        j = net.names.index(rxn) - 1 # skip n__p
        g = net.grid
        i = int(np.searchsorted(g, T9) - 1)
        i = min(max(i, 0), g.size - 2)
        w = (T9 - g[i]) / (g[i + 1] - g[i])
        return net._fwd[j, i] * (1.0 - w) + net._fwd[j, i + 1] * w
    frwrd.__name__ = f"{rxn}_frwrd"
    return frwrd


# We dynamically add the extractor methods to UpdateNuclearRates to support
# the existing output_time_evolution logic in PRIMAT.
for _rxn in _REACTIONS_LARGE:
    setattr(UpdateNuclearRates, f"{_rxn}_frwrd", _make_frwrd(_rxn))
