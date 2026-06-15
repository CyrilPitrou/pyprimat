# -*- coding: utf-8 -*-
"""
Unified nuclear-network construction and use.

This module is the single place where PyPRIMAT turns a named reaction list into
ODE equations.  The lists in ``rates/nuclear/networks/{small,medium,large}.txt``
name the thermonuclear reactions to keep.  The weak ``n <-> p`` conversion is
not stored in those text files because the high-temperature era integrates only
``n`` and ``p`` directly; for MT/LT network solves it is prepended internally as
the first buffer entry.

The physics convention is the usual mass-action BBN form.  A reaction
``r + ... -> p + ...`` contributes a forward flux proportional to the product of
reactant abundances, a backward flux proportional to product abundances and the
detailed-balance coefficient, and each nuclide receives its stoichiometric
coefficient times ``forward - backward``.  ``compile_network`` and
``NetworkKernels`` from :mod:`pyprimat.network_builder` convert that declarative
stoichiometry into fast RHS/Jacobian evaluators.
"""

from __future__ import annotations

import os
import re
import csv
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
    "_REACTIONS_MEDIUM",
    "_KEY12_REACTIONS",
    "check_conservation",
    "compile_network",
    "load_network",
    "load_reaction_names",
    "nuclide_latex",
    "network_jacobian",
    "network_rhs",
    "phase_network",
    "reaction_stoichiometry",
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

# Historical MT order from PRIMAT.  MT always integrates the intersection of the
# selected network with this list, because activating the full network before
# the deuterium bottleneck opens makes the BDF problem unnecessarily stiff.
ORDER_MT = [
    "nTOp", "Be7dTOaap", "Be7nTOLi7p", "Be7nTOaa", "He3aTOBe7g",
    "He3dTOap", "He3nTOtp", "Li6pTOBe7g", "Li7pTOaa", "Li7pTOaag",
    "daTOLi6g", "ddTOHe3n", "ddTOtp", "dpTOHe3g", "npTOdg",
    "taTOLi7g", "tdTOan", "tpTOag",
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

# Matches every other PyPRIMAT nuclide name: an element symbol (one capital
# letter optionally followed by a lowercase letter, e.g. "He", "B", "Na")
# followed by its mass number (e.g. "He3", "B10", "Na23").
_NUCLIDE_NAME_RE = re.compile(r"^([A-Z][a-z]?)(\d+)$")


def nuclide_latex(name):
    """Return the LaTeX form of a PyPRIMAT nuclide name, e.g. for axis labels.

    PyPRIMAT names nuclides as ``"<element symbol><mass number>"`` (e.g.
    ``"He3"``, ``"B10"``), with the neutron and proton as the bare bookkeeping
    names ``"n"`` and ``"p"``.  This maps such a name to the standard
    isotope notation ``${}^{A}\\mathrm{Sym}$`` (e.g. ``"He3"`` ->
    ``r"${}^{3}\\mathrm{He}$"``), suitable for Matplotlib/Plotly labels and
    Streamlit tables (which both support a LaTeX subset via ``$...$``).

    Parameters
    ----------
    name : str
        A nuclide name as it appears in ``PyPR.abundance_names``.

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

# Token aliases used when parsing compact PRIMAT names such as ``ddTOHe3n``.
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
    "nTOp", "npTOdg", "dpTOHe3g", "ddTOHe3n", "ddTOtp", "tpTOag",
    "tdTOan", "taTOLi7g", "He3nTOtp", "He3dTOap", "He3aTOBe7g",
    "Be7nTOLi7p", "Li7pTOaa",
]
_KEY12_REACTIONS = ORDER_SMALL[1:]


def _network_dir_from_cwd() -> str:
    """Return the package's network-list directory for import-time defaults.

    ``rates/`` lives inside the ``pyprimat`` package (it is shipped as package
    data), so the path is resolved relative to this file — never the current
    working directory — and works for both editable and regular installs.
    """
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "rates", "nuclear", "networks")
    )


def load_reaction_names(cfg_or_dir, network: str | None = None) -> list[str]:
    """Read a thermonuclear reaction list from ``rates/nuclear/networks``.

    Parameters
    ----------
    cfg_or_dir : PyPRConfig or str
        Either a configuration object with ``data_dir`` or a direct path to
        the ``networks`` directory.  Accepting both forms lets module constants
        be initialised without constructing a full configuration.
    network : str, optional
        Which list to read.  ``"small"`` is special and returns
        :data:`ORDER_SMALL` without touching the filesystem.  Any other value
        is interpreted as ``<network>.txt`` inside ``rates/nuclear/networks``
        unless it already ends in ``.txt``.

    Returns
    -------
    list[str]
        Reaction names in file order, excluding comments and blank lines.

    Example
    -------
    >>> len(load_reaction_names("/repo/rates/nuclear/networks", "small"))
    12
    """
    if hasattr(cfg_or_dir, "data_dir"):
        nets_dir = os.path.join(cfg_or_dir.data_dir, "rates", "nuclear", "networks")
        network = network or cfg_or_dir.network
    else:
        nets_dir = os.fspath(cfg_or_dir)
        network = network or "medium"

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


try:
    _REACTIONS_MEDIUM = load_reaction_names(_network_dir_from_cwd(), "medium")
except OSError:
    # Importing documentation tooling outside the repository should still work.
    _REACTIONS_MEDIUM = []

ORDER_LT = ["nTOp"] + _REACTIONS_MEDIUM


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
        Nuclide keys (PyPRConfig.Nuclides convention: 'n', 'p', 'H2', 'H3',
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
    cfg : PyPRConfig
        Supplies the nuclide tables (``Nuclides`` = [N, Z], ``NuclExcessMass``
        in keV, ``NuclSpin``) and the fundamental constants (kB, hbar, clight,
        ma, me, keV, MeV) in the CGS-erg system used throughout PyPRIMAT.

    Returns
    -------
    (alpha, beta, gamma) : tuple[float, float, float]
        Coefficients of ``alpha * T9**beta * exp(gamma/T9)`` (the backward /
        forward rate ratio).

    Example
    -------
    >>> cfg = PyPRConfig()
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
    """Split a compact PRIMAT reaction name into nuclide and separator tokens."""
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


def reaction_stoichiometry(name):
    """Return ``(reactants, products)`` as nuclide-multiplicity dictionaries.

    The compact PRIMAT names concatenate nuclide tokens.  For example
    ``ddTOHe3n`` means ``d + d -> He3 + n`` and is returned as
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
    >>> reaction_stoichiometry("npTOdg")
    ({'n': 1, 'p': 1}, {'H2': 1})
    """
    if name == "nTOp":
        # The physical reaction is n → p + e⁻(Bm), but the ODE state vector
        # does not track the emitted electron.  The lepton charge bookkeeping
        # lives in NetworkDefinition.lepton_dZ (= -1 for nTOp), so the caller
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
    """Map a compact reaction name to the ``<reactants>TO<products>`` file name.

    Some historical code used names without the explicit ``TO`` separator.  This
    helper keeps that mapping in the same module that owns network construction.

    Example
    -------
    >>> to_filename("npdg")
    'npTOdg'
    """
    if name == "nTOp":
        return "nTOp"
    if "TO" in name:
        return name

    _, _, _, _, db, _ = _reaction_catalog(_default_data_dir())

    if name in db: return name

    tokens = _tokenise(name)
    n_nuclide = len([t for t in tokens if t != "g"])
    
    # Try to find the reaction in db by splitting tokens at different points
    for i in range(1, n_nuclide):
        candidate = "".join(tokens[:i]) + "TO" + "".join(tokens[i:])
        # Normalize tokens (d->H2 etc) for matching if needed, but CSV uses PRIMAT tokens
        if candidate in db:
            return candidate

    raise ValueError(f"cannot map reaction name {name!r} to a TO-separated filename")


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
    >>> phase_network(["nTOp"], ["n", "p"])
    [({0: 1}, {1: 1})]
    """
    idx = {s: i for i, s in enumerate(species)}
    net = []
    for name in order:
        react, prod = reaction_stoichiometry(name)
        net.append(({idx[s]: c for s, c in react.items()},
                    {idx[s]: c for s, c in prod.items()}))
    return net


def _sym(multiplicities):
    """Return the identical-particle symmetry factor for one reaction side."""
    s = 1
    for c in multiplicities.values():
        s *= factorial(c)
    return s


def network_rhs(Y, rhoBBN, r, network):
    """Reference mass-action RHS for a compiled-by-hand network description.

    This pure-Python implementation is used as an exact, readable oracle for
    tests.  Production solves use :class:`NetworkKernels`, but both evaluate the
    same formula: forward flux minus backward flux, distributed by net
    stoichiometric coefficients.
    """
    dY = np.zeros(len(Y))
    for i, (react, prod) in enumerate(network):
        R = sum(react.values())
        P = sum(prod.values())
        Ff = r[2 * i] * rhoBBN ** (R - 1) / _sym(react)
        for s, c in react.items():
            Ff *= Y[s] ** c
        Fb = r[2 * i + 1] * rhoBBN ** (P - 1) / _sym(prod)
        for s, c in prod.items():
            Fb *= Y[s] ** c
        net = Ff - Fb
        for s, c in react.items():
            dY[s] -= c * net
        for s, c in prod.items():
            dY[s] += c * net
    return dY


def _dmonomial(Y, terms, u):
    """Differentiate ``prod_s Y[s]**terms[s]`` with respect to ``Y[u]``."""
    if u not in terms:
        return 0.0
    v = terms[u] * Y[u] ** (terms[u] - 1)
    for s, c in terms.items():
        if s != u:
            v *= Y[s] ** c
    return v


def network_jacobian(Y, rhoBBN, r, network):
    """Reference analytic Jacobian matching :func:`network_rhs`."""
    n = len(Y)
    J = np.zeros((n, n))
    for i, (react, prod) in enumerate(network):
        R = sum(react.values())
        P = sum(prod.values())
        cf = r[2 * i] * rhoBBN ** (R - 1) / _sym(react)
        cb = r[2 * i + 1] * rhoBBN ** (P - 1) / _sym(prod)
        for u in range(n):
            dnet = cf * _dmonomial(Y, react, u) - cb * _dmonomial(Y, prod, u)
            if dnet == 0.0:
                continue
            for s, c in react.items():
                J[s, u] -= c * dnet
            for s, c in prod.items():
                J[s, u] += c * dnet
    return J


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

    ``names`` includes the prepended weak ``nTOp`` entry when present, while the
    text network files name only thermonuclear reactions.  ``network`` is the
    corresponding index-based stoichiometry used by :func:`compile_network`.

    Example
    -------
    >>> cfg = PyPRConfig({"network": "medium"})
    >>> net = load_network(cfg, era="MT")
    >>> net.names[0], len(net.species)
    ('nTOp', 12)
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
    lepton_dZ: list[int]   # net lepton charge per reaction: e.g. -1 for nTOp (Bm emitted)
    # Per-reaction provenance string (aligned with ``names``), read from the
    # first ``#`` header line of each rate table at load time (e.g. "And06" for
    # npTOdg).  ``None`` when the network was built without source bookkeeping
    # (e.g. directly from ``reaction_names`` in a test).
    sources: list[str] | None = None
    # Per-reaction rate-table path (aligned with ``names``): the on-disk
    # ``rates/nuclear/tables/<name>.txt`` each forward rate was loaded from.
    # ``None`` for entries with no rate table (``nTOp``, whose weak rates are
    # supplied at solve time) and ``None`` for the whole list when the network
    # was built without source bookkeeping. Used by the GUI reactions table to
    # offer a download button per reaction.
    files: list[str] | None = None

    def __post_init__(self):
        self.index = {s: i for i, s in enumerate(self.species)}
        self.n_reac = len(self.network)
        self._buf = np.empty(2 * self.n_reac)

    def reaction_equation(self, i):
        """Human-readable equation for reaction ``i`` as ``a + b <-> c + d``.

        Built from the index-based stoichiometry ``self.network[i]`` and the
        ``self.species`` name list, so a reaction such as ``ddTOHe3n`` (stored
        as ``({H2: 2}, {He3: 1, n: 1})``) is rendered ``H2 + H2 <-> He3 + n``.
        Stoichiometric multiplicities are expanded into repeated tokens to
        mirror the compact PRIMAT reaction name.  The weak entry ``nTOp`` is
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
        ``('npTOdg', 'n + p <-> H2', 'And06', '.../tables/npTOdg.txt')``.

        Returns
        -------
        list[tuple[str, str, str, str | None]]
            One tuple per reaction, in solver/buffer order (``nTOp`` first).
            The last element is the rate-table path, or ``None`` for entries
            with no table (e.g. the weak ``nTOp`` conversion).
        """
        src = self.sources if self.sources is not None else [""] * len(self.names)
        files = self.files if self.files is not None else [None] * len(self.names)
        return [
            (name, self.reaction_equation(i), src[i], files[i])
            for i, name in enumerate(self.names)
        ]

    def apply_variations(self, cfg):
        """Update the active forward rate tables ``self._fwd`` by applying any
        variation parameters (p_* and NP_delta_*) from the configuration.

        This allows Monte Carlo loops to reuse the same network objects while
        refreshing the rates at the start of each solve.
        """
        NP = cfg.rescale_nuclear_rates
        # Skip names[0] which is always nTOp (handled separately in the solver)
        for i, name in enumerate(self.names[1:]):
            p = getattr(cfg, f"p_{name}")
            NP_delta = getattr(cfg, f"NP_delta_{name}")

            if p == 0.0 and (not NP or NP_delta == 0.0):
                # No variation: revert to median
                self._fwd[i] = self._fwd_median[i]
            else:
                # Apply p uncertainty: median * exp(p * log(expsigma))
                variation = np.exp(p * np.log(self._expsigma[i]))
                if NP:
                    variation += NP_delta
                self._fwd[i] = self._fwd_median[i] * variation

    def fill_buffer(self, T_t, nTOp_frwrd, nTOp_bkwrd, clamp=True):
        """Fill the forward/backward rate buffer at photon temperature ``T_t``.

        ``T_t`` is in kelvin.  Forward rates are linearly interpolated in the
        active rate tables.  Backward rates are obtained from detailed balance.
        Clamping is applied if ``clamp=True`` to prevent low-temperature
        reverse-rate blow-up (standard for the large network's LT era).
        """
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
        
        if clamp:
            np.minimum(bwd, self._bwd_cap, out=bwd)

        r[2::2] = fwd
        r[3::2] = bwd
        return r


def _default_data_dir() -> str:
    """Package data root (``cfg.data_dir`` is always this same path).

    Defined here so :func:`reaction_stoichiometry` and :func:`to_filename` can
    reach :func:`_reaction_catalog` without constructing a throwaway
    ``PyPRConfig`` (which re-reads ``nuclides.csv`` and would create a
    config<->nuclear circular import).
    """
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=None)
def _reaction_catalog(data_dir: str):
    """Load nuclide metadata, reaction stoichiometry and detailed balance tables.

    Parameters
    ----------
    data_dir : str
        Package data root, i.e. ``cfg.data_dir`` (the directory containing
        ``rates/``).  This is a fixed path for a given PyPRIMAT installation,
        so the result is cached with :func:`functools.lru_cache`: the three
        CSV files under ``rates/nuclear/data/`` are read at most once per
        process instead of on every :func:`load_network`,
        :func:`reaction_stoichiometry` or :func:`to_filename` call.
    """
    base = os.path.join(data_dir, "rates", "nuclear", "data")
    tables_dir = os.path.join(data_dir, "rates", "nuclear", "tables")
    _, nuc_rows = _read_csv(os.path.join(base, "nuclides.csv"))
    nuc_order = [row[0] for row in nuc_rows]
    nuc_NZ = {row[0]: (int(row[1]), int(row[2])) for row in nuc_rows}

    _, db_rows = _read_csv(os.path.join(base, "detailed_balance.csv"))
    db = {row[0]: (float(row[2]), float(row[3]), float(row[4])) for row in db_rows}

    # Prefer the generated data copy.  The repository-root mirror is accepted so
    # users can inspect or regenerate the catalog without changing loader code.
    rxn_path = os.path.join(base, "reactions_large.csv")
    if not os.path.exists(rxn_path):
        rxn_path = os.path.join(data_dir, "rates", "nuclear", "reactions_large.csv")
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

    For ``npTOdg`` the correction is taken from a polynomial fit to a dedicated
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
        Reaction identifier (e.g. ``"npTOdg"``).
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
    >>> f = _qed_nuclear_rescale("npTOdg", T9)
    >>> (f > 1.0).all()
    True
    """
    # Fine structure constant (CODATA 2018)
    ALPHA = 1.0 / 137.035999084
    # Electron mass [MeV] (PDG)
    ME_MEV = 0.51099895

    if name == "npTOdg":
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
        "dpTOHe3g":  (1, 2,  1, 1,  (13135.723 + 7288.971 - 14931.219) * 1e-3),
        "tpTOag":    (1, 3,  1, 1,  (14949.811 + 7288.971 -  2424.916) * 1e-3),
        "taTOLi7g":  (1, 3,  2, 4,  (14949.811 + 2424.916 - 14907.105) * 1e-3),
        "He3aTOBe7g":(2, 3,  2, 4,  (14931.219 + 2424.916 - 15769.000) * 1e-3),
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


def _read_reaction_source(table_path):
    """Extract the data source label from a rate table's first ``#`` line.

    Every rate table under ``rates/nuclear/tables/`` starts with a header such
    as ``# n + p > d + g   [npTOdg]   ref=And06``.  The ``ref=`` field names the
    experimental/theoretical compilation the rate was taken from (here the
    ``And06`` = Ando et al. 2006 evaluation).  This helper returns that label so
    the verbose log and the GUI can show each reaction's provenance.

    Parameters
    ----------
    table_path : str
        Path to the rate table ``.txt`` file.

    Returns
    -------
    str
        The text after ``ref=`` on the first ``#`` line (e.g. ``"And06"``).  If
        the header has no ``ref=`` field, the whole comment line (minus the
        leading ``#``) is returned; if the file cannot be read, ``"?"``.
    """
    try:
        with open(table_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if "ref=" in line:
                        return line.split("ref=", 1)[1].strip()
                    # No explicit ref= field: fall back to the bare comment.
                    return line.lstrip("#").strip()
                break  # first non-comment line reached without a header
    except OSError:
        pass
    return "?"


def load_network(cfg, subset_file=None, era: str = "LT", reaction_names=None):
    """Build the selected network from its text reaction list.

    Parameters
    ----------
    cfg : PyPRConfig
        Configuration with repository paths and temperature boundaries.
    subset_file : str, optional
        Legacy filename or modern filename.  When omitted, ``cfg.network`` picks
        ``small.txt``, ``medium.txt`` or ``large.txt``.
    era : {"MT", "LT"}
        ``"MT"`` keeps only the intersection with :data:`ORDER_MT`; ``"LT"``
        keeps the full selected list.
    reaction_names : sequence[str], optional
        Direct reaction list, mainly for tests.

    Returns
    -------
    NetworkDefinition
        Species, stoichiometry, rate tables and detailed-balance coefficients in
        the exact order expected by the solver buffer.

    Example
    -------
    >>> net = load_network(PyPRConfig({"network": "large"}), era="LT")
    >>> len(net.names)  # nTOp + 423 thermonuclear reactions
    424
    """
    if reaction_names is None:
        reaction_names = load_reaction_names(cfg, subset_file or cfg.network)
    # Reject literal duplicate entries (e.g. the same line twice in a network
    # file) rather than silently dropping the repeat -- a duplicate is far
    # more likely to be a copy-paste mistake than an intentional no-op.
    seen_entries = set()
    for entry in reaction_names:
        if entry in seen_entries:
            raise ValueError(
                f"reaction entry {entry!r} is already present in network "
                f"{getattr(cfg, 'network', subset_file)!r} (duplicate line "
                f"in the network file)")
        seen_entries.add(entry)
    selected = list(reaction_names)

    # Pre-parse entries before the MT intersection: each entry is either a bare
    # reaction name ("npTOdg") or a "bare_name, filename.txt" pair that points
    # to a non-default rate table.  We extract bare names for the intersection
    # and keep the filename mapping for the rate-table loading loop below.
    bare_to_file = {}
    bare_names = []
    for entry in selected:
        parts = re.split(r'[, ]+', entry, maxsplit=1)
        if len(parts) > 1:
            bare, fname = parts[0].strip(), parts[1].strip()
        else:
            bare = entry.strip()
            fname = bare + ".txt"
        bare_names.append(bare)
        bare_to_file[bare] = fname

    era = era.upper()
    if era == "MT" and getattr(cfg, "network", None) == "small":
        allowed = set(bare_names)
        selected = [name for name in ORDER_SMALL if name != "nTOp" and name in allowed]
    elif era == "MT":
        allowed = set(bare_names)
        selected = [name for name in ORDER_MT if name != "nTOp" and name in allowed]
    elif era == "LT":
        selected = bare_names
    else:
        raise ValueError(f"era must be 'MT' or 'LT', got {era!r}")

    tables_dir, data_dir, nuc_order, nuc_NZ, db, rxn_map = _reaction_catalog(cfg.data_dir)

    # cfg.amax: if set, drop any reaction whose stoichiometry involves a nuclide
    # with mass number A = N + Z > amax.  Only meaningful for network="large";
    # silently no-ops for small/medium (their nuclides all have A ≤ 11).
    amax = getattr(cfg, "amax", None)

    parsed = []
    active_nuclides = {"n", "p"}
    weak_indices = {0}
    for name in selected:
        # Look up the custom filename if provided, otherwise default to name.txt
        filename = bare_to_file.get(name, name + ".txt")

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

        # amax filter: skip this reaction if any nuclide has A > amax.
        if amax is not None:
            all_nuclides = set(react) | set(prod)
            if any(sum(nuc_NZ[s]) > amax for s in all_nuclides if s in nuc_NZ):
                continue

        parsed.append((name, filename, react, prod, is_weak, net_lepton_dZ))
        active_nuclides.update(react)
        active_nuclides.update(prod)

    if era == "MT" and getattr(cfg, "network", None) != "small":
        # The MT era historically carries a fixed set of nuclides through the
        # solver even when only a subset of reactions is active.  For standard
        # networks this is all of SPECIES_MD (12 nuclides).  For custom
        # networks we only add SPECIES_MD members that actually appear in the
        # file's full reaction list (before the MT intersection), so we don't
        # carry columns for species that are completely absent from the network.
        file_nuclides: set[str] = {"n", "p"}
        for bn in bare_names:
            if bn in rxn_map:
                r, _ = _side_counts(rxn_map[bn][0])
                p, _ = _side_counts(rxn_map[bn][1])
                file_nuclides.update(r)
                file_nuclides.update(p)
        active_nuclides.update(s for s in SPECIES_MD if s in file_nuclides)

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

    # nTOp: net lepton dZ = -1 (Bm on products, Z_Bm = -1).
    # This balances the +1 nuclear dZ (n→p converts Z=0 to Z=1), making the
    # reaction electrically neutral to check_conservation.
    names = ["nTOp"]
    network = [({idx["n"]: 1}, {idx["p"]: 1})]
    lepton_dZ_list = [-1]   # index 0 = nTOp (electron emitted: dZ = -1)
    # Provenance label per reaction, read from each table's header (ref= field).
    # nTOp has no rate table here (its weak rates are supplied at solve time), so
    # we label it as the tabulated weak n<->p conversion.
    sources = ["weak n<->p"]
    # Rate-table path per reaction (aligned with ``names``).  nTOp has no rate
    # table (its weak rates are supplied at solve time), hence ``None``.
    files = [None]
    fwd_median, fwd_expsigma, abg = [], [], []
    for name, filename, react_names, prod_names, is_weak, net_lepton_dZ in parsed:
        names.append(name)
        if is_weak:
            weak_indices.add(len(names) - 1)
        lepton_dZ_list.append(net_lepton_dZ)
        table_path = os.path.join(tables_dir, filename)
        sources.append(_read_reaction_source(table_path))
        files.append(table_path)

        network.append((
            {idx[s]: c for s, c in react_names.items()},
            {idx[s]: c for s, c in prod_names.items()},
        ))

        data = np.loadtxt(os.path.join(tables_dir, filename), unpack=True)
        T9_src = data[0]
        # Resample from the file's own T9 grid to the master grid.  When all
        # tables share the same grid (the common case) this is nearly a no-op.
        fwd_median.append(_resample_rate_table(T9_src, data[1], grid))
        fwd_expsigma.append(_resample_rate_table(T9_src, data[2], grid))
        abg.append(list(db.get(name, (0.0, 0.0, 0.0))))

    fwd_median = np.asarray(fwd_median)
    fwd_expsigma = np.asarray(fwd_expsigma)
    abg = np.asarray(abg)

    # Apply QED corrections to radiative-capture rates when requested.
    # The correction (Pitrou & Pospelov 2020) accounts for pair-production in the
    # final-state photon.  Multiplying into fwd_median makes the corrected value
    # the new median so that p_* and NP_delta_* variations work relative to it.
    if getattr(cfg, "nuclear_qed_corrections", False):
        for i, rname in enumerate(names[1:]):   # names[0] is nTOp, handled separately
            factor = _qed_nuclear_rescale(rname, grid)
            if factor is not None:
                fwd_median[i] *= factor

    # Active forward rates (initially median)
    fwd = fwd_median.copy()

    # Build the reverse-rate cap: bwd is clamped so it never exceeds the value
    # it would have had at T_nucl (the nucleosynthesis onset temperature).
    # Below T_nucl the exp(γ/T9) factor of exothermic reverse rates can grow
    # by many orders of magnitude, producing a stiff "blow-up" for heavy nuclides
    # in the large network that would prevent BDF convergence.  Pinning the cap
    # at T_nucl preserves detailed balance near BBN onset and removes the low-T
    # divergence safely (see module docstring for the large-network caveats).
    j = int(np.searchsorted(grid, cfg.T_nucl / 1.0e9))  # index of T9 ≈ T_nucl/10⁹
    j = min(max(j, 0), grid.size - 1)
    a_, b_, g_ = abg[:, 0], abg[:, 1], abg[:, 2]
    T9c = grid[j]                              # T9 at the capping temperature
    bwd_cap = a_ * T9c ** b_ * np.exp(np.minimum(g_ / T9c, _EXP_CAP)) * fwd[:, j]

    return NetworkDefinition(species, N, Z, network, weak_indices, names, grid,
                             fwd, fwd_median, fwd_expsigma, abg, bwd_cap,
                             lepton_dZ=lepton_dZ_list, sources=sources,
                             files=files)


class UpdateNuclearRates:
    """Build era networks and temperature-dependent rate buffers.

    This class unifies all networks (small, medium, large) under a single
    architecture: each solver era (MT, LT) has a corresponding
    ``NetworkDefinition`` which manages stoichiometric network compilation and
    fast, vectorised rate-buffer filling.
    """

    def __init__(self, cfg):
        if cfg.verbose:
            print(f"[rates] Building {cfg.network!r} network from text lists.")

        self._selected_names = load_reaction_names(cfg, cfg.network)
        self._mt_net = load_network(cfg, era="MT", reaction_names=self._selected_names)
        self._lt_net = load_network(cfg, era="LT", reaction_names=self._selected_names)

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
            print(f"[rates] MT network: {len(self._mt_net.names)-1} reactions over "
                  f"{len(self._mt_net.species)} nuclides.")
            print(f"[rates] LT network: {len(self._lt_net.names)-1} reactions over "
                  f"{len(self._lt_net.species)} nuclides.")
            self.print_reactions()

    def describe_reactions(self):
        """Return the LT (full) network's reactions as
        ``(name, equation, source, file)`` tuples.

        Thin delegate to :meth:`NetworkDefinition.describe_reactions` for the LT
        network (the complete selected reaction set; the MT era only uses a fixed
        18-reaction subset).  The fourth element is the rate-table path (``None``
        for the weak ``nTOp`` entry).  Used by the verbose console listing and by
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
        j = net.names.index(rxn) - 1 # skip nTOp
        g = net.grid
        i = int(np.searchsorted(g, T9) - 1)
        i = min(max(i, 0), g.size - 2)
        w = (T9 - g[i]) / (g[i + 1] - g[i])
        return net._fwd[j, i] * (1.0 - w) + net._fwd[j, i + 1] * w
    frwrd.__name__ = f"{rxn}_frwrd"
    return frwrd


# We dynamically add the extractor methods to UpdateNuclearRates to support
# the existing output_time_evolution logic in PyPR.
for _rxn in _REACTIONS_MEDIUM:
    setattr(UpdateNuclearRates, f"{_rxn}_frwrd", _make_frwrd(_rxn))
