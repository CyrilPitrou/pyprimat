# -*- coding: utf-8 -*-
"""
nuclide_table.py
================
Offline helpers (used only by the rate/network *generation* command, never at
PyPRIMAT run time) that turn the reaction list extracted from AC2024 +
PRIMAT-main.m into:

  1. the **set of nuclides** the network touches, each resolved to its
     (N, Z, A, charge Q, mass excess, spin), and
  2. the **detailed-balance coefficients** (alpha, beta, gamma) of every
     reaction that has a reverse rate.

Why offline.  These quantities never change for a fixed PRIMAT version, and the
NUBASE2020 table is ~760 kB, so we resolve everything once here and bake the
result into small CSV files that PyPRIMAT simply reads at start-up
(``nuclides.csv``, ``detailed_balance.csv``).

Token convention.  Reaction sides come from the AC2024/PRIMAT sources as token
lists that mix spellings: ``a``/``He4``, ``d``/``H2``, ``t``/``H3``, the bare
nucleons ``n``/``p``, ordinary nuclides ``Be9``/``C12``/..., the photon ``g``
(or ``2g``), and the beta-decay leptons ``Bm`` (electron, e^-) and ``Bp``
(positron, e^+).  :func:`resolve_token` maps any of these to a canonical record;
nuclides are keyed by a canonical name (``n``, ``p``, ``H2``, ``H3``, ``He3``,
``He4``, ``Be9``, ...) chosen to match PyPRIMAT's existing ``Nuclides`` keys.
"""
import re
from collections import Counter

# Element symbol -> atomic number Z.  The BBN+ network reaches Na (Z=11); we
# list through Ca (Z=20) so the table comfortably covers any token that appears.
_ELEMENT_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
}
_Z_ELEMENT = {z: sym for sym, z in _ELEMENT_Z.items()}

# Short single-letter aliases used in the sources, mapped to (Z, A).
_SHORT = {"n": (0, 1), "p": (1, 1), "d": (1, 2), "t": (1, 3), "a": (2, 4)}


class Token:
    """Resolved reaction token.

    kind : 'nuclide' | 'photon' | 'lepton'
    Z, A : atomic number and mass number (0 for photon/lepton).
    Q    : electric charge in units of e (Z for nuclei; Bm=-1, Bp=+1; 0 otherwise).
    name : canonical nuclide name (None for photon/lepton).
    """
    __slots__ = ("kind", "Z", "A", "Q", "name")

    def __init__(self, kind, Z, A, Q, name):
        self.kind, self.Z, self.A, self.Q, self.name = kind, Z, A, Q, name


def canonical_name(Z, A):
    """Canonical nuclide name from (Z, A), matching PyPRIMAT's ``Nuclides`` keys.

    The bare nucleons are special-cased to ``n``/``p``; everything else is the
    element symbol followed by the mass number (``H2``, ``H3``, ``He4``,
    ``Be9``, ``C12``, ...).  This deliberately yields ``H2`` (not ``d``) so the
    generated ``nuclides.csv`` lines up with the names PyPRIMAT already uses.
    """
    if (Z, A) == (0, 1):
        return "n"
    if (Z, A) == (1, 1):
        return "p"
    return f"{_Z_ELEMENT[Z]}{A}"


def resolve_token(tok):
    """Resolve one reaction token (e.g. ``'a'``, ``'He4'``, ``'Be9'``, ``'Bm'``,
    ``'g'``) to a :class:`Token`.

    Photons (``g``) carry no baryon number or charge; the beta leptons ``Bm``
    (e^-) and ``Bp`` (e^+) carry charge -1 / +1 and A=0 -- both are needed for
    the formal charge/baryon conservation check but are *not* tracked species.
    """
    if tok in ("g", "gamma"):
        return Token("photon", 0, 0, 0, None)
    if tok == "Bm":                       # beta-minus: emits an electron
        return Token("lepton", 0, 0, -1, None)
    if tok == "Bp":                       # beta-plus: emits a positron
        return Token("lepton", 0, 0, +1, None)
    if tok in _SHORT:
        Z, A = _SHORT[tok]
        return Token("nuclide", Z, A, Z, canonical_name(Z, A))
    m = re.fullmatch(r"([A-Z][a-z]?)(\d+)", tok)
    if not m:
        raise ValueError(f"cannot resolve reaction token {tok!r}")
    sym, A = m.group(1), int(m.group(2))
    if sym not in _ELEMENT_Z:
        raise ValueError(f"unknown element symbol {sym!r} in token {tok!r}")
    Z = _ELEMENT_Z[sym]
    return Token("nuclide", Z, A, Z, canonical_name(Z, A))


# ---------------------------------------------------------------------------
# NUBASE2020 reader (general: keyed by (Z, A), not a fixed nuclide list)
# ---------------------------------------------------------------------------
def _parse_spin(jpi_field):
    """Leading J (integer or fraction) of a NUBASE ``Jpi`` field (``3/2-*`` -> 1.5)."""
    m = re.match(r"\s*\(?(\d+)(?:/(\d+))?", jpi_field)
    if m is None:
        return None
    return float(m.group(1)) / float(m.group(2)) if m.group(2) else float(m.group(1))


def load_nubase_all(nubase_path):
    """Read mass excesses and spins of *every* ground state in a NUBASE2020 file.

    Fixed-width columns (0-indexed): A = ``[0:3]``, Z = ``[4:7]``, isomer index
    ``[7]`` (``0`` = ground state), mass excess [keV] = ``[18:31]``, Jpi =
    ``[88:102]``.  Estimated values carry a trailing ``#`` we strip.

    Returns ``{(Z, A): (mass_excess_keV, spin_J)}``.
    """
    table = {}
    with open(nubase_path, encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("#") or len(line) < 102:
                continue
            if line[7] != "0":                      # keep ground states only
                continue
            try:
                A = int(line[0:3])
                Z = int(line[4:7])
            except ValueError:
                continue
            excess = float(line[18:31].replace("#", ""))
            table[(Z, A)] = (excess, _parse_spin(line[88:102]))
    return table


def build_nuclide_table(reactions, nubase_path):
    """Deduce the nuclide set from the reaction list and attach NUBASE properties.

    ``reactions`` is a list of dicts with ``reactants``/``products`` token
    lists.  Photons and leptons are dropped; every remaining distinct nuclide is
    resolved to (N, Z, A, Q) and matched to its NUBASE mass excess [keV] and
    ground-state spin.

    Returns an ``Ordered‑ish`` dict ``name -> record`` where record is
    ``dict(name, N, Z, A, Q, excess_keV, spin)``, ordered by increasing (Z, A)
    so the file is deterministic and human-readable.
    """
    nubase = load_nubase_all(nubase_path)
    seen = {}
    for rxn in reactions:
        for tok in rxn["reactants"] + rxn["products"]:
            t = resolve_token(tok)
            if t.kind != "nuclide":
                continue
            if t.name in seen:
                continue
            if (t.Z, t.A) not in nubase:
                raise ValueError(
                    f"nuclide {t.name} (Z={t.Z}, A={t.A}) not found in NUBASE "
                    f"file {nubase_path}")
            excess, spin = nubase[(t.Z, t.A)]
            seen[t.name] = dict(name=t.name, N=t.A - t.Z, Z=t.Z, A=t.A, Q=t.Z,
                                excess_keV=excess, spin=spin)
    # Deterministic order: by (Z, A), so n, p, H2, H3, He3, He4, ... come first.
    return {rec["name"]: rec
            for rec in sorted(seen.values(), key=lambda r: (r["Z"], r["A"]))}


# ---------------------------------------------------------------------------
# Formal conservation check (baryon number A and electric charge Q)
# ---------------------------------------------------------------------------
def conservation_residual(reactants, products):
    """Return ``(dA, dQ)`` = products-minus-reactants of (baryon number, charge).

    Both must be 0 for a physical reaction.  Photons contribute (0, 0); the beta
    leptons contribute (0, -1) for ``Bm`` and (0, +1) for ``Bp``.  This is a
    *formal* (exact integer) check -- no floating point involved.
    """
    def totals(side):
        A = Q = 0
        for tok in side:
            t = resolve_token(tok)
            A += t.A
            Q += t.Q
        return A, Q
    Ar, Qr = totals(reactants)
    Ap, Qp = totals(products)
    return Ap - Ar, Qp - Qr


# ---------------------------------------------------------------------------
# Detailed balance, reusing PyPRIMAT's validated physics
# ---------------------------------------------------------------------------
class _DBConfig:
    """Minimal stand-in for ``PyPRConfig`` exposing exactly what
    :func:`pypr.nuclear_data.detailed_balance` reads: the nuclide property dicts
    (built here for the *whole* large network) and the fundamental constants
    (copied verbatim from a real ``PyPRConfig``).  This lets the offline
    generator reuse the same, already-validated detailed-balance code that
    PyPRIMAT uses for its 62 reactions, but over an arbitrary nuclide set."""

    def __init__(self, nuclide_table):
        from pypr.config import PyPRConfig
        base = PyPRConfig()
        for k in ("keV", "kB", "MeV", "ma", "me", "clight", "hbar"):
            setattr(self, k, getattr(base, k))
        self.Nuclides = {n: [r["N"], r["Z"]] for n, r in nuclide_table.items()}
        self.NuclExcessMass = {n: r["excess_keV"] for n, r in nuclide_table.items()}
        self.NuclSpin = {n: r["spin"] for n, r in nuclide_table.items()}


def make_detailed_balance(nuclide_table):
    """Return ``db(reactants, products) -> (Q_keV, alpha, beta, gamma)``.

    ``reactants``/``products`` are token lists (any spelling); photons and
    leptons are dropped, the rest canonicalised, and the result handed to
    PyPRIMAT's :func:`pypr.nuclear_data.detailed_balance`.  ``Q_keV`` is the
    energy released (positive = exothermic).  Reactions that emit a lepton
    (decays) have no reverse rate and must not be passed here.
    """
    from generate_from_primat.nuclear_data import detailed_balance
    cfg = _DBConfig(nuclide_table)

    def to_canonical(side):
        out = []
        for tok in side:
            t = resolve_token(tok)
            if t.kind == "nuclide":           # drop photons (no mass/spin)
                out.append(t.name)
        return out

    def db(reactants, products):
        rc, pc = to_canonical(reactants), to_canonical(products)
        alpha, beta, gamma = detailed_balance(rc, pc, cfg)
        # gamma = -Q/(kB*1e9 K); recover Q in keV for the stored table.
        Q_keV = -gamma * cfg.kB * 1e9 / cfg.keV
        return Q_keV, alpha, beta, gamma

    return db


def is_decay(reactants, products):
    """True if the reaction emits a beta lepton (``Bm``/``Bp``) -- i.e. it is a
    weak decay with no reverse rate, so it has no detailed-balance coefficients."""
    return any(tok in ("Bm", "Bp") for tok in reactants + products)
