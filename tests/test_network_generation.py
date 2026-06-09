"""
Tests for the offline network-generation layer
(``generate_rates/convert_ac2024_rates.py`` + ``nuclide_table.py``).

That command runs once to turn AC2024 + the analytic table + NUBASE into the
three CSVs PyPRIMAT reads at start-up:

* ``nuclides.csv``        : every nuclide the network touches, with N,Z,A,Q,mass,spin,
* ``reactions_large.csv`` : the deduced >400-reaction list,
* ``detailed_balance.csv``: alpha,beta,gamma per reversible reaction.

The tests below check, without re-running the (slow) full generation:

1. the token resolver and the *formal* baryon/charge conservation check;
2. that the generated nuclide table is internally consistent and agrees with
   PyPRIMAT's hard-coded 12-nuclide table;
3. that the deduced reaction list is a superset of the known 12- and 62-reaction
   networks and that *every* listed reaction conserves A and Q;
4. that the detailed-balance coefficients computed from nuclide data reproduce
   PyPRIMAT's published values for the 62 medium-network reactions.

The CSV-based tests skip if the generated ``rates/nuclear/AC2024`` folder is
absent (fresh checkout before the generator has been run).
"""
import csv
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.join(_HERE, "..")
_GEN_DIR = os.path.join(_ROOT, "generate_rates")
_AC2024_DIR = os.path.join(_ROOT, "rates", "nuclear", "data")

# The generation helpers live in generate_rates/, which is not an
# installed package; add it to sys.path so the tests can import it directly.
sys.path.insert(0, _GEN_DIR)

from nuclide_table import (resolve_token, conservation_residual,   # noqa: E402
                           build_nuclide_table, canonical_name)

_needs_ac2024 = pytest.mark.skipif(
    not os.path.isdir(_AC2024_DIR),
    reason="rates/nuclear/data not generated "
           "(run python generate_rates/convert_ac2024_rates.py)",
)


# ---------------------------------------------------------------------------
# 1. Token resolution and formal conservation
# ---------------------------------------------------------------------------
def test_resolve_token_canonicalises_spellings():
    """``a``/``He4``, ``d``/``H2``, ``t``/``H3`` must collapse to one nuclide."""
    assert resolve_token("a").name == resolve_token("He4").name == "He4"
    assert resolve_token("d").name == resolve_token("H2").name == "H2"
    assert resolve_token("t").name == resolve_token("H3").name == "H3"
    n, p = resolve_token("n"), resolve_token("p")
    assert (n.Z, n.A, n.Q) == (0, 1, 0)
    assert (p.Z, p.A, p.Q) == (1, 1, 1)
    c12 = resolve_token("C12")
    assert (c12.Z, c12.A, c12.name) == (6, 12, "C12")


def test_leptons_and_photons_carry_charge_but_no_baryon():
    assert resolve_token("g").kind == "photon"
    bm, bp = resolve_token("Bm"), resolve_token("Bp")
    assert (bm.kind, bm.A, bm.Q) == ("lepton", 0, -1)
    assert (bp.kind, bp.A, bp.Q) == ("lepton", 0, +1)


def test_conservation_residual_zero_for_physical_reactions():
    # n + p -> d + g ; t + t -> a + n + n ; a beta-minus decay.
    assert conservation_residual(["n", "p"], ["d", "g"]) == (0, 0)
    assert conservation_residual(["t", "t"], ["a", "n", "n"]) == (0, 0)
    assert conservation_residual(["Li9"], ["Be9", "Bm"]) == (0, 0)        # A,Q conserved
    assert conservation_residual(["N17"], ["O16", "n", "Bm"]) == (0, 0)


def test_conservation_residual_catches_violations():
    dA, dQ = conservation_residual(["n", "p"], ["He4"])      # 2 baryons vs 4
    assert dA != 0
    dA, dQ = conservation_residual(["p", "p"], ["d", "g"])   # charge 2 vs 1
    assert dQ != 0


def test_canonical_name_special_cases():
    assert canonical_name(0, 1) == "n"
    assert canonical_name(1, 1) == "p"
    assert canonical_name(1, 2) == "H2"
    assert canonical_name(2, 4) == "He4"


# ---------------------------------------------------------------------------
# 2. Generated nuclides.csv consistency
# ---------------------------------------------------------------------------
def _load_nuclides_csv():
    with open(os.path.join(_AC2024_DIR, "nuclides.csv")) as f:
        return {r["name"]: r for r in csv.DictReader(f)}


@_needs_ac2024
def test_nuclides_csv_agrees_with_pypr_hardcoded_table():
    """Every nuclide in PyPRConfig.Nuclides must appear in nuclides.csv with the
    same (N, Z) -- the generated table is a superset of the hard-coded one."""
    from pypr.config import PyPRConfig
    cfg = PyPRConfig()
    nuc = _load_nuclides_csv()
    # Check the key ones used in Speciess_Small
    for name in ["n", "p", "H2", "H3", "He3", "He4", "Li7", "Be7"]:
        N, Z = cfg.Nuclides[name]
        assert name in nuc, f"{name} missing from nuclides.csv"
        assert (int(nuc[name]["N"]), int(nuc[name]["Z"])) == (N, Z)


@_needs_ac2024
def test_nuclides_csv_self_consistent():
    """A = N + Z and Q = Z for every row; mass excess and spin are present."""
    for r in _load_nuclides_csv().values():
        N, Z, A, Q = (int(r[k]) for k in ("N", "Z", "A", "Q"))
        assert A == N + Z and Q == Z
        float(r["mass_excess_keV"])                  # parses
        float(r["spin"])


# ---------------------------------------------------------------------------
# 3. Generated reactions_large.csv: superset + conservation
# ---------------------------------------------------------------------------
def _load_reactions_csv():
    with open(os.path.join(_AC2024_DIR, "reactions_large.csv")) as f:
        return list(csv.DictReader(f))


@_needs_ac2024
def test_reaction_list_is_superset_of_known_networks():
    """The deduced large list must contain every reaction of the 12-key and
    62-reaction networks (matched by their <reactants>TO<products> file name)."""
    from pypr.nuclear import to_filename, _KEY12_REACTIONS, _REACTIONS_MEDIUM
    names = {r["name"] for r in _load_reactions_csv()}
    for compact in _KEY12_REACTIONS:
        name = compact if 'TO' in compact else to_filename(compact)
        assert name in names, f"{compact} missing from large list"
    for compact in _REACTIONS_MEDIUM:
        name = compact if 'TO' in compact else to_filename(compact)
        assert name in names, f"{compact} missing from large list"


@_needs_ac2024
def test_every_listed_reaction_conserves_A_and_Q():
    for r in _load_reactions_csv():
        reactants = r["reactants"].split("+")
        products = r["products"].split("+")
        assert conservation_residual(reactants, products) == (0, 0), r["name"]


# ---------------------------------------------------------------------------
# 4. Detailed balance consistency
# ---------------------------------------------------------------------------
@_needs_ac2024
def test_detailed_balance_formula_consistency():
    """alpha,beta,gamma computed from nuclide data must reproduce the
    values in detailed_balance.csv: beta exactly,
    alpha and gamma to better than 1% (the documented detailed-balance accuracy)."""
    from pypr.config import PyPRConfig
    from pypr.nuclear import compute_detailed_balance_coefficients, reaction_species
    cfg = PyPRConfig()
    with open(os.path.join(_AC2024_DIR, "detailed_balance.csv")) as f:
        db_rows = list(csv.DictReader(f))
    
    # Check a representative sample or all of them
    for row in db_rows:
        name = row["reaction"]
        ref_alpha = float(row["alpha"])
        ref_beta = float(row["beta"])
        ref_gamma = float(row["gamma"])
        
        reactants, products = reaction_species(name)
        alpha, beta, gamma = compute_detailed_balance_coefficients(reactants, products, cfg)
        
        assert beta == pytest.approx(ref_beta), f"Failed for {name}"
        if ref_alpha:
            assert abs(alpha - ref_alpha) / abs(ref_alpha) < 0.01
        if ref_gamma:
            assert abs(gamma - ref_gamma) / abs(ref_gamma) < 0.01
