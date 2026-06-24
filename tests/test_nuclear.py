"""
Tests for the auto-derivation fallback in ``reaction_stoichiometry`` and the
duplicate-entry check in ``load_network`` (Item 4).

``reaction_stoichiometry`` normally locates the reactant/product split via the
detailed-balance exponent ``beta`` looked up in ``detailed_balance.csv``.  For a
reaction added directly to a network file that has no ``detailed_balance.csv``
entry yet, the literal ``"TO"`` token in the compact name marks the split
directly.  This fallback path is exercised here with synthetic reaction names
that are not present in ``detailed_balance.csv``:

1. ``ppTOdBp`` (p + p -> d + e+, the pp-fusion reaction): a balanced synthetic
   name.  The fallback must derive ``({"p": 2}, {"H2": 1, "Bp": 1})`` and pass
   the same A/Z conservation check as ``check_conservation``.
2. ``ppTOd`` (p + p -> d, missing the positron): an unbalanced synthetic name
   (charge mismatch).  The fallback must raise ``ValueError`` naming the
   reaction and the A/Z imbalance, instead of silently returning bad
   stoichiometry or raising a cryptic ``KeyError``.

``load_network`` also gained a check that rejects a network reaction list
containing the same entry twice (most likely a copy-paste mistake in a network
file), raising ``ValueError`` instead of silently dropping or double-counting
the repeat.
"""
import pytest

from primat.config import PRIMATConfig
from primat.network_data import load_network, reaction_stoichiometry


def test_auto_derived_stoichiometry_for_unknown_reaction():
    """A synthetic name absent from detailed_balance.csv is split at the
    literal "TO" token, with H2/H3/He4 aliases (d/t/a) resolved and Bp/Bm
    bookkeeping tokens kept as-is."""
    react, prod = reaction_stoichiometry("ppTOdBp")
    assert react == {"p": 2}
    assert prod == {"H2": 1, "Bp": 1}


def test_auto_derived_stoichiometry_conserves_A_and_Z():
    """The A/Z totals of the auto-derived reactants and products agree, using
    the same nuclide (N, Z) data as check_conservation: p+p (A=2, Z=2) vs
    d + e+ (A=2, Z=1+1=2)."""
    from primat.network_data import _reaction_catalog, _default_data_dir, _LEPTON_Z
    _, _, _, nuc_NZ, _, _ = _reaction_catalog(_default_data_dir())

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

    react, prod = reaction_stoichiometry("ppTOdBp")
    assert totals(react) == totals(prod) == (2, 2)


def test_unbalanced_synthetic_reaction_raises_value_error():
    """``ppTOd`` (p + p -> d) drops the positron, leaving Z unbalanced
    (reactants Z=2, products Z=1).  The fallback must raise ValueError naming
    the reaction and the imbalance, not silently return bad stoichiometry."""
    with pytest.raises(ValueError, match="ppTOd.*conserve"):
        reaction_stoichiometry("ppTOd")


def test_reaction_with_no_TO_token_raises_value_error():
    """A name with no detailed_balance.csv entry and no "TO" separator cannot
    be split into reactants/products at all."""
    with pytest.raises(ValueError, match="cannot be derived"):
        reaction_stoichiometry("ppdtg")


def test_duplicate_reaction_entry_raises_value_error():
    """A network reaction list containing the same entry twice raises
    ValueError naming the duplicated entry, instead of silently dropping or
    double-counting it."""
    cfg = PRIMATConfig({"network": "small", "verbose": False})
    with pytest.raises(ValueError, match="n_p__d_g.*already present"):
        load_network(cfg, era="LT",
                      reaction_names=["n_p__d_g", "n_p__d_g", "d_d__He3_n"])
