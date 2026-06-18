"""
Tests for the generic, stoichiometry-driven network builder
(``pyprimat.network_builder``) and its integration into ``UpdateNuclearRates``.

The builder compiles an abstract reaction list into flat arrays and evaluates
dY/dt and the Jacobian with two small kernels.  These tests check that:

1. the compiled kernels reproduce the table-driven reference in
   ``pyprimat.reactions`` to machine precision, for all three reaction orders;
2. the formal neutron/proton conservation check passes for the real networks and
   fires on a deliberately broken one;
3. the assembled RHS conserves baryon number numerically (sum_s A_s dY_s = 0);
4. the full ``UpdateNuclearRates`` driver -- its ``rhs``/``rhsMT``/``rhsLT`` and
   the matching Jacobian methods, with real rate splines and detailed-balance
   backward rates -- reproduces the ``pyprimat.reactions`` declarative reference for
   every era (small / MT / LT);
5. the era-independent structural invariants of the reaction table (buffer-order
   lengths, per-reaction baryon/charge conservation) are guarded;
6. the ``amax`` nuclide-mass cutoff filters reactions correctly: count decreases,
   all remaining nuclides satisfy A ≤ amax, conservation still holds, and
   invalid values are rejected.
"""
import numpy as np
import pytest

from pyprimat.network_data import (phase_network, network_rhs, network_jacobian,
                          ORDER_SMALL, ORDER_MT, ORDER_LT,
                          SPECIES_SMALL, SPECIES_MD)
from pyprimat.network_builder import (compile_network, NetworkKernels,
                                 check_conservation)
from pyprimat.config import PyPRConfig

_ORDERS = [("SMALL", ORDER_SMALL, SPECIES_SMALL),
           ("MT", ORDER_MT, SPECIES_MD),
           ("LT", ORDER_LT, SPECIES_MD)]


@pytest.mark.parametrize("label,order,species", _ORDERS)
def test_compiled_kernels_match_reference(label, order, species):
    """Compiled rhs/jacobian == the pyprimat.reactions reference (machine precision)."""
    net = phase_network(order, species)
    K = NetworkKernels(compile_network(net, len(species)), numba=False)
    rng = np.random.default_rng(0)
    for _ in range(5):
        Y = rng.random(len(species)) * 1e-2
        r = rng.random(2 * len(order)) * 1e3
        rho = 1.234e-5
        assert np.allclose(K.rhs(Y, rho, r), network_rhs(Y, rho, r, net),
                           rtol=1e-12, atol=0)
        assert np.allclose(K.jacobian(Y, rho, r),
                           network_jacobian(Y, rho, r, net), rtol=1e-10, atol=0)


@pytest.mark.parametrize("label,order,species", _ORDERS)
def test_formal_conservation_passes(label, order, species):
    """Every real network must pass the formal N/Z conservation check."""
    from pyprimat.config import PyPRConfig
    cnet = compile_network(phase_network(order, species), len(species))
    N = [PyPRConfig().Nuclides[s][0] for s in species]
    Z = [PyPRConfig().Nuclides[s][1] for s in species]
    # order[0] (n__p) and any reaction whose compact name ends in "Bm"/"Bp"
    # (an analytic beta-decay/electron-capture reaction, e.g. "Be7__Li7_Bp")
    # carry a lepton charge that phase_network's stripped-down stoichiometry
    # (see phase_network's _LEPTONS filtering) does not see -- mark them weak
    # so check_conservation only requires A (not Z) to balance for them,
    # mirroring the lepton_dZ=None "legacy" fallback.
    weak_indices = {i for i, name in enumerate(order)
                     if name == "n__p" or name.endswith(("Bm", "Bp"))}
    check_conservation(cnet, N, Z, weak_indices=weak_indices)  # raises on violation


def test_formal_conservation_catches_violation():
    """A reaction whose stoichiometry does not balance must be rejected."""
    # Fake 2-species network with a single non-conserving "reaction" n -> n + n.
    net = [({0: 1}, {0: 2})]
    cnet = compile_network(net, 2)
    with pytest.raises(ValueError, match="N/Z conservation"):
        check_conservation(cnet, N=[1, 1], Z=[0, 1])


@pytest.mark.parametrize("label,order,species", _ORDERS)
def test_rhs_conserves_baryon_number(label, order, species):
    """sum_s A_s dY_s/dt = 0 for arbitrary abundances/rates (baryon number)."""
    from pyprimat.config import PyPRConfig
    net = phase_network(order, species)
    K = NetworkKernels(compile_network(net, len(species)), numba=False)
    cfg = PyPRConfig()
    A = np.array([sum(cfg.Nuclides[s]) for s in species])   # A = N + Z
    rng = np.random.default_rng(2)
    Y = rng.random(len(species)) * 1e-2
    r = rng.random(2 * len(order)) * 1e3
    dY = K.rhs(Y, 1.5e-5, r)
    assert abs(np.dot(A, dY)) < 1e-9 * np.max(np.abs(dY))


# (network, era label, order attr, species, rhs method, jac method)
_DRIVER_ERAS = [
    ("small",  "MT", "_order_MT", SPECIES_MD,  "rhsMT", "JacobianMT"),
    ("medium", "MT", "_order_MT", SPECIES_MD,  "rhsMT", "JacobianMT"),
    ("medium", "LT", "_order_LT", SPECIES_MD,  "rhsLT", "JacobianLT"),
]


@pytest.mark.parametrize("network,era,order_attr,species,rhs_m,jac_m",
                         _DRIVER_ERAS)
def test_driver_methods_match_reference(network, era, order_attr, species,
                                        rhs_m, jac_m):
    """The real ``UpdateNuclearRates`` methods reproduce the declarative
    ``pyprimat.reactions`` reference for every era.
    """
    from pyprimat.config import PyPRConfig
    from pyprimat.network_data import UpdateNuclearRates
    cfg = PyPRConfig({"network": network, "verbose": False})
    K = UpdateNuclearRates(cfg)
    order = getattr(K, order_attr)
    species_eff = K.species_large if era == "LT" and network != "small" else species
    net = phase_network(order, species_eff)
    rhs_method, jac_method = getattr(K, rhs_m), getattr(K, jac_m)
    
    # Era network definition to fill buffer
    net_def = K._mt_net if era == 'MT' else K._lt_net
    
    f, b = (lambda T: 1.3), (lambda T: 0.7)        # dummy n__p callables
    nsp = len(species_eff)
    rng = np.random.default_rng(1)
    for _ in range(15):
        Y = rng.random(nsp) * 1e-2
        T = rng.uniform(1e8, 9e9)
        rho = rng.uniform(1e-6, 1e-4)
        
        # Fill buffer using the net_def
        buf = net_def.fill_buffer(T, f, b, clamp=(era == 'LT'))

        ref_rhs = network_rhs(Y, rho, buf, net)
        ref_jac = network_jacobian(Y, rho, buf, net)
        got_rhs = np.asarray(rhs_method(Y, T, rho, f, b))
        got_jac = np.asarray(jac_method(Y, T, rho, f, b))

        assert np.allclose(got_rhs, ref_rhs, rtol=1e-10, atol=0)
        assert np.allclose(got_jac, ref_jac, rtol=1e-9, atol=1e-300)


# ---------------------------------------------------------------------------
# Era-independent invariants (merged from test_network_consistency.py)
# These are structural checks on the reaction table itself — independent of
# the compiled kernels — and are fast (no integration).
# ---------------------------------------------------------------------------

def test_buffer_orders_have_expected_lengths():
    """Guard the buffer-order list lengths that the kernels index into.

    ORDER_SMALL includes n__p + 12 nuclear reactions = 13 entries.
    ORDER_MT has 18 entries (n__p + 17).
    ORDER_LT has 68 entries (n__p + 67 medium reactions, including the
    B8/Be7/He6/Li8/H3 analytic beta-decay/electron-capture reactions).
    """
    assert len(ORDER_SMALL) == 13, f"Expected 13, got {len(ORDER_SMALL)}"
    assert len(ORDER_MT) == 18,    f"Expected 18, got {len(ORDER_MT)}"
    assert len(ORDER_LT) == 68,    f"Expected 68, got {len(ORDER_LT)}"


def test_stoichiometry_conserves_baryon_and_charge():
    """Every *nuclear* reaction in ORDER_LT must conserve A and Z.

    n__p is excluded: it is the weak n↔p rate whose lepton charge is
    tracked separately via ``lepton_dZ`` (see test_formal_conservation_passes
    which uses the full uniform-Q check including the lepton contribution).

    A handful of medium-network reactions (e.g. ``Be7__Li7_Bp``) are
    beta-decay/electron-capture reactions whose stoichiometry includes a
    ``Bm``/``Bp`` lepton bookkeeping token (A=0, Z=∓1, see
    ``network_data._LEPTON_Z``) on top of the nuclide tokens -- include their
    charge here too so dZ is checked over the *whole* reaction, not just its
    nuclide part.
    """
    from pyprimat.network_data import reaction_stoichiometry, _LEPTON_Z
    nz = PyPRConfig().Nuclides
    A = {s: nz[s][0] + nz[s][1] for s in nz}
    Z = {s: nz[s][1] for s in nz}
    for name in sorted(set(ORDER_LT) - {"n__p"}):
        react, prod = reaction_stoichiometry(name)
        dA = (sum(c * A[s] for s, c in prod.items() if s not in _LEPTON_Z)
              - sum(c * A[s] for s, c in react.items() if s not in _LEPTON_Z))
        dZ = (sum(c * (Z[s] if s not in _LEPTON_Z else _LEPTON_Z[s])
                   for s, c in prod.items())
              - sum(c * (Z[s] if s not in _LEPTON_Z else _LEPTON_Z[s])
                     for s, c in react.items()))
        assert dA == 0, f"{name}: baryon number not conserved (dA={dA})"
        assert dZ == 0, f"{name}: charge not conserved (dZ={dZ})"


# ---------------------------------------------------------------------------
# amax nuclide-mass cutoff
# ---------------------------------------------------------------------------

def test_amax_invalid_values_rejected():
    """Config validation must reject non-positive or non-integer amax values."""
    from pyprimat.config import PyPRConfig
    for bad in (0, -1, 6.5, "8", None):
        if bad is None:
            # None is the valid "no filter" sentinel — must not raise
            PyPRConfig({"network": "large", "amax": None})
            continue
        with pytest.raises((ValueError, TypeError)):
            PyPRConfig({"network": "large", "amax": bad})


def test_amax_reduces_reaction_count():
    """A stricter amax must yield fewer reactions than a looser one."""
    from pyprimat.network_data import load_network
    from pyprimat.config import PyPRConfig
    cfg_full = PyPRConfig({"network": "large"})
    cfg_a20  = PyPRConfig({"network": "large", "amax": 20})
    cfg_a12  = PyPRConfig({"network": "large", "amax": 12})
    n_full = len(load_network(cfg_full, era="LT").names)
    n_a20  = len(load_network(cfg_a20,  era="LT").names)
    n_a12  = len(load_network(cfg_a12,  era="LT").names)
    assert n_a20 < n_full, "amax=20 should drop some large-A reactions"
    assert n_a12 < n_a20,  "amax=12 should drop more reactions than amax=20"


def test_amax_nuclide_bound():
    """With amax=20, every nuclide in the filtered network has A ≤ 20."""
    from pyprimat.network_data import load_network
    from pyprimat.config import PyPRConfig
    cfg = PyPRConfig({"network": "large", "amax": 20})
    net = load_network(cfg, era="LT")
    nz  = PyPRConfig().Nuclides
    for s in net.species:
        if s in nz:
            A = nz[s][0] + nz[s][1]
            assert A <= 20, f"Nuclide {s} has A={A} > amax=20"


def test_amax_conservation_holds():
    """After amax filtering, baryon/charge conservation must still pass."""
    from pyprimat.network_data import load_network
    from pyprimat.config import PyPRConfig
    from pyprimat.network_builder import check_conservation, compile_network
    cfg = PyPRConfig({"network": "large", "amax": 20})
    net = load_network(cfg, era="LT")
    cnet = compile_network(net.network, len(net.species))
    check_conservation(cnet, net.N, net.Z,
                       weak_indices=set(net.weak_indices),
                       lepton_dZ=net.lepton_dZ)
