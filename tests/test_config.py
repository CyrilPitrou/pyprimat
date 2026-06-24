"""Tests for PRIMATConfig: defaults, overrides, derived quantities, nuclide data."""
import warnings
import pytest
from primat.config import PRIMATConfig, DEFAULT_PARAMS


def test_default_construction():
    cfg = PRIMATConfig()
    assert cfg.Omegabh2 > 0
    assert cfg.is_small is True
    assert cfg.numerical_precision > 0


def test_user_override():
    cfg = PRIMATConfig({"Omegabh2": 0.020})
    assert cfg.Omegabh2 == pytest.approx(0.020)


def test_unknown_key_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PRIMATConfig({"not_a_real_param": 42})
    assert any("not_a_real_param" in str(x.message) for x in w)


def test_unknown_key_does_not_raise():
    PRIMATConfig({"totally_unknown": 99})


def test_omegabh2_to_eta0b_positive():
    cfg = PRIMATConfig()
    assert cfg.Omegabh2_to_eta0b > 0


def test_nuclides_keys():
    cfg = PRIMATConfig()
    expected_subset = {"n", "p", "H2", "H3", "He3", "He4", "He6",
                       "Li6", "Li7", "Be7", "Li8", "B8"}
    assert expected_subset.issubset(set(cfg.Nuclides.keys()))


def test_p_rxn_typo_warns():
    """A p_<rxn> override whose reaction name isn't in the network must warn.

    Before this check existed, a typo'd reaction name (e.g. a stray
    underscore, or a name from a different network) was silently accepted: it
    became a no-op dict entry in cfg.p_rxn with no signal that the rate
    variation the caller asked for was never actually applied.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PRIMATConfig({"p_not_a_real_reaction": 0.5})
    assert any("p_not_a_real_reaction" in str(x.message) for x in w)


def test_NP_delta_rxn_typo_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PRIMATConfig({"NP_delta_not_a_real_reaction": 0.5})
    assert any("NP_delta_not_a_real_reaction" in str(x.message) for x in w)


def test_p_rxn_valid_reaction_does_not_warn():
    """A genuine reaction name (from the small network) must not warn."""
    cfg = PRIMATConfig()
    rxn = next(iter(cfg.p_rxn))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg2 = PRIMATConfig({f"p_{rxn}": 0.3})
    assert not any(rxn in str(x.message) for x in w)
    assert getattr(cfg2, f"p_{rxn}") == pytest.approx(0.3)


def test_nuclides_NZ_values():
    cfg = PRIMATConfig()
    assert cfg.Nuclides["He4"] == [2, 2]
    assert cfg.Nuclides["H2"]  == [1, 1]
    assert cfg.Nuclides["Li7"] == [4, 3]
    assert cfg.Nuclides["n"]   == [1, 0]
    assert cfg.Nuclides["p"]   == [0, 1]


def test_p_rate_keys_count():
    """``p_rxn``/``NP_delta_rxn`` carry one MCMC weight per *configured*
    network's reaction (small/large±amax), not always the full large set --
    see the "Corrected bug on MC" fix in ``PRIMATConfig.__init__``, which reads
    ``load_reaction_names(self.data_dir, self.network)`` rather than the
    hardcoded ``_REACTIONS_LARGE`` list."""
    from primat.network_data import load_network

    cfg = PRIMATConfig()  # default network="small" -> 12 reactions
    assert cfg.network == "small"
    assert len(cfg.p_rxn) == 12

    cfg_amax8 = PRIMATConfig({"network": "large", "amax": 8})
    net_amax8 = load_network(cfg_amax8, era="LT")
    n_thermonuclear = len(net_amax8.names) - 1  # exclude the prepended n__p
    assert len(cfg_amax8.p_rxn) == n_thermonuclear == 67


def test_physical_constants_positive():
    cfg = PRIMATConfig()
    for attr in ("me", "mn", "mp", "Mpl", "kB", "MeV"):
        assert getattr(cfg, attr) > 0, f"cfg.{attr} should be positive"


def test_config_dynamic_rate_attrs():
    """Dynamic ``p_*`` and ``NP_delta_*`` attrs round-trip through the backing dicts."""
    cfg = PRIMATConfig()

    # p_<reaction> attribute routes to cfg.p_rxn dict
    cfg.p_n_p__d_g = 0.5
    assert cfg.p_rxn["n_p__d_g"] == 0.5
    assert cfg.p_n_p__d_g == 0.5

    # NP_delta_<reaction> routes to cfg.NP_delta_rxn dict
    cfg.NP_delta_d_p__He3_g = 0.1
    assert cfg.NP_delta_rxn["d_p__He3_g"] == 0.1
    assert cfg.NP_delta_d_p__He3_g == 0.1

    # Unknown prefix falls through to object.__setattr__
    cfg.some_random_param = 42
    assert cfg.some_random_param == 42
