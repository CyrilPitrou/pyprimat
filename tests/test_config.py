"""Tests for PyPRConfig: defaults, overrides, derived quantities, nuclide data."""
import warnings
import pytest
from pyprimat.config import PyPRConfig, DEFAULT_PARAMS


def test_default_construction():
    cfg = PyPRConfig()
    assert cfg.Omegabh2 > 0
    assert cfg.is_small is True
    assert cfg.numerical_precision > 0


def test_user_override():
    cfg = PyPRConfig({"Omegabh2": 0.020})
    assert cfg.Omegabh2 == pytest.approx(0.020)


def test_unknown_key_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PyPRConfig({"not_a_real_param": 42})
    assert any("not_a_real_param" in str(x.message) for x in w)


def test_unknown_key_does_not_raise():
    PyPRConfig({"totally_unknown": 99})


def test_omegabh2_to_eta0b_positive():
    cfg = PyPRConfig()
    assert cfg.Omegabh2_to_eta0b > 0


def test_nuclides_keys():
    cfg = PyPRConfig()
    expected_subset = {"n", "p", "H2", "H3", "He3", "He4", "He6",
                       "Li6", "Li7", "Be7", "Li8", "B8"}
    assert expected_subset.issubset(set(cfg.Nuclides.keys()))


def test_nuclides_NZ_values():
    cfg = PyPRConfig()
    assert cfg.Nuclides["He4"] == [2, 2]
    assert cfg.Nuclides["H2"]  == [1, 1]
    assert cfg.Nuclides["Li7"] == [4, 3]
    assert cfg.Nuclides["n"]   == [1, 0]
    assert cfg.Nuclides["p"]   == [0, 1]


def test_p_rate_keys_count():
    """``p_rxn``/``NP_delta_rxn`` carry one MCMC weight per *configured*
    network's reaction (small/medium/large), not always the medium set --
    see the "Corrected bug on MC" fix in ``PyPRConfig.__init__``, which reads
    ``load_reaction_names(self.data_dir, self.network)`` rather than the
    hardcoded ``_REACTIONS_MEDIUM`` list."""
    import re
    from pyprimat.network_data import _REACTIONS_MEDIUM, load_reaction_names

    cfg = PyPRConfig()  # default network="small" -> 12 reactions
    assert cfg.network == "small"
    assert len(cfg.p_rxn) == 12

    cfg_medium = PyPRConfig({"network": "medium"})
    bare_names = [re.split(r'[, ]+', entry, maxsplit=1)[0]
                   for entry in load_reaction_names(cfg_medium, "medium")]
    assert len(cfg_medium.p_rxn) == len(bare_names) == len(_REACTIONS_MEDIUM) == 67


def test_physical_constants_positive():
    cfg = PyPRConfig()
    for attr in ("me", "mn", "mp", "Mpl", "kB", "MeV"):
        assert getattr(cfg, attr) > 0, f"cfg.{attr} should be positive"


def test_config_dynamic_rate_attrs():
    """Dynamic ``p_*`` and ``NP_delta_*`` attrs round-trip through the backing dicts."""
    cfg = PyPRConfig()

    # p_<reaction> attribute routes to cfg.p_rxn dict
    cfg.p_npTOdg = 0.5
    assert cfg.p_rxn["npTOdg"] == 0.5
    assert cfg.p_npTOdg == 0.5

    # NP_delta_<reaction> routes to cfg.NP_delta_rxn dict
    cfg.NP_delta_dpTOHe3g = 0.1
    assert cfg.NP_delta_rxn["dpTOHe3g"] == 0.1
    assert cfg.NP_delta_dpTOHe3g == 0.1

    # Unknown prefix falls through to object.__setattr__
    cfg.some_random_param = 42
    assert cfg.some_random_param == 42
