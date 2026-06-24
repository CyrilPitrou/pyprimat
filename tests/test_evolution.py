# -*- coding: utf-8 -*-
"""
test_evolution.py
==================
Unit tests for ``primat.evolution`` (the unified time-evolution schema,
``PRIMAT.md`` S7), independent of any actual BBN solve: round-trips a
synthetic ``EvolutionResult`` through :func:`dump_evolution`/
:func:`load_evolution`, both to a string and to a real file on disk.
"""
import numpy as np

from primat.evolution import EvolutionResult, dump_evolution, load_evolution


def _make_result():
    t = np.array([1.0, 2.0, 3.0])
    return EvolutionResult(
        t=t,
        a=np.array([0.1, 0.2, 0.3]),
        T_gamma=np.array([10.0, 5.0, 1.0]),
        T_nu={"e": np.array([9.0, 4.5, 0.9]),
              "mu": np.array([9.1, 4.6, 1.0]),
              "tau": np.array([9.2, 4.7, 1.1])},
        Y={"n": np.array([1.0, 0.9, 0.8]), "p": np.array([0.0, 0.1, 0.2])},
    )


def test_dump_evolution_header_and_round_trip(tmp_path):
    result = _make_result()
    path = tmp_path / "evolution.tsv"

    text = dump_evolution(result, path=str(path))

    header = text.splitlines()[0].split("\t")
    assert header == ["t_s", "a", "T_gamma_MeV", "T_nue_MeV", "T_numu_MeV",
                       "T_nutau_MeV", "Y_n", "Y_p"]
    assert path.exists()
    assert path.read_text() == text

    loaded = load_evolution(str(path))
    np.testing.assert_allclose(loaded.t, result.t)
    np.testing.assert_allclose(loaded.a, result.a)
    np.testing.assert_allclose(loaded.T_gamma, result.T_gamma)
    for flavour in ("e", "mu", "tau"):
        np.testing.assert_allclose(loaded.T_nu[flavour], result.T_nu[flavour])
    for species in ("n", "p"):
        np.testing.assert_allclose(loaded.Y[species], result.Y[species])


def test_dump_evolution_without_path_writes_nothing(tmp_path):
    result = _make_result()
    text = dump_evolution(result)
    assert isinstance(text, str)
    assert list(tmp_path.iterdir()) == []
