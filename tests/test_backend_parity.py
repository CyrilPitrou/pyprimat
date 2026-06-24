"""
Backend parity: primat._primat_c (C) vs. primat.main.PRIMAT (Python).

Why this test exists
---------------------
CLAUDE.md's "Keeping primat-c and primat in sync" section mandates that any
change to the physics/numerics of one backend be mirrored in the other, and
that "the two backends must also agree on the *shape* of their outputs
(same result-dict keys, ...) so callers can switch backends transparently."
This file is that check: it pins down (1) the result-dict *shape* exactly,
and (2) the numerical agreement *level* the two backends currently achieve,
so a future change that silently widens the gap is caught.

Known gap (not yet root-caused)
--------------------------------
For ``network="small"`` at default settings, the C and Python backends
agree on Neff/YP to ~1e-5 but differ in D/H by ~1.7e-8 absolute (~7e-4
relative) -- *outside* CLAUDE.md's stated +/-3e-9 D/H regression tolerance
for the Python backend's own reference values. That tolerance exists to
resolve flag-level effects (e.g. incomplete_decoupling, QED_corrections) at
the 1e-2..1e-3 level in Neff; the C/Python gap checked here is a distinct,
coarser cross-backend budget (1e-3 relative in D/H) until the discrepancy is
diagnosed. Tightening this bound is a fair signal that the discrepancy has
been understood/fixed; loosening it should not happen without updating this
docstring.
"""
import numpy as np
import pytest

from primat.backend import HAS_C_BACKEND, run_bbn
from primat.evolution import dump_evolution, load_evolution

pytestmark = [pytest.mark.slow, pytest.mark.solve, pytest.mark.backend]

requires_c_backend = pytest.mark.skipif(
    not HAS_C_BACKEND, reason="primat._primat_c C extension is not built"
)

# Keys always present in solve()'s result dict (see primat/main.py), i.e.
# regardless of network/flags -- excludes the conditional keys
# (Li6oLi7/YCNO/Neff/Omeganurel/OneOverOmeganunr).
_ALWAYS_KEYS = {"YPCMB", "YPBBN", "DoH", "He3oH", "He3oHe4", "Li7oH"}


@requires_c_backend
def test_backend_result_dict_shape_matches():
    """C and Python backends return the same result-dict keys for 'small'."""
    params = {"network": "small"}
    r_c = run_bbn(params, force_backend="c")
    r_py = run_bbn(params, force_backend="python")

    assert _ALWAYS_KEYS <= r_c.keys()
    assert _ALWAYS_KEYS <= r_py.keys()
    # Standard background (the default here) always provides the neutrino
    # sector hooks (see PRIMAT.solve()'s final_nu-guarded keys).
    assert {"Neff", "Omeganurel", "OneOverOmeganunr"} <= r_c.keys()
    assert {"Neff", "Omeganurel", "OneOverOmeganunr"} <= r_py.keys()
    # r_c carries one extra "Y_final" sub-dict the Python solve() result does
    # not (a bonus the C wrapper adds, see _wrapper.c's results_to_dict);
    # every other key must match exactly.
    assert r_c.keys() - {"Y_final"} == r_py.keys()
    assert isinstance(r_c["Y_final"], dict)


@requires_c_backend
def test_backend_small_network_numerical_agreement():
    """C vs. Python agreement budget for network='small' (see module docstring)."""
    params = {"network": "small"}
    r_c = run_bbn(params, force_backend="c")
    r_py = run_bbn(params, force_backend="python")

    assert r_c["YPBBN"] == pytest.approx(r_py["YPBBN"], abs=1e-5)
    assert r_c["Neff"] == pytest.approx(r_py["Neff"], abs=1e-3)
    # Known ~1.7e-8 absolute (~7e-4 relative) gap (see module docstring);
    # budgeted well above CLAUDE.md's tighter +/-3e-9 same-backend tolerance.
    assert r_c["DoH"] == pytest.approx(r_py["DoH"], rel=1e-3)


@requires_c_backend
def test_backend_large_amax8_numerical_agreement():
    """C vs. Python agreement for network='large', amax=8 (PRIMAT.md S8.2's
    second reference config, alongside 'small' above)."""
    params = {"network": "large", "amax": 8}
    r_c = run_bbn(params, force_backend="c")
    r_py = run_bbn(params, force_backend="python")

    assert r_c["YPBBN"] == pytest.approx(r_py["YPBBN"], abs=1e-5)
    assert r_c["Neff"] == pytest.approx(r_py["Neff"], abs=1e-3)
    assert r_c["DoH"] == pytest.approx(r_py["DoH"], rel=1e-3)


@pytest.mark.parametrize("params", [
    {"network": "small"},
    {"network": "large", "amax": 8},
], ids=["small", "large_amax8"])
def test_evolution_round_trip_matches_in_memory_result(params, tmp_path):
    """dump_evolution/load_evolution round-trips the Python backend's
    in-memory EvolutionResult (PRIMAT.md S7.3/S7.4) at full precision --
    the disk file is a derived convenience, not a separate source of truth.

    This does not (yet) compare against the C backend's evolution output:
    the C backend has no writer for the unified schema yet (PRIMAT.md S7.6 /
    CPLAN.md Phase 6/7's C-side piece is still pending -- run_bbn raises for
    force_backend="c" with output_time_evolution=True, see
    test_run_bbn_c_backend_rejects_output_time_evolution below). Once that
    lands, this test should grow a cross-backend half that interpolates one
    backend's series onto the other's timestamps (mirrors
    test_custom_background.py's table-interpolation comparison pattern) at
    PRIMAT.md S8.2's documented 1e-5 relative tolerance.
    """
    p = dict(params, output_time_evolution=True, output_file=None)
    result = run_bbn(p, force_backend="python")
    evo = result["evolution"]

    path = tmp_path / "evolution.tsv"
    dump_evolution(evo, path=str(path))
    loaded = load_evolution(str(path))

    np.testing.assert_allclose(loaded.t, evo.t)
    np.testing.assert_allclose(loaded.a, evo.a)
    np.testing.assert_allclose(loaded.T_gamma, evo.T_gamma)
    for flavour in ("e", "mu", "tau"):
        np.testing.assert_allclose(loaded.T_nu[flavour], evo.T_nu[flavour])
    assert loaded.Y.keys() == evo.Y.keys()
    for species in evo.Y:
        np.testing.assert_allclose(loaded.Y[species], evo.Y[species])


@requires_c_backend
def test_run_bbn_auto_prefers_c_backend():
    """force_backend=None/'auto' dispatches to C whenever it is available."""
    r_auto = run_bbn({"network": "small"})
    r_c = run_bbn({"network": "small"}, force_backend="c")
    assert r_auto == r_c


def test_run_bbn_rejects_unknown_force_backend():
    with pytest.raises(ValueError, match="force_backend"):
        run_bbn({"network": "small"}, force_backend="nope")


def test_run_bbn_validates_params_regardless_of_backend():
    """An invalid --network surfaces PRIMATConfig's ValueError pre-dispatch."""
    with pytest.raises(ValueError, match="network must be"):
        run_bbn({"network": "no_such_network"})


def test_run_bbn_python_only_features_force_python_backend(monkeypatch):
    """extra_rho/custom_network/background always force the Python backend,
    even when the C backend is requested implicitly via 'auto'."""
    calls = []
    import primat.backend as backend_mod

    def fake_python_solve(params, extra_rho, custom_network, background):
        calls.append((extra_rho, custom_network, background))
        return {"YPBBN": 0.0}

    monkeypatch.setattr(backend_mod, "_python_solve", fake_python_solve)
    run_bbn({"network": "small"}, extra_rho=[lambda Tg: 0.0])
    assert len(calls) == 1


@requires_c_backend
def test_run_bbn_c_backend_rejects_python_only_features():
    with pytest.raises(ValueError, match="incompatible"):
        run_bbn({"network": "small"}, force_backend="c", extra_rho=[lambda Tg: 0.0])


def test_run_bbn_c_backend_rejects_output_time_evolution():
    """The unified EvolutionResult (primat.evolution, PRIMAT.md S7.3) has no
    C-side equivalent yet (cprimat_run does not return per-step arrays) --
    force_backend="c" must raise rather than silently produce the legacy,
    non-unified TSV cprimat_run still writes to disk on its own."""
    with pytest.raises(ValueError, match="output_time_evolution"):
        run_bbn({"network": "small", "output_time_evolution": True}, force_backend="c")


def test_run_bbn_auto_falls_back_to_python_for_output_time_evolution(monkeypatch):
    """'auto' silently prefers Python when output_time_evolution=True is
    requested, since the C backend has no unified-EvolutionResult equivalent
    yet (see module docstring in primat/backend.py)."""
    import primat.backend as backend_mod

    calls = []

    def fake_python_solve(params, extra_rho, custom_network, background):
        calls.append(params)
        return {"YPBBN": 0.0}

    monkeypatch.setattr(backend_mod, "_python_solve", fake_python_solve)
    monkeypatch.setattr(backend_mod, "HAS_C_BACKEND", True)
    run_bbn({"network": "small", "output_time_evolution": True})
    assert len(calls) == 1


@requires_c_backend
def test_run_bbn_c_backend_honors_rates_overlay(tmp_path):
    """rates_dir/user_rates_dir are supported on the C backend too (see
    primat-c's cpr_config_resolve_rates_path, primat-c/src/config.c): a
    user_rates_dir-supplied network file is loadable end-to-end through
    force_backend="c" exactly like a shipped one."""
    net_dir = tmp_path / "nuclear" / "networks"
    net_dir.mkdir(parents=True)
    (net_dir / "overlaynet.txt").write_text(
        "n_p__d_g, n_p__d_g_primat.txt\nd_p__He3_g, d_p__He3_g_primat.txt\n"
    )
    params = {"network": "overlaynet", "user_rates_dir": str(tmp_path)}
    r_c = run_bbn(params, force_backend="c")
    r_py = run_bbn(params, force_backend="python")
    assert r_c["YPBBN"] == pytest.approx(r_py["YPBBN"], abs=1e-3)


def test_run_bbn_auto_prefers_c_backend_for_rates_overlay(tmp_path, monkeypatch):
    """'auto' dispatches to the C backend for a rates_dir/user_rates_dir
    request too, now that both backends support the overlay -- only
    extra_rho/custom_network/background (Python-only features) force
    Python."""
    import primat.backend as backend_mod

    calls = []

    def fake_c_run_bbn(params, package_dir):
        calls.append(params)
        return {"YPBBN": 0.0}

    monkeypatch.setattr(backend_mod, "HAS_C_BACKEND", True)
    monkeypatch.setattr(backend_mod, "_c_ext", type("M", (), {"run_bbn": staticmethod(fake_c_run_bbn)}))
    run_bbn({"network": "small", "user_rates_dir": str(tmp_path)})
    assert len(calls) == 1
