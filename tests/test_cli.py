"""
Tests for the ``primat`` console-script CLI.

``primat.cli.main()`` is invoked in-process (no subprocess) with an
explicit ``argv`` list, which is exactly what the ``primat`` console
script does at startup.  Each invocation runs one full small-network solve
(~1.2 s), so these tests are marked ``slow``/``solve`` like the other
single-solve tests in the "solve" tier.
"""
import json
import re

import pytest

from primat.cli import main
from primat.credits import cli_credits_text
from tests.reference_values import (
    DOH_ABS_TOL,
    DOH_REFERENCE,
    NEFF_ABS_TOL,
    NEFF_REFERENCE,
    YPBBN_ABS_TOL,
    YPBBN_REFERENCE,
)

pytestmark = [pytest.mark.slow, pytest.mark.solve]


def test_cli_default_summary(capsys):
    """No flags: default (small-network) run, human-readable summary.

    Parses the printed values rather than matching a literal string, and
    compares against the centralised CLAUDE.md tolerances in
    tests/reference_values.py, so a routine default-parameter tweak (e.g.
    commit e00f062's rate_grid_npts/sampling_temperature_per_decade bump)
    does not require refreshing a hard-coded pin here.
    """
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    neff = float(re.search(r"Neff\s*=\s*([\d.]+)", out).group(1))
    yp   = float(re.search(r"YP \(BBN\)\s*=\s*([\d.]+)", out).group(1))
    doh  = float(re.search(r"D/H\s*=\s*([\d.eE+-]+)", out).group(1))
    assert neff == pytest.approx(NEFF_REFERENCE, abs=NEFF_ABS_TOL)
    assert yp   == pytest.approx(YPBBN_REFERENCE, abs=YPBBN_ABS_TOL)
    assert doh  == pytest.approx(DOH_REFERENCE, abs=DOH_ABS_TOL)
    assert "Li6/Li7" not in out


def test_cli_json_matches_default_summary(capsys):
    """--json prints the full results dict, parseable and consistent."""
    rc = main(["--json"])
    assert rc == 0
    results = json.loads(capsys.readouterr().out)
    assert results["Neff"]   == pytest.approx(NEFF_REFERENCE, abs=NEFF_ABS_TOL)
    assert results["YPBBN"]  == pytest.approx(YPBBN_REFERENCE, abs=YPBBN_ABS_TOL)
    assert results["DoH"]    == pytest.approx(DOH_REFERENCE, abs=DOH_ABS_TOL)
    assert "Li6oLi7" not in results


def test_cli_omegabh2_override_changes_doh(capsys):
    """--Omegabh2 is forwarded to PRIMATConfig and changes the result."""
    rc = main(["--Omegabh2", "0.024", "--json"])
    assert rc == 0
    results = json.loads(capsys.readouterr().out)
    # A higher baryon density measurably increases D/H away from the
    # Omegabh2=0.022425 reference value above.
    assert results["DoH"] != pytest.approx(2.4349347363779478e-05, rel=1e-6)


def test_cli_network_accepts_any_network_file(capsys):
    """--network accepts any name with a rates/nuclear/networks/<name>.txt
    file, not just 'small'/'small_parthenope'/'large'.

    'small_parthenope' (12-reaction network using Parthenope 3.0 rate
    tables) uses different reaction rates from 'small', so YPBBN differs
    from the default-network reference value above while remaining a
    physically reasonable abundance.
    """
    rc = main(["--network", "small_parthenope", "--json"])
    assert rc == 0
    results = json.loads(capsys.readouterr().out)
    assert 0.24 < results["YPBBN"] < 0.25
    assert results["YPBBN"] != pytest.approx(0.24699534223598402, rel=1e-6)


def test_cli_network_rejects_unknown_name():
    """An unknown --network name surfaces PRIMATConfig's ValueError."""
    with pytest.raises(ValueError, match="network must be"):
        main(["--network", "no_such_network"])


def test_cli_network_error_mentions_data_tree():
    """The missing-network error should point at data/nuclear/networks."""
    with pytest.raises(ValueError, match=r"data/nuclear/networks"):
        main(["--network", "no_such_network"])


def test_cli_network_error_lists_overlay_candidates(tmp_path):
    """A custom overlay should be named explicitly in the missing-network error."""
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    expected = overlay / "networks" / "custom.txt"

    with pytest.raises(ValueError, match=re.escape(str(expected))):
        main(["--set", f"user_rates_dir={overlay}", "--network", "custom"])


def test_cli_set_expands_tilde_in_path_values(monkeypatch, tmp_path, capsys):
    """Quoted ``~`` paths passed through ``--set`` should resolve to HOME.

    The CLI forwards ``--set`` values as raw strings, so path parameters must
    normalize home-directory prefixes inside the config layer rather than
    relying on shell expansion.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "custom").mkdir()

    rc = main(["--set", "user_rates_dir=~/custom", "--json"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "tables and networks located in" in err
    assert str((tmp_path / "custom").resolve()) in err


def test_cli_help_shows_named_output_path_flags(capsys):
    """``primat --help`` documents the four output-path flags as basic options.

    These paths are user-facing CLI knobs, so they must appear in the printed
    help instead of being buried under the hidden ``--set KEY=VALUE`` escape
    hatch.
    """
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--output_file FILE" in out
    assert "--output_final_file FILE" in out
    assert "--output_background_file FILE" in out
    assert "--output_mc_file FILE" in out
    assert not re.search(r"(?m)^\s+--set\b", out)


def test_cli_credits_prints_short_text(capsys):
    """--credits prints the attribution text without install/run guidance."""
    rc = main(["--credits"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.rstrip("\n") == cli_credits_text()
    assert "pip install primat" not in out


def test_cli_mc_output_announces_path(capsys, tmp_path):
    """The MC TSV writer must also emit a visible [output] line."""
    out_path = tmp_path / "mc_samples.tsv"
    rc = main(["--mc", "1", "--output_mc_samples", "--output_mc_file", str(out_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[output] MC samples (1 sample) written to" in out
    assert str(out_path.resolve()) in out


def test_cli_mc_output_file_without_enable_flag_does_not_write(capsys, tmp_path):
    """The filename option alone should not force MC sample output."""
    out_path = tmp_path / "mc_samples.tsv"
    rc = main(["--mc", "1", "--output_mc_file", str(out_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[output] MC samples" not in out
    assert not out_path.exists()


def test_cli_mc_summary_includes_all_displayed_sigmas(capsys):
    """The human-readable MC summary should print sigma for every displayed
    ratio, not only the first few observables.

    This exercises a network that actually produces Li6/Li7 and CNO so the
    optional lines are present in the output.
    """
    rc = main(["--network", "large", "--mc", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert re.search(r"He3/He4\s*=\s*[\d.eE+-]+\s+\+/-\s+[\d.eE+-]+", out)
    assert re.search(r"Li6/Li7\s*=\s*[\d.eE+-]+\s+\+/-\s+[\d.eE+-]+", out)
    assert re.search(r"CNO \(mass\)\s*=\s*[\d.eE+-]+\s+\+/-\s+[\d.eE+-]+", out)
