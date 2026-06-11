"""
Tests for the ``pyprimat`` console-script CLI.

``pyprimat.cli.main()`` is invoked in-process (no subprocess) with an
explicit ``argv`` list, which is exactly what the ``pyprimat`` console
script does at startup.  Each invocation runs one full small-network solve
(~1.2 s), so these tests are marked ``slow``/``solve`` like the other
single-solve tests in the "solve" tier.
"""
import json

import pytest

from pyprimat.cli import main

pytestmark = [pytest.mark.slow, pytest.mark.solve]


def test_cli_default_summary(capsys):
    """No flags: default (small-network) run, human-readable summary."""
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    # Default small-network run, loading the shipped fingerprinted weak-rate
    # cache (rates/weak/nTOp_*.txt).  Within the CLAUDE.md tolerances
    # (YP 0.2469156 +/-1e-5, D/H 2.43647e-5 +/-3e-9).
    assert "Neff     = 3.04397730" in out
    assert "YP (BBN) = 0.24691081" in out
    assert "D/H      = 2.4365492e-05" in out


def test_cli_json_matches_default_summary(capsys):
    """--json prints the full results dict, parseable and consistent."""
    rc = main(["--json"])
    assert rc == 0
    results = json.loads(capsys.readouterr().out)
    assert results["Neff"]   == pytest.approx(3.043977298557919, rel=1e-12)
    assert results["YPBBN"]  == pytest.approx(0.24691080737348087, rel=1e-12)
    assert results["DoH"]    == pytest.approx(2.4365492014999738e-05, rel=1e-12)


def test_cli_omegabh2_override_changes_doh(capsys):
    """--Omegabh2 is forwarded to PyPRConfig and changes the result."""
    rc = main(["--Omegabh2", "0.024", "--json"])
    assert rc == 0
    results = json.loads(capsys.readouterr().out)
    # A higher baryon density measurably increases D/H away from the
    # Omegabh2=0.022425 reference value above.
    assert results["DoH"] != pytest.approx(2.4365492014999738e-05, rel=1e-6)
