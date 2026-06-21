"""
Tests for the ``pyprimat`` console-script CLI.

``pyprimat.cli.main()`` is invoked in-process (no subprocess) with an
explicit ``argv`` list, which is exactly what the ``pyprimat`` console
script does at startup.  Each invocation runs one full small-network solve
(~1.2 s), so these tests are marked ``slow``/``solve`` like the other
single-solve tests in the "solve" tier.
"""
import json
import re

import pytest

from pyprimat.cli import main
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
    """--Omegabh2 is forwarded to PyPRConfig and changes the result."""
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
    """An unknown --network name surfaces PyPRConfig's ValueError."""
    with pytest.raises(ValueError, match="network must be"):
        main(["--network", "no_such_network"])
