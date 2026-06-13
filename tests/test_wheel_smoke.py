"""
"pip install" smoke test: build a wheel, install it in a
clean virtual environment, and run a small BBN solve.

Why this test exists
---------------------
The editable install used during development (``pip install -e .`` / running
straight from a git checkout) resolves ``rates/`` relative to the source tree
no matter how the path is computed, so a bug in the package-data path
resolution (e.g. the stale ``../rates`` left over from the package
reorganisation) is invisible there.  It only shows up once the
package is installed as a *wheel* into ``site-packages`` -- a different
directory layout entirely.  Building the wheel also exercises the
``[tool.setuptools.package-data]`` declaration in ``pyproject.toml``: if a
required file under ``rates/`` were ever excluded, the import would still
succeed but ``PyPR(...).solve()`` would fail with a ``FileNotFoundError``
deep inside ``pyprimat.nuclear``/``pyprimat.weak_rates``.

The venv is created with ``--system-site-packages`` so the already-installed
numpy/scipy/joblib (and any optional numba/vegas/numdifftools) are reused --
this test checks the *PyPRIMAT* packaging, not whether its dependencies can
be downloaded.
"""
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.wheel]

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.wheel
def test_wheel_install_smoke_solve():
    """Build a wheel, pip-install it in a clean venv, and run a small solve.

    Steps:
      1. ``pip wheel`` the repo root into a temporary directory (using the
         setuptools build backend already installed in this environment, so
         no network access is required).
      2. Create a fresh venv (``--system-site-packages`` to reuse numpy/scipy/
         joblib already present) and ``pip install --no-deps`` the wheel.
      3. In that venv, run a default-configuration small-network solve and
         check YP/D-H against the loose CLAUDE.md tolerances used by
         ``tests/test_regression.py``'s default-precision checks.

    A failure here most likely means ``rates/`` data files are missing from
    the wheel, or a path is computed relative to the source tree instead of
    the installed package (``pyprimat.config.PyPRConfig.data_dir``).
    """
    with tempfile.TemporaryDirectory(prefix="pyprimat_wheel_") as tmp:
        tmp_path = Path(tmp)
        wheel_dir = tmp_path / "wheel"
        venv_dir  = tmp_path / "venv"

        # ------------------------------------------------------------
        # 1. Build the wheel (no build isolation: reuse the setuptools
        #    already installed here, avoiding any network access).
        # ------------------------------------------------------------
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", str(REPO_ROOT),
             "-w", str(wheel_dir), "--no-deps", "--no-build-isolation", "-q"],
            check=True,
        )
        wheels = list(wheel_dir.glob("*.whl"))
        assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

        # ------------------------------------------------------------
        # 2. Fresh venv + install the wheel (reusing system site-packages
        #    for numpy/scipy/joblib/numba/...).
        # ------------------------------------------------------------
        venv.create(venv_dir, with_pip=True, system_site_packages=True)
        venv_python = venv_dir / "bin" / "python"
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--no-deps", "-q",
             str(wheels[0])],
            check=True,
        )

        # ------------------------------------------------------------
        # 3. Smoke solve: default config, small network.  save_nTOp
        #    defaults to False, so this does not write into site-packages.
        # ------------------------------------------------------------
        smoke_script = (
            "from pyprimat import PyPR\n"
            "r = PyPR({'network': 'small', 'verbose': False, 'debug': False}).solve()\n"
            "print(r['YPBBN'], r['DoH'])\n"
        )
        result = subprocess.run(
            [str(venv_python), "-c", smoke_script],
            check=True, capture_output=True, text=True,
        )

    # Same loose tolerances as tests/test_regression.py::test_small_network_*
    yp_str, doh_str = result.stdout.split()
    yp, doh = float(yp_str), float(doh_str)
    assert yp  == pytest.approx(0.2469983, abs=1e-4)
    assert doh == pytest.approx(2.43490e-5, rel=2e-3)
