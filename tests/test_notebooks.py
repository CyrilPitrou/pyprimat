"""
Notebook smoke test.

Executes the two *fast* demonstration notebooks end-to-end with
``papermill`` and checks they run without raising:

* ``AbundanceEvolution.ipynb`` -- small/medium/large solves, ~5 s total.
* ``CompareSmallNetworks.ipynb`` -- two small-network solves (~4 s total).

This is a regression guard against import-path bugs (the notebooks still
imported the pre-reorganisation ``pypr`` package name)
and against API drift in ``pyprimat.main.PyPR``: a renamed/removed attribute
that the notebooks rely on (``r.A``, ``r.abundance_names``, ``r[name](t)``,
...) makes one of these cells raise, and papermill re-raises that as a
``CellExecutionError`` here.

The heavier notebooks (``StandardPlots.ipynb`` with its 1000-job Monte Carlo
scan, ``MonteCarloRates.ipynb`` with ``num_mc=500``, ...) are *not* covered
here -- they take minutes to hours and are out of scope for a CI smoke test.

Requires ``papermill`` (an optional ``notebooks`` extra, see
``pyproject.toml``); skipped if not installed.
"""
import shutil
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.notebook]

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"

FAST_NOTEBOOKS = [
    "AbundanceEvolution.ipynb",
    "CompareSmallNetworks.ipynb",
]


@pytest.mark.parametrize("name", FAST_NOTEBOOKS)
def test_fast_notebook_executes(name, tmp_path, monkeypatch):
    """Run a fast demo notebook with papermill; fail if any cell raises."""
    papermill = pytest.importorskip("papermill")

    # Headless plotting backend: notebooks call plt.savefig()/plt.show(),
    # which would otherwise try (and fail) to open a GUI window in CI.
    monkeypatch.setenv("MPLBACKEND", "Agg")

    # Run from a throwaway copy of notebooks/, so plt.savefig('plots/...')
    # (a relative path resolved against cwd) writes into tmp_path instead
    # of overwriting the tracked PDFs in notebooks/plots/.
    work_dir = tmp_path / "notebooks"
    shutil.copytree(NOTEBOOKS_DIR, work_dir)

    papermill.execute_notebook(
        str(work_dir / name), str(work_dir / f"out_{name}"),
        cwd=str(work_dir),
        progress_bar=False,
    )
