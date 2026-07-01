"""
Notebook smoke test.

Executes demonstration notebooks end-to-end with ``papermill`` and checks
they run without raising. Two tiers:

* ``FAST_NOTEBOOKS`` -- no Monte Carlo, run as-is:
    - ``AbundanceEvolution.ipynb`` -- small/large(amax=8)/large solves, ~5 s.
    - ``CompareSmallNetworks.ipynb`` -- two small-network solves, ~4 s.
* ``MC_NOTEBOOKS`` -- normally run a Monte Carlo scan at publication-quality
  sample counts (``num_mc``/``N_MC`` ~100-500, sometimes over a parameter
  grid too, e.g. ``StandardPlots.ipynb``'s 20 eta points x 100 MC samples).
  The cell that sets the sample count is tagged ``parameters`` in each of
  these notebooks, so papermill overrides it down to ``MC_NOTEBOOK_NUM_MC``
  (3) here -- enough to exercise the MC code path (`run_bbn`/`run_mc`
  wiring, plotting of central value + error band) without paying for
  publication-quality statistics:
    - ``AbundancesNrelat.ipynb``, ``AbundancesXi.ipynb`` -- ~21/11-point
      grids x 3 MC samples.
    - ``PosteriorBaryons.ipynb`` -- ~17-point grid x 3 MC samples.
    - ``StandardPlots.ipynb`` -- 20-point grid x 3 MC samples.
    - ``MonteCarloRates.ipynb`` -- 3 full-BBN MC samples (no grid).

This is a regression guard against import-path bugs (the notebooks still
imported the pre-reorganisation ``pypr`` package name)
and against API drift in ``primat.main.PRIMAT``/``primat.backend.run_mc``: a
renamed/removed attribute that the notebooks rely on (``r.A``,
``r.abundance_names``, ``r[name](t)``, ``run_mc(...)``'s return shape, ...)
makes one of these cells raise, and papermill re-raises that as a
``CellExecutionError`` here.

``Sensitivity.ipynb``, ``AnimatedAbundances.ipynb`` and
``AbundancesXi``/``AbundancesNrelat``'s plotting/animation cells beyond the
MC scan itself are out of scope for now.

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

# Notebook name -> papermill parameter dict overriding its MC sample count
# (the cell tagged "parameters" in each of these notebooks).
MC_NOTEBOOK_NUM_MC = 3
MC_NOTEBOOKS = {
    "AbundancesNrelat.ipynb": {"num_mc": MC_NOTEBOOK_NUM_MC},
    "AbundancesXi.ipynb": {"num_mc": MC_NOTEBOOK_NUM_MC},
    "PosteriorBaryons.ipynb": {"num_mc": MC_NOTEBOOK_NUM_MC},
    "StandardPlots.ipynb": {"num_mc": MC_NOTEBOOK_NUM_MC},
    "MonteCarloRates.ipynb": {"N_MC": MC_NOTEBOOK_NUM_MC},
}


def _run_notebook(name, tmp_path, monkeypatch, parameters=None):
    """Execute one notebook with papermill from a throwaway copy of notebooks/."""
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
        parameters=parameters or {},
    )


@pytest.mark.parametrize("name", FAST_NOTEBOOKS)
def test_fast_notebook_executes(name, tmp_path, monkeypatch):
    """Run a fast demo notebook with papermill; fail if any cell raises."""
    _run_notebook(name, tmp_path, monkeypatch)


@pytest.mark.parametrize("name", MC_NOTEBOOKS)
def test_mc_notebook_executes_with_few_samples(name, tmp_path, monkeypatch):
    """Run an MC demo notebook with its sample count cut to 3, via papermill
    parameter injection into the notebook's tagged "parameters" cell."""
    _run_notebook(name, tmp_path, monkeypatch, parameters=MC_NOTEBOOKS[name])
