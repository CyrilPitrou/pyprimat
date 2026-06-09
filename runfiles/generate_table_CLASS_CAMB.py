"""
Generate a PRIMAT-style BBN table with Monte Carlo uncertainties.

Reproduces PRIMAT_Yp_DH_ErrorMC_100_<year>.dat using PyPRIMAT.

Output: results/PyPRIMAT_Yp_DH_ErrorMC_<N_MC>_<year>.dat

Columns: Ombh2, eta10, DeltaN, Yp(CMB), Yp^BBN, sig(Yp^BBN), D/H, sig(D/H)

Strategy for efficiency
-----------------------
For each DeltaN value the n<->p weak-rate tables are computed once and saved to
disk (compute_nTOp=True, save_nTOp=True).  All subsequent runs for
that DeltaN load the saved tables, so only the nuclear-network ODE is
re-integrated.  tau_n variation affects only the weak-rate normalisation, which
is re-evaluated for each instance.

Checkpointing / resuming (IMPORTANT)
------------------------------------
The full grid is ~130k ODE integrations and takes hours.  Earlier versions kept
*every* result in memory and wrote the table only at the very end, so any
interruption -- the OS out-of-memory (OOM) killer reaping the long-lived joblib
worker pool, a single solver failure, a reboot -- threw away the whole run.  (The
"it runs but at some point it stops" symptom is most often the OOM killer; see
the mitigations below.)

The computation is now **checkpointed per DeltaN**.  After each DeltaN block is
finished its results are written atomically to

    results/checkpoints_MC<N_MC>/dN_<i>.npz

On start-up every existing checkpoint that matches the current grid is loaded and
its DeltaN is skipped, so re-running the script simply resumes where it stopped
-- no work is repeated.  The final .dat is assembled from the checkpoints, so
even a partial run yields a usable (partial) table.  Delete the checkpoint
directory to force a clean recomputation.

Memory mitigations: per-worker exceptions are caught (a bad grid point becomes
NaN instead of killing the run); the joblib worker pool is torn down and
respawned after every DeltaN so worker memory cannot accumulate across the whole
grid; and the large per-DeltaN arrays are freed each iteration.  If the OOM
killer still strikes, lower ``N_JOBS`` below (fewer concurrent workers => lower
peak memory) and just re-run -- checkpoints make that cheap.

Run from the repo root:
    python runfiles/generate_table_CLASS_CAMB.py

Estimated run time: ~1–3 hours on a modern multicore machine (n_jobs=-1).
"""

import sys
import os
import time
import datetime
import numpy as np
from joblib import Parallel, delayed

# Ensure the repo root is importable when the script is run from any directory.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pypr.main import PyPR
from pypr.config import DEFAULT_PARAMS, PyPRConfig

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

# Omega_b h^2 grid (matches PRIMAT reference file)
Ombh2_coarse1 = np.arange(0.005, 0.020, 0.001)                 # 0.005–0.019, 15 pts
Ombh2_fine    = np.round(np.arange(0.020, 0.0241, 0.0002), 4)  # 0.020–0.024, 21 pts
Ombh2_coarse2 = np.arange(0.025, 0.041, 0.001)                 # 0.025–0.040, 16 pts
Ombh2_grid    = np.concatenate([Ombh2_coarse1, Ombh2_fine, Ombh2_coarse2])

# DeltaN grid
DeltaN_grid = np.array([
    -3., -2.5, -2., -1.5, -1., -0.75, -0.5, -0.25,
     0., 0.25, 0.5, 0.75, 1., 1.5, 2., 2.5, 3.,
     3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7.
])

n_omega  = len(Ombh2_grid)
n_deltaN = len(DeltaN_grid)
print(f"Omega_b h^2 grid: {n_omega} points  ({Ombh2_grid[0]:.4f} – {Ombh2_grid[-1]:.4f})")
print(f"DeltaN grid:      {n_deltaN} points  ({DeltaN_grid[0]} – {DeltaN_grid[-1]})")
print(f"Total grid points: {n_omega * n_deltaN}")

# ---------------------------------------------------------------------------
# Run parameters
# ---------------------------------------------------------------------------

N_MC      = 10    # MC samples per grid point (use at least 100)
SEED_BASE = 12345  # for reproducibility
N_JOBS    = -1     # joblib workers; -1 = all cores.  Lower this (e.g. 4) if the
                   # OS OOM killer stops the run -- fewer workers => less memory.

# Precompute the conversion factor eta10 = 1e10 * eta_0 / (Omega_b h^2)
_cfg0 = PyPRConfig()
OMBH2_TO_ETA10 = 1e10 * _cfg0.Omegabh2_to_eta0b
print(f"eta10 / (Omega_b h^2) = {OMBH2_TO_ETA10:.6f}")

# The 12 nuclear rate variation keys used in the small network
ALL_P_KEYS = [k for k in DEFAULT_PARAMS if k.startswith('p_')]
RATE_KEYS  = ALL_P_KEYS[:12]
print(f"Rate keys ({len(RATE_KEYS)}): {RATE_KEYS}")

# Options shared by every PyPR call (weak-rate tables are loaded from disk)
BASE_OPTS = {
    'verbose':      False,
    'debug':        False,
    'compute_nTOp': False,   # load pre-tabulated weak rates
    'save_nTOp':    False,
    'network': 'small',
    'tau_n':             DEFAULT_PARAMS['tau_n'],      # central neutron lifetime [s]
    'std_tau_n':         DEFAULT_PARAMS['std_tau_n'],  # 1σ uncertainty [s]
}
print(f"tau_n = {BASE_OPTS['tau_n']} ± {BASE_OPTS['std_tau_n']} s")

# ---------------------------------------------------------------------------
# Checkpointing: save each DeltaN block to disk so the run is resumable.
# ---------------------------------------------------------------------------

import hashlib
import gc

# One .npz per DeltaN, under a directory tagged by N_MC (different N_MC => a
# different, non-conflicting checkpoint set).
CKPT_DIR = os.path.join(repo_root, 'results', f'checkpoints_MC{N_MC}')


def _grid_signature():
    """Fingerprint the grid + MC settings so stale checkpoints are never reused.

    If the Omega_b h^2 grid, the DeltaN grid, N_MC or the RNG seed change, the
    stored results no longer correspond to the current run: the signature stored
    in each checkpoint will mismatch and that DeltaN will be recomputed.
    """
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(Ombh2_grid, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(DeltaN_grid, dtype=np.float64).tobytes())
    h.update(f'{N_MC}|{SEED_BASE}'.encode())
    return h.hexdigest()


GRID_SIG = _grid_signature()


def _ckpt_path(i_dN):
    """Absolute path of the checkpoint file for DeltaN index ``i_dN``."""
    return os.path.join(CKPT_DIR, f'dN_{i_dN:02d}.npz')


def _load_checkpoint(i_dN):
    """Return the stored arrays for DeltaN index ``i_dN``, or None.

    None is returned when the checkpoint is absent, unreadable (e.g. a half-
    written file from a process killed mid-save), or was produced for a different
    grid (signature mismatch) -- in every case the block is simply recomputed.
    """
    path = _ckpt_path(i_dN)
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path)
        if str(d['grid_sig']) != GRID_SIG:
            return None
        return d
    except Exception:
        return None


def _save_checkpoint(i_dN, DeltaN, YPCMB, YPBBN, DoH, sig_YPBBN, sig_DoH):
    """Atomically write one DeltaN block (all Omega_b h^2 at once) to disk.

    Each array has length ``n_omega`` and is aligned with ``Ombh2_grid``.  The
    write goes to a temporary file that is then ``os.replace``-d into place, so a
    crash during the write can never leave a corrupt checkpoint behind.
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    tmp = _ckpt_path(i_dN) + '.tmp.npz'   # np.savez keeps the .npz we give it
    np.savez(tmp,
             grid_sig=GRID_SIG, DeltaN=float(DeltaN), Ombh2_grid=Ombh2_grid,
             YPCMB=YPCMB, YPBBN=YPBBN, DoH=DoH,
             sig_YPBBN=sig_YPBBN, sig_DoH=sig_DoH)
    os.replace(tmp, _ckpt_path(i_dN))


def _recycle_worker_pool():
    """Tear down and respawn the joblib (loky) worker pool.

    Called after each DeltaN so that any memory accumulated in long-lived worker
    processes over thousands of solves is released, instead of growing until the
    OS OOM killer stops the run.
    """
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass   # best-effort; never let pool recycling break the computation


# ---------------------------------------------------------------------------
# Worker functions (must be module-level for joblib multiprocessing)
# ---------------------------------------------------------------------------

def _run_central(Ombh2, DeltaN):
    """Central BBN prediction: all p_* = 0, tau_n at its central value.

    On any failure the point is returned as NaN rather than raising, so a single
    pathological grid point cannot abort (and discard) the whole DeltaN block.
    """
    try:
        params = {
            **BASE_OPTS,
            'Omegabh2': float(Ombh2),
            'DeltaNeff': float(DeltaN),
        }
        inst = PyPR(params=params)
        inst.solve()
        return (
            inst.get_quantity('YPBBN'),
            inst.get_quantity('YPCMB'),
            inst.get_quantity('DoH'),
        )
    except Exception as exc:
        print(f"\n  [warn] central failed at Ombh2={Ombh2}, DeltaN={DeltaN}: {exc}",
              file=sys.stderr, flush=True)
        return (np.nan, np.nan, np.nan)


def _run_mc_sample(Ombh2, DeltaN, p_vals, tau_n_val):
    """One MC sample: randomised nuclear rates and tau_n.

    As with _run_central, failures return NaN so one bad sample does not abort
    the block; the NaN is excluded from the std via np.nanstd below.
    """
    try:
        params = {
            **BASE_OPTS,
            'Omegabh2': float(Ombh2),
            'DeltaNeff': float(DeltaN),
            'tau_n':     float(tau_n_val),
        }
        for k, v in zip(RATE_KEYS, p_vals):
            params[k] = float(v)
        inst = PyPR(params=params)
        inst.solve()
        return inst.get_quantity('YPBBN'), inst.get_quantity('DoH')
    except Exception:
        return np.nan, np.nan

# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

# all_results[(Ombh2, DeltaN)] = dict with YPCMB, YPBBN, sig_YPBBN, DoH, sig_DoH
all_results = {}

t_wall = time.time()

def _store_block(DeltaN, YPCMB, YPBBN, DoH, sig_YPBBN, sig_DoH):
    """Copy one DeltaN block of per-Omega arrays into the global all_results."""
    for i_om, Ombh2 in enumerate(Ombh2_grid):
        all_results[(float(Ombh2), float(DeltaN))] = {
            'YPCMB':     float(YPCMB[i_om]),
            'YPBBN':     float(YPBBN[i_om]),
            'sig_YPBBN': float(sig_YPBBN[i_om]),
            'DoH':       float(DoH[i_om]),
            'sig_DoH':   float(sig_DoH[i_om]),
        }


for i_dN, DeltaN in enumerate(DeltaN_grid):
    # ------------------------------------------------------------------
    # 0. Resume: if this DeltaN was already computed in a previous run, load
    #    its checkpoint and skip straight to the next one.
    # ------------------------------------------------------------------
    ck = _load_checkpoint(i_dN)
    if ck is not None:
        _store_block(DeltaN, ck['YPCMB'], ck['YPBBN'], ck['DoH'],
                     ck['sig_YPBBN'], ck['sig_DoH'])
        print(f"[{i_dN+1:2d}/{n_deltaN}] DeltaN = {DeltaN:+.2f}  [loaded from checkpoint]",
              flush=True)
        continue

    t0 = time.time()
    print(f"[{i_dN+1:2d}/{n_deltaN}] DeltaN = {DeltaN:+.2f}", end='', flush=True)

    # ------------------------------------------------------------------
    # 1. Compute n<->p weak-rate tables for this DeltaN and save to disk.
    #    Ombh2 does not affect these tables (only the nuclear-network ODE).
    # ------------------------------------------------------------------
    _seed_params = {
        **BASE_OPTS,
        'compute_nTOp': True,
        'save_nTOp':    True,
        'Omegabh2': 0.022425,
        'DeltaNeff': float(DeltaN),
    }
    _ = PyPR(params=_seed_params)   # side-effect: saves rates/weak/*.txt
    print(f"  [weak {time.time()-t0:.0f}s]", end='', flush=True)

    # ------------------------------------------------------------------
    # 2. Central values for every Ombh2 (parallel).
    # ------------------------------------------------------------------
    t1 = time.time()
    central_raw = Parallel(n_jobs=N_JOBS)(
        delayed(_run_central)(Ombh2, DeltaN)
        for Ombh2 in Ombh2_grid
    )
    central_arr = np.array(central_raw)            # (n_omega, 3): YPBBN, YPCMB, DoH
    c_YPBBN = central_arr[:, 0]
    c_YPCMB = central_arr[:, 1]
    c_DoH   = central_arr[:, 2]
    print(f"  [central {time.time()-t1:.0f}s]", end='', flush=True)

    # ------------------------------------------------------------------
    # 3. MC samples: parallel over all (Ombh2, i_mc) pairs.
    #    Draw nuclear rate offsets p_i ~ N(0,1) and
    #    tau_n ~ N(BASE_OPTS['tau_n'], BASE_OPTS['std_tau_n']).
    # ------------------------------------------------------------------
    t2 = time.time()
    # Draw ONE shared set of MC samples for all Ombh2 points and reuse the same
    # seed for every DeltaN.  This removes the MC noise that would otherwise cause
    # sigma estimates to vary erratically between neighbouring grid points or
    # between DeltaN values (with N_MC=100 the noise on std is ~7% if samples
    # differ per point).
    rng     = np.random.default_rng(SEED_BASE)
    p_all   = rng.standard_normal((N_MC, len(RATE_KEYS)))   # shared across Ombh2
    tau_all = (BASE_OPTS['tau_n']
               + BASE_OPTS['std_tau_n'] * rng.standard_normal(N_MC))

    mc_raw = Parallel(n_jobs=N_JOBS)(
        delayed(_run_mc_sample)(
            Ombh2_grid[i_om], DeltaN,
            p_all[i_mc],
            tau_all[i_mc]
        )
        for i_om in range(n_omega)
        for i_mc in range(N_MC)
    )

    mc_arr   = np.array(mc_raw).reshape(n_omega, N_MC, 2)
    mc_YPBBN = mc_arr[:, :, 0]   # (n_omega, N_MC)
    mc_DoH   = mc_arr[:, :, 1]
    # nanstd: a failed (NaN) sample is excluded rather than poisoning the sigma.
    sig_YPBBN = np.nanstd(mc_YPBBN, axis=1)
    sig_DoH   = np.nanstd(mc_DoH,   axis=1)
    print(f"  [MC {time.time()-t2:.0f}s]", flush=True)

    # ------------------------------------------------------------------
    # 4. Checkpoint this DeltaN block to disk, then store it in memory.
    #    The checkpoint is written first so progress survives even if the
    #    process is killed immediately afterwards.
    # ------------------------------------------------------------------
    _save_checkpoint(i_dN, DeltaN, c_YPCMB, c_YPBBN, c_DoH, sig_YPBBN, sig_DoH)
    _store_block(DeltaN, c_YPCMB, c_YPBBN, c_DoH, sig_YPBBN, sig_DoH)

    # ------------------------------------------------------------------
    # 5. Release this block's memory and recycle the worker pool so memory
    #    cannot accumulate across the (long) outer loop.
    # ------------------------------------------------------------------
    del central_raw, central_arr, mc_raw, mc_arr, mc_YPBBN, mc_DoH, p_all, tau_all
    gc.collect()
    _recycle_worker_pool()

print(f"\nTotal wall time: {(time.time()-t_wall)/60:.1f} min")

# ---------------------------------------------------------------------------
# Write output file
# ---------------------------------------------------------------------------

os.makedirs(os.path.join(repo_root, 'results'), exist_ok=True)
year    = datetime.date.today().year
outfile = os.path.join(repo_root, 'results',
                       f'PyPRIMAT_Yp_DH_ErrorMC_{N_MC}_{year}.dat')

# Fixed column widths (chars, left-aligned).
# Data rows are prefixed with "  " to align with the "# " on the header line.
W = dict(Ombh2=12, eta10=12, DeltaN=16, Yp=16, YpBBN=16, sigYp=16, DH=16)

HEADER_LINE = (
    "# "
    + f"{'Ombh2':<{W['Ombh2']}}"
    + f"{'eta10':<{W['eta10']}}"
    + f"{'DeltaN':<{W['DeltaN']}}"
    + f"{'Yp':<{W['Yp']}}"
    + f"{'Yp^BBN':<{W['YpBBN']}}"
    + f"{'sig(Yp^BBN)':<{W['sigYp']}}"
    + f"{'D/H':<{W['DH']}}"
    + "sig(D/H)"
)


def fmt_row(Ombh2, eta10, DeltaN, YPCMB, YPBBN, sig_YPBBN, DoH, sig_DoH):
    """Format one data row with fixed column widths."""
    return (
        "  "   # aligns with "# " on the header line
        + f"{Ombh2:.6f}"    .ljust(W['Ombh2'])
        + f"{eta10:.6g}"    .ljust(W['eta10'])
        + f"{DeltaN:.6f}"   .ljust(W['DeltaN'])
        + f"{YPCMB:.6f}"    .ljust(W['Yp'])
        + f"{YPBBN:.6f}"    .ljust(W['YpBBN'])
        + f"{sig_YPBBN:.7g}".ljust(W['sigYp'])
        + f"{DoH:.7g}"      .ljust(W['DH'])
        + f"{sig_DoH:.7g}"
        + "\n"
    )


HEADER = f"""\
# BBN prediction of the primordial abundances (He-4 and Deuterium) as a function of
# 1)the baryon density $\\Omega_b h^2$ and
# 2)the number of extra relativistic degrees of freedom $\\Delta N$
#
# $\\Delta N=0$ is the number of extra relativistic species (which mimic decoupled neutrinos).
# If $\\Delta N=0$, then $N_{{eff}} = 3.0440$ because of QED effects and Incomplete Neutrino decoupling (arXiv:2008.01074).
#
# Computation performed with PyPRIMAT (Python port of PRIMAT by Cyril Pitrou 2018-{year})
# Details on arXiv:1801.08023 and update arXiv:2011.11320
# Last update {year} (notably including the LUNA rate for the d(p,g)He3 reaction)
#
# Neutron Decay rate $\\tau_n$ is {BASE_OPTS['tau_n']}s (+-{BASE_OPTS['std_tau_n']}s), following Particle Data Group {year}
# CMB temperature is 2.7255 K (without taking into account a possible uncertainty)
#
# He4 is given either as $Y^BBN_P = 4 Y_{{He4}}$ where $Y_{{He4}}$ is the ratio of He-4 number density to baryons number density.
# He4 is also provided as $Y_P = Y_{{He4}} * m_{{He4}} /[Y_{{He4}} m_{{He4}} + (1 - 4*Y_{{He4}}) m_H1]$
# where $m_{{He4}}=4.0026032541$ and $m_H1=1.00782503223$ are atomic masses.
#
# Deuterium is given as the ratio of its number density to the H1 number density and noted $D/H$.
#
# eta10 is $10^{{10}} eta_0$ where $eta_0$ is the density ratio between baryons and photons AT THE END OF BBN.
# Hence eta10 ignores the extra production of He4 by stars after BBN. Whenever possible, working with $\\Omega_b h^2$ is better.
#
# Errors are computed with a Monte-Carlo method on {N_MC} samples, varying nuclear rates and tau_n.
# See PRIMAT paper (2018) for details.
#
{HEADER_LINE}"""

# Assemble from whatever results are available.  A partial run (some DeltaN
# still missing because it was interrupted) still produces a valid, usable table
# containing the completed DeltaN blocks; re-run the script to fill in the rest.
n_written  = 0
n_missing  = 0
with open(outfile, 'w') as f:
    f.write(HEADER + '\n')
    for DeltaN in DeltaN_grid:
        for Ombh2 in Ombh2_grid:
            r = all_results.get((float(Ombh2), float(DeltaN)))
            if r is None:
                n_missing += 1
                continue
            eta10 = Ombh2 * OMBH2_TO_ETA10
            f.write(fmt_row(Ombh2, eta10, DeltaN,
                            r['YPCMB'], r['YPBBN'], r['sig_YPBBN'],
                            r['DoH'], r['sig_DoH']))
            n_written += 1

print(f"Written {n_written} rows to {outfile}")
if n_missing:
    print(f"WARNING: {n_missing} grid points still missing "
          f"({n_missing // n_omega} DeltaN block(s) not yet computed). "
          f"Re-run the script to compute them; finished blocks are cached in "
          f"{os.path.relpath(CKPT_DIR, repo_root)} and will not be recomputed.")
print(f"File size: {os.path.getsize(outfile) / 1024:.1f} kB")
