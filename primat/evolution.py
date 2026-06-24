# -*- coding: utf-8 -*-
"""
evolution.py
============
Unified time-evolution schema (``PRIMAT.md`` S7): the shared columns both
the Python and C backends populate, so a notebook can plot nuclide evolution
from *either* backend's output without caring which one ran.

``EvolutionResult`` is the in-memory primary artifact -- populated by
``NuclearNetwork.solve()`` as ``self.evolution`` (and surfaced as
``PRIMAT.solve()``'s returned dict's ``"evolution"`` key) whenever
``cfg.output_time_evolution=True``, with *no* disk I/O required to get it.
Disk output (the ``cfg.output_file`` TSV) is a derived convenience: it is
produced by calling :func:`dump_evolution` on that same object, not
something the solver opens a file for directly.

Columns (tab-separated, ``#``-free header line -- see :data:`_CORE_COLUMNS`):
``t_s  a  T_gamma_MeV  T_nue_MeV  T_numu_MeV  T_nutau_MeV  Y_<nuclide> ...``
The ``Y_<nuclide>`` block is network-dependent (small/large have different
nuclide lists), so the header line is the source of truth -- :func:`load_evolution`
reads it dynamically rather than assuming a fixed column count.

Per-reaction flux columns (today small/small_parthenope-only in the legacy
Python writer) and the ``a``/``T_nu`` columns when the active background has
no scale-factor/neutrino-sector tracking (e.g. a minimal custom background)
are deferred/omitted from this unified schema -- see ``PRIMAT.md`` S7.2.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import numpy as np

# Column order for the always-present, cross-backend-comparable block.  The
# Y_<nuclide> block (network-dependent) follows these six.
_CORE_COLUMNS = ("t_s", "a", "T_gamma_MeV", "T_nue_MeV", "T_numu_MeV", "T_nutau_MeV")
_Y_PREFIX = "Y_"


@dataclass
class EvolutionResult:
    """In-memory unified time-evolution result (``PRIMAT.md`` S7.3).

    Attributes
    ----------
    t : np.ndarray
        Cosmic time [s].
    a : np.ndarray
        Scale factor (``np.nan`` everywhere if the background has no
        scale-factor relation, e.g. a minimal custom background).
    T_gamma : np.ndarray
        Photon temperature [MeV].
    T_nu : dict of str -> np.ndarray
        Per-flavour neutrino temperature [MeV], keyed ``"e"``/``"mu"``/
        ``"tau"`` (``np.nan`` arrays if the background tracks no neutrino
        sector).
    Y : dict of str -> np.ndarray
        Per-nuclide mass-fraction abundance, keyed by nuclide name, in
        network order (``n``/``p`` first).
    """
    t: np.ndarray
    a: np.ndarray
    T_gamma: np.ndarray
    T_nu: dict
    Y: dict = field(default_factory=dict)


def dump_evolution(result, path=None):
    """Serialise ``result`` to the shared TSV schema (module docstring).

    Always returns the TSV text; additionally writes it to ``path`` if
    given (relative paths resolve against the current working directory,
    like ``cfg.output_file`` elsewhere in this package). Called by:
    ``NuclearNetwork._write_time_evolution`` (the ``cfg.output_file``
    convenience path); and ``primat-gui``'s download buttons, which call
    this lazily on ``run.evolution`` to produce the file text for
    ``st.download_button(data=...)`` -- never via a tempfile (see
    ``PRIMAT.md`` S7.5).

    Parameters
    ----------
    result : EvolutionResult
    path : str, optional

    Returns
    -------
    str
        The TSV text (header line + one row per time step).
    """
    names = list(_CORE_COLUMNS) + [_Y_PREFIX + s for s in result.Y]
    columns = [result.t, result.a, result.T_gamma,
               result.T_nu["e"], result.T_nu["mu"], result.T_nu["tau"]] + list(result.Y.values())
    data = np.column_stack(columns)

    buf = io.StringIO()
    # comments='' (no leading "# ") matches the convention already used by
    # the rest of this package's TSV writers (e.g. background.py's
    # write_time_evolution), so a plain `header.split("\t")` recovers the
    # column names without stripping a comment marker first.
    np.savetxt(buf, data, delimiter='\t', header="\t".join(names), comments='')
    text = buf.getvalue()

    if path is not None:
        out_path = os.path.abspath(path)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(text)

    return text


def load_evolution(path):
    """Parse the shared TSV schema written by either backend's
    ``dump_evolution``/equivalent writer, returning the same
    :class:`EvolutionResult` structure as ``solve()``'s in-memory
    ``run.evolution`` -- for the case of reloading a previously-saved run
    without re-solving.

    Any column beyond the core block + ``Y_<nuclide>`` block (e.g. a
    backend-specific bonus column) is simply ignored, so a file containing
    extra columns is still loadable.

    Parameters
    ----------
    path : str

    Returns
    -------
    EvolutionResult
    """
    with open(path) as f:
        header = f.readline().strip().split("\t")
        data = np.loadtxt(f)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    col = {name: data[:, i] for i, name in enumerate(header)}
    Y = {name[len(_Y_PREFIX):]: arr for name, arr in col.items()
         if name.startswith(_Y_PREFIX)}

    return EvolutionResult(
        t=col["t_s"], a=col["a"], T_gamma=col["T_gamma_MeV"],
        T_nu={"e": col["T_nue_MeV"], "mu": col["T_numu_MeV"], "tau": col["T_nutau_MeV"]},
        Y=Y,
    )
