# -*- coding: utf-8 -*-
"""
pyprimat.gui.custom_rates
==========================

Helpers backing the GUI's "Customise Reactions" panel: parsing a user-uploaded
rate table, resampling it the same way :func:`pyprimat.network_data.load_network`
does, and packing/unpacking a customisation as an in-memory zip so it can be
exported/re-imported without ever touching disk (the GUI may be running on a
read-only deployment).

The customisation itself is the small JSON-serialisable structure threaded
through ``params_items``/``PyPR(custom_network=...)`` (see
``pyprimat.network_data.UpdateNuclearRates``):

    {"removed": [name, ...], "replaced": {name: raw_table_text, ...}}

``raw_table_text`` is the verbatim text of the uploaded file (2 or 3
whitespace-separated columns: T9 [GK], rate, optional uncertainty) -- *not*
pre-resampled -- so :func:`pyprimat.network_data.load_network`'s
``_resample_rate_table`` remains the single interpolation path used both at
solve time and when previewing/exporting the "effective" (on-grid) table.
"""
import io
import zipfile

import numpy as np
import streamlit as st

from pyprimat.network_data import _resample_rate_table


def parse_rate_upload(fh):
    """Parse an uploaded rate-table file into raw ``(T9, rate, err)`` arrays.

    Parameters
    ----------
    fh : file-like
        2- or 3-column whitespace-separated text (as produced by
        ``st.file_uploader``, or a plain file object): ``T9 [GK]``, ``rate``,
        and an optional third uncertainty column.

    Returns
    -------
    (T9, rate, err) : tuple of np.ndarray
        ``err`` is an all-zero array of the same length when the upload has
        only 2 columns.

    Raises
    ------
    ValueError
        If the file does not parse as 2 or 3 numeric columns.

    Notes
    -----
    Emits an ``st.warning`` (not an error -- the table is still usable) if its
    T9 range does not cover the master grid's span (``rate_grid_T9_min``..
    ``rate_grid_T9_max``, default 0.001-10 GK): outside the upload's own range,
    ``_resample_rate_table`` extrapolates the log-log cubic spline, which can
    be inaccurate far from the data.
    """
    data = np.loadtxt(fh, unpack=True)
    if data.ndim != 2 or data.shape[0] not in (2, 3):
        raise ValueError(
            f"expected 2 or 3 columns (T9, rate[, err]), got shape {data.shape}"
        )
    T9, rate = data[0], data[1]
    err = data[2] if data.shape[0] == 3 else np.zeros_like(rate)
    if T9.min() > 0.001 or T9.max() < 10.0:
        st.warning(
            f"Uploaded table spans T9 = [{T9.min():.3g}, {T9.max():.3g}] GK, "
            "narrower than the standard grid [0.001, 10] GK -- values outside "
            "this range are extrapolated."
        )
    return T9, rate, err


def effective_table_text(cfg, T9, rate, err, name="custom"):
    """Return the on-grid table text actually fed to the solver.

    Resamples ``(T9, rate, err)`` onto the master T9 grid
    (``cfg.rate_grid_{npts,T9_min,T9_max}``) with the exact same
    :func:`pyprimat.network_data._resample_rate_table` log-log cubic
    interpolation used by ``load_network``, then formats it as a 3-column text
    table mirroring the shipped ``rates/nuclear/tables/*.txt`` files. This is
    what the Download tab offers for a replaced reaction, so the user can
    verify exactly what was used (as opposed to the raw upload).

    Parameters
    ----------
    cfg : PyPRConfig
        Supplies the master-grid parameters.
    T9, rate, err : np.ndarray
        Raw uploaded arrays, as returned by :func:`parse_rate_upload`.
    name : str
        Reaction name, written into the header's ``ref=`` field.

    Returns
    -------
    str
        Table text with a one-line ``#`` header followed by ``T9 rate err``
        rows on the master grid.
    """
    grid = np.logspace(np.log10(cfg.rate_grid_T9_min),
                        np.log10(cfg.rate_grid_T9_max),
                        cfg.rate_grid_npts)
    rate_grid = _resample_rate_table(T9, rate, grid)
    err_grid = _resample_rate_table(T9, err, grid)
    lines = [f"# ref=custom upload ({name})"]
    for t9, r, e in zip(grid, rate_grid, err_grid):
        lines.append(f"{t9:.6e} {r:.6e} {e:.6e}")
    return "\n".join(lines)


def export_zip(cfg, custom_network, kept_names):
    """Pack a customisation into an in-memory zip mirroring the repo layout.

    Parameters
    ----------
    cfg : PyPRConfig
        Supplies the master-grid parameters (for resampling replaced tables).
    custom_network : dict
        ``{"removed": [...], "replaced": {name: raw_text, ...}}``.
    kept_names : sequence[str]
        The full ordered list of reaction names actually in the network after
        removal (i.e. ``cfg.network``'s list minus ``custom_network["removed"]``).

    Returns
    -------
    bytes
        Zip file contents with ``networks/custom.txt`` (one reaction name per
        line, using the ``name, name_custom.txt`` syntax for replaced
        reactions -- see ``load_network``'s ``bare_to_file`` parsing) and one
        ``tables/<name>_custom.txt`` per replaced reaction (the resampled,
        on-grid table from :func:`effective_table_text`).
    """
    replaced = custom_network.get("replaced", {})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        lines = []
        for name in kept_names:
            if name in replaced:
                lines.append(f"{name}, {name}_custom.txt")
            else:
                lines.append(name)
        zf.writestr("networks/custom.txt", "\n".join(lines) + "\n")
        for name, raw_text in replaced.items():
            T9, rate, err = parse_rate_upload(io.StringIO(raw_text))
            table_text = effective_table_text(cfg, T9, rate, err, name=name)
            zf.writestr(f"tables/{name}_custom.txt", table_text)
    return buf.getvalue()


def import_zip(fh):
    """Rebuild a ``custom_network`` dict from a zip produced by :func:`export_zip`.

    Parameters
    ----------
    fh : file-like
        The uploaded zip file.

    Returns
    -------
    dict
        ``{"removed": [], "replaced": {name: raw_text, ...}}``. ``"removed"``
        is always empty on import: ``networks/custom.txt`` only lists the
        reactions that were *kept*, so removal is implicit (any reaction
        absent from the file is treated as removed by the caller, which
        compares against the full ``cfg.network`` list).
    """
    replaced = {}
    with zipfile.ZipFile(fh) as zf:
        net_text = zf.read("networks/custom.txt").decode()
        kept_names = []
        for line in net_text.splitlines():
            line = line.strip()
            if not line:
                continue
            bare = line.split(",", 1)[0].strip()
            kept_names.append(bare)
        for info in zf.infolist():
            if info.filename.startswith("tables/") and info.filename.endswith("_custom.txt"):
                bare = info.filename[len("tables/"):-len("_custom.txt")]
                replaced[bare] = zf.read(info.filename).decode()
    return {"kept": kept_names, "replaced": replaced}


def import_single(fh, reaction_name):
    """Build a ``custom_network`` fragment that replaces one reaction's table.

    Parameters
    ----------
    fh : file-like
        The uploaded raw rate-table file (2 or 3 columns).
    reaction_name : str
        Bare name of the reaction to replace.

    Returns
    -------
    dict
        ``{"removed": [], "replaced": {reaction_name: raw_text}}``.
    """
    raw_text = fh.read()
    if isinstance(raw_text, bytes):
        raw_text = raw_text.decode()
    # Validate it parses before storing (raises ValueError on malformed input).
    parse_rate_upload(io.StringIO(raw_text))
    return {"removed": [], "replaced": {reaction_name: raw_text}}
