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
import os
import zipfile

import numpy as np
import streamlit as st

from pyprimat.network_data import _resample_rate_table, reaction_stoichiometry


def validate_new_reaction(name):
    """Validate a brand-new reaction name and return its readable equation.

    Backs the GUI's "Add a new reaction" pop-up: a user types a reaction name
    in the ``a_b__c_d`` syntax (reactants and products separated by ``__``,
    nuclides within a side by ``_``; ``g`` denotes a photon, ``Bm``/``Bp`` an
    emitted electron/positron, and ``d``/``t``/``a`` alias H2/H3/He4).  The
    name need not exist in the shipped catalog: its stoichiometry is derived
    from the name itself by :func:`pyprimat.network_data.reaction_stoichiometry`,
    which also checks baryon-number and electric-charge conservation and that
    every nuclide token is known.

    Parameters
    ----------
    name : str
        Candidate reaction name, e.g. ``"He3_d__He4_p"``.

    Returns
    -------
    str
        A human-readable equation such as ``"He3 + d -> He4 + p"`` (reactants
        and products joined with ``+`` and separated by ``->``), suitable for
        confirming back to the user what was parsed.

    Raises
    ------
    ValueError
        If the name is empty, cannot be tokenised, has no ``__``/``TO``
        separator, references an unknown nuclide, or does not conserve A/Z.

    Example
    -------
    >>> validate_new_reaction("He3_d__He4_p")
    'He3 + d -> He4 + p'
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("enter a reaction name (e.g. 'He3_d__He4_p').")
    if "__" not in name:
        raise ValueError(
            "name must use the 'a_b__c_d' syntax: reactants and products "
            "separated by a double underscore '__'.")
    try:
        react, prod = reaction_stoichiometry(name)
    except (ValueError, KeyError) as exc:
        raise ValueError(str(exc)) from exc

    def _side(counts):
        return " + ".join(s for s, c in counts.items() for _ in range(int(c)))

    return f"{_side(react)} -> {_side(prod)}"


def parse_rate_upload(fh):
    """Parse an uploaded rate-table file into raw ``(T9, rate, err, header)``.

    Parameters
    ----------
    fh : file-like
        2- or 3-column whitespace-separated text (as produced by
        ``st.file_uploader``, or a plain file object): ``T9 [GK]``, ``rate``,
        and an optional third uncertainty column.  Leading ``#``-prefixed
        lines are the uploader's own header/provenance comment, preserved
        verbatim (see ``header``) rather than discarded.

    Returns
    -------
    (T9, rate, err, header) : tuple of (np.ndarray, np.ndarray, np.ndarray, list[str])
        ``err`` is an all-zero array of the same length when the upload has
        only 2 columns.  ``header`` is the list of the upload's own leading
        ``#``-prefixed lines (possibly empty), preserved so a re-exported zip
        carries the original provenance rather than a generic "custom upload"
        label (see :func:`effective_table_text`).

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
    if hasattr(fh, "read"):
        text = fh.read()
    else:
        text = fh
    if isinstance(text, bytes):
        text = text.decode()
    header = [line for line in text.splitlines() if line.startswith("#")]
    data = np.loadtxt(io.StringIO(text), unpack=True)
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
    return T9, rate, err, header


def effective_table_text(cfg, T9, rate, err, name="custom", source_header=()):
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
        Reaction name, written into the header's ``ref=`` field (only used
        when ``source_header`` is empty).
    source_header : sequence[str]
        The uploader's own ``#``-prefixed header lines (from
        :func:`parse_rate_upload`), preserved verbatim ahead of a bookkeeping
        line, rather than being replaced by a generic label.

    Returns
    -------
    str
        Table text with one or more ``#`` header lines followed by
        ``T9 rate err`` rows on the master grid.
    """
    grid = np.logspace(np.log10(cfg.rate_grid_T9_min),
                        np.log10(cfg.rate_grid_T9_max),
                        cfg.rate_grid_npts)
    rate_grid = _resample_rate_table(T9, rate, grid)
    err_grid = _resample_rate_table(T9, err, grid)
    lines = list(source_header) if source_header else [f"# ref=custom upload ({name})"]
    lines.append(f"# custom rate (reinterpolated): {name}")
    for t9, r, e in zip(grid, rate_grid, err_grid):
        lines.append(f"{t9:.6e} {r:.6e} {e:.6e}")
    return "\n".join(lines)


def export_zip(cfg, custom_network, kept_names, network_filename="custom"):
    """Pack a customisation into an in-memory zip mirroring the repo layout.

    Parameters
    ----------
    cfg : PyPRConfig
        Supplies the master-grid parameters (for resampling replaced tables).
    custom_network : dict
        ``{"removed": [...], "replaced": {name: raw_text, ...},
        "added": {name: raw_text, ...}}``.
    kept_names : sequence[str]
        The full ordered list of reaction names actually in the network after
        removal *and* additions (i.e. ``cfg.network``'s list minus
        ``custom_network["removed"]``, plus ``custom_network["added"]``).  In
        the GUI this is read off the solved network's reaction list, so added
        reactions are already included.
    network_filename : str
        Basename (without ``.txt``) for the network file under ``networks/``,
        e.g. the user-chosen custom-network title (sanitised). Defaults to
        ``"custom"`` for callers that don't have a user-chosen title (e.g. the
        post-run Reactions-tab export of a legacy "Customise Reactions"
        session).

    Returns
    -------
    bytes
        Zip file contents with ``networks/<network_filename>.txt`` (one
        reaction name per line, using the ``name, name_custom.txt`` syntax for
        reactions carrying an uploaded table -- see ``load_network``'s
        ``bare_to_file`` parsing) and one ``tables/<name>/<name>_custom.txt``
        per such reaction (the resampled, on-grid table from
        :func:`effective_table_text`, per-reaction-folder layout matching the
        shipped ``rates/nuclear/tables/<name>/`` tree). Replaced and added
        reactions are written identically; on re-import they are told apart by
        whether the name belongs to the selected network.
    """
    # Replaced (override a kept reaction) and added (brand-new) reactions are
    # both backed by an uploaded table, so they are written to the zip the same
    # way -- merge them into one map of custom tables.
    custom_tables = {**custom_network.get("replaced", {}),
                     **custom_network.get("added", {})}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        lines = []
        for name in kept_names:
            if name in custom_tables:
                lines.append(f"{name}, {name}_custom.txt")
            else:
                lines.append(name)
        zf.writestr(f"networks/{network_filename}.txt", "\n".join(lines) + "\n")
        for name, raw_text in custom_tables.items():
            T9, rate, err, header = parse_rate_upload(raw_text)
            table_text = effective_table_text(cfg, T9, rate, err, name=name,
                                              source_header=header)
            zf.writestr(f"tables/{name}/{name}_custom.txt", table_text)
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
        ``{"kept": [name, ...], "replaced": {name: raw_text, ...}, "title": str}``.
        Removal is implicit: the single file under ``networks/`` only lists
        the reactions that were *kept*, so any reaction of the selected
        network absent from ``kept`` is treated as removed by the caller.
        Brand-new (added) reactions also appear in ``kept`` with their table
        in ``replaced``; the caller tells them apart from replacements by
        checking which ``kept`` names do *not* belong to the selected
        network.  ``title`` is the network file's basename (without
        ``.txt``), recovered without needing a separate metadata file.
    """
    replaced = {}
    try:
        zf = zipfile.ZipFile(fh)
    except zipfile.BadZipFile:
        raise ValueError(
            "the uploaded file is not a valid zip archive (expected one "
            "produced by the 'Save custom network' button)."
        ) from None
    with zf:
        net_files = [info.filename for info in zf.infolist()
                    if info.filename.startswith("networks/")
                    and info.filename.endswith(".txt")]
        if len(net_files) != 1:
            raise ValueError(
                f"expected exactly one file under 'networks/', found {len(net_files)}."
            )
        net_filename = net_files[0]
        title = os.path.basename(net_filename)[: -len(".txt")]
        net_text = zf.read(net_filename).decode()
        kept_names = []
        for line in net_text.splitlines():
            line = line.strip()
            if not line:
                continue
            bare = line.split(",", 1)[0].strip()
            kept_names.append(bare)
        for info in zf.infolist():
            if info.filename.startswith("tables/") and info.filename.endswith("_custom.txt"):
                bare = os.path.basename(info.filename)[: -len("_custom.txt")]
                replaced[bare] = zf.read(info.filename).decode()
    return {"kept": kept_names, "replaced": replaced, "title": title}


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
    parse_rate_upload(raw_text)
    return {"removed": [], "replaced": {reaction_name: raw_text}}
