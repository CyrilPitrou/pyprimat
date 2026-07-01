# -*- coding: utf-8 -*-
"""
primat.gui.custom_rates
==========================

Helpers backing the GUI's "Customise Reactions" panel: parsing a user-uploaded
rate table, resampling it the same way :func:`primat.network_data.load_network`
does, and packing/unpacking a customisation as an in-memory zip so it can be
exported/re-imported without ever touching disk (the GUI may be running on a
read-only deployment).

The customisation itself is the small JSON-serialisable structure threaded
through ``params_items``/``PRIMAT(custom_network=...)`` (see
``primat.network_data.UpdateNuclearRates``):

    {"removed": [name, ...], "replaced": {name: raw_table_text, ...}}

``raw_table_text`` is the verbatim text of the uploaded file (2 or 3
whitespace-separated columns: T9 [GK], rate, optional uncertainty) -- *not*
pre-resampled -- so :func:`primat.network_data.load_network`'s
``_resample_rate_table`` remains the single interpolation path used both at
solve time and when previewing/exporting the "effective" (on-grid) table.
"""
import io
import math
import os
import re
import zipfile

import numpy as np
import streamlit as st

from primat.network_data import (
    _resample_rate_table, reaction_stoichiometry, reaction_display_name,
    load_reaction_names, _load_decay_table,
)


def validate_new_reaction(name):
    """Validate a brand-new reaction name and return its readable equation.

    Backs the GUI's "Add a new reaction" pop-up: a user types a reaction name
    in the ``a_b__c_d`` syntax (reactants and products separated by ``__``,
    nuclides within a side by ``_``; ``g`` denotes a photon, ``Bm``/``Bp`` an
    emitted electron/positron, and ``d``/``t``/``a`` alias H2/H3/He4).  The
    name need not exist in the shipped catalog: its stoichiometry is derived
    from the name itself by :func:`primat.network_data.reaction_stoichiometry`,
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


# Shown to the user whenever an uploaded rate table fails to parse (see
# parse_rate_upload's ValueError, caught at every upload site in
# params_form.py) -- the opening lines of a real shipped table
# (B12_t__C15_g), illustrating the expected layout: optional leading
# '#'-comments, then whitespace-separated columns T9 [GK], rate[, error].
RATE_TABLE_FORMAT_EXAMPLE = """\
# B12 + t > C15 + g   [B12_t__C15_g]   ref=TALYS2, Koning et al. 2023
# detailed balance: alpha=1.1104e+11 beta=1.5 gamma=-214.052  Q=18.4456
# T9                 rate                error
1.000000e-03   1.000000e-35   1.000000e+02
1.018629e-03   1.014102e-35   1.000000e+02
1.037605e-03   1.025274e-35   1.000000e+02
"""


def show_rate_format_help():
    """Explain the expected rate-table layout, with a real shipped example.

    Called wherever an upload fails :func:`parse_rate_upload`'s validation,
    so the user immediately sees what is expected instead of just the bare
    parse error.
    """
    st.info(
        "Expected format: optional leading lines starting with `#` "
        "(comments/provenance, ignored), followed by 2 or 3 "
        "whitespace-separated numeric columns -- `T9 [GK]`, `rate`, and an "
        "optional `error` (uncertainty factor). Example (first lines of a "
        "shipped table):"
    )
    st.code(RATE_TABLE_FORMAT_EXAMPLE, language=None)


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


def stamp_upload(name, raw_text):
    """Prepend a provenance header to a freshly uploaded rate-table text.

    Called right where a user's uploaded file is first accepted as a custom
    rate table (the "New rate table for <name>" and "Add a new rate"
    uploaders in ``params_form``), so that *every* later view of this text --
    the "Show rate table" preview popup (:func:`primat.gui.params_form
    ._current_table_text`), the per-reaction "Source" column
    (:func:`primat.network_data._reaction_source_from_lines`, which reads
    this very header back), and a re-exported zip -- carries an unambiguous
    "this is a primat-loaded custom table for reaction X" label, even when
    the uploaded file itself had no ``#`` header at all.

    Parameters
    ----------
    name : str
        Bare reaction name this table is for.
    raw_text : str
        The verbatim uploaded file contents (already validated by
        :func:`parse_rate_upload`).

    Returns
    -------
    str
        ``raw_text`` with two new leading lines: a one-line provenance
        comment naming the reaction (in the same human-readable
        ``"react1 + react2 > prod1 + prod2   [name]"`` form as the shipped
        tables' own headers, see :func:`reaction_display_name`), then a full
        line of ``#`` as a visual separator from whatever header the upload
        itself carried.
    """
    lines = [
        f"# {reaction_display_name(name)}   [{name}]   (custom rate)",
        "#" * 70,
    ]
    return "\n".join(lines) + "\n" + raw_text


def _strip_own_stamp(name, header):
    """Drop :func:`stamp_upload`'s own two-line preamble from ``header``.

    ``header`` (as returned by :func:`parse_rate_upload`) is read straight
    off an already-:func:`stamp_upload`-ed table -- e.g. when
    :func:`export_zip` re-parses a stored "kept" table to recompute its
    on-grid form -- so it starts with ``stamp_upload``'s own bookkeeping
    lines (``"# {react} > {prod}   [{name}]   (custom rate)"`` + a
    ``"#"*70`` fence). Passing those straight through to
    :func:`effective_table_text` as ``source_header`` would duplicate that
    bookkeeping underneath the new reinterpolation header it writes itself;
    only a genuine header carried by the *original* upload (if any),
    following that preamble, is worth preserving.

    Parameters
    ----------
    name : str
        Bare reaction name (must match the preamble's own ``name``).
    header : sequence[str]
        Leading ``#``-lines as returned by :func:`parse_rate_upload`.

    Returns
    -------
    list[str]
    """
    header = list(header)
    own_stamp = f"# {reaction_display_name(name)}   [{name}]   (custom rate)"
    if len(header) >= 2 and header[0] == own_stamp and header[1] == "#" * 70:
        return header[2:]
    return header


def effective_table_text(cfg, T9, rate, err, name="custom", source_header=()):
    """Return the on-grid table text actually fed to the solver.

    Resamples ``(T9, rate, err)`` onto the master T9 grid
    (``cfg.rate_grid_{npts,T9_min,T9_max}``) with the exact same
    :func:`primat.network_data._resample_rate_table` log-log cubic
    interpolation used by ``load_network``, then formats it as a 3-column text
    table mirroring the shipped ``data/nuclear/tables/*.txt`` files. This is
    what the Download tab offers for a replaced reaction, so the user can
    verify exactly what was used (as opposed to the raw upload).

    Parameters
    ----------
    cfg : PRIMATConfig
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
    # Provenance line first, then a long "#"-fence so it's visually obvious
    # that whatever the *original* uploaded file's own header said (preserved
    # verbatim below) is a separate, prior provenance -- not something
    # primat itself wrote.
    lines = [
        f"# {reaction_display_name(name)}   [{name}]   "
        "(custom rate reinterpolated by primat)",
        "#" * 70,
    ]
    lines.extend(source_header)
    for t9, r, e in zip(grid, rate_grid, err_grid):
        lines.append(f"{t9:.6e}   {r:.6e}   {e:.6e}")
    return "\n".join(lines)


def decay_override_table_text(name, rate_s):
    """Synthetic constant-rate table text for a user-overridden decay rate.

    Decay rates are T9-independent (see ``_load_decay_table``); routing an
    override through ``load_network``'s existing ``custom_tables`` mechanism
    (checked *before* the decays.txt branch, so an override always wins, see
    ``load_network``'s rate-loading loop) needs at least 4 points for the
    cubic log-log resampling in :func:`primat.network_data._resample_rate_table`,
    so this repeats the same rate across a handful of grid points rather than
    using decays.txt's single-row format. Used both to feed the solver and
    (rarely, if a decay override could not be expressed via the dedicated
    ``tables/decays.txt`` zip entry -- see :func:`export_zip`) as a fallback
    table representation.
    """
    grid = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)
    lines = [f"# {name}: decay rate overridden in primat (was log(2)/halflife)"]
    lines += [f"{t9:.6e}   {rate_s:.6e}   {1.0:.6e}" for t9 in grid]
    return "\n".join(lines)


def _shipped_table_dir(cfg, name):
    return os.path.join(cfg._resolved_data_dir, "nuclear", "tables", name)


def _match_shipped_file(cfg, name, raw_text):
    """If ``raw_text`` is byte-identical to an on-disk ``tables/<name>/*.txt``
    file, return that file's basename; else ``None``.

    Distinguishes "the user picked an existing alternate shipped table from
    the dropdown" (e.g. a ``*_parthenope3.0.txt`` sibling) -- which keeps its
    real name and content unaltered -- from "the user actually uploaded new
    content", which gets the ``_newnetwork`` treatment in :func:`export_zip`.
    """
    folder = _shipped_table_dir(cfg, name)
    try:
        candidates = os.listdir(folder)
    except OSError:
        return None
    for fname in candidates:
        if not fname.endswith(".txt"):
            continue
        try:
            with open(os.path.join(folder, fname)) as f:
                if f.read() == raw_text:
                    return fname
        except OSError:
            continue
    return None




def export_zip(cfg, custom_network, kept_names, network_filename="custom"):
    """Pack a customisation into an in-memory zip mirroring the repo layout.

    Parameters
    ----------
    cfg : PRIMATConfig
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
        reaction name per line; every per-reaction-table reaction is written
        as ``name, <filename>`` -- the filename is *always* explicit, never
        implied, even for an unmodified shipped default) and one
        ``tables/<name>/<filename>`` per such reaction:

        * An unmodified reaction (still using a shipped table, default or an
          alternate like ``*_parthenope3.0.txt``) keeps its real, unaltered
          filename and content -- so picking an existing alternate from the
          dropdown is never confused with a genuine edit.
        * A genuinely new/uploaded/edited table is written as
          ``<name>_newnetwork.txt``, with the primat-provenance header from
          :func:`effective_table_text`.
        * A decay reaction (Bm/Bp, rate from the shared ``decays.txt``, not a
          per-reaction file) has no table file at all; if its rate has been
          overridden the network-file line is instead ``name, <rate_s>`` --
          a bare number where every other reaction has a filename. There is
          no separate ``decays.txt`` entry in the zip: ``rate_s`` *is* the
          override, right there in the line that names the reaction.

        The zip is fully self-contained: *every* kept reaction's table is
        included, not just replaced/added ones, so it reproduces the exact
        network even on an install whose shipped tables might differ.
    """
    # Replaced (override a kept reaction) and added (brand-new) reactions are
    # both backed by an uploaded table, so they are written to the zip the same
    # way -- merge them into one map of custom tables.
    custom_tables = {**custom_network.get("replaced", {}),
                     **custom_network.get("added", {})}
    decay_table = _load_decay_table(os.path.join(cfg._resolved_data_dir, "nuclear", "tables"))

    # Decay-reaction overrides are pulled out of custom_tables here: they get
    # their rate written inline in the network file, not a per-reaction
    # table file.
    decay_overrides = {}
    for name in list(custom_tables):
        if name in decay_table:
            raw_text = custom_tables.pop(name)
            try:
                T9, rate, err, header = parse_rate_upload(raw_text)
                decay_overrides[name] = float(np.asarray(rate).reshape(-1)[0])
            except (ValueError, IndexError):
                pass

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        lines = []
        for name in kept_names:
            if name in decay_table:
                if name in decay_overrides:
                    lines.append(f"{name}, {decay_overrides[name]:.6e}")
                else:
                    lines.append(name)
                continue
            if name in custom_tables:
                raw_text = custom_tables[name]
                shipped_name = _match_shipped_file(cfg, name, raw_text)
                if shipped_name is not None:
                    # Just an existing shipped table picked from the
                    # dropdown -- not an edit. The shipped default already
                    # carries the "_primat" suffix on disk (see
                    # convert_ac2024_rates.py), so it reads unambiguously as
                    # "primat's own rate"; an already-distinctly-named
                    # alternate (e.g. "*_parthenope3.0.txt") keeps its name.
                    lines.append(f"{name}, {shipped_name}")
                    zf.writestr(f"tables/{name}/{shipped_name}", raw_text)
                    continue
                try:
                    T9, rate, err, header = parse_rate_upload(raw_text)
                    table_text = effective_table_text(
                        cfg, T9, rate, err, name=name,
                        source_header=_strip_own_stamp(name, header))
                except ValueError:
                    table_text = raw_text
                # Prefer the basename already agreed on at upload time
                # ("<name>_custom_<uploaded filename>", see the "New rate
                # table for <name>" uploader in params_form.py) so a
                # downloaded zip's filename matches what the dialog itself
                # showed throughout editing. Only legacy callers with no
                # "filenames" entry (e.g. the post-run Reactions-tab export
                # of an old-style "Customise Reactions" session) fall back
                # to a generic name suffixed with this network's own title.
                fname = (custom_network.get("filenames", {}).get(name)
                          or f"{name}_{network_filename}.txt")
                lines.append(f"{name}, {fname}")
                zf.writestr(f"tables/{name}/{fname}", table_text)
                continue
            # Unmodified shipped-default reaction: copy its on-disk table
            # verbatim (no resampling needed, it is already a valid rate
            # file) under its own real name, so the zip does not depend on
            # the importing install's own shipped tables/<name>/ folder.
            path = os.path.join(cfg._resolved_data_dir, "nuclear", "tables", name,
                                f"{name}_primat.txt")
            try:
                with open(path) as f:
                    table_text = f.read()
            except OSError:
                lines.append(name)
                continue
            fname = f"{name}_primat.txt"
            lines.append(f"{name}, {fname}")
            zf.writestr(f"tables/{name}/{fname}", table_text)
        zf.writestr(f"networks/{network_filename}.txt", "\n".join(lines) + "\n")
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
        ``{"kept": [name, ...], "replaced": {name: raw_text, ...},
        "filenames": {name: basename, ...}, "decay_overrides":
        {name: rate_s, ...}, "title": str}``.
        Removal is implicit: the single file under ``networks/`` only lists
        the reactions that were *kept*, so any reaction of the selected
        network absent from ``kept`` is treated as removed by the caller.
        Brand-new (added) reactions also appear in ``kept`` with their table
        in ``replaced``; the caller tells them apart from replacements by
        checking which ``kept`` names do *not* belong to the selected
        network.  ``replaced`` carries one entry per kept reaction that has a
        per-reaction table file in the zip (shipped-default, alternate, or
        genuinely new -- see :func:`export_zip`); decay reactions have no
        such file and instead contribute to ``decay_overrides`` if their
        network-file line carries a bare number (the overridden rate)
        instead of a filename. ``filenames`` carries the zip's own basename
        for each such reaction (e.g. ``"B8_d__Be7_He3_primat.txt"``), so the
        Reactions tab's "File" column shows *something* meaningful after a
        round trip rather than ``None`` -- this is purely the in-zip
        filename, not a real on-disk path. ``title`` is the network file's
        basename (without ``.txt``), recovered without needing a separate
        metadata file.
    """
    replaced = {}
    filenames = {}
    decay_overrides = {}
    try:
        zf = zipfile.ZipFile(fh)
    except zipfile.BadZipFile:
        raise ValueError(
            "the uploaded file is not a valid zip archive (expected one "
            "produced by the 'Download network details' button)."
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
            parts = re.split(r'[, ]+', line, maxsplit=1)
            bare = parts[0].strip()
            kept_names.append(bare)
            if len(parts) > 1:
                # A decay reaction's overridden rate is written directly as
                # a bare number in the spot every other reaction uses for a
                # filename (see export_zip) -- no separate decays.txt entry.
                try:
                    decay_overrides[bare] = float(parts[1].strip())
                except ValueError:
                    pass  # it's a filename, not a rate; handled below.
        for info in zf.infolist():
            if info.filename.startswith("tables/") and info.filename.count("/") == 2:
                # "tables/<name>/<filename>" -- any per-reaction table file,
                # default-named, alternate-shipped, or genuinely new.
                bare, fname = info.filename.split("/")[1:3]
                replaced[bare] = zf.read(info.filename).decode()
                filenames[bare] = fname
    return {"kept": kept_names, "replaced": replaced, "filenames": filenames,
            "decay_overrides": decay_overrides, "title": title}


def kept_to_custom_network(cfg, kept, replaced, decay_overrides=None, filenames=None):
    """Build the ``{"removed", "replaced", "added"}`` dict from an imported zip.

    Shared by both the sidebar's "Import custom network" dialog
    (``primat.gui.params_form``) and the post-run Reactions tab's own
    importer (``primat.gui.panels``) -- lives here, not in ``params_form``,
    so ``panels`` can call it without a circular import (``params_form``
    already imports from ``panels``).

    ``removed`` is computed against the *full, unfiltered* large-network
    reaction list -- not some amax-restricted view -- so that every
    catalog reaction absent from ``kept`` is actually excluded from the
    solved network. (An earlier version derived an "implied amax" from the
    heaviest category among ``kept`` and only marked reactions *within* that
    band as removed; every reaction above it was then neither removed nor
    kept, so ``UpdateNuclearRates`` silently treated "not removed" as "keep"
    and the imported network solved with hundreds of unwanted extra
    reactions -- this is why that derivation is gone.)

    Parameters
    ----------
    cfg : PRIMATConfig
        Used only to resolve ``data/nuclear/networks/large.txt``.
    kept : sequence[str]
        Reaction names kept in the imported network.
    replaced : dict[str, str]
        ``{name: raw_table_text}`` for every reaction the zip carried a table
        for (the exported zip's format includes one for *every* kept
        reaction, not just genuinely customised ones -- see ``export_zip``).
    decay_overrides : dict[str, float], optional
        ``{name: rate_s}`` parsed from the zip's ``tables/decays.txt`` (see
        :func:`import_zip`). Only entries that actually differ from the
        shipped ``decays.txt`` rate are turned into a synthetic
        ``replaced`` table entry (:func:`decay_override_table_text`) --
        an unmodified decay reaction needs no override at all.
    filenames : dict[str, str], optional
        ``{name: basename}`` from :func:`import_zip`, the in-zip filename
        for each reaction in ``replaced`` -- threaded through into the
        returned dict's own ``"filenames"`` key purely so the Reactions
        tab's "File" column has *something* to show after a round trip
        (``UpdateNuclearRates`` reads it via ``custom_network["filenames"]``;
        see its docstring) instead of ``None``.

    Returns
    -------
    dict
        ``{"removed": [...], "replaced": {...}, "added": {...},
        "filenames": {...}}``, the shape ``UpdateNuclearRates`` expects.
    """
    entries = load_reaction_names(cfg, "large")
    bare_names = {re.split(r'[, ]+', e, maxsplit=1)[0].strip() for e in entries}
    kept_set = set(kept)
    removed = sorted(bare_names - kept_set)
    added = {n: replaced[n] for n in kept_set - bare_names if n in replaced}
    true_replaced = {n: t for n, t in replaced.items() if n not in added}
    if decay_overrides:
        shipped = _load_decay_table(os.path.join(cfg._resolved_data_dir, "nuclear", "tables"))
        for name, rate_s in decay_overrides.items():
            shipped_entry = shipped.get(name)
            if shipped_entry is None or not math.isclose(
                rate_s, shipped_entry[0], rel_tol=1e-9, abs_tol=0.0
            ):
                true_replaced[name] = decay_override_table_text(name, rate_s)
    true_filenames = {n: f for n, f in (filenames or {}).items() if n in true_replaced}
    return {"removed": removed, "replaced": true_replaced, "added": added,
            "filenames": true_filenames}


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
