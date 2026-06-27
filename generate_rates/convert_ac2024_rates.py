# -*- coding: utf-8 -*-
"""
convert_ac2024_rates.py
=======================
Build a per-reaction rate-table set on the standard 500-point log-uniform T9
grid from two sources:

  1. the tabulated ``BBNRatesAC2024.dat`` compilation (interpolated), and
  2. the analytic rate formulas, hard-coded in the ``_ANALYTIC_REACTIONS`` table
     below (originally from ``PRIMAT-main.m``'s ``DefineAnalyticRates``), each
     evaluated directly on the grid.

Together these cover the reactions PRIMAT splits between a tabulated file and
analytic forms; the analytic blocks are exactly the ones *not* in the tabulated
file (PRIMAT comments out an analytic form once it moves to the table).  All
output files use the ``<reactants>TO<products>`` naming convention.

Because the analytic formulas are hard-coded, the full rate set is regenerable
from ``BBNRatesAC2024.dat`` alone -- the ``PRIMATreference/`` folder is *not*
needed at run time.  The Mathematica extractor is kept only to refresh the
hard-coded table when PRIMAT-main.m changes (``--dump-analytic``).

Source format (one block per reaction)::

    *- n + p > d + g ;            <- reactants ">" products, optional short name
    *%And06                       <- reference tag
    ! ... optional comment lines (start with ! or #)
       2.22457 4.71614e+09 1.5 -25.815      <- Q, alpha, beta, gamma
         0.001    4.4140E+04    1.0045E+00  <- T9, rate=exp(mu), error=exp(sigma)
         ...                                   (60 rows, grid 0.001..10)

Output:
  * ``primat/rates/nuclear/tables/<name>/<name><suffix>.txt`` for every
    *non-decay* reaction (one folder per reaction, ``<name>`` being the
    ``<reactants>TO<products>``-derived bare name; ``<suffix>`` defaults to
    ``"_primat"``, so the shipped PRIMAT-default table is e.g.
    ``n_p__d_g_primat.txt``): a header line (``#`` comment, ignored by
    ``numpy.loadtxt``) recording the reaction, its reference and its
    detailed-balance coefficients, then three columns ``T9  rate  error`` on
    the 500-point grid.  Alternate-source variants (a different ``--suffix``,
    e.g. a Parthenope-extracted table) land as a sibling file in the same
    per-reaction folder, e.g. ``tables/n_p__d_g/n_p__d_g_parthenope3.0.txt``.
  * ``primat/rates/nuclear/tables/decays.txt`` for every radioactive-decay
    reaction (Bm/Bp on the products side): one row each with
    ``name  halflife_s  rate_s^-1  uncertainty  ref`` -- decay rates don't
    depend on T9, so a 500-row table per reaction would be redundant (see
    :func:`write_decay_file`). Always unsuffixed: ``--suffix`` never applies
    to it, since ``network_data._load_decay_table`` hardcodes this filename.
  * ``<datadir>/detailed_balance.csv``: reaction, Q, alpha, beta, gamma for all
    reactions (the backward rate is ``alpha * T9**beta * exp(gamma/T9)`` times
    the forward rate).
  * ``primat/rates/nuclear/networks/large.txt``: the reaction names from
    ``<datadir>/reactions_large.csv``, one per line, each paired with its
    explicit filename (``name, name<suffix>.txt``) per ``load_network``'s
    "never imply the filename" convention -- except decay reactions, written
    bare (see above).

The naming convention ``<reactants>TO<products>`` (e.g. ``n + p > d + g`` ->
``npTOdg``) cleanly separates the initial and final state, removing the
ambiguity of the old prefix-free names (``npdg``).

Usage::

    python generate_rates/convert_ac2024_rates.py \
        --input generate_rates/BBNRatesAC2024.dat \
        --primat generate_rates/PRIMAT-Main.m
"""
import argparse
import math
import os
import re
import sys

import numpy as np
from scipy.interpolate import interp1d

# Make the script self-contained when run as `python generate_rates/convert_ac2024_rates.py`
# from the repo root: put both this script's directory (for the sibling
# `nuclide_table` / `nuclear_data` imports) and the repo root (for
# `from primat.config import PRIMATConfig`) on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):     # generate_rates/ and repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Standard target grid: single-sourced from PyPRIMAT's own master-grid
# defaults (primat.config.DEFAULT_PARAMS["rate_grid_*"]) so the generator
# and the runtime grid-resampling in pypr.nuclear.UpdateNuclearRates can never
# silently drift apart.  DEFAULT_PARAMS is a plain dict,
# so importing it has no side effects (no PRIMATConfig instantiation).
from primat.config import DEFAULT_PARAMS
from primat.network_data import _RATE_SYNTAX_, _format_name
GRID_NPTS = DEFAULT_PARAMS["rate_grid_npts"]
GRID_T9_MIN = DEFAULT_PARAMS["rate_grid_T9_min"]
GRID_T9_MAX = DEFAULT_PARAMS["rate_grid_T9_max"]

# Output directory for per-reaction rate tables (.txt). Hardcoded -- this is
# the only location PyPRIMAT's load_network reads rate tables from, so there
# is no use case for writing them elsewhere.
TABDIR = "primat/rates/nuclear/tables"

# Output file listing the large-network reactions, one name per line: the
# first column of reactions_large.csv, kept in sync with it by
# write_network_files.
LARGE_NETWORK_FILE = "primat/rates/nuclear/networks/large.txt"

# Numbers may use Fortran 'D'/'d' double-precision exponents (e.g. 1.1133D+10).
_NUM = r"[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eEdD][-+]?[0-9]+)?"
_FLOATS_RE = re.compile(rf"^\s*(?:{_NUM})(?:\s+{_NUM})*\s*$")


def _to_float(token):
    return float(token.replace("D", "e").replace("d", "e"))


def standard_grid():
    return np.logspace(np.log10(GRID_T9_MIN), np.log10(GRID_T9_MAX), GRID_NPTS)


# ---------------------------------------------------------------------------
# Reference-tag expansion
#
# BBNRatesAC2024.dat and PRIMAT-Main.m tag each rate's source with a short,
# cryptic key (e.g. ``And06``, ``CF88``). _REFERENCE_NAMES maps every such key
# to a human-readable citation, so the headers written into the per-reaction
# rate tables (write_reaction_file, write_analytic_file) and decays.txt
# (write_decay_file) name the actual source instead of the bare tag.  Compound
# tags joining two references with '&' (e.g. ``CF88&MF89``, ``TUNL&Cam08``)
# are expanded part-by-part by expand_ref below; tags with no entry here
# (e.g. the cross-reference marker ``=li7pa``) are passed through unchanged.
# ---------------------------------------------------------------------------
_REFERENCE_NAMES = {
    "NACRE": "NACRE, Angulo et al. 1999",
    "NACRE II": "NACRE, Xu et al. 2010, 2011",
    "DAACV04": "Descouvemont et al. 2004",
    "ILCCF10": "Iliadis et al. 2010",
    "CF88": "Caughlan& Fowler 1988",
    "MF89": "Malaney& Fowler 1989",
    "Boy93": "Boyd et al. 1993",
    "Bal95": "Balbes et al. 1995",
    "Hei98": "Heil et al. 1998",
    "Rau94": "Rauscher et al. 1994",
    "Des99": "Descouvemont 1999",
    "Bea01": "Beaumel et al. 2001",
    "Des99Bea01": "Descouvemont 1999 & Beaumel et al. 2001",
    "Tan03": "Tang et al. 2003",
    "Tang03": "Tang et al. 2003",
    "Wan91": "Wang et al. 1991",
    "Efr96": "Efros et al. 1996",
    "Wie87": "Wiescher et al. 1987",
    "Bar97": "Bardayan& Smith 1997",
    "Bar97C": "Bardayan& Smith 1997",
    "Koe91": "Koehler& Graff 1991",
    "And06": "Ando et al. 2006",
    "Ser04": "Serpico et al. 2004",
    "Wag69": "Wagoner 1969",
    "Has09": "Hashimoto et al. 2009",
    "Has09c": "Hashimoto et al. 2009",
    "Wie89": "Wiescher et al. 1989",
    "FK90": "Fukugita& Kajino 1990",
    "Bru91": "Brune et al. 1991",
    "Bec92": "Becchetti et al. 1992",
    "Iga95": "Igashira et al. 1995",
    "Cyb08": "Cyburt& Davids 2008",
    "Miz00": "Mizoi et al. 2000",
    "Nag06": "Nagai et al. 2006",
    "Men12": "Mendes et al. 2012",
    "Kaw91": "Kawano et al. 1991",
    "Cam08": "Camargo et al. 2008",
    "Ili16": "Iliadis et al. 2016",
    "Rij19": "Rijal et al. 2019",
    "Gar21": "Gariazzo et al. 2021",
    "Yeh22": "Yeh et al. 2022",
    "Trezzi2017": "Trezzi et al. (Luna) 2017",
    "Moscoso2021": "Moscoso et al. 2021",
    # Not in the user-supplied list but heavily used by decays.txt -- names
    # taken from the source comments right above _ANALYTIC_REACTIONS (Aud03 =
    # the half-lives PRIMAT-Main.m hard-codes, sourced from Audi 2003; Nubase
    # = the later NUBASE2020 half-lives used for the Decay-Time-era nuclides).
    "Aud03": "Audi et al. 2003",
    "Nubase": "NUBASE2020, Kondev et al. 2021",
    "Gom17": "Gómez et al. 2017",
    "deSouza19a": "de Souza et al. 2019",
    "deSouza19b": "de Souza et al. 2019",
    "deSouza2020": "de Souza et al. 2020",
    "TALYS2": "TALYS2, Koning et al. 2023",
    "Bar16": "Barbagallo et al. 2016",
    "CGXSV12": "Coc et al. 2012",
}


def expand_ref(ref):
    """Expand a (possibly compound) AC2024/PRIMAT-Main.m reference tag.

    A plain tag (e.g. ``"CF88"``) is looked up directly in
    :data:`_REFERENCE_NAMES`. A compound tag joining several references with
    ``'&'`` (e.g. ``"CF88&MF89"``) is split and each part expanded
    independently, then rejoined with ``" & "``. Unrecognised tags (no entry
    in the dict -- e.g. the AC2024 cross-reference marker ``"=li7pa"``) are
    returned unchanged so nothing is silently lost.
    """
    if not ref:
        return ref
    if ref in _REFERENCE_NAMES:
        return _REFERENCE_NAMES[ref]
    if "&" in ref:
        return " & ".join(_REFERENCE_NAMES.get(part, part) for part in ref.split("&"))
    return ref


# ---------------------------------------------------------------------------
# Naming conventions
#
# Two different naming systems are used by design, for two different audiences:
#   * Rate FILENAMES (<reactants>TO<products>.txt, built by reaction_name below)
#     use the short single-letter tokens a/d/t for He4/H2/H3, matching AC2024's
#     own spelling and keeping filenames short.
#   * The CSVs (reactions_large.csv etc., built in write_network_files via
#     nuclide_table.resolve_token/canonical_name) spell nuclides out in full
#     (He4/H2/H3/...), matching PyPRIMAT's runtime ``Nuclides`` keys.
# _CANON_TOKEN below only affects filenames; resolve_token/canonical_name in
# nuclide_table.py are the single source of truth for the CSV spelling.
# ---------------------------------------------------------------------------

# Canonical short tokens so the same nuclide always yields the same name,
# whether the source spells it 'He4' or 'a' (AC2024 uses a/d/t; PRIMAT-main.m
# mixes He4/a, etc.).
_CANON_TOKEN = {"He4": "a", "H2": "d", "H3": "t"}


def reaction_name(reactants, products):
    """Build the canonical reaction name in :data:`primat.network_data._RATE_SYNTAX_`.

    Default ``"spaced"`` syntax joins tokens with ``"_"`` and separates the
    reactant/product sides with ``"__"`` (e.g. ``"n_p__d_g"``); legacy
    ``"compact"`` syntax concatenates tokens and separates sides with the
    literal ``"TO"`` (e.g. ``"npTOdg"``).  See :func:`primat.network_data._format_name`.
    """
    def canon(side):
        return [_CANON_TOKEN.get(t, t) for t in side]
    return _format_name(canon(reactants), canon(products), syntax=_RATE_SYNTAX_)


def _parse_side(text):
    """'n + p' -> ['n', 'p'];  'He4 + 2n' -> ['He4', 'n', 'n'] (order preserved)."""
    out = []
    for tok in text.split("+"):
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"^(\d+)(.+)$", tok)          # multiplicity prefix, e.g. 2n, 2a
        if m:
            out.extend([m.group(2)] * int(m.group(1)))
        else:
            out.append(tok)
    return out


def parse_blocks(path):
    """Parse the AC2024 file into a list of reaction dicts.

    Each dict has: reactants, products, name, ref, Q, alpha, beta, gamma,
    T9 (array), rate (array), error (array).
    """
    lines = open(path, encoding="latin-1").read().splitlines()

    # indices of reaction-code lines ('*-')
    starts = [i for i, l in enumerate(lines) if l.startswith("*-")]
    starts.append(len(lines))

    blocks = []
    for a, b in zip(starts[:-1], starts[1:]):
        code = lines[a]
        # '*- n + p > d + g ; tpg'  ->  reactants/products
        body = code[2:].split(";")[0]
        if ">" not in body:
            raise ValueError(f"cannot parse reaction code: {code!r}")
        lhs, rhs = body.split(">", 1)
        reactants, products = _parse_side(lhs), _parse_side(rhs)

        ref = ""
        abc = None
        data = []
        for l in lines[a + 1:b]:
            s = l.strip()
            if s.startswith("*%"):
                ref = s[2:].strip()
                continue
            if not s or s[0] in "*!#":
                continue
            if not _FLOATS_RE.match(s):
                continue
            vals = [_to_float(x) for x in s.split()]
            if abc is None and len(vals) == 4:
                abc = vals                 # Q, alpha, beta, gamma
            elif len(vals) == 3:
                data.append(vals)
        if abc is None or not data:
            raise ValueError(f"incomplete block for {code!r}")

        data = np.array(data, float)
        blocks.append(dict(
            reactants=reactants, products=products,
            name=reaction_name(reactants, products),
            ref=ref, Q=abc[0], alpha=abc[1], beta=abc[2], gamma=abc[3],
            T9=data[:, 0], rate=data[:, 1], error=data[:, 2],
        ))
    return blocks


def interp_loglog(x_src, y_src, x_dst, kind="cubic"):
    """Interpolate y(x) on the target grid in log-log space.

    Reaction rates and their (multiplicative) errors are smooth and positive in
    log-log, so this is the physically appropriate scheme; it reproduces the
    existing key-reaction tables to a few parts in 1e5.  Falls back to linear
    interpolation of y vs log10(x) if any value is non-positive.
    """
    lx_src, lx_dst = np.log10(x_src), np.log10(x_dst)
    if np.all(y_src > 0):
        f = interp1d(lx_src, np.log10(y_src), kind=kind,
                     bounds_error=False, fill_value="extrapolate")
        return 10.0 ** f(lx_dst)
    f = interp1d(lx_src, y_src, kind="linear",
                 bounds_error=False, fill_value="extrapolate")
    return f(lx_dst)


def write_reaction_file(block, grid, outdir, suffix=""):
    """Write one reaction's rate table to a .txt file.

    Args:
        block  : reaction dict from parse_blocks (keys: T9, rate, error, name, …).
        grid   : 1-D T9 array to write.  When equal to block["T9"] (i.e. with
                 --keep-source-grid) the rates are written directly without
                 reinterpolation; otherwise log-log cubic interpolation is applied.
        outdir : directory containing one subfolder per reaction.
        suffix : appended to the reaction name before ".txt" (alternate-source
                 variants land as a sibling file inside the same per-reaction
                 folder as the PRIMAT-default table -- the per-reaction-folder
                 mechanism for multiple candidate tables, see network_data.py's
                 available_rate_tables()).
    """
    if np.array_equal(grid, block["T9"]):
        # Native grid — no interpolation needed.
        rate = block["rate"]
        err  = block["error"]
    else:
        rate = interp_loglog(block["T9"], block["rate"], grid)
        err  = interp_loglog(block["T9"], block["error"], grid)
    header = (
        f"{' + '.join(block['reactants'])} > {' + '.join(block['products'])}"
        f"   [{block['name']}]   ref={expand_ref(block['ref'])}\n"
        f"detailed balance: alpha={block['alpha']:.6g} beta={block['beta']:.6g} "
        f"gamma={block['gamma']:.6g}  Q={block['Q']:.6g}\n"
        f"T9                 rate                error"
    )
    reaction_dir = os.path.join(outdir, block["name"])
    os.makedirs(reaction_dir, exist_ok=True)
    path = os.path.join(reaction_dir, block["name"] + suffix + ".txt")
    np.savetxt(path, np.column_stack([grid, rate, err]),
               fmt=["%.6e", "%.6e", "%.6e"], delimiter="   ", header=header)
    return path


# ---------------------------------------------------------------------------
# Analytic reactions from PRIMAT-main.m (function DefineAnalyticRates)
# ---------------------------------------------------------------------------
# Each block there reads:
#     source="..."; reac="reactants > products ; name"; f=...;
#     forward[T9_]:=<Mathematica expr>;  AddReaction[reac,source,f,forward,Bool];
# We translate the Mathematica forward expression to a Python callable of T9.

_MFUNC = {"Exp": "exp", "Log10": "log10", "Log": "log", "Sqrt": "sqrt",
          "Max": "maximum", "Min": "minimum"}
_MNS = {"exp": np.exp, "log10": np.log10, "log": np.log, "sqrt": np.sqrt,
        "maximum": np.maximum, "minimum": np.minimum, "where": np.where,
        "__builtins__": {}}
_MTOK = re.compile(r"\s*((?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:e[-+]?[0-9]+)?"
                   r"|[A-Za-z][A-Za-z0-9]*|>=|<=|==|[-+*/^(),\[\]{}=<>])")


def _strip_mathematica_comments(text):
    """Remove (* ... *) comments, honouring nesting."""
    out, depth, i = [], 0, 0
    while i < len(text):
        if text[i:i + 2] == "(*":
            depth += 1
            i += 2
        elif text[i:i + 2] == "*)" and depth > 0:
            depth -= 1
            i += 2
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out)


def _match_bracket(s, i, op, cl):
    depth = 0
    while i < len(s):
        if s[i] == op:
            depth += 1
        elif s[i] == cl:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unbalanced {op}{cl} in {s!r}")


def _split_top(s, sep=","):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def _convert_E_pow(s):
    """Rewrite Mathematica E^(...) as Exp[...] so it becomes a normal call."""
    out, i = "", 0
    while i < len(s):
        if s[i:i + 2] == "E^" and i + 2 < len(s) and s[i + 2] == "(":
            j = _match_bracket(s, i + 2, "(", ")")
            out += "Exp[" + _convert_E_pow(s[i + 3:j]) + "]"
            i = j + 1
        else:
            out += s[i]
            i += 1
    return out


def _is_value(tok):       # token that can start/end an implicit-mult value
    return tok[0].isalnum() or tok[0] == "."


def _leaf_to_python(expr):
    """Translate a Mathematica arithmetic leaf (no With/If) to Python.

    Handles the two features that make Mathematica != Python:

    1. **Implicit multiplication** -- Mathematica writes ``214. T9^0.075`` and
       ``2 Pi`` meaning ``214.*T9**0.075`` and ``2*Pi``.  We tokenise and insert
       a ``*`` between any value-ending token (number, identifier, ``)`` or
       ``]``) and any value-starting token (number, identifier or ``(``).  A
       function call like ``Exp[`` is *not* split, because ``[`` is not a
       value-start (and is handled by the bracket replacement below).
    2. **Different syntax** -- ``E^(...)`` is pre-rewritten to ``Exp[...]`` by
       :func:`_convert_E_pow`; function names map to numpy (Exp->exp, ...),
       ``[]`` call brackets become ``()``, and ``^`` becomes ``**``.
    """
    toks = []
    s, i = _convert_E_pow(expr), 0
    # Tokenise the expression (numbers, identifiers, operators, brackets),
    # skipping whitespace.
    while i < len(s):
        m = _MTOK.match(s, i)
        if not m:
            if s[i].isspace():
                i += 1
                continue
            raise ValueError(f"cannot tokenise {s[i:][:20]!r}")
        toks.append(m.group(1))
        i = m.end()
    out = []
    for k, tk in enumerate(toks):
        # Insert the implicit-multiplication '*' (see rule 1 above).
        if k and (_is_value(toks[k - 1]) or toks[k - 1] in (")", "]")) \
                and (_is_value(tk) or tk == "("):
            out.append("*")
        out.append(_MFUNC.get(tk, tk))           # Exp->exp, Max->maximum, ...
    # Convert call brackets and the power operator to Python.
    return "".join(out).replace("[", "(").replace("]", ")").replace("^", "**")


def _mathematica_to_python(expr):
    """Translate a Mathematica forward[T9_] body (With/If/arithmetic) to Python.

    ``With[{...}, body]`` and ``If[c, a, b]`` may appear anywhere (including
    nested inside arithmetic), so each such construct is translated recursively
    and spliced back in via a placeholder before the surrounding arithmetic
    leaf is tokenised.
    """
    e = expr.strip()
    while e.startswith("(") and _match_bracket(e, 0, "(", ")") == len(e) - 1:
        e = e[1:-1].strip()

    subs = {}
    while True:
        m = re.search(r"\b(With|If)\[", e)
        if not m:
            break
        lb = m.end() - 1
        rb = _match_bracket(e, lb, "[", "]")
        inner = e[lb + 1:rb]
        if m.group(1) == "With":
            seg = _split_top(inner)
            binds, body = seg[0].strip()[1:-1], ",".join(seg[1:])
            names, vals = [], []
            for b in _split_top(binds):
                nm, v = b.split("=", 1)
                names.append(nm.strip())
                vals.append(_mathematica_to_python(v))
            trans = (f"(lambda {','.join(names)}: {_mathematica_to_python(body)})"
                     f"({','.join(vals)})")
        else:  # If
            c, a, b = _split_top(inner)[:3]
            trans = (f"where({_mathematica_to_python(c)},"
                     f"{_mathematica_to_python(a)},{_mathematica_to_python(b)})")
        ph = f"SUBZZ{len(subs)}"
        subs[ph] = trans
        e = e[:m.start()] + ph + e[rb + 1:]

    py = _leaf_to_python(e)
    for ph, trans in subs.items():
        py = re.sub(rf"\b{ph}\b", f"({trans})", py)
    return py


def analytic_rate_function(forward_expr):
    """Return a callable f(T9_array) for a Mathematica forward[T9_] expression."""
    py = _mathematica_to_python(forward_expr.replace("*^", "e"))
    return eval("lambda T9: " + py, dict(_MNS))   # noqa: S307 (trusted source)


def extract_analytic_from_primat(primat_path):
    """Re-extract the analytic reactions from a PRIMAT-main.m source file.

    This is *only* needed to regenerate the hard-coded ``_ANALYTIC_REACTIONS``
    table below when PRIMAT-main.m changes (use ``--dump-analytic``).  A normal
    run uses the hard-coded table and never touches ``PRIMATreference/``, so the
    rate files can be regenerated from ``BBNRatesAC2024.dat`` alone.

    The analytic reactions live in PRIMAT's ``DefineAnalyticRates`` function as
    blocks ``source="..."; reac="..."; forward[T9_]:=<expr>; AddReaction[...]``.
    The variables are reassigned before each ``AddReaction`` call, so we split on
    ``AddReaction[`` and take the last assignment of each in the preceding chunk.

    Returns
    -------
    list[tuple[str, str, float, str]]
        ``(source, reac, f, forward)`` per reaction, where ``reac`` is the PRIMAT
        string "reactants > products ; name", ``f`` is the (constant,
        temperature-independent) multiplicative uncertainty factor passed to
        ``AddReaction`` -- PRIMAT sets it just above each call as ``f=2.``,
        ``f=3.``, ... -- and ``forward`` is the raw Mathematica ``forward[T9]``
        expression (whitespace collapsed to single spaces, which preserves
        Mathematica's space-as-multiplication).

    The uncertainty factor is constant in T9: the rate's 1-sigma band is
    ``[rate/f, rate*f]``, exactly as for a tabulated reaction whose ``error``
    column would be the constant ``f``.  Decays carry no uncertainty (PRIMAT
    passes the literal ``1`` rather than setting ``f=...``), so a chunk with no
    ``f=`` assignment defaults to ``f=1.0``.
    """
    text = open(primat_path, encoding="latin-1").read()
    start = text.index("DefineAnalyticRates")
    end = text.index("TabulatedReactions", start)
    region = _strip_mathematica_comments(text[start:end])

    entries = []
    for chunk in region.split("AddReaction[")[:-1]:
        rm = list(re.finditer(r'reac\s*=\s*"([^"]*)"', chunk))
        sm = list(re.finditer(r'source\s*=\s*"([^"]*)"', chunk))
        # The uncertainty factor is the last standalone 'f = <number> ;'
        # assignment in the chunk (PRIMAT reassigns it before every reaction).
        fm_unc = list(re.finditer(r"\bf\s*=\s*([^;]+?)\s*;", chunk))
        fm = list(re.finditer(r"forward\[T9_\]\s*:?=\s*(.+?);", chunk, re.S))
        if not rm or not fm:
            continue
        reac = rm[-1].group(1).strip()
        ref = sm[-1].group(1).strip() if sm else ""
        f = _to_float(fm_unc[-1].group(1).strip()) if fm_unc else 1.0
        forward = re.sub(r"\s+", " ", fm[-1].group(1).strip())
        if ">" not in reac.split(";")[0]:        # skip pure decays without "reac>prod"
            continue
        entries.append((ref, reac, f, forward))
    return entries


def build_analytic_blocks(entries):
    """Turn ``(source, reac, forward)`` entries into writable rate blocks.

    ``entries`` is normally the hard-coded ``_ANALYTIC_REACTIONS``, but may also
    be the output of :func:`extract_analytic_from_primat`.  Each Mathematica
    ``forward`` expression is translated to a Python callable of T9; an entry
    whose translation fails is skipped and reported rather than aborting.

    Returns
    -------
    (blocks, skipped) : tuple[list[dict], list[tuple[str, str]]]
        ``blocks`` carry reactants, products, name, ref, rate (callable) and the
        original expr; ``skipped`` is ``(name, error_message)`` for any failures.
    """
    blocks, skipped = [], []
    for ref, reac, f, forward in entries:
        body = reac.split(";")[0]
        if ">" not in body:
            continue
        lhs, rhs = body.split(">", 1)
        reactants, products = _parse_side(lhs), _parse_side(rhs)
        name = reaction_name(reactants, products)
        try:
            rate = analytic_rate_function(forward)
        except Exception as exc:                 # noqa: BLE001
            skipped.append((name, str(exc)))
            continue
        blocks.append(dict(reactants=reactants, products=products, name=name,
                           ref=ref, f=f, rate=rate, expr=forward))
    return blocks, skipped


def write_analytic_file(block, grid, outdir, suffix=""):
    # block["rate"](grid) may be a T9-independent scalar (constants, decays) or
    # an array already shaped like `grid`; broadcast_to states that intent
    # explicitly (replaces the `* np.ones_like(grid)` idiom).
    rate = np.array(np.broadcast_to(block["rate"](grid), grid.shape), dtype=float)
    # Analytic rates carry a single constant multiplicative uncertainty factor
    # f (the AddReaction argument): the 1-sigma band is [rate/f, rate*f]. We
    # store it as the (temperature-independent) error column, matching the
    # tabulated-reaction convention. Decays have f=1 (no uncertainty).
    err = block["f"] * np.ones_like(grid)
    header = (
        f"{' + '.join(block['reactants'])} > {' + '.join(block['products'])}"
        f"   [{block['name']}]   ref={expand_ref(block['ref'])}\n"
        f"forward[T9] = {block['expr']}   uncertainty factor f = {block['f']:g}\n"
        f"T9                 rate                error"
    )
    reaction_dir = os.path.join(outdir, block["name"])
    os.makedirs(reaction_dir, exist_ok=True)
    path = os.path.join(reaction_dir, block["name"] + suffix + ".txt")
    np.savetxt(path, np.column_stack([grid, rate, err]),
               fmt=["%.6e", "%.6e", "%.6e"], delimiter="   ", header=header)
    return path


def write_decay_file(decay_blocks, outdir, suffix=""):
    """Write ``decays.txt``: one row per radioactive-decay reaction.

    Decay rates are constants (no T9 dependence -- a 500-row table repeating
    the same number, as ``write_analytic_file`` would produce, is wasteful and
    obscures the physically meaningful quantity).  Instead each reaction gets
    one row with its **half-life** ``halflife_s`` (the conventional way decay
    constants are quoted/curated in nuclear-data tables) and the derived rate
    ``rate_s^-1 = ln(2) / halflife_s`` that ``network_data.py`` uses directly
    as the (T-independent) forward rate.  ``uncertainty`` is the same
    multiplicative 1-sigma factor ``f`` as the other analytic reactions
    (always 1 for decays today; kept so ``p_<rxn>``/``delta_<rxn>``
    rate-variation continues to work unchanged).

    Two of the ``Li9`` branches (``Li9TOBe9Bm``/``Li9TOaanBm``) already have
    their branching ratio folded into ``forward_expr`` in
    ``_ANALYTIC_REACTIONS`` (``* 0.492`` / ``* 0.508``); ``halflife_s`` for
    those rows is therefore the *partial* half-life
    (``T1/2_total / branching_ratio``), the standard nuclear-data convention.

    Parameters
    ----------
    decay_blocks : list[dict]
        Blocks (as built by :func:`build_analytic_blocks`) for which
        :func:`nuclide_table.is_decay` is true.
    outdir, suffix : str
        Output directory and optional filename suffix (mirrors
        :func:`write_analytic_file`).

    Returns
    -------
    str
        Path to the written ``decays.txt``.
    """
    lines = [
        "# Decay reactions of the `large` network: constant (T9-independent) rates.",
        "# rate_s^-1 = ln(2) / halflife_s.  'uncertainty' is the multiplicative",
        "# 1-sigma factor f (currently always 1 -- no published uncertainty on",
        "# these half-lives -- kept so p_<rxn>/delta_<rxn> variations keep",
        "# working: exp(p * log(f)) = 1 when f = 1).",
        "# Generated by convert_ac2024_rates.py from PRIMAT-Main.m's analytic",
        "# decay-rate expressions (see _ANALYTIC_REACTIONS).",
        "#",
        f"#{'name':<15s} {'halflife_s':>14s} {'rate_s^-1':>14s} {'uncertainty':>11s}  ref",
    ]
    for blk in sorted(decay_blocks, key=lambda b: b["name"]):
        # block["rate"] is T9-independent for decays; evaluate at any T9.
        rate = float(np.broadcast_to(blk["rate"](np.array(1.0)), ()))
        halflife_s = math.log(2.0) / rate
        ref = expand_ref(blk["ref"]) or "-"
        lines.append(f"{blk['name']:<16s} {halflife_s:14.6e} {rate:14.6e} "
                      f"{blk['f']:11g}  {ref}")
    path = os.path.join(outdir, "decays" + suffix + ".txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Hard-coded analytic reactions (the single source of truth at run time).
#
# Each entry is (source, reac, forward) where `reac` is the PRIMAT reaction
# string "reactants > products ; name" and `forward` is the raw Mathematica
# forward[T9] expression. These were extracted once from PRIMAT-Main.m's
# DefineAnalyticRates; embedding them here makes the rate files regenerable from
# BBNRatesAC2024.dat alone, with no PRIMATreference/ folder. To refresh this
# table after PRIMAT-Main.m changes, run:
#     python generate_rates/convert_ac2024_rates.py --dump-analytic <path-to-PRIMAT-Main.m>
# and paste the printed literal back here.
# ---------------------------------------------------------------------------
_ANALYTIC_REACTIONS = [
    ('Nag06', 'd + n  > t + g ; dng', 2.0,
     'With[{T923=T9^(2/3)},(214. T9^0.075+7.42T9)]'),
    ('Nag06', 't+t>a+n+n;ttn', 3.0,
     'With[{T923=T9^(2/3),T913=T9^(1/3),T943=T9^(4/3),T953=T9^(5/3)},(1/T923 1.67*^9 E^(-4.872/T913) (1. -0.272 T9+0.086 T913-0.455 T923+0.148 T943+0.225 T953))]'),
    ('Wag69', 'He3 +n > He4 + g ; hng', 3.0,
     '6.62*(1+905*T9)'),
    ('CF88', 'He3 + t > He4 + d ; htd', 10.0,
     'With[{T9A=T9/(1.+0.128*T9),T932=T9^(3/2)},With[{T9A13=T9A^(1./3.),T9A56=T9A^(5./6.)},5.46*^9*T9A56/T932*Exp[-7.733/T9A13] ]]'),
    ('CF88', 'He3 + t > He4 + n + p ; htp', 10.0,
     'With[{T9A=T9/(1.+0.115*T9),T932=T9^(3/2)},With[{T9A13=T9A^(1./3.),T9A56=T9A^(5./6.)}, 7.71*^9*T9A56/T932*Exp[-7.733/T9A13] ]]'),
    ('NACRE', 'a + a + n > Be9 + g ; aang', 1.25,
     'With[{T932=T9^(3/2),T923=T9^(2/3),T913=T9^(1/3)}, With[{he4abe8= 2.43*^9*(1.+74.5*T9)/T923*Exp[-13.49/T913-(T9/0.15)^2]+6.09*^5/T932*Exp[-1.054/T9]}, If[T9<0.03, (he4abe8)* 6.69*^-12*(1.-192*T9+2.48*^4*T9^2-1.50*^6*T9^3+4.13*^7*T9^4-3.90*^8*T9^5), (he4abe8)* 2.42*^-12*(1.-1.52*Log10[T9]+0.448*(Log10[T9])^2+0.435*(Log10[T9])^3)]]]'),
    ('CF88&MF89', 'Li7 + t > a + a + n + n; li7ta', 3.0,
     'With[{T923=T9^(2/3),T913=T9^(1/3)},8.81*^+11/T923*Exp[-11.333/T913]]'),
    ('CF88&MF89', 'Li7 + He3 > a + a + n + p; li7haa', 3.0,
     'With[{T923=T9^(2/3),T913=T9^(1/3)},1.11*^+13/T923*Exp[-17.989/T913]]'),
    ('Bal95', 'Li8 + d > Li9 + p ; li8dp', 3.0,
     'With[{T923=T9^(2/3),T913=T9^(1/3)},9.63*^6/T923*Exp[-10.324/T913]*(1.+0.404*T913)*74.]'),
    ('Has09c', 'Li8 + d > Li7 + t ; li8dt', 3.0,
     'With[{T923=T9^(2/3),T913=T9^(1/3)},(3.02*^8/T9^0.624*Exp[-3.51/T9]+5.82*^11/T923*Exp[-19.72/T913]*(1.0+0.280*T913))]'),
    ('CF88&MF89', 'Be7 + t > a + a + n + p ; be7t', 3.0,
     'With[{T923=T9^(2/3),T913=T9^(1/3)},2.91*^+12/T923*Exp[-13.729/T913]]'),
    ('CF88&MF89', 'Be7 + He3 > 2a + p + p  ; be7h', 3.0,
     'With[{T923=T9^(2/3),T913=T9^(1/3)},6.11*^+13/T923*Exp[-21.793/T913]]'),
    ('Wie89', 'C9 + a > N12 + p ; c9an', 3.0,
     'With[{T923=T9^(2/3),T932=T9^(3/2),T913=T9^(1/3),T943=T9^(4/3),T953=T9^(5/3)},(1.668*^+15/T923*Exp[-31.272/T913-(T9/.307)^2]* (1.+1.33*^-2*T913-6.42*T923-.599*T9+14.4*T943+3.42*T953)+56.8/T932*Exp[-5.292/T9]+1.7*^+5/T932*Exp[-14.08/T9]+6.52*^7/T932*Exp[-23.09/T9])]'),
    ('CF88', 'Li6+n>t+a;tan', 3.0,
     'With[{T9A=T9/(1.+49.18*T9)},With[{ T9A32=T9A^(3./2.),T932=T9^(3/2)}, (1.80*^+8*(1.-.261*T9A32/T932)*.935+2.72*^9/T932*Exp[(55.494-57.884)/T9]*.935) ]]'),
    ('FK90', 'He3 + t > Li6 + g ; htg', 3.0,
     'With[{T92=T9^2,T923=T9^(2/3),T932=T9^(3/2),T913=T9^(1/3),T943=T9^(4/3),T953=T9^(5/3)}, 2.21*^5/T923*Exp[-7.720/T913]*(1.+2.68*T923+0.868*T9+0.192*T943+0.174*T953+0.044*T92)]'),
    ('CF88', 'a + n + p > Li6 + g ; anpg', 3.0,
     'If[T9>1, 4.62*^-6/T9^2*(1.+0.075*T9)*Exp[-19.353/T9],0]'),
    ('MF89', 'Li6 + n > Li7 + g ; li6ng', 3.0,
     '5.10*^3'),
    ('MF89', 'Li6 + d > Li7 + p ; li6dp', 3.0,
     'With[{T923=T9^(2/3),T913=T9^(1/3)}, 1.48*^12/T923*Exp[-10.135/T913]]'),
    ('MF89', 'Li6 + d > Be7 + n ; li6dn', 3.0,
     'With[{T923=T9^(2/3),T913=T9^(1/3)}, 1.48*^12/T923*Exp[-10.135/T913]]'),
    ('CF88', 'Li6 + a > B10 + g ; li6ag', 3.0,
     'With[{T923=T9^(2/3),T932=T9^(3/2),T913=T9^(1/3),T943=T9^(4/3),T953=T9^(5/3)}, (4.06*^06/T923*Exp[-18.79/T913-(T9/1.326)^2]*(1.+0.022*T913+1.54*T923+0.239*T9+2.2*T943+0.869*T953) +1.91*^3/T932*Exp[-3.484/T9]+1.01*^4/T9*Exp[-7.269/T9])]'),
    ('NACRE', 'Li7 + a > B10 + n ; li7an / b10na', 1.08,
     '1.325*1.66*^7*(1.+1.064*T9)*1/1.3242*Exp[-32.3755/T9]'),
    ('MF89&Hei98', 'Li7+n>Li8+g;li7ng', 3.0,
     'With[{T932=T9^(3/2)},(6.015*^3 +1.141*^4/T932*Exp[-2.576/T9])]'),
    ('MF89', 'Li7 + d > Li8 + p ; li7dp ! Q<0 !', 3.0,
     'With[{T932=T9^(3/2)}, 8.31*^8/T932*Exp[-6.998/T9]]'),
    ('Rau94', 'Li8 + n > Li9 + g ; li8ng', 3.0,
     'With[{T932=T9^(3/2)},(3.260*^3 +6.328*^4/T932*Exp[-2.866/T9])]'),
    ('Men12', 'Li8 + p > a + a + n ; li8pn', 1.0,
     'With[{T932=T9^(3/2),T913=T9^(1/3),T923=T9^(2/3),T92=T9^2,T93=T9^3,T94=T9^4,T95=T9^5}, If[T9<5,( 5.36*^8/T932*Exp[-4.41/T9]+1.99*^8/T932*Exp[-7.08/T9]+5.85*^10/T923*Exp[-8.50/T913]* (1.-1.70*T9+0.849*T92-0.175*T93+1.62*^-2*T94-5.60*^-4*T95)), 7.777*^7]]'),
    ('Bal95', 'Li8 + d > Be9 + n ; li8dn', 3.0,
     'With[{T913=T9^(1/3),T923=T9^(2/3)},9.63*^6/T923*Exp[-10.324/T913]*(1.+0.404*T913)*188.]'),
    ('Rau94', 'Be9 + n > Be10 + g ; be9ng', 3.0,
     'With[{T913=T9^(1/3),T923=T9^(2/3),T932=T9^(3/2)},(1.01*^3 +1.01*^4/T932*Exp[-6.487/T9]+5.41*^4/T932*Exp[-8.471/T9])]'),
    ('NACRE', 'Be9 + p > a + a + p + n ; be9pn', 1.05,
     '5.06*^7*Exp[-21.479/T9]*(1.+1.26*T9-0.0302*T9^2)'),
    ('NACRE', 'B11 + p > C11 + n ; b11pn ! Q < 0 !', 1.1,
     '1.36*^8*Exp[-32.085/T9]*(1.+0.963*T9-0.285*T9^2+3.36*^-2*T9^3-1.37*^-3*T9^4)'),
    ('Rau94', 'Be10 + n > Be11 + g ; be10ng', 3.0,
     'With[{T932=T9^(3/2)},(5.96*^2 +6.67*^5/T932*Exp[-14.85/T9]) ]'),
    ('Rau94', 'Be11 + n > Be12 + g ; be11ng', 3.0,
     'With[{T932=T9^(3/2)}, 3.56*^2 ]'),
    ('Des99Bea01', 'B8 + p > C9 + g ; b8pg', 3.0,
     'With[{T932=T9^(3/2),T913=T9^(1/3),T92=T9^2},6.253*^5*Exp[-11.971/T913]*(1.-7.03*^-2*T9+6.25*^-3*T92)]'),
    ('NACRE', 'a + a + a > C12 + 2g ; aaag', 1.15,
     'With[{T932=T9^(3/2),T923=T9^(2/3),T913=T9^(1/3)}, With[{he4abe8=2.43*^9*(1.+74.5*T9)/T923*Exp[-13.49/T913-(T9/0.15)^2]+6.09*^5/T932*Exp[-1.054/T9], be8agc12= 2.76*^7*(1.+5.47*T9+326*T9^2)/T923*Exp[-23.570/T913-(T9/0.4)^2]+130.7/T932*Exp[-3.338/T9]+2.51*^4/T932*Exp[-20.307/T9]}, If[T9<0.03, he4abe8*be8agc12*3.07*^-16*(1.-29.1*T9+1308*T9^2), he4abe8*be8agc12*3.44*^-16*(1.+0.0158/T9^0.65)]]]'),
    ('Tang03', 'C11+p>N12+g;c11pg', 3.0,
     'With[{T923=T9^(2/3),T932=T9^(3/2),T913=T9^(1/3),T943=T9^(4/3),T953=T9^(5/3)},(1.670*^2*Exp[-4.166/T9]/T932+2.148*^5*Exp[-13.281/T913]/T923* (1.+4.639*T913-2.641*T923-1.543*T9+2.030*T943+4.657*T953))]'),
    ('CF88', 'B10 + a > N13 + n ; b10an', 3.0,
     'With[{T923=T9^(2/3),T913=T9^(1/3)}, 1.2*^13/T923*Exp[-27.989/T913-(T9/9.589)^2] ]'),
    ('Wan91', 'B11+a>C14+p;b11ap', 3.0,
     'With[{T923=T9^(2/3),T932=T9^(3/2),T913=T9^(1/3),T943=T9^(4/3),T953=T9^(5/3)},(8.403*^15*Exp[-31.914/T913-(T9/0.3432)^2]*(1.+0.022*T913+5.712*T923+0.642*T9+15.982*T943+4.062*T953) +5.44*^-3/T932*Exp[-2.868/T9]+2.419*^2/T932*Exp[-5.147/T9]+4.899*^2/T932*Exp[-5.157/T9]+4.944*^6/T9^(3/5)*Exp[-11.26/T9])]'),
    ('Rau94', 'C11+n>C12+g;c11ng', 3.0,
     'With[{T932=T9^(3/2)},(3.18*^4 +3.30*^3/T932*Exp[-0.917/T9]+1.05*^6/T932*Exp[-5.57/T9])]'),
    ('Aud03', 'He6>Li6+Bm;', 1.0,
     'Log[2]/8.0670*^-1'),
    ('Aud03', 'Li8>2a+Bm;', 1.0,
     'Log[2]/8.4030*^-1'),
    ('Aud03', 'Li9>Be9+Bm;', 1.0,
     'Log[2]/1.7830*^-1 * 0.492'),
    ('Aud03', 'Li9>a+a+n+Bm;', 1.0,
     'Log[2]/1.7830*^-1 * 0.508'),
    ('Aud03', 'Be11>B11+Bm;', 1.0,
     'Log[2]/(1.3810*^1)'),
    ('Aud03', 'Be12>B12+Bm;', 1.0,
     'Log[2]/(2.15*^-2)'),
    ('Aud03', 'B8>a+a+Bp;', 1.0,
     'Log[2]/(7.70*^-1)'),
    ('Aud03', 'B12>C12+Bm;', 1.0,
     'Log[2]/(2.02*^-2)'),
    ('Aud03', 'B13>C13+Bm;', 1.0,
     'Log[2]/(1.733*^-2)'),
    ('Aud03', 'B14>C14+Bm;', 1.0,
     'Log[2]/(1.25*^-2)'),
    ('Aud03', 'B15>C15+Bm;', 1.0,
     'Log[2]/(9.87*^-3)'),
    ('Aud03', 'C9>a+a+p+Bp;', 1.0,
     'Log[2]/(1.26*^-1)'),
    ('Aud03', 'C10>B10+Bp;', 1.0,
     'Log[2]/(19.29)'),
    ('Aud03', 'C11>B11+Bp;', 1.0,
     'Log[2]/1.2234*^3'),
    ('Aud03', 'C15>N15+Bm;', 1.0,
     'Log[2]/2.449'),
    ('Aud03', 'C16>N16+Bm;', 1.0,
     'Log[2]/7.4700*^-1'),
    ('Aud03', 'N12>C12+Bp;', 1.0,
     'Log[2]/1.100*^-2'),
    ('Aud03', 'N13>C13+Bp;', 1.0,
     'Log[2]/5.979*^2'),
    ('Aud03', 'N16>O16+Bm;', 1.0,
     'Log[2]/7.13'),
    ('Aud03', 'N17>O16+n+Bm;', 1.0,
     'Log[2]/4.1730'),
    ('Aud03', 'O13>N13+Bp;', 1.0,
     'Log[2]/8.58*^-3'),
    ('Aud03', 'O14>N14+Bp;', 1.0,
     'Log[2]/70.598'),
    ('Aud03', 'O15>N15+Bp;', 1.0,
     'Log[2]/122.24'),
    ('Aud03', 'O19>F19+Bm;', 1.0,
     'Log[2]/26.464'),
    ('Aud03', 'O20>F20+Bm;', 1.0,
     'Log[2]/13.51'),
    ('Aud03', 'F17>O17+Bp;', 1.0,
     'Log[2]/64.49'),
    ('Aud03', 'F18>O18+Bp;', 1.0,
     'Log[2]/6.5863*^3'),
    ('Aud03', 'F20>Ne20+Bm;', 1.0,
     'Log[2]/11.1630'),
    ('Aud03', 'Ne18>F18+Bp;', 1.0,
     'Log[2]/1.6720'),
    ('Aud03', 'Ne19>F19+Bp;', 1.0,
     'Log[2]/17.296'),
    ('Aud03', 'Ne23>Na23+Bm;', 1.0,
     'Log[2]/37.240'),
    ('Aud03', 'Na20>Ne20+Bp;', 1.0,
     'Log[2]/4.4790*^-1'),
    ('Aud03', 'Na21>Ne21+Bp;', 1.0,
     'Log[2]/22.49'),
    # The next two decays (t->He3 and Be7->Li7) are not part of PRIMAT-Main.m's
    # "Decay Rates" block above (they were added to the `large` network
    # separately, commit 6221e43); they use the same Log[2]/T1/2[s] convention
    # as the rest of this table, with T1/2 from Aud03 (12.32 yr and 53.29 d).
    ('Aud03', 't>He3+Bm;', 1.0,
     'Log[2]/(12.32*86400*365.2422)'),
    ('Aud03', 'Be7>Li7+Bp;', 1.0,
     'Log[2]/(53.29*86400)'),
    # Long-lived decays relevant for the Decay Time (DT) era (T < 0.001 MeV,
    # t > 10^6 s): half-lives from NUBASE2020 (Kondev et al. 2021).
    # These were added to PRIMAT-Main_decays.m's analytic block and to
    # rates/nuclear/networks/large.txt; the conversion factor
    # 86400*365.2422 = seconds per Julian year.
    # C14 -> N14 + e^-: T1/2 = 5700 yr (NUBASE2020).
    ('Nubase', 'C14>N14+Bm;', 1.0,
     'Log[2]/(5700*86400*365.2422)'),
    # Be10 -> B10 + e^-: T1/2 = 1.387 Myr = 1387000 yr (NUBASE2020).
    ('Nubase', 'Be10>B10+Bm;', 1.0,
     'Log[2]/(1387000*86400*365.2422)'),
    # Na22 -> Ne22 + e^+ (beta-plus): T1/2 = 2.6019 yr (NUBASE2020).
    ('Nubase', 'Na22>Ne22+Bp;', 1.0,
     'Log[2]/(2.6019*86400*365.2422)'),
    ('Efr96', 'He4 + 2n  > He6 + g ;', 3.0,
     'If[T9<2, (2.65*^-3*T9^2.555*Exp[0.181/Max[T9,.1]]), (2.93*^-1*T9^(-3.51*^-1)*Exp[-5.24/T9])]'),
    ('Iga95', 'O16 + n  > O17 + g ;', 3.0,
     '(2.7*^1 +1.38*^4*T9 )'),
    ('CF88', 'N14 + n  > C14 + p ;', 3.0,
     'With[{T912=T9^(1/2)},( 7.19*^5*(1.+.361*T912+.502*T9)+3.34*^8/T912*Exp[-4.983/T9])*.333]'),
    ('CF88', 'O14 + n  > N14 + p ;', 3.0,
     'With[{T912=T9^(1/2)},(6.74*^7*(1.+0.658*T912+0.379*T9)*2.99 )]'),
    ('Wie87', 'O14 + a  > Ne18 + g ;', 3.0,
     'With[{T932=T9^(3/2)},( 1.16*^-1/T932*Exp[-11.73/T9]+3.40*^1/T932*Exp[-22.61/79]+9.10*^-3*T9^5*Exp[-12.159])]'),
    ('NACRE', 'C11 + a  > N14 + p ;', 2.0,
     'With[{T913=T9^(1/3),T92=T9^2},(0.2719*3.01*^16*Exp[-31.884/T913]* Exp[-1.379*T9+.215*T92-2.13*^-2*T92*T9+8*^-4*T92*T92]*(1.+0.14*Exp[-.275/T9-.210*T9]) )]'),
    ('Bar97C', 'O14 + a  > F17 + p ;', 3.0,
     'With[{T932=T9^(3/2),T923=T9^(2/3),T913=T9^(1/3),T943=T9^(4/3),T953=T9^(5/3)}, With[{offset=1.330*^5/T932*Exp[-11.86/T9]+8.42*^-47*T932*Exp[-0.453/T9]+6.74*^4/T932*Exp[-13.60/T9]+1.21*^7/T932*Exp[-22.51/T9]+1.26*^8/T932*Exp[-26.00/T9]}, (offset+If[T9<1, 7.906*^15/T923*Exp[-40.33/T913]*(1.-1.884*^1*T913+2.446*^2*T923-7.735*^2*T9+9.485*^2*T943-3.961*^2*T953),0])]]'),
    ('Koe91', 'O17 + n > C14 + a ;', 3.0,
     'With[{T932=T9^(3/2)},( 3.11*^4 +9.18*^5/T932*Exp[-1.961/T9]+7.02*^7/T932*Exp[-2.759/T9])]'),
    ('NACRE', 'F17 + n > N14 + a ;', 1.05,
     '(1.38*^8*T9^0.053*Exp[-(55.0-54.943)/T9]* (1.+.039*Exp[-.012/T9+.217*T9])/1.478 )'),
    ('CF88', 'F18 + n > N15 + a ;', 3.0,
     'With[{T912=T9^(1/2)},( 3.14*^8*(1.-0.641*T912+0.108*T9)*2.)]'),
    ('Kaw91', 'C14 + d  > N15 + n ;', 3.0,
     'With[{T923=T9^(2/3)},(4.27*^13/T923*Exp[-16.939])]'),
    ('CF88', 'p + p + n > d + p ;', 3.0,
     'With[{T923=T9^(2/3),T913=T9^(1/3)},(1.35*^7*Exp[-3.720/T913]*(1.+0.784*T913+0.346*T923+0.690*T9)/2.3590*^9)]'),
    ('Kaw91', 'C14 + n > C15 + g ;', 3.0,
     '(3240. *T9 )'),
    ('CF88', 'O16 + p  > N13 + a ;', 3.0,
     'With[{T953=T9^(5/3),T932=T9^(3/2)}, With[{T9A=T9/(1.+7.76*^-2*T9+2.64*^-2*T953/(1.+7.76*^-2*T9)^(2./3.))}, With[{T9A13=T9A^(1./3.),T9A56=T9A^(5./6.)}, With[{SVRev=1.88*^18*T9A56/T932*Exp[-35.829/T9A13]*1.7232*^-1}, With[{SVDir=SVRev/0.172255*Exp[-60.5573/T9]}, SVDir]]]]]'),
    ('TUNL&Cam08', 'Li8 + p > Be9 + g ;', 3.0,
     'With[{T923=T9^(2/3),T913=T9^(1/3),T932=T9^(3/2)}, (3.516*^6/T923*Exp[-8.5155/T913]+2.669*^4/T932*Exp[-1.010/T9] )]'),
    ('Wan91', 'B11 + a  > N15 + g ;', 3.0,
     'With[{T932=T9^(3/2)},(643./T932*Exp[-5.1526/T9] )]'),
]


# ---------------------------------------------------------------------------
# Network / nuclide / detailed-balance generation
# ---------------------------------------------------------------------------
# The rate *files* above give PyPRIMAT the numbers; the files below give it the
# *structure* it needs to assemble the large network without any hand-coding:
#   * reactions_large.csv : one row per reaction (name, reactants, products),
#   * nuclides.csv        : every nuclide it touches, with N, Z, A, Q, mass, spin,
#   * detailed_balance.csv: alpha, beta, gamma per reversible reaction.
# All three are derived here, once, from AC2024 + the analytic table + NUBASE.


def _canon_side(tokens):
    """Canonicalise a reactant/product token list for *reaction-identity*
    comparisons (order-independent, spelling-independent): nuclides become
    their :func:`nuclide_table.canonical_name`, photons become ``'g'``, and
    leptons (``Bm``/``Bp``) are kept as-is."""
    from nuclide_table import resolve_token
    out = []
    for tok in tokens:
        t = resolve_token(tok)
        out.append(t.name if t.kind == "nuclide" else ("g" if t.kind == "photon" else tok))
    return tuple(sorted(out))


def check_name_collisions(tab_blocks, ana_blocks):
    """Check <reactants>TO<products> name collisions across both block lists.

    Two blocks can legitimately share a name: PRIMAT moves some reactions from
    an analytic formula to a tabulated rate as data improve, so the *same*
    reaction may appear in both ``tab_blocks`` and ``ana_blocks`` -- in that
    case :func:`unified_reactions`/``write_*_file`` intentionally let the
    analytic version win (last write wins).  These are reported as
    "overrides" and are not an error.

    But if two *different* reactions canonicalise to the same
    ``<reactants>TO<products>`` name, one would silently overwrite the other's
    rate file -- this is always a bug (a naming collision, not an override),
    so it raises ``ValueError`` listing the offending names.
    """
    by_name = {}
    for blk in tab_blocks + ana_blocks:
        by_name.setdefault(blk["name"], []).append(blk)

    overrides, bad = [], []
    for name, blks in by_name.items():
        if len(blks) < 2:
            continue
        signatures = {(_canon_side(b["reactants"]), _canon_side(b["products"]))
                      for b in blks}
        if len(signatures) == 1:
            overrides.append(name)
        else:
            bad.append(name)

    if bad:
        raise ValueError(
            f"{len(bad)} <reactants>TO<products> name(s) collide between "
            f"distinct reactions, so one would silently overwrite the other's "
            f"rate file: {sorted(bad)}. Give one of them a different short "
            f"name in the source.")
    if overrides:
        print(f"  ({len(overrides)} reaction(s) given as both tabulated and "
              f"analytic; analytic version wins: {sorted(overrides)})")


def unified_reactions(tab_blocks, ana_blocks):
    """Merge tabulated and analytic blocks into one de-duplicated reaction list.

    A reaction may appear in both sources (PRIMAT moves a rate from analytic to
    tabulated as data improve); the analytic blocks are written last, so the
    analytic version wins the rate file and we keep it here too.  Each entry is
    ``dict(name, reactants, products, ref, source)`` with token lists exactly as
    parsed from the sources (``a``/``He4``/... spellings preserved; canonicalised
    downstream by :mod:`nuclide_table`).
    """
    merged = {}
    for blk in tab_blocks:
        merged[blk["name"]] = dict(name=blk["name"], reactants=blk["reactants"],
                                   products=blk["products"], ref=blk["ref"],
                                   source="tabulated")
    for blk in ana_blocks:                    # analytic overrides on collision
        merged[blk["name"]] = dict(name=blk["name"], reactants=blk["reactants"],
                                   products=blk["products"], ref=blk["ref"],
                                   source="analytic")
    return list(merged.values())


def write_network_files(reactions, tab_blocks, nubase_path, outdir, suffix="_primat"):
    """Write nuclides.csv, reactions_large.csv, detailed_balance.csv and
    large.txt, after the formal conservation check and a detailed-balance
    cross-check.

    Steps:
      1. Deduce the nuclide set and resolve N,Z,A,Q,mass,spin from NUBASE.
      2. **Formal** (integer) check that every reaction conserves baryon number
         A and charge Q; abort on any violation.
      3. Compute (alpha, beta, gamma) for every reversible (non-decay) reaction
         from nuclide data, and cross-check against the AC2024 tabulated values.
      4. Emit the three CSVs PyPRIMAT reads at run time, plus
         ``LARGE_NETWORK_FILE`` (the large-network reaction list, i.e. the
         ``name`` column of ``reactions_large.csv``).

    ``suffix`` (default ``"_primat"``, matching the per-reaction rate files
    written by :func:`write_reaction_file`/:func:`write_analytic_file`) is
    appended to each non-decay reaction's filename in ``large.txt``, which
    ``load_network`` (``primat/network_data.py``) always spells out
    explicitly rather than implying -- see CLAUDE.md's "Adding a new
    reaction" section. Decay reactions (Bm/Bp) are written bare: their rate
    lives in the shared, unsuffixed ``decays.txt``, not a per-reaction file.
    """
    from nuclide_table import (build_nuclide_table, conservation_residual,
                               make_detailed_balance, is_decay)

    # 1. Nuclide set + properties --------------------------------------------
    nuclides = build_nuclide_table(reactions, nubase_path)

    # 2. Formal A & Q conservation check (abort on violation) -----------------
    violations = []
    for rxn in reactions:
        dA, dQ = conservation_residual(rxn["reactants"], rxn["products"])
        if dA != 0 or dQ != 0:
            violations.append((rxn["name"], dA, dQ))
    if violations:
        msg = "\n".join(f"  {n}: dA={dA:+d}, dQ={dQ:+d}" for n, dA, dQ in violations)
        raise ValueError(f"{len(violations)} reaction(s) violate A/Q conservation:\n{msg}")
    print(f"formal check OK: all {len(reactions)} reactions conserve A and Q")

    # 3. Detailed balance + cross-check vs AC2024 tabulated coefficients ------
    db = make_detailed_balance(nuclides)
    coeffs = {}           # name -> (Q_keV, alpha, beta, gamma); reversible only
    db_skipped = []
    for rxn in reactions:
        if is_decay(rxn["reactants"], rxn["products"]):
            continue      # decays have no reverse rate -> no detailed balance
        try:
            coeffs[rxn["name"]] = db(rxn["reactants"], rxn["products"])
        except Exception as exc:                       # noqa: BLE001
            db_skipped.append((rxn["name"], str(exc)))
    if db_skipped:
        print(f"  ({len(db_skipped)} reactions without detailed balance: "
              f"{[n for n, _ in db_skipped][:8]}{'...' if len(db_skipped) > 8 else ''})")

    # Cross-check: computed gamma/alpha vs the AC2024 file's own coefficients.
    worst_g = worst_a = 0.0
    for blk in tab_blocks:
        if blk["name"] not in coeffs:
            continue
        _, alpha, beta, gamma = coeffs[blk["name"]]
        if blk["gamma"]:
            worst_g = max(worst_g, abs(gamma - blk["gamma"]) / abs(blk["gamma"]))
        if blk["alpha"]:
            worst_a = max(worst_a, abs(alpha - blk["alpha"]) / abs(blk["alpha"]))
    print(f"detailed-balance cross-check vs AC2024: max |Δγ/γ|={worst_g:.2%}, "
          f"max |Δα/α|={worst_a:.2%}")

    # 4. Emit the CSVs --------------------------------------------------------
    with open(os.path.join(outdir, "nuclides.csv"), "w") as f:
        f.write("name,N,Z,A,Q,mass_excess_keV,spin\n")
        for r in nuclides.values():
            f.write(f"{r['name']},{r['N']},{r['Z']},{r['A']},{r['Q']},"
                    f"{r['excess_keV']:.6f},{r['spin']:g}\n")

    sorted_reactions = sorted(reactions, key=lambda r: r["name"])

    with open(os.path.join(outdir, "reactions_large.csv"), "w") as f:
        # reactants/products are '+'-joined canonical token lists (a->He4 etc.);
        # multiplicity is explicit by repetition (e.g. He4+He4+n).
        from nuclide_table import resolve_token

        def canon(side):
            return "+".join(resolve_token(t).name or t for t in side)
        f.write("name,reactants,products,source,ref\n")
        for rxn in sorted_reactions:
            f.write(f"{rxn['name']},{canon(rxn['reactants'])},"
                    f"{canon(rxn['products'])},{rxn['source']},{rxn['ref']}\n")

    # large.txt is the network-definition file load_network reads for
    # network="large": one reaction name per line, identical to (and kept in
    # sync with) reactions_large.csv's first column.
    os.makedirs(os.path.dirname(LARGE_NETWORK_FILE), exist_ok=True)
    with open(LARGE_NETWORK_FILE, "w") as f:
        for rxn in sorted_reactions:
            if is_decay(rxn["reactants"], rxn["products"]):
                f.write(f"{rxn['name']}\n")
            else:
                f.write(f"{rxn['name']}, {rxn['name']}{suffix}.txt\n")

    with open(os.path.join(outdir, "detailed_balance.csv"), "w") as f:
        f.write("reaction,Q_keV,alpha,beta,gamma\n")
        for name in sorted(coeffs):
            Q, alpha, beta, gamma = coeffs[name]
            f.write(f"{name},{Q:.6g},{alpha:.8g},{beta:g},{gamma:.8g}\n")

    print(f"wrote nuclides.csv ({len(nuclides)} nuclides), "
          f"reactions_large.csv ({len(reactions)} reactions), "
          f"detailed_balance.csv ({len(coeffs)} reversible reactions)")


def _validate_decay_halflives(decay_blocks, nubase_path):
    """Cross-check coded half-lives against the NUBASE2020 table.

    For each decay block (as built by :func:`build_analytic_blocks` for
    reactions identified as decays by :func:`nuclide_table.is_decay`), look up
    the *parent* nuclide's half-life in NUBASE2020 and compare it to the value
    embedded in ``_ANALYTIC_REACTIONS``.  A warning is printed when the ratio
    differs by more than 1%.

    The comparison is approximate (not exact) for two reasons:

    1. NUBASE quotes half-lives in human-friendly units (seconds, minutes, days,
       years …); the coded value converts via hard-coded unit factors that may
       differ by a part in 10^4 from the exact value (e.g. 365.2422 days/yr
       vs. 365.25 used for tritium).
    2. Branching-ratio decays (e.g. ``Li9TOBe9Bm`` / ``Li9TOaanBm``) embed the
       branching fraction directly in the expression; their coded value is the
       *partial* half-life, which is longer than the NUBASE total by 1/BR.

    The check is therefore informational only -- it catches gross errors (factor
    of 2 or more) while tolerating unit-convention rounding and branch-ratio
    folding.

    Parameters
    ----------
    decay_blocks : list[dict]
        Blocks for which :func:`nuclide_table.is_decay` is ``True``, as
        returned by :func:`build_analytic_blocks`.
    nubase_path : str
        Path to the NUBASE2020 fixed-width text file
        (``generate_rates/nubase_4.mas20.txt``).

    Returns
    -------
    None
        All output is printed; no exceptions are raised.

    Example
    -------
    Running this as part of ``_generate_analytic`` after PRIMAT-Main_decays.m
    changes catches accidental copy-paste of wrong half-lives::

        # A freshly coded wrong half-life would produce:
        # WARNING: Be10TOB10Bm parent Be10 NUBASE T1/2 = 4.377e+13 s,
        #          coded T1/2 = 4.382e+13 s (ratio = 1.001, OK)
    """
    import math
    from nuclide_table import load_nubase_all, resolve_token

    # NUBASE half-life fields (columns 70-78, 79-80) are text-formatted with
    # physical units (s, m, h, d, ky, My, …). The reader load_nubase_all
    # returns only (excess, spin), not the half-life, so we re-read the file
    # here for the raw half-life string.  Parse column offsets from the file
    # header: T # at col 70-78, unit at col 79-80.
    _UNIT_TO_S = {
        "ys": 1e-24, "zs": 1e-21, "as": 1e-18, "fs": 1e-15,
        "ps": 1e-12, "ns": 1e-9,  "us": 1e-6,  "ms": 1e-3,
        "s":  1.0,   "m":  60.0,  "h":  3600.0,
        "d":  86400.0,
        "y":  86400.0 * 365.2422,   # Julian year: 365.2422 d
        "ky": 86400.0 * 365.2422 * 1e3,
        "My": 86400.0 * 365.2422 * 1e6,
        "Gy": 86400.0 * 365.2422 * 1e9,
        "Ty": 86400.0 * 365.2422 * 1e12,
    }

    # Build a (Z, A) -> half-life_s dict from the NUBASE file.
    #
    # NUBASE2020 fixed-width column layout for the half-life:
    #   col 70-78 (0-indexed): T # (half-life value as a float string,
    #             "stbl" for stable, "p-unst" for particle-unstable).
    #   col 79-80: half-life unit (2 characters), e.g. "s ", "m ", "h ",
    #             "d ", "y ", "ms", "us", "ky", "My", "Gy", ...
    # The unit is always in columns 79-80 (0-indexed); for "ky"/"My" the
    # 'k'/'M' prefix is at column 79 and 'y' is at column 80.
    halflife_nubase = {}   # (Z, A) -> half-life in seconds (None for stable)
    with open(nubase_path, encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("#") or len(line) < 82:
                continue
            if line[7] != "0":  # ground states only
                continue
            try:
                A = int(line[0:3])
                Z = int(line[4:7])
            except ValueError:
                continue
            t_str = line[70:78].strip().replace("#", "")
            # Unit occupies columns 78-80 (0-indexed, 1-based: 79-81).
            # For plain units like "s", "d", "y" the prefix slot at col 78 is
            # a space; for "ky"/"My"/"Gy" the SI prefix sits at col 78.
            # Always strip trailing spaces to get the 1- or 2-character unit.
            unit  = line[78:80].strip()
            if t_str in ("stbl", "p-unst", ""):
                halflife_nubase[(Z, A)] = None  # stable or particle-unstable
                continue
            try:
                t_val = float(t_str)
            except ValueError:
                halflife_nubase[(Z, A)] = None
                continue
            s_per_unit = _UNIT_TO_S.get(unit)
            if s_per_unit is None:
                halflife_nubase[(Z, A)] = None
                continue
            halflife_nubase[(Z, A)] = t_val * s_per_unit

    ok = True
    for blk in decay_blocks:
        # The parent is the sole reactant nuclide (decays have one reactant).
        parent_toks = [t for t in blk["reactants"]
                       if t not in ("Bm", "Bp", "g", "n", "p")]
        if not parent_toks:
            continue
        parent_name = parent_toks[0]
        try:
            tok = resolve_token(parent_name)
        except ValueError:
            continue
        if tok.kind != "nuclide":
            continue
        key = (tok.Z, tok.A)
        nubase_t12 = halflife_nubase.get(key)
        if nubase_t12 is None:
            continue   # stable or unknown

        # Coded rate from the block (evaluated at any T9, since it is constant).
        rate_coded = float(blk["rate"](1.0))
        if rate_coded <= 0.0:
            continue
        coded_t12 = math.log(2.0) / rate_coded   # partial half-life [s]

        # Ratio coded / NUBASE (the coded value may be a partial half-life if a
        # branching fraction is folded in, so ratio > 1 is expected for those).
        ratio = coded_t12 / nubase_t12
        label = f"{blk['name']}: NUBASE T1/2 = {nubase_t12:.4g} s, coded T1/2 = {coded_t12:.4g} s (ratio = {ratio:.4g})"
        if abs(ratio - 1.0) > 0.01 and ratio < 0.99:
            # ratio > 1 (partial half-life) is expected; ratio < 1 by > 1% is an error.
            print(f"  WARNING: {label} -- ratio < 1 by > 1%, check coded value!")
            ok = False
        elif ratio > 2.0:
            # Partial half-life is expected to be longer than NUBASE total, but
            # by at most 1/BR_min ≈ 2× for typical branching.  Much larger
            # ratios suggest a wrong nuclide or unit conversion.
            print(f"  WARNING: {label} -- coded T1/2 is > 2× NUBASE total, check!")
            ok = False
        else:
            print(f"  OK:      {label}")
    if ok:
        print(f"  decay half-life cross-check vs NUBASE2020: all {len(decay_blocks)} entries OK")


def _dump_analytic_literal(primat_path):
    """Print the ``_ANALYTIC_REACTIONS`` Python literal extracted from
    PRIMAT-main.m, so it can be pasted back into this file to update the
    hard-coded table after PRIMAT-main.m changes."""
    entries = extract_analytic_from_primat(primat_path)
    print(f"# {len(entries)} analytic reactions extracted from {primat_path}")
    print("_ANALYTIC_REACTIONS = [")
    for ref, reac, f, forward in entries:
        print(f"    ({ref!r}, {reac!r}, {f!r},")
        print(f"     {forward!r}),")
    print("]")


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="generate_rates/BBNRatesAC2024.dat",
                   help="the tabulated AC2024 reaction-rate compilation")
    p.add_argument("--nubase", default="generate_rates/nubase_4.mas20.txt",
                   help="the NUBASE2020 evaluation (nuclide masses and spins)")
    p.add_argument("--datadir", default="primat/rates/csv",
                   help="the directory for network structure files (.csv)")
    p.add_argument("--suffix", default="_primat",
                   help="suffix for generated per-reaction rate files (default "
                        "'_primat', matching the explicit filenames written into "
                        "small.txt/large.txt; pass '' for the legacy unsuffixed "
                        "naming). Never applied to decays.txt, which always keeps "
                        "its fixed, unsuffixed name (see _load_decay_table).")
    p.add_argument("--primat", default=None,
                   help="re-extract the analytic reactions from this PRIMAT-main.m "
                        "instead of using the hard-coded table (for verification)")
    p.add_argument("--dump-analytic", metavar="PRIMAT_PATH", default=None,
                   help="print the _ANALYTIC_REACTIONS literal extracted from the "
                        "given PRIMAT-main.m and exit (to regenerate the table)")
    p.add_argument("--keep-source-grid", action="store_true",
                   help="write each tabulated reaction on its own native T9 grid "
                        "(~60 points from the AC2024 file) instead of reinterpolating "
                        "onto the standard 500-point grid.  Analytic reactions always "
                        "use the standard grid.  PyPRIMAT's load_network resamples all "
                        "tables to a master grid at init, so mixing grids is safe.")
    return p.parse_args(argv)


def _generate_tabulated(args, grid):
    """Stage 1: parse AC2024 and write one rate file per tabulated reaction.

    Returns the parsed blocks (needed downstream for the CSV/cross-check
    stage).  With ``--keep-source-grid``, each file is written on its own
    native AC2024 T9 grid (~60 points) instead of the standard grid;
    PyPRIMAT's ``load_network`` resamples all tables to a master grid at init,
    so mixing grids is safe.
    """
    tab_blocks = parse_blocks(args.input)
    for blk in tab_blocks:
        blk_grid = blk["T9"] if args.keep_source_grid else grid
        write_reaction_file(blk, blk_grid, TABDIR, args.suffix)
    print(f"parsed {len(tab_blocks)} tabulated reactions from {args.input}")
    return tab_blocks


def _generate_analytic(args, grid):
    """Stage 2: build and write one rate file per analytic reaction.

    By default the embedded ``_ANALYTIC_REACTIONS`` table is the source (the
    single source of truth, so the rate set is regenerable from
    ``BBNRatesAC2024.dat`` alone). ``--primat`` is a verification override:
    re-extract the same entries from ``PRIMAT-Main.m`` and check they
    reproduce the same files. Returns the built blocks (needed downstream for
    the collision check and the CSV stage).
    """
    if args.primat:
        entries = extract_analytic_from_primat(args.primat)
        source = args.primat
    else:
        entries = _ANALYTIC_REACTIONS
        source = "the embedded _ANALYTIC_REACTIONS table"
    ana_blocks, skipped = build_analytic_blocks(entries)

    # Decay reactions (Bm/Bp on the products side) get one row each in the
    # single decays.txt table instead of a 500-row constant-value file: their
    # rate doesn't depend on T9, so a per-reaction table is redundant and
    # obscures the half-life, the quantity nuclear-data references quote.
    from nuclide_table import is_decay
    decay_blocks = [b for b in ana_blocks if is_decay(b["reactants"], b["products"])]
    other_blocks = [b for b in ana_blocks if b not in decay_blocks]

    for blk in other_blocks:
        write_analytic_file(blk, grid, TABDIR, args.suffix)
    if decay_blocks:
        _validate_decay_halflives(decay_blocks, args.nubase)
        # decays.txt is always unsuffixed: network_data._load_decay_table
        # hardcodes that filename, and the file is shared (one row per decay)
        # rather than per-reaction, so args.suffix never applies to it.
        write_decay_file(decay_blocks, TABDIR, "")
    print(f"built {len(ana_blocks)} analytic reactions from {source} "
          f"({len(decay_blocks)} decays -> decays.txt, "
          f"{len(other_blocks)} -> per-reaction tables)")
    if skipped:
        print(f"  ({len(skipped)} analytic blocks skipped: "
              f"{[n for n, _ in skipped]})")
    return ana_blocks


def main(argv=None):
    args = _parse_args(argv)

    if args.dump_analytic:
        _dump_analytic_literal(args.dump_analytic)
        return

    os.makedirs(TABDIR, exist_ok=True)
    os.makedirs(args.datadir, exist_ok=True)
    grid = standard_grid()

    # 1+2. Write the rate tables (tabulated then analytic; on a name collision
    #      the analytic file is written last and wins, see check_name_collisions).
    tab_blocks = _generate_tabulated(args, grid)
    ana_blocks = _generate_analytic(args, grid)

    # 3. Check <reactants>TO<products> name collisions across both sets:
    #    same-reaction overrides are reported, distinct-reaction collisions abort.
    check_name_collisions(tab_blocks, ana_blocks)
    n_files = len({blk["name"] for blk in tab_blocks + ana_blocks})
    print(f"wrote {n_files} rate files to {TABDIR}/")

    # 4. Network structure: deduce nuclides + reactions + detailed balance,
    #    run the formal A/Q conservation check, and emit the three CSVs (plus
    #    large.txt) that PyPRIMAT reads at run time to assemble the large
    #    network.
    reactions = unified_reactions(tab_blocks, ana_blocks)
    write_network_files(reactions, tab_blocks, args.nubase, args.datadir, args.suffix)


if __name__ == "__main__":
    main()
