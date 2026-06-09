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
  * ``<outdir>/<reactants>TO<products>.txt`` for every reaction: a header line
    (``#`` comment, ignored by ``numpy.loadtxt``) recording the reaction, its
    reference and its detailed-balance coefficients, then three columns
    ``T9  rate  error`` on the 500-point grid.
  * ``<outdir>/detailed_balance.csv``: reaction, Q, alpha, beta, gamma for all
    reactions (the backward rate is ``alpha * T9**beta * exp(gamma/T9)`` times
    the forward rate).

The naming convention ``<reactants>TO<products>`` (e.g. ``n + p > d + g`` ->
``npTOdg``) cleanly separates the initial and final state, removing the
ambiguity of the old prefix-free names (``npdg``).

Usage::

    python generate_from_primat/convert_ac2024_rates.py \
        --input generate_from_primat/BBNRatesAC2024.dat \
        --primat generate_from_primat/PRIMAT-main.m \
        --outdir Rates/nuclear/tables \
        --suffix "_parthenope" \
        --produce-csv
"""
import argparse
import os
import re

import numpy as np
from scipy.interpolate import interp1d

# Standard target grid: 500 points, log-uniform from T9 = 1e-3 to 1e1.
GRID_NPTS = 500
GRID_T9_MIN = 1.0e-3
GRID_T9_MAX = 1.0e1

# Numbers may use Fortran 'D'/'d' double-precision exponents (e.g. 1.1133D+10).
_NUM = r"[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eEdD][-+]?[0-9]+)?"
_FLOATS_RE = re.compile(rf"^\s*(?:{_NUM})(?:\s+{_NUM})*\s*$")


def _to_float(token):
    return float(token.replace("D", "e").replace("d", "e"))


def standard_grid():
    return np.logspace(np.log10(GRID_T9_MIN), np.log10(GRID_T9_MAX), GRID_NPTS)


# Canonical short tokens so the same nuclide always yields the same name,
# whether the source spells it 'He4' or 'a' (AC2024 uses a/d/t; PRIMAT-main.m
# mixes He4/a, etc.).
_CANON_TOKEN = {"He4": "a", "H2": "d", "H3": "t"}


def reaction_name(reactants, products):
    """Build the canonical ``<reactants>TO<products>`` short name."""
    def canon(side):
        return "".join(_CANON_TOKEN.get(t, t) for t in side)
    return canon(reactants) + "TO" + canon(products)


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
        outdir : directory path for output files.
        suffix : appended to the reaction name before ".txt".
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
        f"   [{block['name']}]   ref={block['ref']}\n"
        f"detailed balance: alpha={block['alpha']:.6g} beta={block['beta']:.6g} "
        f"gamma={block['gamma']:.6g}  Q={block['Q']:.6g}\n"
        f"T9                 rate                error"
    )
    path = os.path.join(outdir, block["name"] + suffix + ".txt")
    np.savetxt(path, np.column_stack([grid, rate, err]),
               fmt=["%.6e", "%.6e", "%.6e"], header=header)
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
    rate = np.asarray(block["rate"](grid) * np.ones_like(grid), dtype=float)
    # Analytic rates carry a single constant multiplicative uncertainty factor
    # f (the AddReaction argument): the 1-sigma band is [rate/f, rate*f]. We
    # store it as the (temperature-independent) error column, matching the
    # tabulated-reaction convention. Decays have f=1 (no uncertainty).
    err = block["f"] * np.ones_like(grid)
    header = (
        f"{' + '.join(block['reactants'])} > {' + '.join(block['products'])}"
        f"   [{block['name']}]   ref={block['ref']}  (analytic, PRIMAT-main.m)\n"
        f"forward[T9] = {block['expr']}   uncertainty factor f = {block['f']:g}\n"
        f"T9                 rate                error"
    )
    path = os.path.join(outdir, block["name"] + suffix + ".txt")
    np.savetxt(path, np.column_stack([grid, rate, err]),
               fmt=["%.6e", "%.6e", "%.6e"], header=header)
    return path


# ---------------------------------------------------------------------------
# Hard-coded analytic reactions (the single source of truth at run time).
#
# Each entry is (source, reac, forward) where `reac` is the PRIMAT reaction
# string "reactants > products ; name" and `forward` is the raw Mathematica
# forward[T9] expression. These were extracted once from PRIMAT-main.m's
# DefineAnalyticRates; embedding them here makes the rate files regenerable from
# BBNRatesAC2024.dat alone, with no PRIMATreference/ folder. To refresh this
# table after PRIMAT-main.m changes, run:
#     python generate_from_primat/convert_ac2024_rates.py --dump-analytic <path-to-PRIMAT-main.m>
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
    ('', 'Li8>2a+Bm;', 1.0,
     'Log[2]/8.4030*^-1'),
    ('', 'Li9>Be9+Bm;', 1.0,
     'Log[2]/1.7830*^-1 * 0.492'),
    ('', 'Li9>a+a+n+Bm;', 1.0,
     'Log[2]/1.7830*^-1 * 0.508'),
    ('', 'Be11>B11+Bm;', 1.0,
     'Log[2]/(1.3810*^1)'),
    ('', 'Be12>B12+Bm;', 1.0,
     'Log[2]/(2.15*^-2)'),
    ('', 'B8>a+a+Bp;', 1.0,
     'Log[2]/(7.70*^-1)'),
    ('', 'B12>C12+Bm;', 1.0,
     'Log[2]/(2.02*^-2)'),
    ('', 'B13>C13+Bm;', 1.0,
     'Log[2]/(1.733*^-2)'),
    ('', 'B14>C14+Bm;', 1.0,
     'Log[2]/(1.25*^-2)'),
    ('', 'B15>C15+Bm;', 1.0,
     'Log[2]/(9.87*^-3)'),
    ('', 'C9>a+a+p+Bp;', 1.0,
     'Log[2]/(1.26*^-1)'),
    ('', 'C10>B10+Bp;', 1.0,
     'Log[2]/(19.29)'),
    ('', 'C11>B11+Bp;', 1.0,
     'Log[2]/1.2234*^3'),
    ('', 'C15>N15+Bm;', 1.0,
     'Log[2]/2.449'),
    ('', 'C16>N16+Bm;', 1.0,
     'Log[2]/7.4700*^-1'),
    ('', 'N12>C12+Bp;', 1.0,
     'Log[2]/1.100*^-2'),
    ('', 'N13>C13+Bp;', 1.0,
     'Log[2]/5.979*^2'),
    ('', 'N16>O16+Bm;', 1.0,
     'Log[2]/7.13'),
    ('', 'N17>O16+n+Bm;', 1.0,
     'Log[2]/4.1730'),
    ('', 'O13>N13+Bp;', 1.0,
     'Log[2]/8.58*^-3'),
    ('', 'O14>N14+Bp;', 1.0,
     'Log[2]/70.598'),
    ('', 'O15>N15+Bp;', 1.0,
     'Log[2]/122.24'),
    ('', 'O19>F19+Bm;', 1.0,
     'Log[2]/26.464'),
    ('', 'O20>F20+Bm;', 1.0,
     'Log[2]/13.51'),
    ('', 'F17>O17+Bp;', 1.0,
     'Log[2]/64.49'),
    ('', 'F18>O18+Bp;', 1.0,
     'Log[2]/6.5863*^3'),
    ('', 'F20>Ne20+Bm;', 1.0,
     'Log[2]/11.1630'),
    ('', 'Ne18>F18+Bp;', 1.0,
     'Log[2]/1.6720'),
    ('', 'Ne19>F19+Bp;', 1.0,
     'Log[2]/17.296'),
    ('', 'Ne23>Na23+Bm;', 1.0,
     'Log[2]/37.240'),
    ('', 'Na20>Ne20+Bp;', 1.0,
     'Log[2]/4.4790*^-1'),
    ('', 'Na21>Ne21+Bp;', 1.0,
     'Log[2]/22.49'),
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


def write_network_files(reactions, tab_blocks, nubase_path, outdir):
    """Write nuclides.csv, reactions_large.csv and detailed_balance.csv, after
    the formal conservation check and a detailed-balance cross-check.

    Steps:
      1. Deduce the nuclide set and resolve N,Z,A,Q,mass,spin from NUBASE.
      2. **Formal** (integer) check that every reaction conserves baryon number
         A and charge Q; abort on any violation.
      3. Compute (alpha, beta, gamma) for every reversible (non-decay) reaction
         from nuclide data, and cross-check against the AC2024 tabulated values.
      4. Emit the three CSVs PyPRIMAT reads at run time.
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

    with open(os.path.join(outdir, "reactions_large.csv"), "w") as f:
        # reactants/products are '+'-joined canonical token lists (a->He4 etc.);
        # multiplicity is explicit by repetition (e.g. He4+He4+n).
        from nuclide_table import resolve_token

        def canon(side):
            return "+".join(resolve_token(t).name or t for t in side)
        f.write("name,reactants,products,source,ref\n")
        for rxn in sorted(reactions, key=lambda r: r["name"]):
            f.write(f"{rxn['name']},{canon(rxn['reactants'])},"
                    f"{canon(rxn['products'])},{rxn['source']},{rxn['ref']}\n")

    with open(os.path.join(outdir, "detailed_balance.csv"), "w") as f:
        f.write("reaction,Q_keV,alpha,beta,gamma\n")
        for name in sorted(coeffs):
            Q, alpha, beta, gamma = coeffs[name]
            f.write(f"{name},{Q:.6g},{alpha:.8g},{beta:g},{gamma:.8g}\n")

    print(f"wrote nuclides.csv ({len(nuclides)} nuclides), "
          f"reactions_large.csv ({len(reactions)} reactions), "
          f"detailed_balance.csv ({len(coeffs)} reversible reactions)")


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="generate_from_primat/BBNRatesAC2024.dat",
                   help="the tabulated AC2024 reaction-rate compilation")
    p.add_argument("--nubase", default="generate_from_primat/nubase_4.mas20.txt",
                   help="the NUBASE2020 evaluation (nuclide masses and spins)")
    p.add_argument("--tabdir", default="pyprimat/rates/nuclear/tables",
                   help="the directory for reaction rate tables (.txt)")
    p.add_argument("--outdir", dest="tabdir",
                   help="alias for --tabdir")
    p.add_argument("--datadir", default="pyprimat/rates/nuclear/tables",
                   help="the directory for network structure files (.csv)")
    p.add_argument("--suffix", default="",
                   help="optional suffix for generated rate files")
    p.add_argument("--primat", default=None,
                   help="re-extract the analytic reactions from this PRIMAT-main.m "
                        "instead of using the hard-coded table (for verification)")
    p.add_argument("--dump-analytic", metavar="PRIMAT_PATH", default=None,
                   help="print the _ANALYTIC_REACTIONS literal extracted from the "
                        "given PRIMAT-main.m and exit (to regenerate the table)")
    p.add_argument("--produce-csv", action="store_true",
                   help="if set, regenerate the CSV data files (nuclides, reactions_large, detailed_balance)")
    p.add_argument("--keep-source-grid", action="store_true",
                   help="write each tabulated reaction on its own native T9 grid "
                        "(~60 points from the AC2024 file) instead of reinterpolating "
                        "onto the standard 500-point grid.  Analytic reactions always "
                        "use the standard grid.  PyPRIMAT's load_network resamples all "
                        "tables to a master grid at init, so mixing grids is safe.")
    args = p.parse_args(argv)

    if args.dump_analytic:
        _dump_analytic_literal(args.dump_analytic)
        return

    os.makedirs(args.tabdir, exist_ok=True)
    os.makedirs(args.datadir, exist_ok=True)
    grid = standard_grid()

    # 1. Tabulated reactions from the AC2024 file (interpolated onto the grid,
    #    or kept on their native grid when --keep-source-grid is set).
    tab_blocks = parse_blocks(args.input)
    for blk in tab_blocks:
        # With --keep-source-grid, write on the native AC2024 T9 grid (~60 pts)
        # rather than the standard 500-pt grid; load_network resamples at init.
        blk_grid = blk["T9"] if args.keep_source_grid else grid
        write_reaction_file(blk, blk_grid, args.tabdir, args.suffix)
    print(f"parsed {len(tab_blocks)} tabulated reactions from {args.input}")

    # 2. Analytic reactions (evaluated on the grid).  If --primat is provided,
    #    extract them from that file. Otherwise, skip analytic generation.
    ana_blocks = []
    if args.primat:
        entries = extract_analytic_from_primat(args.primat)
        ana_blocks, skipped = build_analytic_blocks(entries)
        for blk in ana_blocks:
            write_analytic_file(blk, grid, args.tabdir, args.suffix)
        print(f"built {len(ana_blocks)} analytic reactions from {args.primat}")
        if skipped:
            print(f"  ({len(skipped)} analytic blocks skipped: "
                  f"{[n for n, _ in skipped]})")
    else:
        print("Skipping analytic reaction generation (no --primat provided)")

    # 3. Report any <reactants>TO<products> name collisions across both sets.
    names = {}
    for blk in tab_blocks + ana_blocks:
        names[blk["name"]] = names.get(blk["name"], 0) + 1
    collisions = sorted(n for n, c in names.items() if c > 1)
    print(f"wrote {len(set(names))} rate files to {args.tabdir}/")
    if collisions:
        print(f"WARNING: {len(collisions)} colliding TO-names "
              f"(last write wins): {collisions}")

    # 4. Network structure: deduce nuclides + reactions + detailed balance,
    #    run the formal A/Q conservation check, and emit the three CSVs that
    #    PyPRIMAT reads at run time to assemble the large network.
    if args.produce_csv:
        reactions = unified_reactions(tab_blocks, ana_blocks)
        write_network_files(reactions, tab_blocks, args.nubase, args.datadir)
    else:
        print("Skipping CSV network structure file generation (--produce-csv not set)")


if __name__ == "__main__":
    main()
