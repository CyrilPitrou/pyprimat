#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
doc/generate_tab_reactions.py
==============================
Regenerate doc/tab_reactions.tex from the *current* nuclear-network data
(primat/data/csv/reactions_large.csv, primat/data/nuclear/networks/{small,large}.txt).

Why this exists
----------------
tab_reactions.tex was originally produced by hand/one-off script for
PyPRIMAT v0.1.0, back when the code had three named networks ("small",
"medium", "large"). The "medium" network was later removed in favour of
`network="large", amax=8` (see primat/config.py's DEFAULT_PARAMS["amax"]
docstring), which changed the reaction count of that intermediate tier from
62 to 68 -- so the old table's counts/captions no longer match the code.
There is no committed generator for these tables, so this script recreates
one, driven directly by primat.network_data.load_network (the same function
the solver uses), to guarantee the three tiers documented here
(small / large-amax8 / large) exactly match what a user gets by setting
those parameters.

Run from the repository root:

    python doc/generate_tab_reactions.py

and recompile the LaTeX document afterwards.
"""
import csv
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from primat.network_data import load_network
from primat.config import PRIMATConfig

CSV_PATH = os.path.join(_ROOT, "primat", "data", "csv", "reactions_large.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "tab_reactions.tex")

# Species with a dedicated shorthand symbol instead of isotope notation
# (matches the convention used throughout the main text and appendices).
_SHORTHAND = {
    "n": "n", "p": "p",
    "H2": "d", "H3": "t", "He4": r"\alpha",
    "g": r"\gamma", "Bm": "e^-", "Bp": "e^+",
}
_NAME_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def symbol(species):
    """LaTeX symbol for a species token from reactions_large.csv (e.g. 'He4', 'C12', 'g')."""
    if species in _SHORTHAND:
        return _SHORTHAND[species]
    m = _NAME_RE.match(species)
    if not m:
        raise ValueError(f"Unrecognised species token: {species!r}")
    el, mass = m.groups()
    return rf"{{}}^{{{mass}}}\mathrm{{{el}}}"


def side_tex(tokens):
    return " + ".join(symbol(t) for t in tokens.split("+"))


def source_tex(ref, kind):
    """Render the ref= short code as LaTeX, with the (borrowed) / dagger conventions."""
    if ref.startswith("="):
        return ref[1:] + " (borrowed)"
    code = ref.replace("&", r"\&")
    if kind == "analytic":
        code += r"$^\dagger$"
    return code


def load_csv_rows():
    rows = {}
    with open(CSV_PATH, newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row["name"]] = row
    return rows


def tex_row(name, rows):
    row = rows[name]
    reactants, products = row["reactants"], row["products"]
    arrow = r"\to" if ("Bm" in products.split("+") or "Bp" in products.split("+")) else r"\leftrightarrow"
    eq = f"${side_tex(reactants)} {arrow} {side_tex(products)}$"
    return eq, source_tex(row["ref"], row["source"])


def two_col_table(names, rows, caption, label):
    lines = [
        r"\begin{center}\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}r l l @{\hspace{0.4em}} r l l@{}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        r"\# & Reaction & Source & \# & Reaction & Source\\\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{6}}{{r}}{{\footnotesize\itshape table~\ref{{{label}}} continued}}\\\toprule",
        r"\# & Reaction & Source & \# & Reaction & Source\\\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    n = len(names)
    half = (n + 1) // 2
    left, right = names[:half], names[half:]
    for i in range(half):
        eq_l, src_l = tex_row(left[i], rows)
        cells = f"{i + 1} & {eq_l} & {src_l}"
        if i < len(right):
            eq_r, src_r = tex_row(right[i], rows)
            cells += f" & {half + i + 1} & {eq_r} & {src_r}"
        else:
            cells += " & & &"
        lines.append(cells + r"\\")
    lines += [r"\end{longtable}", r"\end{center}", ""]
    return "\n".join(lines)


def main():
    rows = load_csv_rows()

    small = load_network(PRIMATConfig({"network": "small"})).names[1:]
    amax8 = load_network(PRIMATConfig({"network": "large", "amax": 8})).names[1:]
    full = load_network(PRIMATConfig({"network": "large"})).names[1:]

    small_set, amax8_set = set(small), set(amax8)
    g2 = [n for n in amax8 if n not in small_set]     # large(amax=8) additions beyond small
    g3 = [n for n in full if n not in amax8_set]      # large additions beyond large(amax=8)

    assert len(small) == 12, len(small)
    print(f"small: {len(small)}, large(amax=8) additions: {len(g2)}, "
          f"large additions: {len(g3)}  (totals: {len(small)}, "
          f"{len(small) + len(g2)}, {len(small) + len(g2) + len(g3)})")

    out = []
    out.append(two_col_table(
        small, rows,
        r"Small network (12 reactions). $^\dagger$ = analytic fit hard-coded "
        r"in PRIMAT-Main.m.",
        "tab:rxn-small"))
    out.append(two_col_table(
        g2, rows,
        rf"\code{{large}} network additions at \code{{amax=8}} beyond the small "
        rf"network ({len(g2)} reactions): this is the exact reaction set of the "
        rf"former \code{{medium}} network, reproduced by "
        rf"\code{{\{{\"network\": \"large\", \"amax\": 8\}}}}.",
        "tab:rxn-medium"))
    out.append(two_col_table(
        g3, rows,
        rf"Full \code{{large}} network additions beyond \code{{amax=8}} "
        rf"({len(g3)} reactions).",
        "tab:rxn-large"))

    with open(OUT_PATH, "w") as fh:
        fh.write("\n".join(out))
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
