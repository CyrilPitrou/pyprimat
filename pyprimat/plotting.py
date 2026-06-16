# -*- coding: utf-8 -*-
"""
pyprimat.plotting
=================

Shared plotting conventions for nuclide abundance curves, so that every front
end (the ``notebooks/`` figures and the Streamlit GUI) draws the *same* nuclide
with the *same* colour and line style.

Convention
----------
* **Colour encodes the chemical element** (the proton number Z): every isotope
  of hydrogen is one colour, every isotope of helium another, and so on.  This
  is what the eye groups first, so it carries the most salient coordinate.
* **Line style encodes the isotope** within an element: the lightest isotope is
  solid, the next is dashed, then dash-dot, dotted, and progressively finer
  custom dash patterns.  Isotopes are ranked by mass number A, so the ordering
  is reproducible across plots that show different subsets of the network.
* The free neutron ``n`` is not a chemical element; it is drawn in black, solid,
  as a neutral reference curve.

The single public entry point is :func:`nuclide_styles`, which takes the list of
nuclide names (as used throughout PyPRIMAT, e.g. ``["n", "p", "H2", "He4",
"Li7", ...]``) and returns, for each name, a ``(color, linestyle, label)``
triple ready to pass to ``matplotlib``'s ``plot(..., color=, linestyle=,
label=)``.

Example
-------
>>> from pyprimat.plotting import nuclide_styles
>>> styles = nuclide_styles(["n", "p", "H2", "H3", "He3", "He4"])
>>> color, ls, label = styles["He4"]
>>> # color is the helium colour; ls distinguishes He4 from He3; label = '$^{4}$He'
"""
import re

# ---------------------------------------------------------------------------
# Fixed element -> colour map.
#
# Keyed by element symbol so the colour of, say, carbon is identical in every
# figure regardless of which other elements are present.  The palette is a
# hand-ordered set of well-separated matplotlib named colours covering Z = 1
# (H) through Z = 11 (Na), the heaviest element in the `large` network.  Order
# roughly follows the spectrum (warm light elements -> cool heavy ones) so
# neighbouring elements remain visually distinct.
# ---------------------------------------------------------------------------
ELEMENT_COLORS = {
    "H":  "#d62728",   # red
    "He": "#1f77b4",   # blue
    "Li": "#2ca02c",   # green
    "Be": "#9467bd",   # purple
    "B":  "#ff7f0e",   # orange
    "C":  "#8c564b",   # brown
    "N":  "#e377c2",   # pink
    "O":  "#17becf",   # cyan
    "F":  "#bcbd22",   # olive
    "Ne": "#7f7f7f",   # grey
    "Na": "#393b79",   # dark indigo
}
NEUTRON_COLOR = "black"

# ---------------------------------------------------------------------------
# Line styles, ordered from most to least prominent.  The first four are the
# matplotlib named styles; the rest are explicit (offset, dash-pattern) tuples
# giving distinct finer patterns -- needed because some elements have many
# isotopes in the `large` network (boron has eight: B8..B15).
# ---------------------------------------------------------------------------
LINESTYLES = [
    "solid",
    "dashed",
    "dashdot",
    "dotted",
    (0, (5, 1)),            # long dash, tight gap
    (0, (3, 1, 1, 1)),      # dash-dot, tight
    (0, (1, 1)),            # densely dotted
    (0, (5, 1, 1, 1, 1, 1)),# dash-dot-dot
    (0, (7, 2, 1, 2)),      # long dash - dot
    (0, (3, 2, 3, 2, 1, 2)),# dash-dash-dot
]

# Pretty labels for the light nuclides that are conventionally written with a
# special symbol rather than a mass-superscript (the rest fall back to the
# generic "$^{A}$El" form built in _pretty_label).
_SPECIAL_LABELS = {
    "n":  "n",
    "p":  "p",
    "H2": "D",      # deuterium
    "H3": "T",      # tritium
}


def parse_nuclide(name):
    """Split a PyPRIMAT nuclide name into ``(element_symbol, mass_number A)``.

    Handles the two non-standard names used as bookkeeping aliases in the
    network: the free neutron ``"n"`` -> ``("n", 1)`` (treated specially by the
    colour map) and the free proton ``"p"`` -> ``("H", 1)`` (hydrogen-1).  Every
    other name is of the form ``<element><A>`` (e.g. ``"He4"``, ``"Be7"``) and
    is split with a regex into its leading letters (the element symbol) and
    trailing digits (the mass number).

    Parameters
    ----------
    name : str
        Nuclide name as used in ``abundance_names`` / ``run.A`` keys.

    Returns
    -------
    (str, int)
        Element symbol and mass number.  For the neutron the symbol is the
        sentinel ``"n"`` (not a real element), with A = 1.

    Examples
    --------
    >>> parse_nuclide("He4")
    ('He', 4)
    >>> parse_nuclide("p")
    ('H', 1)
    >>> parse_nuclide("n")
    ('n', 1)
    """
    if name == "n":
        return ("n", 1)
    if name == "p":
        return ("H", 1)
    m = re.match(r"^([A-Za-z]+?)(\d+)$", name)
    if m:
        return (m.group(1), int(m.group(2)))
    # Fallback: a bare element symbol with no mass number (unexpected in the
    # standard networks) -- treat A as 0 so it sorts first within its element.
    return (name, 0)


def _pretty_label(name, element, A):
    """Return a LaTeX-free-ish display label, e.g. ``"$^{4}$He"`` or ``"D"``."""
    if name in _SPECIAL_LABELS:
        return _SPECIAL_LABELS[name]
    return rf"$^{{{A}}}${element}"


def nuclide_styles(names):
    """Map each nuclide name to a ``(color, linestyle, label)`` triple.

    Colour is fixed per chemical element (:data:`ELEMENT_COLORS`); line style
    distinguishes isotopes of the same element, assigned by ascending mass
    number A so the lightest isotope present is solid, the next dashed, etc.
    (cycling through :data:`LINESTYLES`).  The free neutron is black/solid.

    Because the isotope ranking is computed *within each element from the names
    actually passed in*, the same nuclide can in principle receive a different
    line style in two plots that show different isotope subsets -- but the
    colour (the dominant visual cue) is always identical, and in practice the
    notebooks/GUI pass the full network's nuclide list, so the styles are
    stable too.

    Parameters
    ----------
    names : sequence of str
        Nuclide names (e.g. ``run.abundance_names``).

    Returns
    -------
    dict
        ``{name: (color, linestyle, label)}`` for every input name.

    Example
    -------
    >>> styles = nuclide_styles(["n", "p", "H2", "He3", "He4"])
    >>> styles["p"][0] == styles["H2"][0]   # same hydrogen colour
    True
    >>> styles["p"][1] != styles["H2"][1]   # different isotope line styles
    True
    """
    # Group the names by element and sort each group by mass number, so that the
    # isotope -> line-style assignment is by ascending A.
    by_element = {}
    parsed = {}
    for nm in names:
        element, A = parse_nuclide(nm)
        parsed[nm] = (element, A)
        by_element.setdefault(element, []).append(nm)
    for element in by_element:
        by_element[element].sort(key=lambda nm: parsed[nm][1])

    styles = {}
    for nm in names:
        element, A = parsed[nm]
        if element == "n":
            color, ls = NEUTRON_COLOR, "solid"
        else:
            color = ELEMENT_COLORS.get(element, "black")
            rank = by_element[element].index(nm)
            ls = LINESTYLES[rank % len(LINESTYLES)]
        styles[nm] = (color, ls, _pretty_label(nm, element, A))
    return styles
