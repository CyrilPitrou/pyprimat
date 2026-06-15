# PyPRIMAT — documentation

This folder holds the extensive PyPRIMAT user + physics documentation.
The current document describes **PyPRIMAT version 0.1.0**; when the code moves to
v0.2.0 (architectural changes), a new `PyPRIMAT_documentation_v0.2.0.{tex,pdf}`
should be added alongside.

| File | Purpose |
|------|---------|
| `PyPRIMAT_documentation_v0.1.0.tex` | The LaTeX source (usage, plasma thermodynamics, weak interactions, nuclear reactions, sensitivity, appendices A–G). |
| `PyPRIMAT_doc_figures.ipynb` | Jupyter notebook that regenerates every figure into `figures/`. Uses only the public PyPRIMAT API. |
| `figures/` | PDF figures included by the `.tex` (one per `\includegraphics`). |
| `tab_reactions.tex`, `tab_nuclides.tex` | Generated reaction-list and nuclide-data tables, `\input` by the document. |
| `PyPRIMAT_documentation_v0.1.0.pdf` | Compiled output. |

## Rebuilding

Run from the **repository root** (so `import pyprimat` and the `rates/` data
files resolve), then compile from this folder:

```bash
# 1. regenerate the figures (~30 s)
jupyter nbconvert --to notebook --execute --inplace \
    doc/PyPRIMAT_doc_figures.ipynb

# 2. compile the document
cd doc
latexmk -pdf PyPRIMAT_documentation_v0.1.0.tex
```

You can also open the notebook in Jupyter and run *Kernel → Restart & Run All*.

The physics equations are cross-referenced to Pitrou, Coc, Uzan & Vangioni,
*Physics Reports* **754** (2018) 1; an annotated copy of its source is in
`../biblio/PhysReptRevised.tex`.
