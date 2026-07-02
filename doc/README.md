# PRIMAT — documentation

This folder holds the extensive PRIMAT user + physics documentation, updated
for **primat version 0.3.1** (package renamed from `PyPRIMAT`/`pyprimat`;
covers the C backend, the GUI's custom-network builder, Monte-Carlo
rate-uncertainty propagation, and the `data_dir`/`user_nuclear_dir` overlay).

| File | Purpose |
|------|---------|
| `primat_documentation_v0.3.1.tex` | The LaTeX source (usage, plasma thermodynamics, weak interactions, nuclear reactions, sensitivity, appendices A–G). |
| `primat_doc_figures.ipynb` | Jupyter notebook that regenerates every figure into `figures/`. Uses only the public primat API (Python backend). |
| `generate_tab_reactions.py` | Regenerates `tab_reactions.tex` from the current `small`/`large` network + rate-table data (no run needed). |
| `figures/` | PDF figures included by the `.tex` (one per `\includegraphics`). |
| `tab_reactions.tex`, `tab_nuclides.tex` | Generated reaction-list and nuclide-data tables, `\input` by the document. |
| `primat_documentation_v0.3.1.pdf` | Compiled output. |

## Rebuilding

Run from the **repository root** (so `import primat` and the `primat/data/`
data files resolve), then compile from this folder:

```bash
# 1. regenerate the figures (~30 s)
jupyter nbconvert --to notebook --execute --inplace \
    doc/primat_doc_figures.ipynb

# 2. regenerate the reaction-list appendix table (only needed if the
#    network/rate-table data changed; no BBN run required, <1 s)
python doc/generate_tab_reactions.py

# 3. compile the document
cd doc
latexmk -pdf primat_documentation_v0.3.1.tex
```

You can also open the notebook in Jupyter and run *Kernel → Restart & Run All*.

The physics equations are cross-referenced to Pitrou, Coc, Uzan & Vangioni,
*Physics Reports* **754** (2018) 1; an annotated copy of its source is
`../biblio/Pitrou_etal_PhysReptArxivVersion.pdf`.
