# generate_rates/

Offline scripts that produce the rate tables shipped under `pyprimat/rates/`.
None of these run at BBN-solve time — they are run once (or whenever an
upstream source changes) to regenerate committed data files.

## Pipeline map

- **`convert_ac2024_rates.py`** — the main entry point. Builds the
  500-point log-uniform-T9 rate-table set from `BBNRatesAC2024.dat` (tabulated
  reactions) plus a hard-coded analytic-rate table (reactions PRIMAT evaluates
  in closed form). Produces:
  - `pyprimat/rates/nuclear/tables/*.txt` — per-reaction rate tables.
  - `pyprimat/rates/nuclear/data/*.csv` — `nuclides.csv`, `reactions_large.csv`,
    `detailed_balance.csv`.
  - `pyprimat/rates/nuclear/networks/large.txt` — the large-network reaction
    list.

  It imports `nuclide_table.py` (nuclide property table, NUBASE2020 parsing,
  detailed-balance helper) and, through it, `nuclear_data.py` (the
  `detailed_balance` reverse-rate-coefficient formula).

- **`generate_qed_tables.py`** — independent of the above. Recomputes the
  analytic QED plasma-pressure correction tables (`pyprimat.qed_pressure`)
  and writes them to `pyprimat/rates/plasma/`.

- **`parthenope3.0_extract/`** — a separate, self-contained sub-pipeline that
  extracts the 12 PyPRIMAT *small-network* rates directly from the Parthenope
  3.0 Fortran source (verbatim code fragments, not retyped formulas). See its
  own `parthenope3.0_extract/README.md` for the method and how to run it.

- **`PRIMAT-Main_decays.m`** — the original Mathematica
  source `convert_ac2024_rates.py`'s analytic-rate table was extracted from.
  Kept only as a reference to regenerate that table if PRIMAT-Main.m changes
  (`--dump-analytic`); not needed for a normal rate-table rebuild.

- **`nubase_4.mas20.txt`**, **`BBNRatesAC2024*.dat`** — upstream source data
  (NUBASE2020 mass table; the AC2024 tabulated-rate compilation).
