# Parthenope-3.0 small-network rate extraction

Reproducible extraction of the 12 PyPRIMAT *small-network* thermonuclear rates
from the Parthenope 3.0 Fortran source, using the **default `PIS2020`** nuclear
rate selection (`dpg=ddn=ddp="PIS2020"`).

## Method (why this is faithful)

The long polynomial fits in `parthenope3.0.f` are error-prone to retype, so we
**do not** transcribe them.  Instead `frag_*.f` are *verbatim* line ranges cut
from `parthenope3.0.f`:

| fragment | parthenope3.0.f lines | reaction (f-index) | pyprimat name |
|----------|----------------------|--------------------|---------------|
| frag_temp | 1744–1790 | temperature-power factors | — |
| frag_r12  | 1794–1818 | H(n,g)H2     f(12) | npTOdg |
| frag_r20  | 1913–1952 | H2(p,g)He3   f(20) PIS2020 | dpTOHe3g |
| frag_r28  | 2315–2359 | H2(d,n)He3   f(28) PIS2020 | ddTOHe3n |
| frag_r29  | 2500–2545 | H2(d,p)H3    f(29) PIS2020 | ddTOtp |
| frag_r21  | 2139–2146 | H3(p,g)He4   f(21) | tpTOag |
| frag_r30  | 2677–2703 | H3(d,n)He4   f(30) | tdTOan |
| frag_r26  | 2246–2277 | He4(t,g)Li7  f(26) | taTOLi7g |
| frag_r16  | 1843–1866 | He3(n,p)H3   f(16) | He3nTOtp |
| frag_r31  | 2705–2740 | He3(d,p)He4  f(31) | He3dTOap |
| frag_r27  | 2279–2311 | He4(He3,g)Be7 f(27) | He3aTOBe7g |
| frag_r17  | 1868–1891 | Be7(n,p)Li7  f(17) | Be7nTOLi7p |
| frag_r24  | 2184–2232 | Li7(p,a)He4  f(24) | Li7pTOaa |
| frag_ex   | 3550–3575 | underflow-safe `ex()` | — |

`assemble.py` wraps these fragments in a loop over the PyPRIMAT master T9 grid
(500 log-spaced points, 1e-3…10 GK) and emits, per reaction:

    t9   f(idx)   sqrt((1+drate_up)/(1+drate_lo))

The **error column** is Parthenope's *full* propagated 1-sigma multiplicative
envelope: the statistical `fp`/`fm` fits **plus** the systematic floor and
inflation factor that Parthenope folds into `drate` for the Serpico reactions
(e.g. Li7(p,a) carries an 8% floor).  For the PIS2020 d-reactions `drate` is
exactly `fp/f-1`, `fm/f-1`, so the envelope reduces to `sqrt(fp/fm)`.

`postprocess.py` then repairs the unphysical low-T9 tail: below the temperature
where a published fit first extrapolates to a negative rate (well outside the
range Parthenope ever evaluates it), the rate is filled by log-log linear
extrapolation of the two lowest valid points and the error is held constant.
That region is dynamically frozen for BBN; the fill only keeps the table
positive/monotone for PyPRIMAT's log-log resampler.

## Rebuild

```bash
SRC=/path/to/parthenope3.0/parthenope3.0.f   # update if relocated
# (re-cut frag_*.f with: sed -n 'A,Bp' "$SRC" > frag_rXX.f, ranges above)
python3 assemble.py
export SDKROOT=$(xcrun --show-sdk-path)       # macOS linker only
gfortran -ffixed-line-length-none -L"$SDKROOT/usr/lib" -o genrates genrates.f
./genrates
python3 postprocess.py    # writes *_parthenope3.0.txt into rates/nuclear/tables
```

Output tables are consumed via the network file
`rates/nuclear/networks/small_parthenope3.0.txt`.
