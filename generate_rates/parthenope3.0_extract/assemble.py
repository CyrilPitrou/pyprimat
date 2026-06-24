#!/usr/bin/env python3
"""Assemble a standalone Parthenope-3.0 rate generator from verbatim fragments.

We embed the exact fixed-form Fortran code for each small-network reaction
(extracted by line range from parthenope3.0.f, PIS2020 default branches) inside
a loop over the PyPRIMAT master T9 grid and emit, per reaction:

    t9   f(idx)   sqrt((1+drate2)/(1+drate1))

where drate1/drate2 are Parthenope's full propagated lower/upper fractional
deviations (statistical fp/fm *plus* the systematic floor and inflation factor).
"""

# (primat reaction name, Parthenope f-index, fragment file, output unit)
REACTIONS = [
    ("npTOdg",     12, "frag_r12.f", 21),
    ("dpTOHe3g",   20, "frag_r20.f", 22),
    ("ddTOHe3n",   28, "frag_r28.f", 23),
    ("ddTOtp",     29, "frag_r29.f", 24),
    ("tpTOag",     21, "frag_r21.f", 25),
    ("tdTOan",     30, "frag_r30.f", 26),
    ("taTOLi7g",   26, "frag_r26.f", 27),
    ("He3nTOtp",   16, "frag_r16.f", 28),
    ("He3dTOap",   31, "frag_r31.f", 29),
    ("He3aTOBe7g", 27, "frag_r27.f", 30),
    ("Be7nTOLi7p", 17, "frag_r17.f", 31),
    ("Li7pTOaa",   24, "frag_r24.f", 32),
]

def read(fn):
    with open(fn) as f:
        return f.read().rstrip("\n") + "\n"

lines = []
lines.append("      program genrates")
lines.append("      implicit double precision (a-h,o-z)")
lines.append("      integer nchrat, ii, iu, NPTS")
lines.append("      parameter (NPTS=500)")
lines.append("      double precision f(40), drate(40,2)")
lines.append("      double precision t9min, t9max, err")
lines.append("      nchrat=1")
lines.append("      t9min=1.d-3")
lines.append("      t9max=10.d0")
# open one output file per reaction
for name, idx, frag, unit in REACTIONS:
    lines.append("      open(%d,file='out_%s.dat',status='replace')" % (unit, name))
lines.append("      do ii=1,NPTS")
lines.append("        t9=t9min*(t9max/t9min)**(dble(ii-1)/dble(NPTS-1))")
# Re-initialise drate defaults each iteration exactly like the Fortran does
# (so reactions whose drate is only set under a validity condition fall back
# to the same -0.9/+9 default Parthenope uses).
lines.append("        if (nchrat.ne.0) then")
lines.append("          do iu=1,40")
lines.append("            drate(iu,1)=-.9d0")
lines.append("            drate(iu,2)=9.d0")
lines.append("          enddo")
lines.append("        endif")
# temperature factors (verbatim)
lines.append(read("frag_temp.f").rstrip("\n"))
# each reaction block + its write
for name, idx, frag, unit in REACTIONS:
    lines.append("C===== %s  (f(%d)) =====" % (name, idx))
    lines.append(read(frag).rstrip("\n"))
    lines.append("      err=dsqrt((1.d0+drate(%d,2))/(1.d0+drate(%d,1)))" % (idx, idx))
    lines.append("      write(%d,'(es16.8,1x,es16.8,1x,es16.8)') t9,f(%d),err"
                 % (unit, idx))
lines.append("      enddo")
for name, idx, frag, unit in REACTIONS:
    lines.append("      close(%d)" % unit)
lines.append("      end")
lines.append("")
# EX helper (verbatim)
lines.append(read("frag_ex.f").rstrip("\n"))

with open("genrates.f", "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote genrates.f")
