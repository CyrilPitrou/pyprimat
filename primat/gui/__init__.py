# -*- coding: utf-8 -*-
"""
primat.gui — optional Streamlit front end for PyPRIMAT.

This subpackage is intentionally kept import-light: nothing here (or in
``primat/__init__.py``) imports ``streamlit``/``plotly`` at module load
time, so ``import primat`` continues to work even when the optional
``gui`` extra (``pip install "PyPRIMAT[gui]"``) is not installed.

Launch the GUI with::

    primat-gui                              # console script (after `pip install ".[gui]"`)
    streamlit run primat/gui/app.py         # from a source checkout
    python -m primat.gui.launcher           # equivalent to the console script
"""
