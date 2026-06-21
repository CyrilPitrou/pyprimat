# -*- coding: utf-8 -*-
"""
pyprimat.gui — optional Streamlit front end for PyPRIMAT.

This subpackage is intentionally kept import-light: nothing here (or in
``pyprimat/__init__.py``) imports ``streamlit``/``plotly`` at module load
time, so ``import pyprimat`` continues to work even when the optional
``gui`` extra (``pip install "PyPRIMAT[gui]"``) is not installed.

Launch the GUI with::

    pyprimat-gui                              # console script (after `pip install ".[gui]"`)
    streamlit run pyprimat/gui/app.py         # from a source checkout
    python -m pyprimat.gui.launcher           # equivalent to the console script
"""
