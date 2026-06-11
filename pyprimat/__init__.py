# -*- coding: utf-8 -*-
"""
pyprimat — core package for PyPRIMAT.

Public API::

    from pyprimat import PyPR
    result = PyPR({"Omegabh2": 0.022425}).solve()
"""

from .main import PyPR, mc_uncertainty
from .nuclear import nuclide_latex

__all__ = ["PyPR", "mc_uncertainty", "nuclide_latex"]
