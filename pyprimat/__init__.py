# -*- coding: utf-8 -*-
"""
pyprimat — core package for PyPRIMAT.

Public API::

    from pyprimat import PyPR
    result = PyPR({"Omegabh2": 0.022425}).solve()
"""

from .main import PyPR, mc_uncertainty
from .background import Background, StandardBackground
from .nuclear_network import NuclearNetwork
from .network_data import nuclide_latex

__all__ = ["PyPR", "mc_uncertainty", "Background", "StandardBackground",
           "NuclearNetwork", "nuclide_latex"]
