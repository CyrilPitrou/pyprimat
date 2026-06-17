# -*- coding: utf-8 -*-
"""
pyprimat — core package for PyPRIMAT.

Public API::

    from pyprimat import PyPR
    result = PyPR({"Omegabh2": 0.022425}).solve()
"""

from importlib.metadata import version as _version, PackageNotFoundError

from .main import PyPR, mc_uncertainty
from .background import Background, StandardBackground
from .nuclear_network import NuclearNetwork
from .network_data import nuclide_latex

# Single source of truth for the version is pyproject.toml; we read it back
# from the installed distribution metadata so the number is never duplicated.
try:
    __version__ = _version("PyPRIMAT")
except PackageNotFoundError:
    # Running from a source checkout that was never installed (e.g. no
    # `pip install -e .`): metadata is absent, so fall back to a sentinel.
    __version__ = "0.0.0+unknown"

__all__ = ["PyPR", "mc_uncertainty", "Background", "StandardBackground",
           "NuclearNetwork", "nuclide_latex", "__version__"]
