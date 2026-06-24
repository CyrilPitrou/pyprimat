# -*- coding: utf-8 -*-
"""
nuclear_data.py
================
Offline helper used by ``generate_rates/nuclide_table.py`` (and, through it,
``convert_ac2024_rates.py``) to compute detailed-balance reverse-rate
coefficients for the AC2024 rate-conversion pipeline.

Only :func:`detailed_balance` is live. The module previously also carried a
hard-coded ``_DETAILED_BALANCE`` lookup table and ``load_nubase``/
``reaction_species`` helpers; all three were unused (``reaction_species`` was
additionally broken, importing a non-existent ``primat.nuclear`` module)
and have been removed -- the equivalent, validated versions live in
``primat.network_data``, which is what PyPRIMAT itself and the test suite
import.
"""

from collections import Counter
from math import factorial

import numpy as np

__all__ = ['detailed_balance']


def detailed_balance(reactants, products, cfg):
    """Compute the reverse-rate coefficients (alpha, beta, gamma)."""
    keV, kB, MeV = cfg.keV, cfg.kB, cfg.MeV
    ma_e, me_e = cfg.ma * MeV, cfg.me * MeV
    NZ, EX, SP = cfg.Nuclides, cfg.NuclExcessMass, cfg.NuclSpin

    def mass(s):
        n, z = NZ[s]
        return (n + z) * ma_e + EX[s] * keV - z * me_e

    def binding(s):
        n, z = NZ[s]
        return n * EX["n"] + z * EX["p"] - EX[s]

    n_in, n_out = len(reactants), len(products)
    beta = 1.5 * (n_in - n_out)
    Q = keV * (sum(binding(s) for s in products) - sum(binding(s) for s in reactants))
    gamma = -Q / (kB * 1e9)

    def quantum_factor(side):
        val = 1.0
        for s, m in Counter(side).items():
            term = (2 * SP[s] + 1) * (2 * np.pi / mass(s) / (kB * 1e9)) ** (-1.5)
            val *= term ** m / factorial(m)
        return val

    units = ((ma_e / cfg.clight**2) / (cfg.hbar * cfg.clight)**3) ** (n_in - n_out)
    alpha = quantum_factor(reactants) / quantum_factor(products) * units
    return alpha, beta, gamma
