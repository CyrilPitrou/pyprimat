# -*- coding: utf-8 -*-
"""
Test-only reference RHS/Jacobian oracle for the nuclear network.

``network_rhs``/``network_jacobian`` are a deliberately independent,
pure-Python re-implementation of the mass-action ODE that
:class:`primat.network_builder.NetworkKernels` evaluates with compiled
(numba-able) kernels. They exist *only* to give ``tests/test_network_builder.py``
an exact, easy-to-read oracle to check the production kernels against -- see
FUTURE.md P1.3 (moved here, out of ``primat/network_data.py``, since
production code never calls them).
"""

from math import factorial

import numpy as np

__all__ = ["network_rhs", "network_jacobian"]


def _sym(multiplicities):
    """Return the identical-particle symmetry factor for one reaction side."""
    s = 1
    for c in multiplicities.values():
        s *= factorial(c)
    return s


def network_rhs(Y, rhoBBN, r, network):
    """Reference mass-action RHS for a compiled-by-hand network description.

    This pure-Python implementation is used as an exact, readable oracle for
    tests.  Production solves use :class:`NetworkKernels`, but both evaluate the
    same formula: forward flux minus backward flux, distributed by net
    stoichiometric coefficients.
    """
    dY = np.zeros(len(Y))
    for i, (react, prod) in enumerate(network):
        R = sum(react.values())
        P = sum(prod.values())
        Ff = r[2 * i] * rhoBBN ** (R - 1) / _sym(react)
        for s, c in react.items():
            Ff *= Y[s] ** c
        Fb = r[2 * i + 1] * rhoBBN ** (P - 1) / _sym(prod)
        for s, c in prod.items():
            Fb *= Y[s] ** c
        net = Ff - Fb
        for s, c in react.items():
            dY[s] -= c * net
        for s, c in prod.items():
            dY[s] += c * net
    return dY


def _dmonomial(Y, terms, u):
    """Differentiate ``prod_s Y[s]**terms[s]`` with respect to ``Y[u]``."""
    if u not in terms:
        return 0.0
    v = terms[u] * Y[u] ** (terms[u] - 1)
    for s, c in terms.items():
        if s != u:
            v *= Y[s] ** c
    return v


def network_jacobian(Y, rhoBBN, r, network):
    """Reference analytic Jacobian matching :func:`network_rhs`."""
    n = len(Y)
    J = np.zeros((n, n))
    for i, (react, prod) in enumerate(network):
        R = sum(react.values())
        P = sum(prod.values())
        cf = r[2 * i] * rhoBBN ** (R - 1) / _sym(react)
        cb = r[2 * i + 1] * rhoBBN ** (P - 1) / _sym(prod)
        for u in range(n):
            dnet = cf * _dmonomial(Y, react, u) - cb * _dmonomial(Y, prod, u)
            if dnet == 0.0:
                continue
            for s, c in react.items():
                J[s, u] -= c * dnet
            for s, c in prod.items():
                J[s, u] += c * dnet
    return J
