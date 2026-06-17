# -*- coding: utf-8 -*-
"""
network_builder.py
==================
Generic, stoichiometry-driven assembly of the nuclear-network right-hand side
(dY/dt) and its Jacobian.

--------------------------------------------------------------------------------
What problem this solves
--------------------------------------------------------------------------------
A BBN run integrates a stiff system of ODEs, one per nuclide, coupling the
abundances Y_s through hundreds of nuclear reactions.  Writing the algebra of
every reaction by hand (as the original PRIMAT kernels did) is unmaintainable and
hopeless for the ~433-reaction "large" network.  Instead we describe a network
*abstractly* -- as a plain list of reactions, each a pair of
``{species_index: multiplicity}`` dicts -- and **compile** it once into flat
NumPy arrays (:func:`compile_network` -> :class:`CompiledNetwork`).  Two small
array-driven kernels (:func:`_rhs_kernel`, :func:`_jac_kernel`) then evaluate
dY/dt and J for *any* such network.  The numbers are identical, to round-off, to
the hand-written reference -- but the same code path serves the small (8-species),
medium (12) and large (~59) networks.

--------------------------------------------------------------------------------
Lifecycle / how the pieces fit together  (READ THIS FIRST)
--------------------------------------------------------------------------------
There are two very different time-scales, and keeping them straight is the key to
understanding this module:

  * **Once per run (setup):** :func:`compile_network` turns the abstract network
    into a :class:`CompiledNetwork` (immutable flat arrays), and
    :class:`NetworkKernels` binds it to the kernels (JIT-compiling them with
    numba if available).  This happens a handful of times total -- once for each
    era's network -- in ``UpdateNuclearRates`` (see :mod:`pyprimat.network_data`).
    A ``CompiledNetwork`` is therefore *not* recomputed during integration; it is
    a fixed description of the reaction topology.

  * **Every solver step (hot path):** ``scipy``'s BDF integrator calls the ODE
    right-hand side and Jacobian many times per accepted step.  Each such call
    (in :meth:`PyPR.solve`) does two things:
      1. fill the *rate buffer* ``r`` -- the forward/backward reaction rates at
         the current temperature T (interpolated from tables), ``r[2i]`` forward
         and ``r[2i+1]`` backward for reaction ``i``;
      2. call :meth:`NetworkKernels.rhs` / :meth:`NetworkKernels.jacobian`, which
         run :func:`_rhs_kernel` / :func:`_jac_kernel` over the compiled arrays
         with the current abundances ``Y``, baryon density ``rho`` and ``r``.
    These two kernels are the innermost loop of the whole BBN computation, which
    is why they are flat-array and numba-JIT-friendly (no Python objects, no
    per-reaction function calls).

So: the *topology* (who reacts with whom, the stoichiometry) is compiled once;
only the *rates* and *abundances* change from step to step.

--------------------------------------------------------------------------------
The mathematics the kernels implement
--------------------------------------------------------------------------------
Each reaction ``i`` has reactants with multiplicities c_s^react and products with
c_s^prod (e.g. d + d -> He4 + g has c_d^react = 2, c_He4^prod = 1).  Mass-action
kinetics gives a net flux F_i = F_forward - F_backward with

    F_forward  = r[2i]   * rho**(R-1) / sym(reactants) * prod_s Y_s**(c_s^react)
    F_backward = r[2i+1] * rho**(P-1) / sym(products)  * prod_s Y_s**(c_s^prod)

and every nuclide's equation accumulates its net stoichiometric share of it:

    dY_s/dt += (c_s^prod - c_s^react) * F_i .

Here, per reaction:
  * R = sum_s c_s^react and P = sum_s c_s^prod are the total reactant/product
    multiplicities; the density powers rho**(R-1), rho**(P-1) come from writing
    the rate per baryon (an n-body rate scales as number-density**(n-1), and
    Y_s = n_s/n_baryon factors the n_baryon = rho-dependence into rho**(n-1)).
  * sym(side) = prod_s (c_s!) is the **symmetry factor** that avoids double
    counting identical reactants/products (d + d carries 1/2!, three alphas 1/3!).
  * ``prod_s Y_s**c_s`` is the **monomial** -- the product of abundances raised to
    their multiplicities -- which is where the Y-dependence (and hence the
    Jacobian) lives.

The Jacobian J[s, u] = d(dY_s/dt)/dY_u follows by differentiating the two
monomials with the power rule; see :func:`_jac_kernel`.

``r`` is the flat forward/backward rate buffer exactly as filled by
``UpdateNuclearRates`` (via ``NetworkDefinition.fill_buffer``).
"""
from math import factorial

import numpy as np

__all__ = ["compile_network", "CompiledNetwork", "NetworkKernels",
           "check_conservation"]


class CompiledNetwork:
    """Flat-array encoding of a network's *topology*, ready for the numba kernels.

    This is a plain data container (no methods): the immutable result of
    :func:`compile_network`, built **once per network at run setup** and then read
    -- never modified -- by the kernels on every solver step.  It holds everything
    about *who reacts with whom and with what stoichiometry*; the time-varying
    rates and abundances are passed in separately at evaluation time.

    Storage layout.  The reactions have wildly different sizes (1-body decays up
    to 3-body reactions), so a ragged structure is flattened into **padded 2-D
    arrays**: one row per reaction, columns padded with zeros to the network-wide
    maximum, plus a companion ``*_len`` array giving the number of valid columns
    in each row.  This fixed-shape, object-free layout is what lets numba JIT the
    kernels into tight machine-code loops.  Indices and integer powers are
    ``int64``; net coefficients and prefactors are ``float64``.

    Fields (``n_rx`` reactions, ``n_sp`` species):
      reactant monomial : ri_idx (species indices), ri_pow (their multiplicities),
                          ri_len (how many reactant species)
      product  monomial : pi_idx, pi_pow, pi_len   (same, product side)
      net update        : af_idx, af_co, af_len    (dY[af_idx] += af_co * flux;
                          af_co = c^prod - c^react, only the species that change)
      jacobian variables: vr_idx, vr_len           (union of reactant+product
                          species: the only Y the flux -- and so the only Jacobian
                          columns -- this reaction depends on)
      density/symmetry  : Rm1 (=R-1), Pm1 (=P-1), invsr (=1/sym_reactants),
                          invsp (=1/sym_products)

    Worked example -- reaction ``d + d -> He4 + g`` with species order
    ``[n, p, d, He4]`` (photons are not tracked), as reaction row ``i``:
      ri_idx[i] = [2],      ri_pow[i] = [2],   ri_len[i] = 1   # two deuterons
      pi_idx[i] = [3],      pi_pow[i] = [1],   pi_len[i] = 1   # one He4
      af_idx[i] = [2, 3],   af_co[i] = [-2, +1], af_len[i] = 2 # d: -2, He4: +1
      vr_idx[i] = [2, 3],   vr_len[i] = 2                      # flux depends on d, He4
      Rm1[i] = 1, Pm1[i] = 0, invsr[i] = 1/2! = 0.5, invsp[i] = 1
    """

    __slots__ = ("n_rx", "n_sp", "ri_idx", "ri_pow", "ri_len",
                 "pi_idx", "pi_pow", "pi_len", "af_idx", "af_co", "af_len",
                 "vr_idx", "vr_len", "Rm1", "Pm1", "invsr", "invsp")


def _pad(rows, width, dtype):
    """Stack ragged integer/float ``rows`` into an ``(n, width)`` padded array."""
    out = np.zeros((len(rows), width), dtype=dtype)
    for i, row in enumerate(rows):
        out[i, :len(row)] = row
    return out


def compile_network(network, n_sp):
    """Compile an abstract network into a :class:`CompiledNetwork`.

    Run once per network at setup.  For each reaction it derives, from the two
    stoichiometry dicts, everything the kernels need: the two monomials
    (which species, to which power), the net per-species change
    (``+products - reactants``), the symmetry factors, the density powers, and
    the set of variables the flux depends on.  These ragged per-reaction lists
    are then padded into the fixed-shape arrays of a :class:`CompiledNetwork`.

    Parameters
    ----------
    network : list[(reactants, products)]
        Each side a ``{species_index: multiplicity}`` dict, as produced by
        :func:`pyprimat.reactions.phase_network`.  Example reaction ``d + d -> He4 + g``
        with ``d`` at index 2 and ``He4`` at index 3 is ``({2: 2}, {3: 1})``.
    n_sp : int
        Number of species in the abundance vector Y.

    Returns
    -------
    CompiledNetwork
        Immutable flat-array description; see that class for the field layout and
        a worked example.
    """
    ri_idx, ri_pow, pi_idx, pi_pow = [], [], [], []
    af_idx, af_co, vr_idx = [], [], []
    Rm1, Pm1, invsr, invsp = [], [], [], []

    for react, prod in network:
        ri_idx.append(list(react.keys()))
        ri_pow.append(list(react.values()))
        pi_idx.append(list(prod.keys()))
        pi_pow.append(list(prod.values()))
        R = sum(react.values())
        P = sum(prod.values())
        Rm1.append(float(R - 1))
        Pm1.append(float(P - 1))
        # Symmetry factor prod_s (m_s!) on each side (e.g. d+d over-counts by 2!).
        sr = 1
        for c in react.values():
            sr *= factorial(c)
        sp = 1
        for c in prod.values():
            sp *= factorial(c)
        invsr.append(1.0 / sr)
        invsp.append(1.0 / sp)
        # Net stoichiometric change per species: +product, -reactant.
        net = {}
        for s, c in react.items():
            net[s] = net.get(s, 0) - c
        for s, c in prod.items():
            net[s] = net.get(s, 0) + c
        af = [(s, c) for s, c in net.items() if c != 0]
        af_idx.append([s for s, _ in af])
        af_co.append([float(c) for _, c in af])
        # Species the reaction flux depends on (union of both monomials): the
        # only columns of the Jacobian this reaction can contribute to.
        vr_idx.append(sorted(set(react) | set(prod)))

    cn = CompiledNetwork()
    cn.n_rx = len(network)
    cn.n_sp = n_sp
    MR = max((len(r) for r in ri_idx), default=1) or 1
    MP = max((len(r) for r in pi_idx), default=1) or 1
    MA = max((len(r) for r in af_idx), default=1) or 1
    MV = max((len(r) for r in vr_idx), default=1) or 1
    cn.ri_idx = _pad(ri_idx, MR, np.int64); cn.ri_pow = _pad(ri_pow, MR, np.int64)
    cn.pi_idx = _pad(pi_idx, MP, np.int64); cn.pi_pow = _pad(pi_pow, MP, np.int64)
    cn.af_idx = _pad(af_idx, MA, np.int64); cn.af_co = _pad(af_co, MA, np.float64)
    cn.vr_idx = _pad(vr_idx, MV, np.int64)
    cn.ri_len = np.array([len(r) for r in ri_idx], np.int64)
    cn.pi_len = np.array([len(r) for r in pi_idx], np.int64)
    cn.af_len = np.array([len(r) for r in af_idx], np.int64)
    cn.vr_len = np.array([len(r) for r in vr_idx], np.int64)
    cn.Rm1 = np.array(Rm1); cn.Pm1 = np.array(Pm1)
    cn.invsr = np.array(invsr); cn.invsp = np.array(invsp)
    return cn


def check_conservation(compiled, N, Z, weak_indices=(), lepton_dZ=None):
    """Formally verify the assembled stoichiometry conserves the right charges.

    This is an *exact integer* check (no floating point, no test abundances): for
    each reaction the net coefficients ``af_co`` (``+products − reactants`` per
    species) are dotted with the species' N and Z.  Lepton charge contributions
    (from emitted electrons/positrons not in the ODE state vector) are added via
    ``lepton_dZ`` to restore uniform charge conservation across all reactions.

    Conservation rules:
      * **A = N + Z** must be conserved by every reaction (nucleons are conserved).
      * **Electric charge Z** must be conserved by every reaction, including weak
        ones — the n↔p weak rate emits an electron (A=0, Z=−1) whose charge is
        accounted for via ``lepton_dZ`` rather than appearing in the ODE species.
      * **Neutron number N** must be conserved by every *nuclear* (strong/EM)
        reaction.  Weak processes convert n↔p, so they change N by ±1 — this is
        expected and they are therefore exempt from the N check (listed in
        ``weak_indices``).

    A violation means the reaction list is physically inconsistent, so we raise
    rather than integrate nonsense.

    Parameters
    ----------
    compiled : CompiledNetwork
    N, Z : sequence[int]
        Neutron and proton number of each species, indexed as the abundance
        vector (i.e. the ``species`` order used to build the network).
    weak_indices : iterable[int]
        Reaction indices (rows of ``compiled``) that are weak processes; these
        may change N (n↔p) and are exempted from the N-conservation check.
        They must still conserve Z (verified via ``lepton_dZ``).
    lepton_dZ : sequence[int] or None
        Net electric charge carried by emitted/absorbed leptons, one entry per
        reaction.  E.g. ``lepton_dZ[0] = -1`` for n__p (β⁻ electron emitted
        with Z=−1 that balances the nuclear ΔZ = +1 from n→p).  When ``None``,
        the old behaviour (A-only for weak reactions) is used as a fallback.
    """
    N = np.asarray(N); Z = np.asarray(Z)
    weak = set(weak_indices)
    bad = []
    for i in range(compiled.n_rx):
        dN = dZ_nuc = 0   # nuclear N and Z changes (from ODE species only)
        for k in range(compiled.af_len[i]):
            s = compiled.af_idx[i, k]
            c = int(round(compiled.af_co[i, k]))
            dN += c * int(N[s])
            dZ_nuc += c * int(Z[s])
        # Total electric-charge change including emitted/absorbed leptons.
        # A_lepton = 0, so leptons do NOT contribute to baryon number dA.
        ldZ = int(lepton_dZ[i]) if (lepton_dZ is not None and i < len(lepton_dZ)) else 0
        dZ_total = dZ_nuc + ldZ      # should be 0 for all reactions
        dA = dN + dZ_nuc             # baryon number change (A_lepton = 0): should be 0
        if i in weak:
            # Weak: N is not conserved (n↔p), but A and total Q must be zero.
            # Fall back to A-only check if no lepton info is provided (legacy).
            if lepton_dZ is not None:
                if dA != 0 or dZ_total != 0:
                    bad.append((i, dN, dZ_nuc))
            else:
                if dA != 0:                   # legacy: only baryon number A=N+Z
                    bad.append((i, dN, dZ_nuc))
        elif dN != 0 or dZ_nuc != 0:         # nuclear: N and Z separately
            bad.append((i, dN, dZ_nuc))
    if bad:
        raise ValueError(
            f"network violates N/Z conservation in {len(bad)} reaction(s) "
            f"(index, dN, dZ): {bad[:5]}{'...' if len(bad) > 5 else ''}")


# ---------------------------------------------------------------------------
# Kernels (plain Python; JIT-compiled by NetworkKernels when numba is present)
# ---------------------------------------------------------------------------
def _rhs_kernel(Y, rho, r, ri_idx, ri_pow, ri_len, pi_idx, pi_pow, pi_len,
                Rm1, Pm1, invsr, invsp, af_idx, af_co, af_len):
    """Right-hand side dY/dt of the whole network at one (Y, rho, r).

    This is the inner loop called on **every** solver evaluation.  It implements
    the mass-action law from the module docstring: for each reaction it builds the
    net flux (forward minus backward) and adds each nuclide's stoichiometric share
    of it to that nuclide's derivative.

    Parameters mirror :class:`CompiledNetwork` (passed as bare arrays so numba can
    JIT the function): ``Y`` abundances, ``rho`` baryon density, ``r`` the flat
    rate buffer (``r[2i]`` forward, ``r[2i+1]`` backward for reaction ``i``).
    Returns the derivative vector ``dY`` (same shape as ``Y``).
    """
    n_rx = ri_len.shape[0]
    dY = np.zeros(Y.shape[0])
    for i in range(n_rx):
        # Forward flux  F_forward = rate * rho**(R-1) / sym_reactants * monomial,
        # where the monomial is prod_s Y_s**(reactant multiplicity).  We start
        # with the prefactor and multiply in one abundance power per reactant.
        Ff = r[2 * i] * rho ** Rm1[i] * invsr[i]
        for k in range(ri_len[i]):
            Ff *= Y[ri_idx[i, k]] ** ri_pow[i, k]
        # Backward flux: identical construction on the product side.
        Fb = r[2 * i + 1] * rho ** Pm1[i] * invsp[i]
        for k in range(pi_len[i]):
            Fb *= Y[pi_idx[i, k]] ** pi_pow[i, k]
        # Net flux of reaction i; distribute it to the nuclides it changes, each
        # weighted by its net stoichiometric coefficient (+product, -reactant).
        net = Ff - Fb
        for k in range(af_len[i]):
            dY[af_idx[i, k]] += af_co[i, k] * net
    return dY


def _jac_kernel(Y, rho, r, ri_idx, ri_pow, ri_len, pi_idx, pi_pow, pi_len,
                Rm1, Pm1, invsr, invsp, af_idx, af_co, af_len, vr_idx, vr_len):
    """Analytic Jacobian J[s, u] = d(dY_s/dt)/dY_u of the whole network.

    The stiff BDF integrator needs the Jacobian at every step.  We supply it
    analytically (rather than by finite differences) for accuracy and speed.

    Derivation.  From the RHS, dY_s/dt = sum_i (c_s^prod - c_s^react)_i * F_i, so
        J[s, u] = sum_i (net coeff of s in i) * dF_i/dY[u] .
    Only the two **monomials** of F_i depend on Y, so we just need dF_i/dY[u].
    With the forward/backward prefactors cf, cb (the rate*rho**.../sym parts,
    independent of Y),
        F_i = cf * M_react(Y) - cb * M_prod(Y),   M_side = prod_s Y_s**(c_s).
    Differentiating a monomial by the power rule: if species ``u`` appears in the
    monomial with power p (p = 0 if absent), then
        d/dY[u] prod_s Y_s**(c_s) = p * Y[u]**(p-1) * prod_{s != u} Y_s**(c_s).
    Hence dF_i/dY[u] = cf * dM_react/dY[u] - cb * dM_prod/dY[u].

    Sparsity.  F_i depends only on the species in its two monomials, i.e. on
    ``vr_idx[i]`` (their union).  Looping ``u`` over just those columns -- instead
    of all n_sp -- is what keeps the Jacobian affordable for the large network.

    The two monomial-derivative loops are inlined (not factored into a helper) so
    the whole kernel JIT-compiles as a single numba nopython function.
    Returns the dense ``(n_sp, n_sp)`` matrix J.
    """
    n_rx = ri_len.shape[0]
    n_sp = Y.shape[0]
    J = np.zeros((n_sp, n_sp))
    for i in range(n_rx):
        cf = r[2 * i] * rho ** Rm1[i] * invsr[i]        # forward prefactor (Y-independent)
        cb = r[2 * i + 1] * rho ** Pm1[i] * invsp[i]    # backward prefactor (Y-independent)
        # Differentiate F_i only w.r.t. the species it actually involves.
        for vj in range(vr_len[i]):
            u = vr_idx[i, vj]
            # --- d(reactant monomial)/dY[u] -------------------------------
            # Find u's power in the reactant monomial (0 if u is not a reactant).
            pu = 0
            for k in range(ri_len[i]):
                if ri_idx[i, k] == u:
                    pu = ri_pow[i, k]
            dmr = 0.0
            if pu != 0:                       # power rule: p * Y[u]**(p-1) * (rest)
                dmr = pu * Y[u] ** (pu - 1)
                for k in range(ri_len[i]):    # multiply in every *other* reactant
                    if ri_idx[i, k] != u:
                        dmr *= Y[ri_idx[i, k]] ** ri_pow[i, k]
            # --- d(product monomial)/dY[u] (same construction) ------------
            pu = 0
            for k in range(pi_len[i]):
                if pi_idx[i, k] == u:
                    pu = pi_pow[i, k]
            dmp = 0.0
            if pu != 0:
                dmp = pu * Y[u] ** (pu - 1)
                for k in range(pi_len[i]):
                    if pi_idx[i, k] != u:
                        dmp *= Y[pi_idx[i, k]] ** pi_pow[i, k]
            # dF_i/dY[u]; skip if this variable does not affect the flux.
            dnet = cf * dmr - cb * dmp
            if dnet == 0.0:
                continue
            # Scatter into column u of every row the reaction changes.
            for k in range(af_len[i]):
                J[af_idx[i, k], u] += af_co[i, k] * dnet
    return J


class NetworkKernels:
    """Bind a :class:`CompiledNetwork` to its (optionally JIT-compiled) kernels.

    Created once per network (alongside the ``CompiledNetwork``).  It simply holds
    the compiled arrays and the chosen kernel implementations, and exposes the two
    methods the ODE solver calls on every step:

      * ``rhs(Y, rho, r)``      -> dY/dt           (:func:`_rhs_kernel`)
      * ``jacobian(Y, rho, r)`` -> J = d(dY/dt)/dY (:func:`_jac_kernel`)

    ``r`` is the rate buffer already filled for the current temperature.  The
    methods are thin wrappers that unpack the ``CompiledNetwork`` fields into the
    positional arguments the bare-array kernels expect.

    Pass ``numba=True`` to JIT-compile the kernels (strongly recommended: these
    two functions dominate the BBN integration cost).  If numba is unavailable the
    plain-Python kernels are used unchanged, so results are identical -- only
    slower.
    """

    def __init__(self, compiled, numba=False):
        self._c = compiled
        rhs, jac = _rhs_kernel, _jac_kernel
        if numba:
            try:
                from numba import njit
                rhs = njit(cache=True)(_rhs_kernel)
                jac = njit(cache=True)(_jac_kernel)
            except Exception:                            # numba absent/broken
                pass
        self._rhs, self._jac = rhs, jac

    def rhs(self, Y, rho, r):
        c = self._c
        return self._rhs(Y, rho, r, c.ri_idx, c.ri_pow, c.ri_len,
                         c.pi_idx, c.pi_pow, c.pi_len, c.Rm1, c.Pm1,
                         c.invsr, c.invsp, c.af_idx, c.af_co, c.af_len)

    def jacobian(self, Y, rho, r):
        c = self._c
        return self._jac(Y, rho, r, c.ri_idx, c.ri_pow, c.ri_len,
                         c.pi_idx, c.pi_pow, c.pi_len, c.Rm1, c.Pm1,
                         c.invsr, c.invsp, c.af_idx, c.af_co, c.af_len,
                         c.vr_idx, c.vr_len)
