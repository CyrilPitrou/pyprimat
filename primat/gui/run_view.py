# -*- coding: utf-8 -*-
"""
primat.gui.run_view
======================

:class:`GuiRun` adapts one :func:`primat.backend.run_bbn` result (a plain
dict, identical in shape whether the C or Python backend produced it) plus a
cheaply-built ``PRIMATConfig``/``UpdateNuclearRates`` pair into the same
minimal read-only interface :mod:`primat.gui.panels` needs from a solved
``primat.PRIMAT`` instance: ``get_quantity``/``__getitem__``/
``abundance_names``/``A``/``Z``/``N``/``T_of_t``/``cfg``/``nucl``.

This is what lets the GUI's result/evolution panels run against *either*
backend's output -- including the C backend, which has no live Python
object to introspect at all (``primat.backend.run_bbn`` always returns a
plain dict, never a ``PRIMAT`` instance, per its module docstring). The two
nuclide-interpolator-shaped accessors (``__getitem__``, ``T_of_t``) are built
lazily from the discrete ``EvolutionResult`` arrays via
:func:`primat.evolution.Y_interpolator`/:func:`primat.evolution.T_gamma_interpolator`,
not from any backend-specific continuous interpolator.
"""
from primat.evolution import T_gamma_interpolator, Y_interpolator


class GuiRun:
    """Backend-agnostic stand-in for a solved ``primat.PRIMAT`` instance.

    Args:
        results: dict. A :func:`primat.backend.run_bbn` result (must include
            an ``"evolution"`` key, i.e. the run was made with
            ``output_time_evolution=True``).
        cfg: primat.config.PRIMATConfig. Built from the same ``params`` (no
            solve required -- see ``primat.gui.app._build_preview``, which
            already does this for the Reactions tab).
        nucl: primat.network_data.UpdateNuclearRates. Same ``cfg``/
            ``custom_network``, also solve-free.
    """

    def __init__(self, results, cfg, nucl):
        self.results = results
        self.evolution = results["evolution"]
        self.cfg = cfg
        self.nucl = nucl
        self.A = {name: NZ[0] + NZ[1] for name, NZ in cfg.Nuclides.items()}
        self.Z = {name: NZ[1] for name, NZ in cfg.Nuclides.items()}
        self.N = {name: NZ[0] for name, NZ in cfg.Nuclides.items()}
        self._Y_interp = {}   # name -> cached Y(t) interpolator
        self._T_of_t = None   # cached T_gamma(t) interpolator

    @property
    def abundance_names(self):
        """Tracked nuclide names, in the order ``evolution.Y`` carries them
        (n/p first, per ``primat.evolution``'s module docstring)."""
        return list(self.evolution.Y.keys())

    def primat_results(self):
        return self.results

    def get_quantity(self, quantity):
        """Same contract as ``primat.main.PRIMAT.get_quantity``: a result-dict
        key (``'YPBBN'``, ``'DoH'``, ...) or a nuclide name for its final Y.

        ``results["Y_final"]`` is a bonus the C backend's wrapper adds (see
        ``primat/_primat_c/_wrapper.c``); the Python backend's own
        ``PRIMAT.solve()`` result dict never has it (confirmed by
        ``tests/test_backend_parity.py::test_backend_result_dict_shape_matches``),
        since Python keeps per-nuclide finals on the live ``nuclear.Y_final``
        attribute instead. Falling back to ``self.evolution.Y[name][-1]``
        (the last sampled time step, always present since ``GuiRun`` requires
        ``output_time_evolution=True``) keeps this working under either
        backend -- see the two evol_Y/Y_final agreement checks in
        ``primat-c/tests/unit/test_api.c``.
        """
        if quantity in self.results:
            return self.results[quantity]
        Y_final = self.results.get("Y_final")
        if Y_final is not None and quantity in Y_final:
            return Y_final[quantity]
        if quantity in self.evolution.Y:
            return self.evolution.Y[quantity][-1]
        raise ValueError(
            f"Unknown quantity '{quantity}'. "
            f"Valid result keys: {list(self.results.keys())}. "
            f"Valid nuclide names: {list(self.evolution.Y.keys())}."
        )

    def __getitem__(self, name):
        """``Y(t)`` callable for nuclide ``name`` (mirrors ``PRIMAT.__getitem__``),
        built lazily from the discrete ``evolution`` arrays and cached."""
        if name not in self._Y_interp:
            if name not in self.evolution.Y:
                raise KeyError(
                    f"Unknown species '{name}'. Available: {self.abundance_names}"
                )
            self._Y_interp[name] = Y_interpolator(self.evolution, name)
        return self._Y_interp[name]

    @property
    def T_of_t(self):
        """``T_gamma(t)`` [MeV] callable (mirrors ``PRIMAT.T_of_t``), built
        lazily from the discrete ``evolution`` arrays and cached."""
        if self._T_of_t is None:
            self._T_of_t = T_gamma_interpolator(self.evolution)
        return self._T_of_t
