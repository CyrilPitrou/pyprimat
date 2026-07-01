# -*- coding: utf-8 -*-
"""
primat.gui.launcher
======================

Console-script entry point for the ``primat-gui`` command.

Streamlit's runner (``streamlit run ...``) needs a *path* to a script file,
not an importable module, so this launcher resolves the on-disk location of
``primat/gui/app.py`` -- which works identically whether primat was
installed into site-packages via ``pip install ".[gui]"`` or is being run
from a source checkout -- and hands it to Streamlit's CLI exactly as if the
user had typed ``streamlit run <path>``.
"""
import argparse
import importlib.resources
import os
import sys


def main(argv=None):
    """Launch the primat Streamlit GUI.

    Parameters
    ----------
    argv : list of str, optional
        Extra arguments forwarded to ``streamlit run`` (e.g. ``--server.port
        8502``), plus this launcher's own ``--backend`` flag. Defaults to
        ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code from the Streamlit process.

    Notes
    -----
    ``streamlit`` and ``plotly`` are optional dependencies (the ``gui``
    extra): if they are not installed, this prints an actionable message
    instead of an ``ImportError`` traceback.

    ``--backend {auto,c,python}`` (default ``auto``, same default as
    :func:`primat.backend.run_bbn`) selects which BBN backend every solve
    triggered from this GUI process uses. It is consumed here rather than
    forwarded to Streamlit, and passed down to ``primat/gui/app.py`` via the
    ``PRIMAT_GUI_BACKEND`` environment variable (a plain CLI flag cannot
    reach ``app.py`` since Streamlit re-execs it as a script on every rerun
    without our ``argv``). This is what lets ``primat-gui --backend python``
    exercise the pure-Python backend for testing/development even though the
    GUI otherwise defaults to the faster C backend (``CLAUDE.md``: "primat
    is ... a fast C engine ... the default").

    >>> main(["--backend", "python"])  # doctest: +SKIP
    """
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.exit(
            "The primat GUI requires the optional 'gui' extra.\n"
            "Install it with:\n\n"
            '    pip install "primat[gui]"\n'
        )

    if argv is None:
        argv = sys.argv[1:]

    # Peel off our own --backend flag before handing the rest of argv to
    # Streamlit's CLI, which would otherwise reject an unrecognised option.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=["auto", "c", "python"], default="auto")
    known, streamlit_argv = parser.parse_known_args(argv)
    os.environ["PRIMAT_GUI_BACKEND"] = known.backend

    # Resolve the installed location of app.py (works from site-packages and
    # from a source checkout alike).
    app_path = importlib.resources.files("primat.gui") / "app.py"

    # streamlit.web.cli.main() is a click command that reads sys.argv, so we
    # rewrite argv to look like a direct `streamlit run <app_path> ...` call.
    sys.argv = ["streamlit", "run", str(app_path), *streamlit_argv]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
