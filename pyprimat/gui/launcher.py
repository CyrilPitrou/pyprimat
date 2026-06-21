# -*- coding: utf-8 -*-
"""
pyprimat.gui.launcher
======================

Console-script entry point for the ``pyprimat-gui`` command.

Streamlit's runner (``streamlit run ...``) needs a *path* to a script file,
not an importable module, so this launcher resolves the on-disk location of
``pyprimat/gui/app.py`` -- which works identically whether PyPRIMAT was
installed into site-packages via ``pip install ".[gui]"`` or is being run
from a source checkout -- and hands it to Streamlit's CLI exactly as if the
user had typed ``streamlit run <path>``.
"""
import importlib.resources
import sys


def main(argv=None):
    """Launch the PyPRIMAT Streamlit GUI.

    Parameters
    ----------
    argv : list of str, optional
        Extra arguments forwarded to ``streamlit run`` (e.g. ``--server.port
        8502``). Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code from the Streamlit process.

    Notes
    -----
    ``streamlit`` and ``plotly`` are optional dependencies (the ``gui``
    extra): if they are not installed, this prints an actionable message
    instead of an ``ImportError`` traceback.
    """
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.exit(
            "The PyPRIMAT GUI requires the optional 'gui' extra.\n"
            "Install it with:\n\n"
            '    pip install "PyPRIMAT[gui]"\n'
        )

    if argv is None:
        argv = sys.argv[1:]

    # Resolve the installed location of app.py (works from site-packages and
    # from a source checkout alike).
    app_path = importlib.resources.files("pyprimat.gui") / "app.py"

    # streamlit.web.cli.main() is a click command that reads sys.argv, so we
    # rewrite argv to look like a direct `streamlit run <app_path> ...` call.
    sys.argv = ["streamlit", "run", str(app_path), *argv]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
