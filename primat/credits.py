"""Shared credits text for the GUI and command-line entry points.

The project exposes the same attribution copy in several places:

- the Streamlit GUI shows the full text, including installation and usage
  hints for people arriving from the browser;
- the Python and C command-line tools print a shorter variant without those
  usage hints, because the terminal user already has the executable in hand.

Keeping the text in one module avoids the small but annoying drift that tends
to happen when the same paragraph is duplicated across a GUI footer, a CLI
flag, and a dialog box.
"""

_CREDITS_CORE = (
    "primat is developed by Cyril Pitrou (https://www2.iap.fr/users/pitrou/) "
    "with features related to neutrino physics written by Julien Froustey.\n\n"
    "The story started in the 1980s with BBN codes written by Elisabeth "
    "Vangioni and Alain Coc which eventually lead to 'ezbbn', a large "
    "nuclear network FORTRAN code whose nuclear rates tables were maintained "
    "by Alain Coc.\n"
    "PRIMAT, initially a Mathematica code, was based on "
    "'ezbbn' with improved neutrino physics. It is now translated into a "
    "python code, but it also relies on a C backend to improve its "
    "performance."
)

_CREDITS_USAGE = (
    "\n\nYou can install it in a terminal via 'pip install primat' and learn "
    "how to run it with 'primat --help'.\n\n"
    "For notebooks, examples and documentation, download the source code "
    "(https://github.com/CyrilPitrou/primat).\n\n"
    "Please cite the publication (https://arxiv.org/abs/1801.08023) if you "
    "use it."
)

_CREDITS_CLI_SUFFIX = (
    "\n\nFor notebooks, examples and documentation, download the source code "
    "(https://github.com/CyrilPitrou/primat).\n"
    "Please cite the publication (https://arxiv.org/abs/1801.08023) if you "
    "use it."
)


def gui_credits_text():
    """Return the full credits text shown by the Streamlit GUI.

    The GUI includes the installation and ``--help`` hint because it is the
    main entry point for users browsing the project in a browser and may be
    their first exposure to the package.

    Returns
    -------
    str
        Multi-paragraph Markdown-friendly credits text.

    Example
    -------
        >>> "pip install primat" in gui_credits_text()
        True
    """
    return _CREDITS_CORE + _CREDITS_USAGE


def cli_credits_text():
    """Return the shorter credits text printed by the CLI ``--credits`` flag.

    The terminal variant omits the installation / usage sentence because that
    information is redundant once the user already invoked the executable.

    Returns
    -------
    str
        Multi-paragraph credits text without the install-and-run guidance.

    Example
    -------
        >>> "pip install primat" in cli_credits_text()
        False
    """
    return _CREDITS_CORE + _CREDITS_CLI_SUFFIX
