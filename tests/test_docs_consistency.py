"""Guard against README/CLAUDE.md documentation staling relative to the code.

Both docs quote specific PyPRConfig defaults and specific
runfiles/PyPRIMAT_reference_run.py parameter names/values (CLAUDE.md's
"Validation before committing" section says references were produced with
particular settings). Neither file is machine-checked by anything else, so a
config refactor can silently leave them wrong (this happened: CLAUDE.md used
to cite a `n_temperature_table`/`sampling_nTOp` that no longer exist). These
tests assert the quoted facts still hold, so a future config change that
breaks them fails a test instead of just leaving stale prose.
"""
import ast
import os

import pytest

from pyprimat.config import PyPRConfig

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_save_nTOp_defaults_match_readme():
    """README's n<->p weak-rate section states save_nTOp/save_nTOp_thermal default True."""
    cfg = PyPRConfig()
    assert cfg.save_nTOp is True
    assert cfg.save_nTOp_thermal is True


def _reference_run_options():
    """Parse MyOptions out of PyPRIMAT_reference_run.py without running it.

    The script performs an expensive multi-minute solve as a side effect of
    import, so we extract the literal dict via the AST instead of importing
    the module.
    """
    path = os.path.join(REPO_ROOT, "runfiles", "PyPRIMAT_reference_run.py")
    tree = ast.parse(open(path).read(), filename=path)
    # MyOptions references module-level names (e.g. "Omegabh2": omegabh2), so
    # literal_eval alone can't resolve it; evaluate against a namespace built
    # from this module's own simple top-level literal assignments instead of
    # importing the module (which would trigger its expensive solve()).
    namespace = {}
    my_options_node = None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        if node.targets[0].id == 'MyOptions':
            my_options_node = node.value
            continue
        try:
            namespace[node.targets[0].id] = ast.literal_eval(node.value)
        except ValueError:
            pass

    if my_options_node is None:
        raise AssertionError("MyOptions dict not found in PyPRIMAT_reference_run.py")
    code = compile(ast.Expression(body=my_options_node), filename=path, mode='eval')
    return eval(code, {}, namespace)


@pytest.mark.parametrize("key,expected", [
    ("sampling_temperature_per_decade", 2000),
    ("numerical_precision", 1e-10),
    ("sampling_nTOp_per_decade", 125),
    ("T_start_cosmo_MeV", 100.0),
])
def test_reference_run_params_match_claude_md(key, expected):
    """The param names/values CLAUDE.md quotes for the reference run must exist verbatim."""
    options = _reference_run_options()
    assert key in options, f"{key!r} no longer in PyPRIMAT_reference_run.py's MyOptions"
    assert options[key] == expected


def test_reference_run_params_are_known_to_config():
    """Every MyOptions key must be a real PyPRConfig field (catches silent typos)."""
    options = _reference_run_options()
    with _no_warning_context():
        PyPRConfig(options)


class _no_warning_context:
    """Fail the test if PyPRConfig(options) emits an 'unknown parameter' warning."""

    def __enter__(self):
        import warnings
        self._cw = warnings.catch_warnings(record=True)
        self._records = self._cw.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        self._cw.__exit__(*exc)
        unknown = [r for r in self._records if "unknown parameter" in str(r.message)]
        assert not unknown, f"PyPRConfig reported unknown keys: {unknown}"
