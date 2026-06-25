# -*- coding: utf-8 -*-
"""
setup.py -- builds the optional primat._primat_c extension (PRIMAT.md S5.2).

setuptools reads project metadata from pyproject.toml; this file exists
only to declare the compiled extension, since pyproject.toml's
[project]/[tool.setuptools] tables have no equivalent for Extension
objects. The extension links primat-c's sources directly (excluding
main.c/cli.c, the standalone-binary entry points, which are irrelevant to
the Python bridge and main.c's int main() would collide with Python's own).

If a C compiler is unavailable (or compilation fails for any other
reason), pip install still succeeds: setuptools' build_ext step is
declared optional via cmdclass below, and primat.backend.HAS_C_BACKEND
falls back to the pure-Python implementation at import time (see
primat/backend.py).
"""
import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

CPRIMAT_SRC = os.path.join("primat-c", "src")
CPRIMAT_INCLUDE = os.path.join("primat-c", "include")

# Every primat-c source file except the standalone CLI binary's entry
# point (main.c, defines int main()) and its argv/--set parsing (cli.c,
# unused by the Python bridge, which goes through cpr_config_set_by_name
# directly with values already parsed on the Python side).
# setuptools requires sources to be relative to this file (not absolute),
# hence the os.path.join with the (relative) CPRIMAT_SRC above rather than
# an absolute repo-root path.
_EXCLUDED = {"main.c", "cli.c"}
_cprimat_sources = sorted(
    os.path.join(CPRIMAT_SRC, f)
    for f in os.listdir(CPRIMAT_SRC)
    if f.endswith(".c") and f not in _EXCLUDED
)

ext_modules = [
    Extension(
        "primat._primat_c",
        sources=["primat/_primat_c/_wrapper.c"] + _cprimat_sources,
        include_dirs=[CPRIMAT_INCLUDE],
        # _GNU_SOURCE: exposes M_PI on Linux (glibc) with -std=c11.
        # _USE_MATH_DEFINES: exposes M_PI on Windows (MSVC).
        define_macros=[("_GNU_SOURCE", None), ("_USE_MATH_DEFINES", None)],
        # No -march=native: wheels must run on any host of the target
        # platform/arch, not just the build machine's own CPU.
        # MSVC does not accept -std=c11 or -O2 (it uses /std:c11 and /O2).
        extra_compile_args=["-std=c11", "-O2"] if os.name != "nt" else ["/std:c11", "/O2"],
        libraries=["m"] if os.name != "nt" else [],
    )
]


class optional_build_ext(build_ext):
    """Let the C extension fail to build without failing the install.

    A missing/broken C compiler should degrade to the pure-Python backend
    (primat.backend.HAS_C_BACKEND probes for the compiled module at import
    time), not break `pip install primat` altogether.
    """

    def run(self):
        try:
            super().run()
        except Exception as exc:  # noqa: BLE001 -- intentionally broad: any
            # build_ext failure (missing compiler, primat-c source error,
            # platform-specific compile error, ...) should degrade
            # gracefully rather than abort the whole `pip install`.
            print(f"WARNING: primat._primat_c C extension build failed "
                  f"({exc}); falling back to the pure-Python backend.")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # noqa: BLE001 -- see run()'s docstring
            print(f"WARNING: could not build extension {ext.name} ({exc}); "
                  f"falling back to the pure-Python backend.")


#setup(
#    ext_modules=ext_modules,
#)
           
setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": optional_build_ext},
)
