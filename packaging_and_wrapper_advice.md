# Packaging and Architecture Advice: Hybrid C/Python BBN Suite (`primat`)

To make the codebase easy to distribute, install, and use, we need a packaging strategy that accommodates:
1. **Easy installation** (`pip install primat`) with a fast C backend.
2. **Robustness**: Falling back to the pure-Python implementation if compilation fails or a compiler is absent.
3. **A unified CLI**: Letting users easily run the Python implementation, the C implementation, or the GUI.
4. **Standalone C usage**: Allowing C users to compile and run `CPRIMAT` independently of Python.

Below is the recommended strategy and step-by-step design.

---

## 1. Recommended Directory Layout

Keep the Python package (`pyprimat`) and C package (`CPRIMAT`) side-by-side inside the repository. This structure keeps standalone C compilation clean while allowing Python packaging tools to find and compile the C code:

```
PRIMAT_suite/
├── pyproject.toml           # Modern package configuration (metadata, entry points)
├── setup.py                 # Defines the C extension module compilation
├── README.md                # Top-level installation instructions
├── pyprimat/                # The Python package
│   ├── __init__.py          # Package initialization (exports the high-level API)
│   ├── main.py              # Contains PyPR class
│   ├── backend.py           # Handles import of C extension and Python fallback
│   ├── cli.py               # Python/C CLI entry points
│   ├── cprimat_wrapper.c    # Python C-API wrapper code (bridges Python <-> C)
│   └── gui/                 # Streamlit interface
└── CPRIMAT/                 # Pure C code repository
    ├── Makefile             # Standalone C compilation
    ├── include/             # C header files
    ├── src/                 # C source files
    └── tests/               # C unit tests
```

---

## 2. The C-Python Extension Wrapper (`pyprimat/cprimat_wrapper.c`)

Python and C cannot directly communicate. Python works with high-level Python Objects (dictionaries, lists, strings), whereas C expects binary structs and pointers.

The wrapper `pyprimat/cprimat_wrapper.c` serves as a translator. It uses Python's official C-API (`Python.h`) to unpack Python variables, call the C engine (`cprimat_run`), and wrap the output back into Python-friendly types.

Because the C codebase provides `cpr_config_set_by_name(...)` to dynamically set configuration variables, the wrapper can dynamically loop over any configuration dictionary passed from Python without hardcoding each parameter name.

Here is the complete implementation of the wrapper:

```c
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "cprimat/api.h"
#include "cprimat/config.h"

/* 
 * This C function executes when you run `_cprimat.run(config_dict, data_dir)` in Python.
 * It parses the Python dict, constructs the C CPRConfig, runs BBN, and returns a Python dict of results.
 */
static PyObject* py_cprimat_run(PyObject* self, PyObject* args) {
    PyObject* py_cfg_dict;
    const char* data_dir;

    // 1. Unpack arguments passed from Python: expected one dictionary and one string
    if (!PyArg_ParseTuple(args, "O!s", &PyDict_Type, &py_cfg_dict, &data_dir)) {
        return NULL;
    }

    CPRConfig cfg;
    char* errmsg = NULL;

    // 2. Initialize the C configuration struct with default parameters
    if (cpr_config_init_defaults(&cfg, data_dir, &errmsg) != 0) {
        PyErr_SetString(PyExc_RuntimeError, errmsg ? errmsg : "Failed to initialize C defaults");
        free(errmsg);
        return NULL;
    }

    // 3. Loop through keys/values in the Python dictionary to override default settings
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(py_cfg_dict, &pos, &key, &value)) {
        const char* key_str = PyUnicode_AsUTF8(key);
        if (!key_str) continue;

        CPRParam param;
        
        // Map Python types to the C parameter union
        if (PyBool_Check(value)) {
            param.type = CPR_BOOL;
            param.v.b = (value == Py_True) ? 1 : 0;
        } else if (PyLong_Check(value)) {
            param.type = CPR_INT;
            param.v.i = PyLong_AsLong(value);
        } else if (PyFloat_Check(value)) {
            param.type = CPR_DOUBLE;
            param.v.d = PyFloat_AsDouble(value);
        } else if (PyUnicode_Check(value)) {
            param.type = CPR_STRING;
            param.v.s = PyUnicode_AsUTF8(value);
        } else {
            // Ignore unsupported config types
            continue; 
        }

        // Apply config option dynamically by name
        if (cpr_config_set_by_name(&cfg, key_str, param, &errmsg) != 0) {
            PyErr_Format(PyExc_ValueError, "Failed setting parameter '%s': %s", key_str, errmsg ? errmsg : "unknown error");
            free(errmsg);
            cpr_config_free(&cfg);
            return NULL;
        }
    }

    // Validate the resulting config flags/invariants
    if (cpr_config_validate(&cfg, &errmsg) != 0) {
        PyErr_SetString(PyExc_ValueError, errmsg ? errmsg : "Config validation failed");
        free(errmsg);
        cpr_config_free(&cfg);
        return NULL;
    }

    // 4. Run the BBN simulation
    CPRResults results;
    if (cprimat_run(&cfg, &results, &errmsg) != 0) {
        PyErr_SetString(PyExc_RuntimeError, errmsg ? errmsg : "C integration run failed");
        free(errmsg);
        cpr_config_free(&cfg);
        return NULL;
    }

    // 5. Pack C results back into a standard Python dictionary
    PyObject* py_results = PyDict_New();
    PyDict_SetItemString(py_results, "YPBBN", PyFloat_FromDouble(results.YPBBN));
    PyDict_SetItemString(py_results, "YPCMB", PyFloat_FromDouble(results.YPCMB));
    PyDict_SetItemString(py_results, "DoH", PyFloat_FromDouble(results.DoH));
    PyDict_SetItemString(py_results, "He3oH", PyFloat_FromDouble(results.He3oH));
    PyDict_SetItemString(py_results, "He3oHe4", PyFloat_FromDouble(results.He3oHe4));
    PyDict_SetItemString(py_results, "Li7oH", PyFloat_FromDouble(results.Li7oH));

    if (results.has_Li6oLi7) {
        PyDict_SetItemString(py_results, "Li6oLi7", PyFloat_FromDouble(results.Li6oLi7));
    }
    if (results.has_YCNO) {
        PyDict_SetItemString(py_results, "YCNO", PyFloat_FromDouble(results.YCNO));
    }
    if (results.has_Neff) {
        PyDict_SetItemString(py_results, "Neff", PyFloat_FromDouble(results.Neff));
    }

    // Clean up C resources
    cprimat_results_free(&results);
    cpr_config_free(&cfg);

    return py_results;
}

// Module registration tables
static PyMethodDef ModuleMethods[] = {
    {"run", py_cprimat_run, METH_VARARGS, "Runs C BBN code"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cprimatmodule = {
    PyModuleDef_HEAD_INIT,
    "_cprimat",
    "C backend wrapper for PRIMAT computations",
    -1,
    ModuleMethods
};

// Module entry point (called upon import _cprimat)
PyMODINIT_FUNC PyInit__cprimat(void) {
    return PyModule_Create(&cprimatmodule);
}
```

---

## 3. Python Packaging Setup (`setup.py` & `pyproject.toml`)

To make `pip install primat` compile the C wrapper and the main C code into the binary library `_cprimat`, we define a Python extension target.

### A. Add `setup.py` at the root
Create `setup.py` to target compiling both files:

```python
from setuptools import setup, Extension, find_packages
import os

# Define the sources for the C extension module.
# We exclude 'main.c' and 'cli.c' because they contain main() entry points
# for the standalone C binary, which conflict with the Python extension.
c_source_dir = os.path.join("CPRIMAT", "src")
c_sources = [
    os.path.join(c_source_dir, f)
    for f in os.listdir(c_source_dir)
    if f.endswith(".c") and f not in ("main.c", "cli.c")
]

# Add the wrapper code
c_sources.append("pyprimat/cprimat_wrapper.c")

cprimat_extension = Extension(
    "pyprimat._cprimat",
    sources=c_sources,
    include_dirs=["CPRIMAT/include"],
    extra_compile_args=["-std=c11", "-O2"],
)

setup(
    packages=find_packages(),
    ext_modules=[cprimat_extension],
)
```

### B. Update `pyproject.toml`
Ensure `setuptools` builds the package and defines high-level package information:

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "primat"
version = "0.2.0"
description = "Precise Big Bang Nucleosynthesis computations in Python and C"
readme = "README.md"
license = { file = "LICENCE" }
authors = [
    { name = "Cyril Pitrou", email = "pitrou@iap.fr" },
    { name = "Julien Froustey" },
]
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "scipy",
    "joblib",
    "plotly",
]

[project.scripts]
primat = "pyprimat.cli:main"           # Auto-selects fast C backend, falls back to Python
primat-python = "pyprimat.cli:main_py" # Explicitly runs Python backend
primat-c = "pyprimat.cli:main_c"       # Explicitly runs C backend
primat-gui = "pyprimat.gui.launcher:main"

[project.optional-dependencies]
gui = ["streamlit", "pandas"]
notebooks = ["matplotlib", "pandas", "papermill"]
```

---

## 4. The Python Wrapper & Fallback Mechanism (`pyprimat/backend.py`)

A fallback mechanism protects users who do not have compilers by importing the C extension inside a `try-except` block. If it fails, Python falls back to the pure Python implementation seamlessly.

Create `pyprimat/backend.py`:

```python
"""
pyprimat/backend.py
Handles backend routing between C Extension and Pure Python.
"""
import logging
import os

logger = logging.getLogger("primat")

# Try importing the compiled C extension
HAS_C_BACKEND = False
try:
    from . import _cprimat
    HAS_C_BACKEND = True
except ImportError:
    logger.warning(
        "C extension (_cprimat) is not compiled or failed to import. "
        "Falling back to pure Python implementation (which will be slower)."
    )

def run_bbn(config_dict, force_backend=None):
    """
    Solves BBN using either the fast C backend or pure Python.
    
    force_backend: 'C' or 'python' to bypass automatic routing.
    """
    # Locating default data folder
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "rates"))
    
    use_c = (force_backend == 'C') or (force_backend is None and HAS_C_BACKEND)
    
    if use_c:
        if not HAS_C_BACKEND:
            raise RuntimeError("C backend requested but not compiled/available.")
        # Call the C extension
        return _cprimat.run(config_dict, data_dir)
    else:
        # Fallback to the Python implementation
        from .main import PyPR
        pr = PyPR(config_dict)
        return pr.solve()
```

---

## 5. Structuring the CLIs (`pyprimat/cli.py`)

You can satisfy both C-centric and Python-centric users via three CLI entry points, routing through `pyprimat.cli`:

```python
"""
pyprimat/cli.py
Command Line Interface dispatcher.
"""
import sys
import argparse
from .backend import run_bbn, HAS_C_BACKEND

def main():
    """Default entry point ('primat'). Automatically selects C if available."""
    run(force_backend=None)

def main_py():
    """Explicitly run the Python engine ('primat-python')."""
    run(force_backend='python')

def main_c():
    """Explicitly run the C engine ('primat-c')."""
    run(force_backend='C')

def run(force_backend):
    parser = argparse.ArgumentParser(description="PRIMAT BBN Calculator CLI")
    parser.add_argument("config_file", help="Path to config file (.ini or .json)")
    # Add other common flags...
    args = parser.parse_args()

    # Read config file into a dictionary
    config_dict = parse_config(args.config_file) 

    try:
        results = run_bbn(config_dict, force_backend=force_backend)
        print_results(results)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
```

---

## 6. Good Practices for Standalone C Users

If a user wants to run the **C version entirely on its own** without Python:

1. **Independent C Build Pipeline**: Keep the C build pipeline completely detached from Python dependencies. The current `CPRIMAT/Makefile` which compiles `build/cprimat` using only standard library C and standard math libraries is **perfect** for this.
2. **Compile Command**: Standalone compilation remains simple:
   ```bash
   cd CPRIMAT
   make
   ```
3. **C CLI Usage**: The user runs the compiled C CLI directly:
   ```bash
   ./build/cprimat examples/run_small.ini
   ```
4. **Providing a CMake Alternative (Recommended)**:
   While a `Makefile` is excellent for Linux/macOS, adding a `CMakeLists.txt` in `CPRIMAT` is best practice for cross-platform compatibility (especially for Windows users using MSVC). It also makes it easy to integrate with C++ packages later or build shared/static libraries cleanly.

---

## 7. Next Step: Pre-building Binary Wheels (cibuildwheel)

If you compile C extensions dynamically on user machines via `pip`, the user must have `gcc`/`clang` installed, which might fail on some platforms. 

To eliminate compiling on installation:
- Use **`cibuildwheel`** (a tool run in GitHub Actions).
- It compiles `_cprimat` automatically for macOS (x86 & M-series), Windows, and Linux.
- It produces pre-compiled wheel files (`.whl`) and uploads them to PyPI.
- When a user runs `pip install primat`, they instantly download the pre-compiled binary wheel corresponding to their system architecture. Compilation is skipped entirely, and they get the fast C version instantly.
