/* _wrapper.c -- CPython bridge to primat-c's cprimat_run (PRIMAT.md S5.1).
 *
 * Exposes a single function, run_bbn(params, data_dir) -> dict, that:
 *   1. builds a CPRConfig with cpr_config_init_defaults(data_dir),
 *   2. applies every (key, value) in `params` via cpr_config_set_by_name
 *      (the same generic by-name setter the C CLI/ini parser uses, so this
 *      wrapper does not need a field-by-field mapping table),
 *   3. validates with cpr_config_validate,
 *   4. runs cprimat_run, and
 *   5. converts the resulting CPRResults into a plain Python dict with
 *      exactly the same key set PRIMAT.solve() returns (see
 *      primat/main.py's solve(); kept in sync per CLAUDE.md's backend-parity
 *      mandate -- see tests/test_backend_parity.py).
 *
 * `data_dir` is the directory containing rates/ (cfg.data_dir on the Python
 * side); the rates_dir/user_rates_dir Python-side overlay (config.py's
 * resolve_rates_path) has no C-side equivalent yet, so only the shipped
 * rates/ tree is reachable through this bridge for now.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cprimat/api.h"
#include "cprimat/config.h"

#include <stdlib.h>
#include <string.h>

/* Converts one Python value into a CPRParam. CPR_STRING points at a
 * strdup'd copy (returned via *owned, so the caller can free it after the
 * cpr_config_set_by_name call -- config.c's F_STRING case strdup's its own
 * copy, so the temporary here does not need to outlive that call). Returns
 * 0 on success; 1 (with a Python exception already set) for an
 * unsupported value type. */
static int py_to_cprparam(PyObject *value, CPRParam *out, char **owned)
{
    *owned = NULL;
    if (value == Py_None) {
        out->type = CPR_NONE;
        return 0;
    }
    if (PyBool_Check(value)) {
        out->type = CPR_BOOL;
        out->v.b = (value == Py_True) ? 1 : 0;
        return 0;
    }
    if (PyLong_Check(value)) {
        out->type = CPR_INT;
        out->v.i = PyLong_AsLong(value);
        return 0;
    }
    if (PyFloat_Check(value)) {
        out->type = CPR_DOUBLE;
        out->v.d = PyFloat_AsDouble(value);
        return 0;
    }
    if (PyUnicode_Check(value)) {
        const char *s = PyUnicode_AsUTF8(value);
        if (!s)
            return 1;
        *owned = strdup(s);
        out->type = CPR_STRING;
        out->v.s = *owned;
        return 0;
    }
    PyErr_Format(PyExc_TypeError,
                 "unsupported parameter value type %s (expected bool/int/"
                 "float/str/None)", Py_TYPE(value)->tp_name);
    return 1;
}

/* Builds a Python list of floats from a caller-owned double array of
 * length n. Returns NULL (with a Python exception set) on allocation
 * failure. */
static PyObject *doubles_to_list(const double *arr, size_t n)
{
    PyObject *list = PyList_New((Py_ssize_t)n);
    if (!list)
        return NULL;
    for (size_t i = 0; i < n; i++) {
        PyObject *o = PyFloat_FromDouble(arr[i]);
        if (!o) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, (Py_ssize_t)i, o);
    }
    return list;
}

/* Builds the "evolution" sub-dict (PRIMAT.md S7.2/S7.3): plain Python
 * lists (not numpy arrays -- this extension carries no numpy C-API
 * dependency), converted to an EvolutionResult Python-side by
 * primat/backend.py via np.asarray. "Y" is itself a sub-dict keyed by
 * nuclide name, column-sliced out of r->evol_Y's row-major layout, mirroring
 * primat.evolution.EvolutionResult.Y. Returns NULL (with a Python
 * exception set) on failure. */
static PyObject *evolution_to_dict(const CPRResults *r)
{
    PyObject *eo = PyDict_New();
    if (!eo)
        return NULL;

#define SETLIST(key, arr) \
    do { \
        PyObject *o = doubles_to_list((arr), r->n_evolution); \
        if (!o || PyDict_SetItemString(eo, key, o) < 0) { Py_XDECREF(o); Py_DECREF(eo); return NULL; } \
        Py_DECREF(o); \
    } while (0)

    SETLIST("t", r->evol_t);
    SETLIST("a", r->evol_a);
    SETLIST("T_gamma", r->evol_T_gamma);
    SETLIST("T_nue", r->evol_Tnue);
    SETLIST("T_numu", r->evol_Tnumu);
    SETLIST("T_nutau", r->evol_Tnutau);
#undef SETLIST

    PyObject *Y = PyDict_New();
    if (!Y) { Py_DECREF(eo); return NULL; }
    double *col = malloc(r->n_evolution * sizeof(double));
    if (!col) { Py_DECREF(Y); Py_DECREF(eo); PyErr_NoMemory(); return NULL; }
    for (size_t s = 0; s < r->n_nuclides; s++) {
        for (size_t i = 0; i < r->n_evolution; i++)
            col[i] = r->evol_Y[i * r->n_nuclides + s];
        PyObject *o = doubles_to_list(col, r->n_evolution);
        if (!o || PyDict_SetItemString(Y, r->nuclide_names[s], o) < 0) {
            Py_XDECREF(o); free(col); Py_DECREF(Y); Py_DECREF(eo); return NULL;
        }
        Py_DECREF(o);
    }
    free(col);
    if (PyDict_SetItemString(eo, "Y", Y) < 0) { Py_DECREF(Y); Py_DECREF(eo); return NULL; }
    Py_DECREF(Y);

    return eo;
}

/* Builds a Python dict mirroring PRIMAT.solve()'s result dict (main.py),
 * plus a "Y_final" sub-dict of every tracked nuclide's final mass
 * fraction (mirrors NuclearNetwork.Y_final, used by get_quantity's
 * nuclide-name fallback), and an "evolution" sub-dict (PRIMAT.md S7.3)
 * when cfg.output_time_evolution requested it. */
static PyObject *results_to_dict(const CPRResults *r)
{
    PyObject *d = PyDict_New();
    if (!d)
        return NULL;

#define SET(key, val) \
    do { \
        PyObject *o = PyFloat_FromDouble(val); \
        if (!o || PyDict_SetItemString(d, key, o) < 0) { Py_XDECREF(o); Py_DECREF(d); return NULL; } \
        Py_DECREF(o); \
    } while (0)

    SET("YPCMB", r->YPCMB);
    SET("YPBBN", r->YPBBN);
    SET("DoH", r->DoH);
    SET("He3oH", r->He3oH);
    SET("He3oHe4", r->He3oHe4);
    SET("Li7oH", r->Li7oH);
    if (r->has_Li6oLi7) SET("Li6oLi7", r->Li6oLi7);
    if (r->has_YCNO) SET("YCNO", r->YCNO);
    if (r->has_Neff) SET("Neff", r->Neff);
    if (r->has_Omeganurel) SET("Omeganurel", r->Omeganurel);
    if (r->has_OneOverOmeganunr) SET("OneOverOmeganunr", r->OneOverOmeganunr);
#undef SET

    PyObject *yfinal = PyDict_New();
    if (!yfinal) { Py_DECREF(d); return NULL; }
    for (size_t i = 0; i < r->n_nuclides; i++) {
        PyObject *val = PyFloat_FromDouble(r->Y_final[i]);
        if (!val || PyDict_SetItemString(yfinal, r->nuclide_names[i], val) < 0) {
            Py_XDECREF(val);
            Py_DECREF(yfinal);
            Py_DECREF(d);
            return NULL;
        }
        Py_DECREF(val);
    }
    if (PyDict_SetItemString(d, "Y_final", yfinal) < 0) {
        Py_DECREF(yfinal);
        Py_DECREF(d);
        return NULL;
    }
    Py_DECREF(yfinal);

    if (r->has_evolution) {
        PyObject *eo = evolution_to_dict(r);
        if (!eo || PyDict_SetItemString(d, "evolution", eo) < 0) {
            Py_XDECREF(eo);
            Py_DECREF(d);
            return NULL;
        }
        Py_DECREF(eo);
    }

    return d;
}

static PyObject *primat_c_run_bbn(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *params;
    const char *data_dir;

    if (!PyArg_ParseTuple(args, "Os", &params, &data_dir))
        return NULL;
    if (!PyDict_Check(params)) {
        PyErr_SetString(PyExc_TypeError, "params must be a dict");
        return NULL;
    }

    CPRConfig cfg;
    char *errmsg = NULL;
    if (cpr_config_init_defaults(&cfg, data_dir, &errmsg)) {
        PyErr_Format(PyExc_RuntimeError, "cpr_config_init_defaults failed: %s",
                     errmsg ? errmsg : "(no message)");
        free(errmsg);
        return NULL;
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    int failed = 0;
    while (PyDict_Next(params, &pos, &key, &value)) {
        if (!PyUnicode_Check(key)) {
            PyErr_SetString(PyExc_TypeError, "params keys must be str");
            failed = 1;
            break;
        }
        const char *name = PyUnicode_AsUTF8(key);
        if (!name) { failed = 1; break; }

        char *owned = NULL;
        CPRParam p;
        if (py_to_cprparam(value, &p, &owned)) {
            failed = 1;
            break;
        }
        char *set_err = NULL;
        int rc = cpr_config_set_by_name(&cfg, name, p, &set_err);
        free(owned);
        if (rc) {
            PyErr_Format(PyExc_ValueError, "%s", set_err ? set_err : "cpr_config_set_by_name failed");
            free(set_err);
            failed = 1;
            break;
        }
    }
    if (failed) {
        cpr_config_free(&cfg);
        return NULL;
    }

    if (cpr_config_validate(&cfg, &errmsg)) {
        PyErr_Format(PyExc_ValueError, "%s", errmsg ? errmsg : "cpr_config_validate failed");
        free(errmsg);
        cpr_config_free(&cfg);
        return NULL;
    }

    CPRResults results;
    int rc = cprimat_run(&cfg, &results, &errmsg);
    cpr_config_free(&cfg);
    if (rc) {
        PyErr_Format(PyExc_RuntimeError, "cprimat_run failed: %s",
                     errmsg ? errmsg : "(no message)");
        free(errmsg);
        return NULL;
    }

    PyObject *d = results_to_dict(&results);
    cprimat_results_free(&results);
    return d;
}

static PyMethodDef primat_c_methods[] = {
    {"run_bbn", primat_c_run_bbn, METH_VARARGS,
     "run_bbn(params: dict, data_dir: str) -> dict\n\n"
     "Run one cprimat_run-equivalent BBN computation and return the result "
     "dict (same keys as primat.PRIMAT.solve()), plus a 'Y_final' sub-dict "
     "of every tracked nuclide's final mass fraction."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef primat_c_module = {
    PyModuleDef_HEAD_INIT,
    "_primat_c",
    "C-extension bridge to the primat-c BBN solver.",
    -1,
    primat_c_methods
};

PyMODINIT_FUNC PyInit__primat_c(void)
{
    return PyModule_Create(&primat_c_module);
}
