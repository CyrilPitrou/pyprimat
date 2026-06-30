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
 * `data_dir` is the data root directory (cfg._resolved_data_dir on the Python
 * side — the equivalent of primat/data/, passed in from backend.py).
 * `user_nuclear_dir` (additive nuclear overlay) is an ordinary params key
 * applied generically via cpr_config_set_by_name in step 2, so it reaches
 * the C-side cpr_config_resolve_rates_path without special-casing here.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cprimat/api.h"
#include "cprimat/config.h"
#include "cprimat/mc.h"

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

/* Converts a Python sequence of float-convertible items into a malloc'd
 * double array of length `*out_n` (the sequence's own length). Returns
 * NULL (with a Python exception set, nothing allocated) on failure;
 * an empty sequence yields a non-NULL zero-length allocation so callers
 * can still memcpy/free it uniformly. */
static double *seq_to_doubles(PyObject *seq_obj, size_t *out_n)
{
    PyObject *seq = PySequence_Fast(seq_obj, "expected a sequence of float");
    if (!seq)
        return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    double *arr = malloc((n > 0 ? (size_t)n : 1) * sizeof(double));
    if (!arr) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        double v = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
        if (v == -1.0 && PyErr_Occurred()) {
            free(arr);
            Py_DECREF(seq);
            return NULL;
        }
        arr[i] = v;
    }
    Py_DECREF(seq);
    *out_n = (size_t)n;
    return arr;
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

/* Parses one raw "T9 rate [err]" text blob (the GUI's verbatim
 * custom_network["replaced"/"added"] value, np.loadtxt-compatible: '#'
 * comment lines and blank lines skipped, whitespace-separated columns) into
 * heap T9/rate/err arrays, mirroring Python's np.loadtxt + np.zeros_like
 * err fallback (network_data.py's custom_tables consumer). Returns 0 on
 * success (caller owns *T9/*rate/*err, each length *n), 1 (with a Python
 * exception set) if no data row parses. */
static int parse_custom_table_text(const char *text, double **T9, double **rate,
                                    double **err, size_t *n)
{
    size_t cap = 16, count = 0;
    double *t9 = malloc(cap * sizeof(double));
    double *rt = malloc(cap * sizeof(double));
    double *er = malloc(cap * sizeof(double));
    if (!t9 || !rt || !er) { free(t9); free(rt); free(er); return PyErr_NoMemory() != NULL; }

    char *copy = strdup(text);
    if (!copy) { free(t9); free(rt); free(er); return PyErr_NoMemory() != NULL; }

    char *saveline = NULL;
    for (char *line = strtok_r(copy, "\n", &saveline); line; line = strtok_r(NULL, "\n", &saveline)) {
        while (*line == ' ' || *line == '\t') line++;
        if (*line == '\0' || *line == '#') continue;

        char *end1, *end2, *end3;
        double v1 = strtod(line, &end1);
        if (end1 == line) continue; /* not a data row */
        double v2 = strtod(end1, &end2);
        if (end2 == end1) continue; /* needs at least T9 and rate */
        double v3 = strtod(end2, &end3);
        if (end3 == end2) v3 = 0.0; /* no third column: zero error, mirrors np.zeros_like */

        if (count == cap) {
            cap *= 2;
            t9 = realloc(t9, cap * sizeof(double));
            rt = realloc(rt, cap * sizeof(double));
            er = realloc(er, cap * sizeof(double));
        }
        t9[count] = v1; rt[count] = v2; er[count] = v3;
        count++;
    }
    free(copy);

    if (count == 0) {
        free(t9); free(rt); free(er);
        PyErr_SetString(PyExc_ValueError, "custom_network rate table has no data rows");
        return 1;
    }
    *T9 = t9; *rate = rt; *err = er; *n = count;
    return 0;
}

static void free_custom_network(CPRCustomNetwork *c)
{
    if (!c) return;
    free(c->removed);
    for (size_t i = 0; i < c->n_tables; i++) {
        free(c->tables[i].T9); free(c->tables[i].rate); free(c->tables[i].err);
    }
    free(c->tables);
}

/* Parses the GUI's custom_network dict ({"removed": [str], "replaced":
 * {name: text}, "added": {name: text}, "filenames": ... (ignored, display-
 * only)}) into a heap CPRCustomNetwork (see network_data.h), merging
 * "replaced"/"added" into one `tables` array exactly as Python merges them
 * into one custom_tables dict before calling load_network. `custom_network`
 * may be Py_None, in which case *out is zeroed (n_removed=n_tables=0) and
 * cpr_load_network treats a NULL/all-empty CPRCustomNetwork as a no-op.
 * Returns 0 on success (caller must free_custom_network), 1 (with a Python
 * exception set) on a malformed dict or unparseable table. */
static int dict_to_custom_network(PyObject *custom_network, CPRCustomNetwork *out)
{
    memset(out, 0, sizeof(*out));
    if (custom_network == NULL || custom_network == Py_None)
        return 0;
    if (!PyDict_Check(custom_network)) {
        PyErr_SetString(PyExc_TypeError, "custom_network must be a dict or None");
        return 1;
    }

    PyObject *removed_obj = PyDict_GetItemString(custom_network, "removed");
    if (removed_obj && removed_obj != Py_None) {
        PyObject *seq = PySequence_Fast(removed_obj, "custom_network['removed'] must be a sequence of str");
        if (!seq) return 1;
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        out->removed = malloc((size_t)n * sizeof(*out->removed));
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
            const char *s = PyUnicode_Check(item) ? PyUnicode_AsUTF8(item) : NULL;
            if (!s) {
                PyErr_SetString(PyExc_TypeError, "custom_network['removed'] must be a sequence of str");
                Py_DECREF(seq); free_custom_network(out); return 1;
            }
            snprintf(out->removed[i], 64, "%s", s);
        }
        out->n_removed = (size_t)n;
        Py_DECREF(seq);
    }

    const char *subkeys[2] = {"replaced", "added"};
    for (int k = 0; k < 2; k++) {
        PyObject *sub = PyDict_GetItemString(custom_network, subkeys[k]);
        if (!sub || sub == Py_None) continue;
        if (!PyDict_Check(sub)) {
            PyErr_Format(PyExc_TypeError, "custom_network['%s'] must be a dict", subkeys[k]);
            free_custom_network(out); return 1;
        }
        PyObject *name_obj, *text_obj;
        Py_ssize_t pos = 0;
        while (PyDict_Next(sub, &pos, &name_obj, &text_obj)) {
            const char *name = PyUnicode_Check(name_obj) ? PyUnicode_AsUTF8(name_obj) : NULL;
            const char *text = PyUnicode_Check(text_obj) ? PyUnicode_AsUTF8(text_obj) : NULL;
            if (!name || !text) {
                PyErr_Format(PyExc_TypeError, "custom_network['%s'] must map str -> str", subkeys[k]);
                free_custom_network(out); return 1;
            }
            double *T9, *rate, *err;
            size_t n;
            if (parse_custom_table_text(text, &T9, &rate, &err, &n)) {
                free_custom_network(out); return 1;
            }
            out->tables = realloc(out->tables, (out->n_tables + 1) * sizeof(*out->tables));
            CPRCustomTable *ct = &out->tables[out->n_tables++];
            snprintf(ct->name, sizeof(ct->name), "%s", name);
            ct->T9 = T9; ct->rate = rate; ct->err = err; ct->n = n;
        }
    }
    return 0;
}

static PyObject *primat_c_run_bbn(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    PyObject *params;
    const char *data_dir;
    PyObject *custom_network = NULL;

    int show_progress = 1; /* default: show phase markers (matches Python backend) */
    static char *kwlist[] = {"params", "data_dir", "custom_network", "show_progress", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Os|Oi", kwlist,
                                       &params, &data_dir, &custom_network, &show_progress))
        return NULL;
    if (!PyDict_Check(params)) {
        PyErr_SetString(PyExc_TypeError, "params must be a dict");
        return NULL;
    }

    CPRCustomNetwork custom;
    if (dict_to_custom_network(custom_network, &custom))
        return NULL;

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
        free_custom_network(&custom);
        return NULL;
    }

    if (cpr_config_validate(&cfg, &errmsg)) {
        PyErr_Format(PyExc_ValueError, "%s", errmsg ? errmsg : "cpr_config_validate failed");
        free(errmsg);
        cpr_config_free(&cfg);
        free_custom_network(&custom);
        return NULL;
    }
    cfg.show_progress = show_progress;

    CPRResults results;
    int rc = cprimat_run(&cfg, &custom, &results, &errmsg);
    cpr_config_free(&cfg);
    free_custom_network(&custom);
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

/* Converts a Python `params` dict into a CPRParamSet array suitable for
 * cpr_mc_uncertainty's `base_params` (each worker thread re-applies these
 * to its own CPRConfig -- see mc.c's worker_setup). Unlike run_bbn's
 * streaming cpr_config_set_by_name loop, mc needs the (key, value) pairs
 * collected up front since they are reused by every worker. `key`s point
 * directly into the live `params` dict's str objects (valid for the
 * lifetime of this call, no copy needed); CPR_STRING `value`s are
 * strdup'd via py_to_cprparam and tracked in `*out_owned` for the caller
 * to free after the cpr_mc_uncertainty call returns. Returns 0 on
 * success, 1 (with a Python exception set, partial state already freed)
 * on failure. */
static int dict_to_paramset(PyObject *params, CPRParamSet **out_set,
                             char ***out_owned, size_t *out_n)
{
    Py_ssize_t n = PyDict_Size(params);
    CPRParamSet *set = malloc((size_t)n * sizeof(CPRParamSet));
    char **owned = calloc((size_t)n, sizeof(char *));
    if ((n > 0 && (!set || !owned))) {
        free(set); free(owned);
        PyErr_NoMemory();
        return 1;
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    size_t idx = 0;
    while (PyDict_Next(params, &pos, &key, &value)) {
        if (!PyUnicode_Check(key)) {
            PyErr_SetString(PyExc_TypeError, "params keys must be str");
            goto fail;
        }
        const char *name = PyUnicode_AsUTF8(key);
        if (!name) goto fail;

        CPRParam p;
        if (py_to_cprparam(value, &p, &owned[idx]))
            goto fail;
        set[idx].key = name;
        set[idx].value = p;
        idx++;
    }

    *out_set = set;
    *out_owned = owned;
    *out_n = (size_t)n;
    return 0;

fail:
    for (size_t i = 0; i < (size_t)n; i++) free(owned[i]);
    free(owned);
    free(set);
    return 1;
}

static PyObject *primat_c_run_mc(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    PyObject *params, *quantities_obj;
    const char *data_dir;
    int num_mc;
    int seed = 0;
    int n_jobs = -1;
    PyObject *custom_network = NULL;
    PyObject *prev_centrals_obj = NULL;
    PyObject *prev_values_obj = NULL;
    int progress = 1; /* default: show progress (mirrors Python backend's progress=True) */

    static char *kwlist[] = {"params", "data_dir", "num_mc", "quantities",
                              "seed", "n_jobs", "custom_network",
                              "prev_centrals", "prev_values", "progress", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OsiO|iiOOOi", kwlist,
                                      &params, &data_dir, &num_mc, &quantities_obj,
                                      &seed, &n_jobs, &custom_network,
                                      &prev_centrals_obj, &prev_values_obj, &progress))
        return NULL;
    if (!PyDict_Check(params)) {
        PyErr_SetString(PyExc_TypeError, "params must be a dict");
        return NULL;
    }

    CPRCustomNetwork custom;
    if (dict_to_custom_network(custom_network, &custom))
        return NULL;
    PyObject *quantities_seq = PySequence_Fast(quantities_obj,
        "quantities must be a sequence of str");
    if (!quantities_seq)
        return NULL;
    Py_ssize_t n_q = PySequence_Fast_GET_SIZE(quantities_seq);
    const char **quantities = malloc((size_t)n_q * sizeof(char *));
    if (n_q > 0 && !quantities) {
        Py_DECREF(quantities_seq);
        free_custom_network(&custom);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < n_q; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(quantities_seq, i);
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "quantities must be a sequence of str");
            free(quantities);
            Py_DECREF(quantities_seq);
            free_custom_network(&custom);
            return NULL;
        }
        quantities[i] = PyUnicode_AsUTF8(item);
        if (!quantities[i]) {
            free(quantities);
            Py_DECREF(quantities_seq);
            free_custom_network(&custom);
            return NULL;
        }
    }

    CPRParamSet *paramset = NULL;
    char **owned = NULL;
    size_t n_params = 0;
    if (dict_to_paramset(params, &paramset, &owned, &n_params)) {
        free(quantities);
        Py_DECREF(quantities_seq);
        free_custom_network(&custom);
        return NULL;
    }

    /* Incremental-reuse arrays (primat/backend.py's run_mc, the C-side
     * counterpart of mc_uncertainty's `prev`): prev_centrals is a sequence
     * of n_q floats; prev_values is a sequence of n_q sequences, all of the
     * same length n_prev -- both optional/None for an ordinary from-scratch
     * run. backend.py is responsible for only passing these when prev is
     * sample-compatible with this call (same seed/params/custom_network/
     * quantities); this wrapper does not re-check that. */
    double *prev_centrals = NULL;
    double **prev_values = NULL;
    size_t n_prev = 0;
    if (prev_centrals_obj != NULL && prev_centrals_obj != Py_None) {
        size_t n_pc = 0;
        prev_centrals = seq_to_doubles(prev_centrals_obj, &n_pc);
        if (!prev_centrals || (Py_ssize_t)n_pc != n_q) {
            free(prev_centrals);
            if (n_pc != (size_t)n_q && prev_centrals)
                PyErr_SetString(PyExc_ValueError, "prev_centrals length must match quantities");
            free(quantities);
            Py_DECREF(quantities_seq);
            free_custom_network(&custom);
            for (size_t i = 0; i < n_params; i++) free(owned[i]);
            free(owned);
            free(paramset);
            return NULL;
        }
    }
    PyObject *prev_values_seq = NULL;
    if (prev_centrals != NULL && prev_values_obj != NULL && prev_values_obj != Py_None) {
        prev_values_seq = PySequence_Fast(prev_values_obj, "prev_values must be a sequence");
        if (!prev_values_seq || PySequence_Fast_GET_SIZE(prev_values_seq) != n_q) {
            Py_XDECREF(prev_values_seq);
            if (prev_values_seq)
                PyErr_SetString(PyExc_ValueError, "prev_values length must match quantities");
            free(prev_centrals);
            free(quantities);
            Py_DECREF(quantities_seq);
            free_custom_network(&custom);
            for (size_t i = 0; i < n_params; i++) free(owned[i]);
            free(owned);
            free(paramset);
            return NULL;
        }
        prev_values = calloc((size_t)n_q, sizeof(double *));
        for (Py_ssize_t i = 0; i < n_q; i++) {
            size_t n_this = 0;
            prev_values[i] = seq_to_doubles(PySequence_Fast_GET_ITEM(prev_values_seq, i), &n_this);
            int bad = !prev_values[i] || (i > 0 && n_this != n_prev);
            if (i == 0 && prev_values[i]) n_prev = n_this;
            if (bad) {
                if (prev_values[i])
                    PyErr_SetString(PyExc_ValueError, "prev_values entries must all have the same length");
                for (Py_ssize_t j = 0; j <= i; j++) free(prev_values[j]);
                free(prev_values);
                Py_DECREF(prev_values_seq);
                free(prev_centrals);
                free(quantities);
                Py_DECREF(quantities_seq);
                free_custom_network(&custom);
                for (size_t k = 0; k < n_params; k++) free(owned[k]);
                free(owned);
                free(paramset);
                return NULL;
            }
        }
    }

    CPRMCResult out;
    char *errmsg = NULL;
    int rc = cpr_mc_uncertainty(num_mc, quantities, (size_t)n_q, data_dir,
                                 paramset, n_params, seed, n_jobs, &custom,
                                 prev_centrals, (const double * const *)prev_values, n_prev,
                                 progress, &out, &errmsg);
    free_custom_network(&custom);

    free(prev_centrals);
    if (prev_values) {
        for (Py_ssize_t i = 0; i < n_q; i++) free(prev_values[i]);
        free(prev_values);
    }
    Py_XDECREF(prev_values_seq);

    for (size_t i = 0; i < n_params; i++) free(owned[i]);
    free(owned);
    free(paramset);
    free(quantities);
    Py_DECREF(quantities_seq);

    if (rc) {
        PyErr_Format(PyExc_RuntimeError, "cpr_mc_uncertainty failed: %s",
                     errmsg ? errmsg : "(no message)");
        free(errmsg);
        return NULL;
    }

    PyObject *d = PyDict_New();
    if (!d) { cpr_mc_result_free(&out); return NULL; }
    for (size_t i = 0; i < out.n; i++) {
        PyObject *item = PyDict_New();
        PyObject *vals = NULL;
        if (!item
            || PyDict_SetItemString(item, "central", PyFloat_FromDouble(out.items[i].central)) < 0
            || PyDict_SetItemString(item, "mean", PyFloat_FromDouble(out.items[i].mean)) < 0
            || PyDict_SetItemString(item, "std", PyFloat_FromDouble(out.items[i].std)) < 0
            || !(vals = doubles_to_list(out.items[i].values, (size_t)num_mc))
            || PyDict_SetItemString(item, "values", vals) < 0
            || PyDict_SetItemString(d, out.items[i].name, item) < 0) {
            Py_XDECREF(vals);
            Py_XDECREF(item);
            Py_DECREF(d);
            cpr_mc_result_free(&out);
            return NULL;
        }
        Py_DECREF(vals);
        Py_DECREF(item);
    }
    cpr_mc_result_free(&out);
    return d;
}

static PyMethodDef primat_c_methods[] = {
    {"run_bbn", (PyCFunction)primat_c_run_bbn, METH_VARARGS | METH_KEYWORDS,
     "run_bbn(params: dict, data_dir: str, custom_network: dict|None = None) -> dict\n\n"
     "Run one cprimat_run-equivalent BBN computation and return the result "
     "dict (same keys as primat.PRIMAT.solve()), plus a 'Y_final' sub-dict "
     "of every tracked nuclide's final mass fraction. `custom_network` is "
     "the GUI 'Customise Reactions' override ({'removed': [...], "
     "'replaced': {...}, 'added': {...}}, see network_data.h's CPRCustomNetwork)."},
    {"run_mc", (PyCFunction)primat_c_run_mc, METH_VARARGS | METH_KEYWORDS,
     "run_mc(params, data_dir, num_mc, quantities, seed=0, n_jobs=-1, custom_network=None,\n"
     "       prev_centrals=None, prev_values=None) -> dict\n\n"
     "Run cpr_mc_uncertainty (primat-c/src/mc.c): num_mc threaded MC samples "
     "of every quantity in `quantities` (result-dict key or nuclide name), "
     "perturbing nuclear rates and tau_n. Returns {name: {central, mean, "
     "std, values}}; values has length num_mc. Note: the C side uses a "
     "pthread/xoshiro256** RNG, not NumPy's default_rng, so samples are "
     "statistically but not bit-for-bit comparable to the Python backend's "
     "mc_uncertainty (see mc.h). `prev_centrals` (length len(quantities)) "
     "and `prev_values` (a sequence of len(quantities) equal-length sample "
     "sequences) reuse a previously computed C-side result instead of "
     "recomputing it -- the incremental-extension counterpart of Python's "
     "mc_uncertainty(..., prev=...); the caller (primat/backend.py) is "
     "responsible for only passing these when compatible with this call's "
     "seed/params/custom_network/quantities."},
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
