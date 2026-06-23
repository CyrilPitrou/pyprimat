#include "cprimat/table_io.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Splits `line` in place on whitespace and/or commas, returning pointers to
 * up to `max_fields` field starts in `fields[]` (each NUL-terminated by
 * overwriting the separator). Returns the field count. */
static size_t split_fields(char *line, char **fields, size_t max_fields)
{
    size_t n = 0;
    char *p = line;
    while (*p && n < max_fields) {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (!*p) break;
        fields[n++] = p;
        while (*p && !isspace((unsigned char)*p) && *p != ',') p++;
        if (*p) { *p = '\0'; p++; }
    }
    return n;
}

int cpr_table_read(const char *path, size_t n_cols_hint, CPRTable *out,
                    char **errmsg)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        char buf[4352];
        snprintf(buf, sizeof(buf), "cannot open table file '%s'", path);
        *errmsg = strdup(buf);
        return 1;
    }

    out->cols = NULL;
    out->n_cols = n_cols_hint;
    out->n_rows = 0;
    size_t cap = 0;

    char line[8192];
    int lineno = 0;
    char *fields[256];

    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char *s = line;
        while (isspace((unsigned char)*s)) s++;
        if (*s == '\0' || *s == '#')
            continue;

        char linecopy[8192];
        strncpy(linecopy, line, sizeof(linecopy) - 1);
        linecopy[sizeof(linecopy) - 1] = '\0';
        size_t nf = split_fields(linecopy, fields, 256);
        if (nf == 0)
            continue;

        if (out->n_cols == 0) {
            out->n_cols = nf;
        } else if (nf != out->n_cols) {
            char buf[4352];
            snprintf(buf, sizeof(buf),
                      "%s:%d: expected %zu columns, found %zu",
                      path, lineno, out->n_cols, nf);
            *errmsg = strdup(buf);
            fclose(f);
            cpr_table_free(out);
            return 1;
        }

        if (out->n_rows == cap) {
            cap = cap ? cap * 2 : 64;
            if (!out->cols) {
                out->cols = calloc(out->n_cols, sizeof(double *));
                for (size_t c = 0; c < out->n_cols; c++)
                    out->cols[c] = NULL;
            }
            for (size_t c = 0; c < out->n_cols; c++)
                out->cols[c] = realloc(out->cols[c], cap * sizeof(double));
        }

        for (size_t c = 0; c < out->n_cols; c++) {
            errno = 0;
            char *endptr;
            double v = strtod(fields[c], &endptr);
            /* ERANGE alone is not necessarily an error: strtod also sets
             * it on underflow to a subnormal (e.g. a rate-table column
             * value like 1.617129e-308, seen in
             * Li7_d__Li8_p_primat.txt) -- a successfully parsed, merely
             * tiny, value. Only overflow to +-HUGE_VAL is a genuine
             * parse failure. */
            if (endptr == fields[c] || (errno == ERANGE && fabs(v) == HUGE_VAL)) {
                char buf[4352];
                snprintf(buf, sizeof(buf),
                          "%s:%d: cannot parse '%s' as a number",
                          path, lineno, fields[c]);
                *errmsg = strdup(buf);
                fclose(f);
                cpr_table_free(out);
                return 1;
            }
            out->cols[c][out->n_rows] = v;
        }
        out->n_rows++;
    }

    fclose(f);

    if (out->n_rows == 0 && out->n_cols == 0) {
        char buf[4352];
        snprintf(buf, sizeof(buf), "%s: no data rows found", path);
        *errmsg = strdup(buf);
        return 1;
    }

    return 0;
}

void cpr_table_free(CPRTable *t)
{
    if (!t->cols) return;
    for (size_t c = 0; c < t->n_cols; c++)
        free(t->cols[c]);
    free(t->cols);
    t->cols = NULL;
    t->n_cols = t->n_rows = 0;
}
