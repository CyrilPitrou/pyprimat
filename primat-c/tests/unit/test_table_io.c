/* test_table_io.c -- reads real rates/ files and checks row/col counts and
 * a couple of known values, to confirm the generic reader handles every
 * shape it needs to (whitespace-separated rate tables, comma-separated
 * NEVO tables, single-column NEVOGrid). */
#include "table_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

static int close(double a, double b) { return fabs(a - b) < 1e-9 * fabs(b) + 1e-300; }

int main(void)
{
    char *err = NULL;
    CPRTable t;

    /* n_p__d_g rate table: T9, rate, error -- 3 columns. */
    if (cpr_table_read("../primat/data/nuclear/tables/n_p__d_g/n_p__d_g_primat.txt",
                        3, &t, &err)) {
        printf("FAIL n_p__d_g read: %s\n", err);
        return 1;
    }
    CHECK(t.n_cols == 3, "n_p__d_g has 3 columns");
    CHECK(t.n_rows > 100, "n_p__d_g has many rows");
    CHECK(close(t.cols[0][0], 1.0e-3), "n_p__d_g first T9 == 1e-3");
    CHECK(close(t.cols[1][0], 4.414e+04), "n_p__d_g first rate == 4.414e4");
    cpr_table_free(&t);

    /* NEVO col_1_7: comma-separated, 7 columns, 600 rows. */
    if (cpr_table_read("../primat/data/NEVO/NEVOPRIMAT_col_1_7.csv", 0, &t, &err)) {
        printf("FAIL NEVO col_1_7 read: %s\n", err);
        return 1;
    }
    CHECK(t.n_cols == 7, "NEVO col_1_7 has 7 columns");
    CHECK(t.n_rows == 600, "NEVO col_1_7 has 600 rows");
    CHECK(close(t.cols[0][0], 1.277497374999999984e-02), "NEVO col_1_7 first x");
    cpr_table_free(&t);

    /* NEVOGrid: single column. */
    if (cpr_table_read("../primat/data/NEVO/NEVOGrid.csv", 1, &t, &err)) {
        printf("FAIL NEVOGrid read: %s\n", err);
        return 1;
    }
    CHECK(t.n_cols == 1, "NEVOGrid has 1 column");
    CHECK(close(t.cols[0][0], 3.289640684128431782e-03), "NEVOGrid first value");
    cpr_table_free(&t);

    /* QED_pressure_correction_e2.txt: 4 columns (T, dP_a, derivatives…), 4 comment lines. */
    if (cpr_table_read("../primat/data/plasma/QED_pressure_correction_e2.txt", 4, &t, &err)) {
        printf("FAIL QED_pressure_correction_e2 read: %s\n", err);
        return 1;
    }
    CHECK(t.n_cols == 4, "QED_pressure_correction_e2 has 4 columns");
    CHECK(close(t.cols[0][0], 1.0e-3), "QED_pressure_correction_e2 first T == 1e-3");
    cpr_table_free(&t);

    /* QED_pressure_correction_e3.txt: 4 columns (T, dP_e3, derivatives…), 4 comment lines. */
    if (cpr_table_read("../primat/data/plasma/QED_pressure_correction_e3.txt", 4, &t, &err)) {
        printf("FAIL QED_pressure_correction_e3 read: %s\n", err);
        return 1;
    }
    CHECK(t.n_cols == 4, "QED_pressure_correction_e3 has 4 columns");
    CHECK(close(t.cols[0][0], 1.0e-3), "QED_pressure_correction_e3 first T == 1e-3");
    cpr_table_free(&t);

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
