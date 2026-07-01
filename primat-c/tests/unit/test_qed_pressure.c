/* test_qed_pressure.c -- checks the QED plasma-pressure correction port
 * against (a) the closed-form ultra-relativistic limit I01(0)=I2m1(0)=
 * pi^2/12, and (b) reference values from primat.qed_pressure._dPa/_dPe3
 * at T=10 and T=1 MeV (computed via Python's scipy.integrate.quad, same
 * tolerance 1e-13 -- see qed_pressure.h's top comment for why these two
 * implementations must agree to high precision: both modules must produce
 * numerically interchangeable QED tables). Also exercises
 * cpr_qed_compute_tables end-to-end and a save/reload roundtrip through
 * table_io.c, matching the file format Python's save_qed_tables produces.
 */
#include "qed_pressure.h"
#include "table_io.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

int main(void)
{
    double pi2_12 = M_PI * M_PI / 12.0;
    CHECK(fabs(cpr_qed_I01(0.0) - pi2_12) < 1e-10, "I01(0) matches pi^2/12");
    CHECK(fabs(cpr_qed_I2m1(0.0) - pi2_12) < 1e-10, "I2m1(0) matches pi^2/12");

    double alpha = 1.0 / 137.035999084;
    double me    = 0.5109989461;

    CHECK(fabs(cpr_qed_dPa(10.0, alpha, me) - (-15.859101755349474)) < 1e-8,
          "dPa(10 MeV) matches Python reference");
    CHECK(fabs(cpr_qed_dPe3(10.0, alpha, me) - 1.4167436751938973) < 1e-8,
          "dPe3(10 MeV) matches Python reference");
    CHECK(fabs(cpr_qed_dPa(1.0, alpha, me) - (-0.0013324374015057094)) < 1e-10,
          "dPa(1 MeV) matches Python reference");
    CHECK(fabs(cpr_qed_dPe3(1.0, alpha, me) - 0.00013362887257597347) < 1e-10,
          "dPe3(1 MeV) matches Python reference");

    /* End-to-end table computation + spline derivatives. */
    char *err = NULL;
    CPRQEDTables t;
    int rc = cpr_qed_compute_tables(1e-3, 1e2, 200, alpha, me, &t, &err);
    CHECK(rc == 0, "cpr_qed_compute_tables succeeds");
    CHECK(t.n == 200, "table has requested length");

    /* At T=10 MeV (an interior grid point, large n means a nearby knot
     * exists) the tabulated dP_e2/dP_e3 should match the direct evaluation
     * closely (same underlying formula, just sampled on the log grid). */
    size_t i10 = 0;
    for (size_t i = 0; i < t.n; i++) if (t.T[i] > 9.0 && t.T[i] < 11.0) { i10 = i; break; }
    CHECK(fabs(t.dP_e2[i10] - cpr_qed_dPa(t.T[i10], alpha, me)) < 1e-6,
          "tabulated dP_e2 near T=10 matches direct cpr_qed_dPa");

    /* The spline derivative of dP_e2 should match a centered finite
     * difference of the direct (non-tabulated) function to a few percent
     * -- this is the same loose check qed_pressure.py's docstring quotes
     * ("<0.01% at all T" for the well-resolved analytic functions; a
     * coarser tolerance is used here since the grid is much sparser than
     * Python's default n_pts=500). */
    double T0 = t.T[i10];
    double dT = T0 * 1e-4;
    double fd = (cpr_qed_dPa(T0 + dT, alpha, me) - cpr_qed_dPa(T0 - dT, alpha, me)) / (2.0 * dT);
    CHECK(fabs(t.d_dP_e2_dT[i10] - fd) < 1e-2 * fabs(fd),
          "spline d(dP_e2)/dT matches finite difference to 1%");

    /* Save and reload via table_io.c, exactly the "file mode" path
     * plasma.c will use; verifies cpr_qed_save_tables' on-disk format is
     * self-consistent.  The saved files are QED_pressure_correction_e2.txt/
     * QED_pressure_correction_e3.txt, each with 4 columns. */
    rc = cpr_qed_save_tables(&t, "/tmp", &err);
    CHECK(rc == 0, "cpr_qed_save_tables succeeds");

    CPRTable loaded;
    rc = cpr_table_read("/tmp/QED_pressure_correction_e2.txt", 4, &loaded, &err);
    CHECK(rc == 0, "saved QED_pressure_correction_e2.txt reloads via cpr_table_read");
    CHECK(loaded.n_rows == t.n, "reloaded e2 table has the same row count");
    CHECK(fabs(loaded.cols[0][i10] - t.T[i10]) < 1e-6, "reloaded e2 T column matches");
    CHECK(fabs(loaded.cols[1][i10] - t.dP_e2[i10]) < fabs(t.dP_e2[i10]) * 1e-5 + 1e-12,
          "reloaded dP_a column (col 1) matches to file precision (%.6E)");
    CHECK(fabs(loaded.cols[2][i10] - t.d_dP_e2_dT[i10])
              < fabs(t.d_dP_e2_dT[i10]) * 1e-5 + 1e-12,
          "reloaded d(dP_a)/dT column (col 2) matches to file precision (%.6E)");
    cpr_table_free(&loaded);

    rc = cpr_table_read("/tmp/QED_pressure_correction_e3.txt", 4, &loaded, &err);
    CHECK(rc == 0, "saved QED_pressure_correction_e3.txt reloads via cpr_table_read");
    CHECK(loaded.n_rows == t.n, "reloaded e3 table has the same row count");
    CHECK(fabs(loaded.cols[0][i10] - t.T[i10]) < 1e-6, "reloaded e3 T column matches");
    CHECK(fabs(loaded.cols[1][i10] - t.dP_e3[i10]) < fabs(t.dP_e3[i10]) * 1e-5 + 1e-12,
          "reloaded dP_e3 column (col 1) matches to file precision (%.6E)");
    cpr_table_free(&loaded);

    cpr_qed_tables_free(&t);

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
