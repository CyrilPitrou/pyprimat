/* test_ini.c -- verifies cpr_ini_load applies KEY=VALUE overrides on top of
 * defaults, and that examples/run_large_amax8.ini round-trips amax/network. */
#include "cprimat/config.h"
#include "cprimat/ini.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
        if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
        else printf("ok: %s\n", msg); \
    } while (0)

int main(void)
{
    char *err = NULL;
    CPRConfig cfg;
    if (cpr_config_init_defaults(&cfg, "../primat/data", &err)) {
        printf("FAIL cpr_config_init_defaults: %s\n", err);
        return 1;
    }

    if (cpr_ini_load(&cfg, "examples/run_large_amax8.ini", &err)) {
        printf("FAIL cpr_ini_load: %s\n", err);
        return 1;
    }
    CHECK(strcmp(cfg.network, "large") == 0, "network == large after ini load");
    CHECK(cfg.amax == 8, "amax == 8 after ini load");
    CHECK(cfg.Omegabh2_ == 0.022425, "Omegabh2 == 0.022425 after ini load");

    if (cpr_config_validate(&cfg, &err)) {
        printf("FAIL cpr_config_validate: %s\n", err);
        failures++;
    } else {
        printf("ok: validate succeeds after ini load\n");
    }

    cpr_config_free(&cfg);

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
