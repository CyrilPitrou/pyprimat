/* log.c -- see cprimat/log.h. */
#include "log.h"

#include <stdarg.h>
#include <stdio.h>

void cpr_log(const CPRConfig *cfg, const char *tag, const char *fmt, ...)
{
    if (!cfg->verbose) return;

    printf("[%s-c] ", tag);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}
