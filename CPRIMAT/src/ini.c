#include "cprimat/ini.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *trim(char *s)
{
    while (isspace((unsigned char)*s)) s++;
    if (*s == '\0') return s;
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) *end-- = '\0';
    return s;
}

int cpr_ini_load(CPRConfig *cfg, const char *path, char **errmsg)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        char buf[4352];
        snprintf(buf, sizeof(buf), "cannot open ini file '%s'", path);
        *errmsg = strdup(buf);
        return 1;
    }

    char line[4096];
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char *s = trim(line);
        if (*s == '\0' || *s == '#' || *s == ';')
            continue;

        /* Split at the first '=' if present, else the first run of
         * whitespace (the "KEY VALUE" form). */
        char *eq = strchr(s, '=');
        char *key, *val;
        if (eq) {
            *eq = '\0';
            key = trim(s);
            val = trim(eq + 1);
        } else {
            char *sp = s;
            while (*sp && !isspace((unsigned char)*sp)) sp++;
            if (*sp == '\0') {
                fprintf(stderr, "%s:%d: warning: ignoring line with no value: '%s'\n",
                        path, lineno, s);
                continue;
            }
            *sp = '\0';
            key = trim(s);
            val = trim(sp + 1);
        }
        if (*key == '\0') {
            fprintf(stderr, "%s:%d: warning: ignoring line with empty key\n", path, lineno);
            continue;
        }

        CPRParam value = cpr_parse_literal(val);
        char *set_err = NULL;
        if (cpr_config_set_by_name(cfg, key, value, &set_err)) {
            fprintf(stderr, "%s:%d: warning: %s\n", path, lineno,
                    set_err ? set_err : "could not set key");
            free(set_err);
        }
    }

    fclose(f);
    return 0;
}
