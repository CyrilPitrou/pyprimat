/* ini.h -- minimal ".ini"-ish config file loader (port of cli.py's --set
 * semantics' .ini handling).
 *
 * Format: one "KEY=VALUE" or "KEY VALUE" per (trimmed) line; lines starting
 * with '#' or ';', and blank lines, are ignored. No sections. VALUE is
 * parsed exactly like --set (cpr_parse_literal): int, then double, then
 * true/false/none (case-insensitive), else the literal string (surrounding
 * quotes stripped if present).
 */
#ifndef CPRIMAT_INI_H
#define CPRIMAT_INI_H

#include "config.h"

/* Loads `path`, applying every KEY=VALUE line to `cfg` via
 * cpr_config_set_by_name (so p_<rxn>/delta_<rxn> keys and ordinary
 * fields are both handled, with the same type-coercion rules).  An unknown
 * key or type mismatch from cpr_config_set_by_name is treated as a warning
 * (printed to stderr, mirroring Python's warnings.warn) and does not abort
 * the load -- only a missing/unreadable file is an error.
 * Returns 0 on success, nonzero with *errmsg set (caller frees) if `path`
 * cannot be opened. */
int cpr_ini_load(CPRConfig *cfg, const char *path, char **errmsg);

#endif /* CPRIMAT_INI_H */
