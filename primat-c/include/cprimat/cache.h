/* cache.h -- fingerprinted on-disk cache support (port of
 * primat/cache_utils.py + primat/weak_rates/cache.py).
 *
 * The fingerprint hash MUST reproduce Python's byte-for-byte:
 *   sha256(json.dumps(fingerprint_dict, sort_keys=True, separators=(",", ":")))[:16]
 * so that a cache file written by either implementation is a cache hit for
 * the other. See cache.c for the canonical-JSON serialisation rules this
 * requires (key sort order, bool/int/float/string/null formatting).
 */
#ifndef CPRIMAT_CACHE_H
#define CPRIMAT_CACHE_H

#include "cprimat/config.h"
#include <stddef.h>

/* One (key, value) pair of a fingerprint dict, pre-serialisation. */
typedef struct {
    const char *key;
    CPRParam value;
} CPRFPField;

/* Serialises `fields` (in any order -- sorted internally by key, matching
 * Python's sort_keys=True) to canonical JSON, e.g.
 * {"a":1,"b":true,"c":null}. Returned string is malloc'd; caller frees. */
char *cpr_fingerprint_json(const CPRFPField *fields, size_t n);

/* Returns the 16-hex-character sha256 prefix of `json_str` (matches
 * primat.cache_utils.fingerprint_hash applied to the already-serialised
 * dict). Returned string is malloc'd; caller frees. */
char *cpr_sha256_hex16(const char *json_str);

/* Convenience: cpr_sha256_hex16(cpr_fingerprint_json(fields, n)), freeing
 * the intermediate JSON string. Malloc'd; caller frees. */
char *cpr_fingerprint_hash(const CPRFPField *fields, size_t n);

/* Builds the n<->p weak-rate cache fingerprint (nTOp_<hash>.txt), mirroring
 * weak_rates.cache._weak_rate_fingerprint(cfg). `out` must have room for at
 * least 20 fields (16 listed fields + format_version + the 3 explicitly
 * duplicated ones, matching the Python dict construction order -- final
 * field count after dedup by key is 16). Returns the number of fields
 * written (for cpr_fingerprint_json/_hash's `n` argument). Field .key
 * pointers point into static string literals (safe to outlive `cfg`); any
 * CPR_STRING .value.v.s pointers alias cfg's own string fields and must not
 * outlive `cfg`. */
size_t cpr_weak_rate_fingerprint(const CPRConfig *cfg, CPRFPField *out);

/* Builds the thermal-correction cache fingerprint (nTOp_thermal_<hash>.txt),
 * mirroring weak_rates.cache._thermal_fingerprint(cfg). `out` must have room
 * for at least 8 fields. Returns the number of fields written. */
size_t cpr_thermal_fingerprint(const CPRConfig *cfg, CPRFPField *out);

/* Reads the `# fingerprint_hash: <hash>` header line of a cache file
 * written by write_cache_with_fingerprint / cpr_cache_write. Returns a
 * malloc'd hash string, or NULL if the file is missing/has no such header
 * (mirrors cache_utils.read_cache_fingerprint_hash). Caller frees. */
char *cpr_cache_read_fingerprint_hash(const char *path);

/* Writes a fingerprinted cache file in the same layout as
 * cache_utils.write_cache_with_fingerprint: an optional human-readable
 * `col_header` line, then "fingerprint_hash: <hash>" and
 * "fingerprint: <json>" lines (each "# "-prefixed, matching
 * numpy.savetxt's header convention), then an optional "provenance: <str>"
 * line (see write_cache_with_fingerprint's `provenance` docstring -- pass
 * NULL to omit; deliberately NOT part of the fingerprint/hash), then
 * `n_cols` columns of `n_rows` values each in "%.18e" format,
 * space-separated. `columns[c][r]` is column c, row r. Returns 0 on
 * success, nonzero on I/O failure. */
int cpr_cache_write(const char *path, const CPRFPField *fields, size_t n_fields,
                     const char *col_header, double **columns, size_t n_cols,
                     size_t n_rows, const char *provenance);

#endif /* CPRIMAT_CACHE_H */
