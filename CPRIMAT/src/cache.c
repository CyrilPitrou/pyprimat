#include "cprimat/cache.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ===========================================================================
 * SHA-256 (public-domain style, single-buffer one-shot implementation --
 * no streaming API needed since every fingerprint JSON blob is short).
 * ===========================================================================
 */
static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256(const unsigned char *msg, size_t len, unsigned char out[32])
{
    uint32_t h[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };

    /* Pad: msg || 0x80 || zeros || 64-bit big-endian bit length, to a
     * multiple of 64 bytes. */
    size_t padded_len = ((len + 9 + 63) / 64) * 64;
    unsigned char *buf = calloc(padded_len, 1);
    memcpy(buf, msg, len);
    buf[len] = 0x80;
    uint64_t bitlen = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++)
        buf[padded_len - 1 - i] = (unsigned char)(bitlen >> (8 * i));

    for (size_t off = 0; off < padded_len; off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)buf[off + 4*i] << 24) | ((uint32_t)buf[off + 4*i + 1] << 16)
                 | ((uint32_t)buf[off + 4*i + 2] << 8) | (uint32_t)buf[off + 4*i + 3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }

        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + SHA256_K[i] + w[i];
            uint32_t S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    free(buf);

    for (int i = 0; i < 8; i++) {
        out[4*i]   = (unsigned char)(h[i] >> 24);
        out[4*i+1] = (unsigned char)(h[i] >> 16);
        out[4*i+2] = (unsigned char)(h[i] >> 8);
        out[4*i+3] = (unsigned char)(h[i]);
    }
}

char *cpr_sha256_hex16(const char *json_str)
{
    unsigned char digest[32];
    sha256((const unsigned char *)json_str, strlen(json_str), digest);
    char *hex = malloc(17);
    for (int i = 0; i < 8; i++)
        snprintf(hex + 2*i, 3, "%02x", digest[i]);
    return hex;
}

/* ===========================================================================
 * Canonical JSON serialisation -- must match
 * json.dumps(d, sort_keys=True, separators=(",", ":")) exactly.
 * ===========================================================================
 */

/* Python float repr: the shortest decimal string that round-trips to the
 * same IEEE-754 double. We brute-force the precision (1..17 significant
 * digits via "%.*e"/"%.*g") rather than porting Grisu/dtoa -- the
 * fingerprint fields are always "nice" config values (0.0, 0.001, 40.0,
 * ...), so this is never on a hot path and correctness-by-construction
 * (verify round-trip with strtod) matters more than speed here. */
/* Python's float repr (CPython's format_float_short, mode 'r'): find the
 * shortest decimal digit string that round-trips to the same double, then
 * format it fixed-point if its decimal exponent is in [-4, 16), else
 * scientific -- e.g. repr(40.0)=="40.0", repr(0.001)=="0.001",
 * repr(1e+16)=="1e+16", repr(1e-05)=="1e-05". Using "%.*e" (always
 * scientific) to find the shortest round-tripping digit string avoids %g's
 * premature switch to scientific notation at low precision (e.g. "%.1g" of
 * 40.0 is "4e+01", not "40"), which is the bug this replaces. */
static void format_python_float(double v, char *buf, size_t bufsize)
{
    if (v == 0.0) {
        snprintf(buf, bufsize, "%s", signbit(v) ? "-0.0" : "0.0");
        return;
    }

    char e_str[64];
    int p;
    for (p = 1; p <= 17; p++) {
        snprintf(e_str, sizeof(e_str), "%.*e", p - 1, v);
        if (strtod(e_str, NULL) == v)
            break;
    }

    /* e_str is "[-]d[.ddd]e±EE". Extract sign, significant digits (no '.'),
     * and the decimal exponent of the leading digit. */
    const char *s = e_str;
    int neg = 0;
    if (*s == '-') { neg = 1; s++; }
    char digits[32];
    int nd = 0;
    digits[nd++] = *s++; /* leading digit */
    if (*s == '.') {
        s++;
        while (*s && *s != 'e' && *s != 'E')
            digits[nd++] = *s++;
    }
    while (*s && *s != 'e' && *s != 'E') s++;
    s++; /* skip 'e' */
    int decexp = atoi(s);

    char out[64];
    size_t len = 0;
    if (neg) out[len++] = '-';

    if (decexp >= -4 && decexp < 16) {
        if (decexp >= 0) {
            int int_digits = decexp + 1;
            for (int i = 0; i < int_digits; i++)
                out[len++] = (i < nd) ? digits[i] : '0';
            out[len++] = '.';
            if (int_digits < nd) {
                for (int i = int_digits; i < nd; i++) out[len++] = digits[i];
            } else {
                out[len++] = '0';
            }
        } else {
            out[len++] = '0';
            out[len++] = '.';
            for (int i = 0; i < -decexp - 1; i++) out[len++] = '0';
            for (int i = 0; i < nd; i++) out[len++] = digits[i];
        }
    } else {
        out[len++] = digits[0];
        if (nd > 1) {
            out[len++] = '.';
            for (int i = 1; i < nd; i++) out[len++] = digits[i];
        }
        len += snprintf(out + len, sizeof(out) - len, "e%+03d", decexp);
    }
    out[len] = '\0';
    snprintf(buf, bufsize, "%s", out);
}

static void json_escape_append(char **buf, size_t *cap, size_t *len, const char *s)
{
    size_t need = *len + strlen(s) * 2 + 4;
    if (need > *cap) {
        *cap = need * 2;
        *buf = realloc(*buf, *cap);
    }
    (*buf)[(*len)++] = '"';
    for (const char *p = s; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '"' || c == '\\') {
            (*buf)[(*len)++] = '\\';
            (*buf)[(*len)++] = (char)c;
        } else if (c == '\n') { (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 'n'; }
        else if (c == '\t') { (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 't'; }
        else if (c < 0x20) {
            *len += snprintf(*buf + *len, 8, "\\u%04x", c);
        } else {
            (*buf)[(*len)++] = (char)c;
        }
    }
    (*buf)[(*len)++] = '"';
}

static void buf_append(char **buf, size_t *cap, size_t *len, const char *s)
{
    size_t slen = strlen(s);
    if (*len + slen + 1 > *cap) {
        *cap = (*len + slen + 1) * 2;
        *buf = realloc(*buf, *cap);
    }
    memcpy(*buf + *len, s, slen + 1);
    *len += slen;
}

char *cpr_fingerprint_json(const CPRFPField *fields, size_t n)
{
    /* Sort a local copy of the field indices by key (byte-wise, matching
     * Python's default string comparison for plain ASCII identifiers). */
    size_t *order = malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) order[i] = i;
    for (size_t i = 1; i < n; i++) {
        size_t j = i;
        while (j > 0 && strcmp(fields[order[j-1]].key, fields[order[j]].key) > 0) {
            size_t t = order[j-1]; order[j-1] = order[j]; order[j] = t;
            j--;
        }
    }

    size_t cap = 256, len = 0;
    char *buf = malloc(cap);
    buf_append(&buf, &cap, &len, "{");
    for (size_t k = 0; k < n; k++) {
        const CPRFPField *f = &fields[order[k]];
        if (k > 0) buf_append(&buf, &cap, &len, ",");
        json_escape_append(&buf, &cap, &len, f->key);
        buf_append(&buf, &cap, &len, ":");
        char num[64];
        switch (f->value.type) {
        case CPR_NONE:
            buf_append(&buf, &cap, &len, "null");
            break;
        case CPR_BOOL:
            buf_append(&buf, &cap, &len, f->value.v.b ? "true" : "false");
            break;
        case CPR_INT:
            snprintf(num, sizeof(num), "%ld", f->value.v.i);
            buf_append(&buf, &cap, &len, num);
            break;
        case CPR_DOUBLE:
            format_python_float(f->value.v.d, num, sizeof(num));
            buf_append(&buf, &cap, &len, num);
            break;
        case CPR_STRING:
            json_escape_append(&buf, &cap, &len, f->value.v.s);
            break;
        }
    }
    buf_append(&buf, &cap, &len, "}");
    free(order);
    return buf;
}

char *cpr_fingerprint_hash(const CPRFPField *fields, size_t n)
{
    char *json = cpr_fingerprint_json(fields, n);
    char *hash = cpr_sha256_hex16(json);
    free(json);
    return hash;
}

/* ===========================================================================
 * Weak-rate / thermal fingerprint builders (port of weak_rates/cache.py).
 * ===========================================================================
 */
static CPRParam pb(int b) { CPRParam p; p.type = CPR_BOOL; p.v.b = b; return p; }
static CPRParam pi(long i) { CPRParam p; p.type = CPR_INT; p.v.i = i; return p; }
static CPRParam pd(double d) { CPRParam p; p.type = CPR_DOUBLE; p.v.d = d; return p; }
static CPRParam ps(const char *s) {
    CPRParam p;
    if (s) { p.type = CPR_STRING; p.v.s = s; }
    else { p.type = CPR_NONE; }
    return p;
}

/* WEAK_RATE_FORMAT_VERSION in weak_rates/cache.py. */
#define WEAK_RATE_FORMAT_VERSION 1

size_t cpr_weak_rate_fingerprint(const CPRConfig *cfg, CPRFPField *out)
{
    size_t n = 0;
    out[n++] = (CPRFPField){"format_version", pi(WEAK_RATE_FORMAT_VERSION)};
    out[n++] = (CPRFPField){"sampling_nTOp_per_decade", pi(cfg->sampling_nTOp_per_decade)};
    out[n++] = (CPRFPField){"radiative_corrections", pb(cfg->radiative_corrections)};
    out[n++] = (CPRFPField){"finite_mass_corrections", pb(cfg->finite_mass_corrections)};
    out[n++] = (CPRFPField){"munuOverTnu", pd(cfg->munuOverTnu)};
    out[n++] = (CPRFPField){"QED_corrections", pb(cfg->QED_corrections)};
    out[n++] = (CPRFPField){"incomplete_decoupling", pb(cfg->incomplete_decoupling)};
    out[n++] = (CPRFPField){"spectral_distortions", pb(cfg->spectral_distortions)};
    out[n++] = (CPRFPField){"analytic_distortions", pb(cfg->analytic_distortions)};
    out[n++] = (CPRFPField){"y_SZ", pd(cfg->y_SZ)};
    out[n++] = (CPRFPField){"y_gray", pd(cfg->y_gray)};
    out[n++] = (CPRFPField){"T_start_cosmo_MeV", pd(cfg->T_start_cosmo_MeV)};
    out[n++] = (CPRFPField){"T_end_MeV", pd(cfg->T_end_MeV)};
    out[n++] = (CPRFPField){"nevo_file", ps(cfg->nevo_file)};
    out[n++] = (CPRFPField){"nevo_spectral_file", ps(cfg->nevo_spectral_file)};
    out[n++] = (CPRFPField){"nevo_file_prefix", ps(cfg->nevo_file_prefix)};
    return n; /* 16 entries; sampling_nTOp_per_decade/radiative_corrections/
                 finite_mass_corrections each appear once here already
                 (the Python dict's apparent "duplicate" assignment from
                 looping over _WEAK_RATE_BG_FIELDS after the literal dict
                 is a no-op re-write of the same key -- not represented
                 twice in a dict, so not duplicated here either). */
}

size_t cpr_thermal_fingerprint(const CPRConfig *cfg, CPRFPField *out)
{
    size_t n = 0;
    out[n++] = (CPRFPField){"format_version", pi(WEAK_RATE_FORMAT_VERSION)};
    out[n++] = (CPRFPField){"sampling_nTOp_thermal_per_decade", pi(cfg->sampling_nTOp_thermal_per_decade)};
    out[n++] = (CPRFPField){"T_end_MeV", pd(cfg->T_end_MeV)};
    out[n++] = (CPRFPField){"T_start_cosmo_MeV", pd(cfg->T_start_cosmo_MeV)};
    out[n++] = (CPRFPField){"QED_corrections", pb(cfg->QED_corrections)};
    out[n++] = (CPRFPField){"incomplete_decoupling", pb(cfg->incomplete_decoupling)};
    out[n++] = (CPRFPField){"nevo_file", ps(cfg->nevo_file)};
    out[n++] = (CPRFPField){"nevo_file_prefix", ps(cfg->nevo_file_prefix)};
    return n;
}

/* ===========================================================================
 * Cache file read/write (port of cache_utils.read_cache_fingerprint_hash /
 * write_cache_with_fingerprint).
 * ===========================================================================
 */
char *cpr_cache_read_fingerprint_hash(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    char line[2048];
    char *result = NULL;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '#') break;
        const char *prefix = "# fingerprint_hash:";
        if (strncmp(line, prefix, strlen(prefix)) == 0) {
            char *val = line + strlen(prefix);
            while (*val == ' ') val++;
            char *end = val + strlen(val);
            while (end > val && (end[-1] == '\n' || end[-1] == '\r' || end[-1] == ' '))
                end--;
            *end = '\0';
            result = strdup(val);
            break;
        }
    }
    fclose(f);
    return result;
}

int cpr_cache_write(const char *path, const CPRFPField *fields, size_t n_fields,
                     const char *col_header, double **columns, size_t n_cols,
                     size_t n_rows)
{
    char *json = cpr_fingerprint_json(fields, n_fields);
    char *hash = cpr_sha256_hex16(json);

    char tmp_path[4200];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", path, (int)getpid());

    FILE *f = fopen(tmp_path, "w");
    if (!f) { free(json); free(hash); return 1; }

    if (col_header && *col_header)
        fprintf(f, "# %s\n", col_header);
    fprintf(f, "# fingerprint_hash: %s\n", hash);
    fprintf(f, "# fingerprint: %s\n", json);
    for (size_t r = 0; r < n_rows; r++) {
        for (size_t c = 0; c < n_cols; c++) {
            if (c > 0) fputc(' ', f);
            fprintf(f, "%.18e", columns[c][r]);
        }
        fputc('\n', f);
    }
    fclose(f);
    free(json);
    free(hash);

    if (rename(tmp_path, path) != 0)
        return 1;
    return 0;
}
