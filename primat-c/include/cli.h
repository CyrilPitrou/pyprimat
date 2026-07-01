/* cli.h -- `cprimat` executable entry point (port of primat/cli.py).
 *
 * cpr_cli_main() parses argv into a fully validated CPRConfig, then either
 * services a standalone subcommand (--cache-info/--cache-clear) or runs the
 * full BBN solve via cprimat_run (api.h), mirroring Python's
 * `PRIMAT(params).primat_results()`, and prints/writes the results.
 */
#ifndef CPRIMAT_CLI_H
#define CPRIMAT_CLI_H

int cpr_cli_main(int argc, char **argv);

#endif /* CPRIMAT_CLI_H */
