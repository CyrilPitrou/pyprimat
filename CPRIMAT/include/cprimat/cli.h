/* cli.h -- `cprimat` executable entry point (port of pyprimat/cli.py).
 *
 * Phase 0 scope: parse argv into a fully validated CPRConfig and print it
 * back (or run --cache-info/--cache-clear). The actual BBN solve
 * (cprimat_run, mirroring PyPR(params).PyPRresults()) lands in a later
 * phase once background.c/nuclear_network.c exist; cpr_cli_main() already
 * has the hook point (see the TODO in cli.c) so wiring it in later needs no
 * argv-parsing changes.
 */
#ifndef CPRIMAT_CLI_H
#define CPRIMAT_CLI_H

int cpr_cli_main(int argc, char **argv);

#endif /* CPRIMAT_CLI_H */
