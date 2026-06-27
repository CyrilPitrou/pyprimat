CC ?= cc
# -MMD/-MP emit a per-object .d file listing the headers it #includes, so a
# header edit (e.g. config.h, only #included transitively) correctly marks
# every .o that depends on it stale 
CFLAGS ?= -std=c11 -Wall -Wextra -O2 -Iinclude -MMD -MP -pthread -D_GNU_SOURCE
LDFLAGS ?= -lm -pthread

BUILD := build
SRC := $(wildcard src/*.c)
OBJ := $(patsubst src/%.c,$(BUILD)/%.o,$(SRC))

UNIT_SRC := $(wildcard tests/unit/*.c)
UNIT_BIN := $(patsubst tests/unit/%.c,$(BUILD)/unit/%,$(UNIT_SRC))

.PHONY: all debug test bench clean

all: $(BUILD)/cprimat

debug: CFLAGS := -std=c11 -Wall -Wextra -O0 -g -fsanitize=address,undefined -Iinclude -MMD -MP -pthread -D_GNU_SOURCE
debug: LDFLAGS := -lm -pthread -fsanitize=address,undefined
debug: all

$(BUILD)/%.o: src/%.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

-include $(OBJ:.o=.d)

$(BUILD)/cprimat: $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD)/unit/%: tests/unit/%.c $(filter-out $(BUILD)/main.o,$(OBJ))
	@mkdir -p $(BUILD)/unit
	$(CC) $(CFLAGS) $< $(filter-out $(BUILD)/main.o,$(OBJ)) -o $@ $(LDFLAGS)

# test_weak_rates_thermal.c #includes src/weak_rates.c directly (to reach
# its file-static L_CCRTh_compute), so it must not also link build/weak_rates.o
# (duplicate symbols) -- override the generic pattern rule above for this
# one binary.
$(BUILD)/unit/test_weak_rates_thermal: tests/unit/test_weak_rates_thermal.c $(filter-out $(BUILD)/main.o $(BUILD)/weak_rates.o,$(OBJ))
	@mkdir -p $(BUILD)/unit
	$(CC) $(CFLAGS) $< $(filter-out $(BUILD)/main.o $(BUILD)/weak_rates.o,$(OBJ)) -o $@ $(LDFLAGS)

test: $(UNIT_BIN)
	@for t in $(UNIT_BIN); do echo "== $$t =="; $$t || exit 1; done

# Times the two CLAUDE.md reference runs (small, large+amax=8) end-to-end
# and reports the ratio against the recorded Python baseline (CPLAN.md S12)
# -- examples/baseline_timings.txt, captured once via
# `time python runfiles/primat_run.py`-equivalent calls (see that file's
# header for how/when to refresh it). `--data-dir ../primat/data` matches
# every other CPRIMAT invocation in this repo (tests, examples/*.ini);
# `mkdir -p results` only matters on a fresh checkout (cpr_nuclear_network_
# write_final_result's own mkdir_p creates it too, but cfg->output_final_
# result defaults to False via the CLI path used here -- see cli.c).
bench: $(BUILD)/cprimat
	@mkdir -p results
	@echo "== CPRIMAT benchmark (small, large+amax=8) vs. Python baseline =="
	@echo "-- small --"; \
	 start=$$(date +%s.%N); \
	 ./$(BUILD)/cprimat --data-dir ../primat/data --ini examples/run_small.ini >/dev/null; \
	 end=$$(date +%s.%N); \
	 awk -v s=$$start -v e=$$end 'BEGIN { printf "cprimat: %.3f s\n", e - s }'
	@echo "-- large, amax=8 --"; \
	 start=$$(date +%s.%N); \
	 ./$(BUILD)/cprimat --data-dir ../primat/data --ini examples/run_large_amax8.ini >/dev/null; \
	 end=$$(date +%s.%N); \
	 awk -v s=$$start -v e=$$end 'BEGIN { printf "cprimat: %.3f s\n", e - s }'
	@echo "Python baseline timings (recorded once, see examples/baseline_timings.txt):"
	@cat examples/baseline_timings.txt

clean:
	rm -rf $(BUILD)
