PROJECTS     = kuwahara kuwahara-torch kuwahara-numba kuwahara-jax
IMG         ?= $(PWD)/assets/cowboy.jpg
TURBINE     ?=
TORCH_EXTRAS = $(if $(TURBINE),--extra turbine,)
KERNEL_SIZES ?= 3,5,7,9,11,15,21,31
OUT_DIR     ?= $(PWD)/out
GENERATED    = $(OUT_DIR)/generated
RUNS_DIR     = $(OUT_DIR)/runs
IMAGES_DIR   = $(RUNS_DIR)/images

.DEFAULT_GOAL := help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Benchmark targets:"
	@echo "  bench              Run all benchmarks (torch eager + compile, numba, jax)"
	@echo "  bench-torch        Run PyTorch eager / compile benchmark"
	@echo "  bench-torch-eager  Run PyTorch eager benchmark"
	@echo "  bench-torch-compile Run PyTorch compile benchmark (inductor)"
	@echo "  bench-torch-turbine-cpu Run PyTorch compile benchmark (turbine_cpu)"
	@echo "  bench-numba        Run Numba benchmark"
	@echo "  bench-jax          Run JAX benchmark"
	@echo "  bench-sweep        Sweep kernel sizes and save results"
	@echo "  bench-plot         Plot benchmark results"
	@echo "  bench-report       Sweep + plot"
	@echo ""
	@echo "  Set TURBINE=1 to include turbine_cpu variant (installs iree-turbine)"
	@echo ""
	@echo "Export targets:"
	@echo "  export         Export all backends (torch + jax)"
	@echo "  export-torch   Export PyTorch model via torch.export"
	@echo "  export-jax     Export JAX model via jax.export"
	@echo ""
	@echo "Query targets:"
	@echo "  list-backends         List all backends and variants"
	@echo "  list-inductor-backends List torch backend variants"
	@echo ""
	@echo "Development targets:"
	@echo "  format         Auto-format all sources"
	@echo "  lint           Lint all sources"
	@echo "  typecheck      Type-check all sources"
	@echo "  check          Format + lint + typecheck"

$(GENERATED) $(IMAGES_DIR):
	mkdir -p $@

bench-torch-eager: $(IMAGES_DIR) sync
	uv run --directory kuwahara-torch bench-torch $(IMG) --variant eager --save-output-dir $(IMAGES_DIR)

bench-torch-compile: $(IMAGES_DIR) sync
	uv run --directory kuwahara-torch bench-torch $(IMG) --variant inductor --save-output-dir $(IMAGES_DIR)

bench-torch-turbine-cpu: $(IMAGES_DIR) sync
	uv run --directory kuwahara-torch bench-torch $(IMG) --variant turbine_cpu --save-output-dir $(IMAGES_DIR)

bench-torch: bench-torch-eager bench-torch-compile $(if $(TURBINE),bench-torch-turbine-cpu)

bench-numba: sync
	uv run --directory kuwahara-numba bench-numba $(IMG)

bench-jax: sync
	uv run --directory kuwahara-jax bench-jax $(IMG)

bench: bench-torch bench-numba bench-jax

export-torch: $(GENERATED) sync
	uv run --directory kuwahara-torch export-torch --kernel-size 35 --output $(GENERATED)/torch

export-jax: $(GENERATED) sync
	uv run --directory kuwahara-jax export-jax --kernel-size 35 --output $(GENERATED)/jax

export: export-torch export-jax

bench-sweep: sync
	uv run scripts/run_benchmarks.py

bench-plot: sync
	uv run scripts/plot_results.py --metric timing
	uv run scripts/plot_results.py --metric flops

bench-list:
	uv run scripts/plot_results.py list

bench-resume:
	uv run scripts/run_benchmarks.py --resume

bench-report: bench-sweep bench-plot

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

list-inductor-backends: sync
	uv run --directory kuwahara-torch list-inductor-backends

list-backends: sync
	@echo "=== kuwahara-torch ==="
	uv run --directory kuwahara-torch list-inductor-backends
	@echo ""
	@echo "=== kuwahara-jax ==="
	uv run --directory kuwahara-jax bench-jax --list-variants
	@echo ""
	@echo "=== kuwahara-numba ==="
	uv run --directory kuwahara-numba bench-numba --list-variants

format:
	@for p in $(PROJECTS); do \
		uv run --directory $$p ruff format src; \
	done

lint:
	@for p in $(PROJECTS); do \
		uv run --directory $$p ruff check$(if $(FIX), --fix,) src; \
	done

typecheck:
	@for p in $(PROJECTS); do \
		uv run --directory $$p ty check src; \
	done

purge:
	@for p in $(PROJECTS); do \
		rm -rf $$p/.venv; \
	done

lock:
	@for p in $(PROJECTS); do \
		uv lock --directory $$p; \
	done

sync:
	uv sync --directory kuwahara-torch $(TORCH_EXTRAS)
	uv sync --directory kuwahara-numba
	uv sync --directory kuwahara-jax

check: format lint typecheck

.PHONY: help bench bench-torch bench-torch-eager bench-torch-compile bench-torch-turbine-cpu bench-numba bench-jax bench-sweep bench-plot bench-report export export-torch export-jax list-backends list-inductor-backends format lint typecheck purge lock sync check
