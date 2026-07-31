PROJECTS     = kuwahara kuwahara-torch kuwahara-numba kuwahara-jax
TURBINE     ?=
TORCH_EXTRAS = $(if $(TURBINE),--extra turbine,)
OUT_DIR     ?= $(PWD)/out
GENERATED    = $(OUT_DIR)/generated

.DEFAULT_GOAL := help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Benchmark targets:"
	@echo "  bench              Run benchmark sweep then plot"
	@echo "  bench-run          Run benchmark sweep only"
	@echo "  bench-plot         Plot latest benchmark results (timing + flops)"
	@echo "  bench-list         List all runs with timestamps and status"
	@echo ""
	@echo "  Variables:"
	@echo "    PROJECT=<dir>    Filter backends by project dir (e.g., kuwahara-torch)"
	@echo "    RESUME=<id>      Resume a previous run (omit id to resume latest)"
	@echo "    TURBINE=1        Include turbine_cpu variant (installs iree-turbine)"
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

$(GENERATED):
	mkdir -p $@

bench: sync
	uv run scripts/run_benchmarks.py \
		$(if $(PROJECT),--project $(PROJECT),) \
		$(if $(RESUME),--resume $(RESUME),)
	uv run scripts/plot_results.py --metric timing
	uv run scripts/plot_results.py --metric flops

bench-run: sync
	uv run scripts/run_benchmarks.py \
		$(if $(PROJECT),--project $(PROJECT),) \
		$(if $(RESUME),--resume $(RESUME),)

bench-plot: sync
	uv run scripts/plot_results.py --metric timing
	uv run scripts/plot_results.py --metric flops

bench-list:
	uv run scripts/plot_results.py list

export-torch: $(GENERATED) sync
	uv run --directory kuwahara-torch export-torch --kernel-size 35 --output $(GENERATED)/torch

export-jax: $(GENERATED) sync
	uv run --directory kuwahara-jax export-jax --kernel-size 35 --output $(GENERATED)/jax

export: export-torch export-jax

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

.PHONY: help bench bench-run bench-plot bench-list export export-torch export-jax list-backends list-inductor-backends format lint typecheck purge lock sync check
