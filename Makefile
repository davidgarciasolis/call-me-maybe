SHELL := /bin/bash
.DEFAULT_GOAL := install

SGOINFRE ?= $(HOME)/sgoinfre
VENV_DIR ?= $(SGOINFRE)/venvs/call-me-maybe
PYTHON ?= python3.10
THREADS ?= 2

export OMP_NUM_THREADS := $(THREADS)
export MKL_NUM_THREADS := $(THREADS)
export UV_CACHE_DIR ?= $(SGOINFRE)/uv-cache
export UV_LINK_MODE := copy
export UV_PROJECT_ENVIRONMENT := $(VENV_DIR)
export HF_HOME ?= $(SGOINFRE)/huggingface_cache
export HF_HUB_CACHE ?= $(HF_HOME)/hub
export HF_DATASETS_CACHE ?= $(HF_HOME)/datasets
export TMPDIR := $(SGOINFRE)/tmp

.PHONY: install run debug clean lint prepare

prepare:
	mkdir -p "$(UV_CACHE_DIR)" "$(HF_HUB_CACHE)" "$(HF_DATASETS_CACHE)" \
		"$(TMPDIR)" "$(dir $(VENV_DIR))"

install: prepare
	uv sync --python "$(PYTHON)"

run: prepare
	uv run --python python3.10 python -u -m src \
		--functions_definition "data/input/functions_definition.json" \
		--input "data/input/function_calling_tests.json" \
		--output "data/output/function_calling_results.json"

debug: prepare
	uv run --python "$(PYTHON)" python -m pdb -m src \
		--functions_definition "data/input/functions_definition.json" \
		--input "data/input/function_calling_tests.json" \
		--output "data/output/function_calling_results.json"

clean:
	find src llm_sdk -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache

lint: prepare
	uv run --python "$(PYTHON)" flake8 src
	uv run --python "$(PYTHON)" mypy src --warn-return-any \
		--warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs --follow-imports=skip
