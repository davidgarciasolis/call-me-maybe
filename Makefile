SHELL := /bin/bash
.DEFAULT_GOAL := install

PROJECT_DIR := $(CURDIR)
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

.PHONY: install run debug clean lint info prepare

prepare:
	mkdir -p "$(UV_CACHE_DIR)" "$(HF_HUB_CACHE)" "$(HF_DATASETS_CACHE)" \
		"$(TMPDIR)" "$(dir $(VENV_DIR))"

install: prepare
	uv sync --python "$(PYTHON)"

run: prepare
	uv run --python "$(PYTHON)" python -u -m src

debug: prepare
	uv run --python "$(PYTHON)" python -m pdb -m src

clean:
	find src llm_sdk -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache

lint: prepare
	uv run --python "$(PYTHON)" flake8 .
	uv run --python "$(PYTHON)" mypy . --warn-return-any \
		--warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

info:
	@echo "Project: $(PROJECT_DIR)"
	@echo "Storage: $(SGOINFRE)"
	@echo "UV cache: $(UV_CACHE_DIR)"
	@echo "Hugging Face cache: $(HF_HOME)"
	@echo "Hugging Face Hub cache: $(HF_HUB_CACHE)"
	@echo "Virtual environment: $(UV_PROJECT_ENVIRONMENT)"
	@echo "Temporary files: $(TMPDIR)"
	@df -h "$(SGOINFRE)"
