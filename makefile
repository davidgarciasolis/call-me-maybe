SHELL := /bin/bash

PROJECT_DIR := $(CURDIR) SGOINFRE := $(HOME)/sgoinfre VENV_DIR := $(SGOINFRE)/venvs/call-me-maybe

export UV_CACHE_DIR := $(SGOINFRE)/uv-cache export UV_LINK_MODE := copy export UV_PROJECT_ENVIRONMENT := $(VENV_DIR) export HF_HOME := $(SGOINFRE)/huggingface_cache export HF_HUB_CACHE := $(HF_HOME)/hub export HF_DATASETS_CACHE := $(HF_HOME)/datasets export TMPDIR := $(SGOINFRE)/tmp

.PHONY: install run info

install: mkdir -p "(UVCACHEDIR)""(HF_HOME)" "(TMPDIR)""(SGOINFRE)/venvs" uv venv "$(VENV_DIR)" --python 3.10 uv sync --python "$(VENV_DIR)/bin/python"

run: uv run --python "$(VENV_DIR)/bin/python" python -u -m src

info: @echo "Project: $(PROJECT_DIR)" @echo "UV cache: $(UV_CACHE_DIR)" @echo "Hugging Face cache: $(HF_HOME)" @echo "Virtual environment: $(VENV_DIR)" @df -h "$(SGOINFRE)"