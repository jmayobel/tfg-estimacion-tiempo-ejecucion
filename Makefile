.PHONY: help venv clean-venv install install-dev test coverage lint format clean azure-deploy azure-collect azure-download azure-stop azure-delete

VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Dataset Creator - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make venv          Create virtual environment in .venv"
	@echo "  make clean-venv    Remove virtual environment"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies (includes testing tools)"
	@echo "  make test          Run test suite"
	@echo "  make coverage      Run tests with coverage report"
	@echo "  make lint          Run code linting with flake8"
	@echo "  make format        Format code with black"
	@echo "  make clean         Remove generated files and caches"
	@echo ""
	@echo "Docker Execution:"
	@echo "  make run-docker    Run the orchestrator inside a Docker container"
	@echo ""
	@echo "Azure Deployment:"
	@echo "  make azure-deploy     Deploy Azure VMs (prompts for configuration)"
	@echo "  make azure-collect    Trigger dataset collection on remote hosts"
	@echo "  make azure-download   Download datasets from Azure VMs"
	@echo "  make azure-stop       Stop (deallocate) all Azure VMs"
	@echo "  make azure-delete     Delete Azure resource group (WARNING: destructive)"
	@echo ""

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

clean-venv:
	rm -rf $(VENV)

install: venv
	$(PIP) install -r scripts/requirements.txt

install-dev: venv
	$(PIP) install -r requirements-dev.txt

test: venv
	$(PYTHON) -m pytest tests/ -v

coverage: venv
	$(PYTHON) -m pytest tests/ --cov=scripts --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

lint: venv
	$(PYTHON) -m flake8 scripts/ tests/ --max-line-length=100 --ignore=E203,W503

format: venv
	$(PYTHON) -m black scripts/ tests/ --line-length=100

clean:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf scripts/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

run-docker:
	@echo "Running orchestrator in Docker..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		HOST_OS="macos" PWD="$$(pwd)" docker compose -p ds-orchestrator -f docker-compose.orchestrator.yml up --build; \
	elif grep -qi microsoft /proc/version 2>/dev/null; then \
		HOST_OS="wsl2" PWD="$$(pwd)" docker compose -p ds-orchestrator -f docker-compose.orchestrator.yml up --build; \
	else \
		HOST_OS="linux" PWD="$$(pwd)" docker compose -p ds-orchestrator -f docker-compose.orchestrator.yml up --build; \
	fi

# ============================================================================
# Azure Deployment Targets
# ============================================================================

azure-deploy:
	@echo "Deploying Azure VMs..."
	@echo "Make sure you've configured azure/parameters.json with your SSH public key"
	@read -p "Continue? (y/N) " confirm && [ "$$confirm" = "y" ] || exit 1
	cd azure && ./deploy.sh

azure-collect:
	@echo "Triggering dataset collection on remote hosts..."
	@if [ ! -f scripts/remote_hosts.txt ]; then \
		echo "ERROR: scripts/remote_hosts.txt not found"; \
		echo "Copy examples/remote_hosts.example.txt to scripts/remote_hosts.txt and configure it"; \
		exit 1; \
	fi
	./scripts/collect_remote.sh

azure-download:
	@echo "Downloading datasets from Azure VMs..."
	./scripts/download_datasets.sh

azure-stop:
	@echo "Stopping (deallocating) Azure VMs to save costs..."
	@rg_detected=$$(grep -E '^Resource Group:' azure/connection-info.txt 2>/dev/null | cut -d':' -f2 | xargs); \
	rg_env=$$(grep -E '^AZURE_RESOURCE_GROUP=' .env 2>/dev/null | cut -d'=' -f2 | xargs); \
	rg_default=$${rg_detected:-$${rg_env:-dataset-creator-rg}}; \
	read -p "Resource group name (default: $$rg_default): " rg; \
	rg=$${rg:-$$rg_default}; \
	echo "Checking VMs in resource group: $$rg..."; \
	vms=$$(az vm list -g "$$rg" --query "[].name" -o tsv); \
	if [ -z "$$vms" ]; then \
		echo "No VMs to deallocate or resource group '$$rg' does not exist."; \
	else \
		echo "Deallocating VMs: $$vms"; \
		echo "$$vms" | xargs -I {} az vm deallocate -g "$$rg" -n {} --no-wait; \
		echo "✓ Deallocation initiated in the background (this may take a few minutes)"; \
	fi

azure-delete:
	@echo "⚠️  WARNING: This will DELETE the entire Azure resource group!"
	@echo "    This action is IRREVERSIBLE and will delete all VMs, disks, and networks."
	@echo ""
	@rg_detected=$$(grep -E '^Resource Group:' azure/connection-info.txt 2>/dev/null | cut -d':' -f2 | xargs); \
	rg_env=$$(grep -E '^AZURE_RESOURCE_GROUP=' .env 2>/dev/null | cut -d'=' -f2 | xargs); \
	rg_default=$${rg_detected:-$${rg_env:-dataset-creator-rg}}; \
	read -p "Resource group name (default: $$rg_default): " rg; \
	rg=$${rg:-$$rg_default}; \
	echo "You are about to delete resource group: $$rg"; \
	read -p "Type 'DELETE' to confirm: " confirm; \
	if [ "$$confirm" = "DELETE" ]; then \
		az group delete -g "$$rg" --yes --no-wait; \
		echo "✓ Resource group deletion initiated"; \
	else \
		echo "Cancelled"; \
	fi
