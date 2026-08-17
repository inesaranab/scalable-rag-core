.PHONY: help install dev up down deploy infra test size weigh logs run-local

help:
	@echo "RAG Platform Commands:"
	@echo "  make install  - Install Python dependencies"
	@echo "  make dev      - Run FastAPI server locally"
	@echo "  make up       - Start local DBs (Docker, detached)"
	@echo "  make down     - Stop local DBs"
	@echo "  make logs     - Follow the local DBs' logs"
	@echo "  make run-local - Run the pipeline locally into Qdrant"
	@echo "  make infra    - Apply Terraform"
	@echo "  make deploy   - Deploy to Azure AKS via Helm"
	@echo "  make test     - Run the test suite"
	@echo "  make size     - Show what this project is costing in disk"
	@echo "  make weigh PKG=name - Size a dependency before installing it"

install:
	uv sync

# Local development environment
up:
	docker compose up -d

down:
	docker compose down

# Stream the containers' logs; Ctrl+C detaches without stopping them.
logs:
	docker compose logs -f

# The whole pipeline on this machine: real Ray, real Qdrant, fake embedder.
run-local:
	QDRANT_HOST=localhost PYTHONPATH=. uv run python scripts/run_local.py

# Run the API locally, reloading on change
dev:
	uv run uvicorn services.api.app.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env

# Infrastructure
infra:
	cd infra/terraform && terraform init && terraform apply

# Kubernetes deployment
deploy:
	helm dependency update deploy/helm/api
	helm upgrade --install api deploy/helm/api --namespace default
	helm upgrade --install ray-cluster kuberay/ray-cluster -f deploy/ray/ray-cluster.yaml

test:
	uv run pytest

# What a dependency would cost, before committing to it.
# Usage: make weigh PKG="unstructured[pdf]"
weigh:
	@test -n "$(PKG)" || (echo 'usage: make weigh PKG="package[extra]"' && exit 1)
	@uv run --with requests python scripts/weigh.py "$(PKG)"

# What this project occupies, and what the machine has left.
size:
	@echo "venv:    $$(du -sh .venv 2>/dev/null | cut -f1)"
	@echo "data:    $$(du -sh data 2>/dev/null | cut -f1)"
	@echo "docker:  $$(docker system df --format '{{.Size}}' 2>/dev/null | head -1)"
	@echo "free:    $$(df -h /System/Volumes/Data | awk 'NR==2{print $$4}')"
