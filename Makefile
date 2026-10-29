# OmniVoice TTS server — common tasks.
#
#   make build          # production image (frontend baked in)
#   make run            # serve on GPU 0 -> http://localhost:9001
#   make status         # health, queue and GPU state
#   make logs           # follow container logs
#   make bench          # quick latency/throughput check against a running server
#   make dev-up         # hot-reload dev stack (API + Vite)
#   make test           # unit tests (no GPU needed)
#   make lint           # ruff

SHELL          := /bin/bash
COMPOSE        := docker compose
COMPOSE_PROD   := docker compose -f docker-compose.prod.yml
IMAGE          ?= omnivoice-tts:prod
PORT           ?= 9001
URL            ?= http://localhost:${PORT}
GPU            ?= 0

.DEFAULT_GOAL := help
.PHONY: help build run stop restart logs status shell dev-up dev-down test lint fmt \
        bench bench-load bench-stream bench-quality voices prepare-voices clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## Build the production image (UI + API in one container)
	$(COMPOSE_PROD) build

run: ## Start the production container on GPU $(GPU)
	# --force-recreate: a rebuilt :prod image must replace the running container
	GPU=$(GPU) $(COMPOSE_PROD) up -d --force-recreate
	@echo "waiting for the model..."
	@for i in $$(seq 1 60); do curl -sf $(URL)/readyz >/dev/null && break; sleep 2; done
	@curl -s $(URL)/health | head -c 400; echo

stop: ## Stop the production container
	$(COMPOSE_PROD) down

restart: stop run ## Restart the production container

logs: ## Follow container logs
	$(COMPOSE_PROD) logs -f

status: ## Show health, queue, GPU and voice state
	@curl -s $(URL)/health | python3 -m json.tool || echo "server not reachable at $(URL)"
	@nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

shell: ## Shell inside the running container
	$(COMPOSE_PROD) exec server bash

dev-up: ## Start the hot-reload dev stack (API :9001, Vite :5173)
	$(COMPOSE) up -d --build

dev-down: ## Stop the dev stack
	$(COMPOSE) down

test: ## Run the unit test suite (no GPU required)
	cd server && .venv/bin/python -m pytest -q

lint: ## Lint the server code
	cd server && .venv/bin/ruff check app tests ../scripts ../bench

fmt: ## Auto-fix lint findings
	cd server && .venv/bin/ruff check --fix app tests ../scripts ../bench

bench: ## End-to-end latency/throughput check (full + SSE + WS)
	server/.venv/bin/python bench/load_test.py --url $(URL) --mode full --concurrency 1 --requests 8
	server/.venv/bin/python bench/load_test.py --url $(URL) --mode sse --concurrency 1 --requests 8
	server/.venv/bin/python bench/load_test.py --url $(URL) --mode ws --concurrency 1 --requests 8

plan: ## Measure this machine and print the env values it should run with
	server/.venv/bin/python scripts/plan.py --url $(URL)

probe: ## Batching probe: is this GPU saturated at batch size 1? (stops the container)
	-$(COMPOSE_PROD) stop
	server/.venv/bin/python bench/batch_probe.py
	-$(COMPOSE_PROD) start

bench-load: ## Concurrency sweep (writes bench/results/)
	bash bench/run_all.sh $(URL)

voices: ## List registered voices
	@curl -s $(URL)/v1/voices | python3 -m json.tool

prepare-voices: ## Rebuild the built-in voice references (needs the ASR service)
	server/.venv/bin/python scripts/prepare_voices.py

clean:
	$(COMPOSE_PROD) down --rmi local -v --remove-orphans || true
