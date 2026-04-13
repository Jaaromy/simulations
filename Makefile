.PHONY: help edit run-casino run-evolution run-ising test

help:
	@echo "Usage:"
	@echo "  make          - open notebook browser (pick any simulation)"
	@echo "  make edit     - same as above"
	@echo "  make test     - run all tests"
	@echo ""
	@echo "Run as read-only app:"
	@echo "  make run-casino"
	@echo "  make run-evolution"
	@echo "  make run-ising"

edit:
	uv run marimo edit

run-casino:
	uv run marimo run notebooks/casino_comparison.py

run-evolution:
	uv run marimo run notebooks/evolution_exploration.py

run-ising:
	uv run marimo run notebooks/ising_exploration.py

test:
	uv run pytest
