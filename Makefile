.PHONY: help edit run-casino run-evolution run-ising run-pendulum run-sir run-collisions test build-site serve-site deploy-local

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
	@echo "  make run-pendulum"
	@echo "  make run-sir"
	@echo "  make run-collisions"
	@echo ""
	@echo "Local deployment (mirrors GitHub Pages):"
	@echo "  make build-site   - export all notebooks to _site/"
	@echo "  make serve-site   - serve _site/ on http://localhost:8080"
	@echo "  make deploy-local - build-site + serve-site"

edit:
	uv run marimo edit

run-casino:
	uv run marimo run notebooks/casino_comparison.py

run-evolution:
	uv run marimo run notebooks/evolution_exploration.py

run-ising:
	uv run marimo run notebooks/ising_exploration.py

run-pendulum:
	uv run marimo run notebooks/double_pendulum.py

run-sir:
	uv run marimo run notebooks/sir_epidemic.py

run-collisions:
	uv run marimo run notebooks/ball_collisions.py

test:
	uv run pytest

build-site:
	uv build --wheel
	mkdir -p _site/wheels
	cp dist/simulations-*.whl _site/wheels/
	for nb in notebooks/*.py; do \
		name=$$(basename $$nb .py); \
		uv run marimo export html-wasm $$nb -o _site/$$name/ --mode run -f; \
	done
	cp pages/index.html _site/index.html
	touch _site/.nojekyll

serve-site:
	@echo "Serving on http://localhost:8080"
	uv run python -m http.server 8080 --directory _site

deploy-local: build-site serve-site
