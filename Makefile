.PHONY: help edit run-casino run-logistic run-ising run-pendulum run-sir run-collisions run-n-pendulum run-planetary-gravity test test-js build-site serve-site deploy-local

help:
	@echo "Usage:"
	@echo "  make          - open notebook browser (pick any simulation)"
	@echo "  make edit     - same as above"
	@echo "  make test     - run all tests"
	@echo ""
	@echo "Run as read-only app:"
	@echo "  make run-casino"
	@echo "  make run-logistic"
	@echo "  make run-ising"
	@echo "  make run-pendulum"
	@echo "  make run-n-pendulum"
	@echo "  make run-planetary-gravity"
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

run-logistic:
	uv run marimo run notebooks/logistic_population_growth.py

run-ising:
	uv run marimo run notebooks/ising_exploration.py

run-pendulum:
	uv run marimo run notebooks/double_pendulum.py

run-sir:
	uv run marimo run notebooks/sir_epidemic.py

run-collisions:
	uv run marimo run notebooks/ball_collisions.py

run-n-pendulum:
	uv run marimo run notebooks/n_pendulum.py

run-planetary-gravity:
	uv run marimo run notebooks/planetary_gravity.py

test:
	uv run pytest
	$(MAKE) test-js

test-js:
	node --test tests_js/*.test.mjs

build-site:
	uv build --wheel
	mkdir -p _site/wheels
	cp dist/simulations-*.whl _site/wheels/
	for nb in notebooks/*.py; do \
		name=$$(basename $$nb .py); \
		uv run marimo export html-wasm $$nb -o _site/$$name/ --mode run -f; \
		uv run python3 -c "f='_site/$$name/index.html'; b=open('pages/back_link.html').read(); c=open(f).read(); open(f,'w').write(c.replace('<body>','<body>'+b,1))"; \
	done
	cp pages/index.html _site/index.html
	touch _site/.nojekyll

serve-site:
	@echo "Serving on http://localhost:8080"
	uv run python -m http.server 8080 --directory _site

deploy-local: build-site serve-site
