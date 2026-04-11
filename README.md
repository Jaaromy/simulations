# Simulations

A Python framework for interactive simulations with parameterized inputs and live charts. Built with [Marimo](https://marimo.io) for the UI and [Plotly](https://plotly.com/python/) for visualization.

## Simulations

| Domain | Simulation | Description |
|---|---|---|
| Casino | Blackjack | Bankroll over N hands with optional basic strategy |
| Casino | Roulette | Bankroll over N spins with configurable bet type and wheel |
| Casino | Slots | Bankroll over N spins with configurable RTP |
| Evolution | Genetics | Logistic population growth with stochastic birth/death |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra ui --extra dev
```

## Running notebooks

```bash
# Launch as a clean interactive app
uv run marimo run notebooks/casino_comparison.py
uv run marimo run notebooks/evolution_exploration.py

# Open in edit/development mode
uv run marimo edit notebooks/casino_comparison.py
```

## Running tests

```bash
uv run pytest
```

## Project structure

```
src/simulations/
├── core/           # Simulation ABC, params protocol, result types
├── casino/         # blackjack · roulette · slots
└── evolution/      # genetics (logistic population growth)

notebooks/          # Marimo interactive notebooks (.py)
tests/              # Headless pytest tests (no UI dependencies)
```

## Adding a new simulation

1. Create `src/simulations/<domain>/<name>/params.py` — frozen dataclass with `validate()` and `to_dict()`
2. Create `src/simulations/<domain>/<name>/game.py` — subclass `Simulation[YourParams, TimeSeriesResult]`
3. Add a notebook in `notebooks/` following the params cell → simulation cell → chart cell pattern
4. Add tests in `tests/<domain>/`

## License

Public domain (Unlicense).
