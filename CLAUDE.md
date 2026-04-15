# Simulations

## Setup

```bash
uv sync --extra ui --extra dev   # IMPORTANT: bare `uv sync` omits marimo and pytest
```

## Commands

- `uv run pytest` — run tests
- `uv run marimo run notebooks/<name>.py` — launch notebook as a clean app UI
- `uv run marimo edit notebooks/<name>.py` — open notebook in edit/development mode

## Marimo notebooks

- All imports (`numpy`, `plotly`, etc.) must live in a **single cell** and be returned so other cells can receive them as arguments. NEVER import the same name in two different cells — Marimo requires all variable names to be unique across cells.
- Variable names must be unique across all cells. Use the `_` prefix for cell-local variables to avoid cross-cell conflicts.
- **Never claim a notebook is complete without running `uv run marimo run notebooks/<name>.py` and confirming it starts without errors.**

## Plotting performance

Only plot the points needed to accurately represent the data — no more.

- Downsample long time series before plotting. A trajectory with 10,000 steps does not need 10,000 points on screen; sample to the resolution the chart can actually display (e.g. 500–1,000 points for a typical line chart).
- Preserve shape-critical points when downsampling: keep local minima/maxima and the first/last point so the visual remains faithful to the underlying data.
- Never pass raw simulation output arrays directly to Plotly without considering their length first.

## Architecture

- Simulation logic has **zero UI imports** — UI imports sims, never the reverse. This keeps all sims headlessly testable.
- Notebooks live in `notebooks/` as plain `.py` Marimo files, not `.ipynb`.
- NEVER use `@st.cache_data` or Streamlit — the UI framework is Marimo.
- New simulation domains go under `src/simulations/<domain>/` and must subclass `Simulation[P, R]` from `core/base.py`.

## Simulation accuracy — top priority

Correctness of the mathematical model is the highest priority. A fast, clean, well-structured simulation that produces wrong numbers is worthless.

### Theory-first workflow (required for every new simulation)

Before writing any simulation code:

1. Write down the mathematical model — identify expected value per step, conserved quantities, equilibrium points, and limit behavior.
2. Derive key invariants analytically (closed-form formulas, not measured from a run).
3. Write failing test stubs that assert against those derivations.
4. Implement the simulation.
5. Tests pass organically — or reveal bugs.

Never write tests after the fact by fitting assertions to observed output.

### Three required test layers for every simulation

**1. Seed reproducibility** — same seed → bit-identical output across runs.

**2. Limit cases (deterministic)** — set parameters to extremes where the answer is exactly known with no statistics:
- Zero-rate / zero-probability inputs (e.g., `birth_rate=0` → constant population, `p_win=0` → exact loss every step)
- Always-win inputs (mock RNG to force win branch → verify exact payout arithmetic)
- These catch wrong signs, off-by-one in payouts, and missing factors before any statistical test can.

**3. Theory cross-validation (statistical)** — run a large batch and assert the observed mean is within a tight band (≤ 5 SE) of the closed-form theoretical value:
- House edge / expected value per step
- Equilibrium / fixed point (e.g., `N* = K*(b-d)/(b+d)` for logistic growth)
- The tolerance must be derived from the actual per-step variance, not chosen arbitrarily.

### What makes a theory test an accuracy anchor

A theory test is only as good as the independence of its expected value from the implementation. The expected value must come from:
- Published math (standard probability, physics, biology)
- Independent derivation documented in the test file
- A separate reference implementation

Never derive the "expected" value by running the simulation and recording the output.

### Single-step trace tests for complex logic

For simulations with non-trivial per-step logic (card games, multi-branch decisions):
- Construct a minimal controlled input (fixed deck order, mock RNG returning index 0)
- Manually trace the exact outcome on paper
- Assert the exact net result — no tolerance bands

These catch subtle logic bugs (wrong soft-hand rule, missing double-down, incorrect payout multiplier) that aggregate statistical tests will never surface.

### Real-time / JavaScript simulations

Simulations requiring live browser animation (e.g. pendulums, collisions) may implement their physics in JavaScript rather than Python. The accuracy-first rule is unchanged — all three required test layers (seed reproducibility, limit cases, theory cross-validation) must still be enforced, and JS tests live under `tests_js/` and run via `make test-js`. Where a trusted Python reference exists (e.g. `src/simulations/physics/pendulum` for N=2), JS output must be cross-validated against it to ≤1e-4 relative error over a fixed trajectory.
