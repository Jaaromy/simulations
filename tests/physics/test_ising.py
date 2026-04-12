"""
Fidelity tests for IsingSimulation.

Three layers of coverage:
  1. Seed reproducibility  — same seed → bit-identical output.
  2. Limit cases           — deterministic bounds requiring no statistics.
  3. Theory cross-validation — Onsager exact solution within 5 SE.

Mathematical model
------------------
2D ferromagnetic Ising model on an L×L periodic lattice.

    Hamiltonian: H = −J Σ_{<ij>} s_i s_j,   J = 1,  k_B = 1

Each spin s_i ∈ {−1, +1}.  Metropolis dynamics at temperature T:
- Flip spin i if ΔE ≤ 0, else flip with probability exp(−ΔE/T).
- ΔE for flipping s_i: ΔE = 2J · s_i · Σ_{j∈nn(i)} s_j = 2 · s_i · neighbor_sum

Key analytical results
----------------------
Critical temperature (Onsager 1944):
    Tc = 2 / ln(1 + √2) ≈ 2.2692   (J = 1, k_B = 1)

Spontaneous magnetization below Tc:
    M_eq(T) = (1 − sinh(2/T)^{−4})^{1/8}

Energy per spin for a fully aligned grid (all s_i = +1, periodic L×L):
    Each site has 4 neighbors; each bond counted once in 2 directions → 2 bonds/site.
    E/N = −J · 2 = −2

Conserved sign under Metropolis:
    |M| ∈ [0, 1] always (magnetization bounded by construction).

ΔE range:
    neighbor_sum ∈ {−4, −2, 0, +2, +4}  →  ΔE ∈ {−8, −4, 0, +4, +8}.

SE derivation for theory cross-validation
------------------------------------------
At T = 1.5 (well below Tc ≈ 2.269):
    M_theory = (1 − sinh(2/1.5)^{−4})^{1/8} ≈ 0.984

Fluctuations of |M| after equilibration on an L×L lattice (L = 60):
    N = L² = 3600 spins
    Fraction up: p = (1 + M)/2 ≈ 0.992
    Single-spin variance: p(1−p) ≈ 0.00794
    Independent fluctuations at high ordering dominate; correlation length is
    finite below Tc, so blocks of size ≈ ξ² are correlated.

    Conservative estimate: SE ≈ 0.005 for M averaged over 500 equilibrated sweeps.
    Tolerance = 5 · SE = 0.025  (2.5 percentage points of magnetization).
"""

import numpy as np
import pytest

from simulations.physics.ising.params import IsingParams
from simulations.physics.ising.model import IsingSimulation, IsingResult


TC = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.2692


@pytest.fixture
def sim() -> IsingSimulation:
    return IsingSimulation()


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------

def test_result_type(sim: IsingSimulation) -> None:
    result = sim.run(IsingParams(grid_size=10, n_sweeps=10, n_snapshots=5))
    assert isinstance(result, IsingResult)


def test_magnetization_bounds(sim: IsingSimulation) -> None:
    result = sim.run(IsingParams(grid_size=20, n_sweeps=50, n_snapshots=10, seed=0))
    assert ((result.magnetization >= 0.0) & (result.magnetization <= 1.0)).all()


def test_final_grid_shape_and_values(sim: IsingSimulation) -> None:
    L = 15
    result = sim.run(IsingParams(grid_size=L, n_sweeps=5, n_snapshots=5))
    assert result.final_grid.shape == (L, L)
    assert set(np.unique(result.final_grid)).issubset({-1, 1})


def test_snapshot_count(sim: IsingSimulation) -> None:
    """n_snapshots controls how many magnetization readings are recorded."""
    result = sim.run(IsingParams(grid_size=10, n_sweeps=100, n_snapshots=20, seed=1))
    assert len(result.magnetization) == len(result.steps)
    assert len(result.magnetization) >= 2  # at least first and last


def test_steps_monotone(sim: IsingSimulation) -> None:
    result = sim.run(IsingParams(grid_size=10, n_sweeps=100, n_snapshots=10, seed=2))
    assert (np.diff(result.steps) > 0).all()


def test_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError, match="temperature"):
        IsingSimulation().run(IsingParams(temperature=0.0))


def test_invalid_grid_size_raises() -> None:
    with pytest.raises(ValueError, match="grid_size"):
        IsingSimulation().run(IsingParams(grid_size=1))


def test_invalid_initial_state_raises() -> None:
    with pytest.raises(ValueError, match="initial_state"):
        IsingSimulation().run(IsingParams(initial_state="diagonal"))


# ---------------------------------------------------------------------------
# 1. Seed reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducibility(sim: IsingSimulation) -> None:
    """Same seed must produce bit-identical magnetization arrays and final grids."""
    params = IsingParams(grid_size=30, n_sweeps=100, n_snapshots=20, seed=99)
    r1 = sim.run(params)
    r2 = sim.run(params)
    np.testing.assert_array_equal(r1.magnetization, r2.magnetization)
    np.testing.assert_array_equal(r1.final_grid, r2.final_grid)


def test_different_seeds_differ(sim: IsingSimulation) -> None:
    p1 = IsingParams(grid_size=20, n_sweeps=50, seed=1)
    p2 = IsingParams(grid_size=20, n_sweeps=50, seed=2)
    r1 = sim.run(p1)
    r2 = sim.run(p2)
    assert not np.array_equal(r1.final_grid, r2.final_grid)


# ---------------------------------------------------------------------------
# 2. Limit cases — deterministic, no statistics
# ---------------------------------------------------------------------------

def test_aligned_grid_stable_at_near_zero_temperature(sim: IsingSimulation) -> None:
    """
    A fully aligned grid at T ≈ 0:
    Every flip raises energy by ΔE = 8 (all 4 neighbors aligned).
    Accept prob = exp(−8/T) → 0 for T = 0.001.
    After 50 sweeps the grid must remain fully magnetised.

    Deterministic: no statistics, no tolerance.
    """
    params = IsingParams(
        grid_size=20,
        temperature=0.001,
        n_sweeps=50,
        n_snapshots=10,
        initial_state="aligned_up",
        seed=0,
    )
    result = sim.run(params)
    assert result.magnetization[-1] == 1.0, (
        f"Aligned grid at T=0.001 should stay fully magnetised; "
        f"got |M| = {result.magnetization[-1]}"
    )
    assert (result.final_grid == 1).all()


def test_aligned_down_gives_magnetization_one(sim: IsingSimulation) -> None:
    """|M| = 1 regardless of which direction is aligned."""
    params = IsingParams(
        grid_size=15,
        temperature=0.001,
        n_sweeps=20,
        n_snapshots=5,
        initial_state="aligned_down",
        seed=3,
    )
    result = sim.run(params)
    assert result.magnetization[-1] == 1.0


def test_energy_per_spin_fully_aligned() -> None:
    """
    Analytical result: fully aligned L×L periodic grid.
        Each bond contributes −1 (J=1, both spins +1).
        Bonds per site = 2 (each of 4 bonds shared between 2 sites).
        E/N = −2.

    Wrong sign, wrong factor, or double-counting all fail here immediately.
    """
    grid = np.ones((20, 20), dtype=np.int8)
    e_per_spin = IsingSimulation._energy_per_spin(grid)
    assert e_per_spin == -2.0, f"Expected E/N = −2.0 for aligned grid, got {e_per_spin}"


def test_energy_per_spin_antiferromagnetic() -> None:
    """
    Checkerboard (antiferromagnetic) pattern: all bonds connect opposite spins.
    Each bond contributes +1 → E/N = +2.
    """
    grid = np.fromfunction(lambda i, j: 1 - 2 * ((i + j) % 2), (20, 20), dtype=np.int8)
    e_per_spin = IsingSimulation._energy_per_spin(grid)
    assert e_per_spin == 2.0, f"Expected E/N = +2.0 for antiferromagnet, got {e_per_spin}"


def test_high_temperature_disordered(sim: IsingSimulation) -> None:
    """
    At T = 10 >> Tc ≈ 2.269: thermal noise overwhelms alignment.
    Starting from random, |M| must remain small (< 0.3) on average.
    No tight statistics needed — this is a gross sanity check.
    """
    params = IsingParams(
        grid_size=50,
        temperature=10.0,
        n_sweeps=500,
        n_snapshots=50,
        initial_state="random",
        seed=7,
    )
    result = sim.run(params)
    mean_mag = result.magnetization[len(result.magnetization) // 2 :].mean()
    assert mean_mag < 0.3, (
        f"At T=10 >> Tc, expected |M| < 0.3 but got {mean_mag:.3f}"
    )


def test_initial_magnetization_aligned_up(sim: IsingSimulation) -> None:
    """Snapshot at sweep 0 from aligned_up must show |M| = 1.0."""
    params = IsingParams(
        grid_size=20, temperature=2.0, n_sweeps=100,
        n_snapshots=10, initial_state="aligned_up", seed=0,
    )
    result = sim.run(params)
    assert result.steps[0] == 0
    assert result.magnetization[0] == 1.0


# ---------------------------------------------------------------------------
# Single-step trace — exact ΔE arithmetic
# ---------------------------------------------------------------------------

def test_single_step_delta_e_zero_entropy() -> None:
    """
    Manual trace: 5×5 all-+1 grid, T = 0.001 (deterministic: all flips rejected).

    Every spin has ns = 4 (4 aligned neighbours).
    ΔE = 2 · (+1) · 4 = +8 > 0; accept prob = exp(−8/0.001) ≈ 0.

    After one sweep, grid must be identical and |M| = 1.
    Any wrong sign in ΔE (e.g. −2 instead of +2) would make ΔE = −8 < 0,
    accepting every flip and immediately magnetising to 0.
    """
    params = IsingParams(
        grid_size=5,
        temperature=0.001,
        n_sweeps=1,
        n_snapshots=2,
        initial_state="aligned_up",
        seed=42,
    )
    result = IsingSimulation().run(params)
    assert (result.final_grid == 1).all(), (
        "All spins should remain +1 after one sweep at T=0.001"
    )


# ---------------------------------------------------------------------------
# 3. Theory cross-validation — Onsager exact solution within 5 SE
# ---------------------------------------------------------------------------

def test_onsager_magnetization_below_tc(sim: IsingSimulation) -> None:
    """
    Onsager (1952) exact spontaneous magnetization for T < Tc:

        M_eq(T) = (1 − sinh(2J / k_B T)^{−4})^{1/8},   J = 1, k_B = 1.

    At T = 1.5 (below Tc ≈ 2.269):
        sinh(2/1.5) = sinh(4/3) ≈ 1.6984
        1.6984^{−4} ≈ 0.1202
        M_theory = (1 − 0.1202)^{1/8} ≈ 0.984

    Protocol:
    - L = 60 grid, start from aligned_up (already in ordered phase).
    - Run 2000 sweeps; discard first 1000 for equilibration.
    - Average |M| over remaining 1000 sweeps.
    - Tolerance = 5 SE ≈ 0.025 (conservative; see module docstring).

    Expected value (M_theory) is derived from Onsager, NOT from running the sim.
    """
    T = 1.5
    M_theory = (1.0 - np.sinh(2.0 / T) ** (-4)) ** 0.125

    params = IsingParams(
        grid_size=60,
        temperature=T,
        n_sweeps=2000,
        n_snapshots=200,
        initial_state="aligned_up",
        seed=42,
    )
    result = sim.run(params)

    # Discard first half as equilibration
    eq_idx = len(result.magnetization) // 2
    mean_mag = result.magnetization[eq_idx:].mean()

    tolerance = 0.025  # 5 SE (see module docstring derivation)
    assert abs(mean_mag - M_theory) < tolerance, (
        f"Onsager predicts |M| = {M_theory:.4f} at T={T}; "
        f"measured {mean_mag:.4f} (deviation {abs(mean_mag - M_theory):.4f} > 5 SE = {tolerance})"
    )


def test_low_temperature_highly_ordered(sim: IsingSimulation) -> None:
    """
    At T = 0.5 << Tc: Onsager gives M ≈ 0.9999.
    Observed |M| must exceed 0.97 — a loose floor derived from theory.
    Starting from aligned_up so no equilibration needed.
    """
    T = 0.5
    M_theory = (1.0 - np.sinh(2.0 / T) ** (-4)) ** 0.125  # ≈ 0.9999

    params = IsingParams(
        grid_size=40,
        temperature=T,
        n_sweeps=500,
        n_snapshots=50,
        initial_state="aligned_up",
        seed=5,
    )
    result = sim.run(params)
    mean_mag = result.magnetization[len(result.magnetization) // 2 :].mean()

    assert mean_mag > M_theory - 0.03, (
        f"T={T}: Onsager M_theory={M_theory:.4f}, observed {mean_mag:.4f}"
    )
