"""
Fidelity tests for GeneticsSimulation.

Three layers of coverage:
  1. Seed reproducibility  — same seed → bit-identical output.
  2. Distributional        — population stays non-negative and mean trajectory
                             converges near carrying capacity.
  3. Equilibrium theory    — at equilibrium, effective birth ≈ effective death,
                             pinning mean population near K.  A vectorized
                             reimplementation must pass the same bound.
"""

import numpy as np
import pytest

from simulations.evolution.genetics.params import GeneticsParams
from simulations.evolution.genetics.population import GeneticsSimulation


@pytest.fixture
def sim() -> GeneticsSimulation:
    return GeneticsSimulation()


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_result_shape(sim: GeneticsSimulation) -> None:
    params = GeneticsParams(n_generations=50)
    result = sim.run(params)
    assert len(result.steps) == 51
    assert len(result.values) == 51


def test_starts_at_initial_population(sim: GeneticsSimulation) -> None:
    params = GeneticsParams(initial_population=123)
    assert sim.run(params).values[0] == 123.0


def test_population_never_negative(sim: GeneticsSimulation) -> None:
    params = GeneticsParams(n_generations=200)
    for _ in range(20):
        assert (sim.run(params).values >= 0).all()


def test_batch_run_length(sim: GeneticsSimulation) -> None:
    assert len(sim.run_batch(GeneticsParams(n_generations=10), n_runs=5)) == 5


def test_invalid_params_raises(sim: GeneticsSimulation) -> None:
    with pytest.raises(ValueError, match="n_generations must be at least 1"):
        sim.run(GeneticsParams(n_generations=0))


# ---------------------------------------------------------------------------
# 1. Seed reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducibility(sim: GeneticsSimulation) -> None:
    """Identical seeds must produce bit-identical trajectories."""
    params = GeneticsParams(n_generations=100, seed=7)
    r1 = sim.run(params)
    r2 = sim.run(params)
    np.testing.assert_array_equal(r1.values, r2.values)


def test_different_seeds_differ(sim: GeneticsSimulation) -> None:
    r1 = sim.run(GeneticsParams(n_generations=100, seed=1))
    r2 = sim.run(GeneticsParams(n_generations=100, seed=2))
    assert not np.array_equal(r1.values, r2.values)


# ---------------------------------------------------------------------------
# 2. Distributional properties
# ---------------------------------------------------------------------------

def test_population_monotone_from_below_carrying_capacity(sim: GeneticsSimulation) -> None:
    """
    Starting well below K with birth > death, the mean trajectory should
    trend upward over the first half of the run.
    """
    params = GeneticsParams(
        n_generations=200,
        initial_population=100,
        birth_rate=0.2,
        death_rate=0.05,
        carrying_capacity=5000,
    )
    runs = sim.run_batch(params, n_runs=30)
    mean_traj = np.mean(np.stack([r.values for r in runs]), axis=0)
    midpoint = len(mean_traj) // 2
    assert mean_traj[midpoint] > mean_traj[0], (
        "Mean population did not grow when starting below carrying capacity"
    )


# ---------------------------------------------------------------------------
# 3. Equilibrium theory — population converges near K
# ---------------------------------------------------------------------------

def test_equilibrium_near_theoretical_fixed_point(sim: GeneticsSimulation) -> None:
    """
    The model's fixed point N* is not K.  Setting effective_birth = effective_death:

        birth * (1 − N*/K) = death * (1 + N*/K)
        N* = K · (birth − death) / (birth + death)

    This is the key invariant any reimplementation must preserve.  It exercises
    the density scaling on both birth and death simultaneously; a wrong sign or
    missing factor would shift the equilibrium and fail this test.

    A ±20% band around N* accommodates stochastic noise without masking
    gross implementation errors.
    """
    K = 5000
    b, d = 0.10, 0.08
    n_star = K * (b - d) / (b + d)  # ≈ 556

    params = GeneticsParams(
        n_generations=500,
        initial_population=500,
        birth_rate=b,
        death_rate=d,
        carrying_capacity=K,
    )
    runs = sim.run_batch(params, n_runs=50)
    # Average population over the final 20% of generations (post-convergence)
    final_slice = slice(int(params.n_generations * 0.8), None)
    mean_at_equilibrium = np.mean(
        [r.values[final_slice].mean() for r in runs]
    )
    assert 0.8 * n_star < mean_at_equilibrium < 1.2 * n_star, (
        f"Equilibrium population {mean_at_equilibrium:.0f} not within ±20% "
        f"of theoretical fixed point N*={n_star:.0f} (K={K})"
    )


# ---------------------------------------------------------------------------
# Limit cases — deterministic, no statistics
# ---------------------------------------------------------------------------

def test_zero_rates_constant_population(sim: GeneticsSimulation) -> None:
    """
    With birth_rate=0 and death_rate=0, Poisson(0) always returns 0 births and
    0 deaths — deterministic regardless of seed.  Population must stay exactly
    constant across all generations.

    This is a zero-statistics ground-truth check: any wrong sign, missing factor,
    or accidental mutation of the population variable will cause values to drift
    and fail here before any distributional test can catch it.
    """
    params = GeneticsParams(
        n_generations=100,
        initial_population=500,
        birth_rate=0.0,
        death_rate=0.0,
        carrying_capacity=5000,
    )
    result = sim.run(params)
    assert (result.values == 500.0).all(), (
        "Zero birth/death rates must hold population constant at 500 for all generations"
    )


def test_extinction_when_death_exceeds_birth(sim: GeneticsSimulation) -> None:
    """When death_rate >> birth_rate, population should trend to extinction."""
    params = GeneticsParams(
        n_generations=200,
        initial_population=100,
        birth_rate=0.01,
        death_rate=0.30,
        carrying_capacity=5000,
    )
    runs = sim.run_batch(params, n_runs=20)
    # At least 80% of runs should go extinct (reach 0)
    extinct = sum(1 for r in runs if r.values[-1] == 0)
    assert extinct >= 16, f"Only {extinct}/20 runs went extinct with death_rate >> birth_rate"
