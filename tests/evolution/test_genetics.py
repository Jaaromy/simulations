"""
Fidelity tests for GeneticsSimulation.

Three layers of coverage:
  1. Seed reproducibility  — same seed → bit-identical output.
  2. Limit cases           — deterministic bounds that require no statistics.
  3. Theory cross-validation — observed mean within 5 SE of closed-form N*.

SE derivation for theory test
------------------------------
The fixed point of the logistic model:

    N* = K · (b − d) / (b + d)

At N*, both effective rates equal 2bd/(b+d), so per-step variance of ΔN is:

    Var(ΔN) = (eff_birth + eff_death) · N*
            = 4bd/(b+d) · K(b−d)/(b+d)
            = 4bdK(b−d) / (b+d)²

The stationary variance of N around N* (Ornstein-Uhlenbeck approximation):

    Var_stat ≈ Var(ΔN) / (2α),  where α = b − d  (restoring rate)
             = 2bdK / (b+d)²

Correlation time τ ≈ 1/α steps.  Effective independent samples from n_runs
runs each averaged over a window of w steps:

    n_eff = n_runs · w / τ = n_runs · w · (b − d)

SE of grand mean = sqrt(Var_stat / n_eff).
"""

import unittest.mock as mock

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
# 3. Equilibrium theory — population converges near N* within 5 SE
# ---------------------------------------------------------------------------

def test_equilibrium_near_theoretical_fixed_point(sim: GeneticsSimulation) -> None:
    """
    Fixed point: N* = K·(b−d)/(b+d) ≈ 556 for the parameters below.

    Tolerance is derived analytically — not chosen arbitrarily:
        Var_stat  = 2·b·d·K / (b+d)²         (stationary variance, O-U approx)
        τ         = 1/(b−d)                    (correlation time in steps)
        n_eff     = n_runs · window · (b−d)    (effective independent samples)
        SE        = sqrt(Var_stat / n_eff)
        tolerance = 5 · SE

    With the parameters below: SE ≈ 5.0, tolerance ≈ 25 (≈ ±4.5% of N*).
    """
    K = 5000
    b, d = 0.10, 0.08
    n_runs = 50
    n_generations = 500
    n_star = K * (b - d) / (b + d)  # = 5000·0.02/0.18 ≈ 555.6

    # Analytically derived SE (see module docstring for full derivation)
    var_stat = 2 * b * d * K / (b + d) ** 2      # ≈ 2469
    window = int(n_generations * 0.2)             # = 100 steps
    tau = 1.0 / (b - d)                           # = 50 steps
    n_eff = n_runs * window / tau                  # = 100 effective samples
    se = np.sqrt(var_stat / n_eff)                 # ≈ 4.97
    tolerance = 5 * se                             # ≈ 24.9

    params = GeneticsParams(
        n_generations=n_generations,
        initial_population=500,
        birth_rate=b,
        death_rate=d,
        carrying_capacity=K,
    )
    runs = sim.run_batch(params, n_runs=n_runs)
    final_slice = slice(int(n_generations * 0.8), None)
    mean_at_equilibrium = np.mean(
        [r.values[final_slice].mean() for r in runs]
    )
    assert abs(mean_at_equilibrium - n_star) < tolerance, (
        f"Equilibrium mean {mean_at_equilibrium:.1f} deviates from N*={n_star:.1f} "
        f"by more than 5 SE ({tolerance:.1f})"
    )


# ---------------------------------------------------------------------------
# Limit cases — deterministic, no statistics
# ---------------------------------------------------------------------------

def test_single_step_exact_arithmetic() -> None:
    """
    Verify exact per-step arithmetic with a mocked RNG.

    Setup: population=500, K=5000, birth_rate=0.10, death_rate=0.08

    Manual derivation:
        density          = 500 / 5000 = 0.10
        effective_birth  = 0.10 × (1 − 0.10) = 0.09
        effective_death  = 0.08 × (1 + 0.10) = 0.088
        λ_births         = 0.09 × 500 = 45.0
        λ_deaths         = 0.088 × 500 = 44.0
        births (mocked)  = 40
        deaths (mocked)  = 30
        new population   = max(500 + 40 − 30, 0) = 510

    Wrong signs, missing density factors, or swapped birth/death assignments
    all shift λ_births or λ_deaths and cause this test to fail before any
    statistical test can surface the error.
    """
    sim = GeneticsSimulation()
    params = GeneticsParams(
        n_generations=1,
        initial_population=500,
        birth_rate=0.10,
        death_rate=0.08,
        carrying_capacity=5000,
    )

    mock_rng = mock.MagicMock()
    mock_rng.poisson.side_effect = [40, 30]  # first call: births; second: deaths

    with mock.patch(
        "simulations.evolution.genetics.population.np.random.default_rng",
        return_value=mock_rng,
    ):
        result = sim.run(params)

    assert result.values[0] == 500.0
    assert result.values[1] == 510.0, (
        f"Expected 510 (500+40−30) but got {result.values[1]}"
    )
    # Confirm the correct λ values were passed to the RNG (float-exact comparison)
    calls = [c.args[0] for c in mock_rng.poisson.call_args_list]
    np.testing.assert_allclose(calls, [45.0, 44.0], rtol=1e-12, err_msg=(
        f"Expected Poisson calls [45.0, 44.0] but got {calls}"
    ))


def test_zero_birth_rate_population_monotonically_nonincreasing(sim: GeneticsSimulation) -> None:
    """
    With birth_rate=0, effective_birth=0 exactly, so rng.poisson(0)=0 always.
    Births are deterministically zero every step; population can only decrease.

    Asserts the strict invariant: values[i+1] ≤ values[i] for all i.
    This is a zero-statistics deterministic bound — no distributional tolerance.
    A bug that accidentally adds births when birth_rate=0 breaks this immediately.
    """
    params = GeneticsParams(
        n_generations=50,
        initial_population=200,
        birth_rate=0.0,
        death_rate=0.10,
        carrying_capacity=5000,
        seed=0,
    )
    result = sim.run(params)
    diffs = np.diff(result.values)
    assert (diffs <= 0).all(), (
        f"Population increased on steps {np.where(diffs > 0)[0].tolist()} "
        "despite birth_rate=0"
    )


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
