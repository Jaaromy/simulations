"""
Fidelity tests for RouletteSimulation.

Three layers of coverage:
  1. Seed reproducibility  — same seed → bit-identical output.
                             When a vectorized implementation is added, run it
                             with the same seed and assert array equality against
                             the scalar result to confirm exact equivalence.
  2. Distributional        — run a large batch and verify the mean, std, and
                             bust rate are within statistically tight bounds.
  3. House-edge theory     — cross-validate the per-spin expected value against
                             the closed-form theoretical value for each bet type.
"""

import numpy as np
import pytest
import simulations.casino.roulette.game as _roulette_game

from simulations.casino.roulette.game import RouletteSimulation
from simulations.casino.roulette.params import RouletteParams


@pytest.fixture
def sim() -> RouletteSimulation:
    return RouletteSimulation()


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_result_shape(sim: RouletteSimulation) -> None:
    params = RouletteParams(n_spins=50)
    result = sim.run(params)
    assert len(result.steps) == 51
    assert len(result.values) == 51


def test_starts_at_initial_bankroll(sim: RouletteSimulation) -> None:
    params = RouletteParams(initial_bankroll=750.0)
    assert sim.run(params).values[0] == 750.0


def test_bankroll_never_negative(sim: RouletteSimulation) -> None:
    params = RouletteParams(n_spins=500, initial_bankroll=500.0, bet_size=10.0)
    for _ in range(20):
        assert (sim.run(params).values >= 0).all()


def test_batch_run_length(sim: RouletteSimulation) -> None:
    assert len(sim.run_batch(RouletteParams(n_spins=10), n_runs=7)) == 7


def test_invalid_params_raises(sim: RouletteSimulation) -> None:
    with pytest.raises(ValueError, match="bet_size cannot exceed initial_bankroll"):
        sim.run(RouletteParams(initial_bankroll=50.0, bet_size=100.0))


# ---------------------------------------------------------------------------
# 1. Seed reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducibility(sim: RouletteSimulation) -> None:
    """Identical seeds must produce bit-identical trajectories."""
    params = RouletteParams(n_spins=200, seed=42)
    r1 = sim.run(params)
    r2 = sim.run(params)
    np.testing.assert_array_equal(r1.values, r2.values)


def test_different_seeds_differ(sim: RouletteSimulation) -> None:
    """Different seeds should (overwhelmingly) produce different trajectories."""
    r1 = sim.run(RouletteParams(n_spins=200, seed=1))
    r2 = sim.run(RouletteParams(n_spins=200, seed=2))
    assert not np.array_equal(r1.values, r2.values)


# ---------------------------------------------------------------------------
# 2. Distributional properties
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Limit cases — deterministic, no statistics
# ---------------------------------------------------------------------------
# Mock the RNG at the module level so rng.random() always returns a constant,
# forcing every spin to win or lose without relying on statistical samples.
# These catch wrong payout multipliers, sign errors, and off-by-one bugs
# that aggregate tests might miss.

def test_always_lose_exact_bankroll(sim: RouletteSimulation, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When rng.random() always returns 1.0 (> any p_win), every spin loses.
    Final bankroll must equal exactly initial_bankroll - n_spins × bet_size.
    """
    class _AlwaysLoseRng:
        def random(self, n: int) -> np.ndarray:
            return np.ones(n)

    monkeypatch.setattr(_roulette_game.np.random, "default_rng", lambda seed: _AlwaysLoseRng())
    params = RouletteParams(
        n_spins=10, initial_bankroll=500.0, bet_size=10.0, bet_type="red_black"
    )
    result = sim.run(params)
    assert result.values[-1] == pytest.approx(500.0 - 10 * 10.0)


def test_always_win_exact_bankroll(sim: RouletteSimulation, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When rng.random() always returns 0.0 (< any p_win), every spin wins.
    red_black payout = 1×bet, so final = initial + n_spins × bet_size.
    Catches a wrong payout multiplier or missing win branch.
    """
    class _AlwaysWinRng:
        def random(self, n: int) -> np.ndarray:
            return np.zeros(n)

    monkeypatch.setattr(_roulette_game.np.random, "default_rng", lambda seed: _AlwaysWinRng())
    params = RouletteParams(
        n_spins=5, initial_bankroll=500.0, bet_size=10.0,
        bet_type="red_black", wheel_type="european",
    )
    result = sim.run(params)
    assert result.values[-1] == pytest.approx(500.0 + 5 * 10.0 * 1)


def test_always_win_single_number_exact_payout(
    sim: RouletteSimulation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    single_number payout = 35×bet.  Winning every spin confirms the 35:1 multiplier.
    """
    class _AlwaysWinRng:
        def random(self, n: int) -> np.ndarray:
            return np.zeros(n)

    monkeypatch.setattr(_roulette_game.np.random, "default_rng", lambda seed: _AlwaysWinRng())
    params = RouletteParams(
        n_spins=3, initial_bankroll=1000.0, bet_size=10.0,
        bet_type="single_number", wheel_type="european",
    )
    result = sim.run(params)
    assert result.values[-1] == pytest.approx(1000.0 + 3 * 10.0 * 35)


def test_final_bankroll_std_in_range(sim: RouletteSimulation) -> None:
    """
    After N spins the std of final bankrolls across runs should be close to
    bet_size * sqrt(N) (each spin ≈ ±bet on a near-fair coin).
    """
    N, runs = 500, 300
    params = RouletteParams(n_spins=N, initial_bankroll=100_000.0, bet_size=10.0)
    finals = np.array([r.final_value for r in sim.run_batch(params, runs)])

    theoretical_std = 10.0 * np.sqrt(N)  # ≈ 223.6
    observed_std = finals.std()
    # Allow ±30% relative tolerance — the test catches gross implementation errors.
    assert 0.7 * theoretical_std < observed_std < 1.3 * theoretical_std, (
        f"Std {observed_std:.1f} outside expected range "
        f"({0.7 * theoretical_std:.1f}, {1.3 * theoretical_std:.1f})"
    )


# ---------------------------------------------------------------------------
# 3. House-edge cross-validation against theory
# ---------------------------------------------------------------------------
# Theoretical per-spin expected gain for each (wheel, bet_type) combination.
# Source: standard roulette math.
#   p_win * payout - (1 - p_win) * 1  (all quantities in units of bet_size)
_THEORY = {
    # (wheel_type, bet_type): expected_net_per_unit_bet
    ("european", "red_black"):      -1 / 37,          # ≈ -0.02703
    ("american", "red_black"):      -2 / 38,          # ≈ -0.05263
    ("european", "dozen"):          -1 / 37,          # same house edge
    ("american", "dozen"):          -2 / 38,
    ("european", "single_number"):  -1 / 37,
    ("american", "single_number"):  -2 / 38,
}


# Per-spin standard deviation (in units of bet_size) for each bet type.
# Var[X] = m²·p + (1−p) − (m·p − (1−p))²  where m = payout multiplier, p = p_win.
# Source: standard roulette variance formulas.
_STD_PER_UNIT: dict[str, float] = {
    "red_black":     1.0,    # near-50/50, std ≈ 1
    "dozen":         1.37,   # 3:1 payout, std ≈ 1.37
    "single_number": 5.84,   # 35:1 payout, std ≈ 5.84
}


@pytest.mark.parametrize("wheel,bet_type", list(_THEORY.keys()))
def test_house_edge(sim: RouletteSimulation, wheel: str, bet_type: str) -> None:
    """
    Mean per-spin net gain must be within 5 standard errors of the theoretical
    value.  SE is computed from the actual per-spin variance for each bet type,
    which varies considerably: red/black std ≈ 1×bet, single_number std ≈ 5.8×bet.
    """
    BET = 10.0
    N_SPINS = 1_000
    N_RUNS = 200
    # Use a large bankroll so busts don't skew the per-spin calculation.
    params = RouletteParams(
        n_spins=N_SPINS,
        initial_bankroll=1_000_000.0,
        bet_size=BET,
        bet_type=bet_type,  # type: ignore[arg-type]
        wheel_type=wheel,   # type: ignore[arg-type]
    )
    results = sim.run_batch(params, N_RUNS)
    mean_net_per_spin = np.mean(
        [(r.final_value - r.values[0]) / N_SPINS for r in results]
    )
    expected = _THEORY[(wheel, bet_type)] * BET

    std_per_spin = _STD_PER_UNIT[bet_type] * BET
    se = std_per_spin / np.sqrt(N_RUNS * N_SPINS)
    tolerance = 5 * se

    assert abs(mean_net_per_spin - expected) < tolerance, (
        f"{wheel} {bet_type}: observed {mean_net_per_spin:.4f}/spin, "
        f"expected {expected:.4f}/spin (±{tolerance:.4f})"
    )
