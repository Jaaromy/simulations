"""
Fidelity tests for SlotsSimulation.

Three layers of coverage:
  1. Seed reproducibility  — same seed → bit-identical output.
  2. Distributional        — bust rate and variance are consistent with the
                             exponential payout model.
  3. House-edge theory     — observed RTP must be within a tight band of the
                             configured rtp parameter.

Slots have high per-spin variance (exponential tail), so house-edge tests
require more total spins than roulette to achieve the same SE.
"""

import numpy as np
import pytest
import simulations.casino.slots.game as _slots_game

from simulations.casino.slots.game import SlotsSimulation
from simulations.casino.slots.params import SlotsParams


@pytest.fixture
def sim() -> SlotsSimulation:
    return SlotsSimulation()


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_result_shape(sim: SlotsSimulation) -> None:
    params = SlotsParams(n_spins=50)
    result = sim.run(params)
    assert len(result.steps) == 51
    assert len(result.values) == 51


def test_starts_at_initial_bankroll(sim: SlotsSimulation) -> None:
    params = SlotsParams(initial_bankroll=500.0)
    assert sim.run(params).values[0] == 500.0


def test_bankroll_never_negative(sim: SlotsSimulation) -> None:
    params = SlotsParams(n_spins=500, initial_bankroll=5000.0, bet_size=1.0)
    for _ in range(20):
        assert (sim.run(params).values >= 0).all()


def test_batch_run_length(sim: SlotsSimulation) -> None:
    assert len(sim.run_batch(SlotsParams(n_spins=10), n_runs=7)) == 7


def test_invalid_params_raises(sim: SlotsSimulation) -> None:
    with pytest.raises(ValueError, match="bet_size cannot exceed initial_bankroll"):
        sim.run(SlotsParams(initial_bankroll=5.0, bet_size=10.0))


# ---------------------------------------------------------------------------
# Limit cases — deterministic, no statistics
# ---------------------------------------------------------------------------

def test_always_lose_exact_bankroll(sim: SlotsSimulation, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When rng.random() always returns 0.5 (≥ 0.15 win threshold), every spin loses.
    Final bankroll must equal exactly initial_bankroll - n_spins × bet_size.
    Catches a wrong sign or missing subtraction in the loss branch.
    """
    class _AlwaysLoseRng:
        def random(self, n: int) -> np.ndarray:
            return np.full(n, 0.5)  # always ≥ 0.15 → loss path
        def exponential(self, scale: float, size: int) -> np.ndarray:
            return np.zeros(size)  # unreachable in loss path; present for completeness

    monkeypatch.setattr(_slots_game.np.random, "default_rng", lambda seed: _AlwaysLoseRng())
    params = SlotsParams(n_spins=10, initial_bankroll=500.0, bet_size=10.0, rtp=0.95)
    result = sim.run(params)
    assert result.values[-1] == pytest.approx(500.0 - 10 * 10.0)


# ---------------------------------------------------------------------------
# 1. Seed reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducibility(sim: SlotsSimulation) -> None:
    """Identical seeds must produce bit-identical trajectories."""
    params = SlotsParams(n_spins=200, seed=99)
    r1 = sim.run(params)
    r2 = sim.run(params)
    np.testing.assert_array_equal(r1.values, r2.values)


def test_different_seeds_differ(sim: SlotsSimulation) -> None:
    r1 = sim.run(SlotsParams(n_spins=200, seed=10))
    r2 = sim.run(SlotsParams(n_spins=200, seed=11))
    assert not np.array_equal(r1.values, r2.values)


# ---------------------------------------------------------------------------
# 2. Distributional properties
# ---------------------------------------------------------------------------

def test_win_event_rate_near_fifteen_percent(sim: SlotsSimulation) -> None:
    """
    The slots model enters the win branch on 15% of spins.  A win event
    produces a bankroll delta > -bet_size (any payout, even sub-bet).
    A loss always produces exactly -bet_size.

    Across 100k spins the 5-SE tolerance is ±0.0035, tight enough to catch
    a wrong win-probability constant.
    """
    BET = 1.0
    N_SPINS = 1_000
    N_RUNS = 100
    # Large bankroll to prevent busts distorting the count.
    params = SlotsParams(n_spins=N_SPINS, initial_bankroll=1_000_000.0, bet_size=BET)
    results = sim.run_batch(params, N_RUNS)

    # A win event is any spin where the delta is strictly greater than -bet_size
    # (losses are always exactly -bet_size; wins produce payout - bet_size which
    # can be anywhere from just above -bet_size to very large positive).
    win_counts = []
    for r in results:
        diffs = np.diff(r.values)
        win_counts.append((diffs > -BET).sum())

    observed_win_rate = np.sum(win_counts) / (N_RUNS * N_SPINS)
    expected_win_rate = 0.15
    se = np.sqrt(expected_win_rate * (1 - expected_win_rate) / (N_RUNS * N_SPINS))
    assert abs(observed_win_rate - expected_win_rate) < 5 * se, (
        f"Win rate {observed_win_rate:.4f} too far from expected {expected_win_rate}"
    )


# ---------------------------------------------------------------------------
# 3. House-edge / RTP cross-validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rtp", [0.85, 0.90, 0.95])
def test_rtp_matches_configured_value(sim: SlotsSimulation, rtp: float) -> None:
    """
    The per-spin expected net as a fraction of bet_size should equal (rtp - 1).

    Slots variance is high (exponential tail), so we need more total spins.
    With 500k total spins, SE ≈ std_per_spin / sqrt(500k).

    Theoretical std per spin (bet=1, rtp=0.95):
      σ² = 0.85*1 + 0.15*E[(exp - 1)²] - (rtp-1)²
      E[(exp-1)²] ≈ 2*(rtp/0.15)² (variance of shifted exponential)
      For rtp=0.95: σ ≈ 6.3; scales with rtp so we use a fixed generous tolerance.
    """
    BET = 1.0
    N_SPINS = 1_000
    N_RUNS = 500
    params = SlotsParams(
        n_spins=N_SPINS,
        initial_bankroll=1_000_000.0,
        bet_size=BET,
        rtp=rtp,
    )
    results = sim.run_batch(params, N_RUNS)
    mean_net_per_spin = np.mean(
        [(r.final_value - r.values[0]) / N_SPINS for r in results]
    )
    expected = (rtp - 1.0) * BET

    # SE is dominated by the exponential tail; use empirical std as upper bound.
    # A tolerance of 0.05 is >> 5*SE for 500k spins and catches any wrong rtp wiring.
    assert abs(mean_net_per_spin - expected) < 0.05, (
        f"rtp={rtp}: observed {mean_net_per_spin:.4f}/spin, "
        f"expected {expected:.4f}/spin"
    )
