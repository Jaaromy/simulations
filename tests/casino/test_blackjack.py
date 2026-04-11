import random

import numpy as np
import pytest

from simulations.casino.blackjack.game import BlackjackSimulation, _Shoe, _play_hand
from simulations.casino.blackjack.params import BlackjackParams


class _IndexZeroRng:
    """Mock RNG: always returns index 0, turning the shoe into a deterministic FIFO queue."""
    def randint(self, a: int, b: int) -> int:
        return 0


@pytest.fixture
def sim() -> BlackjackSimulation:
    return BlackjackSimulation()


def test_result_shape(sim: BlackjackSimulation) -> None:
    params = BlackjackParams(n_hands=50)
    result = sim.run(params)
    assert len(result.steps) == 51
    assert len(result.values) == 51


def test_bankroll_never_negative(sim: BlackjackSimulation) -> None:
    params = BlackjackParams(n_hands=200, initial_bankroll=500.0, bet_size=10.0)
    for _ in range(10):
        result = sim.run(params)
        assert (result.values >= 0).all(), "Bankroll went negative"


def test_starts_at_initial_bankroll(sim: BlackjackSimulation) -> None:
    params = BlackjackParams(initial_bankroll=750.0)
    result = sim.run(params)
    assert result.values[0] == 750.0


def test_params_snapshot_captured(sim: BlackjackSimulation) -> None:
    params = BlackjackParams(n_hands=30, bet_size=5.0)
    result = sim.run(params)
    assert result.params_snapshot["n_hands"] == 30
    assert result.params_snapshot["bet_size"] == 5.0


def test_batch_run_length(sim: BlackjackSimulation) -> None:
    params = BlackjackParams(n_hands=20)
    results = sim.run_batch(params, n_runs=5)
    assert len(results) == 5


def test_invalid_params_raises(sim: BlackjackSimulation) -> None:
    with pytest.raises(ValueError, match="bet_size cannot exceed initial_bankroll"):
        sim.run(BlackjackParams(initial_bankroll=100.0, bet_size=200.0))


def test_hilo_bankroll_never_negative(sim: BlackjackSimulation) -> None:
    params = BlackjackParams(
        n_hands=200, initial_bankroll=500.0, bet_size=10.0,
        use_hilo=True, hilo_max_bet_units=8,
    )
    for _ in range(10):
        result = sim.run(params)
        assert (result.values >= 0).all(), "Hi/Lo bankroll went negative"


def test_hilo_result_shape(sim: BlackjackSimulation) -> None:
    params = BlackjackParams(n_hands=50, use_hilo=True)
    result = sim.run(params)
    assert len(result.steps) == 51
    assert len(result.values) == 51


def test_hilo_requires_basic_strategy(sim: BlackjackSimulation) -> None:
    with pytest.raises(ValueError, match="use_hilo requires use_basic_strategy"):
        sim.run(BlackjackParams(use_hilo=True, use_basic_strategy=False))


def test_hilo_params_in_snapshot(sim: BlackjackSimulation) -> None:
    params = BlackjackParams(use_hilo=True, hilo_max_bet_units=4)
    result = sim.run(params)
    assert result.params_snapshot["use_hilo"] is True
    assert result.params_snapshot["hilo_max_bet_units"] == 4


# ---------------------------------------------------------------------------
# Fidelity — seed reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducibility(sim: BlackjackSimulation) -> None:
    """Identical seeds must produce bit-identical trajectories."""
    params = BlackjackParams(n_hands=200, seed=42)
    r1 = sim.run(params)
    r2 = sim.run(params)
    np.testing.assert_array_equal(r1.values, r2.values)


def test_different_seeds_differ(sim: BlackjackSimulation) -> None:
    r1 = sim.run(BlackjackParams(n_hands=200, seed=1))
    r2 = sim.run(BlackjackParams(n_hands=200, seed=2))
    assert not np.array_equal(r1.values, r2.values)


# ---------------------------------------------------------------------------
# Fidelity — house-edge cross-validation
# ---------------------------------------------------------------------------

def test_basic_strategy_rtp_in_range(sim: BlackjackSimulation) -> None:
    """
    Basic strategy house edge ≈ 0.5% → RTP ≈ 99.5%.

    With 200k hands (n_runs=100, n_hands=2000):
      std per hand ≈ 1.14 × bet = 11.4
      SE of RTP ≈ 11.4 / (sqrt(200k) × 10) ≈ 0.00255
      ±1.5% band = ±5.9 SE — essentially impossible to fail by chance.

    Catches subtle strategy bugs (e.g., wrong soft-17 rule, missed double-down)
    that shift the house edge by > 1% but wouldn't move it by 3%.
    """
    BET = 10.0
    N_HANDS = 2_000
    N_RUNS = 100
    BANKROLL = 1_000_000.0  # avoid busts skewing the calculation

    params = BlackjackParams(
        n_hands=N_HANDS,
        initial_bankroll=BANKROLL,
        bet_size=BET,
        use_basic_strategy=True,
        use_hilo=False,
    )
    results = sim.run_batch(params, N_RUNS)
    mean_net_per_hand = np.mean(
        [(r.final_value - r.values[0]) / N_HANDS for r in results]
    )
    rtp = 1.0 + mean_net_per_hand / BET

    assert 0.985 <= rtp <= 1.005, (
        f"Basic strategy RTP {rtp:.4f} outside expected range [0.985, 1.005]"
    )


# ---------------------------------------------------------------------------
# 4. Single-hand trace tests — deterministic, no statistics
# ---------------------------------------------------------------------------
# These tests construct a shoe with known cards at the front and use
# _IndexZeroRng (always pops index 0) to make the deal a FIFO draw.
# Each scenario verifies exact net gain for a specific hand outcome.
# A wrong payout multiplier, sign error, or off-by-one in game logic
# will show up here as a wrong number before any statistical test can catch it.

def test_trace_natural_blackjack_pays_1_5x() -> None:
    """Player [10, A] vs dealer [5, 7] → blackjack → +1.5 × bet."""
    shoe = [10, 11, 5, 7] + [2] * 20
    net, _ = _play_hand(shoe, bet=10.0, use_basic=True, running_count=0, rng=_IndexZeroRng())
    assert net == 15.0, f"Expected +15 for blackjack, got {net}"


def test_trace_both_blackjack_pushes() -> None:
    """Player [10, A] vs dealer [10, A] → both blackjack → push → 0."""
    shoe = [10, 11, 10, 11] + [2] * 20
    net, _ = _play_hand(shoe, bet=10.0, use_basic=True, running_count=0, rng=_IndexZeroRng())
    assert net == 0.0, f"Expected 0 for BJ vs BJ, got {net}"


def test_trace_player_bust_loses_bet() -> None:
    """
    Player [10, 6] = hard 16 vs dealer up 7 → basic strategy hits → draws 10 → busts → -bet.

    Shoe layout (each pop(0) takes the front):
      player deal: 10, 6 | dealer deal: 7, 5 | player hit: 10
    """
    shoe = [10, 6, 7, 5, 10] + [2] * 20
    net, _ = _play_hand(shoe, bet=10.0, use_basic=True, running_count=0, rng=_IndexZeroRng())
    assert net == -10.0, f"Expected -10 for player bust, got {net}"


def test_trace_dealer_bust_player_wins() -> None:
    """
    Player [10, 7] = 17, stands. Dealer [10, 6] = 16 must hit → draws 10 → busts → +bet.

    Shoe layout:
      player deal: 10, 7 | dealer deal: 10, 6 | dealer hit: 10
    """
    shoe = [10, 7, 10, 6, 10] + [2] * 20
    net, _ = _play_hand(shoe, bet=10.0, use_basic=True, running_count=0, rng=_IndexZeroRng())
    assert net == 10.0, f"Expected +10 for dealer bust, got {net}"


def test_trace_push_returns_zero() -> None:
    """Player [10, 7] = 17 vs dealer [10, 7] = 17 → both stand → tie → 0."""
    shoe = [10, 7, 10, 7] + [2] * 20
    net, _ = _play_hand(shoe, bet=10.0, use_basic=True, running_count=0, rng=_IndexZeroRng())
    assert net == 0.0, f"Expected 0 for push, got {net}"


def test_trace_double_down_win() -> None:
    """
    Hard 11 vs dealer 5 → basic strategy doubles → player draws 10 → 21.
    Dealer [5, 6] = 11, must hit → draws 10 → 21 → push. No wait, let's pick
    a dealer that busts.

    Shoe layout (FIFO):
      player deal: 7, 4  (hard 11 → doubles)
      dealer deal: 5, 6  (dealer has 11, must hit)
      player double card: 10  → player = 21
      dealer hit: 10 → dealer = 21 → push would be 0

    Use dealer bust instead:
      player deal: 7, 4  (hard 11)
      dealer deal: 5, 6  (hard 11, must hit)
      player double: 10  → 21
      dealer hit: 8  → dealer = 19 → player 21 > dealer 19 → +2×bet

    Net = +20 (double-down on 10 bet, won).
    """
    shoe = [7, 4, 5, 6, 10, 8] + [2] * 20
    net, _ = _play_hand(shoe, bet=10.0, use_basic=True, running_count=0, rng=_IndexZeroRng())
    assert net == 20.0, f"Expected +20 for double-down win, got {net}"


def test_trace_double_down_lose() -> None:
    """
    Hard 11 vs dealer 5 → doubles → player draws 2 → 13.
    Dealer [5, 8] = 13, hits → draws 9 → 22, busts → player wins 2×bet.

    Wait, if dealer busts player wins regardless. Use dealer standing 20:
      player deal: 7, 4  (hard 11 → doubles)
      dealer deal: 10, 10  (dealer 20, stands)
      player double: 2  → player = 13, dealer 20 → player loses 2×bet

    Net = -20.
    """
    shoe = [7, 4, 10, 10, 2] + [2] * 20
    net, _ = _play_hand(shoe, bet=10.0, use_basic=True, running_count=0, rng=_IndexZeroRng())
    assert net == -20.0, f"Expected -20 for double-down loss, got {net}"


def test_trace_split_aces_each_wins() -> None:
    """
    Split aces: player [A, A] vs dealer 6.
    Each ace gets one card. Both get 10 → 21. Dealer [6, 5] = 11, hits → draws 10 → 21 → push.

    Use dealer bust:
      player deal: A, A
      dealer deal: 6, 5  (hard 11, must hit)
      split card 1: 10  → hand1 = [A, 10] = 21
      split card 2: 10  → hand2 = [A, 10] = 21
      dealer hit: 10  → dealer = 21 (push both)

    To get a clear win, pick dealer bust:
      player deal: A, A
      dealer deal: 6, 5
      split card 1: 10 → 21
      split card 2: 10 → 21
      dealer hit: 10 → 21 → push on both → net 0

    Instead, dealer [6, 7] = 13, hits → 10 → 23 (bust) → both hands win.
      Shoe: A, A, 6, 7, 10, 10, 10, [padding]
    Net = +20 (two bets won).
    """
    shoe = [11, 11, 6, 7, 10, 10, 10] + [2] * 20
    net, _ = _play_hand(shoe, bet=10.0, use_basic=True, running_count=0, rng=_IndexZeroRng())
    assert net == 20.0, f"Expected +20 for split aces both winning, got {net}"


# ---------------------------------------------------------------------------
# _Shoe unit tests
# ---------------------------------------------------------------------------

def test_shoe_deal_advances_cursor() -> None:
    """deal() returns the first two cards in shuffle order and advances by 2."""
    rng = random.Random(0)
    shoe = _Shoe(deck_count=1, rng=rng)
    total = shoe.remaining()
    c1, c2 = shoe.deal()
    assert shoe.remaining() == total - 2
    assert c1 in range(2, 12)
    assert c2 in range(2, 12)


def test_shoe_draw_advances_cursor() -> None:
    """draw() returns one card and advances cursor by 1."""
    rng = random.Random(7)
    shoe = _Shoe(deck_count=1, rng=rng)
    total = shoe.remaining()
    card = shoe.draw()
    assert shoe.remaining() == total - 1
    assert card in range(2, 12)


def test_shoe_initial_size() -> None:
    """A 4-deck shoe has 4 × 52 = 208 cards (13 values × 4 suits × 4 decks)."""
    rng = random.Random(0)
    shoe = _Shoe(deck_count=4, rng=rng)
    assert shoe.remaining() == 208


def test_shoe_ensure_reshuffles_below_cut() -> None:
    """ensure() returns True and resets remaining when below cut threshold."""
    rng = random.Random(0)
    shoe = _Shoe(deck_count=1, rng=rng)
    # Drain until just above cut
    cut = 10
    while shoe.remaining() > cut + 1:
        shoe.draw()
    # Now drain to exactly cut
    while shoe.remaining() > cut:
        shoe.draw()
    assert shoe.remaining() <= cut
    reshuffled = shoe.ensure(cut)
    assert reshuffled is True
    assert shoe.remaining() == 52  # 1-deck shoe fully restored


def test_shoe_ensure_no_reshuffle_above_cut() -> None:
    """ensure() returns False and does not reshuffle when above cut threshold."""
    rng = random.Random(0)
    shoe = _Shoe(deck_count=1, rng=rng)
    cut = 10
    remaining_before = shoe.remaining()
    reshuffled = shoe.ensure(cut)
    assert reshuffled is False
    assert shoe.remaining() == remaining_before


def test_shoe_sequential_draw_order() -> None:
    """Sequential draws return cards in the pre-shuffled order (no skipping)."""
    rng = random.Random(42)
    shoe = _Shoe(deck_count=1, rng=rng)
    # Re-create with same seed to get the shuffled list directly
    rng2 = random.Random(42)
    import simulations.casino.blackjack.game as _bj
    expected = _bj._build_shoe(1)
    rng2.shuffle(expected)
    drawn = [shoe.draw() for _ in range(10)]
    assert drawn == expected[:10]


def test_shoe_seed_reproducibility() -> None:
    """Two _Shoe objects with the same seed produce identical draw sequences."""
    shoe1 = _Shoe(deck_count=4, rng=random.Random(99))
    shoe2 = _Shoe(deck_count=4, rng=random.Random(99))
    seq1 = [shoe1.draw() for _ in range(20)]
    seq2 = [shoe2.draw() for _ in range(20)]
    assert seq1 == seq2
