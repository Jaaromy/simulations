import random

import numpy as np

from simulations.casino.blackjack.params import BlackjackParams
from simulations.core.base import Simulation
from simulations.core.results import TimeSeriesResult

_BLACKJACK_VALUE = 21          # Natural / bust threshold
_ACE_SOFT_VALUE = 11           # Ace counted as soft (high)
_ACE_SOFT_REDUCTION = 10       # Subtract when converting soft ace to hard (11 → 1)
_DEALER_STAND_THRESHOLD = 17   # Dealer (and basic-strategy player) stands at 17+
_BLACKJACK_PAYOUT = 1.5        # Natural blackjack pays 3:2
_DOUBLE_DOWN_MULTIPLIER = 2    # Double-down doubles the wager
_CARDS_PER_DECK = 52           # Standard deck size
_MIN_DECKS_FOR_COUNT = 0.5     # True count unreliable below this many decks remaining
_MIN_CUT_CARD = 15             # Minimum absolute cut-card position (cards from end)
_SHOE_CUT_FRACTION = 4         # Cut card placed at 1/_SHOE_CUT_FRACTION from the end

# Card values: 2-9 face value, 10/J/Q/K = 10, A = 11 (soft) or 1 (hard)
_CARD_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, _ACE_SOFT_VALUE]

# Hi/Lo count: +1 for low cards (2-6), 0 neutral (7-9), -1 for high cards (10, A)
_HI_LO: dict[int, int] = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, _ACE_SOFT_VALUE: -1}


def _build_shoe(deck_count: int) -> list[int]:
    return _CARD_VALUES * 4 * deck_count


def _hand_value(hand: list[int]) -> int:
    total = sum(hand)
    aces = hand.count(_ACE_SOFT_VALUE)
    while total > _BLACKJACK_VALUE and aces:
        total -= _ACE_SOFT_REDUCTION
        aces -= 1
    return total


class _Shoe:
    """Pre-shuffled shoe with O(1) draw via an advancing cursor."""

    __slots__ = ("_cards", "_cursor", "_deck_count", "_rng")

    def __init__(self, deck_count: int, rng: random.Random) -> None:
        self._deck_count = deck_count
        self._rng = rng
        self._cards: list[int] = []
        self._cursor = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._cards = _build_shoe(self._deck_count)
        self._rng.shuffle(self._cards)
        self._cursor = 0

    def remaining(self) -> int:
        return len(self._cards) - self._cursor

    def draw(self) -> int:
        card = self._cards[self._cursor]
        self._cursor += 1
        return card

    def deal(self) -> list[int]:
        a = self._cards[self._cursor]
        b = self._cards[self._cursor + 1]
        self._cursor += 2
        return [a, b]

    def __len__(self) -> int:
        return self.remaining()

    def ensure(self, cut_card: int) -> bool:
        """Reshuffle if below cut card. Returns True if reshuffled."""
        if self.remaining() <= cut_card:
            self._reshuffle()
            return True
        return False


def _deal(shoe: "_Shoe | list[int]", rng: random.Random) -> list[int]:
    if isinstance(shoe, _Shoe):
        return shoe.deal()
    return [shoe.pop(rng.randint(0, len(shoe) - 1)) for _ in range(2)]


def _draw(shoe: "_Shoe | list[int]", rng: random.Random) -> int:
    if isinstance(shoe, _Shoe):
        return shoe.draw()
    return shoe.pop(rng.randint(0, len(shoe) - 1))


def _basic_strategy_hit(player: list[int], dealer_up: int) -> bool:
    """Basic strategy hit/stand decision, called after double/split checks."""
    pv = _hand_value(player)
    has_soft_ace = _ACE_SOFT_VALUE in player and sum(player) <= _BLACKJACK_VALUE

    if pv <= _ACE_SOFT_VALUE:
        return True

    if has_soft_ace:
        if pv <= _DEALER_STAND_THRESHOLD:
            return True
        if pv == 18:  # Soft 18: hit vs 9, 10, A
            return dealer_up in (9, 10, _ACE_SOFT_VALUE)
        return False  # Soft 19+: stand

    # Hard hands
    if pv >= _DEALER_STAND_THRESHOLD:
        return False
    # Hard 12: stand vs 4–6, hit otherwise
    if pv == 12:
        return dealer_up not in (4, 5, 6)
    # Hard 13–16: stand vs 2–6, hit vs 7+
    return dealer_up not in range(2, 7)


def _should_double(player: list[int], dealer_up: int) -> bool:
    """Basic strategy doubling rules (2-card hands only)."""
    if len(player) != 2:
        return False
    pv = _hand_value(player)
    has_soft_ace = _ACE_SOFT_VALUE in player

    if has_soft_ace:
        if pv == 19:  # A8: double vs 6
            return dealer_up == 6
        if pv == 18:  # A7: double vs 3-6
            return dealer_up in range(3, 7)
        if pv == 17:  # A6: double vs 3-6
            return dealer_up in range(3, 7)
        if pv in (15, 16):  # A4/A5: double vs 4-6
            return dealer_up in range(4, 7)
        if pv in (13, 14):  # A2/A3: double vs 5-6
            return dealer_up in range(5, 7)
        return False

    if pv == 11:
        return True  # Always double hard 11
    if pv == 10:
        return dealer_up in range(2, 10)  # vs 2-9
    if pv == 9:
        return dealer_up in range(3, 7)   # vs 3-6
    return False


def _should_split(player: list[int], dealer_up: int) -> bool:
    """Basic strategy splitting rules (2-card pairs only)."""
    if len(player) != 2 or player[0] != player[1]:
        return False
    card = player[0]
    if card == _ACE_SOFT_VALUE:  # Aces: always split
        return True
    if card == 8:   # 8s: always split
        return True
    if card in (5, 10):  # 5s (better as hard 10), 10s: never split
        return False
    if card == 9:   # 9s: split vs 2-6, 8-9
        return dealer_up in (2, 3, 4, 5, 6, 8, 9)
    if card == 7:   # 7s: split vs 2-7
        return dealer_up in range(2, 8)
    if card == 6:   # 6s: split vs 2-6
        return dealer_up in range(2, 7)
    if card == 4:   # 4s: split vs 5-6
        return dealer_up in range(5, 7)
    if card in (2, 3):  # 2s/3s: split vs 2-7
        return dealer_up in range(2, 8)
    return False


def _play_single_hand(
    hand: list[int],
    dealer_up: int,
    shoe: "_Shoe | list[int]",
    bet: float,
    use_basic: bool,
    running_count: int,
    rng: random.Random,
) -> tuple[float, list[int], int]:
    """Play out one hand. Returns (effective_bet, final_hand, running_count)."""
    if use_basic and _should_double(hand, dealer_up):
        card = _draw(shoe, rng)
        running_count += _HI_LO[card]
        hand.append(card)
        return bet * _DOUBLE_DOWN_MULTIPLIER, hand, running_count

    while True:
        pv = _hand_value(hand)
        if pv >= _BLACKJACK_VALUE:
            break
        should_hit = _basic_strategy_hit(hand, dealer_up) if use_basic else pv < _DEALER_STAND_THRESHOLD
        if not should_hit:
            break
        card = _draw(shoe, rng)
        running_count += _HI_LO[card]
        hand.append(card)

    return bet, hand, running_count


def _play_hand(
    shoe: "_Shoe | list[int]",
    bet: float,
    use_basic: bool,
    running_count: int,
    rng: random.Random,
) -> tuple[float, int]:
    """Play one round. Returns (net gain/loss, updated running count)."""
    player = _deal(shoe, rng)
    dealer = _deal(shoe, rng)
    dealer_up = dealer[0]

    running_count += sum(_HI_LO[c] for c in player)
    running_count += _HI_LO[dealer_up]

    # Natural blackjack check
    if _hand_value(player) == _BLACKJACK_VALUE:
        running_count += _HI_LO[dealer[1]]
        if _hand_value(dealer) == _BLACKJACK_VALUE:
            return 0.0, running_count
        return bet * _BLACKJACK_PAYOUT, running_count

    # Split
    if use_basic and _should_split(player, dealer_up):
        split_aces = player[0] == _ACE_SOFT_VALUE
        hands: list[list[int]] = [[player[0]], [player[1]]]
        for hand in hands:
            card = _draw(shoe, rng)
            running_count += _HI_LO[card]
            hand.append(card)

        played: list[tuple[list[int], float]] = []
        for hand in hands:
            if split_aces:
                # Split aces receive one card only, no further action
                played.append((hand, bet))
            else:
                eff_bet, final_hand, running_count = _play_single_hand(
                    hand, dealer_up, shoe, bet, use_basic, running_count, rng
                )
                played.append((final_hand, eff_bet))

        running_count += _HI_LO[dealer[1]]
        while _hand_value(dealer) < _DEALER_STAND_THRESHOLD:
            card = _draw(shoe, rng)
            running_count += _HI_LO[card]
            dealer.append(card)
        dv = _hand_value(dealer)

        net = 0.0
        for hand, hand_bet in played:
            pv = _hand_value(hand)
            if pv > _BLACKJACK_VALUE:
                net -= hand_bet
            elif dv > 21 or pv > dv:
                net += hand_bet
            elif pv < dv:
                net -= hand_bet
        return net, running_count

    # Normal hand (with possible double-down)
    eff_bet, player, running_count = _play_single_hand(
        player, dealer_up, shoe, bet, use_basic, running_count, rng
    )

    pv = _hand_value(player)
    if pv > _BLACKJACK_VALUE:
        running_count += _HI_LO[dealer[1]]
        return -eff_bet, running_count

    running_count += _HI_LO[dealer[1]]

    while _hand_value(dealer) < _DEALER_STAND_THRESHOLD:
        card = _draw(shoe, rng)
        running_count += _HI_LO[card]
        dealer.append(card)

    dv = _hand_value(dealer)

    if dv > _BLACKJACK_VALUE or pv > dv:
        return eff_bet, running_count
    if pv == dv:
        return 0.0, running_count
    return -eff_bet, running_count


def _true_count(running_count: int, cards_remaining: int) -> float:
    """Convert running count to true count (per remaining deck)."""
    decks_remaining = cards_remaining / _CARDS_PER_DECK
    if decks_remaining < _MIN_DECKS_FOR_COUNT:
        return 0.0
    return running_count / decks_remaining


def _hilo_bet(true_count: float, base_bet: float, max_units: int) -> float:
    """Bet spreading: 1 unit at TC≤1, ramp up to max_units at TC≥max_units."""
    units = max(1, min(max_units, int(true_count)))
    return base_bet * units


class BlackjackSimulation(Simulation[BlackjackParams, TimeSeriesResult]):
    """Simulates player bankroll over N hands of blackjack."""

    @property
    def name(self) -> str:
        return "Blackjack"

    @property
    def description(self) -> str:
        return "Tracks bankroll over N hands using optional basic strategy and Hi/Lo counting."

    def run(self, params: BlackjackParams) -> TimeSeriesResult:
        params.validate()
        rng = random.Random(params.seed)
        shoe = _Shoe(params.deck_count, rng)
        cut_card = max(_MIN_CUT_CARD, len(_build_shoe(params.deck_count)) // _SHOE_CUT_FRACTION)
        running_count = 0

        bankroll = params.initial_bankroll
        bankrolls = np.empty(params.n_hands + 1)
        bankrolls[0] = bankroll

        for i in range(params.n_hands):
            if bankroll < params.bet_size:
                bankrolls[i + 1:] = 0.0
                break

            if shoe.ensure(cut_card):
                running_count = 0

            if params.use_hilo:
                tc = _true_count(running_count, len(shoe))
                bet = _hilo_bet(tc, params.bet_size, params.hilo_max_bet_units)
                bet = min(bet, bankroll)
            else:
                bet = params.bet_size

            gain, running_count = _play_hand(
                shoe, bet, params.use_basic_strategy, running_count, rng
            )
            bankroll = max(bankroll + gain, 0.0)
            bankrolls[i + 1] = bankroll

        return TimeSeriesResult(
            simulation_name=self.name,
            params_snapshot=params.to_dict(),
            steps=np.arange(params.n_hands + 1),
            values=bankrolls,
        )
