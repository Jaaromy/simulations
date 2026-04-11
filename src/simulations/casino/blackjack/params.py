from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BlackjackParams:
    n_hands: int = 100
    initial_bankroll: float = 1000.0
    bet_size: float = 10.0
    deck_count: int = 6
    use_basic_strategy: bool = True
    use_hilo: bool = False
    hilo_max_bet_units: int = 8  # max bet = bet_size * this, when true count is very high
    seed: int | None = None

    def validate(self) -> None:
        if self.n_hands < 1:
            raise ValueError("n_hands must be at least 1")
        if self.initial_bankroll <= 0:
            raise ValueError("initial_bankroll must be positive")
        if self.bet_size <= 0:
            raise ValueError("bet_size must be positive")
        if self.bet_size > self.initial_bankroll:
            raise ValueError("bet_size cannot exceed initial_bankroll")
        if self.deck_count < 1:
            raise ValueError("deck_count must be at least 1")
        if self.use_hilo and not self.use_basic_strategy:
            raise ValueError("use_hilo requires use_basic_strategy=True")
        if self.hilo_max_bet_units < 1:
            raise ValueError("hilo_max_bet_units must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
