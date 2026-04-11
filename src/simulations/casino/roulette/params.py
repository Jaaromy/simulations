from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class RouletteParams:
    n_spins: int = 100
    initial_bankroll: float = 1000.0
    bet_size: float = 10.0
    bet_type: Literal["red_black", "single_number", "dozen"] = "red_black"
    wheel_type: Literal["american", "european"] = "european"
    seed: int | None = None

    def validate(self) -> None:
        if self.n_spins < 1:
            raise ValueError("n_spins must be at least 1")
        if self.initial_bankroll <= 0:
            raise ValueError("initial_bankroll must be positive")
        if self.bet_size <= 0:
            raise ValueError("bet_size must be positive")
        if self.bet_size > self.initial_bankroll:
            raise ValueError("bet_size cannot exceed initial_bankroll")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
