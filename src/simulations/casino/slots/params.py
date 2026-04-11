from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SlotsParams:
    n_spins: int = 100
    initial_bankroll: float = 1000.0
    bet_size: float = 1.0
    rtp: float = 0.95  # return-to-player percentage (0.0–1.0)
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
        if not (0.0 < self.rtp < 1.0):
            raise ValueError("rtp must be between 0 and 1 exclusive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
