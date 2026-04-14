from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LogisticParams:
    n_generations: int = 100
    initial_population: int = 100
    birth_rate: float = 0.15      # offspring per individual per generation
    death_rate: float = 0.05      # base probability of death per individual
    carrying_capacity: int = 5000  # logistic cap
    mutation_rate: float = 0.01   # placeholder for future trait evolution
    seed: int | None = None

    def validate(self) -> None:
        if self.n_generations < 1:
            raise ValueError("n_generations must be at least 1")
        if self.initial_population < 1:
            raise ValueError("initial_population must be at least 1")
        if not (0.0 <= self.birth_rate <= 1.0):
            raise ValueError("birth_rate must be between 0 and 1")
        if not (0.0 <= self.death_rate <= 1.0):
            raise ValueError("death_rate must be between 0 and 1")
        if self.carrying_capacity < 1:
            raise ValueError("carrying_capacity must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
