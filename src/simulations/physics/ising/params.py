from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class IsingParams:
    grid_size: int = 50           # L; lattice is L × L
    temperature: float = 2.0      # T in units where J = 1, k_B = 1
    n_sweeps: int = 1000          # one sweep = L² Metropolis flip attempts
    n_snapshots: int = 100        # how many (|M|, E/N) readings to record
    initial_state: str = "random" # "random" | "aligned_up" | "aligned_down"
    seed: int | None = None

    def validate(self) -> None:
        if self.grid_size < 2:
            raise ValueError(f"grid_size must be ≥ 2, got {self.grid_size}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.n_sweeps < 1:
            raise ValueError(f"n_sweeps must be ≥ 1, got {self.n_sweeps}")
        if self.n_snapshots < 2:
            raise ValueError(f"n_snapshots must be ≥ 2, got {self.n_snapshots}")
        if self.initial_state not in {"random", "aligned_up", "aligned_down"}:
            raise ValueError(
                f"initial_state must be 'random', 'aligned_up', or 'aligned_down'; "
                f"got {self.initial_state!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
