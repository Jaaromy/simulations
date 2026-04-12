from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SIRParams:
    """Parameters for the SIR epidemic model.

    Population is divided into three compartments:
        S  — Susceptible (can catch the disease)
        I  — Infectious (currently spreading the disease)
        R  — Recovered / removed (immune or deceased)

    N = S + I + R is conserved throughout the simulation.
    """

    N: float = 10_000.0    # total population size
    I0: float = 10.0       # initial number infected
    beta: float = 0.3      # transmission rate (contacts × prob. of transmission per day)
    gamma: float = 0.05    # recovery rate (1/gamma = mean infectious period, days)
    t_end: float = 365.0   # simulation duration, days
    dt: float = 0.1        # RK4 integration time step, days
    n_snapshots: int = 500 # output resolution (points kept after downsampling)

    def validate(self) -> None:
        if self.N <= 0.0:
            raise ValueError(f"N must be > 0, got {self.N}")
        if self.I0 < 0.0 or self.I0 > self.N:
            raise ValueError(f"I0 must be in [0, N], got I0={self.I0}, N={self.N}")
        if self.beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}")
        if self.t_end <= 0.0:
            raise ValueError(f"t_end must be > 0, got {self.t_end}")
        if self.dt <= 0.0 or self.dt >= self.t_end:
            raise ValueError(f"dt must be in (0, t_end), got dt={self.dt}, t_end={self.t_end}")
        if self.n_snapshots < 2:
            raise ValueError(f"n_snapshots must be >= 2, got {self.n_snapshots}")

    @property
    def R0(self) -> float:
        """Basic reproduction number: expected secondary infections per case in a fully
        susceptible population.  R₀ > 1 → epidemic grows; R₀ ≤ 1 → epidemic dies out."""
        if self.gamma == 0.0:
            return float("inf")
        return self.beta / self.gamma

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
