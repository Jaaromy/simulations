from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PendulumParams:
    """Parameters for a double-pendulum simulation.

    Angles (theta1_0, theta2_0) are in degrees measured from the downward
    vertical.  Internally the model converts to radians before integrating.
    """

    theta1_0: float = 120.0   # initial angle of upper bob, degrees from vertical
    theta2_0: float = -20.0   # initial angle of lower bob, degrees from vertical
    omega1_0: float = 0.0     # initial angular velocity of upper bob, rad/s
    omega2_0: float = 0.0     # initial angular velocity of lower bob, rad/s
    m1: float = 1.0           # mass of upper bob, kg
    m2: float = 1.0           # mass of lower bob, kg
    l1: float = 1.0           # length of upper rod, m
    l2: float = 1.0           # length of lower rod, m
    g: float = 9.81           # gravitational acceleration, m/s²
    t_end: float = 30.0       # simulation duration, s
    dt: float = 0.001         # RK4 integration time step, s
    n_snapshots: int = 1000   # output resolution (points kept after downsampling)

    def validate(self) -> None:
        if self.m1 <= 0.0:
            raise ValueError(f"m1 must be > 0, got {self.m1}")
        if self.m2 <= 0.0:
            raise ValueError(f"m2 must be > 0, got {self.m2}")
        if self.l1 <= 0.0:
            raise ValueError(f"l1 must be > 0, got {self.l1}")
        if self.l2 <= 0.0:
            raise ValueError(f"l2 must be > 0, got {self.l2}")
        if self.g < 0.0:
            raise ValueError(f"g must be >= 0, got {self.g}")
        if self.t_end <= 0.0:
            raise ValueError(f"t_end must be > 0, got {self.t_end}")
        if self.dt <= 0.0 or self.dt >= self.t_end:
            raise ValueError(f"dt must be in (0, t_end), got dt={self.dt}, t_end={self.t_end}")
        if self.n_snapshots < 2:
            raise ValueError(f"n_snapshots must be >= 2, got {self.n_snapshots}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
