from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CollisionParams:
    """Parameters for the 2D ball collision simulation.

    Physics are handled by Matter.js in the browser.  This dataclass
    defines the configuration that is serialised into the JavaScript
    engine initialisation.

    Units are intentionally dimensionless / pixel-space because Matter.js
    works in arbitrary pixel units, not SI.  The slider labels in the
    notebook explain the physical meaning of each parameter.
    """

    n_balls: int = 10
    friction: float = 0.05         # Coulomb friction coefficient [0, 1]
    friction_air: float = 0.005    # air drag coefficient [0, 0.1]
    restitution: float = 0.8       # coefficient of restitution [0, 1]
    mass_min: float = 1.0          # minimum ball mass (arbitrary units)
    mass_max: float = 5.0          # maximum ball mass (arbitrary units)
    speed: float = 5.0             # initial speed magnitude (px per engine tick)
    angle_spread: float = 360.0    # launch angle cone width in degrees; 360 = omnidirectional
    canvas_width: int = 800
    canvas_height: int = 600
    gravity_y: float = 1.0         # vertical gravity scale; 0 = zero-g billiards mode
    show_velocity_vectors: bool = False

    def validate(self) -> None:
        if not (0 <= self.n_balls <= 200):
            raise ValueError(f"n_balls must be in [0, 200], got {self.n_balls}")
        if not (0.0 <= self.friction <= 1.0):
            raise ValueError(f"friction must be in [0, 1], got {self.friction}")
        if not (0.0 <= self.friction_air <= 0.1):
            raise ValueError(f"friction_air must be in [0, 0.1], got {self.friction_air}")
        if not (0.0 <= self.restitution <= 1.0):
            raise ValueError(f"restitution must be in [0, 1], got {self.restitution}")
        if self.mass_min <= 0.0:
            raise ValueError(f"mass_min must be > 0, got {self.mass_min}")
        if self.mass_max <= 0.0:
            raise ValueError(f"mass_max must be > 0, got {self.mass_max}")
        if self.mass_min > self.mass_max:
            raise ValueError(
                f"mass_min must be <= mass_max, got {self.mass_min} > {self.mass_max}"
            )
        if self.speed < 0.0:
            raise ValueError(f"speed must be >= 0, got {self.speed}")
        if not (0.0 < self.angle_spread <= 360.0):
            raise ValueError(f"angle_spread must be in (0, 360], got {self.angle_spread}")
        if self.canvas_width <= 0:
            raise ValueError(f"canvas_width must be > 0, got {self.canvas_width}")
        if self.canvas_height <= 0:
            raise ValueError(f"canvas_height must be > 0, got {self.canvas_height}")
        if self.gravity_y < 0.0:
            raise ValueError(f"gravity_y must be >= 0, got {self.gravity_y}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
