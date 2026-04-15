from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_AU = 1.496e11  # metres per AU
_G = 6.67430e-11  # m³ kg⁻¹ s⁻²


@dataclass
class BodyIC:
    """Initial conditions for a single gravitational body."""

    name: str
    mass_kg: float
    x_m: float
    y_m: float
    vx_m_s: float
    vy_m_s: float
    radius_m: float
    color: str


def solar_system_preset() -> list[BodyIC]:
    """Sun + 8 planets + Luna on circular orbits along +x axis."""

    sun_mass = 1.989e30

    def circ_v(r: float, parent_mass: float) -> float:
        return math.sqrt(_G * parent_mass / r)

    bodies: list[BodyIC] = []

    # Sun at origin
    bodies.append(
        BodyIC(
            name="Sun",
            mass_kg=sun_mass,
            x_m=0.0,
            y_m=0.0,
            vx_m_s=0.0,
            vy_m_s=0.0,
            radius_m=6.957e8,
            color="#FDB813",
        )
    )

    planets = [
        ("Mercury", 0.387, 3.301e23, 2.440e6, "#B5B5B5"),
        ("Venus",   0.723, 4.867e24, 6.052e6, "#E8C47A"),
        ("Earth",   1.000, 5.972e24, 6.371e6, "#4B9CD3"),
        ("Mars",    1.524, 6.390e23, 3.390e6, "#C1440E"),
        ("Jupiter", 5.203, 1.898e27, 6.991e7, "#C88B3A"),
        ("Saturn",  9.537, 5.683e26, 5.823e7, "#EAD6A6"),
        ("Uranus",  19.19, 8.681e25, 2.536e7, "#7DE8E8"),
        ("Neptune", 30.07, 1.024e26, 2.462e7, "#4B70DD"),
    ]

    earth_x = 0.0
    earth_vy = 0.0

    for name, a_au, mass, radius, color in planets:
        r = a_au * _AU
        v = circ_v(r, sun_mass)
        bodies.append(
            BodyIC(
                name=name,
                mass_kg=mass,
                x_m=r,
                y_m=0.0,
                vx_m_s=0.0,
                vy_m_s=v,
                radius_m=radius,
                color=color,
            )
        )
        if name == "Earth":
            earth_x = r
            earth_vy = v

    # Luna: Earth position + 384_400_000 m along x
    luna_a = 384_400_000.0
    earth_mass = 5.972e24
    luna_v_around_earth = circ_v(luna_a, earth_mass)
    bodies.append(
        BodyIC(
            name="Luna",
            mass_kg=7.342e22,
            x_m=earth_x + luna_a,
            y_m=0.0,
            vx_m_s=0.0,
            vy_m_s=earth_vy + luna_v_around_earth,
            radius_m=1.737e6,
            color="#CCCCCC",
        )
    )

    return bodies


@dataclass
class GravityParams:
    """Parameters for the real-time N-body gravitational simulation."""

    bodies: list[BodyIC] = field(default_factory=solar_system_preset)
    G: float = _G
    softening_m: float = 1e7
    dt_s: float = 3600.0
    substeps: int = 4
    time_warp: float = 86400.0
    canvas_width: int = 900
    canvas_height: int = 720
    view_scale_au: float = 3.2
    view_center: str = "sun"
    trail_enabled: bool = True
    trail_length: int = 600
    show_labels: bool = True
    log_radius_scale: bool = True
    merge_enabled: bool = True
    preset: str = "solar_system"

    def validate(self) -> None:
        if len(self.bodies) < 1:
            raise ValueError("bodies must contain at least 1 body")
        for b in self.bodies:
            if b.mass_kg <= 0:
                raise ValueError(f"body '{b.name}' mass_kg must be positive, got {b.mass_kg}")
            if b.radius_m <= 0:
                raise ValueError(f"body '{b.name}' radius_m must be positive, got {b.radius_m}")
        if self.G <= 0:
            raise ValueError(f"G must be positive, got {self.G}")
        if self.softening_m < 0:
            raise ValueError(f"softening_m must be >= 0, got {self.softening_m}")
        if self.dt_s <= 0:
            raise ValueError(f"dt_s must be positive, got {self.dt_s}")
        if self.substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {self.substeps}")
        if self.time_warp <= 0:
            raise ValueError(f"time_warp must be positive, got {self.time_warp}")
        if self.view_scale_au <= 0:
            raise ValueError(f"view_scale_au must be positive, got {self.view_scale_au}")
        if self.view_center not in ("sun", "centroid"):
            raise ValueError(f"view_center must be 'sun' or 'centroid', got '{self.view_center}'")
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ValueError("canvas dimensions must be positive")
        if self.trail_length < 0:
            raise ValueError(f"trail_length must be >= 0, got {self.trail_length}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bodies": [
                {
                    "name": b.name,
                    "massKg": b.mass_kg,
                    "xM": b.x_m,
                    "yM": b.y_m,
                    "vxMs": b.vx_m_s,
                    "vyMs": b.vy_m_s,
                    "radiusM": b.radius_m,
                    "color": b.color,
                }
                for b in self.bodies
            ],
            "G": self.G,
            "softeningM": self.softening_m,
            "dtS": self.dt_s,
            "substeps": self.substeps,
            "timeWarp": self.time_warp,
            "canvasWidth": self.canvas_width,
            "canvasHeight": self.canvas_height,
            "viewScaleAu": self.view_scale_au,
            "viewCenter": self.view_center,
            "trailEnabled": self.trail_enabled,
            "trailLength": self.trail_length,
            "showLabels": self.show_labels,
            "logRadiusScale": self.log_radius_scale,
            "mergeEnabled": self.merge_enabled,
            "canvasId": "c",
        }


def parse_ic_text(text: str) -> list[BodyIC]:
    """Parse textarea text into list[BodyIC].

    Format: one body per line, 8 comma-separated values:
        name, mass_kg, x_AU, y_AU, vx_km_s, vy_km_s, radius_km, color

    Lines starting with '#' or blank are skipped.
    AU → m (×1.496e11), km/s → m/s (×1000), km → m (×1000).
    Raises ValueError with line number if a line has fewer than 8 fields.
    """
    bodies: list[BodyIC] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",", 7)]
        if len(parts) < 8:
            raise ValueError(
                f"Line {lineno}: expected 8 comma-separated fields, got {len(parts)}: {line!r}"
            )
        name = parts[0]
        mass_kg = float(parts[1])
        x_m = float(parts[2]) * _AU
        y_m = float(parts[3]) * _AU
        vx_m_s = float(parts[4]) * 1000.0
        vy_m_s = float(parts[5]) * 1000.0
        radius_m = float(parts[6]) * 1000.0
        color = parts[7]
        bodies.append(
            BodyIC(
                name=name,
                mass_kg=mass_kg,
                x_m=x_m,
                y_m=y_m,
                vx_m_s=vx_m_s,
                vy_m_s=vy_m_s,
                radius_m=radius_m,
                color=color,
            )
        )
    return bodies


def bodies_to_text(bodies: list[BodyIC]) -> str:
    """Serialize list[BodyIC] to textarea text (reverse of parse_ic_text).

    Units: AU, km/s, km. 6 significant figures.
    Prepends a header comment line.
    """
    header = "# name, mass_kg, x_AU, y_AU, vx_km_s, vy_km_s, radius_km, color"
    def fmt(v: float) -> str:
        return f"{v:.6g}"

    lines = [header]
    for b in bodies:
        x_au = b.x_m / _AU
        y_au = b.y_m / _AU
        vx_km_s = b.vx_m_s / 1000.0
        vy_km_s = b.vy_m_s / 1000.0
        radius_km = b.radius_m / 1000.0
        lines.append(
            f"{b.name}, {fmt(b.mass_kg)}, {fmt(x_au)}, {fmt(y_au)}, "
            f"{fmt(vx_km_s)}, {fmt(vy_km_s)}, {fmt(radius_km)}, {b.color}"
        )
    return "\n".join(lines)
