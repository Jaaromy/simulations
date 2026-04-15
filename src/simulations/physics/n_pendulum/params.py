from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any

@dataclass
class NPendulumParams:
    """Parameters for the real-time N-link pendulum simulation."""
    n_links: int = 2                     # number of links (1-5)
    masses: list[float] = field(default_factory=lambda: [1.0, 1.0])
    lengths: list[float] = field(default_factory=lambda: [1.0, 1.0])
    g: float = 9.81
    # List of per-pendulum ICs: each element is [theta1_deg, theta2_deg, ...]
    initial_conditions: list[list[float]] = field(
        default_factory=lambda: [[120.0, -60.0]]
    )
    # Angular velocities matching ICs (rad/s), default all zero
    initial_omegas: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0]]
    )
    trail_enabled: bool = True
    trail_length: int = 200              # number of frames to keep in trail
    canvas_width: int = 700
    canvas_height: int = 600
    dt: float = 1.0 / 240.0             # physics timestep (s)
    substeps: int = 2                    # physics steps per animation frame
    scale_px_per_m: float = 180.0       # rendering scale

    def validate(self) -> None:
        if not (1 <= self.n_links <= 5):
            raise ValueError(f"n_links must be 1-5, got {self.n_links}")
        if len(self.masses) != self.n_links:
            raise ValueError("len(masses) must equal n_links")
        if len(self.lengths) != self.n_links:
            raise ValueError("len(lengths) must equal n_links")
        for m in self.masses:
            if m <= 0:
                raise ValueError("all masses must be positive")
        for l in self.lengths:
            if l <= 0:
                raise ValueError("all lengths must be positive")
        if self.g < 0:
            raise ValueError("g must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n_links,
            "masses": self.masses,
            "lengths": self.lengths,
            "g": self.g,
            "pendulums": [
                {
                    "thetas": ic,
                    "omegas": self.initial_omegas[i] if i < len(self.initial_omegas) else [0.0] * self.n_links,
                }
                for i, ic in enumerate(self.initial_conditions)
            ],
            "dt": self.dt,
            "substeps": self.substeps,
            "trailEnabled": self.trail_enabled,
            "trailLength": self.trail_length,
            "scalePxPerM": self.scale_px_per_m,
            "canvasId": "c",
        }


def parse_ic_text(text: str, n_links: int, n_pendulums: int) -> tuple[list[list[float]], list[list[float]]]:
    """Parse IC textarea text into (initial_conditions, initial_omegas).

    Format: one line per pendulum, N comma-separated angle values in degrees.
    Lines with fewer than N values are zero-padded.
    Extra pendulums beyond text lines get tiny perturbations of line 0.

    Returns:
        (initial_conditions, initial_omegas) — omegas default to all zeros.
    """
    import random
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    parsed: list[list[float]] = []
    for line in lines[:n_pendulums]:
        parts = [float(x) for x in line.split(",")]
        while len(parts) < n_links:
            parts.append(0.0)
        parsed.append(parts[:n_links])

    # Pad missing pendulums with tiny perturbations of first IC
    base = parsed[0] if parsed else [90.0] * n_links
    while len(parsed) < n_pendulums:
        perturbed = [a + (random.random() - 0.5) * 0.2 for a in base]
        parsed.append(perturbed)

    omegas = [[0.0] * n_links for _ in parsed]
    return parsed, omegas
