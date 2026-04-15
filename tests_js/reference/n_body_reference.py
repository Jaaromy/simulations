"""
N-body gravitational reference implementation — pure Python + NumPy, Velocity-Verlet integration.

State per body: position (x, y), velocity (vx, vy), mass, radius, alive flag.

Algorithm (Velocity-Verlet):
    1. Compute accelerations a(t) for all alive pairs, softened:
           a_i = G * Σ_{j≠i} m_j * (r_j - r_i) / (|r_j - r_i|² + ε²)^(3/2)
    2. Half-step velocity:   v(t + dt/2) = v(t) + ½ a(t) dt
    3. Full-step position:   r(t + dt)   = r(t) + v(t + dt/2) dt
    4. Recompute accelerations a(t + dt)
    5. Half-step velocity:   v(t + dt)   = v(t + dt/2) + ½ a(t + dt) dt

Optional merging (merge=True):
    After each position update, check all alive pairs O(n²).
    If |r_i - r_j| < r_i + r_j, merge into the more massive body:
        new_mass     = m_i + m_j
        new_velocity = (m_i * v_i + m_j * v_j) / new_mass  (momentum conservation)
        new_radius   = (r_i³ + r_j³)^(1/3)                (volume conservation)
        new_position = position of heavier body
    The lighter body is marked alive=False.

Conserved quantities (no merge, no softening):
    Total energy:    E = ½ Σ_i m_i |v_i|² − G Σ_{i<j} m_i m_j / |r_ij|
    Linear momentum: p = Σ_i m_i v_i
    Angular momentum: L = Σ_i m_i (r_i × v_i)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _accelerations(
    x: np.ndarray,
    y: np.ndarray,
    mass: np.ndarray,
    alive: np.ndarray,
    G: float,
    softening: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute gravitational accelerations for all alive bodies.

    Returns (ax, ay) arrays of shape (n,); dead bodies get 0.
    """
    n = len(x)
    ax = np.zeros(n, dtype=np.float64)
    ay = np.zeros(n, dtype=np.float64)
    eps2 = softening * softening

    alive_idx = np.where(alive)[0]
    for i in alive_idx:
        for j in alive_idx:
            if i == j:
                continue
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dist2 = dx * dx + dy * dy + eps2
            inv_dist3 = dist2 ** (-1.5)
            factor = G * mass[j] * inv_dist3
            ax[i] += factor * dx
            ay[i] += factor * dy

    return ax, ay


def _check_merges(
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    mass: np.ndarray,
    radius: np.ndarray,
    alive: np.ndarray,
) -> None:
    """
    In-place merge: if two alive bodies overlap, merge the lighter into the heavier.
    Conserves momentum and volume. Mutates arrays in place.
    """
    n = len(x)
    # Repeat until no more merges (cascade)
    changed = True
    while changed:
        changed = False
        alive_idx = [i for i in range(n) if alive[i]]
        for ii in range(len(alive_idx)):
            i = alive_idx[ii]
            if not alive[i]:
                continue
            for jj in range(ii + 1, len(alive_idx)):
                j = alive_idx[jj]
                if not alive[j]:
                    continue
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < radius[i] + radius[j]:
                    # Merge: keep heavier, remove lighter
                    heavy, light = (i, j) if mass[i] >= mass[j] else (j, i)
                    new_mass = mass[heavy] + mass[light]
                    # Center-of-mass position
                    x[heavy] = (mass[heavy] * x[heavy] + mass[light] * x[light]) / new_mass
                    y[heavy] = (mass[heavy] * y[heavy] + mass[light] * y[light]) / new_mass
                    # Momentum conservation
                    vx[heavy] = (mass[heavy] * vx[heavy] + mass[light] * vx[light]) / new_mass
                    vy[heavy] = (mass[heavy] * vy[heavy] + mass[light] * vy[light]) / new_mass
                    # Volume conservation: r_new = (r_heavy³ + r_light³)^(1/3)
                    radius[heavy] = (radius[heavy] ** 3 + radius[light] ** 3) ** (1.0 / 3.0)
                    mass[heavy] = new_mass
                    alive[light] = False
                    changed = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(
    bodies: list[dict],
    dt: float,
    n_steps: int,
    G: float = 6.67430e-11,
    softening: float = 1e9,
    merge: bool = False,
) -> list[dict]:
    """
    Integrate N-body gravitational system for n_steps steps using Velocity-Verlet.

    Parameters
    ----------
    bodies    : list of dicts with keys: name, massKg, xM, yM, vxMs, vyMs, radiusM
    dt        : time step (s)
    n_steps   : number of integration steps
    G         : gravitational constant (m³ kg⁻¹ s⁻²)
    softening : softening length (m) — avoids singularity at close approach
    merge     : if True, merge bodies that physically overlap after each step

    Returns
    -------
    List of n_steps dicts (one per step, NOT including step 0):
        {"t": float, "bodies": [{"x": float, "y": float, "vx": float, "vy": float,
                                  "mass": float, "alive": bool}, ...]}
    """
    n = len(bodies)

    x = np.array([b["xM"] for b in bodies], dtype=np.float64)
    y = np.array([b["yM"] for b in bodies], dtype=np.float64)
    vx = np.array([b["vxMs"] for b in bodies], dtype=np.float64)
    vy = np.array([b["vyMs"] for b in bodies], dtype=np.float64)
    mass = np.array([b["massKg"] for b in bodies], dtype=np.float64)
    radius = np.array([b["radiusM"] for b in bodies], dtype=np.float64)
    alive = np.ones(n, dtype=bool)

    snapshots: list[dict] = []

    def _snapshot(t: float) -> dict:
        return {
            "t": t,
            "bodies": [
                {
                    "x": float(x[i]),
                    "y": float(y[i]),
                    "vx": float(vx[i]),
                    "vy": float(vy[i]),
                    "mass": float(mass[i]),
                    "alive": bool(alive[i]),
                }
                for i in range(n)
            ],
        }

    # Initial accelerations
    ax, ay = _accelerations(x, y, mass, alive, G, softening)

    for step in range(n_steps):
        t_next = (step + 1) * dt

        # Half-step velocity
        vx_half = vx + 0.5 * dt * ax
        vy_half = vy + 0.5 * dt * ay

        # Full-step position
        x = x + dt * vx_half
        y = y + dt * vy_half

        # Optional merge check
        if merge:
            _check_merges(x, y, vx_half, vy_half, mass, radius, alive)

        # Recompute accelerations at new position
        ax, ay = _accelerations(x, y, mass, alive, G, softening)

        # Half-step velocity again (complete the full step)
        vx = vx_half + 0.5 * dt * ax
        vy = vy_half + 0.5 * dt * ay

        snapshots.append(_snapshot(t_next))

    return snapshots


# ---------------------------------------------------------------------------
# Gold trajectory generation
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent

_SUN = dict(name="Sun", massKg=1.989e30, xM=0.0, yM=0.0, vxMs=0.0, vyMs=0.0, radiusM=6.957e8)
_EARTH = dict(name="Earth", massKg=5.972e24, xM=1.496e11, yM=0.0, vxMs=0.0, vyMs=29784.0, radiusM=6.371e6)
_LUNA = dict(
    name="Luna",
    massKg=7.342e22,
    xM=1.496e11 + 3.844e8,
    yM=0.0,
    vxMs=0.0,
    vyMs=29784.0 + 1022.0,
    radiusM=1.737e6,
)

_CASES = [
    dict(
        filename="gold_trajectory_2body.json",
        bodies=[_SUN, _EARTH],
        dt=3600.0,
        n_steps=1000,
        G=6.67430e-11,
        softening=1e9,
        merge=False,
        save_every=1,
    ),
    dict(
        filename="gold_trajectory_solar.json",
        bodies=[_SUN, _EARTH, _LUNA],
        dt=3600.0,
        n_steps=8760,
        G=6.67430e-11,
        softening=1e9,
        merge=False,
        save_every=24,
    ),
]


if __name__ == "__main__":
    for case in _CASES:
        fname = case.pop("filename")
        save_every = case.pop("save_every")
        bodies_cfg = case["bodies"]

        trajectory = simulate(**case)

        # Subsample: keep every save_every-th step
        saved = [trajectory[i] for i in range(0, len(trajectory), save_every)]

        config = {
            "dt": case["dt"],
            "n_steps": case["n_steps"],
            "G": case["G"],
            "softening": case["softening"],
            "merge": case["merge"],
            "save_every": save_every,
            "bodies": bodies_cfg,
        }

        out = {
            "config": config,
            "trajectory": saved,
        }

        outpath = _HERE / fname
        with open(outpath, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"Wrote {outpath} — {len(saved)} snapshots")
