"""
N-link pendulum reference implementation — pure Python + NumPy, RK4 integration.

Generalised coordinates: θ = (θ_0, …, θ_{N-1}) measured from downward vertical.
State vector: [θ_0, ω_0, θ_1, ω_1, ..., θ_{N-1}, ω_{N-1}]

Equations of motion: M(θ) α = −C(θ, ω) − G(θ)

Mass matrix M (N×N, 0-indexed):
    M[i][j] = (Σ_{k=max(i,j)}^{N-1} m_k) · l_i · l_j · cos(θ_i − θ_j)

Coriolis/centripetal vector C (N-element):
    C[i] = Σ_{j=0}^{N-1} (Σ_{k=max(i,j)}^{N-1} m_k) · l_i · l_j · sin(θ_i − θ_j) · ω_j²

Gravity vector G (N-element):
    G[i] = (Σ_{k=i}^{N-1} m_k) · g · l_i · sin(θ_i)

Kinetic energy:
    T = ½ ωᵀ M ω
      = ½ Σ_i Σ_j (Σ_{k≥max(i,j)} m_k) · l_i · l_j · ω_i · ω_j · cos(θ_i − θ_j)

Potential energy (pivot at origin, y upward):
    V = −Σ_i (Σ_{k≥i} m_k) · g · l_i · cos(θ_i)

Bob Cartesian positions:
    x[i] = Σ_{j=0}^{i} l_j · sin(θ_j)
    y[i] = −Σ_{j=0}^{i} l_j · cos(θ_j)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Suffix-mass lookup table  tail_mass[k] = Σ_{j=k}^{N-1} m_j
# ---------------------------------------------------------------------------

def _tail_masses(masses: list[float]) -> np.ndarray:
    """Return tail_mass[k] = sum of masses from index k to end (inclusive)."""
    m = np.array(masses, dtype=np.float64)
    return np.cumsum(m[::-1])[::-1]


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------

def _build_M(thetas: np.ndarray, lengths: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Build N×N mass matrix."""
    n = len(thetas)
    M = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            k = max(i, j)
            M[i, j] = tail[k] * lengths[i] * lengths[j] * math.cos(thetas[i] - thetas[j])
    return M


def _build_C(thetas: np.ndarray, omegas: np.ndarray, lengths: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Build N-element Coriolis/centripetal vector."""
    n = len(thetas)
    C = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            k = max(i, j)
            C[i] += tail[k] * lengths[i] * lengths[j] * math.sin(thetas[i] - thetas[j]) * omegas[j] ** 2
    return C


def _build_G(thetas: np.ndarray, lengths: np.ndarray, tail: np.ndarray, g: float) -> np.ndarray:
    """Build N-element gravity vector."""
    n = len(thetas)
    G = np.zeros(n, dtype=np.float64)
    for i in range(n):
        G[i] = tail[i] * g * lengths[i] * math.sin(thetas[i])
    return G


def _derivs(
    state: np.ndarray,
    lengths: np.ndarray,
    tail: np.ndarray,
    g: float,
) -> np.ndarray:
    """
    Return derivative of state vector [θ_0, ω_0, ..., θ_{N-1}, ω_{N-1}].

    dθ_i/dt = ω_i
    dω_i/dt = α_i  where M α = -(C + G)
    """
    n = len(lengths)
    thetas = state[0::2]
    omegas = state[1::2]

    M = _build_M(thetas, lengths, tail)
    C = _build_C(thetas, omegas, lengths, tail)
    G = _build_G(thetas, lengths, tail, g)

    rhs = -(C + G)
    alphas = np.linalg.solve(M, rhs)

    dstate = np.empty_like(state)
    dstate[0::2] = omegas
    dstate[1::2] = alphas
    return dstate


def _energy(
    thetas: np.ndarray,
    omegas: np.ndarray,
    lengths: np.ndarray,
    tail: np.ndarray,
    g: float,
) -> float:
    """
    Total mechanical energy E = T + V.

    T = ½ ωᵀ M ω
    V = −Σ_i tail[i] · g · l_i · cos(θ_i)
    """
    M = _build_M(thetas, lengths, tail)
    T = 0.5 * float(omegas @ M @ omegas)
    V = float(-np.sum(tail * g * lengths * np.cos(thetas)))
    return T + V


def _cartesian(thetas: np.ndarray, lengths: np.ndarray) -> tuple[list[float], list[float]]:
    """
    Compute Cartesian positions of each bob.

    x[i] = Σ_{j=0}^{i} l_j sin(θ_j)
    y[i] = −Σ_{j=0}^{i} l_j cos(θ_j)
    """
    n = len(thetas)
    x = [0.0] * n
    y = [0.0] * n
    cx, cy = 0.0, 0.0
    for i in range(n):
        cx += lengths[i] * math.sin(thetas[i])
        cy -= lengths[i] * math.cos(thetas[i])
        x[i] = cx
        y[i] = cy
    return x, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(
    n: int,
    masses: list[float],
    lengths: list[float],
    g: float,
    thetas0_deg: list[float],
    omegas0: list[float],
    dt: float,
    n_steps: int,
) -> list[dict]:
    """
    Integrate an N-link pendulum for n_steps steps using 4th-order Runge-Kutta.

    Parameters
    ----------
    n          : number of links
    masses     : bob masses [m_0, …, m_{N-1}] (kg)
    lengths    : rod lengths [l_0, …, l_{N-1}] (m)
    g          : gravitational acceleration (m/s²)
    thetas0_deg: initial angles in degrees (from downward vertical)
    omegas0    : initial angular velocities (rad/s)
    dt         : time step (s)
    n_steps    : number of integration steps

    Returns
    -------
    List of (n_steps + 1) snapshot dicts:
        {"t": float, "thetas": [...], "omegas": [...], "x": [...], "y": [...], "energy": float}
    """
    assert len(masses) == n
    assert len(lengths) == n
    assert len(thetas0_deg) == n
    assert len(omegas0) == n

    lengths_arr = np.array(lengths, dtype=np.float64)
    tail = _tail_masses(masses)

    # Build initial state [θ_0, ω_0, θ_1, ω_1, ...]
    state = np.empty(2 * n, dtype=np.float64)
    state[0::2] = np.deg2rad(thetas0_deg)
    state[1::2] = omegas0

    snapshots: list[dict] = []

    def _snapshot(t: float, st: np.ndarray) -> dict:
        thetas = st[0::2]
        omegas = st[1::2]
        x, y = _cartesian(thetas, lengths_arr)
        e = _energy(thetas, omegas, lengths_arr, tail, g)
        return {
            "t": t,
            "thetas": thetas.tolist(),
            "omegas": omegas.tolist(),
            "x": x,
            "y": y,
            "energy": e,
        }

    snapshots.append(_snapshot(0.0, state))

    for step in range(n_steps):
        t = step * dt
        k1 = _derivs(state, lengths_arr, tail, g)
        k2 = _derivs(state + 0.5 * dt * k1, lengths_arr, tail, g)
        k3 = _derivs(state + 0.5 * dt * k2, lengths_arr, tail, g)
        k4 = _derivs(state + dt * k3, lengths_arr, tail, g)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        snapshots.append(_snapshot((step + 1) * dt, state))

    return snapshots


# ---------------------------------------------------------------------------
# Gold trajectory generation
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent

_CASES = [
    dict(
        filename="gold_trajectory_n2.json",
        n=2,
        masses=[1.0, 1.0],
        lengths=[1.0, 1.0],
        g=9.81,
        thetas0_deg=[120.0, -60.0],
        omegas0=[0.0, 0.0],
        dt=1.0 / 240,
        n_steps=1200,
    ),
    dict(
        filename="gold_trajectory_n3.json",
        n=3,
        masses=[1.0, 1.0, 1.0],
        lengths=[0.8, 0.7, 0.6],
        g=9.81,
        thetas0_deg=[120.0, -60.0, 45.0],
        omegas0=[0.0, 0.0, 0.0],
        dt=1.0 / 240,
        n_steps=1200,
    ),
]


if __name__ == "__main__":
    for case in _CASES:
        fname = case.pop("filename")
        data = simulate(**case)
        out = _HERE / fname
        with open(out, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"Wrote {out} — {len(data)} snapshots")
