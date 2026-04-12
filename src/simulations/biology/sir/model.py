"""
SIR epidemic model — compartmental ODE system, RK4 integration.

Model equations
---------------
Three compartments: S (susceptible), I (infectious), R (recovered).
N = S + I + R is conserved.

    dS/dt = −β S I / N
    dI/dt =  β S I / N  − γ I
    dR/dt =  γ I

Parameters:
    β  — transmission rate (day⁻¹): contact rate × probability of transmission
    γ  — recovery rate (day⁻¹): 1/γ is the mean infectious period

Key analytical results
----------------------

Basic reproduction number:
    R₀ = β / γ

    R₀ > 1 → epidemic grows from small I.
    R₀ ≤ 1 → I decreases monotonically from any IC with S < N.

Epidemic threshold:
    The epidemic peak occurs when dI/dt = 0, i.e. when S* = γN/β = N/R₀.
    If S₀/N < 1/R₀ the peak is never reached — no epidemic.

Conservation:
    S(t) + I(t) + R(t) = N  exactly (analytically).
    RK4 maintains this to within O(dt⁴).

β = 0 limit (no transmission):
    dI/dt = −γ I  →  I(t) = I₀ exp(−γ t)
    Exact, closed-form; used as a deterministic test anchor.

Final epidemic size (implicit equation for R∞ = R(∞)):
    S∞ = N exp(−R₀ · R∞/N)
    → R∞ = N − S∞ − 0  (I∞ = 0 by definition)

Phase-plane invariant:
    dI/dS = −1 + N/(R₀ S)
    → I = −S + (N/R₀) ln S + C   where C = I₀ + S₀ − (N/R₀) ln S₀
    This curve is the exact phase-plane trajectory regardless of time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from simulations.core.base import Simulation
from simulations.biology.sir.params import SIRParams


@dataclass
class SIRResult:
    """Result of one SIR simulation run."""

    simulation_name: str
    params_snapshot: dict[str, Any]
    t: np.ndarray   # (n,) time in days
    S: np.ndarray   # (n,) susceptible count
    I: np.ndarray   # (n,) infectious count
    R: np.ndarray   # (n,) recovered count
    R0: float       # basic reproduction number β/γ


class SIRSimulation(Simulation[SIRParams, SIRResult]):
    """SIR compartmental epidemic model integrated with 4th-order Runge-Kutta."""

    @property
    def name(self) -> str:
        return "SIR"

    @property
    def description(self) -> str:
        return "SIR epidemic model — β/γ threshold, herd immunity, final size."

    def run(self, params: SIRParams) -> SIRResult:
        params.validate()

        N = params.N
        beta = params.beta
        gamma = params.gamma
        dt = params.dt

        # Initial conditions
        S0 = N - params.I0
        I0 = params.I0
        R0_val = 0.0

        n_steps = int(round(params.t_end / dt))

        # Allocate full-resolution arrays
        S_arr = np.empty(n_steps + 1)
        I_arr = np.empty(n_steps + 1)
        R_arr = np.empty(n_steps + 1)
        S_arr[0] = S0
        I_arr[0] = I0
        R_arr[0] = R0_val

        state = np.array([S0, I0, R0_val], dtype=np.float64)
        for i in range(n_steps):
            k1 = self._derivs(state, beta, gamma, N)
            k2 = self._derivs(state + 0.5 * dt * k1, beta, gamma, N)
            k3 = self._derivs(state + 0.5 * dt * k2, beta, gamma, N)
            k4 = self._derivs(state + dt * k3, beta, gamma, N)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # Clamp to [0, N] to prevent tiny numerical negatives
            state = np.clip(state, 0.0, N)
            S_arr[i + 1] = state[0]
            I_arr[i + 1] = state[1]
            R_arr[i + 1] = state[2]

        # Downsample to n_snapshots
        idx = np.round(np.linspace(0, n_steps, params.n_snapshots)).astype(int)
        idx = np.clip(idx, 0, n_steps)

        return SIRResult(
            simulation_name=self.name,
            params_snapshot=params.to_dict(),
            t=idx * dt,
            S=S_arr[idx],
            I=I_arr[idx],
            R=R_arr[idx],
            R0=params.R0,
        )

    @staticmethod
    def _derivs(state: np.ndarray, beta: float, gamma: float, N: float) -> np.ndarray:
        """Return [dS/dt, dI/dt, dR/dt]."""
        S, I, _R = state
        force_of_infection = beta * S * I / N
        return np.array([
            -force_of_infection,
             force_of_infection - gamma * I,
             gamma * I,
        ])
