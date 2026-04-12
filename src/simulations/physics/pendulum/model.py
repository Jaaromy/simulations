"""
Double pendulum — exact Lagrangian equations of motion, RK4 integration.

Coordinate convention
---------------------
θ₁  angle of upper rod from the downward vertical (rad, +ve counterclockwise)
θ₂  angle of lower rod from the downward vertical (rad, +ve counterclockwise)
ω₁  = dθ₁/dt  (rad/s)
ω₂  = dθ₂/dt  (rad/s)

Pivot is at the origin; y increases upward.
Bob positions:
    x₁ = l₁ sin θ₁,         y₁ = −l₁ cos θ₁
    x₂ = l₁ sin θ₁ + l₂ sin θ₂,   y₂ = −l₁ cos θ₁ − l₂ cos θ₂

Equations of motion (Wikipedia "Double pendulum", exact non-linear form)
------------------------------------------------------------------------
Let Δ = θ₁ − θ₂.

    D = 2m₁ + m₂ − m₂ cos(2Δ)   (= 2(m₁ + m₂ sin²Δ))

    θ̈₁ = [−g(2m₁+m₂)sinθ₁ − m₂g sin(θ₁−2θ₂)
           − 2 sinΔ · m₂(ω₂²l₂ + ω₁²l₁ cosΔ)] / (l₁ D)

    θ̈₂ = [2 sinΔ · (ω₁²l₁(m₁+m₂) + g(m₁+m₂)cosθ₁
           + ω₂²l₂ m₂ cosΔ)] / (l₂ D)

Conserved quantity — total mechanical energy
--------------------------------------------
    T = ½(m₁+m₂)l₁²ω₁² + ½m₂l₂²ω₂² + m₂l₁l₂ω₁ω₂cos(θ₁−θ₂)
    V = −(m₁+m₂)g l₁ cosθ₁ − m₂g l₂ cosθ₂
    E = T + V  (conserved in exact Hamiltonian dynamics)

For fixed-step RK4 at dt = 0.001 s with moderate initial conditions, the
fractional energy drift over 30 s is typically < 0.1 %.

Normal-mode frequencies (small-angle limit, m₁ = m₂ = m, l₁ = l₂ = l)
----------------------------------------------------------------------
    ω²± = (g/l)(2 ± √2)

    Lower mode (−): θ₂ / θ₁ = +√2  (bobs swing together)
    Upper mode (+): θ₂ / θ₁ = −√2  (bobs swing in opposition)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulations.core.base import Simulation
from simulations.physics.pendulum.params import PendulumParams


@dataclass
class PendulumResult:
    """Result of one double-pendulum integration run."""

    simulation_name: str
    params_snapshot: dict[str, Any]
    t: np.ndarray        # (n,) time in seconds
    theta1: np.ndarray   # (n,) upper bob angle, radians
    theta2: np.ndarray   # (n,) lower bob angle, radians
    omega1: np.ndarray   # (n,) upper bob angular velocity, rad/s
    omega2: np.ndarray   # (n,) lower bob angular velocity, rad/s
    x1: np.ndarray       # (n,) upper bob x-position, m
    y1: np.ndarray       # (n,) upper bob y-position, m
    x2: np.ndarray       # (n,) lower bob x-position, m
    y2: np.ndarray       # (n,) lower bob y-position, m
    energy: np.ndarray   # (n,) total mechanical energy, J


class DoublePendulumSimulation(Simulation[PendulumParams, PendulumResult]):
    """Double pendulum integrated with 4th-order Runge-Kutta."""

    @property
    def name(self) -> str:
        return "DoublePendulum"

    @property
    def description(self) -> str:
        return "Chaotic double pendulum — exact Lagrangian EOMs, RK4 integration."

    def run(self, params: PendulumParams) -> PendulumResult:
        params.validate()

        m1, m2 = params.m1, params.m2
        l1, l2 = params.l1, params.l2
        g = params.g
        dt = params.dt

        # Convert initial angles from degrees to radians
        th1 = np.deg2rad(params.theta1_0)
        th2 = np.deg2rad(params.theta2_0)
        om1 = params.omega1_0
        om2 = params.omega2_0

        n_steps = int(round(params.t_end / dt))

        # Allocate full-resolution storage
        th1_arr = np.empty(n_steps + 1)
        th2_arr = np.empty(n_steps + 1)
        om1_arr = np.empty(n_steps + 1)
        om2_arr = np.empty(n_steps + 1)
        th1_arr[0] = th1
        th2_arr[0] = th2
        om1_arr[0] = om1
        om2_arr[0] = om2

        # RK4 integration
        state = np.array([th1, om1, th2, om2], dtype=np.float64)
        for i in range(n_steps):
            k1 = self._derivs(state, m1, m2, l1, l2, g)
            k2 = self._derivs(state + 0.5 * dt * k1, m1, m2, l1, l2, g)
            k3 = self._derivs(state + 0.5 * dt * k2, m1, m2, l1, l2, g)
            k4 = self._derivs(state + dt * k3, m1, m2, l1, l2, g)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            th1_arr[i + 1] = state[0]
            om1_arr[i + 1] = state[1]
            th2_arr[i + 1] = state[2]
            om2_arr[i + 1] = state[3]

        # Downsample to n_snapshots evenly-spaced indices
        idx = np.round(np.linspace(0, n_steps, params.n_snapshots)).astype(int)
        idx = np.clip(idx, 0, n_steps)

        t = idx * dt
        th1_s = th1_arr[idx]
        th2_s = th2_arr[idx]
        om1_s = om1_arr[idx]
        om2_s = om2_arr[idx]

        # Cartesian positions
        x1_s = l1 * np.sin(th1_s)
        y1_s = -l1 * np.cos(th1_s)
        x2_s = x1_s + l2 * np.sin(th2_s)
        y2_s = y1_s - l2 * np.cos(th2_s)

        energy_s = self._energy(th1_s, th2_s, om1_s, om2_s, m1, m2, l1, l2, g)

        return PendulumResult(
            simulation_name=self.name,
            params_snapshot=params.to_dict(),
            t=t,
            theta1=th1_s,
            theta2=th2_s,
            omega1=om1_s,
            omega2=om2_s,
            x1=x1_s,
            y1=y1_s,
            x2=x2_s,
            y2=y2_s,
            energy=energy_s,
        )

    @staticmethod
    def _derivs(
        state: np.ndarray,
        m1: float,
        m2: float,
        l1: float,
        l2: float,
        g: float,
    ) -> np.ndarray:
        """Return [dθ₁/dt, dω₁/dt, dθ₂/dt, dω₂/dt] for state [θ₁, ω₁, θ₂, ω₂]."""
        th1, om1, th2, om2 = state
        delta = th1 - th2

        # Shared denominator (Wikipedia form): D = 2m₁+m₂ − m₂cos(2Δ)
        D = 2.0 * m1 + m2 - m2 * np.cos(2.0 * delta)

        # θ̈₁
        num1 = (
            -g * (2.0 * m1 + m2) * np.sin(th1)
            - m2 * g * np.sin(th1 - 2.0 * th2)
            - 2.0 * np.sin(delta) * m2 * (om2 ** 2 * l2 + om1 ** 2 * l1 * np.cos(delta))
        )
        alpha1 = num1 / (l1 * D)

        # θ̈₂
        num2 = 2.0 * np.sin(delta) * (
            om1 ** 2 * l1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(th1)
            + om2 ** 2 * l2 * m2 * np.cos(delta)
        )
        alpha2 = num2 / (l2 * D)

        return np.array([om1, alpha1, om2, alpha2])

    @staticmethod
    def _energy(
        th1: np.ndarray,
        th2: np.ndarray,
        om1: np.ndarray,
        om2: np.ndarray,
        m1: float,
        m2: float,
        l1: float,
        l2: float,
        g: float,
    ) -> np.ndarray:
        """Total mechanical energy E = T + V (vectorised over snapshot arrays)."""
        delta = th1 - th2
        T = (
            0.5 * (m1 + m2) * l1 ** 2 * om1 ** 2
            + 0.5 * m2 * l2 ** 2 * om2 ** 2
            + m2 * l1 * l2 * om1 * om2 * np.cos(delta)
        )
        V = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
        return T + V
