"""
2D Ising model with Metropolis dynamics.

Checkerboard (sublattice) decomposition for vectorised updates:
- Colour the grid like a chess board: (i+j)%2 == 0 → sublattice A.
- All neighbours of an A-site are B-sites and vice versa.
- Flip all A-sites simultaneously (they share no bonds), then all B-sites.
- This gives fully vectorised Metropolis sweeps without sequential dependencies.

ΔE for flipping spin s at site i:
    ΔE = 2 · s_i · Σ_{j∈nn(i)} s_j   (J = 1)

Accept if ΔE ≤ 0; else accept with probability exp(−ΔE / T).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulations.core.base import Simulation
from simulations.physics.ising.params import IsingParams


@dataclass
class IsingResult:
    """Result of one Ising simulation run."""

    simulation_name: str
    params_snapshot: dict[str, Any]
    steps: np.ndarray           # sweep indices at which snapshots were taken
    magnetization: np.ndarray   # |mean spin| at each snapshot ∈ [0, 1]
    energy_per_spin: np.ndarray # E/N at each snapshot ∈ [−2, +2]
    final_grid: np.ndarray      # L × L int8 spin grid at end of run


class IsingSimulation(Simulation[IsingParams, IsingResult]):
    """2D ferromagnetic Ising model on a periodic square lattice."""

    @property
    def name(self) -> str:
        return "Ising"

    @property
    def description(self) -> str:
        return "2D ferromagnetic Ising model with Metropolis dynamics."

    def run(self, params: IsingParams) -> IsingResult:
        params.validate()
        rng = np.random.default_rng(params.seed)
        L = params.grid_size
        T = params.temperature

        # --- Initialise grid ---
        if params.initial_state == "random":
            grid = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
        elif params.initial_state == "aligned_up":
            grid = np.ones((L, L), dtype=np.int8)
        else:  # aligned_down — validate() already blocked anything else
            grid = -np.ones((L, L), dtype=np.int8)

        # --- Precompute sublattice (checkerboard) indices ---
        ii, jj = np.indices((L, L))
        parity = (ii + jj) % 2
        sublattices = [
            (ii[parity == c].ravel(), jj[parity == c].ravel())
            for c in (0, 1)
        ]

        # --- Snapshot schedule: n_snapshots evenly spaced, always include 0 ---
        snap_at = set(
            np.round(np.linspace(0, params.n_sweeps, params.n_snapshots)).astype(int)
        )
        snap_at.add(0)
        snap_schedule = sorted(snap_at)

        snap_ptr = 0  # index into snap_schedule
        snap_steps: list[int] = []
        snap_mag: list[float] = []
        snap_energy: list[float] = []

        def _record(sweep: int) -> None:
            snap_steps.append(sweep)
            snap_mag.append(float(np.abs(grid.mean())))
            snap_energy.append(self._energy_per_spin(grid))

        # Record sweep 0 before any updates
        if snap_ptr < len(snap_schedule) and snap_schedule[snap_ptr] == 0:
            _record(0)
            snap_ptr += 1

        # --- Main Metropolis loop ---
        for sweep in range(1, params.n_sweeps + 1):
            for si, sj in sublattices:
                # Neighbour sum with periodic boundary conditions
                ns = (
                    grid[(si - 1) % L, sj].astype(np.int16)
                    + grid[(si + 1) % L, sj].astype(np.int16)
                    + grid[si, (sj - 1) % L].astype(np.int16)
                    + grid[si, (sj + 1) % L].astype(np.int16)
                )
                # ΔE = 2 · s_i · neighbor_sum
                delta_e = 2.0 * grid[si, sj].astype(np.float64) * ns

                # Accept downhill moves unconditionally; uphill with Boltzmann prob
                accept = delta_e <= 0.0
                uphill = ~accept
                if uphill.any():
                    rand = rng.random(int(uphill.sum()))
                    accept[uphill] = rand < np.exp(-delta_e[uphill] / T)

                grid[si[accept], sj[accept]] *= -1

            # Record snapshot if this sweep is in the schedule
            if snap_ptr < len(snap_schedule) and snap_schedule[snap_ptr] == sweep:
                _record(sweep)
                snap_ptr += 1

        return IsingResult(
            simulation_name=self.name,
            params_snapshot=params.to_dict(),
            steps=np.array(snap_steps, dtype=np.int64),
            magnetization=np.array(snap_mag),
            energy_per_spin=np.array(snap_energy),
            final_grid=grid,
        )

    @staticmethod
    def _energy_per_spin(grid: np.ndarray) -> float:
        """
        E/N = −(1/N) Σ_{<ij>} s_i · s_j   (J = 1, each bond counted once).

        Implemented as the sum over all rightward + downward bonds divided by N.
        np.roll shifts the grid by one position so element-wise products give
        nearest-neighbour products along each axis.
        """
        N = grid.size
        bonds = (
            np.sum(grid * np.roll(grid, 1, axis=0))  # vertical bonds
            + np.sum(grid * np.roll(grid, 1, axis=1))  # horizontal bonds
        )
        return float(-bonds) / N
