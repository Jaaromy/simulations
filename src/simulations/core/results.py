from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TimeSeriesResult:
    """Result for simulations that produce a value over discrete time steps.

    steps:  x-axis — hand number, generation, tick, etc.
    values: y-axis — bankroll, population size, etc.
    """

    simulation_name: str
    params_snapshot: dict[str, Any]
    steps: np.ndarray
    values: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_value(self) -> float:
        """Last value in the series."""
        return float(self.values[-1])

    @property
    def net_change(self) -> float:
        """Difference between last and first value."""
        return float(self.values[-1] - self.values[0])


@dataclass
class BatchResult:
    """Container for multiple independent runs of the same simulation."""

    simulation_name: str
    runs: list[TimeSeriesResult]

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def final_values(self) -> np.ndarray:
        """Array of final values across all runs."""
        return np.array([r.final_value for r in self.runs])

    def mean_trajectory(self) -> np.ndarray:
        """Element-wise mean across all run value arrays.
        Assumes all runs have the same number of steps."""
        return np.mean(np.stack([r.values for r in self.runs]), axis=0)

    def std_trajectory(self) -> np.ndarray:
        """Element-wise std deviation across all run value arrays."""
        return np.std(np.stack([r.values for r in self.runs]), axis=0)
