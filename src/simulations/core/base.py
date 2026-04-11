from abc import ABC, abstractmethod
from typing import Generic, TypeVar

P = TypeVar("P")
R = TypeVar("R")


class Simulation(ABC, Generic[P, R]):
    """Base class for all simulations.

    Type parameters:
        P: the params dataclass type
        R: the result type (usually TimeSeriesResult or BatchResult)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable simulation name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of what this simulation models."""
        ...

    @abstractmethod
    def run(self, params: P) -> R:
        """Execute one simulation run and return results."""
        ...

    def run_batch(self, params: P, n_runs: int) -> list[R]:
        """Execute n_runs independent runs with the same params."""
        return [self.run(params) for _ in range(n_runs)]
