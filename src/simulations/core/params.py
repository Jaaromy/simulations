from dataclasses import asdict
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SimulationParams(Protocol):
    """Protocol that all simulation param dataclasses must satisfy."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize params to a plain dict (for snapshotting in results)."""
        ...

    def validate(self) -> None:
        """Raise ValueError if params are invalid."""
        ...


def params_to_dict(params: object) -> dict[str, Any]:
    """Helper: serialize any dataclass to dict via dataclasses.asdict."""
    return asdict(params)  # type: ignore[arg-type]
