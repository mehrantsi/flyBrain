from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isclose


@dataclass(frozen=True, slots=True)
class ModelParameters:
    dt_ms: float = 0.1
    resting_mv: float = -52.0
    reset_mv: float = -52.0
    threshold_mv: float = -45.0
    membrane_tau_ms: float = 20.0
    synapse_tau_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    synapse_weight_mv: float = 0.275
    poisson_rate_hz: float = 150.0
    poisson_rate_2_hz: float = 0.0
    poisson_weight_scale: int = 250

    def __post_init__(self) -> None:
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        for name in ("membrane_tau_ms", "synapse_tau_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.refractory_ms < 0 or self.delay_ms < 0:
            raise ValueError("refractory_ms and delay_ms cannot be negative")
        self._exact_steps(self.delay_ms, "delay_ms")
        self._exact_steps(self.refractory_ms, "refractory_ms")

    def _exact_steps(self, duration_ms: float, name: str) -> int:
        steps = round(duration_ms / self.dt_ms)
        if not isclose(steps * self.dt_ms, duration_ms, abs_tol=1e-9):
            raise ValueError(f"{name} must be an integer multiple of dt_ms")
        return steps

    @property
    def delay_steps(self) -> int:
        return self._exact_steps(self.delay_ms, "delay_ms")

    @property
    def refractory_steps(self) -> int:
        return self._exact_steps(self.refractory_ms, "refractory_ms")

    @property
    def membrane_decay(self) -> float:
        return exp(-self.dt_ms / self.membrane_tau_ms)

    @property
    def synapse_decay(self) -> float:
        return exp(-self.dt_ms / self.synapse_tau_ms)

    @property
    def membrane_synapse_coupling(self) -> float:
        membrane = self.membrane_decay
        synapse = self.synapse_decay
        return (
            self.synapse_tau_ms
            / (self.membrane_tau_ms - self.synapse_tau_ms)
            * (membrane - synapse)
        )

    @property
    def poisson_weight_mv(self) -> float:
        return self.synapse_weight_mv * self.poisson_weight_scale

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)
