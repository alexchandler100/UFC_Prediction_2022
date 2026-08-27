"""Named, direct-index random streams for reproducible simulation paths."""

from __future__ import annotations

from dataclasses import is_dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from .domain import RngDraw, SimulationRunSpec


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        if hasattr(value, "to_dict"):
            return _jsonable(value.to_dict())
        return {
            name: _jsonable(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON cannot contain non-finite floats")
        return value
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_string_bytes(value: object) -> bytes:
    return json.dumps(
        str(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def run_id_for(spec: SimulationRunSpec) -> str:
    return f"sim-{sha256_hex(spec.to_dict())[:24]}"


def _display(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


class NamedRandomStreams:
    """Independent PCG64DXSM generators derived from a stable SHA-256 contract.

    A path's random sequence depends on the run contract, bootstrap member,
    simulation index, and stream name only.  It therefore cannot change with
    worker count, chunking, task order, or whether trace recording is enabled.
    """

    def __init__(
        self,
        spec: SimulationRunSpec,
        simulation_index: int,
        *,
        record: bool = False,
    ) -> None:
        if simulation_index < 0:
            raise ValueError("simulation_index must be nonnegative")
        self.spec = spec
        self.simulation_index = int(simulation_index)
        self.record = bool(record)
        self._generators: dict[str, np.random.Generator] = {}
        self._indices: dict[str, int] = {}
        self._pending: list[RngDraw] = []
        # The seed contract is a flat object with a fixed sorted-key order.
        # Construct its invariant portion once per path instead of recursively
        # normalizing and sorting the same six fields for every named stream.
        self._seed_prefix = b"".join(
            (
                b'{"bootstrap_member":',
                str(int(spec.bootstrap_member)).encode("ascii"),
                b',"matchup_id":',
                _canonical_string_bytes(spec.bout.matchup_id),
                b',"parameter_artifact_id":',
                _canonical_string_bytes(spec.parameter_artifact_id),
                b',"rng_contract":',
                _canonical_string_bytes(spec.rng_contract),
                b',"root_seed":',
                _canonical_string_bytes(spec.root_seed),
                b',"simulation_index":',
                str(self.simulation_index).encode("ascii"),
                b',"stream":',
            )
        )

    def _generator(self, stream: str) -> np.random.Generator:
        if not stream or not stream.strip():
            raise ValueError("stream name is required")
        if stream not in self._generators:
            digest = hashlib.sha256(
                self._seed_prefix + _canonical_string_bytes(stream) + b"}"
            ).digest()
            entropy = [
                int.from_bytes(digest[offset : offset + 4], "little")
                for offset in range(0, 32, 4)
            ]
            seed = np.random.SeedSequence(entropy)
            self._generators[stream] = np.random.Generator(np.random.PCG64DXSM(seed))
            self._indices[stream] = 0
        return self._generators[stream]

    def _record(
        self,
        stream: str,
        distribution: str,
        parameters: dict[str, Any],
        value: Any,
    ) -> None:
        index = self._indices[stream]
        self._indices[stream] = index + 1
        if self.record:
            self._pending.append(
                RngDraw(
                    stream=stream,
                    draw_index=index,
                    distribution=distribution,
                    parameters=tuple(
                        (key, _display(item)) for key, item in sorted(parameters.items())
                    ),
                    value=_display(value),
                )
            )

    def uniform(self, stream: str) -> float:
        generator = self._generator(stream)
        value = float(generator.random())
        self._record(stream, "uniform", {}, value)
        return value

    def exponential(self, stream: str, rate_per_second: float) -> float:
        if not math.isfinite(rate_per_second) or rate_per_second <= 0:
            raise ValueError("exponential rate must be finite and positive")
        generator = self._generator(stream)
        scale = 1.0 / rate_per_second
        value = float(generator.exponential(scale))
        self._record(stream, "exponential", {"rate_per_second": rate_per_second}, value)
        return value

    def normal(self, stream: str, mean: float = 0.0, sd: float = 1.0) -> float:
        if not math.isfinite(mean) or not math.isfinite(sd) or sd < 0:
            raise ValueError("normal parameters must be finite with nonnegative sd")
        generator = self._generator(stream)
        value = float(generator.normal(mean, sd))
        self._record(stream, "normal", {"mean": mean, "sd": sd}, value)
        return value

    def binomial(self, stream: str, trials: int, probability: float) -> int:
        if trials < 0 or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("binomial requires nonnegative trials and probability in [0, 1]")
        generator = self._generator(stream)
        value = int(generator.binomial(int(trials), float(probability)))
        self._record(
            stream,
            "binomial",
            {"trials": trials, "probability": probability},
            value,
        )
        return value

    def weighted_choice(
        self,
        stream: str,
        weighted_values: Iterable[tuple[Any, float]],
    ) -> Any:
        values = tuple(weighted_values)
        if not values:
            raise ValueError("weighted choice requires values")
        weights = [float(weight) for _, weight in values]
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("weighted choice weights must be finite and nonnegative")
        total = math.fsum(weights)
        if total <= 0:
            raise ValueError("weighted choice requires positive total weight")
        draw = self.uniform(stream) * total
        cumulative = 0.0
        for value, weight in values:
            cumulative += weight
            if draw < cumulative:
                return value
        return values[-1][0]

    def drain(self) -> tuple[RngDraw, ...]:
        pending = tuple(self._pending)
        self._pending.clear()
        return pending

    @property
    def draw_counts(self) -> dict[str, int]:
        return dict(self._indices)
