"""Causal, strongly pooled parameter estimation for the fight simulator.

The fitter consumes the mirrored UFCStats bout table (one row per fighter-side)
and, when available, the normalized per-round table.  Every public fitting and
snapshot operation uses a strict ``fight_date < as_of`` cutoff.  Bootstrap
members resample complete event cards, preserving the dependence between the
two sides of a bout and between bouts held at the same event.

This is deliberately an empirical-Bayes first pass.  Sparse fighter estimates
are shrunk heavily toward division/era and global observations.  The resulting
bootstrap spread is a parameter/model uncertainty interval, not a Bayesian
credible interval.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from fight_semantics import method_bucket


PARAMETER_SCHEMA_VERSION = 3
PARAMETER_MODEL_VERSION = "fight-sim-empirical-bayes-card-bootstrap-v3"
TAKEDOWN_CONTROL_PARAMETER_MODEL_VERSION = (
    "fight-sim-empirical-bayes-card-bootstrap-v3+td-control-association-v1"
)
TAKEDOWN_CONTROL_GLOBAL_PRIOR_OPPORTUNITIES = 25.0
TAKEDOWN_CONTROL_CONTEXT_PRIOR_OPPORTUNITIES = 25.0
TAKEDOWN_CONTROL_FIGHTER_PRIOR_OPPORTUNITIES = 12.0
LEGACY_PARAMETER_SCHEMAS = frozenset({2})
PARAMETER_STORAGE_FORMAT = "fight-sim.parameter-ensemble"
PARAMETER_STORAGE_VERSION = 1
PARAMETER_NAMES = (
    "strike_rate_distance",
    "strike_rate_clinch",
    "strike_rate_ground",
    "distance_phase_share",
    "clinch_phase_share",
    "ground_phase_share",
    "strike_accuracy",
    "strike_defense",
    "head_target_share",
    "body_target_share",
    "leg_target_share",
    "strike_power",
    "knockdown_rate_per_landed",
    "finish_after_knockdown",
    "clinch_entry_rate",
    "clinch_exit_rate",
    "takedown_attempt_rate",
    "takedown_accuracy",
    "takedown_defense",
    "ground_control_rate",
    "escape_rate",
    "reversal_after_escape",
    "submission_attempt_rate",
    "submission_finish_probability",
    "submission_defense",
    "ko_resistance",
    "hurt_recovery_per_minute",
    "pace_decay",
    "stamina_recovery_between_rounds",
)

SNAPSHOT_PARAMETER_MODES = frozenset(
    {"full", "context_only", "reliability_weighted"}
)

_SNAPSHOT_COMPOSITION_GROUPS = (
    ("distance_phase_share", "clinch_phase_share", "ground_phase_share"),
    ("head_target_share", "body_target_share", "leg_target_share"),
)

_SNAPSHOT_POSITIVE_RATE_PARAMETERS = frozenset(
    {
        "strike_rate_distance",
        "strike_rate_clinch",
        "strike_rate_ground",
        "clinch_entry_rate",
        "clinch_exit_rate",
        "takedown_attempt_rate",
        "ground_control_rate",
        "escape_rate",
        "submission_attempt_rate",
        "hurt_recovery_per_minute",
    }
)

_COUNT_COLUMNS = (
    "knockdowns",
    "sig_strikes_landed",
    "sig_strikes_attempts",
    "takedowns_landed",
    "takedowns_attempts",
    "sub_attempts",
    "control",
)

_POSITION_COLUMNS = (
    "distance_strikes_attempts",
    "clinch_strikes_attempts",
    "ground_strikes_attempts",
)

_OPTIONAL_OBSERVABLE_COLUMNS = (
    "head_strikes_landed",
    "body_strikes_landed",
    "leg_strikes_landed",
    "reversals",
)

_FIT_FIGHT_COLUMNS = (
    "date",
    "fight_id",
    "event_id",
    "fighter_id",
    "opponent_id",
    "division",
    "result",
    "method",
    "fight_seconds",
    "experience_fights",
    "layoff_days",
    *_COUNT_COLUMNS,
    *_POSITION_COLUMNS,
    *_OPTIONAL_OBSERVABLE_COLUMNS,
    *(f"opponent_{name}" for name in _COUNT_COLUMNS),
    *(f"opponent_{name}" for name in _POSITION_COLUMNS),
    *(f"opponent_{name}" for name in _OPTIONAL_OBSERVABLE_COLUMNS),
)

_FIT_ROUND_COLUMNS = (
    "date",
    "fight_id",
    "event_id",
    "fighter_id",
    "opponent_id",
    "round_number",
    "round_seconds",
    "sig_strikes_attempts",
    "reconciliation_status",
    "_fit_eligible",
)

_TAKEDOWN_CONTROL_ROUND_COLUMNS = (
    "division",
    "takedowns_landed",
    "control",
)


def canonical_json(value: object) -> str:
    """Return the byte-stable JSON representation used for artifact IDs."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _identity_token(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().rstrip("/")
    if text.casefold() in {"", "nan", "none", "<na>"}:
        return ""
    return text.rsplit("/", 1)[-1]


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(_finite(value, low), low), high))


def _utc_iso(value: object) -> str:
    parsed = pd.to_datetime(value, errors="raise", utc=True)
    if isinstance(parsed, pd.DatetimeIndex):
        raise TypeError("as_of must be a scalar timestamp")
    return parsed.isoformat()


def _frame_sha256(frame: pd.DataFrame) -> str:
    """Fingerprint values, column order and dtypes without serializing a CSV."""

    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(
                normalized[column], errors="coerce", utc=True
            ).astype("string")
    header = canonical_json(
        {
            "columns": list(normalized.columns),
            "dtypes": [str(dtype) for dtype in normalized.dtypes],
            "rows": int(len(normalized)),
        }
    ).encode("utf-8")
    values = pd.util.hash_pandas_object(normalized, index=False).to_numpy(
        dtype="uint64"
    )
    digest = sha256(header)
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ParameterFitConfig:
    """Controls pooling and event-card bootstrap fitting."""

    bootstrap_members: int = 200
    random_seed: int = 1729
    era_years: int = 5
    rate_prior_fights: float = 5.0
    probability_prior_attempts: float = 40.0
    rare_event_prior_opportunities: float = 12.0
    division_prior_fights: float = 20.0
    recent_half_life_days: float | None = None

    def __post_init__(self) -> None:
        if self.bootstrap_members <= 0:
            raise ValueError("bootstrap_members must be positive")
        if self.era_years <= 0:
            raise ValueError("era_years must be positive")
        for name in (
            "rate_prior_fights",
            "probability_prior_attempts",
            "rare_event_prior_opportunities",
            "division_prior_fights",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.recent_half_life_days is not None and (
            not math.isfinite(self.recent_half_life_days)
            or self.recent_half_life_days <= 0
        ):
            raise ValueError("recent_half_life_days must be positive when provided")

    @classmethod
    def historical(cls, **overrides: object) -> "ParameterFitConfig":
        values: dict[str, object] = {"bootstrap_members": 64}
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class BootstrapParameterMember:
    member_index: int
    bootstrap_seed: int
    sampled_event_count: int
    context_parameters: dict[str, dict[str, float]]
    fighter_parameters: dict[str, dict[str, float]]
    covariate_effects: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "BootstrapParameterMember":
        return cls(
            member_index=int(value["member_index"]),
            bootstrap_seed=int(value["bootstrap_seed"]),
            sampled_event_count=int(value["sampled_event_count"]),
            context_parameters={
                str(key): {str(k): float(v) for k, v in dict(parameters).items()}
                for key, parameters in dict(value["context_parameters"]).items()
            },
            fighter_parameters={
                str(key): {str(k): float(v) for k, v in dict(parameters).items()}
                for key, parameters in dict(value["fighter_parameters"]).items()
            },
            covariate_effects={
                str(key): float(item)
                for key, item in dict(value["covariate_effects"]).items()
            },
        )


@dataclass(frozen=True)
class ParameterEnsembleArtifact:
    schema_version: int
    model_version: str
    as_of_utc: str
    trained_through: str | None
    input_sha256: str
    config: ParameterFitConfig
    members: tuple[BootstrapParameterMember, ...]
    observed_fights: int
    observed_fighter_sides: int
    observed_round_sides: int
    round_reconciliation_counts: dict[str, int]
    created_at_utc: str
    artifact_sha256: str

    def unhashed_dict(self) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "as_of_utc": self.as_of_utc,
            "trained_through": self.trained_through,
            "input_sha256": self.input_sha256,
            "config": asdict(self.config),
            "members": [member.to_dict() for member in self.members],
            "observed_fights": self.observed_fights,
            "observed_fighter_sides": self.observed_fighter_sides,
            "observed_round_sides": self.observed_round_sides,
            "round_reconciliation_counts": self.round_reconciliation_counts,
            "created_at_utc": self.created_at_utc,
        }
        if self.schema_version not in LEGACY_PARAMETER_SCHEMAS:
            value.pop("created_at_utc")
        return value

    def content_dict(self) -> dict[str, object]:
        """Return deterministic model content, excluding provenance metadata."""

        value = self.unhashed_dict()
        value.pop("created_at_utc", None)
        return value

    def to_dict(self) -> dict[str, object]:
        value = self.unhashed_dict()
        value["created_at_utc"] = self.created_at_utc
        value["artifact_sha256"] = self.artifact_sha256
        return value

    def validate(self) -> "ParameterEnsembleArtifact":
        if self.schema_version not in {
            PARAMETER_SCHEMA_VERSION,
            *LEGACY_PARAMETER_SCHEMAS,
        }:
            raise ValueError("unsupported simulation parameter schema")
        supported_models = {
            PARAMETER_MODEL_VERSION,
            TAKEDOWN_CONTROL_PARAMETER_MODEL_VERSION,
            "fight-sim-empirical-bayes-card-bootstrap-v2",
        }
        if self.model_version not in supported_models:
            raise ValueError("unsupported simulation parameter model")
        if len(self.input_sha256) != 64:
            raise ValueError("parameter input fingerprint is not a SHA-256")
        if len(self.members) != self.config.bootstrap_members:
            raise ValueError("bootstrap member count disagrees with configuration")
        if [member.member_index for member in self.members] != list(
            range(len(self.members))
        ):
            raise ValueError("bootstrap member indices are not contiguous")
        expected_hash = canonical_sha256(self.unhashed_dict())
        if self.artifact_sha256 != expected_hash:
            raise ValueError("simulation parameter artifact hash is invalid")
        if any(
            not isinstance(count, int) or count < 0
            for count in self.round_reconciliation_counts.values()
        ):
            raise ValueError("round reconciliation counts are invalid")
        eligible_rounds = self.round_reconciliation_counts.get("matched", 0)
        eligible_rounds += self.round_reconciliation_counts.get(
            "legacy_unlabeled_eligible", 0
        )
        if self.schema_version in LEGACY_PARAMETER_SCHEMAS and not eligible_rounds:
            eligible_rounds = self.round_reconciliation_counts.get(
                "legacy_unlabeled", 0
            )
        if self.round_reconciliation_counts and eligible_rounds != self.observed_round_sides:
            raise ValueError("eligible round count disagrees with reconciliation summary")
        for member in self.members:
            if "__global__" not in member.context_parameters:
                raise ValueError("bootstrap member is missing global parameters")
            for parameters in (
                list(member.context_parameters.values())
                + list(member.fighter_parameters.values())
            ):
                if set(parameters) != set(PARAMETER_NAMES):
                    raise ValueError("parameter vector has the wrong fields")
                if any(not math.isfinite(float(v)) for v in parameters.values()):
                    raise ValueError("parameter vector contains a non-finite value")
        return self

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ParameterEnsembleArtifact":
        artifact = cls(
            schema_version=int(value["schema_version"]),
            model_version=str(value["model_version"]),
            as_of_utc=str(value["as_of_utc"]),
            trained_through=(
                None
                if value.get("trained_through") is None
                else str(value["trained_through"])
            ),
            input_sha256=str(value["input_sha256"]),
            config=ParameterFitConfig(**dict(value["config"])),
            members=tuple(
                BootstrapParameterMember.from_dict(item)
                for item in list(value["members"])
            ),
            observed_fights=int(value["observed_fights"]),
            observed_fighter_sides=int(value["observed_fighter_sides"]),
            observed_round_sides=int(value["observed_round_sides"]),
            round_reconciliation_counts={
                str(key): int(count)
                for key, count in dict(
                    value.get("round_reconciliation_counts") or {}
                ).items()
            },
            created_at_utc=str(value["created_at_utc"]),
            artifact_sha256=str(value["artifact_sha256"]),
        )
        return artifact.validate()


@dataclass(frozen=True)
class ParameterArtifactInspection:
    """Fast integrity view of a sealed physical parameter artifact.

    This deliberately contains no parameter vectors.  It is sufficient for a
    production pre-commit integrity/cross-hash check; research and simulation
    execution must still call :func:`load_parameter_artifact`.
    """

    storage_format: str
    storage_version: int
    codec: str
    schema_version: int
    model_version: str
    artifact_sha256: str
    input_sha256: str
    members_sha256: str
    bootstrap_members: int
    observed_fights: int
    observed_fighter_sides: int
    observed_round_sides: int
    round_reconciliation_counts: dict[str, int]
    created_at_utc: str
    storage_sha256: str
    fit_inputs_sha256: str | None

    @property
    def members(self) -> range:
        """Expose only cardinality for existing non-materializing gate code."""

        return range(self.bootstrap_members)

    def validate(self) -> "ParameterArtifactInspection":
        if self.storage_format not in {
            PARAMETER_STORAGE_FORMAT,
            "legacy-row-json",
        }:
            raise ValueError("unsupported parameter inspection storage format")
        if self.storage_format == PARAMETER_STORAGE_FORMAT and (
            self.storage_version != PARAMETER_STORAGE_VERSION
        ):
            raise ValueError("unsupported compact parameter storage version")
        if self.schema_version not in {
            PARAMETER_SCHEMA_VERSION,
            *LEGACY_PARAMETER_SCHEMAS,
        }:
            raise ValueError("unsupported simulation parameter schema")
        if self.model_version not in {
            PARAMETER_MODEL_VERSION,
            TAKEDOWN_CONTROL_PARAMETER_MODEL_VERSION,
            "fight-sim-empirical-bayes-card-bootstrap-v2",
        }:
            raise ValueError("unsupported simulation parameter model")
        for name, digest in (
            ("artifact", self.artifact_sha256),
            ("input", self.input_sha256),
            ("members", self.members_sha256),
            ("storage", self.storage_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"parameter {name} commitment is not a SHA-256")
        if self.fit_inputs_sha256 is not None and (
            len(self.fit_inputs_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.fit_inputs_sha256
            )
        ):
            raise ValueError("parameter fit-input commitment is not a SHA-256")
        if self.bootstrap_members <= 0:
            raise ValueError("parameter inspection has no bootstrap members")
        for count in (
            self.observed_fights,
            self.observed_fighter_sides,
            self.observed_round_sides,
        ):
            if count < 0:
                raise ValueError("parameter inspection contains a negative count")
        if any(
            not isinstance(count, int) or count < 0
            for count in self.round_reconciliation_counts.values()
        ):
            raise ValueError("round reconciliation counts are invalid")
        eligible = self.round_reconciliation_counts.get("matched", 0)
        eligible += self.round_reconciliation_counts.get(
            "legacy_unlabeled_eligible", 0
        )
        if self.schema_version in LEGACY_PARAMETER_SCHEMAS and not eligible:
            eligible = self.round_reconciliation_counts.get("legacy_unlabeled", 0)
        if self.round_reconciliation_counts and eligible != self.observed_round_sides:
            raise ValueError(
                "eligible round count disagrees with reconciliation summary"
            )
        return self


def _encode_binary(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_binary(value: object) -> bytes:
    try:
        return base64.b64decode(str(value), validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("compact parameter artifact contains invalid base64") from exc


def _encode_sparse_mapping(
    members: Iterable[BootstrapParameterMember],
    attribute: str,
    value_names: tuple[str, ...],
) -> dict[str, object]:
    members = tuple(members)
    keys = sorted(
        set().union(*(getattr(member, attribute).keys() for member in members))
    )
    key_index = {key: index for index, key in enumerate(keys)}
    offsets = [0]
    indices: list[int] = []
    values: list[float] = []
    for member in members:
        mapping = getattr(member, attribute)
        for key in sorted(mapping):
            indices.append(key_index[key])
            item = mapping[key]
            if value_names == ("value",):
                values.append(float(item))
            else:
                values.extend(float(item[name]) for name in value_names)
        offsets.append(len(indices))
    return {
        "keys": keys,
        "value_names": list(value_names),
        "offsets": _encode_binary(np.asarray(offsets, dtype="<u4").tobytes()),
        "key_indices": _encode_binary(
            np.asarray(indices, dtype="<u4").tobytes()
        ),
        "values_f64": _encode_binary(
            np.asarray(values, dtype="<f8").tobytes()
        ),
    }


def _decode_sparse_mapping(
    value: Mapping[str, object], member_count: int
) -> list[dict[str, object]]:
    keys = [str(item) for item in list(value["keys"])]
    value_names = tuple(str(item) for item in list(value["value_names"]))
    offsets = np.frombuffer(_decode_binary(value["offsets"]), dtype="<u4")
    indices = np.frombuffer(_decode_binary(value["key_indices"]), dtype="<u4")
    numbers = np.frombuffer(_decode_binary(value["values_f64"]), dtype="<f8")
    if len(offsets) != member_count + 1 or offsets[0] != 0:
        raise ValueError("compact parameter mapping has invalid member offsets")
    if int(offsets[-1]) != len(indices):
        raise ValueError("compact parameter mapping has invalid row count")
    if len(numbers) != len(indices) * len(value_names):
        raise ValueError("compact parameter mapping has invalid value count")
    output: list[dict[str, object]] = []
    for member_index in range(member_count):
        start, stop = int(offsets[member_index]), int(offsets[member_index + 1])
        mapping: dict[str, object] = {}
        for row_index in range(start, stop):
            encoded_key = int(indices[row_index])
            if encoded_key >= len(keys):
                raise ValueError("compact parameter mapping key is out of range")
            first = row_index * len(value_names)
            row = numbers[first : first + len(value_names)]
            mapping[keys[encoded_key]] = (
                float(row[0])
                if value_names == ("value",)
                else {
                    name: float(item) for name, item in zip(value_names, row)
                }
            )
        output.append(mapping)
    return output


def _encode_frame(frame: pd.DataFrame) -> dict[str, object]:
    """Encode one fitting frame column-wise without changing scalar bits."""

    encoded_columns: list[dict[str, object]] = []
    for name in frame.columns:
        series = frame[name]
        dtype_text = str(series.dtype)
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            values = pd.to_datetime(series, errors="raise", utc=True).array.asi8
            encoded = {
                "name": str(name),
                "kind": "datetime_ns_utc",
                "dtype": dtype_text,
                "data": _encode_binary(np.asarray(values, dtype="<i8").tobytes()),
            }
        elif pd.api.types.is_bool_dtype(series.dtype):
            encoded = {
                "name": str(name),
                "kind": "bool",
                "dtype": dtype_text,
                "data": _encode_binary(
                    series.to_numpy(dtype=np.uint8).tobytes()
                ),
            }
        elif (
            pd.api.types.is_integer_dtype(series.dtype)
            or pd.api.types.is_float_dtype(series.dtype)
        ) and not pd.api.types.is_extension_array_dtype(series.dtype):
            dtype = np.dtype(series.dtype)
            little = dtype.newbyteorder("<")
            encoded = {
                "name": str(name),
                "kind": "numeric",
                "dtype": dtype.str,
                "data": _encode_binary(
                    series.to_numpy(dtype=little, copy=True).tobytes()
                ),
            }
        else:
            normalized = [
                None if pd.isna(item) else str(item) for item in series.tolist()
            ]
            dictionary = sorted({item for item in normalized if item is not None})
            lookup = {item: index for index, item in enumerate(dictionary)}
            indices = [
                -1 if item is None else lookup[item] for item in normalized
            ]
            encoded = {
                "name": str(name),
                "kind": "dictionary",
                "dtype": dtype_text,
                "dictionary": dictionary,
                "data": _encode_binary(
                    np.asarray(indices, dtype="<i4").tobytes()
                ),
            }
        encoded_columns.append(encoded)
    return {"rows": int(len(frame)), "columns": encoded_columns}


def _decode_frame(value: Mapping[str, object]) -> pd.DataFrame:
    row_count = int(value["rows"])
    output: dict[str, pd.Series] = {}
    for raw_column in list(value["columns"]):
        column = dict(raw_column)
        name = str(column["name"])
        kind = str(column["kind"])
        if kind == "datetime_ns_utc":
            array = np.frombuffer(_decode_binary(column["data"]), dtype="<i8")
            series = pd.Series(pd.to_datetime(array.copy(), utc=True))
        elif kind == "bool":
            array = np.frombuffer(_decode_binary(column["data"]), dtype=np.uint8)
            decoded = array.astype(bool, copy=True)
            series = pd.Series(
                decoded,
                dtype=(
                    "boolean"
                    if str(column.get("dtype", "bool")).startswith("boolean")
                    else bool
                ),
            )
        elif kind == "numeric":
            dtype = np.dtype(str(column["dtype"]))
            array = np.frombuffer(_decode_binary(column["data"]), dtype=dtype)
            series = pd.Series(array.copy())
        elif kind == "dictionary":
            dictionary = [str(item) for item in list(column["dictionary"])]
            indices = np.frombuffer(_decode_binary(column["data"]), dtype="<i4")
            items: list[str | None] = []
            for index in indices:
                integer = int(index)
                if integer == -1:
                    items.append(None)
                elif 0 <= integer < len(dictionary):
                    items.append(dictionary[integer])
                else:
                    raise ValueError("compact frame dictionary index is invalid")
            dtype = str(column.get("dtype", "object"))
            series = pd.Series(items, dtype="string" if dtype.startswith("string") else object)
        else:
            raise ValueError(f"unsupported compact frame column kind: {kind}")
        if len(series) != row_count:
            raise ValueError("compact frame column length disagrees with row count")
        output[name] = series
    return pd.DataFrame(output, columns=list(output))


def _logical_metadata(artifact: ParameterEnsembleArtifact) -> dict[str, object]:
    metadata = artifact.to_dict()
    metadata.pop("members")
    return metadata


def _member_values_sha256(artifact: ParameterEnsembleArtifact) -> str:
    return canonical_sha256([member.to_dict() for member in artifact.members])


def _seal_physical_artifact(
    body: Mapping[str, object],
    artifact: ParameterEnsembleArtifact,
    *,
    fit_inputs: Mapping[str, object] | None,
    member_storage: Mapping[str, object] | None,
) -> dict[str, object]:
    sealed = dict(body)
    sealed["commitments"] = {
        "logical_artifact_sha256": artifact.artifact_sha256,
        "logical_input_sha256": artifact.input_sha256,
        "member_values_sha256": _member_values_sha256(artifact),
        "member_count": len(artifact.members),
        "fit_inputs_sha256": (
            None if fit_inputs is None else canonical_sha256(fit_inputs)
        ),
        "member_storage_sha256": (
            None if member_storage is None else canonical_sha256(member_storage)
        ),
    }
    sealed["storage_sha256"] = canonical_sha256(sealed)
    return sealed


def _encode_columnar_artifact(
    artifact: ParameterEnsembleArtifact,
) -> dict[str, object]:
    member_storage = {
        "member_index": [member.member_index for member in artifact.members],
        "bootstrap_seed": [member.bootstrap_seed for member in artifact.members],
        "sampled_event_count": [
            member.sampled_event_count for member in artifact.members
        ],
        "context_parameters": _encode_sparse_mapping(
            artifact.members, "context_parameters", PARAMETER_NAMES
        ),
        "fighter_parameters": _encode_sparse_mapping(
            artifact.members, "fighter_parameters", PARAMETER_NAMES
        ),
        "covariate_effects": _encode_sparse_mapping(
            artifact.members, "covariate_effects", ("value",)
        ),
    }
    body = {
        "storage_format": PARAMETER_STORAGE_FORMAT,
        "storage_version": PARAMETER_STORAGE_VERSION,
        "codec": "exact-columnar-v1",
        "logical_metadata": _logical_metadata(artifact),
        **member_storage,
    }
    return _seal_physical_artifact(
        body,
        artifact,
        fit_inputs=None,
        member_storage=member_storage,
    )


def _decode_columnar_artifact(
    value: Mapping[str, object]
) -> ParameterEnsembleArtifact:
    indices = [int(item) for item in list(value["member_index"])]
    seeds = [int(item) for item in list(value["bootstrap_seed"])]
    counts = [int(item) for item in list(value["sampled_event_count"])]
    member_count = len(indices)
    if len(seeds) != member_count or len(counts) != member_count:
        raise ValueError("compact parameter member scalar columns disagree")
    contexts = _decode_sparse_mapping(
        dict(value["context_parameters"]), member_count
    )
    fighters = _decode_sparse_mapping(
        dict(value["fighter_parameters"]), member_count
    )
    covariates = _decode_sparse_mapping(
        dict(value["covariate_effects"]), member_count
    )
    members = [
        BootstrapParameterMember(
            member_index=indices[index],
            bootstrap_seed=seeds[index],
            sampled_event_count=counts[index],
            context_parameters=dict(contexts[index]),
            fighter_parameters=dict(fighters[index]),
            covariate_effects={
                str(key): float(item)
                for key, item in covariates[index].items()
            },
        ).to_dict()
        for index in range(member_count)
    ]
    logical = dict(value["logical_metadata"])
    logical["members"] = members
    return ParameterEnsembleArtifact.from_dict(logical)


def _encode_fit_recipe(
    artifact: ParameterEnsembleArtifact,
    inputs: Mapping[str, object],
) -> dict[str, object]:
    encoded_inputs = {
        "fights": _encode_frame(inputs["fights"]),
        "profiles": _encode_frame(inputs["profiles"].reset_index(drop=True)),
        "rounds": _encode_frame(inputs["rounds"]),
        "allow_legacy_unreconciled_rounds": bool(
            inputs["allow_legacy_unreconciled_rounds"]
        ),
        "use_takedown_control_association": bool(
            inputs.get("use_takedown_control_association", False)
        ),
    }
    body = {
        "storage_format": PARAMETER_STORAGE_FORMAT,
        "storage_version": PARAMETER_STORAGE_VERSION,
        "codec": "self-contained-causal-fit-v1",
        "logical_metadata": _logical_metadata(artifact),
        "fit_inputs": encoded_inputs,
    }
    return _seal_physical_artifact(
        body,
        artifact,
        fit_inputs=encoded_inputs,
        member_storage=None,
    )


def _decode_fit_recipe(value: Mapping[str, object]) -> ParameterEnsembleArtifact:
    metadata = dict(value["logical_metadata"])
    if int(metadata["schema_version"]) != PARAMETER_SCHEMA_VERSION:
        raise ValueError("compact fit recipe has an unsupported logical schema")
    inputs = dict(value["fit_inputs"])
    fitter = object.__new__(CausalParameterFitter)
    fitter.raw_fights = _decode_frame(dict(inputs["fights"]))
    profiles = _decode_frame(dict(inputs["profiles"]))
    fitter.fighter_profiles = profiles.set_index("fighter_id", drop=False)
    fitter.round_stats = _decode_frame(dict(inputs["rounds"]))
    fitter.allow_legacy_unreconciled_rounds = bool(
        inputs.get("allow_legacy_unreconciled_rounds", False)
    )
    fitter.use_takedown_control_association = bool(
        inputs.get("use_takedown_control_association", False)
    )
    artifact = fitter.fit(
        metadata["as_of_utc"],
        config=ParameterFitConfig(**dict(metadata["config"])),
        created_at_utc=metadata["created_at_utc"],
    )
    if _logical_metadata(artifact) != metadata:
        raise ValueError(
            "self-contained compact artifact did not reproduce its logical metadata"
        )
    return artifact


def _member_storage_view(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (
            "member_index",
            "bootstrap_seed",
            "sampled_event_count",
            "context_parameters",
            "fighter_parameters",
            "covariate_effects",
        )
    }


def _inspect_compact_value(
    value: Mapping[str, object],
) -> ParameterArtifactInspection:
    if int(value.get("storage_version", -1)) != PARAMETER_STORAGE_VERSION:
        raise ValueError("unsupported compact parameter storage version")
    storage_sha256 = str(value.get("storage_sha256") or "")
    unhashed_storage = dict(value)
    unhashed_storage.pop("storage_sha256", None)
    if storage_sha256 != canonical_sha256(unhashed_storage):
        raise ValueError("compact parameter storage hash is invalid")
    metadata = dict(value["logical_metadata"])
    config = ParameterFitConfig(**dict(metadata["config"]))
    commitments = dict(value["commitments"])
    if commitments.get("logical_artifact_sha256") != metadata.get(
        "artifact_sha256"
    ):
        raise ValueError("compact artifact logical hash commitment is stale")
    if commitments.get("logical_input_sha256") != metadata.get("input_sha256"):
        raise ValueError("compact artifact input hash commitment is stale")
    if int(commitments.get("member_count", -1)) != config.bootstrap_members:
        raise ValueError("compact artifact member-count commitment is stale")
    codec = str(value.get("codec") or "")
    fit_inputs_sha256 = commitments.get("fit_inputs_sha256")
    member_storage_sha256 = commitments.get("member_storage_sha256")
    if codec == "self-contained-causal-fit-v1":
        if fit_inputs_sha256 != canonical_sha256(value["fit_inputs"]):
            raise ValueError("compact artifact fit-input commitment is invalid")
        if member_storage_sha256 is not None:
            raise ValueError("fit-recipe artifact unexpectedly stores member columns")
    elif codec == "exact-columnar-v1":
        if fit_inputs_sha256 is not None:
            raise ValueError("columnar artifact unexpectedly stores fitting inputs")
        if member_storage_sha256 != canonical_sha256(_member_storage_view(value)):
            raise ValueError("compact artifact member-storage commitment is invalid")
        if len(list(value["member_index"])) != config.bootstrap_members:
            raise ValueError("compact member columns disagree with configuration")
    else:
        raise ValueError(f"unsupported compact parameter codec: {codec}")
    inspection = ParameterArtifactInspection(
        storage_format=PARAMETER_STORAGE_FORMAT,
        storage_version=PARAMETER_STORAGE_VERSION,
        codec=codec,
        schema_version=int(metadata["schema_version"]),
        model_version=str(metadata["model_version"]),
        artifact_sha256=str(metadata["artifact_sha256"]),
        input_sha256=str(metadata["input_sha256"]),
        members_sha256=str(commitments.get("member_values_sha256") or ""),
        bootstrap_members=config.bootstrap_members,
        observed_fights=int(metadata["observed_fights"]),
        observed_fighter_sides=int(metadata["observed_fighter_sides"]),
        observed_round_sides=int(metadata["observed_round_sides"]),
        round_reconciliation_counts={
            str(key): int(count)
            for key, count in dict(
                metadata.get("round_reconciliation_counts") or {}
            ).items()
        },
        created_at_utc=str(metadata["created_at_utc"]),
        storage_sha256=storage_sha256,
        fit_inputs_sha256=(
            None if fit_inputs_sha256 is None else str(fit_inputs_sha256)
        ),
    )
    return inspection.validate()


def _read_parameter_physical_value(path: str | Path) -> dict[str, object]:
    payload = Path(path).read_bytes()
    if Path(path).suffix.casefold() == ".gz":
        payload = gzip.decompress(payload)
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("simulation parameter artifact must be a JSON object")
    return value


def inspect_parameter_artifact(path: str | Path) -> ParameterArtifactInspection:
    """Validate sealed commitments without reconstructing bootstrap members."""

    value = _read_parameter_physical_value(path)
    if value.get("storage_format") == PARAMETER_STORAGE_FORMAT:
        return _inspect_compact_value(value)
    # Legacy JSON has no physical commitment wrapper.  It is already a direct
    # logical representation, so validation requires parsing but never refits.
    artifact = ParameterEnsembleArtifact.from_dict(value)
    return ParameterArtifactInspection(
        storage_format="legacy-row-json",
        storage_version=0,
        codec="legacy-row-json",
        schema_version=artifact.schema_version,
        model_version=artifact.model_version,
        artifact_sha256=artifact.artifact_sha256,
        input_sha256=artifact.input_sha256,
        members_sha256=_member_values_sha256(artifact),
        bootstrap_members=len(artifact.members),
        observed_fights=artifact.observed_fights,
        observed_fighter_sides=artifact.observed_fighter_sides,
        observed_round_sides=artifact.observed_round_sides,
        round_reconciliation_counts=dict(artifact.round_reconciliation_counts),
        created_at_utc=artifact.created_at_utc,
        storage_sha256=canonical_sha256(value),
        fit_inputs_sha256=None,
    ).validate()


def save_parameter_artifact(
    path: str | Path,
    artifact: ParameterEnsembleArtifact,
    *,
    materialized: bool = False,
) -> None:
    """Atomically save an exact, versioned artifact in deterministic bytes.

    Normal published artifacts retain the compact self-contained fit recipe so
    their causal inputs remain independently auditable.  Ignored local caches
    can request ``materialized=True`` to store the already-fitted member
    columns instead.  Loading that representation performs integrity checks
    but never repeats the expensive bootstrap fit.
    """

    artifact.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fit_inputs = getattr(artifact, "_compact_fit_inputs", None)
    physical = (
        _encode_columnar_artifact(artifact)
        if materialized
        else (
            _encode_fit_recipe(artifact, fit_inputs)
            if isinstance(fit_inputs, Mapping)
            else _encode_columnar_artifact(artifact)
        )
    )
    payload = (canonical_json(physical) + "\n").encode("utf-8")
    if destination.suffix.casefold() == ".gz":
        # mtime=0 keeps the physical bytes reproducible as well as logical IDs.
        payload = gzip.compress(payload, compresslevel=9, mtime=0)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_parameter_artifact(path: str | Path) -> ParameterEnsembleArtifact:
    value = _read_parameter_physical_value(path)
    if value.get("storage_format") != PARAMETER_STORAGE_FORMAT:
        # Backward compatibility for the original row-oriented logical JSON.
        return ParameterEnsembleArtifact.from_dict(value)
    inspection = _inspect_compact_value(value)
    codec = value.get("codec")
    if codec == "exact-columnar-v1":
        artifact = _decode_columnar_artifact(value)
    elif codec == "self-contained-causal-fit-v1":
        artifact = _decode_fit_recipe(value)
    else:  # pragma: no cover - inspection rejects this first
        raise ValueError(f"unsupported compact parameter codec: {codec}")
    if _member_values_sha256(artifact) != inspection.members_sha256:
        raise ValueError("materialized parameter member commitment is invalid")
    return artifact


def cache_materialized_parameter_artifact(
    artifact: ParameterEnsembleArtifact, cache_dir: str | Path
) -> Path:
    """Persist a fast-loading ignored cache entry keyed by logical identity."""

    artifact.validate()
    destination = Path(cache_dir) / f"parameter-{artifact.artifact_sha256}.json.gz"
    if destination.is_file():
        inspection = inspect_parameter_artifact(destination)
        if (
            inspection.codec != "exact-columnar-v1"
            or inspection.artifact_sha256 != artifact.artifact_sha256
        ):
            raise ValueError(f"materialized parameter cache is invalid: {destination}")
        return destination
    save_parameter_artifact(destination, artifact, materialized=True)
    return destination


def load_parameter_artifact_cached(
    path: str | Path, cache_dir: str | Path
) -> tuple[ParameterEnsembleArtifact, bool, Path]:
    """Load through a content-addressed materialized cache.

    The first access to a recipe artifact still performs its authoritative
    deterministic reconstruction. Later accesses decode exact member columns
    directly and return ``cache_hit=True``.
    """

    source = inspect_parameter_artifact(path)
    cache_path = Path(cache_dir) / f"parameter-{source.artifact_sha256}.json.gz"
    if cache_path.is_file():
        cached = inspect_parameter_artifact(cache_path)
        if (
            cached.codec != "exact-columnar-v1"
            or cached.artifact_sha256 != source.artifact_sha256
            or cached.members_sha256 != source.members_sha256
        ):
            raise ValueError(f"materialized parameter cache is invalid: {cache_path}")
        return load_parameter_artifact(cache_path), True, cache_path
    artifact = load_parameter_artifact(path)
    cache_materialized_parameter_artifact(artifact, cache_dir)
    return artifact, False, cache_path


def _ratio(successes: float, attempts: float, prior: float, strength: float) -> float:
    attempts = max(_finite(attempts), 0.0)
    successes = min(max(_finite(successes), 0.0), attempts)
    return (successes + prior * strength) / (attempts + strength)


def _rate(count: float, minutes: float, prior: float, prior_minutes: float) -> float:
    return (max(_finite(count), 0.0) + prior * prior_minutes) / (
        max(_finite(minutes), 0.0) + prior_minutes
    )


def _weighted_sum(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return 0.0
    return float((values[valid] * frame.loc[valid, "_weight"]).sum())


def _weighted_exposure_minutes(frame: pd.DataFrame, observed_column: str) -> float:
    observed = pd.to_numeric(frame[observed_column], errors="coerce").notna()
    return float(
        (
            pd.to_numeric(frame.loc[observed, "fight_seconds"], errors="coerce")
            * frame.loc[observed, "_weight"]
        ).sum()
        / 60.0
    )


class CausalParameterFitter:
    """Fit card-bootstrap parameter ensembles and construct as-of snapshots."""

    def __init__(
        self,
        raw_fights: pd.DataFrame,
        fighter_profiles: pd.DataFrame | None = None,
        round_stats: pd.DataFrame | None = None,
        *,
        allow_legacy_unreconciled_rounds: bool = False,
        use_takedown_control_association: bool = False,
    ) -> None:
        self.raw_fights = self._normalize_fights(raw_fights)
        self.fighter_profiles = self._normalize_profiles(fighter_profiles)
        self.allow_legacy_unreconciled_rounds = bool(
            allow_legacy_unreconciled_rounds
        )
        self.use_takedown_control_association = bool(
            use_takedown_control_association
        )
        self.round_stats = self._normalize_rounds(
            round_stats,
            allow_legacy_unreconciled_rounds=self.allow_legacy_unreconciled_rounds,
        )

    @property
    def parameter_model_version(self) -> str:
        return (
            TAKEDOWN_CONTROL_PARAMETER_MODEL_VERSION
            if self.use_takedown_control_association
            else PARAMETER_MODEL_VERSION
        )

    @property
    def fit_round_columns(self) -> tuple[str, ...]:
        return (
            (*_FIT_ROUND_COLUMNS, *_TAKEDOWN_CONTROL_ROUND_COLUMNS)
            if self.use_takedown_control_association
            else _FIT_ROUND_COLUMNS
        )

    @staticmethod
    def _normalize_fights(raw: pd.DataFrame) -> pd.DataFrame:
        required = {
            "date",
            "fight_url",
            "event_url",
            "fighter_url",
            "opponent_url",
            "division",
            "result",
            "method",
            "total_fight_time",
            *_COUNT_COLUMNS,
        }
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"raw fights are missing columns: {missing}")
        frame = raw.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True)
        frame["fight_id"] = frame["fight_url"].map(_identity_token)
        frame["event_id"] = frame["event_url"].map(_identity_token)
        frame["fighter_id"] = frame["fighter_url"].map(_identity_token)
        frame["opponent_id"] = frame["opponent_url"].map(_identity_token)
        if frame[["fight_id", "event_id", "fighter_id", "opponent_id"]].eq("").any().any():
            raise ValueError("raw fights contain blank stable IDs")
        frame["fight_seconds"] = pd.to_numeric(
            frame["total_fight_time"], errors="coerce"
        )
        if (frame["fight_seconds"].dropna() < 0).any():
            raise ValueError("raw fights contain negative fight duration")
        # Old UFCStats rows can lack reconstructable duration.  They may still
        # inform landed/attempted probabilities, but contribute no rate or
        # control exposure; absence is never converted to a zero-minute bout.
        frame.loc[frame["fight_seconds"].le(0), "fight_seconds"] = math.nan
        for column in _COUNT_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if (frame[column].dropna() < 0).any():
                raise ValueError(f"raw fights contain negative {column}")
        for column in (*_POSITION_COLUMNS, *_OPTIONAL_OBSERVABLE_COLUMNS):
            if column not in frame:
                frame[column] = math.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if (frame[column].dropna() < 0).any():
                raise ValueError(f"raw fights contain negative {column}")
        counts = frame.groupby("fight_id", sort=False).size()
        invalid = counts[counts != 2]
        if not invalid.empty:
            raise ValueError(
                "every physical fight must have two mirrored sides; invalid IDs: "
                f"{invalid.head().to_dict()}"
            )
        for fight_id, sides in frame.groupby("fight_id", sort=False):
            ids = set(sides["fighter_id"])
            opponents = set(sides["opponent_id"])
            if len(ids) != 2 or ids != opponents:
                raise ValueError(f"mirrored identities disagree for fight {fight_id}")
            if sides["event_id"].nunique() != 1 or sides["date"].nunique() != 1:
                raise ValueError(f"mirrored event metadata disagree for fight {fight_id}")
        opponent_columns = [
            *_COUNT_COLUMNS,
            *_POSITION_COLUMNS,
            *_OPTIONAL_OBSERVABLE_COLUMNS,
        ]
        opponent = frame[["fight_id", "fighter_id", *opponent_columns]].rename(
            columns={
                "fighter_id": "opponent_id_join",
                **{column: f"opponent_{column}" for column in opponent_columns},
            }
        )
        frame = frame.merge(
            opponent,
            left_on=["fight_id", "opponent_id"],
            right_on=["fight_id", "opponent_id_join"],
            how="left",
            validate="one_to_one",
        ).drop(columns="opponent_id_join")
        # Covariates are computed sequentially before any bootstrap weighting.
        frame = frame.sort_values(
            ["date", "event_id", "fight_id", "fighter_id"], kind="stable"
        ).reset_index(drop=True)
        frame["experience_fights"] = frame.groupby("fighter_id").cumcount()
        previous = frame.groupby("fighter_id")["date"].shift()
        frame["layoff_days"] = (frame["date"] - previous).dt.total_seconds() / 86400.0
        return frame

    @staticmethod
    def _normalize_profiles(profiles: pd.DataFrame | None) -> pd.DataFrame:
        if profiles is None:
            return pd.DataFrame(columns=["fighter_id", "name", "dob"])
        required = {"url", "name", "dob"}
        missing = sorted(required - set(profiles.columns))
        if missing:
            raise ValueError(f"fighter profiles are missing columns: {missing}")
        frame = profiles.copy()
        frame["fighter_id"] = frame["url"].map(_identity_token)
        frame["dob"] = pd.to_datetime(frame["dob"], errors="coerce", utc=True)
        if frame["fighter_id"].duplicated().any():
            raise ValueError("fighter profiles contain duplicate stable IDs")
        return frame.set_index("fighter_id", drop=False)

    @staticmethod
    def _normalize_rounds(
        rounds: pd.DataFrame | None,
        *,
        allow_legacy_unreconciled_rounds: bool = False,
    ) -> pd.DataFrame:
        if rounds is None or rounds.empty:
            return pd.DataFrame()
        frame = rounds.copy()
        aliases = {
            "fight_url": "fight_id",
            "event_url": "event_id",
            "fighter_url": "fighter_id",
            "opponent_url": "opponent_id",
            "round": "round_number",
            "round_time_seconds": "round_seconds",
            "exposure_seconds": "round_seconds",
        }
        for source, target in aliases.items():
            if target not in frame and source in frame:
                frame[target] = frame[source]
        for column in ("fight_id", "event_id", "fighter_id", "opponent_id"):
            if column not in frame:
                raise ValueError(f"round statistics are missing {column}")
            frame[column] = frame[column].map(_identity_token)
        required = {"date", "round_number", "sig_strikes_attempts"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"round statistics are missing columns: {missing}")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True)
        frame["round_number"] = pd.to_numeric(
            frame["round_number"], errors="raise"
        ).astype(int)
        if "round_seconds" not in frame:
            frame["round_seconds"] = 300.0
        frame["round_seconds"] = pd.to_numeric(
            frame["round_seconds"], errors="coerce"
        )
        frame["sig_strikes_attempts"] = pd.to_numeric(
            frame["sig_strikes_attempts"], errors="coerce"
        )
        invalid = (
            (frame["round_number"] < 1)
            | (frame["round_number"] > 5)
            | frame["round_seconds"].isna()
            | (frame["round_seconds"] <= 0)
            | (frame["round_seconds"] > 300)
        )
        if invalid.any():
            raise ValueError("round statistics contain invalid round exposure")
        key = ["fight_id", "fighter_id", "round_number"]
        if frame.duplicated(key).any():
            raise ValueError("round statistics contain duplicate fighter-round rows")
        if "reconciliation_status" in frame:
            frame["reconciliation_status"] = (
                frame["reconciliation_status"].astype("string").str.strip().str.casefold()
            )
            allowed = {"matched", "unverifiable", "discrepancy"}
            unexpected = set(frame["reconciliation_status"].dropna()) - allowed
            if unexpected:
                raise ValueError(
                    f"round statistics contain unknown reconciliation status: {sorted(unexpected)}"
                )
            frame["_fit_eligible"] = frame["reconciliation_status"].eq("matched")
        else:
            suffix = "eligible" if allow_legacy_unreconciled_rounds else "excluded"
            frame["reconciliation_status"] = f"legacy_unlabeled_{suffix}"
            frame["_fit_eligible"] = bool(allow_legacy_unreconciled_rounds)
        return frame.sort_values(
            ["date", "event_id", "fight_id", "fighter_id", "round_number"],
            kind="stable",
        ).reset_index(drop=True)

    def _before(self, as_of: object) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
        cutoff = pd.to_datetime(as_of, errors="raise", utc=True)
        if isinstance(cutoff, pd.DatetimeIndex):
            raise TypeError("as_of must be a scalar timestamp")
        fights = self.raw_fights.loc[self.raw_fights["date"] < cutoff].copy()
        rounds = (
            self.round_stats.loc[
                self.round_stats["date"].lt(cutoff)
                & self.round_stats["_fit_eligible"]
            ].copy()
            if not self.round_stats.empty
            else self.round_stats.copy()
        )
        if fights.empty:
            raise ValueError("no completed fights exist strictly before as_of")
        if not rounds.empty:
            identity = ["fight_id", "event_id", "fighter_id", "opponent_id"]
            causal_identities = fights[identity].drop_duplicates()
            checked = rounds.merge(
                causal_identities.assign(_causal_identity=True),
                on=identity,
                how="left",
                validate="many_to_one",
            )
            invalid = checked["_causal_identity"].ne(True)
            if invalid.any():
                example = checked.loc[invalid, identity].iloc[0].to_dict()
                raise ValueError(
                    "eligible round identity is absent from causal doubled bouts: "
                    f"{example}"
                )
        return fights, rounds, cutoff

    @staticmethod
    def _era_key(date: pd.Timestamp, years: int) -> str:
        start = (int(date.year) // years) * years
        return f"{start}-{start + years - 1}"

    @staticmethod
    def _method_bucket(value: object) -> str:
        return method_bucket(value)

    def _prepare_member_rows(
        self,
        fights: pd.DataFrame,
        event_weights: Mapping[str, int],
        cutoff: pd.Timestamp,
        config: ParameterFitConfig,
    ) -> pd.DataFrame:
        frame = fights.copy()
        frame["_weight"] = frame["event_id"].map(event_weights).fillna(0).astype(float)
        if config.recent_half_life_days is not None:
            age_days = (cutoff - frame["date"]).dt.total_seconds() / 86400.0
            frame["_weight"] *= np.power(0.5, age_days / config.recent_half_life_days)
        frame = frame.loc[frame["_weight"] > 0].copy()
        frame["era"] = frame["date"].map(
            lambda value: self._era_key(value, config.era_years)
        )
        frame["is_sub_win"] = (
            frame["result"].astype(str).str.upper().eq("W")
            & frame["method"].map(self._method_bucket).eq("submission")
        ).astype(float)
        frame["is_ko_win"] = (
            frame["result"].astype(str).str.upper().eq("W")
            & frame["method"].map(self._method_bucket).eq("ko_tko")
        ).astype(float)
        frame["is_ko_loss"] = (
            frame["result"].astype(str).str.upper().eq("L")
            & frame["method"].map(self._method_bucket).eq("ko_tko")
        ).astype(float)
        frame["is_sub_loss"] = (
            frame["result"].astype(str).str.upper().eq("L")
            & frame["method"].map(self._method_bucket).eq("submission")
        ).astype(float)
        return frame

    @staticmethod
    def _unpooled_sufficient(frame: pd.DataFrame) -> dict[str, float]:
        minutes = _weighted_exposure_minutes(frame, "sig_strikes_attempts")
        td_minutes = _weighted_exposure_minutes(frame, "takedowns_attempts")
        sub_minutes = _weighted_exposure_minutes(frame, "sub_attempts")
        control_seconds = _weighted_sum(frame, "control")
        control_exposure = float(
            (
                frame.loc[pd.to_numeric(frame["control"], errors="coerce").notna(), "fight_seconds"]
                * frame.loc[pd.to_numeric(frame["control"], errors="coerce").notna(), "_weight"]
            ).sum()
        )
        opponent_control_seconds = _weighted_sum(frame, "opponent_control")
        opponent_control_exposure = float(
            (
                frame.loc[
                    pd.to_numeric(frame["opponent_control"], errors="coerce").notna(),
                    "fight_seconds",
                ]
                * frame.loc[
                    pd.to_numeric(frame["opponent_control"], errors="coerce").notna(),
                    "_weight",
                ]
            ).sum()
        )
        return {
            "minutes": minutes,
            "sig_attempts": _weighted_sum(frame, "sig_strikes_attempts"),
            "distance_minutes": _weighted_exposure_minutes(frame, "distance_strikes_attempts"),
            "clinch_minutes": _weighted_exposure_minutes(frame, "clinch_strikes_attempts"),
            "ground_minutes": _weighted_exposure_minutes(frame, "ground_strikes_attempts"),
            "distance_attempts": _weighted_sum(frame, "distance_strikes_attempts"),
            "clinch_attempts": _weighted_sum(frame, "clinch_strikes_attempts"),
            "ground_attempts": _weighted_sum(frame, "ground_strikes_attempts"),
            "sig_landed": _weighted_sum(frame, "sig_strikes_landed"),
            "head_landed": _weighted_sum(frame, "head_strikes_landed"),
            "body_landed": _weighted_sum(frame, "body_strikes_landed"),
            "leg_landed": _weighted_sum(frame, "leg_strikes_landed"),
            "opp_sig_attempts": _weighted_sum(frame, "opponent_sig_strikes_attempts"),
            "opp_sig_landed": _weighted_sum(frame, "opponent_sig_strikes_landed"),
            "knockdowns": _weighted_sum(frame, "knockdowns"),
            "opp_knockdowns": _weighted_sum(frame, "opponent_knockdowns"),
            "td_minutes": td_minutes,
            "td_attempts": _weighted_sum(frame, "takedowns_attempts"),
            "td_landed": _weighted_sum(frame, "takedowns_landed"),
            "opp_td_attempts": _weighted_sum(frame, "opponent_takedowns_attempts"),
            "opp_td_landed": _weighted_sum(frame, "opponent_takedowns_landed"),
            "reversals": _weighted_sum(frame, "reversals"),
            "sub_minutes": sub_minutes,
            "sub_attempts": _weighted_sum(frame, "sub_attempts"),
            "opp_sub_attempts": _weighted_sum(frame, "opponent_sub_attempts"),
            "sub_wins": _weighted_sum(frame, "is_sub_win"),
            "ko_wins": _weighted_sum(frame, "is_ko_win"),
            "ko_losses": _weighted_sum(frame, "is_ko_loss"),
            "sub_losses": _weighted_sum(frame, "is_sub_loss"),
            "control_seconds": control_seconds,
            "control_exposure": control_exposure,
            "opp_control_seconds": opponent_control_seconds,
            "opp_control_exposure": opponent_control_exposure,
        }

    @staticmethod
    def _sufficient_rows(frame: pd.DataFrame) -> pd.DataFrame:
        """Vectorize all additive sufficient statistics for fast bootstraps."""

        output = frame[["fighter_id", "division", "era"]].copy()
        weight = frame["_weight"].to_numpy(dtype=float)
        seconds = pd.to_numeric(frame["fight_seconds"], errors="coerce").to_numpy(float)

        def weighted(column: str) -> np.ndarray:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            return np.nan_to_num(values, nan=0.0) * weight

        def exposure(column: str, *, minutes: bool) -> np.ndarray:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            observed_seconds = np.where(np.isfinite(values), seconds, 0.0)
            observed_seconds = np.nan_to_num(observed_seconds, nan=0.0) * weight
            return observed_seconds / 60.0 if minutes else observed_seconds

        columns = {
            "minutes": exposure("sig_strikes_attempts", minutes=True),
            "sig_attempts": weighted("sig_strikes_attempts"),
            "distance_minutes": exposure("distance_strikes_attempts", minutes=True),
            "clinch_minutes": exposure("clinch_strikes_attempts", minutes=True),
            "ground_minutes": exposure("ground_strikes_attempts", minutes=True),
            "distance_attempts": weighted("distance_strikes_attempts"),
            "clinch_attempts": weighted("clinch_strikes_attempts"),
            "ground_attempts": weighted("ground_strikes_attempts"),
            "sig_landed": weighted("sig_strikes_landed"),
            "head_landed": weighted("head_strikes_landed"),
            "body_landed": weighted("body_strikes_landed"),
            "leg_landed": weighted("leg_strikes_landed"),
            "opp_sig_attempts": weighted("opponent_sig_strikes_attempts"),
            "opp_sig_landed": weighted("opponent_sig_strikes_landed"),
            "knockdowns": weighted("knockdowns"),
            "opp_knockdowns": weighted("opponent_knockdowns"),
            "td_minutes": exposure("takedowns_attempts", minutes=True),
            "td_attempts": weighted("takedowns_attempts"),
            "td_landed": weighted("takedowns_landed"),
            "opp_td_attempts": weighted("opponent_takedowns_attempts"),
            "opp_td_landed": weighted("opponent_takedowns_landed"),
            "reversals": weighted("reversals"),
            "sub_minutes": exposure("sub_attempts", minutes=True),
            "sub_attempts": weighted("sub_attempts"),
            "opp_sub_attempts": weighted("opponent_sub_attempts"),
            "sub_wins": weighted("is_sub_win"),
            "ko_wins": weighted("is_ko_win"),
            "ko_losses": weighted("is_ko_loss"),
            "sub_losses": weighted("is_sub_loss"),
            "control_seconds": weighted("control"),
            "control_exposure": exposure("control", minutes=False),
            "opp_control_seconds": weighted("opponent_control"),
            "opp_control_exposure": exposure("opponent_control", minutes=False),
        }
        for name, values in columns.items():
            output[name] = values
        return output

    @staticmethod
    def _parameters_from_sufficient(
        stats: Mapping[str, float],
        prior: Mapping[str, float] | None,
        config: ParameterFitConfig,
        *,
        prior_scale: float = 1.0,
        pace_decay: float | None = None,
        rate_multiplier: float = 1.0,
        update_weak_mechanics: bool = False,
    ) -> dict[str, float]:
        population_fit = prior is None
        if prior is None:
            # Conservative UFC-wide seeds used only while estimating the first
            # data-driven global vector.
            prior = {
                "strike_rate_distance": 7.0,
                "strike_rate_clinch": 4.0,
                "strike_rate_ground": 5.0,
                "distance_phase_share": 0.83,
                "clinch_phase_share": 0.085,
                "ground_phase_share": 0.085,
                "strike_accuracy": 0.45,
                "strike_defense": 0.55,
                "head_target_share": 0.62,
                "body_target_share": 0.24,
                "leg_target_share": 0.14,
                "strike_power": 0.50,
                "knockdown_rate_per_landed": 0.025,
                "finish_after_knockdown": 0.18,
                "clinch_entry_rate": 0.35,
                "clinch_exit_rate": 0.70,
                "takedown_attempt_rate": 0.65,
                "takedown_accuracy": 0.40,
                "takedown_defense": 0.60,
                "ground_control_rate": 0.55,
                "escape_rate": 0.45,
                "reversal_after_escape": 0.12,
                "submission_attempt_rate": 0.28,
                "submission_finish_probability": 0.12,
                "submission_defense": 0.58,
                "ko_resistance": 0.55,
                "hurt_recovery_per_minute": 0.35,
                "pace_decay": 0.12,
                "stamina_recovery_between_rounds": 0.15,
            }
        rate_minutes = config.rate_prior_fights * 15.0 * prior_scale
        probability_strength = config.probability_prior_attempts * prior_scale
        rare_strength = config.rare_event_prior_opportunities * prior_scale
        sig_attempts = stats["sig_attempts"]
        sig_landed = stats["sig_landed"]
        opp_attempts = stats["opp_sig_attempts"]
        opp_landed = stats["opp_sig_landed"]
        # Strike-position counts are observed composition, not phase-time
        # occupancy.  They may inform strike policies but must never be used as
        # a denominator for takedown or submission opportunity rates.
        phase_counts = {
            "distance": stats["distance_attempts"],
            "clinch": stats["clinch_attempts"],
            "ground": stats["ground_attempts"],
        }
        observed_phase_attempts = math.fsum(phase_counts.values())
        phase_shares = {
            phase: (
                count + probability_strength * prior[f"{phase}_phase_share"]
            )
            / (observed_phase_attempts + probability_strength)
            for phase, count in phase_counts.items()
        }
        phase_share_total = math.fsum(phase_shares.values())
        phase_shares = {
            phase: value / phase_share_total
            for phase, value in phase_shares.items()
        }
        prior_total_strike_rate = math.fsum(
            prior[f"strike_rate_{phase}"] * prior[f"{phase}_phase_share"]
            for phase in ("distance", "clinch", "ground")
        )
        total_strike_rate = _rate(
            stats["sig_attempts"],
            stats["minutes"],
            prior_total_strike_rate,
            rate_minutes,
        )
        # UFCStats observes the strike partition, not phase exposure time.  The
        # pooled parent phase mix is therefore the explicit occupancy proxy
        # used to convert marginal contribution rates into conditional engine
        # hazards.  At the population fit, each phase has the same conditional
        # engagement rate; fighter/context deviations come only from supported
        # observed composition.
        conditional_strike_rates = {
            phase: total_strike_rate
            * phase_shares[phase]
            / (
                phase_shares[phase]
                if population_fit
                else max(prior[f"{phase}_phase_share"], 1e-6)
            )
            for phase in ("distance", "clinch", "ground")
        }
        takedown_marginal = _rate(
            stats["td_attempts"],
            stats["td_minutes"],
            prior["takedown_attempt_rate"],
            rate_minutes,
        )
        submission_marginal = _rate(
            stats["sub_attempts"],
            stats["sub_minutes"],
            prior["submission_attempt_rate"],
            rate_minutes,
        )
        parameters = {
            "strike_rate_distance": conditional_strike_rates["distance"]
            * rate_multiplier,
            "strike_rate_clinch": conditional_strike_rates["clinch"]
            * rate_multiplier,
            "strike_rate_ground": conditional_strike_rates["ground"]
            * rate_multiplier,
            "distance_phase_share": phase_shares["distance"],
            "clinch_phase_share": phase_shares["clinch"],
            "ground_phase_share": phase_shares["ground"],
            "strike_accuracy": _ratio(
                sig_landed,
                sig_attempts,
                prior["strike_accuracy"],
                probability_strength,
            ),
            "strike_defense": 1.0
            - _ratio(
                opp_landed,
                opp_attempts,
                1.0 - prior["strike_defense"],
                probability_strength,
            ),
            # UFCStats observes target composition but not action-level target
            # choice.  Treat the landed target partition as one strongly
            # pooled multinomial policy and keep the three shares coherent.
            "head_target_share": (
                stats["head_landed"]
                + probability_strength * prior["head_target_share"]
            )
            / (
                stats["head_landed"]
                + stats["body_landed"]
                + stats["leg_landed"]
                + probability_strength
            ),
            "body_target_share": (
                stats["body_landed"]
                + probability_strength * prior["body_target_share"]
            )
            / (
                stats["head_landed"]
                + stats["body_landed"]
                + stats["leg_landed"]
                + probability_strength
            ),
            "leg_target_share": (
                stats["leg_landed"]
                + probability_strength * prior["leg_target_share"]
            )
            / (
                stats["head_landed"]
                + stats["body_landed"]
                + stats["leg_landed"]
                + probability_strength
            ),
            # Damage conversion and state transitions are not identifiable
            # from bout totals. They stay global/division values in v1.
            "strike_power": prior["strike_power"],
            "knockdown_rate_per_landed": _ratio(
                stats["knockdowns"],
                sig_landed,
                prior["knockdown_rate_per_landed"],
                rare_strength,
            ),
            "finish_after_knockdown": (
                _ratio(
                    stats["ko_wins"],
                    max(stats["knockdowns"], stats["ko_wins"]),
                    prior["finish_after_knockdown"],
                    rare_strength,
                )
                if update_weak_mechanics
                else prior["finish_after_knockdown"]
            ),
            "clinch_entry_rate": prior["clinch_entry_rate"],
            "clinch_exit_rate": prior["clinch_exit_rate"],
            # UFCStats exposes whole-fight TD/submission opportunities but no
            # reliable eligible-phase clock.  Keep their strongly pooled
            # whole-fight marginal rates as conservative engine proxies; do
            # not inflate them by a strike-composition pseudo-occupancy.
            "takedown_attempt_rate": takedown_marginal,
            "takedown_accuracy": _ratio(
                stats["td_landed"],
                stats["td_attempts"],
                prior["takedown_accuracy"],
                probability_strength,
            ),
            "takedown_defense": 1.0
            - _ratio(
                stats["opp_td_landed"],
                stats["opp_td_attempts"],
                1.0 - prior["takedown_defense"],
                probability_strength,
            ),
            "ground_control_rate": _ratio(
                stats["control_seconds"],
                stats["control_exposure"],
                prior["ground_control_rate"],
                probability_strength * 60.0,
            ),
            "escape_rate": 1.0
            - _ratio(
                stats["opp_control_seconds"],
                stats["opp_control_exposure"],
                1.0 - prior["escape_rate"],
                probability_strength * 60.0,
            ),
            "reversal_after_escape": _ratio(
                stats["reversals"],
                max(stats["opp_td_landed"], stats["reversals"]),
                prior["reversal_after_escape"],
                rare_strength,
            ),
            "submission_attempt_rate": submission_marginal,
            "submission_finish_probability": _ratio(
                stats["sub_wins"],
                max(stats["sub_attempts"], stats["sub_wins"]),
                prior["submission_finish_probability"],
                rare_strength,
            ),
            "submission_defense": 1.0
            - _ratio(
                stats["sub_losses"],
                max(stats["opp_sub_attempts"], stats["sub_losses"]),
                1.0 - prior["submission_defense"],
                rare_strength,
            ),
            "ko_resistance": 1.0
            - _ratio(
                stats["ko_losses"],
                max(stats["opp_knockdowns"], stats["ko_losses"]),
                1.0 - prior["ko_resistance"],
                rare_strength,
            ),
            "hurt_recovery_per_minute": prior["hurt_recovery_per_minute"],
            "pace_decay": prior["pace_decay"] if pace_decay is None else pace_decay,
            "stamina_recovery_between_rounds": prior[
                "stamina_recovery_between_rounds"
            ],
        }
        rate_bounds = {
            "strike_rate_distance": (0.05, 30.0),
            "strike_rate_clinch": (0.01, 30.0),
            "strike_rate_ground": (0.01, 30.0),
            "clinch_entry_rate": (0.0, 5.0),
            "clinch_exit_rate": (0.0, 5.0),
            "takedown_attempt_rate": (0.001, 5.0),
            "ground_control_rate": (0.0, 5.0),
            "escape_rate": (0.0, 5.0),
            "submission_attempt_rate": (0.0001, 3.0),
            "hurt_recovery_per_minute": (0.0, 5.0),
        }
        for name in PARAMETER_NAMES:
            low, high = rate_bounds.get(name, (0.00001, 0.99999))
            parameters[name] = _clip(parameters[name], low, high)
        target_total = math.fsum(
            parameters[name]
            for name in ("head_target_share", "body_target_share", "leg_target_share")
        )
        if target_total <= 0:  # pragma: no cover - positive priors guarantee this
            raise ValueError("target composition has no probability mass")
        for name in ("head_target_share", "body_target_share", "leg_target_share"):
            parameters[name] /= target_total
        fitted_phase_total = math.fsum(
            parameters[name]
            for name in (
                "distance_phase_share",
                "clinch_phase_share",
                "ground_phase_share",
            )
        )
        for name in (
            "distance_phase_share",
            "clinch_phase_share",
            "ground_phase_share",
        ):
            parameters[name] /= fitted_phase_total
        return parameters

    @staticmethod
    def _fit_covariate_effects(frame: pd.DataFrame) -> dict[str, float]:
        """Fit a regularized log-rate adjustment for observable covariates."""

        attempts = pd.to_numeric(frame["sig_strikes_attempts"], errors="coerce")
        minutes = pd.to_numeric(frame["fight_seconds"], errors="coerce") / 60.0
        age = pd.to_numeric(frame.get("age_years"), errors="coerce")
        experience = pd.to_numeric(frame["experience_fights"], errors="coerce")
        layoff = pd.to_numeric(frame["layoff_days"], errors="coerce")
        valid = attempts.notna() & minutes.gt(0)
        rare = CausalParameterFitter._fit_global_rare_rates(frame)
        if valid.sum() < 20:
            return {
                "age_per_decade": 0.0,
                "log_experience": 0.0,
                "log_layoff_years": 0.0,
                "age_center": 0.0,
                "log_experience_center": 0.0,
                "log_layoff_years_center": 0.0,
                **rare,
            }
        x = np.column_stack(
            [
                np.nan_to_num((age[valid].to_numpy(dtype=float) - 30.0) / 10.0),
                np.log1p(np.maximum(experience[valid].to_numpy(dtype=float), 0.0)),
                np.log1p(
                    np.maximum(
                        np.nan_to_num(layoff[valid].to_numpy(dtype=float), nan=365.0)
                        / 365.25,
                        0.0,
                    )
                ),
            ]
        )
        centers = x.mean(axis=0)
        x -= centers
        y = np.log((attempts[valid].to_numpy(dtype=float) + 0.5) / (minutes[valid].to_numpy(dtype=float) + 0.5))
        y -= y.mean()
        weights = frame.loc[valid, "_weight"].to_numpy(dtype=float)
        root_w = np.sqrt(np.maximum(weights, 0.0))
        xw = x * root_w[:, None]
        yw = y * root_w
        ridge = np.eye(x.shape[1]) * 20.0
        coefficients = np.linalg.solve(xw.T @ xw + ridge, xw.T @ yw)
        coefficients = np.clip(coefficients, -0.35, 0.35)
        return {
            "age_per_decade": float(coefficients[0]),
            "log_experience": float(coefficients[1]),
            "log_layoff_years": float(coefficients[2]),
            "age_center": float(centers[0]),
            "log_experience_center": float(centers[1]),
            "log_layoff_years_center": float(centers[2]),
            **rare,
        }

    @staticmethod
    def _fit_global_rare_rates(frame: pd.DataFrame) -> dict[str, float]:
        """Estimate pooled exogenous outcomes once per physical fight.

        A weak fixed exposure prior stabilizes sparse bootstrap resamples.  The
        rates remain global and are never attributed to either fighter.
        """

        if frame.empty:
            return {
                "global_no_contest_rate_per_minute": 0.0,
                "global_other_finish_rate_per_minute": 0.0,
            }
        minutes = 0.0
        no_contests = 0.0
        other_finishes = 0.0
        for _fight_id, sides in frame.groupby("fight_id", sort=False):
            weight = float(sides["_weight"].iloc[0])
            duration = pd.to_numeric(sides["fight_seconds"], errors="coerce").dropna()
            if duration.empty or float(duration.iloc[0]) <= 0 or weight <= 0:
                continue
            minutes += float(duration.iloc[0]) / 60.0 * weight
            results = sides["result"].astype(str).str.upper().str.strip()
            methods = sides["method"].map(method_bucket)
            if results.isin({"NC", "N/C", "NO CONTEST"}).any():
                no_contests += weight
            elif (results.eq("W") & methods.eq("other")).any():
                other_finishes += weight
        # Roughly 10,000 active minutes of weak regularization prevents a
        # bootstrap member with zero rare events from assigning zero support.
        prior_minutes = 10_000.0
        no_contest_prior = 0.0008
        other_prior = 0.0004
        return {
            "global_no_contest_rate_per_minute": float(
                (no_contests + no_contest_prior * prior_minutes)
                / (minutes + prior_minutes)
            ),
            "global_other_finish_rate_per_minute": float(
                (other_finishes + other_prior * prior_minutes)
                / (minutes + prior_minutes)
            ),
        }

    def _attach_age(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if self.fighter_profiles.empty:
            result["age_years"] = math.nan
            return result
        dobs = result["fighter_id"].map(self.fighter_profiles["dob"])
        result["age_years"] = (
            (result["date"] - dobs).dt.total_seconds() / (365.25 * 86400.0)
        )
        return result

    def _pace_decay_by_fighter(
        self,
        rounds: pd.DataFrame,
        event_weights: Mapping[str, int],
        global_default: float,
    ) -> dict[str, float]:
        if rounds.empty:
            return {}
        frame = rounds.copy()
        frame["_weight"] = frame["event_id"].map(event_weights).fillna(0).astype(float)
        frame = frame.loc[
            (frame["_weight"] > 0)
            & frame["sig_strikes_attempts"].notna()
            & frame["round_seconds"].gt(0)
        ].copy()
        if frame.empty:
            return {}
        frame["pace"] = frame["sig_strikes_attempts"] / (frame["round_seconds"] / 60.0)
        output: dict[str, float] = {}
        for fighter_id, rows in frame.groupby("fighter_id", sort=False):
            first = rows.loc[rows["round_number"].eq(1)]
            later = rows.loc[rows["round_number"].gt(1)]
            if first.empty or later.empty:
                continue
            first_rate = np.average(first["pace"], weights=first["_weight"])
            # Convert average later/first pace into a per-round multiplier.
            ratios: list[float] = []
            ratio_weights: list[float] = []
            for _, row in later.iterrows():
                if first_rate <= 0 or float(row["pace"]) < 0:
                    continue
                ratios.append(
                    (float(row["pace"]) / first_rate)
                    ** (1.0 / (int(row["round_number"]) - 1))
                )
                ratio_weights.append(float(row["_weight"]))
            if ratios:
                observed_multiplier = float(np.average(ratios, weights=ratio_weights))
                observed = 1.0 - observed_multiplier
                exposure = sum(ratio_weights)
                output[str(fighter_id)] = _clip(
                    (observed * exposure + global_default * 10.0) / (exposure + 10.0),
                    0.0,
                    1.0,
                )
        return output

    def _takedown_control_sufficient(
        self,
        rounds: pd.DataFrame,
        event_weights: Mapping[str, int],
        config: ParameterFitConfig,
    ) -> pd.DataFrame:
        """Return additive, interval-censored TD-round control associations.

        UFCStats does not expose exact action order or top/bottom position.
        Each qualifying fighter-round is therefore one same-round opportunity,
        and its response is credited CTRL divided by observed round exposure.
        The mirrored opponent row supplies control conceded after an opponent
        takedown. These are candidate predictors, not claimed causal counts.
        """

        columns = (
            "fighter_id",
            "division",
            "era",
            "td_control_share_sum",
            "td_control_opportunities",
            "opp_td_control_share_sum",
            "opp_td_control_opportunities",
        )
        if rounds.empty:
            return pd.DataFrame(columns=columns)
        required = {
            "fight_id",
            "event_id",
            "fighter_id",
            "opponent_id",
            "date",
            "round_number",
            "round_seconds",
            "division",
            "takedowns_landed",
            "control",
        }
        if required - set(rounds.columns):
            return pd.DataFrame(columns=columns)
        frame = rounds.copy()
        frame["_weight"] = frame["event_id"].map(event_weights).fillna(0).astype(float)
        for name in ("round_seconds", "takedowns_landed", "control"):
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        frame = frame.loc[frame["_weight"].gt(0)].copy()
        if frame.empty:
            return pd.DataFrame(columns=columns)
        opponent = frame[
            [
                "fight_id",
                "round_number",
                "fighter_id",
                "round_seconds",
                "takedowns_landed",
                "control",
            ]
        ].rename(
            columns={
                "fighter_id": "opponent_id_join",
                "round_seconds": "opponent_round_seconds",
                "takedowns_landed": "opponent_takedowns_landed",
                "control": "opponent_control",
            }
        )
        frame = frame.merge(
            opponent,
            left_on=["fight_id", "round_number", "opponent_id"],
            right_on=["fight_id", "round_number", "opponent_id_join"],
            how="left",
            validate="one_to_one",
        )
        frame["era"] = frame["date"].map(
            lambda value: self._era_key(pd.Timestamp(value), config.era_years)
        )
        own_valid = (
            frame["takedowns_landed"].gt(0)
            & frame["control"].notna()
            & frame["round_seconds"].gt(0)
        )
        opponent_valid = (
            frame["opponent_takedowns_landed"].gt(0)
            & frame["opponent_control"].notna()
            & frame["opponent_round_seconds"].gt(0)
        )
        frame["td_control_opportunities"] = np.where(
            own_valid, frame["_weight"], 0.0
        )
        frame["td_control_share_sum"] = np.where(
            own_valid,
            np.clip(frame["control"] / frame["round_seconds"], 0.0, 1.0)
            * frame["_weight"],
            0.0,
        )
        frame["opp_td_control_opportunities"] = np.where(
            opponent_valid, frame["_weight"], 0.0
        )
        frame["opp_td_control_share_sum"] = np.where(
            opponent_valid,
            np.clip(
                frame["opponent_control"] / frame["opponent_round_seconds"],
                0.0,
                1.0,
            )
            * frame["_weight"],
            0.0,
        )
        return frame.loc[:, columns]

    @staticmethod
    def _apply_takedown_control_association(
        parameters: Mapping[str, float],
        stats: Mapping[str, float],
        prior: Mapping[str, float],
        prior_opportunities: float,
    ) -> dict[str, float]:
        result = dict(parameters)
        own_opportunities = max(float(stats.get("td_control_opportunities", 0.0)), 0.0)
        own_sum = float(stats.get("td_control_share_sum", 0.0))
        conceded_opportunities = max(
            float(stats.get("opp_td_control_opportunities", 0.0)), 0.0
        )
        conceded_sum = float(stats.get("opp_td_control_share_sum", 0.0))
        result["ground_control_rate"] = _clip(
            (own_sum + prior_opportunities * prior["ground_control_rate"])
            / (own_opportunities + prior_opportunities),
            0.0,
            1.0,
        )
        conceded_prior = 1.0 - prior["escape_rate"]
        conceded_share = (
            conceded_sum + prior_opportunities * conceded_prior
        ) / (conceded_opportunities + prior_opportunities)
        result["escape_rate"] = _clip(1.0 - conceded_share, 0.0, 1.0)
        return result

    def _fit_member(
        self,
        member_index: int,
        seed: int,
        fights: pd.DataFrame,
        rounds: pd.DataFrame,
        cutoff: pd.Timestamp,
        config: ParameterFitConfig,
    ) -> BootstrapParameterMember:
        event_ids = np.asarray(sorted(fights["event_id"].unique()), dtype=object)
        rng = np.random.Generator(np.random.PCG64DXSM(seed))
        sampled = rng.choice(event_ids, size=len(event_ids), replace=True)
        unique, counts = np.unique(sampled, return_counts=True)
        weights = {str(key): int(value) for key, value in zip(unique, counts)}
        rows = self._prepare_member_rows(fights, weights, cutoff, config)
        rows = self._attach_age(rows)
        covariates = self._fit_covariate_effects(rows)
        sufficient = self._sufficient_rows(rows)
        control_sufficient = self._takedown_control_sufficient(
            rounds, weights, config
        )
        sufficient_names = [
            column
            for column in sufficient.columns
            if column not in {"fighter_id", "division", "era"}
        ]
        global_stats = {
            key: float(value)
            for key, value in sufficient[sufficient_names].sum().items()
        }
        global_parameters = self._parameters_from_sufficient(
            global_stats,
            None,
            config,
            prior_scale=0.25,
            update_weak_mechanics=True,
        )
        if self.use_takedown_control_association and not control_sufficient.empty:
            global_control = {
                name: float(control_sufficient[name].sum())
                for name in (
                    "td_control_share_sum",
                    "td_control_opportunities",
                    "opp_td_control_share_sum",
                    "opp_td_control_opportunities",
                )
            }
            global_parameters = self._apply_takedown_control_association(
                global_parameters,
                global_control,
                global_parameters,
                TAKEDOWN_CONTROL_GLOBAL_PRIOR_OPPORTUNITIES,
            )
        pace_by_fighter = self._pace_decay_by_fighter(
            rounds, weights, global_parameters["pace_decay"]
        )
        contexts: dict[str, dict[str, float]] = {"__global__": global_parameters}
        context_sufficient = sufficient.groupby(
            ["division", "era"], sort=True
        )[sufficient_names].sum()
        for (division, era), context_stats in context_sufficient.iterrows():
            context_key = f"{str(division).strip() or 'Unknown'}|{era}"
            contexts[context_key] = self._parameters_from_sufficient(
                {key: float(value) for key, value in context_stats.items()},
                global_parameters,
                config,
                prior_scale=config.division_prior_fights / config.rate_prior_fights,
                update_weak_mechanics=True,
            )
            if self.use_takedown_control_association and not control_sufficient.empty:
                matching_control = control_sufficient.loc[
                    control_sufficient["division"].astype(str).eq(str(division))
                    & control_sufficient["era"].astype(str).eq(str(era))
                ]
                control_stats = {
                    name: float(matching_control[name].sum())
                    for name in (
                        "td_control_share_sum",
                        "td_control_opportunities",
                        "opp_td_control_share_sum",
                        "opp_td_control_opportunities",
                    )
                }
                contexts[context_key] = self._apply_takedown_control_association(
                    contexts[context_key],
                    control_stats,
                    global_parameters,
                    TAKEDOWN_CONTROL_CONTEXT_PRIOR_OPPORTUNITIES,
                )

        current_era = self._era_key(cutoff, config.era_years)
        fighters: dict[str, dict[str, float]] = {}
        fighter_sufficient = sufficient.groupby("fighter_id", sort=True)[
            sufficient_names
        ].sum()
        latest_by_fighter = (
            rows.sort_values(["date", "event_id", "fight_id"], kind="stable")
            .drop_duplicates("fighter_id", keep="last")
            .set_index("fighter_id")
        )
        for fighter_id, fighter_stats in fighter_sufficient.iterrows():
            latest = latest_by_fighter.loc[fighter_id]
            context_key = f"{str(latest['division']).strip() or 'Unknown'}|{current_era}"
            prior = contexts.get(context_key)
            if prior is None:
                same_division = [
                    (key, value)
                    for key, value in contexts.items()
                    if key.startswith(f"{str(latest['division']).strip() or 'Unknown'}|")
                ]
                prior = same_division[-1][1] if same_division else global_parameters
            fighters[str(fighter_id)] = self._parameters_from_sufficient(
                {key: float(value) for key, value in fighter_stats.items()},
                prior,
                config,
                pace_decay=pace_by_fighter.get(str(fighter_id), prior["pace_decay"]),
            )
            if self.use_takedown_control_association and not control_sufficient.empty:
                matching_control = control_sufficient.loc[
                    control_sufficient["fighter_id"].astype(str).eq(str(fighter_id))
                ]
                control_stats = {
                    name: float(matching_control[name].sum())
                    for name in (
                        "td_control_share_sum",
                        "td_control_opportunities",
                        "opp_td_control_share_sum",
                        "opp_td_control_opportunities",
                    )
                }
                fighters[str(fighter_id)] = self._apply_takedown_control_association(
                    fighters[str(fighter_id)],
                    control_stats,
                    prior,
                    TAKEDOWN_CONTROL_FIGHTER_PRIOR_OPPORTUNITIES,
                )
        return BootstrapParameterMember(
            member_index=member_index,
            bootstrap_seed=seed,
            sampled_event_count=len(event_ids),
            context_parameters=contexts,
            fighter_parameters=fighters,
            covariate_effects=covariates,
        )

    def fit(
        self,
        as_of: object,
        *,
        config: ParameterFitConfig | None = None,
        created_at_utc: object | None = None,
    ) -> ParameterEnsembleArtifact:
        """Fit a strictly causal card-bootstrap ensemble before ``as_of``."""

        config = config or ParameterFitConfig()
        fights, rounds, cutoff = self._before(as_of)
        reconciliation_counts = (
            self.round_stats.loc[self.round_stats["date"].lt(cutoff), "reconciliation_status"]
            .value_counts(dropna=False)
            .sort_index()
            .astype(int)
            .to_dict()
            if not self.round_stats.empty
            else {}
        )
        eligible_profile_ids = set(fights["fighter_id"].astype(str))
        normalized_profiles = self.fighter_profiles.reset_index(drop=True)
        causal_profiles = (
            normalized_profiles.loc[
                normalized_profiles["fighter_id"].astype(str).isin(
                    eligible_profile_ids
                ),
                ["fighter_id", "dob"],
            ]
            .sort_values("fighter_id", kind="stable")
            .reset_index(drop=True)
            if not self.fighter_profiles.empty
            else normalized_profiles[["fighter_id", "dob"]]
        )
        source_parts = {
            "fights": _frame_sha256(fights.loc[:, _FIT_FIGHT_COLUMNS]),
            "rounds": (
                None
                if rounds.empty
                else _frame_sha256(rounds.loc[:, self.fit_round_columns])
            ),
            "profiles": (
                None
                if causal_profiles.empty
                else _frame_sha256(causal_profiles)
            ),
            "strictly_before": cutoff.isoformat(),
            "use_takedown_control_association": (
                self.use_takedown_control_association
            ),
        }
        input_hash = canonical_sha256(source_parts)
        seed_sequence = np.random.SeedSequence(config.random_seed)
        child_seeds = seed_sequence.generate_state(config.bootstrap_members, dtype=np.uint64)
        members = tuple(
            self._fit_member(
                index,
                int(seed),
                fights,
                rounds,
                cutoff,
                config,
            )
            for index, seed in enumerate(child_seeds)
        )
        created = (
            datetime.now(timezone.utc).isoformat()
            if created_at_utc is None
            else _utc_iso(created_at_utc)
        )
        body: dict[str, object] = {
            "schema_version": PARAMETER_SCHEMA_VERSION,
            "model_version": self.parameter_model_version,
            "as_of_utc": cutoff.isoformat(),
            "trained_through": fights["date"].max().date().isoformat(),
            "input_sha256": input_hash,
            "config": asdict(config),
            "members": [member.to_dict() for member in members],
            "observed_fights": int(fights["fight_id"].nunique()),
            "observed_fighter_sides": int(len(fights)),
            "observed_round_sides": int(len(rounds)),
            "round_reconciliation_counts": reconciliation_counts,
            "created_at_utc": created,
        }
        content = dict(body)
        content.pop("created_at_utc")
        artifact = ParameterEnsembleArtifact(
            schema_version=PARAMETER_SCHEMA_VERSION,
            model_version=self.parameter_model_version,
            as_of_utc=cutoff.isoformat(),
            trained_through=str(body["trained_through"]),
            input_sha256=input_hash,
            config=config,
            members=members,
            observed_fights=int(body["observed_fights"]),
            observed_fighter_sides=int(body["observed_fighter_sides"]),
            observed_round_sides=int(body["observed_round_sides"]),
            round_reconciliation_counts={
                str(key): int(count)
                for key, count in reconciliation_counts.items()
            },
            created_at_utc=created,
            artifact_sha256=canonical_sha256(content),
        )
        artifact.validate()
        # The compact on-disk codec may use these frozen, causal inputs to
        # reconstruct exact member values without consulting mutable files.
        object.__setattr__(
            artifact,
            "_compact_fit_inputs",
            {
                "fights": fights.loc[:, _FIT_FIGHT_COLUMNS].copy(),
                "rounds": self.round_stats.loc[
                    self.round_stats["date"].lt(cutoff), self.fit_round_columns
                ].copy()
                if not self.round_stats.empty
                else self.round_stats.copy(),
                "profiles": causal_profiles.set_index("fighter_id", drop=False),
                "allow_legacy_unreconciled_rounds": (
                    self.allow_legacy_unreconciled_rounds
                ),
                "use_takedown_control_association": (
                    self.use_takedown_control_association
                ),
            },
        )
        return artifact

    def _snapshot_metadata(
        self, fighter_id: str, as_of: pd.Timestamp
    ) -> dict[str, object]:
        # Never look beyond the artifact/snapshot cutoff.
        history = self.raw_fights.loc[
            self.raw_fights["fighter_id"].eq(fighter_id)
            & self.raw_fights["date"].lt(as_of)
        ].sort_values(["date", "event_id", "fight_id"], kind="stable")
        profile = (
            self.fighter_profiles.loc[fighter_id]
            if fighter_id in self.fighter_profiles.index
            else None
        )
        name = (
            str(profile.get("name") or "").strip()
            if profile is not None
            else (
                str(history.iloc[-1].get("fighter") or "").strip()
                if not history.empty
                else fighter_id
            )
        )
        dob = profile.get("dob") if profile is not None else pd.NaT
        age = (
            float((as_of - dob).total_seconds() / (365.25 * 86400.0))
            if pd.notna(dob)
            else None
        )
        layoff = (
            int((as_of - history.iloc[-1]["date"]).total_seconds() // 86400.0)
            if not history.empty
            else None
        )
        observed_seconds = float(history["fight_seconds"].sum()) if not history.empty else 0.0
        if not self.round_stats.empty:
            observed_rounds = int(
                self.round_stats.loc[
                    self.round_stats["fighter_id"].eq(fighter_id)
                    & self.round_stats["date"].lt(as_of)
                    & self.round_stats["_fit_eligible"]
                ].shape[0]
            )
        else:
            observed_rounds = int(
                pd.to_numeric(history.get("round"), errors="coerce").fillna(0).sum()
            ) if "round" in history else 0
        return {
            "fighter_id": fighter_id,
            "fighter_name": name,
            "as_of_utc": as_of.isoformat(),
            "age_years": age,
            "experience_fights": int(len(history)),
            "layoff_days": layoff,
            "observed_fight_seconds": observed_seconds,
            "observed_rounds": observed_rounds,
        }

    def _snapshot_reliability_weights(
        self,
        fighter_id: str,
        as_of: pd.Timestamp,
        config: ParameterFitConfig,
    ) -> dict[str, float]:
        """Return causal, parameter-specific second-stage reliability weights.

        The fitted fighter vector is already empirically pooled. This research
        mode deliberately applies one additional exposure-based shrinkage step
        to test whether the remaining fighter deviations are still too noisy.
        Each opportunity definition mirrors the observable denominator used by
        the fitter; unsupported latent mechanics receive no fighter deviation.
        """

        cache = getattr(self, "_snapshot_reliability_cache", None)
        if cache is None:
            cache = {}
            self._snapshot_reliability_cache = cache
        cache_key = (fighter_id, as_of.isoformat(), config)
        if cache_key in cache:
            return dict(cache[cache_key])

        history = self.raw_fights.loc[
            self.raw_fights["fighter_id"].eq(fighter_id)
            & self.raw_fights["date"].lt(as_of)
        ].copy()
        weights = {name: 0.0 for name in PARAMETER_NAMES}
        if history.empty:
            cache[cache_key] = dict(weights)
            return weights

        def observed_sum(column: str) -> float:
            if column not in history:
                return 0.0
            values = pd.to_numeric(history[column], errors="coerce")
            if not values.notna().any():
                return 0.0
            return max(float(values.sum(min_count=1)), 0.0)

        def observed_minutes(column: str) -> float:
            if column not in history:
                return 0.0
            observed = pd.to_numeric(history[column], errors="coerce").notna()
            seconds = pd.to_numeric(
                history.loc[observed, "fight_seconds"], errors="coerce"
            )
            if not seconds.notna().any():
                return 0.0
            return max(float(seconds.sum(min_count=1)) / 60.0, 0.0)

        def reliability(exposure: float, prior: float) -> float:
            if exposure <= 0.0:
                return 0.0
            return _clip(exposure / (exposure + prior), 0.0, 1.0)

        rate_prior_minutes = config.rate_prior_fights * 15.0
        probability_prior = config.probability_prior_attempts
        rare_prior = config.rare_event_prior_opportunities
        strike_rate_weight = reliability(
            observed_minutes("sig_strikes_attempts"), rate_prior_minutes
        )
        for name in (
            "strike_rate_distance",
            "strike_rate_clinch",
            "strike_rate_ground",
        ):
            weights[name] = strike_rate_weight

        phase_attempts = math.fsum(
            observed_sum(name)
            for name in (
                "distance_strikes_attempts",
                "clinch_strikes_attempts",
                "ground_strikes_attempts",
            )
        )
        phase_weight = reliability(phase_attempts, probability_prior)
        for name in _SNAPSHOT_COMPOSITION_GROUPS[0]:
            weights[name] = phase_weight

        sig_attempts = observed_sum("sig_strikes_attempts")
        sig_landed = observed_sum("sig_strikes_landed")
        weights["strike_accuracy"] = reliability(sig_attempts, probability_prior)
        weights["strike_defense"] = reliability(
            observed_sum("opponent_sig_strikes_attempts"), probability_prior
        )
        target_landed = math.fsum(
            observed_sum(name)
            for name in (
                "head_strikes_landed",
                "body_strikes_landed",
                "leg_strikes_landed",
            )
        )
        target_weight = reliability(target_landed, probability_prior)
        for name in _SNAPSHOT_COMPOSITION_GROUPS[1]:
            weights[name] = target_weight
        weights["knockdown_rate_per_landed"] = reliability(
            sig_landed, rare_prior
        )

        weights["takedown_attempt_rate"] = reliability(
            observed_minutes("takedowns_attempts"), rate_prior_minutes
        )
        weights["takedown_accuracy"] = reliability(
            observed_sum("takedowns_attempts"), probability_prior
        )
        weights["takedown_defense"] = reliability(
            observed_sum("opponent_takedowns_attempts"), probability_prior
        )

        if self.use_takedown_control_association and not self.round_stats.empty:
            rounds = self.round_stats.loc[
                self.round_stats["fighter_id"].eq(fighter_id)
                & self.round_stats["date"].lt(as_of)
                & self.round_stats["_fit_eligible"]
            ].copy()
            required = {
                "takedowns_landed",
                "control",
                "fight_id",
                "round_number",
                "opponent_id",
            }
            if required.issubset(rounds.columns):
                own_opportunities = float(
                    (
                        pd.to_numeric(
                            rounds["takedowns_landed"], errors="coerce"
                        ).gt(0)
                        & pd.to_numeric(rounds["control"], errors="coerce").notna()
                    ).sum()
                )
                mirrored = self.round_stats.loc[
                    self.round_stats["date"].lt(as_of)
                    & self.round_stats["_fit_eligible"]
                ][
                    [
                        "fight_id",
                        "round_number",
                        "fighter_id",
                        "takedowns_landed",
                        "control",
                    ]
                ].rename(
                    columns={
                        "fighter_id": "opponent_id_join",
                        "takedowns_landed": "opponent_takedowns_landed",
                        "control": "opponent_control",
                    }
                )
                conceded = rounds.merge(
                    mirrored,
                    left_on=["fight_id", "round_number", "opponent_id"],
                    right_on=["fight_id", "round_number", "opponent_id_join"],
                    how="left",
                    validate="one_to_one",
                )
                conceded_opportunities = float(
                    (
                        pd.to_numeric(
                            conceded["opponent_takedowns_landed"], errors="coerce"
                        ).gt(0)
                        & pd.to_numeric(
                            conceded["opponent_control"], errors="coerce"
                        ).notna()
                    ).sum()
                )
                weights["ground_control_rate"] = reliability(
                    own_opportunities,
                    TAKEDOWN_CONTROL_FIGHTER_PRIOR_OPPORTUNITIES,
                )
                weights["escape_rate"] = reliability(
                    conceded_opportunities,
                    TAKEDOWN_CONTROL_FIGHTER_PRIOR_OPPORTUNITIES,
                )
        else:
            control_observed = pd.to_numeric(
                history["control"], errors="coerce"
            ).notna()
            control_exposure = pd.to_numeric(
                history.loc[control_observed, "fight_seconds"], errors="coerce"
            )
            opponent_control_observed = pd.to_numeric(
                history["opponent_control"], errors="coerce"
            ).notna()
            opponent_control_exposure = pd.to_numeric(
                history.loc[opponent_control_observed, "fight_seconds"],
                errors="coerce",
            )
            weights["ground_control_rate"] = reliability(
                float(control_exposure.sum(min_count=1))
                if control_exposure.notna().any()
                else 0.0,
                probability_prior * 60.0,
            )
            weights["escape_rate"] = reliability(
                float(opponent_control_exposure.sum(min_count=1))
                if opponent_control_exposure.notna().any()
                else 0.0,
                probability_prior * 60.0,
            )

        reversals = observed_sum("reversals")
        opponent_takedowns = observed_sum("opponent_takedowns_landed")
        weights["reversal_after_escape"] = reliability(
            max(reversals, opponent_takedowns), rare_prior
        )
        weights["submission_attempt_rate"] = reliability(
            observed_minutes("sub_attempts"), rate_prior_minutes
        )
        submission_attempts = observed_sum("sub_attempts")
        opponent_submission_attempts = observed_sum("opponent_sub_attempts")
        method = history["method"].map(self._method_bucket)
        result = history["result"].astype(str).str.upper()
        submission_wins = float((result.eq("W") & method.eq("submission")).sum())
        submission_losses = float((result.eq("L") & method.eq("submission")).sum())
        ko_losses = float((result.eq("L") & method.eq("ko_tko")).sum())
        weights["submission_finish_probability"] = reliability(
            max(submission_attempts, submission_wins), rare_prior
        )
        weights["submission_defense"] = reliability(
            max(opponent_submission_attempts, submission_losses), rare_prior
        )
        weights["ko_resistance"] = reliability(
            max(observed_sum("opponent_knockdowns"), ko_losses), rare_prior
        )

        if not self.round_stats.empty:
            later_rounds = self.round_stats.loc[
                self.round_stats["fighter_id"].eq(fighter_id)
                & self.round_stats["date"].lt(as_of)
                & self.round_stats["_fit_eligible"]
                & self.round_stats["round_number"].gt(1)
                & self.round_stats["sig_strikes_attempts"].notna()
            ]
            weights["pace_decay"] = reliability(float(len(later_rounds)), 10.0)
        cache[cache_key] = dict(weights)
        return weights

    @staticmethod
    def _blend_snapshot_parameters(
        fighter: Mapping[str, float],
        context: Mapping[str, float],
        weights: Mapping[str, float],
    ) -> dict[str, float]:
        """Shrink a fighter vector toward its causal context on natural scales."""

        result: dict[str, float] = {}
        composition_names = {
            name for group in _SNAPSHOT_COMPOSITION_GROUPS for name in group
        }
        for name in PARAMETER_NAMES:
            if name in composition_names:
                continue
            weight = _clip(float(weights.get(name, 0.0)), 0.0, 1.0)
            fighter_value = float(fighter[name])
            context_value = float(context[name])
            if name in _SNAPSHOT_POSITIVE_RATE_PARAMETERS:
                result[name] = math.exp(
                    (1.0 - weight) * math.log(max(context_value, 1e-12))
                    + weight * math.log(max(fighter_value, 1e-12))
                )
            else:
                epsilon = 1e-9
                fighter_probability = _clip(fighter_value, epsilon, 1.0 - epsilon)
                context_probability = _clip(context_value, epsilon, 1.0 - epsilon)
                fighter_logit = math.log(
                    fighter_probability / (1.0 - fighter_probability)
                )
                context_logit = math.log(
                    context_probability / (1.0 - context_probability)
                )
                blended_logit = (
                    (1.0 - weight) * context_logit + weight * fighter_logit
                )
                result[name] = 1.0 / (1.0 + math.exp(-blended_logit))

        for group in _SNAPSHOT_COMPOSITION_GROUPS:
            unnormalized = {}
            for name in group:
                weight = _clip(float(weights.get(name, 0.0)), 0.0, 1.0)
                unnormalized[name] = (
                    max(float(context[name]), 1e-12) ** (1.0 - weight)
                    * max(float(fighter[name]), 1e-12) ** weight
                )
            total = math.fsum(unnormalized.values())
            for name, value in unnormalized.items():
                result[name] = value / total
        return result

    @staticmethod
    def _construct_domain_dataclass(cls: type, values: Mapping[str, object]) -> object:
        """Construct a domain dataclass while keeping this fitter schema-isolated."""

        if not is_dataclass(cls):
            return cls(**values)
        accepted = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in accepted})

    def snapshot_for(
        self,
        artifact: ParameterEnsembleArtifact,
        fighter_id: str,
        *,
        division: str,
        member_index: int,
        as_of: object | None = None,
        parameter_mode: str = "full",
        _artifact_validated: bool = False,
    ) -> object:
        """Construct the engine's immutable ``FighterSnapshot`` for one member.

        ``as_of`` may equal or precede the artifact cutoff, but may never be
        later.  A snapshot from an earlier date should be built from an artifact
        fitted to that same cutoff; this method refuses a mismatch to prevent a
        seemingly historical snapshot from carrying future-fit parameters.
        """

        # Batch spec construction validates the immutable artifact once before
        # requesting both sides across every member. Standalone callers retain
        # the defensive validation default.
        if not _artifact_validated:
            artifact.validate()
        artifact_cutoff = pd.to_datetime(artifact.as_of_utc, utc=True)
        requested = (
            artifact_cutoff
            if as_of is None
            else pd.to_datetime(as_of, errors="raise", utc=True)
        )
        if requested != artifact_cutoff:
            raise ValueError("snapshot as_of must exactly match the fitted artifact cutoff")
        if member_index < 0 or member_index >= len(artifact.members):
            raise IndexError("bootstrap member index is out of range")
        if parameter_mode not in SNAPSHOT_PARAMETER_MODES:
            raise ValueError(
                "parameter_mode must be full, context_only, or reliability_weighted"
            )
        stable_id = _identity_token(fighter_id)
        if not stable_id:
            raise ValueError("fighter_id is blank")
        member = artifact.members[member_index]
        current_era = self._era_key(requested, artifact.config.era_years)
        context_key = f"{str(division).strip() or 'Unknown'}|{current_era}"
        context_parameters = member.context_parameters.get(
            context_key, member.context_parameters["__global__"]
        )
        fighter_parameters = member.fighter_parameters.get(stable_id)
        if fighter_parameters is None or parameter_mode == "context_only":
            parameters = context_parameters
            data_quality = "division_era_prior"
        elif parameter_mode == "reliability_weighted":
            parameters = self._blend_snapshot_parameters(
                fighter_parameters,
                context_parameters,
                self._snapshot_reliability_weights(
                    stable_id, requested, artifact.config
                ),
            )
            data_quality = "reliability_weighted_fighter_history"
        else:
            parameters = fighter_parameters
            data_quality = "fighter_history"
        metadata = self._snapshot_metadata(stable_id, requested)
        parameters = dict(parameters)
        effects = member.covariate_effects
        age_value = metadata.get("age_years")
        age_scaled = (
            (float(age_value) - 30.0) / 10.0
            if age_value is not None
            else effects.get("age_center", 0.0)
        )
        experience_scaled = math.log1p(
            max(float(metadata.get("experience_fights") or 0), 0.0)
        )
        layoff_value = metadata.get("layoff_days")
        layoff_scaled = (
            math.log1p(max(float(layoff_value), 0.0) / 365.25)
            if layoff_value is not None
            else effects.get("log_layoff_years_center", 0.0)
        )
        log_rate_adjustment = (
            effects.get("age_per_decade", 0.0)
            * (age_scaled - effects.get("age_center", 0.0))
            + effects.get("log_experience", 0.0)
            * (
                experience_scaled
                - effects.get("log_experience_center", 0.0)
            )
            + effects.get("log_layoff_years", 0.0)
            * (layoff_scaled - effects.get("log_layoff_years_center", 0.0))
        )
        multiplier = (
            1.0
            if parameter_mode == "context_only"
            else _clip(math.exp(log_rate_adjustment), 0.6, 1.4)
        )
        for field_name in (
            "strike_rate_distance",
            "strike_rate_clinch",
            "strike_rate_ground",
        ):
            parameters[field_name] = _clip(
                parameters[field_name] * multiplier, 0.0, 30.0
            )
        try:
            from .domain import FighterParameters, FighterSnapshot
        except ImportError as error:  # pragma: no cover - integration guard
            raise RuntimeError("fight_sim.domain must define the engine contracts") from error
        parameter_object = self._construct_domain_dataclass(
            FighterParameters, parameters
        )
        snapshot_values = {
            **metadata,
            "division": str(division).strip() or "Unknown",
            "parameters": parameter_object,
            "parameter_member_index": member_index,
            "data_quality": data_quality,
            "source_hash": artifact.artifact_sha256,
        }
        return self._construct_domain_dataclass(FighterSnapshot, snapshot_values)


def ensemble_parameter_interval(
    artifact: ParameterEnsembleArtifact,
    fighter_id: str,
    parameter: str,
    *,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> dict[str, float | int]:
    """Summarize bootstrap parameter/model uncertainty for one fighter."""

    if parameter not in PARAMETER_NAMES:
        raise ValueError(f"unknown parameter: {parameter}")
    if not (0 <= quantiles[0] < quantiles[1] <= 1):
        raise ValueError("quantiles must be ordered inside [0, 1]")
    values = [
        member.fighter_parameters[fighter_id][parameter]
        for member in artifact.members
        if fighter_id in member.fighter_parameters
    ]
    if not values:
        raise KeyError(f"fighter is absent from every bootstrap member: {fighter_id}")
    array = np.asarray(values, dtype=float)
    return {
        "n_members": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "lower": float(np.quantile(array, quantiles[0])),
        "upper": float(np.quantile(array, quantiles[1])),
    }


def simulator_config_for_member(
    member: BootstrapParameterMember,
    *,
    base: object | None = None,
) -> object:
    """Build global simulator mechanics from the same bootstrap outer draw."""

    from .domain import SimulatorConfig

    values = base.to_dict() if isinstance(base, SimulatorConfig) else {}
    values["no_contest_rate_per_minute"] = max(
        0.0,
        float(member.covariate_effects.get("global_no_contest_rate_per_minute", 0.0)),
    )
    values["other_finish_rate_per_minute"] = max(
        0.0,
        float(member.covariate_effects.get("global_other_finish_rate_per_minute", 0.0)),
    )
    return SimulatorConfig(**values)
