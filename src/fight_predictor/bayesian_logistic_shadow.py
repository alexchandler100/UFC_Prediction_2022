"""Prospective, paper-only tracking for the frozen Bayesian-logistic blend."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import io
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from market_tracker._common import (
    MarketDataError,
    StoreIntegrityError,
    canonical_hash,
    canonical_json,
    iso_date,
    matchup_id_for,
    nonempty_text,
    probability,
    require_before_event,
    stable_id,
    utc_text,
    validated_git_commit,
    validated_sha256,
)
from market_tracker._storage import atomic_write_text, exclusive_store_lock
from market_tracker.blend import forecast_metrics
from market_tracker.forecasts import AppendResult

from .bayesian_logistic import BayesianLogisticConfig, bayesian_logistic_predict
from .point_in_time import PointInTimeDatasetBuilder, training_fingerprint


POLICY_VERSION = "bayesian-logistic-blend-shadow-v1"
MODEL_VERSION = "fully-bayesian-logistic-group-shrinkage-v1"
SCHEMA_VERSION = 1
SELECTED_BLEND_WEIGHT = 0.5462639465757038
SELECTED_VARIANT = "grouped_tight"
SELECTED_VARIANCE_PRIOR_SHAPE = 3.0
SELECTED_VARIANCE_PRIOR_SCALE = 0.02
MINIMUM_PROSPECTIVE_FIGHTS = 200
MINIMUM_PROSPECTIVE_EVENTS = 20


def default_shadow_config() -> BayesianLogisticConfig:
    """Return the frozen high-precision sampler used for live paper forecasts."""

    return BayesianLogisticConfig(
        burn_in=1_000,
        posterior_draws=1_000,
        chains=2,
        grouped_shrinkage=True,
        variance_prior_shape=SELECTED_VARIANCE_PRIOR_SHAPE,
        variance_prior_scale=SELECTED_VARIANCE_PRIOR_SCALE,
        seed=30_260_829,
    )


def _bounded_probability(value: object, field: str) -> float:
    return probability(value, field)


@dataclass(frozen=True)
class BayesianLogisticShadowForecast:
    """One immutable pre-fight comparison of production, Bayes, and the blend."""

    schema_version: int
    forecast_id: str
    policy_version: str
    matchup_id: str
    event_id: str
    event_date: str
    timing_precision: str
    event_start_utc: str | None
    fighter_id: str
    opponent_id: str
    fighter_name: str
    opponent_name: str
    forecast_issued_at_utc: str
    source_commit_sha: str
    experiment_sha256: str
    training_start: str
    training_through: str
    training_fights: int
    training_fingerprint_sha256: str
    model_id: str
    published_model_id: str
    published_model_probability: float
    bayesian_probability: float
    bayesian_lower_probability: float
    bayesian_upper_probability: float
    frozen_blend_probability: float
    bayesian_weight: float
    calibration_slope: float
    fighter_prior_fights: int
    opponent_prior_fights: int
    mean_chain_difference: float
    paper_only: bool
    candidate_only: bool
    execution_enabled: bool

    FIELDNAMES = tuple(__annotations__)

    @property
    def natural_key(self) -> str:
        return self.matchup_id

    @classmethod
    def create(
        cls,
        *,
        event_id: object,
        event_date: object,
        timing_precision: object,
        event_start_utc: object | None,
        fighter_id: object,
        opponent_id: object,
        fighter_name: object,
        opponent_name: object,
        forecast_issued_at_utc: object,
        source_commit_sha: object,
        experiment_sha256: object,
        training_start: object,
        training_through: object,
        training_fights: object,
        training_fingerprint_sha256: object,
        model_id: object,
        published_model_id: object,
        published_model_probability: object,
        bayesian_probability: object,
        bayesian_lower_probability: object,
        bayesian_upper_probability: object,
        frozen_blend_probability: object,
        calibration_slope: object,
        fighter_prior_fights: object,
        opponent_prior_fights: object,
        mean_chain_difference: object,
    ) -> "BayesianLogisticShadowForecast":
        event = stable_id(event_id, "event_id")
        fighter = stable_id(fighter_id, "fighter_id")
        opponent = stable_id(opponent_id, "opponent_id")
        if fighter == opponent:
            raise MarketDataError("fighter_id and opponent_id must differ")
        issued, event_day, precision, event_start = require_before_event(
            forecast_issued_at_utc,
            event_date=event_date,
            timing_precision=timing_precision,
            event_start_utc=event_start_utc,
            observed_field="forecast_issued_at_utc",
        )
        training_first = iso_date(training_start, "training_start")
        training_last = iso_date(training_through, "training_through")
        if not training_first <= training_last:
            raise MarketDataError("training_start must not follow training_through")
        if training_last > issued.date().isoformat() or training_last >= event_day:
            raise MarketDataError(
                "training_through must be observed by issuance and precede the event"
            )
        try:
            fight_count = int(training_fights)
            fighter_history = int(fighter_prior_fights)
            opponent_history = int(opponent_prior_fights)
        except (TypeError, ValueError) as error:
            raise MarketDataError("fight counts must be integers") from error
        if fight_count < 500 or min(fighter_history, opponent_history) < 0:
            raise MarketDataError("fight counts are outside the supported range")
        values = {
            "published_model_probability": _bounded_probability(
                published_model_probability, "published_model_probability"
            ),
            "bayesian_probability": _bounded_probability(
                bayesian_probability, "bayesian_probability"
            ),
            "bayesian_lower_probability": _bounded_probability(
                bayesian_lower_probability, "bayesian_lower_probability"
            ),
            "bayesian_upper_probability": _bounded_probability(
                bayesian_upper_probability, "bayesian_upper_probability"
            ),
            "frozen_blend_probability": _bounded_probability(
                frozen_blend_probability, "frozen_blend_probability"
            ),
        }
        if not (
            values["bayesian_lower_probability"]
            <= values["bayesian_probability"]
            <= values["bayesian_upper_probability"]
        ):
            raise MarketDataError("Bayesian probability interval is unordered")
        try:
            slope = float(calibration_slope)
            chain_difference = float(mean_chain_difference)
        except (TypeError, ValueError) as error:
            raise MarketDataError("sampler diagnostics must be numeric") from error
        if not math.isfinite(slope) or slope <= 0.0:
            raise MarketDataError("calibration_slope must be positive")
        if not math.isfinite(chain_difference) or chain_difference < 0.0:
            raise MarketDataError("mean_chain_difference must be nonnegative")
        body = {
            "schema_version": SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "matchup_id": matchup_id_for(event, fighter, opponent),
            "event_id": event,
            "event_date": event_day,
            "timing_precision": precision,
            "event_start_utc": event_start,
            "fighter_id": fighter,
            "opponent_id": opponent,
            "fighter_name": nonempty_text(fighter_name, "fighter_name"),
            "opponent_name": nonempty_text(opponent_name, "opponent_name"),
            "forecast_issued_at_utc": utc_text(issued, "forecast_issued_at_utc"),
            "source_commit_sha": validated_git_commit(source_commit_sha),
            "experiment_sha256": validated_sha256(
                experiment_sha256, "experiment_sha256"
            ),
            "training_start": training_first,
            "training_through": training_last,
            "training_fights": fight_count,
            "training_fingerprint_sha256": validated_sha256(
                training_fingerprint_sha256, "training_fingerprint_sha256"
            ),
            "model_id": nonempty_text(model_id, "model_id"),
            "published_model_id": nonempty_text(
                published_model_id, "published_model_id"
            ),
            **values,
            "bayesian_weight": SELECTED_BLEND_WEIGHT,
            "calibration_slope": slope,
            "fighter_prior_fights": fighter_history,
            "opponent_prior_fights": opponent_history,
            "mean_chain_difference": chain_difference,
            "paper_only": True,
            "candidate_only": True,
            "execution_enabled": False,
        }
        forecast_id = canonical_hash(body)
        return cls(forecast_id=forecast_id, **body)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "BayesianLogisticShadowForecast":
        if set(value) != set(cls.FIELDNAMES):
            missing = sorted(set(cls.FIELDNAMES) - set(value))
            extra = sorted(set(value) - set(cls.FIELDNAMES))
            raise MarketDataError(
                f"Bayesian shadow fields differ; missing={missing}, extra={extra}"
            )
        if int(value["schema_version"]) != SCHEMA_VERSION:
            raise MarketDataError("unsupported Bayesian shadow schema")
        if value["policy_version"] != POLICY_VERSION:
            raise MarketDataError("unsupported Bayesian shadow policy")
        if (
            value["paper_only"] is not True
            or value["candidate_only"] is not True
            or value["execution_enabled"] is not False
        ):
            raise MarketDataError("Bayesian shadow must remain paper-only")
        rebuilt = cls.create(
            **{
                key: value[key]
                for key in cls.FIELDNAMES
                if key
                not in {
                    "schema_version",
                    "forecast_id",
                    "policy_version",
                    "matchup_id",
                    "bayesian_weight",
                    "paper_only",
                    "candidate_only",
                    "execution_enabled",
                }
            }
        )
        if value["forecast_id"] != rebuilt.forecast_id:
            raise MarketDataError("Bayesian shadow forecast_id is invalid")
        if value["matchup_id"] != rebuilt.matchup_id:
            raise MarketDataError("Bayesian shadow matchup_id is invalid")
        if float(value["bayesian_weight"]) != SELECTED_BLEND_WEIGHT:
            raise MarketDataError("Bayesian shadow blend weight was changed")
        return rebuilt

    def to_mapping(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.FIELDNAMES}


class BayesianLogisticShadowStore:
    """Crash-safe CSV/JSONL mirrors that never replace a matchup forecast."""

    def __init__(self, csv_path: Path, jsonl_path: Path):
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)

    @staticmethod
    def _index(
        records: Sequence[BayesianLogisticShadowForecast],
    ) -> dict[str, BayesianLogisticShadowForecast]:
        indexed: dict[str, BayesianLogisticShadowForecast] = {}
        natural: set[str] = set()
        for record in records:
            if record.forecast_id in indexed:
                raise StoreIntegrityError("duplicate Bayesian shadow forecast_id")
            if record.natural_key in natural:
                raise StoreIntegrityError("a Bayesian shadow matchup was rewritten")
            indexed[record.forecast_id] = record
            natural.add(record.natural_key)
        return indexed

    def _read_jsonl(self) -> list[BayesianLogisticShadowForecast]:
        if not self.jsonl_path.exists():
            return []
        records = []
        with self.jsonl_path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    raise StoreIntegrityError(
                        f"blank Bayesian shadow JSONL line {number}"
                    )
                try:
                    value = json.loads(line)
                    records.append(BayesianLogisticShadowForecast.from_mapping(value))
                except (json.JSONDecodeError, MarketDataError) as error:
                    raise StoreIntegrityError(
                        f"invalid Bayesian shadow JSONL line {number}: {error}"
                    ) from error
        self._index(records)
        return records

    def _read_csv(self) -> list[BayesianLogisticShadowForecast]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if (
                tuple(reader.fieldnames or ())
                != BayesianLogisticShadowForecast.FIELDNAMES
            ):
                raise StoreIntegrityError("Bayesian shadow CSV schema is invalid")
            records = [
                BayesianLogisticShadowForecast.from_mapping(
                    {
                        **row,
                        "schema_version": int(row["schema_version"]),
                        "training_fights": int(row["training_fights"]),
                        "fighter_prior_fights": int(row["fighter_prior_fights"]),
                        "opponent_prior_fights": int(row["opponent_prior_fights"]),
                        "paper_only": row["paper_only"].casefold() == "true",
                        "candidate_only": row["candidate_only"].casefold() == "true",
                        "execution_enabled": (
                            row["execution_enabled"].casefold() == "true"
                        ),
                        "event_start_utc": row["event_start_utc"] or None,
                    }
                )
                for row in reader
            ]
        self._index(records)
        return records

    def read(self) -> tuple[BayesianLogisticShadowForecast, ...]:
        jsonl = self._read_jsonl()
        csv_rows = self._read_csv()
        if not jsonl:
            return tuple(csv_rows)
        if not csv_rows:
            return tuple(jsonl)
        shared_length = min(len(jsonl), len(csv_rows))
        if jsonl[:shared_length] != csv_rows[:shared_length]:
            raise StoreIntegrityError("Bayesian shadow CSV and JSONL diverged")
        # A process interruption can occur after one atomic mirror replacement
        # but before the other. The longer immutable prefix is authoritative.
        return tuple(jsonl if len(jsonl) >= len(csv_rows) else csv_rows)

    @staticmethod
    def _render_jsonl(
        records: Iterable[BayesianLogisticShadowForecast],
    ) -> str:
        return "".join(
            f"{canonical_json(record.to_mapping())}\n" for record in records
        )

    @staticmethod
    def _render_csv(
        records: Iterable[BayesianLogisticShadowForecast],
    ) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=BayesianLogisticShadowForecast.FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_mapping())
        return output.getvalue()

    def append(
        self, forecasts: Iterable[BayesianLogisticShadowForecast]
    ) -> AppendResult:
        pending = tuple(forecasts)
        lock = self.jsonl_path.with_name(f".{self.jsonl_path.name}.lock")
        with exclusive_store_lock(lock):
            existing = list(self.read())
            indexed = self._index(existing)
            natural = {record.natural_key for record in existing}
            additions = []
            duplicates = []
            for forecast in pending:
                if not isinstance(forecast, BayesianLogisticShadowForecast):
                    raise TypeError("append accepts Bayesian shadow forecasts only")
                prior = indexed.get(forecast.forecast_id)
                if prior is not None:
                    if prior != forecast:
                        raise StoreIntegrityError("Bayesian forecast_id was rewritten")
                    duplicates.append(forecast.forecast_id)
                    continue
                if forecast.natural_key in natural:
                    raise StoreIntegrityError("Bayesian matchup forecast was rewritten")
                additions.append(forecast)
                indexed[forecast.forecast_id] = forecast
                natural.add(forecast.natural_key)
            additions.sort(
                key=lambda item: (item.forecast_issued_at_utc, item.forecast_id)
            )
            combined = [*existing, *additions]
            self._index(combined)
            atomic_write_text(self.jsonl_path, self._render_jsonl(combined))
            atomic_write_text(self.csv_path, self._render_csv(combined))
            return AppendResult(
                added_ids=tuple(item.forecast_id for item in additions),
                duplicate_ids=tuple(duplicates),
                total_records=len(combined),
                dataset_sha256=canonical_hash(
                    [item.to_mapping() for item in combined]
                ),
            )


def _logit(values: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(bounded / (1.0 - bounded))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def frozen_blend(
    published_probability: np.ndarray, bayesian_probability: np.ndarray
) -> np.ndarray:
    return _sigmoid(
        (1.0 - SELECTED_BLEND_WEIGHT) * _logit(published_probability)
        + SELECTED_BLEND_WEIGHT * _logit(bayesian_probability)
    )


def _fit_calibration_slope(target: np.ndarray, raw_probability: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    logits = _logit(raw_probability)

    def objective(slope: float) -> float:
        predicted = _sigmoid(float(slope) * logits)
        bounded = np.clip(predicted, 1e-12, 1.0 - 1e-12)
        return float(
            -np.mean(target * np.log(bounded) + (1 - target) * np.log1p(-bounded))
        )

    result = minimize_scalar(objective, bounds=(0.05, 3.0), method="bounded")
    if not result.success or not math.isfinite(float(result.x)):
        raise RuntimeError("Bayesian shadow calibration failed")
    return float(result.x)


def _upcoming_features(
    builder: PointInTimeDatasetBuilder,
    upcoming: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for _, fight in upcoming.iterrows():
        features = builder.matchup_features(
            str(fight["fighter id"]),
            str(fight["opponent id"]),
            str(fight["date"]),
            str(fight["division"]),
        )
        rows.append(
            {
                **{
                    feature: float(features.iloc[0][feature])
                    for feature in feature_columns
                },
                "fighter_id": str(fight["fighter id"]),
                "opponent_id": str(fight["opponent id"]),
                **fight.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def _validated_chain_difference(diagnostics: Mapping[str, object]) -> float:
    difference = diagnostics.get(
        "mean_absolute_chain_mean_probability_difference"
    )
    if difference is None or not math.isfinite(float(difference)):
        raise RuntimeError("Bayesian shadow chains did not produce a comparison")
    if float(difference) > 0.01:
        raise RuntimeError(
            "Bayesian shadow chains disagree by more than one percentage point"
        )
    chains = diagnostics.get("sampler_chains")
    if not isinstance(chains, list) or len(chains) < 2:
        raise RuntimeError("Bayesian shadow requires two independent chains")
    for chain in chains:
        if not isinstance(chain, dict):
            raise RuntimeError("Bayesian shadow chain diagnostics are invalid")
        acceptance = float(chain.get("retained_acceptance_rate", math.nan))
        if not math.isfinite(acceptance) or not 0.5 <= acceptance <= 1.0:
            raise RuntimeError("Bayesian shadow sampler acceptance is unstable")
    return float(difference)


def build_shadow_forecasts(
    training: pd.DataFrame,
    builder: PointInTimeDatasetBuilder,
    upcoming: pd.DataFrame,
    *,
    forecast_issued_at_utc: datetime | str,
    source_commit_sha: str,
    experiment_sha256: str,
    existing_matchup_ids: Iterable[str] = (),
    event_start_utc: datetime | str | None = None,
    config: BayesianLogisticConfig | None = None,
) -> tuple[BayesianLogisticShadowForecast, ...]:
    """Fit the frozen recipe and return only matchups not already locked."""

    required = {
        "event id",
        "date",
        "fighter id",
        "opponent id",
        "fighter name",
        "opponent name",
        "division",
        "model id",
        "model probability",
        "fighter prior fights",
        "opponent prior fights",
    }
    missing = sorted(required - set(upcoming.columns))
    if missing:
        raise ValueError(f"upcoming Bayesian shadow rows are missing {missing}")
    frame = upcoming.copy()
    for column in ("fighter id", "opponent id", "event id", "model id"):
        frame = frame.loc[frame[column].astype(str).str.strip().ne("")]
    frame["model probability"] = pd.to_numeric(
        frame["model probability"], errors="coerce"
    )
    frame = frame.loc[frame["model probability"].between(0.0, 1.0, inclusive="neither")]
    locked = set(existing_matchup_ids)
    frame = frame.loc[
        [
            matchup_id_for(row["event id"], row["fighter id"], row["opponent id"])
            not in locked
            for _, row in frame.iterrows()
        ]
    ].reset_index(drop=True)
    if frame.empty:
        return ()
    event_ids = set(frame["event id"].astype(str))
    event_dates = set(pd.to_datetime(frame["date"], errors="raise").dt.date)
    if len(event_ids) != 1 or len(event_dates) != 1:
        raise ValueError("Bayesian shadow generation requires exactly one event")
    issued, event_day, timing_precision, normalized_start = require_before_event(
        forecast_issued_at_utc,
        event_date=next(iter(event_dates)),
        timing_precision="timestamp" if event_start_utc is not None else "date",
        event_start_utc=event_start_utc,
        observed_field="forecast_issued_at_utc",
    )
    training_frame = training.copy()
    training_frame["date"] = pd.to_datetime(training_frame["date"], errors="raise")
    if (
        training_frame.empty
        or training_frame["date"].max().date() > issued.date()
        or training_frame["date"].max().date().isoformat() >= event_day
    ):
        raise ValueError(
            "Bayesian shadow training must be observed by issuance and pre-event"
        )
    features = tuple(builder.feature_columns)
    validation_year = issued.year - 1
    calibration_fit = training_frame.loc[
        training_frame["date"].dt.year < validation_year
    ].copy()
    calibration = training_frame.loc[
        training_frame["date"].dt.year == validation_year
    ].copy()
    if len(calibration_fit) < 500 or len(calibration) < 100:
        raise ValueError("Bayesian shadow needs a complete prior year for calibration")
    selected = config or default_shadow_config()
    if (
        not selected.grouped_shrinkage
        or selected.variance_prior_shape != SELECTED_VARIANCE_PRIOR_SHAPE
        or selected.variance_prior_scale != SELECTED_VARIANCE_PRIOR_SCALE
    ):
        raise ValueError("Bayesian shadow sampler changed the frozen prior")
    inner = bayesian_logistic_predict(
        calibration_fit,
        calibration,
        features,
        config=replace(selected, seed=selected.seed + validation_year * 2),
    )
    inner_chain_difference = _validated_chain_difference(inner.diagnostics)
    slope = _fit_calibration_slope(
        calibration["target"].to_numpy(dtype=int), inner.probability
    )
    prediction = _upcoming_features(builder, frame, features)
    final = bayesian_logistic_predict(
        training_frame,
        prediction,
        features,
        config=replace(selected, seed=selected.seed + validation_year * 2 + 1),
    )
    final_chain_difference = _validated_chain_difference(final.diagnostics)
    bayesian_probability = _sigmoid(slope * _logit(final.probability))
    bayesian_lower = _sigmoid(slope * _logit(final.lower_probability))
    bayesian_upper = _sigmoid(slope * _logit(final.upper_probability))
    published_probability = prediction["model probability"].to_numpy(dtype=float)
    blend_probability = frozen_blend(published_probability, bayesian_probability)
    fingerprint = training_fingerprint(training_frame, features)
    model_body = {
        "model_version": MODEL_VERSION,
        "variant": SELECTED_VARIANT,
        "training_fingerprint_sha256": fingerprint,
        "experiment_sha256": validated_sha256(
            experiment_sha256, "experiment_sha256"
        ),
        "calibration_slope": slope,
        "config": asdict(selected),
        "features": list(features),
    }
    model_id = canonical_hash(model_body)[:20]
    mean_chain_difference = max(inner_chain_difference, final_chain_difference)
    forecasts = []
    for index, fight in prediction.reset_index(drop=True).iterrows():
        forecasts.append(
            BayesianLogisticShadowForecast.create(
                event_id=fight["event id"],
                event_date=event_day,
                timing_precision=timing_precision,
                event_start_utc=normalized_start,
                fighter_id=fight["fighter id"],
                opponent_id=fight["opponent id"],
                fighter_name=fight["fighter name"],
                opponent_name=fight["opponent name"],
                forecast_issued_at_utc=issued,
                source_commit_sha=source_commit_sha,
                experiment_sha256=experiment_sha256,
                training_start=training_frame["date"].min().date(),
                training_through=training_frame["date"].max().date(),
                training_fights=len(training_frame),
                training_fingerprint_sha256=fingerprint,
                model_id=model_id,
                published_model_id=fight["model id"],
                published_model_probability=published_probability[index],
                bayesian_probability=bayesian_probability[index],
                bayesian_lower_probability=bayesian_lower[index],
                bayesian_upper_probability=bayesian_upper[index],
                frozen_blend_probability=blend_probability[index],
                calibration_slope=slope,
                fighter_prior_fights=fight["fighter prior fights"],
                opponent_prior_fights=fight["opponent prior fights"],
                mean_chain_difference=mean_chain_difference,
            )
        )
    return tuple(forecasts)


def _loss(target: int, forecast: float) -> float:
    bounded = float(np.clip(forecast, 1e-12, 1.0 - 1e-12))
    return -math.log(bounded if target else 1.0 - bounded)


def _paired_interval(
    rows: Sequence[tuple[str, int, BayesianLogisticShadowForecast]],
    candidate_field: str,
) -> dict[str, object]:
    grouped: dict[str, list[float]] = {}
    for event_id, target, record in rows:
        difference = _loss(target, float(getattr(record, candidate_field))) - _loss(
            target, record.published_model_probability
        )
        grouped.setdefault(event_id, []).append(difference)
    observed = [value for block in grouped.values() for value in block]
    result: dict[str, object] = {
        "definition": f"{candidate_field} minus published production log loss",
        "point_difference": float(np.mean(observed)) if observed else None,
        "event_count": len(grouped),
        "fight_count": len(observed),
        "bootstrap_samples": 0,
        "ci_95_lower": None,
        "ci_95_upper": None,
    }
    if len(grouped) < 2:
        return result
    blocks = [grouped[key] for key in sorted(grouped)]
    generator = random.Random(int(canonical_hash({"blocks": blocks})[:16], 16))
    samples = []
    for _ in range(10_000):
        chosen = [generator.choice(blocks) for _ in blocks]
        values = [value for block in chosen for value in block]
        samples.append(float(np.mean(values)))
    samples.sort()
    result.update(
        {
            "bootstrap_samples": len(samples),
            "ci_95_lower": float(np.quantile(samples, 0.025)),
            "ci_95_upper": float(np.quantile(samples, 0.975)),
        }
    )
    return result


def score_shadow_forecasts(
    forecasts: Sequence[BayesianLogisticShadowForecast],
    outcomes: Mapping[tuple[str, str, str], tuple[int | None, str]],
    completed_events: set[str],
    ambiguous_matchups: set[tuple[str, str, str]],
) -> dict[str, object]:
    """Score locked forecasts from result IDs without changing the ledger."""

    scored: list[tuple[str, int, BayesianLogisticShadowForecast]] = []
    unresolved_completed = 0
    for record in forecasts:
        first, second = sorted((record.fighter_id, record.opponent_id))
        key = (record.event_id, first, second)
        if key in ambiguous_matchups:
            continue
        result = outcomes.get(key)
        if result is None:
            unresolved_completed += int(record.event_id in completed_events)
            continue
        canonical_target, _fight_id = result
        if canonical_target is None:
            continue
        target = int(canonical_target)
        if record.fighter_id != first:
            target = 1 - target
        scored.append((record.event_id, target, record))

    model_fields = {
        "published_production": "published_model_probability",
        "bayesian_logistic": "bayesian_probability",
        "frozen_blend": "frozen_blend_probability",
    }
    metrics = {}
    targets = [target for _, target, _ in scored]
    for name, field in model_fields.items():
        values = [float(getattr(record, field)) for _, _, record in scored]
        metrics[name] = (
            forecast_metrics(values, targets).to_mapping() if values else None
        )
    intervals = {
        "bayesian_minus_production": _paired_interval(
            scored, "bayesian_probability"
        ),
        "blend_minus_production": _paired_interval(
            scored, "frozen_blend_probability"
        ),
    }
    blend_interval = intervals["blend_minus_production"]
    enough_data = (
        len(scored) >= MINIMUM_PROSPECTIVE_FIGHTS
        and len({event for event, _, _ in scored}) >= MINIMUM_PROSPECTIVE_EVENTS
    )
    production = metrics["published_production"]
    blend = metrics["frozen_blend"]
    return {
        "policy_version": POLICY_VERSION,
        "paper_only": True,
        "candidate_only": True,
        "execution_enabled": False,
        "locked_forecasts": len(forecasts),
        "scored_fights": len(scored),
        "settled_events": len({event for event, _, _ in scored}),
        "unresolved_completed_forecasts": unresolved_completed,
        "metrics": metrics,
        "paired_log_loss_intervals": intervals,
        "forecast_dataset_sha256": canonical_hash(
            [record.to_mapping() for record in forecasts]
        ),
        "scored_dataset_sha256": canonical_hash(
            [
                {
                    "forecast_id": record.forecast_id,
                    "event_id": event,
                    "target": target,
                }
                for event, target, record in scored
            ]
        ),
        "promotion_gate": {
            "status": "collecting_prospective_evidence",
            "minimum_scored_fights": MINIMUM_PROSPECTIVE_FIGHTS,
            "minimum_settled_events": MINIMUM_PROSPECTIVE_EVENTS,
            "count_requirements_met": enough_data,
            "log_loss_improvement_requirement_met": (
                enough_data
                and blend_interval["ci_95_upper"] is not None
                and float(blend_interval["ci_95_upper"]) < 0.0
            ),
            "brier_not_worse_requirement_met": (
                enough_data
                and production is not None
                and blend is not None
                and float(blend["brier_score"])
                <= float(production["brier_score"])
            ),
            "automatic_production_change": False,
            "execution_enabled": False,
        },
    }
