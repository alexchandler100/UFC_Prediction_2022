"""Compare the current UFC model with an external-history replay candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from data_handler.data_handler import atomic_write_text
from fight_predictor import PointInTimeDatasetBuilder, TemporalFightPredictor


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "src" / "content" / "data"


def _train(
    raw: pd.DataFrame,
    fighters: pd.DataFrame,
    auxiliary: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    builder = PointInTimeDatasetBuilder(
        raw, fighters, auxiliary_fights=auxiliary
    )
    point_in_time = builder.build()
    predictor = TemporalFightPredictor(point_in_time, builder)
    return point_in_time, predictor.train()


def evaluate(
    raw: pd.DataFrame,
    fighters: pd.DataFrame,
    auxiliary: pd.DataFrame,
    auxiliary_sha256: str,
) -> dict[str, object]:
    if auxiliary.empty:
        raise ValueError("auxiliary history is empty")
    flags = auxiliary["emit_training_target"].astype(str).str.casefold()
    if not flags.isin({"false", "0"}).all():
        raise ValueError("auxiliary history contains a training-label row")
    baseline_rows, baseline = _train(raw, fighters, None)
    enriched_rows, enriched = _train(raw, fighters, auxiliary)
    if baseline_rows["fight_id"].tolist() != enriched_rows["fight_id"].tolist():
        raise ValueError("external replay changed the UFC training-label set or order")
    baseline_holdout = baseline["calibrated_model"]
    enriched_holdout = enriched["calibrated_model"]
    baseline_walk = baseline["walk_forward"]["aggregate"]
    enriched_walk = enriched["walk_forward"]["aggregate"]
    changed = baseline_rows[list(
        column for column in baseline_rows if column.endswith("_diff")
    )].ne(
        enriched_rows[list(
            column for column in enriched_rows if column.endswith("_diff")
        )]
    ).any(axis=1)
    former_one_sided = baseline_rows["has_history_diff"].abs().eq(1.0)
    newly_two_sided = former_one_sided & enriched_rows["has_history_diff"].eq(0.0)
    candidate_improved = (
        float(enriched_holdout["log_loss"]) < float(baseline_holdout["log_loss"])
        and float(enriched_walk["log_loss"]) < float(baseline_walk["log_loss"])
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "auxiliary_sha256": auxiliary_sha256,
        "auxiliary_physical_bouts": int(auxiliary["fight_url"].nunique()),
        "ufc_training_labels_unchanged": True,
        "ufc_feature_rows_changed": int(changed.sum()),
        "one_sided_ufc_histories_made_two_sided": int(newly_two_sided.sum()),
        "baseline": {
            "holdout": baseline_holdout,
            "walk_forward": baseline_walk,
            "holdout_start": baseline["holdout_start"],
            "holdout_end": baseline["holdout_end"],
        },
        "external_history": {
            "holdout": enriched_holdout,
            "walk_forward": enriched_walk,
            "holdout_start": enriched["holdout_start"],
            "holdout_end": enriched["holdout_end"],
        },
        "delta": {
            "holdout_log_loss": float(enriched_holdout["log_loss"])
            - float(baseline_holdout["log_loss"]),
            "holdout_accuracy": float(enriched_holdout["accuracy"])
            - float(baseline_holdout["accuracy"]),
            "walk_forward_log_loss": float(enriched_walk["log_loss"])
            - float(baseline_walk["log_loss"]),
            "walk_forward_accuracy": float(enriched_walk["accuracy"])
            - float(baseline_walk["accuracy"]),
        },
        "candidate_improved_both_log_loss_checks": candidate_improved,
        "approval_note": (
            "This gate tests predictive probability quality, not betting return. "
            "Keep betting disabled until timestamped odds yield enough settled bets."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path,
        default=DATA_ROOT / "processed" / "ufc_fights_reported_doubled.csv",
    )
    parser.add_argument(
        "--fighters", type=Path,
        default=DATA_ROOT / "processed" / "fighter_stats.csv",
    )
    parser.add_argument(
        "--auxiliary", type=Path,
        default=DATA_ROOT / "processed" / "external_mma_auxiliary_doubled.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DATA_ROOT / "external_mma" / "evaluation_report.json",
    )
    args = parser.parse_args()
    auxiliary_bytes = args.auxiliary.resolve().read_bytes()
    report = evaluate(
        pd.read_csv(args.raw.resolve(), low_memory=False),
        pd.read_csv(args.fighters.resolve(), low_memory=False),
        pd.read_csv(args.auxiliary.resolve(), low_memory=False),
        sha256(auxiliary_bytes).hexdigest(),
    )
    atomic_write_text(
        args.output.resolve(),
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
