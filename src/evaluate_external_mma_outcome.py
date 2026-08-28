"""Compare the candidate method/duration model with and without external history."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from data_handler.data_handler import atomic_write_text
from evaluate_winner_feature_challengers import (
    DEFAULT_AUXILIARY,
    DEFAULT_EXTERNAL_REPORT,
    DEFAULT_FIGHTER_STATS,
    DEFAULT_RAW_FIGHTS,
    load_frozen_research_auxiliary,
)
from fight_predictor import PointInTimeDatasetBuilder, evaluate_outcome_model


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "content/data/external_mma/outcome_feature_comparison.json"


def _headline(report: dict[str, object]) -> dict[str, object]:
    return {
        "selected_c": report["selected_c"],
        "holdout_start": report["holdout_start"],
        "holdout_end": report["holdout_end"],
        "holdout_fights": report["holdout_fights"],
        "joint_outcome": report["joint_outcome"],
        "winner": report["winner"],
        "method": report["method"],
        "total_rounds": report["total_rounds"],
    }


def _deltas(
    baseline: dict[str, object],
    external: dict[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {
        "joint_outcome_log_loss": (
            float(external["joint_outcome"]["log_loss"])
            - float(baseline["joint_outcome"]["log_loss"])
        ),
        "winner_log_loss": (
            float(external["winner"]["log_loss"])
            - float(baseline["winner"]["log_loss"])
        ),
        "method_log_loss": (
            float(external["method"]["log_loss"])
            - float(baseline["method"]["log_loss"])
        ),
        "total_round_log_loss": {},
    }
    baseline_totals = baseline["total_rounds"]
    external_totals = external["total_rounds"]
    for name in sorted(set(baseline_totals) & set(external_totals)):
        result["total_round_log_loss"][name] = (
            float(external_totals[name]["log_loss"])
            - float(baseline_totals[name]["log_loss"])
        )
    return result


def evaluate(
    raw: pd.DataFrame,
    fighters: pd.DataFrame,
    auxiliary: pd.DataFrame,
    auxiliary_sha256: str,
) -> dict[str, object]:
    baseline_builder = PointInTimeDatasetBuilder(raw, fighters)
    external_builder = PointInTimeDatasetBuilder(
        raw, fighters, auxiliary_fights=auxiliary
    )
    baseline_point = baseline_builder.build()
    external_point = external_builder.build()
    if baseline_point["fight_id"].tolist() != external_point["fight_id"].tolist():
        raise RuntimeError("external history changed the outcome-model label set")
    if baseline_point["target"].tolist() != external_point["target"].tolist():
        raise RuntimeError("external history changed an outcome-model target")
    baseline_columns = tuple(
        column for column in baseline_point if column.endswith("_diff")
    )
    external_columns = tuple(
        column for column in external_point if column.endswith("_diff")
    )
    if baseline_columns != external_columns:
        raise RuntimeError("external history changed the outcome feature contract")
    _baseline_model, baseline_report = evaluate_outcome_model(
        baseline_point, baseline_columns
    )
    _external_model, external_report = evaluate_outcome_model(
        external_point, external_columns
    )
    deltas = _deltas(baseline_report, external_report)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "candidate_only": True,
        "production_enabled": False,
        "auxiliary_sha256": auxiliary_sha256,
        "ufc_training_labels_unchanged": True,
        "feature_contract_unchanged": True,
        "baseline": _headline(baseline_report),
        "external_history": _headline(external_report),
        "external_minus_baseline": deltas,
        "decision": {
            "joint_outcome_point_log_loss_improved": (
                deltas["joint_outcome_log_loss"] < 0.0
            ),
            "winner_point_log_loss_improved": deltas["winner_log_loss"] < 0.0,
            "method_point_log_loss_improved": deltas["method_log_loss"] < 0.0,
            "production_action": (
                "none; this checks for collateral degradation before a separate "
                "winner-model policy review"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_FIGHTS)
    parser.add_argument("--fighters", type=Path, default=DEFAULT_FIGHTER_STATS)
    parser.add_argument("--auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    parser.add_argument("--external-report", type=Path, default=DEFAULT_EXTERNAL_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    auxiliary, metadata = load_frozen_research_auxiliary(
        arguments.auxiliary.resolve(), arguments.external_report.resolve()
    )
    report = evaluate(
        pd.read_csv(arguments.raw.resolve(), low_memory=False),
        pd.read_csv(arguments.fighters.resolve(), low_memory=False),
        auxiliary,
        str(metadata["auxiliary_sha256"]),
    )
    if not arguments.dry_run:
        atomic_write_text(
            arguments.output.resolve(),
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    deltas = report["external_minus_baseline"]
    print(
        "External-history outcome deltas: "
        f"joint={deltas['joint_outcome_log_loss']:+.6f}, "
        f"winner={deltas['winner_log_loss']:+.6f}, "
        f"method={deltas['method_log_loss']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
