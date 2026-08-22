"""Rebuild candidate upcoming winner/method/duration forecasts offline."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from data_handler import DataHandler
from external_mma import load_approved_auxiliary
from fight_predictor import (
    PointInTimeDatasetBuilder,
    build_outcome_forecast_publication,
    evaluate_outcome_model,
    write_outcome_forecast_publication,
)


ROOT = Path(__file__).resolve().parent
POINT_IN_TIME_PATH = ROOT / "content/data/processed/ufc_fights_point_in_time.csv"
EVALUATION_PATH = ROOT / "content/data/external/outcome_model_evaluation.json"
FORECAST_PATH = ROOT / "content/data/external/outcome_forecasts.json"
CARD_PATH = ROOT / "content/data/external/card_info.json"
VEGAS_PATH = ROOT / "content/data/external/vegas_odds.json"


def main() -> int:
    handler = DataHandler()
    raw = handler.get("ufc_fights_reported_doubled")
    fighters = handler.get("fighter_stats")
    auxiliary = load_approved_auxiliary(
        ROOT / "content/data/processed/external_mma_auxiliary_doubled.csv",
        ROOT / "content/data/external_mma/model_policy.json",
    )
    builder = PointInTimeDatasetBuilder(
        raw, fighters, auxiliary_fights=auxiliary
    )
    # Replay the source state before asking for future matchup features.
    builder.build()
    frame = pd.read_csv(POINT_IN_TIME_PATH, low_memory=False)
    builder.training_data = frame.copy()
    feature_columns = tuple(
        column for column in frame if column.endswith("_diff")
    )
    model, evaluation = evaluate_outcome_model(frame, feature_columns)
    training_hash = sha256(POINT_IN_TIME_PATH.read_bytes()).hexdigest()
    evaluation["training_input_sha256"] = training_hash
    evaluation["feature_count"] = len(feature_columns)
    EVALUATION_PATH.write_text(
        json.dumps(
            evaluation,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    upcoming = pd.read_json(VEGAS_PATH)
    issued = {
        str(value).strip()
        for value in upcoming["forecast issued at"]
        if str(value).strip()
    }
    commits = {
        str(value).strip()
        for value in upcoming["forecast source commit"]
        if str(value).strip()
    }
    if len(issued) != 1 or len(commits) != 1:
        raise ValueError("upcoming forecasts require one issuance and source revision")
    publication = build_outcome_forecast_publication(
        model,
        builder,
        upcoming,
        card,
        selected_c=float(evaluation["selected_c"]),
        training_input_sha256=training_hash,
        model_trained_through=str(frame["date"].max()),
        forecast_issued_at_utc=next(iter(issued)),
        source_commit_sha=next(iter(commits)),
    )
    write_outcome_forecast_publication(FORECAST_PATH, publication)
    print(
        "Candidate outcome forecasts: "
        f"{publication['forecast_matchup_count']}/"
        f"{publication['matchup_count']} matchups"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
