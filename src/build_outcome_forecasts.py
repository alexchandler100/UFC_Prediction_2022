"""Rebuild candidate upcoming winner/method/duration forecasts offline."""

from __future__ import annotations

from hashlib import sha256
from datetime import datetime, timezone
import argparse
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
from fight_predictor.outcome_model import InsufficientVerifiedScheduleData, DiscreteTimeOutcomeModel
from fight_predictor.upcoming_outcomes import publish_upcoming_outcomes


ROOT = Path(__file__).resolve().parent
POINT_IN_TIME_PATH = ROOT / "content/data/processed/ufc_fights_point_in_time.csv"
EVALUATION_PATH = ROOT / "content/data/external/outcome_model_evaluation.json"
FORECAST_PATH = ROOT / "content/data/external/outcome_forecasts.json"
CARD_PATH = ROOT / "content/data/external/card_info.json"
VEGAS_PATH = ROOT / "content/data/external/vegas_odds.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upcoming-cards-only', action='store_true',
                        help='Extend the existing evaluated model to future cards without changing current-card publications.')
    args = parser.parse_args()
    # Git may check out CSV text with CRLF on Windows. Preserve the training
    # fingerprint of the Linux publication when only line endings differ.
    training_bytes = POINT_IN_TIME_PATH.read_bytes()
    training_hash = sha256(training_bytes).hexdigest()
    evaluation = None
    if args.upcoming_cards_only:
        evaluation = json.loads(EVALUATION_PATH.read_text(encoding='utf-8'))
        accepted_hashes = {training_hash, sha256(training_bytes.replace(b'\r\n', b'\n')).hexdigest()}
        if evaluation['training_input_sha256'] not in accepted_hashes:
            raise ValueError('saved outcome evaluation does not match the training data')
        training_hash = evaluation['training_input_sha256']
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
    if args.upcoming_cards_only:
        assert evaluation is not None
        model = None if evaluation.get('status') == 'unavailable_verified_schedule_history' else DiscreteTimeOutcomeModel(
            feature_columns, c_value=float(evaluation['selected_c'])).fit(frame)
        if model is not None:
            model.total_calibration_artifact = evaluation.get('bayesian_total_calibration')
    else:
        try:
            model, evaluation = evaluate_outcome_model(frame, feature_columns)
        except InsufficientVerifiedScheduleData as error:
            model = None
            evaluation = {"status": "unavailable_verified_schedule_history",
                          "reason": str(error), "selected_c": 0.0}
        evaluation["training_input_sha256"] = training_hash
        evaluation["feature_count"] = len(feature_columns)
        EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2, sort_keys=True,
                                  ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
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
    contract = dict(
        selected_c=float(evaluation["selected_c"]),
        training_input_sha256=training_hash,
        model_trained_through=str(frame["date"].max()),
        forecast_issued_at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit_sha=next(iter(commits)),
        unavailable_reason=evaluation.get("reason", "Insufficient verified schedule history."),
    )
    if not args.upcoming_cards_only:
        publication = build_outcome_forecast_publication(model, builder, upcoming, card, **contract)
        write_outcome_forecast_publication(FORECAST_PATH, publication,
            archive_directory=ROOT / "content/data/market/joint_forecast_archive")
    announced = json.loads((FORECAST_PATH.parent / 'all_upcoming_forecasts.json').read_text(encoding='utf-8'))
    publications = publish_upcoming_outcomes(model, builder, announced,
        FORECAST_PATH.parent / 'upcoming_outcome_forecasts',
        archive_directory=ROOT / 'content/data/market/joint_forecast_archive', **contract)
    print(f"Upcoming outcome forecasts: {len(publications)} cards / "
          f"{sum(p['forecast_matchup_count'] for p in publications)} fights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
