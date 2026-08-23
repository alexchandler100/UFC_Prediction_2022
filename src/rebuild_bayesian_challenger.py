"""Rebuild the Bayesian winner challenger from frozen local publications.

This command performs no network access.  It exists so the challenger can be
reproduced, evaluated, and attached to the already-published upcoming card
without running the authoritative UFCStats/odds update.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_handler import DataHandler
from external_mma import load_approved_auxiliary
from fight_predictor import (
    BayesianLogisticChallenger,
    PointInTimeDatasetBuilder,
    TemporalFightPredictor,
)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "content" / "data"
POINT_IN_TIME_PATH = DATA_ROOT / "processed" / "ufc_fights_point_in_time.csv"
WINNER_MODEL_PATH = DATA_ROOT / "external" / "winner_model.json"
BAYESIAN_MODEL_PATH = (
    DATA_ROOT / "external" / "bayesian_winner_challenger.json"
)
CARD_PATH = DATA_ROOT / "external" / "card_info.json"
AUXILIARY_PATH = DATA_ROOT / "processed" / "external_mma_auxiliary_doubled.csv"
AUXILIARY_POLICY_PATH = DATA_ROOT / "external_mma" / "model_policy.json"


def rebuild_bayesian_challenger() -> dict[str, object]:
    handler = DataHandler()
    raw = handler.get("ufc_fights_reported_doubled")
    fighters = handler.get("fighter_stats")
    auxiliary = load_approved_auxiliary(
        AUXILIARY_PATH, AUXILIARY_POLICY_PATH
    )
    builder = PointInTimeDatasetBuilder(
        raw, fighters, auxiliary_fights=auxiliary
    )
    # Replay state to the publication cutoff, then use the exact persisted
    # numeric matrix whose fingerprint is bound into winner_model.json.
    builder.build()
    point_in_time = pd.read_csv(POINT_IN_TIME_PATH, low_memory=False)
    builder.training_data = point_in_time.copy()
    winner = TemporalFightPredictor.load_artifact(
        WINNER_MODEL_PATH, builder, handler
    )
    challenger = BayesianLogisticChallenger.fit(winner)
    challenger.save_artifact(BAYESIAN_MODEL_PATH)
    challenger = BayesianLogisticChallenger.load_artifact(
        BAYESIAN_MODEL_PATH,
        builder=builder,
        base_artifact=winner.artifact(),
    )

    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    vegas = handler.get("vegas_odds", filetype="json")
    annotated = challenger.annotate_upcoming_fights(
        vegas, str(card["date"])
    )
    annotated = challenger.annotate_best_price_expected_returns(
        annotated, handler.bookies
    )
    handler.update_vegas_odds(annotated)
    artifact = challenger.artifact()
    comparison = artifact["temporal_evaluation"]["comparison_to_point_model"]
    print(
        "Bayesian challenger rebuilt: "
        f"{artifact['model_id']}; walk-forward log-loss delta "
        f"{comparison['walk_forward_log_loss_delta_vs_point']:+.6f}"
    )
    return artifact


def main() -> int:
    rebuild_bayesian_challenger()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
