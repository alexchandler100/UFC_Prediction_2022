"""Chronologically evaluate the candidate winner/method/finish-time model."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from fight_predictor import evaluate_outcome_model


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "content" / "data" / "processed" / "ufc_fights_point_in_time.csv"
OUTPUT_PATH = ROOT / "content" / "data" / "external" / "outcome_model_evaluation.json"


def main() -> int:
    frame = pd.read_csv(INPUT_PATH, low_memory=False)
    feature_columns = tuple(column for column in frame.columns if column.endswith("_diff"))
    _model, report = evaluate_outcome_model(frame, feature_columns)
    report["training_input_sha256"] = sha256(INPUT_PATH.read_bytes()).hexdigest()
    report["feature_count"] = len(feature_columns)
    encoded = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    OUTPUT_PATH.write_text(encoded, encoding="utf-8", newline="")
    print(
        "Candidate outcome holdout: "
        f"winner log loss={report['winner']['log_loss']:.4f}, "
        f"method log loss={report['method']['log_loss']:.4f}, "
        f"joint log loss={report['joint_outcome']['log_loss']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
