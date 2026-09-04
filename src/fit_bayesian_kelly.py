"""Fit the paper-only moneyline uncertainty model used for robust Kelly sizing."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from market_tracker.bayesian_kelly import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_REPLAY_PATH,
    fit_market_calibration,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_REPLAY_PATH))
    parser.add_argument("--output", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--created-at-utc")
    args = parser.parse_args(argv)
    source = Path(args.input)
    artifact = fit_market_calibration(
        pd.read_csv(source, low_memory=False),
        source_sha256=sha256(source.read_bytes()).hexdigest(),
        created_at_utc=args.created_at_utc,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    check = artifact["chronological_check"]
    print(
        f"Bayesian Kelly calibration: {artifact['training_fights']} fights / "
        f"{artifact['training_events']} events; later {check['holdout_fights']}-fight "
        f"log-loss change {check['log_loss_change']:+.6f}; {output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
