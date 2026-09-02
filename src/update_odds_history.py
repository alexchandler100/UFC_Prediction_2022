"""Build or validate the compact current-card odds-history publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_tracker import QuoteSnapshotStore
from market_tracker._storage import atomic_write_text
from market_tracker.odds_history import build_odds_history, validate_odds_history


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "content" / "data"
CARD_PATH = DATA / "external" / "card_info.json"
MARKET = DATA / "market"
CSV_PATH = MARKET / "quote_snapshots.csv"
JSONL_PATH = MARKET / "quote_snapshots.jsonl"
OUTPUT_PATH = MARKET / "odds_history.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    snapshots = QuoteSnapshotStore(CSV_PATH, JSONL_PATH).read()
    if args.validate_only:
        publication = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        validate_odds_history(publication, snapshots, card)
        print(
            f"Validated odds history: {publication['matchup_count']} matchups, "
            f"{publication['quote_count']} quotes."
        )
        return 0

    publication = build_odds_history(snapshots, card)
    atomic_write_text(
        OUTPUT_PATH,
        json.dumps(publication, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(
        f"Published odds history: {publication['matchup_count']} matchups, "
        f"{publication['quote_count']} quotes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
