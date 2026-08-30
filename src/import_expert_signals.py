#!/usr/bin/env python3
"""Import verifiable, prospective expert moneyline picks into a local ledger."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Mapping

from market_tracker import ExpertPick, ExpertPickStore, load_expert_source_registry
from market_tracker._common import MarketDataError, payload_hash


ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = ROOT / "content/data/market/expert_source_registry.json"
DEFAULT_LEDGER_DIR = Path.home() / ".ufc-data-lab" / "expert-signals"


def _read_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix.casefold() == ".jsonl":
        rows: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise MarketDataError(f"input line {number} is not an object")
                rows.append(value)
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return [dict(row) for row in csv.DictReader(source)]


def _pick_from_row(row: Mapping[str, object], observed_at: datetime) -> ExpertPick:
    values = dict(row)
    source_text = str(values.pop("source_text", ""))
    source_hash = str(values.get("source_text_sha256", "")).strip()
    if source_text:
        calculated = payload_hash(source_text)
        if source_hash and source_hash.casefold() != calculated:
            raise MarketDataError("source_text_sha256 does not match source_text")
        values["source_text_sha256"] = calculated
    elif not source_hash:
        raise MarketDataError("each input row needs source_text or source_text_sha256")
    values["observed_at_utc"] = observed_at
    return ExpertPick.create(**values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import free public expert moneyline picks before their fights. "
            "Generated ledgers are paper-only and contain no execution support."
        )
    )
    parser.add_argument("--input", type=Path, help="CSV or JSONL input file")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER_DIR)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the existing ledger and registry without importing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policies = load_expert_source_registry(args.registry)
        store = ExpertPickStore(
            args.ledger_dir / "expert_picks.csv",
            args.ledger_dir / "expert_picks.jsonl",
        )
        existing = store.read()
        if args.validate_only:
            print(
                f"Expert signal ledger valid: {len(existing)} picks; "
                f"{sum(policy.enabled for policy in policies.values())} enabled sources."
            )
            return 0
        if args.input is None:
            raise MarketDataError("--input is required unless --validate-only is used")
        observed_at = datetime.now(timezone.utc)
        picks = []
        for number, row in enumerate(_read_rows(args.input), start=1):
            try:
                pick = _pick_from_row(row, observed_at)
                policy = policies.get(pick.analyst_id)
                if policy is None:
                    raise MarketDataError(f"unknown analyst_id {pick.analyst_id!r}")
                policy.validate_url(pick.source_url)
                picks.append(pick)
            except (MarketDataError, TypeError) as error:
                raise MarketDataError(f"input row {number}: {error}") from error
        result = store.append(picks)
        print(
            f"Expert signal import: {len(result.added_ids)} added, "
            f"{len(result.duplicate_ids)} duplicates, {result.total_records} total."
        )
        print(f"Ledger: {args.ledger_dir}")
        print("Status: paper_only; execution_enabled=false")
        return 0
    except (OSError, json.JSONDecodeError, MarketDataError, RuntimeError) as error:
        print(f"expert signals: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
