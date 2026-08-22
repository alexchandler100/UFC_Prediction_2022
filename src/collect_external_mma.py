"""Collect, validate, crosswalk, and prepare licensed external MMA history.

Examples:
  python src/collect_external_mma.py sources
  python src/collect_external_mma.py import-kaggle path/to/pro_mma_fights.csv
  python src/collect_external_mma.py crosswalk
  python src/collect_external_mma.py build-auxiliary
  python src/collect_external_mma.py validate

The collector deliberately has no generic web scraper. Only sources whose
registry entry permits manual/licensed import can write the canonical ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from data_handler.data_handler import atomic_to_csv
from external_mma import (
    CanonicalCsvAdapter,
    ExternalMmaStore,
    KaggleProMmaAdapter,
    build_auxiliary_doubled,
    load_identity_map,
    propose_ufcstats_crosswalk,
)
from external_mma.identity import IDENTITY_COLUMNS, merge_identity_maps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "src" / "content" / "data" / "external_mma"
DEFAULT_RAW_UFC = (
    REPO_ROOT / "src" / "content" / "data" / "processed"
    / "ufc_fights_reported_doubled.csv"
)
DEFAULT_AUXILIARY = (
    REPO_ROOT / "src" / "content" / "data" / "processed"
    / "external_mma_auxiliary_doubled.csv"
)


def _read_identity_frame(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=IDENTITY_COLUMNS)
    return pd.read_csv(path, dtype=object, keep_default_na=False)


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="external-MMA ledger root",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sources", help="show source permissions and readiness")

    kaggle = commands.add_parser(
        "import-kaggle", help="import the CC0 Kaggle version-1 CSV"
    )
    kaggle.add_argument("csv", type=Path)
    kaggle.add_argument(
        "--store-raw", action="store_true",
        help="retain the raw snapshot locally (raw files are gitignored)",
    )

    canonical = commands.add_parser(
        "import-canonical", help="import an authorized canonical provider CSV"
    )
    canonical.add_argument("source_key")
    canonical.add_argument("csv", type=Path)
    canonical.add_argument(
        "--license-confirmed", action="store_true",
        help="confirm that this repository may process the provider export",
    )
    canonical.add_argument("--store-raw", action="store_true")

    crosswalk = commands.add_parser(
        "crosswalk", help="derive strong source-to-UFCStats identity mappings"
    )
    crosswalk.add_argument("--raw-ufc", type=Path, default=DEFAULT_RAW_UFC)

    build = commands.add_parser(
        "build-auxiliary", help="write state-only doubled history for model replay"
    )
    build.add_argument("--output", type=Path, default=DEFAULT_AUXILIARY)

    validate = commands.add_parser(
        "validate", help="validate ledgers, hashes, and auxiliary rows"
    )
    validate.add_argument("--auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    args = parser.parse_args(argv)
    store = ExternalMmaStore(args.data_root.resolve())

    if args.command == "sources":
        _print_json(store.source_registry())
        return 0
    if args.command == "import-kaggle":
        content = args.csv.resolve().read_bytes()
        report = store.import_bytes(
            KaggleProMmaAdapter(), content, store_raw=args.store_raw,
            original_filename=args.csv.name,
        )
        _print_json(report)
        return 0
    if args.command == "import-canonical":
        if not args.license_confirmed:
            parser.error("--license-confirmed is required for a provider export")
        content = args.csv.resolve().read_bytes()
        report = store.import_bytes(
            CanonicalCsvAdapter(args.source_key), content, store_raw=args.store_raw,
            original_filename=args.csv.name,
        )
        _print_json(report)
        return 0
    if args.command == "crosswalk":
        observations = store.observations()
        raw = pd.read_csv(args.raw_ufc.resolve(), low_memory=False)
        proposed, review = propose_ufcstats_crosswalk(observations, raw)
        identity_path = store.root / "identity_map.csv"
        existing = _read_identity_frame(identity_path)
        merged = merge_identity_maps(existing, proposed)
        atomic_to_csv(merged, identity_path, index=False)
        review_path = store.root / "identity_review_queue.csv"
        atomic_to_csv(review, review_path, index=False)
        _print_json(
            {
                "approved_mappings": len(merged),
                "new_proposals": len(proposed),
                "review_required": len(review),
            }
        )
        return 0
    if args.command == "build-auxiliary":
        identity_path = store.root / "identity_map.csv"
        identities = load_identity_map(identity_path)
        auxiliary = build_auxiliary_doubled(store.observations(), identities)
        atomic_to_csv(auxiliary, args.output.resolve(), index=False)
        _print_json(
            {
                "physical_bouts": int(auxiliary["fight_url"].nunique()) if not auxiliary.empty else 0,
                "doubled_rows": len(auxiliary),
                "mapped_ufcstats_fighters": len(set(identities.values())),
                "output": str(args.output.resolve()),
            }
        )
        return 0
    if args.command == "validate":
        report = store.validate()
        auxiliary_path = args.auxiliary.resolve()
        if auxiliary_path.exists() and auxiliary_path.stat().st_size:
            auxiliary = pd.read_csv(auxiliary_path, low_memory=False)
            report["auxiliary_rows"] = len(auxiliary)
            report["auxiliary_bouts"] = int(auxiliary["fight_url"].nunique())
            if not auxiliary["emit_training_target"].astype(str).str.casefold().isin(
                {"false", "0"}
            ).all():
                raise ValueError("auxiliary rows must never emit training targets")
        _print_json(report)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"external MMA collection failed: {error}", file=sys.stderr)
        raise
