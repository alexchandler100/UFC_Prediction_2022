"""Resolve an upcoming matchup by fighter names and launch a small local run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLICATION = REPO_ROOT / "src/content/data/external/simulation_forecasts.json"
DEFAULT_OUTPUT_ROOT = Path.home() / ".ufc-data-lab" / "quick-simulations"


def _normalize(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _name_matches(query: str, name: object) -> bool:
    wanted = _normalize(query)
    available = _normalize(name)
    return bool(wanted) and (wanted == available or wanted in available)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _matchup(
    publication: Mapping[str, Any], first: str, second: str
) -> tuple[dict[str, Any], bool]:
    candidates: list[tuple[dict[str, Any], bool]] = []
    for value in publication.get("matchups", []):
        row = dict(value)
        forward = _name_matches(first, row.get("fighter_name")) and _name_matches(
            second, row.get("opponent_name")
        )
        reverse = _name_matches(first, row.get("opponent_name")) and _name_matches(
            second, row.get("fighter_name")
        )
        if forward:
            candidates.append((row, False))
        if reverse:
            candidates.append((row, True))
    if not candidates:
        raise ValueError(
            f"no current upcoming fight matches {first!r} vs {second!r}; "
            "run with --list to see accepted names"
        )
    unique = {
        (str(row.get("matchup_id")), reversed_order): (row, reversed_order)
        for row, reversed_order in candidates
    }
    if len(unique) != 1:
        raise ValueError(
            "those names match more than one upcoming fight; use the full fighter names"
        )
    return next(iter(unique.values()))


def _artifact_metadata(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    metadata = payload.get("logical_metadata", payload)
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid parameter artifact metadata: {path}")
    return metadata


def _find_parameters(publication: Mapping[str, Any], explicit: Path | None) -> tuple[Path, int]:
    expected_sha = str(publication.get("parameter_artifact_sha256") or "")
    materialized = (
        REPO_ROOT
        / "artifacts/simulations/parameter-materialized-cache"
        / f"parameter-{expected_sha}.json.gz"
    )
    if explicit is not None:
        candidates = [explicit]
    elif expected_sha and materialized.is_file():
        # The smaller publication artifact is a reproducible fit recipe: loading
        # it refits all 200 members. Prefer the larger exact-member cache so a
        # quick run only decodes parameters and never performs a hidden refit.
        members = int(publication.get("bootstrap_members") or 0)
        if members > 0:
            return materialized.resolve(), members
        candidates = [materialized]
    else:
        candidates = sorted(
            (
                path
                for path in (REPO_ROOT / "artifacts/simulations").rglob("*.json.gz")
                if "parameter" in path.name.casefold()
            ),
            key=lambda path: path.stat().st_size,
        )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            metadata = _artifact_metadata(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        artifact_sha = str(metadata.get("artifact_sha256") or "")
        if explicit is None and expected_sha and artifact_sha != expected_sha:
            continue
        config = dict(metadata.get("config") or {})
        members = int(config.get("bootstrap_members") or 0)
        if members > 0:
            return path.resolve(), members
    if explicit is not None:
        raise ValueError(f"could not read a parameter ensemble from {explicit}")
    raise ValueError(
        "the current upcoming-card parameter artifact is not available locally; "
        "run the regular updater first"
    )


def _largest_divisor_at_most(value: int, maximum: int) -> int:
    return max(
        candidate
        for candidate in range(1, min(value, maximum) + 1)
        if value % candidate == 0
    )


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return result or "fight"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an exact, low-path Monte Carlo simulation for a fight on the "
            "current upcoming card and open the local GUI."
        )
    )
    parser.add_argument("fighter_one", nargs="?")
    parser.add_argument("fighter_two", nargs="?")
    parser.add_argument("--paths", type=int, default=100, help="Exact total paths (default: 100)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--traces", type=int, default=32)
    parser.add_argument("--seed", default="quick-gui-v1")
    parser.add_argument("--publication", type=Path, default=DEFAULT_PUBLICATION)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--list", action="store_true", help="List the current upcoming fights")
    parser.add_argument("--no-gui", action="store_true", help="Run without opening the GUI")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print without simulating")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    publication = _load_json(args.publication.resolve())
    matchups = [dict(value) for value in publication.get("matchups", [])]
    if args.list:
        print(f"{publication.get('event_title', 'Upcoming card')} ({publication.get('event_date', '')})")
        for row in sorted(matchups, key=lambda item: int(item.get("bout_order", 999))):
            history = min(
                int(row.get("fighter_prior_ufc_fights") or 0),
                int(row.get("opponent_prior_ufc_fights") or 0),
            )
            note = "" if history >= 3 else " [not enough UFC history]"
            print(f"  {row.get('fighter_name')} vs {row.get('opponent_name')}{note}")
        return 0
    if not args.fighter_one or not args.fighter_two:
        raise SystemExit("provide two fighter names, or use --list")
    if args.paths < 2:
        raise SystemExit("--paths must be at least 2")
    if not 1 <= args.workers <= 64:
        raise SystemExit("--workers must be between 1 and 64")
    if not 0 <= args.traces <= 32:
        raise SystemExit("--traces must be between 0 and 32")

    row, reversed_order = _matchup(publication, args.fighter_one, args.fighter_two)
    minimum_history = min(
        int(row.get("fighter_prior_ufc_fights") or 0),
        int(row.get("opponent_prior_ufc_fights") or 0),
    )
    if minimum_history < 3:
        raise SystemExit(
            "this fight is intentionally excluded because at least one fighter "
            "has fewer than three previous UFC fights"
        )
    parameter_path, available_members = _find_parameters(publication, args.parameters)
    # Two paths per member are the minimum needed for the simulator's
    # deterministic odd/even convergence comparison. Keep the requested total
    # exact while using as much of the parameter ensemble as that permits.
    selected_members = _largest_divisor_at_most(
        args.paths, min(available_members, args.paths // 2)
    )
    paths_per_member = args.paths // selected_members
    if reversed_order:
        red_id, blue_id = str(row["opponent_id"]), str(row["fighter_id"])
        red_name, blue_name = str(row["opponent_name"]), str(row["fighter_name"])
    else:
        red_id, blue_id = str(row["fighter_id"]), str(row["opponent_id"])
        red_name, blue_name = str(row["fighter_name"]), str(row["opponent_name"])

    if args.output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            DEFAULT_OUTPUT_ROOT
            / _slug(str(publication.get("event_date") or "upcoming"))
            / f"{_slug(red_name)}-vs-{_slug(blue_name)}-{args.paths}paths-{timestamp}"
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()

    command = [
        sys.executable,
        "-m",
        "fight_sim",
        "run",
        "--parameters",
        str(parameter_path),
        "--simulator-config",
        str(REPO_ROOT / "SIMULATION_MECHANICS_BASELINE_V1.json"),
        "--red-fighter-id",
        red_id,
        "--blue-fighter-id",
        blue_id,
        "--division",
        str(row["division"]),
        "--scheduled-rounds",
        str(int(row["scheduled_rounds"])),
        "--event-id",
        str(publication.get("event_id") or "local-upcoming"),
        "--matchup-id",
        str(row["matchup_id"]),
        "--root-seed",
        str(args.seed),
        "--bootstrap-members",
        str(selected_members),
        "--initial-paths-per-member",
        str(paths_per_member),
        "--max-paths-per-member",
        str(paths_per_member),
        "--workers",
        str(args.workers),
        "--chunk-size",
        str(min(64, max(1, paths_per_member))),
        "--max-traces",
        str(min(args.traces, args.paths)),
        "--allow-nonconverged-research",
        "--output-dir",
        str(output_dir),
    ]
    if not args.no_gui:
        command.append("--launch-gui")

    print(f"Fight: {red_name} vs {blue_name}", flush=True)
    print(
        f"Paths: {args.paths} total = {selected_members} parameter replicas "
        f"x {paths_per_member} path(s) each",
        flush=True,
    )
    if args.paths < 1000:
        print(
            "Note: this is a quick visual sample; use --paths 2000 or more for "
            "steadier probability estimates.",
            flush=True,
        )
    print(f"Output: {output_dir}", flush=True)
    if args.dry_run:
        print("Dry run only; no simulation was started.")
        return 0
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
