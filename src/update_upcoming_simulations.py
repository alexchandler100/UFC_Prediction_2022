"""Scheduled incremental publication of upcoming-fight simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fight_sim.catalog import execute_upcoming_catalog
from fight_sim.domain import SimulatorConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate newly discovered upcoming UFC matchups and rebuild the "
            "paper-only website publication."
        )
    )
    parser.add_argument("--catalog", default="src/content/data/external/all_upcoming_forecasts.json")
    parser.add_argument(
        "--record-directory",
        default="src/content/data/simulation/upcoming_matchups",
    )
    parser.add_argument(
        "--website-output",
        default="src/content/data/external/simulation_forecasts.json",
    )
    parser.add_argument("--minimum-prior-ufc-fights", type=int, default=3)
    parser.add_argument("--bootstrap-members", type=int, default=64)
    parser.add_argument("--paths-per-member", type=int, default=64)
    parser.add_argument(
        "--simulator-config",
        default="SIMULATION_MECHANICS_BASELINE_V1.json",
        help="Retained mechanics-profile JSON used for every new matchup.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-new-matchups", type=int, default=100)
    parser.add_argument("--max-runtime-seconds", type=float, default=9000.0)
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="Rebuild the website file from existing records without running fights.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_payload = json.loads(Path(args.simulator_config).read_text(encoding="utf-8"))
    config_values = config_payload.get("simulator_config", config_payload)
    if not isinstance(config_values, dict):
        raise ValueError("simulator config must contain a simulator_config object")
    simulator_config = SimulatorConfig(**config_values)
    mechanics_profile_id = config_payload.get("mechanics_profile_id")
    destination, publication = execute_upcoming_catalog(
        catalog_path=args.catalog,
        record_directory=args.record_directory,
        website_output=args.website_output,
        minimum_prior_ufc_fights=args.minimum_prior_ufc_fights,
        bootstrap_members=args.bootstrap_members,
        paths_per_member=args.paths_per_member,
        workers=args.workers,
        chunk_size=args.chunk_size,
        max_new_matchups=args.max_new_matchups,
        max_runtime_seconds=args.max_runtime_seconds,
        simulator_config=simulator_config,
        mechanics_profile_id=mechanics_profile_id,
        publish_only=args.publish_only,
        progress=lambda message: print(message, flush=True),
    )
    print(
        json.dumps(
            {
                "website_output": str(Path(destination).resolve()),
                "events": publication["event_count"],
                "matchups": publication["matchup_count"],
                "available": publication["available_matchups"],
                "withheld": publication["excluded_matchups"],
                "pending": publication["pending_matchups"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
