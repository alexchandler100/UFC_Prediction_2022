"""Command-line entry point for local, candidate-only simulation research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .research import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_FIGHTER_PROFILES,
    DEFAULT_PARAMETER_ARTIFACT,
    DEFAULT_POINT_IN_TIME,
    DEFAULT_RAW_FIGHTS,
    DEFAULT_ROUND_STATS,
    NonConvergedSimulationError,
    execute_analyze,
    execute_backfill,
    execute_backtest,
    execute_diff,
    execute_fit,
    execute_reduce,
    execute_replay,
    execute_run,
)


def _bounded_integer(lower: int, upper: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if not lower <= parsed <= upper:
            raise argparse.ArgumentTypeError(
                f"must be between {lower} and {upper}"
            )
        return parsed

    return parse


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw", default=str(DEFAULT_RAW_FIGHTS), help="Mirrored bout-total CSV")
    parser.add_argument(
        "--profiles", default=str(DEFAULT_FIGHTER_PROFILES), help="Fighter profile CSV"
    )
    parser.add_argument(
        "--round-stats",
        default=str(DEFAULT_ROUND_STATS),
        help="Optional normalized/reconciled fighter-round CSV",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fight_sim",
        description=(
            "Deterministic, candidate-only UFC simulation research. Commands do "
            "not alter production predictions or place wagers."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    backfill = commands.add_parser(
        "backfill", help="Fetch a bounded, resumable batch of UFCStats round tables"
    )
    backfill.add_argument("--max-fights", type=_bounded_integer(1, 100), default=25)
    backfill.add_argument(
        "--checkpoint-every", type=_bounded_integer(1, 25), default=5
    )
    backfill.add_argument(
        "--summary-output",
        default=str(DEFAULT_ARTIFACT_ROOT / "round-backfill-summary.json"),
    )
    backfill.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Re-fetch already stored fights (off by default)",
    )

    fit = commands.add_parser("fit", help="Fit a frozen card-bootstrap parameter ensemble")
    _input_arguments(fit)
    fit.add_argument(
        "--bootstrap-members", type=_bounded_integer(1, 200), default=200
    )
    fit.add_argument("--random-seed", type=int, default=1729)
    fit.add_argument(
        "--as-of", help="Strict UTC data cutoff; defaults to the current time"
    )
    fit.add_argument("--output", default=str(DEFAULT_PARAMETER_ARTIFACT))

    run = commands.add_parser(
        "run", help="Run an adaptive nested simulation for two stable fighter IDs"
    )
    _input_arguments(run)
    run.add_argument("--parameters", default=str(DEFAULT_PARAMETER_ARTIFACT))
    run.add_argument("--red-fighter-id", required=True)
    run.add_argument("--blue-fighter-id", required=True)
    run.add_argument("--division", required=True)
    run.add_argument("--scheduled-rounds", type=int, choices=(3, 5), default=3)
    run.add_argument("--event-id", default="local-research")
    run.add_argument("--matchup-id")
    run.add_argument("--root-seed", default="20220813")
    run.add_argument("--output-dir")
    run.add_argument(
        "--initial-paths-per-member", type=_bounded_integer(2, 8192), default=512
    )
    run.add_argument(
        "--max-paths-per-member", type=_bounded_integer(2, 8192), default=2048
    )
    run.add_argument("--workers", type=_bounded_integer(1, 64), default=1)
    run.add_argument("--chunk-size", type=_bounded_integer(1, 4096), default=64)
    run.add_argument("--max-traces", type=_bounded_integer(0, 32), default=32)
    run.add_argument("--winner-mcse-target", type=_positive_float, default=0.002)
    run.add_argument(
        "--parameter-quantile-tolerance",
        type=_nonnegative_float,
        default=0.01,
    )
    run.add_argument(
        "--allow-nonconverged-research",
        action="store_true",
        help=(
            "Write explicitly labeled nonconverged research outputs; without this "
            "flag only specs/convergence diagnostics are retained"
        ),
    )

    backtest = commands.add_parser(
        "backtest", help="Run bounded, strictly chronological fit/simulation folds"
    )
    _input_arguments(backtest)
    backtest.add_argument(
        "--bootstrap-members", type=_bounded_integer(1, 64), default=64
    )
    backtest.add_argument(
        "--paths-per-matchup", type=_bounded_integer(1, 16384), default=4096
    )
    backtest.add_argument(
        "--first-test-year", type=_bounded_integer(1993, 2100), default=2017
    )
    backtest.add_argument(
        "--last-test-year", type=_bounded_integer(1993, 2100), default=2026
    )
    backtest.add_argument("--max-fights", type=_bounded_integer(1, 500), default=200)
    backtest.add_argument(
        "--min-training-fights", type=_bounded_integer(1, 100000), default=500
    )
    backtest.add_argument("--random-seed", type=int, default=2903)
    backtest.add_argument(
        "--seed-repeats",
        type=_bounded_integer(2, 4),
        default=2,
        help="Independent inner-process repeats; repeat 1 remains authoritative",
    )
    backtest.add_argument(
        "--skip-borderline-rerun",
        action="store_true",
        help=(
            "Disable the default-repository 16,384-path rerun when a primary "
            "joint baseline interval crosses zero"
        ),
    )
    backtest.add_argument(
        "--point-in-time",
        help=(
            "Optional causal point-in-time CSV for incumbent/outcome comparators; "
            "the repository default is used with default raw data"
        ),
    )
    backtest.add_argument("--market-directory", help="Timestamped market capture directory")
    backtest.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip chronological incumbent/outcome and timestamped market comparisons",
    )
    backtest.add_argument("--workers", type=_bounded_integer(1, 64), default=1)
    backtest.add_argument("--chunk-size", type=_bounded_integer(1, 4096), default=64)
    backtest.add_argument(
        "--output", default=str(DEFAULT_ARTIFACT_ROOT / "backtest-summary.json")
    )
    backtest.add_argument(
        "--ledger-output",
        help="Optional detailed JSONL ledger; omitted by the manual workflow",
    )

    replay = commands.add_parser(
        "replay", help="Verify a stored trace or regenerate one from spec plus index"
    )
    replay_source = replay.add_mutually_exclusive_group(required=True)
    replay_source.add_argument("--trace", help="Stored trace JSON for reducer replay")
    replay_source.add_argument("--spec", help="Single or ensemble specs JSON")
    replay.add_argument("--simulation-index", type=_bounded_integer(0, 2**31 - 1))
    replay.add_argument("--bootstrap-member", type=_bounded_integer(0, 100000))
    replay.add_argument("--expected-trace", help="Optional trace to compare during stochastic replay")
    replay.add_argument("--output", help="Optional JSON replay result")

    reduce = commands.add_parser(
        "reduce", help="Reduce and hash-verify one immutable event trace"
    )
    reduce.add_argument("trace", help="Stored trace JSON")
    reduce.add_argument("--output")

    diff = commands.add_parser(
        "diff", help="Report the first event-field divergence between two traces"
    )
    diff.add_argument("expected")
    diff.add_argument("actual")
    diff.add_argument("--output")

    analyze = commands.add_parser(
        "analyze", help="Generate a self-contained local HTML analysis report"
    )
    analyze.add_argument(
        "input",
        help="Run directory, aggregate JSON, or BacktestReport JSON",
    )
    analyze.add_argument("--output")
    analyze.add_argument("--title")
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "backfill":
            result = execute_backfill(
                max_fights=args.max_fights,
                checkpoint_every=args.checkpoint_every,
                summary_output=args.summary_output,
                refresh_existing=args.refresh_existing,
            )
            _print(result)
        elif args.command == "fit":
            artifact = execute_fit(
                raw_path=args.raw,
                profiles_path=args.profiles,
                round_path=args.round_stats,
                output=args.output,
                as_of=args.as_of,
                bootstrap_members=args.bootstrap_members,
                random_seed=args.random_seed,
            )
            _print(
                {
                    "artifact_sha256": artifact.artifact_sha256,
                    "bootstrap_members": len(artifact.members),
                    "observed_fights": artifact.observed_fights,
                    "output": str(Path(args.output).resolve()),
                }
            )
        elif args.command == "run":
            destination, result = execute_run(
                parameter_path=args.parameters,
                raw_path=args.raw,
                profiles_path=args.profiles,
                round_path=args.round_stats,
                red_fighter_id=args.red_fighter_id,
                blue_fighter_id=args.blue_fighter_id,
                division=args.division,
                scheduled_rounds=args.scheduled_rounds,
                event_id=args.event_id,
                matchup_id=args.matchup_id,
                root_seed=args.root_seed,
                output_dir=args.output_dir,
                initial_paths_per_member=args.initial_paths_per_member,
                max_paths_per_member=args.max_paths_per_member,
                workers=args.workers,
                chunk_size=args.chunk_size,
                max_traces=args.max_traces,
                winner_mcse_target=args.winner_mcse_target,
                parameter_quantile_tolerance=args.parameter_quantile_tolerance,
                allow_nonconverged_research=args.allow_nonconverged_research,
            )
            _print(
                {
                    "converged": result.converged,
                    "output_dir": str(destination.resolve()),
                    "total_paths": result.forecast.total_paths,
                }
            )
        elif args.command == "backtest":
            if args.first_test_year > args.last_test_year:
                raise ValueError("first-test-year must not exceed last-test-year")
            ledger, report = execute_backtest(
                raw_path=args.raw,
                profiles_path=args.profiles,
                round_path=args.round_stats,
                output=args.output,
                ledger_output=args.ledger_output,
                bootstrap_members=args.bootstrap_members,
                paths_per_matchup=args.paths_per_matchup,
                first_test_year=args.first_test_year,
                last_test_year=args.last_test_year,
                max_fights=args.max_fights,
                min_training_fights=args.min_training_fights,
                random_seed=args.random_seed,
                workers=args.workers,
                chunk_size=args.chunk_size,
                point_in_time_path=args.point_in_time,
                market_directory=args.market_directory,
                include_baselines=not args.skip_baselines,
                seed_repeats=args.seed_repeats,
                skip_borderline_rerun=args.skip_borderline_rerun,
            )
            _print(
                {
                    "report_sha256": report.report_sha256,
                    "scored_fights": len(ledger),
                    "output": str(Path(args.output).resolve()),
                }
            )
        elif args.command == "replay":
            result = execute_replay(
                trace_path=args.trace,
                spec_path=args.spec,
                simulation_index=args.simulation_index,
                bootstrap_member=args.bootstrap_member,
                expected_trace_path=args.expected_trace,
                output=args.output,
            )
            _print({key: value for key, value in result.items() if key != "trace"})
        elif args.command == "reduce":
            _print(execute_reduce(args.trace, output=args.output))
        elif args.command == "diff":
            result, differs = execute_diff(
                args.expected, args.actual, output=args.output
            )
            _print(result)
            return 1 if differs else 0
        elif args.command == "analyze":
            output = execute_analyze(args.input, output=args.output, title=args.title)
            _print({"output": str(output)})
        else:  # pragma: no cover - argparse enforces the command set
            parser.error(f"unsupported command: {args.command}")
    except NonConvergedSimulationError as error:
        print(str(error), file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as error:
        print(f"fight_sim: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
