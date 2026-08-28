"""Command-line entry point for local, candidate-only simulation research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .domain import SimulatorConfig
from .parameters import SNAPSHOT_PARAMETER_MODES
from .opponent_audit import (
    OpponentAdjustmentAuditConfig,
    execute_opponent_adjustment_audit,
)
from .research import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_FIGHTER_PROFILES,
    DEFAULT_PARAMETER_ARTIFACT,
    DEFAULT_POINT_IN_TIME,
    DEFAULT_POSTERIOR_FIT_CACHE,
    DEFAULT_RAW_FIGHTS,
    DEFAULT_ROUND_STATS,
    NonConvergedSimulationError,
    execute_analyze,
    execute_backfill,
    execute_backtest,
    execute_diff,
    execute_fit,
    execute_posterior_backtest,
    execute_reduce,
    execute_replay,
    execute_run,
)
from .posterior_predictive import validate_completed_fight, write_validation_report
from .performance import execute_benchmark
from .tuning import (
    compare_outcome_mechanics,
    derive_mechanics_profile,
    select_finish_profile,
    select_mechanics_profile,
    validate_finish_profile,
    validate_knockdown_observation_profile,
    validate_mechanics_holdout,
)
from .transition_audit import TransitionAuditConfig, execute_transition_audit
from .upcoming import DEFAULT_PARAMETER_CACHE, execute_upcoming_card


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


def _worker_counts(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be comma-separated integers") from error
    if not parsed or any(item < 1 or item > 64 for item in parsed):
        raise argparse.ArgumentTypeError("worker counts must be between 1 and 64")
    return parsed


def _positive_float_tuple(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be comma-separated numbers"
        ) from error
    if not parsed or any(not item > 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive")
    if tuple(sorted(set(parsed))) != parsed:
        raise argparse.ArgumentTypeError(
            "values must be unique and strictly increasing"
        )
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


def _load_simulator_config(path: str | None) -> SimulatorConfig | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("simulator config must contain a JSON object")
    values = payload.get("simulator_config", payload)
    if not isinstance(values, dict):
        raise ValueError("simulator_config must be a JSON object")
    return SimulatorConfig(**values)


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
    backfill.add_argument("--max-fights", type=_bounded_integer(1, 1000), default=25)
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
    backfill.add_argument(
        "--max-runtime-seconds",
        type=_positive_float,
        default=3000.0,
        help="Checkpoint and stop before this compute budget (maximum 3300 seconds)",
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
    fit.add_argument(
        "--takedown-control-association",
        action="store_true",
        help="Fit the research-only conditional TD/CTRL retention candidate",
    )

    run = commands.add_parser(
        "run", help="Run an adaptive nested simulation for two stable fighter IDs"
    )
    _input_arguments(run)
    run.add_argument("--parameters", default=str(DEFAULT_PARAMETER_ARTIFACT))
    run.add_argument(
        "--simulator-config",
        help="Optional JSON file containing global research mechanics multipliers",
    )
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
    run.add_argument(
        "--launch-gui",
        action="store_true",
        help="Open the optional local desktop explorer after the run completes",
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
    backtest.add_argument(
        "--stack-min-training-fights",
        type=_bounded_integer(1, 100000),
        default=100,
        help="Earlier out-of-fold fights required before evaluating the winner stack",
    )
    backtest.add_argument(
        "--stack-l2-penalty",
        type=_nonnegative_float,
        default=0.01,
        help="Regularization toward incumbent-only stack weights (1, 0)",
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

    posterior = commands.add_parser(
        "posterior-backtest",
        help="Run causal posterior-predictive checks on recent complete events",
    )
    _input_arguments(posterior)
    posterior.add_argument("--last-events", type=_bounded_integer(1, 100), default=20)
    posterior.add_argument(
        "--skip-latest-events",
        type=_bounded_integer(0, 99),
        default=0,
        help="Reserve this many newest complete events after the selected window",
    )
    posterior.add_argument(
        "--cohort-manifest",
        help="Tracked frozen-cohort JSON; requires --cohort-name",
    )
    posterior.add_argument(
        "--cohort-name",
        help="Named cohort from --cohort-manifest",
    )
    posterior.add_argument(
        "--min-prior-ufc-fights", type=_bounded_integer(0, 100), default=3
    )
    posterior.add_argument(
        "--bootstrap-members", type=_bounded_integer(1, 64), default=64
    )
    posterior.add_argument(
        "--paths-per-matchup", type=_bounded_integer(1, 16384), default=4096
    )
    posterior.add_argument("--seed-repeats", type=_bounded_integer(1, 4), default=2)
    posterior.add_argument(
        "--min-training-fights", type=_bounded_integer(1, 100000), default=500
    )
    posterior.add_argument("--random-seed", type=int, default=2903)
    posterior.add_argument(
        "--simulator-config",
        help="Optional JSON file containing global research mechanics multipliers",
    )
    posterior.add_argument("--workers", type=_bounded_integer(1, 64), default=1)
    posterior.add_argument("--chunk-size", type=_bounded_integer(1, 4096), default=64)
    fidelity = posterior.add_mutually_exclusive_group()
    fidelity.add_argument(
        "--quick-screen",
        action="store_true",
        help=(
            "Use 5 events, 16 members, 512 total paths/matchup, and one seed; "
            "screening output cannot serve as final validation or promotion evidence"
        ),
    )
    fidelity.add_argument(
        "--confirmation-screen",
        action="store_true",
        help=(
            "Use 15 events, 32 members, 2,048 total paths/matchup, and one seed "
            "for survivors from the quick screen"
        ),
    )
    posterior.add_argument(
        "--resume",
        action="store_true",
        help="Resume atomically checkpointed fight/seed pairs in the output directory",
    )
    posterior.add_argument(
        "--fit-cache-dir",
        default=str(DEFAULT_POSTERIOR_FIT_CACHE),
        help="Shared ignored cache for materialized causal event-cutoff fits",
    )
    posterior.add_argument(
        "--no-fit-cache",
        action="store_true",
        help="Disable reuse of materialized causal event-cutoff fits",
    )
    posterior.add_argument(
        "--takedown-control-association",
        action="store_true",
        help=(
            "Research-only fit using strongly pooled same-round TD/CTRL "
            "associations for ground retention and escape"
        ),
    )
    posterior.add_argument(
        "--snapshot-parameter-mode",
        choices=tuple(sorted(SNAPSHOT_PARAMETER_MODES)),
        default="full",
        help=(
            "Research policy for fighter deviations: full, division/era "
            "context only, rejected causal opponent-adjusted v1, cross-fitted "
            "equal-bout opponent-adjusted v2 strikes, or a second causal "
            "exposure-weighted shrinkage step"
        ),
    )
    posterior.add_argument(
        "--max-runtime-seconds",
        type=_positive_float,
        default=3300.0,
        help="Checkpoint complete fights and stop (maximum 3300 seconds)",
    )
    posterior.add_argument(
        "--output-dir",
        default=str(DEFAULT_ARTIFACT_ROOT / "posterior-backtest-recent"),
    )

    compare_mechanics = commands.add_parser(
        "compare-outcome-mechanics",
        help="Compare outcome engines on identical completed event cards",
    )
    compare_mechanics.add_argument("baseline", help="Baseline posterior run")
    compare_mechanics.add_argument("candidate", help="Candidate posterior run")
    compare_mechanics.add_argument(
        "--minimum-balanced-events", type=_bounded_integer(1, 100), default=5
    )
    compare_mechanics.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACT_ROOT / "outcome-mechanics-comparison.json"),
    )

    upcoming = commands.add_parser(
        "upcoming-card",
        help=(
            "Fit once and precompute candidate-only distributions for the current "
            "website card"
        ),
    )
    _input_arguments(upcoming)
    upcoming.add_argument(
        "--card",
        default="src/content/data/external/card_info.json",
        help="Current card metadata JSON",
    )
    upcoming.add_argument(
        "--outcomes",
        default="src/content/data/external/outcome_forecasts.json",
        help="Current outcome-forecast card/matchup JSON",
    )
    upcoming.add_argument(
        "--simulator-config",
        help="Validated mechanics-profile JSON selected from held-out evaluation",
    )
    upcoming.add_argument(
        "--parameter-artifact",
        help="Reuse a validated same-card pre-event artifact when no fights were added",
    )
    upcoming.add_argument(
        "--parameter-cache-dir",
        default=str(DEFAULT_PARAMETER_CACHE),
        help=(
            "Ignored content-addressed materialized cache; defaults beneath "
            "artifacts/simulations"
        ),
    )
    upcoming.add_argument(
        "--minimum-prior-ufc-fights", type=_bounded_integer(0, 100), default=3
    )
    upcoming.add_argument(
        "--bootstrap-members", type=_bounded_integer(1, 200), default=200
    )
    upcoming.add_argument(
        "--initial-paths-per-member", type=_bounded_integer(2, 8192), default=512
    )
    upcoming.add_argument(
        "--max-paths-per-member", type=_bounded_integer(2, 8192), default=2048
    )
    upcoming.add_argument("--random-seed", type=int, default=81173)
    upcoming.add_argument("--workers", type=_bounded_integer(1, 64), default=1)
    upcoming.add_argument("--chunk-size", type=_bounded_integer(1, 4096), default=64)
    upcoming.add_argument(
        "--output-dir",
        default=str(DEFAULT_ARTIFACT_ROOT / "upcoming-card"),
    )
    upcoming.add_argument(
        "--website-output",
        default="src/content/data/external/simulation_forecasts.json",
    )
    upcoming.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume validated completed matchups and the next exact balanced "
            "adaptive batch from the output directory"
        ),
    )

    tune = commands.add_parser(
        "derive-mechanics",
        help=(
            "Derive one conservative global mechanics profile from development "
            "events while reserving the newest events as holdout"
        ),
    )
    tune.add_argument(
        "population_run",
        help="Posterior-backtest directory or its forecast-ledger.jsonl.gz",
    )
    tune.add_argument(
        "--holdout-latest-events", type=_bounded_integer(1, 99), default=5
    )
    tune.add_argument(
        "--prior-strength-events", type=_nonnegative_float, default=20.0
    )
    tune.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACT_ROOT / "mechanics-profile.json"),
    )

    select_tune = commands.add_parser(
        "select-mechanics",
        help="Select predeclared mechanics candidates on an intermediate event window",
    )
    select_tune.add_argument(
        "baseline_population_run",
        help="20-event neutral posterior-backtest directory or ledger",
    )
    select_tune.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=REPORT",
        help="Candidate label and population-summary path; repeat for each profile",
    )
    select_tune.add_argument(
        "--selection-events", type=_bounded_integer(1, 99), default=5
    )
    select_tune.add_argument(
        "--skip-latest-events", type=_bounded_integer(0, 99), default=5
    )
    select_tune.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACT_ROOT / "selected-mechanics-profile.json"),
    )

    select_finish = commands.add_parser(
        "select-finishing",
        help="Select finish-conversion candidates on one intermediate cohort",
    )
    select_finish.add_argument("baseline_population_run")
    select_finish.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=REPORT",
        help="Candidate label and population-summary path; repeat for each profile",
    )
    select_finish.add_argument(
        "--objective",
        choices=("duration", "joint"),
        default="duration",
        help="Selection objective; joint preserves the predeclared simulator primary metric",
    )
    select_finish.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACT_ROOT / "selected-finish-profile.json"),
    )

    validate_tune = commands.add_parser(
        "validate-mechanics",
        help="Retain or reject selected mechanics on the untouched newest events",
    )
    validate_tune.add_argument("baseline_population_run")
    validate_tune.add_argument("tuned_population_run")
    validate_tune.add_argument(
        "--holdout-latest-events", type=_bounded_integer(1, 99), default=5
    )
    validate_tune.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACT_ROOT / "validated-mechanics-profile.json"),
    )

    validate_finish = commands.add_parser(
        "validate-finishing",
        help="Retain or reject finish conversion on its untouched holdout",
    )
    validate_finish.add_argument("baseline_holdout_run")
    validate_finish.add_argument("candidate_holdout_run")
    validate_finish.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACT_ROOT / "validated-finish-profile.json"),
    )

    validate_knockdown_observation = commands.add_parser(
        "validate-knockdown-observation",
        help="Validate official-knockdown observation thinning on a locked cohort",
    )
    validate_knockdown_observation.add_argument("baseline_holdout_run")
    validate_knockdown_observation.add_argument("candidate_holdout_run")
    validate_knockdown_observation.add_argument(
        "--output",
        default=str(
            DEFAULT_ARTIFACT_ROOT / "validated-knockdown-observation-profile.json"
        ),
    )

    transitions = commands.add_parser(
        "transition-audit",
        help=(
            "Test strongly pooled fighter-specific KD/TD same-round associations "
            "on a locked chronological holdout"
        ),
    )
    transitions.add_argument("--round-stats", default=str(DEFAULT_ROUND_STATS))
    transitions.add_argument(
        "--holdout-latest-events", type=_bounded_integer(1, 99), default=5
    )
    transitions.add_argument(
        "--context-prior-opportunities", type=_positive_float, default=25.0
    )
    transitions.add_argument(
        "--fighter-prior-opportunities", type=_positive_float, default=12.0
    )
    transitions.add_argument(
        "--bootstrap-replicates", type=_bounded_integer(100, 10000), default=2000
    )
    transitions.add_argument("--random-seed", type=int, default=41041)
    transitions.add_argument(
        "--max-runtime-seconds",
        type=_positive_float,
        default=3000.0,
        help="Hard audit budget (maximum 3300 seconds)",
    )
    transitions.add_argument(
        "--as-of", help="Strict UTC source cutoff; rows on/after it are excluded"
    )
    transitions.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACT_ROOT / "transition-audit.json"),
    )
    transitions.add_argument(
        "--predictions-output",
        default=str(DEFAULT_ARTIFACT_ROOT / "transition-audit-predictions.csv"),
    )

    opponent_audit = commands.add_parser(
        "opponent-adjustment-audit",
        help=(
            "Cross-fit bout-clustered opponent effects on directly observed "
            "next-card statistics before another simulator screen"
        ),
    )
    _input_arguments(opponent_audit)
    opponent_audit.add_argument(
        "--cohort-manifest",
        default=str(Path(__file__).resolve().parents[2] / "SIMULATION_EXPERIMENT_COHORTS_V1.json"),
    )
    opponent_audit.add_argument("--cohort-name", default="development_2024")
    opponent_audit.add_argument(
        "--min-prior-ufc-fights", type=_bounded_integer(1, 99), default=3
    )
    opponent_audit.add_argument(
        "--inner-validation-events", type=_bounded_integer(3, 30), default=8
    )
    opponent_audit.add_argument(
        "--minimum-training-fights", type=_bounded_integer(1, 100000), default=500
    )
    opponent_audit.add_argument(
        "--ridge-grid", type=_positive_float_tuple, default=(5.0, 10.0, 20.0, 40.0)
    )
    opponent_audit.add_argument(
        "--bootstrap-replicates", type=_bounded_integer(100, 10000), default=2000
    )
    opponent_audit.add_argument("--random-seed", type=int, default=52237)
    opponent_audit.add_argument(
        "--max-runtime-seconds",
        type=_positive_float,
        default=3300.0,
        help="Hard audit budget (maximum 3300 seconds)",
    )
    opponent_audit.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACT_ROOT / "opponent-adjustment-audit.json"),
    )
    opponent_audit.add_argument(
        "--predictions-output",
        default=str(
            DEFAULT_ARTIFACT_ROOT / "opponent-adjustment-audit-predictions.csv"
        ),
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

    benchmark = commands.add_parser(
        "benchmark",
        help="Measure deterministic bulk-simulation throughput across worker counts",
    )
    benchmark.add_argument("specs", help="Run specs.json or a single run-spec JSON")
    benchmark.add_argument(
        "--paths-per-member", type=_bounded_integer(1, 65536), default=128
    )
    benchmark.add_argument("--workers", type=_worker_counts, default=(1, 2, 4))
    benchmark.add_argument("--chunk-size", type=_bounded_integer(1, 4096), default=64)
    benchmark.add_argument("--repeats", type=_bounded_integer(1, 10), default=1)
    benchmark.add_argument("--output")

    validate_fight = commands.add_parser(
        "validate-fight",
        help="Compare one completed run with observed UFCStats bout totals",
    )
    validate_fight.add_argument("run", help="Completed simulation run directory")
    validate_fight.add_argument("--fight-id", required=True)
    validate_fight.add_argument(
        "--observed",
        default=str(DEFAULT_RAW_FIGHTS),
        help="Mirrored observed bout-total CSV",
    )
    validate_fight.add_argument("--json-output")
    validate_fight.add_argument("--html-output")

    gui = commands.add_parser(
        "gui", help="Open the optional local desktop explorer for a completed run"
    )
    gui.add_argument("run", help="Completed simulation run directory")
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "backfill":
            if args.max_runtime_seconds > 3300:
                raise ValueError("max-runtime-seconds must not exceed 3300")
            result = execute_backfill(
                max_fights=args.max_fights,
                checkpoint_every=args.checkpoint_every,
                summary_output=args.summary_output,
                refresh_existing=args.refresh_existing,
                max_runtime_seconds=args.max_runtime_seconds,
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
                use_takedown_control_association=(
                    args.takedown_control_association
                ),
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
                simulator_config=_load_simulator_config(args.simulator_config),
            )
            _print(
                {
                    "converged": result.converged,
                    "output_dir": str(destination.resolve()),
                    "total_paths": result.forecast.total_paths,
                }
            )
            if args.launch_gui:
                from .gui import launch_gui

                return launch_gui(destination)
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
                stack_min_training_fights=args.stack_min_training_fights,
                stack_l2_penalty=args.stack_l2_penalty,
            )
            _print(
                {
                    "report_sha256": report.report_sha256,
                    "scored_fights": len(ledger),
                    "winner_stack_status": dict(
                        report.comparisons.get("production_simulation_stack") or {}
                    ).get("status", "unavailable"),
                    "winner_stack_candidate_freeze_recommended": dict(
                        report.comparisons.get("production_simulation_stack") or {}
                    ).get("candidate_freeze_recommended", False),
                    "output": str(Path(args.output).resolve()),
                }
            )
        elif args.command == "posterior-backtest":
            if args.max_runtime_seconds > 3300:
                raise ValueError("max-runtime-seconds must not exceed 3300")
            if args.quick_screen:
                last_events = 5
                bootstrap_members = 16
                paths_per_matchup = 512
                seed_repeats = 1
                fidelity = "screen"
            elif args.confirmation_screen:
                last_events = 15
                bootstrap_members = 32
                paths_per_matchup = 2048
                seed_repeats = 1
                fidelity = "confirm"
            else:
                last_events = args.last_events
                bootstrap_members = args.bootstrap_members
                paths_per_matchup = args.paths_per_matchup
                seed_repeats = args.seed_repeats
                fidelity = (
                    "final"
                    if (
                        bootstrap_members == 64
                        and paths_per_matchup == 4096
                        and seed_repeats >= 2
                    )
                    else "custom"
                )
            destination, report = execute_posterior_backtest(
                raw_path=args.raw,
                profiles_path=args.profiles,
                round_path=args.round_stats,
                output_dir=args.output_dir,
                last_events=last_events,
                skip_latest_events=args.skip_latest_events,
                min_prior_ufc_fights=args.min_prior_ufc_fights,
                bootstrap_members=bootstrap_members,
                paths_per_matchup=paths_per_matchup,
                seed_repeats=seed_repeats,
                min_training_fights=args.min_training_fights,
                random_seed=args.random_seed,
                workers=args.workers,
                chunk_size=args.chunk_size,
                resume=args.resume,
                fit_cache_dir=(None if args.no_fit_cache else args.fit_cache_dir),
                fidelity=fidelity,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
                simulator_config=_load_simulator_config(args.simulator_config),
                use_takedown_control_association=(
                    args.takedown_control_association
                ),
                max_runtime_seconds=args.max_runtime_seconds,
                cohort_manifest_path=args.cohort_manifest,
                cohort_name=args.cohort_name,
                snapshot_parameter_mode=args.snapshot_parameter_mode,
            )
            _print(
                {
                    "eligible_fights": report["selection"]["eligible_fights"],
                    "completed_fights": report["selection"]["completed_fights"],
                    "elapsed_seconds": report["runtime"]["elapsed_seconds"],
                    "output_dir": str(destination.resolve()),
                    "report_sha256": report["report_sha256"],
                    "total_paths": report["runtime"]["total_paths"],
                    "stopped_by_time_limit": report["runtime"][
                        "stopped_by_time_limit"
                    ],
                }
            )
        elif args.command == "upcoming-card":
            website_path, publication = execute_upcoming_card(
                card_path=args.card,
                outcome_path=args.outcomes,
                raw_path=args.raw,
                profiles_path=args.profiles,
                round_path=args.round_stats,
                output_dir=args.output_dir,
                website_output=args.website_output,
                minimum_prior_ufc_fights=args.minimum_prior_ufc_fights,
                bootstrap_members=args.bootstrap_members,
                initial_paths_per_member=args.initial_paths_per_member,
                max_paths_per_member=args.max_paths_per_member,
                random_seed=args.random_seed,
                workers=args.workers,
                chunk_size=args.chunk_size,
                simulator_config=_load_simulator_config(args.simulator_config),
                parameter_artifact_path=args.parameter_artifact,
                parameter_cache_dir=args.parameter_cache_dir,
                resume=args.resume,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            _print(
                {
                    "available_matchups": publication["available_matchups"],
                    "excluded_matchups": publication["excluded_matchups"],
                    "publication_sha256": publication["publication_sha256"],
                    "website_output": str(website_path.resolve()),
                }
            )
        elif args.command == "compare-outcome-mechanics":
            comparison = compare_outcome_mechanics(
                args.baseline,
                args.candidate,
                output=args.output,
                minimum_balanced_events=args.minimum_balanced_events,
            )
            _print(
                {
                    "balanced_events": comparison["balanced_event_count"],
                    "balanced_fights": comparison["balanced_fight_count"],
                    "development_status": comparison["development_status"],
                    "output": str(Path(args.output).resolve()),
                }
            )
        elif args.command == "derive-mechanics":
            profile = derive_mechanics_profile(
                args.population_run,
                output=args.output,
                holdout_latest_events=args.holdout_latest_events,
                prior_strength_events=args.prior_strength_events,
            )
            _print(
                {
                    "development_fights": profile["development_fight_count"],
                    "held_out_events": profile["held_out_event_count"],
                    "mechanics_profile_id": profile["mechanics_profile_id"],
                    "output": str(Path(args.output).resolve()),
                    "profile_sha256": profile["profile_sha256"],
                }
            )
        elif args.command == "select-mechanics":
            candidates = {}
            for value in args.candidate:
                label, separator, path = value.partition("=")
                if not separator or not label.strip() or not path.strip():
                    raise ValueError("candidate must use LABEL=REPORT")
                if label in candidates:
                    raise ValueError(f"duplicate candidate label: {label}")
                candidates[label] = path
            selection = select_mechanics_profile(
                args.baseline_population_run,
                candidates,
                output=args.output,
                selection_events=args.selection_events,
                skip_latest_events=args.skip_latest_events,
            )
            _print(
                {
                    "mechanics_profile_id": selection["mechanics_profile_id"],
                    "output": str(Path(args.output).resolve()),
                    "selected_label": selection["selected_label"],
                    "selection_sha256": selection["selection_sha256"],
                    "selection_status": selection["selection_status"],
                }
            )
        elif args.command == "validate-mechanics":
            validation = validate_mechanics_holdout(
                args.baseline_population_run,
                args.tuned_population_run,
                output=args.output,
                holdout_latest_events=args.holdout_latest_events,
            )
            _print(
                {
                    "mechanics_profile_id": validation["mechanics_profile_id"],
                    "output": str(Path(args.output).resolve()),
                    "validation_sha256": validation["validation_sha256"],
                    "validation_status": validation["validation_status"],
                }
            )
        elif args.command == "select-finishing":
            candidates = {}
            for value in args.candidate:
                label, separator, path = value.partition("=")
                if not separator or not label.strip() or not path.strip():
                    raise ValueError("candidate must use LABEL=REPORT")
                if label in candidates:
                    raise ValueError(f"duplicate candidate label: {label}")
                candidates[label] = path
            selection = select_finish_profile(
                args.baseline_population_run,
                candidates,
                output=args.output,
                objective=args.objective,
            )
            _print(
                {
                    "mechanics_profile_id": selection["mechanics_profile_id"],
                    "output": str(Path(args.output).resolve()),
                    "selected_label": selection["selected_label"],
                    "selection_sha256": selection["selection_sha256"],
                    "selection_status": selection["selection_status"],
                }
            )
        elif args.command == "validate-finishing":
            validation = validate_finish_profile(
                args.baseline_holdout_run,
                args.candidate_holdout_run,
                output=args.output,
            )
            _print(
                {
                    "mechanics_profile_id": validation["mechanics_profile_id"],
                    "output": str(Path(args.output).resolve()),
                    "validation_sha256": validation["validation_sha256"],
                    "validation_status": validation["validation_status"],
                }
            )
        elif args.command == "validate-knockdown-observation":
            validation = validate_knockdown_observation_profile(
                args.baseline_holdout_run,
                args.candidate_holdout_run,
                output=args.output,
            )
            _print(
                {
                    "mechanics_profile_id": validation["mechanics_profile_id"],
                    "output": str(Path(args.output).resolve()),
                    "validation_sha256": validation["validation_sha256"],
                    "validation_status": validation["validation_status"],
                }
            )
        elif args.command == "transition-audit":
            if args.max_runtime_seconds > 3300:
                raise ValueError("max-runtime-seconds must not exceed 3300")
            report, report_path, predictions_path = execute_transition_audit(
                round_path=args.round_stats,
                output=args.output,
                predictions_output=args.predictions_output,
                config=TransitionAuditConfig(
                    holdout_latest_events=args.holdout_latest_events,
                    context_prior_opportunities=args.context_prior_opportunities,
                    fighter_prior_opportunities=args.fighter_prior_opportunities,
                    bootstrap_replicates=args.bootstrap_replicates,
                    random_seed=args.random_seed,
                    max_runtime_seconds=args.max_runtime_seconds,
                    as_of=args.as_of,
                ),
            )
            _print(
                {
                    "output": str(report_path.resolve()),
                    "predictions_output": str(predictions_path.resolve()),
                    "report_sha256": report["report_sha256"],
                    "retained_targets": [
                        name
                        for name, result in report["targets"].items()
                        if result.get("candidate_retained")
                    ],
                }
            )
        elif args.command == "opponent-adjustment-audit":
            if args.max_runtime_seconds > 3300:
                raise ValueError("max-runtime-seconds must not exceed 3300")
            report, report_path, predictions_path = (
                execute_opponent_adjustment_audit(
                    raw_path=args.raw,
                    profiles_path=args.profiles,
                    round_path=args.round_stats,
                    cohort_manifest_path=args.cohort_manifest,
                    cohort_name=args.cohort_name,
                    output=args.output,
                    predictions_output=args.predictions_output,
                    config=OpponentAdjustmentAuditConfig(
                        min_prior_ufc_fights=args.min_prior_ufc_fights,
                        inner_validation_events=args.inner_validation_events,
                        minimum_training_fights=args.minimum_training_fights,
                        ridge_grid=args.ridge_grid,
                        bootstrap_replicates=args.bootstrap_replicates,
                        random_seed=args.random_seed,
                        max_runtime_seconds=args.max_runtime_seconds,
                    ),
                    progress=lambda message: print(
                        message, file=sys.stderr, flush=True
                    ),
                )
            )
            _print(
                {
                    "candidate_advances_to_simulation_screen": report[
                        "candidate_advances_to_simulation_screen"
                    ],
                    "decision": report["decision"],
                    "output": str(report_path.resolve()),
                    "predictions_output": str(predictions_path.resolve()),
                    "report_sha256": report["report_sha256"],
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
        elif args.command == "benchmark":
            benchmark = execute_benchmark(
                args.specs,
                paths_per_member=args.paths_per_member,
                worker_counts=args.workers,
                chunk_size=args.chunk_size,
                repeats=args.repeats,
                output=args.output,
            )
            _print(benchmark)
        elif args.command == "validate-fight":
            run = Path(args.run)
            report = validate_completed_fight(
                run,
                observed_path=args.observed,
                fight_id=args.fight_id,
            )
            json_output = Path(args.json_output) if args.json_output else run / "validation.json"
            html_output = Path(args.html_output) if args.html_output else run / "validation.html"
            written_json, written_html = write_validation_report(
                report,
                json_path=json_output,
                html_path=html_output,
            )
            _print(
                {
                    "actual_outcome": report["actual_outcome"],
                    "actual_outcome_probability": report["actual_outcome_probability"],
                    "html_output": str(written_html.resolve()),
                    "json_output": str(written_json.resolve()),
                    "scored_marginals": report["summary"]["scored_marginals"],
                }
            )
        elif args.command == "gui":
            from .gui import launch_gui

            return launch_gui(args.run)
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
