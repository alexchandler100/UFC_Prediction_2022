"""Publish the website's settled paper bets and bankroll-replay inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import pandas as pd

from market_tracker import (
    ForecastCaptureStore,
    PaperDecisionStore,
    PaperSettlementStore,
    QuoteSnapshotStore,
    TotalRoundsPaperDecisionStore,
    TotalRoundsPaperSettlementStore,
)
from market_tracker.bankroll import (
    archive_upcoming_bet_board,
    bet_support_key,
    build_bet_performance_publication,
    empty_published_bet_archive,
    validate_bet_performance_publication,
    validate_published_bet_archive,
    write_bet_performance_publication,
)
from upcoming_bet_board import validate_upcoming_bet_board


ROOT = Path(__file__).resolve().parent
MARKET = ROOT / "content" / "data" / "market"
RAW = ROOT / "content" / "data" / "processed" / "ufc_fights_reported_doubled.csv"
ARCHIVE = MARKET / "published_bet_snapshots.json"
OUTPUT = MARKET / "bet_performance.json"
PREDICTION_HISTORY = ROOT / "content" / "data" / "external" / "prediction_history.json"
SIMULATION_FORECASTS = ROOT / "content" / "data" / "external" / "simulation_forecasts.json"
REPOSITORY = ROOT.parent
BOARD_REPOSITORY_PATH = "src/content/data/market/upcoming_bet_board.json"


def _identity(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().rstrip("/").rsplit("/", 1)[-1].casefold()


def _completed_results(
    frame: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str, str], int | None],
    dict[tuple[str, str, str], float],
]:
    outcomes: dict[tuple[str, str, str], int | None] = {}
    durations: dict[tuple[str, str, str], float] = {}
    for _, group in frame.groupby("fight_url", sort=False, dropna=False):
        if len(group) != 2:
            continue
        event_ids = {_identity(value) for value in group["event_url"]}
        fighter_ids = {_identity(value) for value in group["fighter_url"]}
        if len(event_ids) != 1 or len(fighter_ids) != 2 or "" in fighter_ids:
            continue
        event_id = next(iter(event_ids))
        fighter_id, opponent_id = sorted(fighter_ids)
        key = (event_id, fighter_id, opponent_id)
        canonical = group[group["fighter_url"].map(_identity).eq(fighter_id)]
        if len(canonical) != 1:
            continue
        result = str(canonical.iloc[0]["result"]).strip().upper()
        target = 1 if result == "W" else 0 if result == "L" else None
        prior = outcomes.get(key, target)
        if key in outcomes and prior != target:
            outcomes.pop(key, None)
            durations.pop(key, None)
            continue
        outcomes[key] = target
        seconds = pd.to_numeric(group["total_fight_time"], errors="coerce").dropna()
        if len(seconds) == 2 and abs(float(seconds.iloc[0]) - float(seconds.iloc[1])) <= 1e-9:
            value = float(seconds.iloc[0])
            if 0.0 < value <= 25.0 * 300.0:
                durations[key] = value
    return outcomes, durations


def _stores() -> dict[str, object]:
    return {
        "quotes": QuoteSnapshotStore(MARKET / "quote_snapshots.csv", MARKET / "quote_snapshots.jsonl"),
        "forecasts": ForecastCaptureStore(MARKET / "forecast_captures.csv", MARKET / "forecast_captures.jsonl"),
        "decisions": PaperDecisionStore(MARKET / "paper_decisions.csv", MARKET / "paper_decisions.jsonl"),
        "settlements": PaperSettlementStore(MARKET / "paper_settlements.csv", MARKET / "paper_settlements.jsonl"),
        "total_decisions": TotalRoundsPaperDecisionStore(MARKET / "total_round_paper_decisions.csv", MARKET / "total_round_paper_decisions.jsonl"),
        "total_settlements": TotalRoundsPaperSettlementStore(MARKET / "total_round_paper_settlements.csv", MARKET / "total_round_paper_settlements.jsonl"),
    }


def _model_support() -> dict[tuple[str, ...], dict[str, object]]:
    if not PREDICTION_HISTORY.exists():
        return {}
    frame = pd.read_json(PREDICTION_HISTORY)
    required = {
        "event id", "fighter id", "opponent id", "model probability",
        "forecast issued at",
    }
    if not required.issubset(frame.columns):
        return {}
    output: dict[tuple[str, ...], dict[str, object]] = {}
    for _, row in frame.iterrows():
        event_id = _identity(row.get("event id"))
        fighter_id = _identity(row.get("fighter id"))
        opponent_id = _identity(row.get("opponent id"))
        probability = pd.to_numeric(row.get("model probability"), errors="coerce")
        issued = str(row.get("forecast issued at") or "")
        if (
            not event_id or not fighter_id or not opponent_id
            or pd.isna(probability) or not 0.0 < float(probability) < 1.0
            or not issued or issued == "NaT"
        ):
            continue
        for side, chance in (
            ("fighter", float(probability)),
            ("opponent", 1.0 - float(probability)),
        ):
            key = bet_support_key(
                event_id=event_id,
                fighter_id=fighter_id,
                opponent_id=opponent_id,
                category="Moneyline",
                side=side,
                selection="",
            )
            output[key] = {
                "probability": chance,
                "source": "production_winner_model",
                "issued_at_utc": issued,
            }
    return output


def _simulation_support() -> dict[tuple[str, ...], dict[str, object]]:
    if not SIMULATION_FORECASTS.exists():
        return {}
    publication = json.loads(SIMULATION_FORECASTS.read_text(encoding="utf-8"))
    legacy_event_id = _identity(publication.get("event_id"))
    legacy_issued = str(publication.get("forecast_issued_at_utc") or "")
    output: dict[tuple[str, ...], dict[str, object]] = {}
    for matchup in publication.get("matchups", []):
        if not isinstance(matchup, dict) or matchup.get("status") != "available":
            continue
        event_id = _identity(matchup.get("event_id")) or legacy_event_id
        issued = str(matchup.get("forecast_issued_at_utc") or legacy_issued)
        fighter_id = _identity(matchup.get("fighter_id"))
        opponent_id = _identity(matchup.get("opponent_id"))
        aggregate = matchup.get("aggregate")
        probabilities = aggregate.get("outcome_probabilities") if isinstance(aggregate, dict) else None
        if not event_id or not issued or not fighter_id or not opponent_id or not isinstance(probabilities, dict):
            continue
        red = math.fsum(
            float(value) for key, value in probabilities.items()
            if str(key).startswith("red_")
        )
        blue = math.fsum(
            float(value) for key, value in probabilities.items()
            if str(key).startswith("blue_")
        )
        if red <= 0.0 or blue <= 0.0:
            continue
        for side, chance in (("fighter", red / (red + blue)), ("opponent", blue / (red + blue))):
            key = bet_support_key(
                event_id=event_id,
                fighter_id=fighter_id,
                opponent_id=opponent_id,
                category="Moneyline",
                side=side,
                selection="",
            )
            output[key] = {
                "probability": chance,
                "source": "frozen_pre_event_monte_carlo",
                "issued_at_utc": issued,
            }
        for total in aggregate.get("total_lines", []):
            if not isinstance(total, dict):
                continue
            over = float(total.get("over") or 0.0)
            under = float(total.get("under") or 0.0)
            if over <= 0.0 or under <= 0.0:
                continue
            line = float(total.get("half_rounds"))
            for side, chance in (("over", over / (over + under)), ("under", under / (over + under))):
                key = bet_support_key(
                    event_id=event_id,
                    fighter_id=fighter_id,
                    opponent_id=opponent_id,
                    category="Total rounds",
                    side=side,
                    selection=f"{side.title()} {line:g} rounds",
                )
                output[key] = {
                    "probability": chance,
                    "source": "frozen_pre_event_monte_carlo",
                    "issued_at_utc": issued,
                }
    return output


def _prior_support() -> dict[str, dict[str, object]]:
    if not OUTPUT.exists():
        return {}
    publication = validate_bet_performance_publication(
        json.loads(OUTPUT.read_text(encoding="utf-8"))
    )
    return {str(item["record_id"]): item for item in publication["records"]}


def update_bet_performance() -> dict[str, object]:
    stores = _stores()
    records = {key: store.read() for key, store in stores.items()}
    archive = empty_published_bet_archive(ARCHIVE)
    outcomes, durations = _completed_results(pd.read_csv(RAW, low_memory=False))
    publication = build_bet_performance_publication(
        decisions=records["decisions"],
        settlements=records["settlements"],
        quotes=records["quotes"],
        forecasts=records["forecasts"],
        total_decisions=records["total_decisions"],
        total_settlements=records["total_settlements"],
        archive=archive,
        outcomes=outcomes,
        durations=durations,
        model_support=_model_support(),
        simulation_support=_simulation_support(),
        prior_support=_prior_support(),
    )
    write_bet_performance_publication(publication, OUTPUT)
    return publication


def backfill_git_board_history() -> int:
    """Recover qualified boards that were genuinely committed before archiving began."""

    git = [
        "git", "-c", f"safe.directory={REPOSITORY.as_posix()}",
    ]
    history = subprocess.run(
        [*git, "log", "--all", "--format=%H", "--", BOARD_REPOSITORY_PATH],
        cwd=REPOSITORY, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    boards: list[dict[str, object]] = []
    for revision in reversed(history):
        payload = subprocess.run(
            [*git, "show", f"{revision}:{BOARD_REPOSITORY_PATH}"],
            cwd=REPOSITORY, check=True, capture_output=True, text=True,
        ).stdout
        boards.append(validate_upcoming_bet_board(json.loads(payload)))
    for board in boards:
        archive_upcoming_bet_board(board, ARCHIVE)
    return len(boards)


def validate_current() -> dict[str, object]:
    validate_published_bet_archive(json.loads(ARCHIVE.read_text(encoding="utf-8")))
    return validate_bet_performance_publication(
        json.loads(OUTPUT.read_text(encoding="utf-8"))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--backfill-git-history", action="store_true")
    args = parser.parse_args()
    if args.validate_only and args.backfill_git_history:
        parser.error("--validate-only and --backfill-git-history cannot be combined")
    if args.backfill_git_history:
        count = backfill_git_board_history()
        print(f"Recovered {count} committed upcoming-bet board version(s).")
    publication = validate_current() if args.validate_only else update_bet_performance()
    print(
        "Paper-bet performance: "
        f"{publication['official_wins']}-{publication['official_losses']} across "
        f"{publication['official_settled_count']} settled official bets; "
        f"{publication['record_count']} timestamped records available."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
