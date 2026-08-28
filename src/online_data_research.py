"""Validated, research-only imports for free UFC data sources.

The production model does not read this module.  It exists to answer three
questions without weakening the point-in-time data contract:

* can a large public MMA archive safely extend pre-UFC fight history;
* do historical UFC rankings improve winner probabilities; and
* can a free odds archive support a genuinely pre-fight market comparison.

Every loader returns an audit alongside usable rows.  Rows with ambiguous
fighter identity, future dates, duplicate bouts, or late odds timestamps are
excluded rather than guessed.
"""

from __future__ import annotations

from collections import defaultdict
from bisect import bisect_right
from datetime import date
from hashlib import sha256
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from external_mma.identity import normalize_name, propose_ufcstats_crosswalk
from external_mma.integration import build_auxiliary_doubled, is_ufc_promotion
from external_mma.schema import ExternalBoutObservation
from fight_predictor.point_in_time import _identity_token


MMA_SOURCE_KEY = "database_complete_mma_v3"
RANKING_FEATURES = (
    "ranking_division_score_diff",
    "ranking_division_known_diff",
    "ranking_champion_diff",
    "ranking_p4p_score_diff",
    "ranking_peak_score_diff",
    "ranking_snapshots_log_diff",
    "ranking_momentum_13w_diff",
)
RANKING_FAMILIES = {
    "current_division_rank": RANKING_FEATURES[:3],
    "current_pound_for_pound_rank": RANKING_FEATURES[3:4],
    "ranking_history": RANKING_FEATURES[4:],
}


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fighter_name_index(
    fighter_stats: pd.DataFrame,
    raw_fights: pd.DataFrame,
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for name, url in fighter_stats[["name", "url"]].itertuples(index=False):
        fighter_id = _identity_token(url)
        if fighter_id:
            index[normalize_name(name)].add(fighter_id)
    for name, url in raw_fights[["fighter", "fighter_url"]].itertuples(index=False):
        fighter_id = _identity_token(url)
        if fighter_id:
            index[normalize_name(name)].add(fighter_id)
    return index


def prepare_rankings(
    rankings: pd.DataFrame,
    fighter_stats: pd.DataFrame,
    raw_fights: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate rankings and map only unique UFCStats fighter names.

    The source contains synthetic ``Top Rank`` categories in which the same
    fighter is listed as both zero and one.  Those categories are rejected;
    real divisional and pound-for-pound lists are retained.
    """

    required = {"date", "weightclass", "fighter", "rank"}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"rankings are missing columns: {sorted(missing)}")
    working = rankings[list(required)].copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["rank"] = pd.to_numeric(working["rank"], errors="coerce")
    invalid = (
        working["date"].isna()
        | working["fighter"].astype(str).str.strip().eq("")
        | working["weightclass"].astype(str).str.strip().eq("")
        | working["rank"].isna()
        | ~working["rank"].between(0, 15)
        | (working["rank"] % 1).ne(0)
    )
    invalid_rows = int(invalid.sum())
    working = working.loc[~invalid].copy()
    synthetic = working["weightclass"].astype(str).str.contains(
        "top rank", case=False, na=False
    )
    synthetic_rows = int(synthetic.sum())
    working = working.loc[~synthetic].copy()

    key = ["date", "weightclass", "fighter"]
    conflicts = working.groupby(key, dropna=False)["rank"].nunique().gt(1)
    conflict_dates = sorted({item[0] for item in conflicts[conflicts].index})
    conflicting_snapshot_rows = int(working["date"].isin(conflict_dates).sum())
    # A weekly page occasionally contains two different lists concatenated
    # together.  Reject its entire date; choosing either copy would be a guess.
    working = working.loc[~working["date"].isin(conflict_dates)].copy()
    duplicate_rows = int(working.duplicated(key).sum())
    working = working.drop_duplicates(key, keep="last")

    identities = _fighter_name_index(fighter_stats, raw_fights)
    working["normalized_name"] = working["fighter"].map(normalize_name)
    working["identity_candidates"] = working["normalized_name"].map(
        lambda item: identities.get(item, set())
    )
    working["fighter_id"] = working["identity_candidates"].map(
        lambda values: next(iter(values)) if len(values) == 1 else ""
    )
    ambiguous = working["identity_candidates"].map(len).gt(1)
    unmatched = working["identity_candidates"].map(len).eq(0)
    mapped = working.loc[working["fighter_id"].ne("")].copy()
    mapped["rank"] = mapped["rank"].astype(int)
    mapped["is_p4p"] = mapped["weightclass"].astype(str).str.contains(
        "pound-for-pound", case=False, na=False
    )
    mapped = mapped[
        ["date", "weightclass", "fighter", "fighter_id", "rank", "is_p4p"]
    ].sort_values(["date", "fighter_id", "is_p4p", "rank"], kind="stable")
    audit = {
        "source_rows": int(len(rankings)),
        "invalid_rows_rejected": invalid_rows,
        "synthetic_top_rank_rows_rejected": synthetic_rows,
        "conflicting_snapshot_dates_rejected": len(conflict_dates),
        "conflicting_snapshot_rows_rejected": conflicting_snapshot_rows,
        "exact_duplicate_rows_removed": duplicate_rows,
        "mapped_rows": int(len(mapped)),
        "mapped_fighters": int(mapped["fighter_id"].nunique()),
        "unmatched_rows": int(unmatched.sum()),
        "ambiguous_name_rows": int(ambiguous.sum()),
        "first_snapshot": mapped["date"].min().date().isoformat(),
        "last_snapshot": mapped["date"].max().date().isoformat(),
        "identity_rule": "normalized name must resolve to exactly one UFCStats ID",
        "fight_join_rule": "latest complete ranking snapshot strictly before bout date",
    }
    return mapped.reset_index(drop=True), audit


def _ranking_side_snapshots(rankings: pd.DataFrame) -> tuple[
    list[pd.Timestamp],
    dict[pd.Timestamp, dict[str, dict[str, float]]],
]:
    """Materialize bounded side features at each weekly snapshot."""

    dates = sorted(pd.Timestamp(value) for value in rankings["date"].unique())
    snapshots: dict[pd.Timestamp, dict[str, dict[str, float]]] = {}
    peak: dict[str, float] = defaultdict(float)
    appeared: dict[str, int] = defaultdict(int)
    history_dates: dict[str, list[pd.Timestamp]] = defaultdict(list)
    history_scores: dict[str, list[float]] = defaultdict(list)
    seen_fighters: set[str] = set()
    rows_by_date = {
        pd.Timestamp(snapshot_date): rows
        for snapshot_date, rows in rankings.groupby("date", sort=True)
    }
    for snapshot_date in dates:
        rows = rows_by_date[snapshot_date]
        by_fighter: dict[str, dict[str, float]] = {}
        current_rows = {
            str(fighter_id): fighter_rows
            for fighter_id, fighter_rows in rows.groupby("fighter_id", sort=False)
        }
        seen_fighters.update(current_rows)
        for fighter_id in sorted(seen_fighters):
            fighter_rows = current_rows.get(fighter_id)
            if fighter_rows is None:
                fighter_rows = rows.iloc[0:0]
            divisional = fighter_rows.loc[~fighter_rows["is_p4p"]]
            p4p = fighter_rows.loc[fighter_rows["is_p4p"]]
            division_rank = (
                int(divisional["rank"].min()) if not divisional.empty else None
            )
            p4p_rank = int(p4p["rank"].min()) if not p4p.empty else None
            division_score = 16.0 - division_rank if division_rank is not None else 0.0
            p4p_score = 16.0 - p4p_rank if p4p_rank is not None else 0.0
            peak[fighter_id] = max(peak[fighter_id], division_score)
            if division_rank is not None:
                appeared[fighter_id] += 1
            history_dates[fighter_id].append(snapshot_date)
            history_scores[fighter_id].append(division_score)
            cutoff = snapshot_date - pd.Timedelta(days=91)
            prior_position = bisect_right(history_dates[fighter_id], cutoff) - 1
            prior_score = (
                history_scores[fighter_id][prior_position]
                if prior_position >= 0 else 0.0
            )
            by_fighter[fighter_id] = {
                "ranking_division_score": division_score,
                "ranking_division_known": float(division_rank is not None),
                "ranking_champion": float(division_rank == 0),
                "ranking_p4p_score": p4p_score,
                "ranking_peak_score": peak[fighter_id],
                "ranking_snapshots_log": math.log1p(appeared[fighter_id]),
                "ranking_momentum_13w": division_score - prior_score,
            }
        snapshots[snapshot_date] = by_fighter
    return dates, snapshots


def add_ranking_features(
    matchups: pd.DataFrame,
    rankings: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add antisymmetric ranking features without using same-day rankings."""

    required = {"date", "fighter_id", "opponent_id", "fight_id"}
    missing = required - set(matchups.columns)
    if missing:
        raise ValueError(f"matchups are missing columns: {sorted(missing)}")
    if matchups["fight_id"].duplicated().any():
        raise ValueError("ranking feature input contains duplicate fight IDs")
    dates, snapshots = _ranking_side_snapshots(rankings)
    date_values = np.asarray(dates, dtype="datetime64[ns]")
    side_names = tuple(item.removesuffix("_diff") for item in RANKING_FEATURES)
    zero = dict.fromkeys(side_names, 0.0)
    output = matchups.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise")
    values: list[list[float]] = []
    covered = 0
    both_covered = 0
    snapshot_dates: list[str] = []
    for fight_date, fighter_id, opponent_id in output[
        ["date", "fighter_id", "opponent_id"]
    ].itertuples(index=False, name=None):
        position = int(np.searchsorted(date_values, np.datetime64(fight_date), side="left")) - 1
        if position < 0:
            fighter = zero
            opponent = zero
            snapshot_dates.append("")
        else:
            snapshot_date = dates[position]
            snapshot = snapshots[snapshot_date]
            fighter = snapshot.get(str(fighter_id), zero)
            opponent = snapshot.get(str(opponent_id), zero)
            snapshot_dates.append(snapshot_date.date().isoformat())
        fighter_known = bool(fighter["ranking_division_known"] or fighter["ranking_p4p_score"])
        opponent_known = bool(opponent["ranking_division_known"] or opponent["ranking_p4p_score"])
        covered += int(fighter_known or opponent_known)
        both_covered += int(fighter_known and opponent_known)
        values.append([float(fighter[name]) - float(opponent[name]) for name in side_names])
    output[list(RANKING_FEATURES)] = np.asarray(values, dtype=float)
    output["ranking_snapshot_date"] = snapshot_dates
    same_or_future = output["ranking_snapshot_date"].ne("") & (
        pd.to_datetime(output["ranking_snapshot_date"]) >= output["date"]
    )
    if same_or_future.any():
        raise RuntimeError("ranking join used a same-day or future snapshot")
    audit = {
        "fight_rows": int(len(output)),
        "fights_with_at_least_one_currently_ranked_fighter": covered,
        "fights_with_two_currently_ranked_fighters": both_covered,
        "strictly_prior_snapshot_verified": True,
    }
    return output, audit


def prepare_pre_event_odds(
    odds: pd.DataFrame,
    matchups: pd.DataFrame,
    *,
    minimum_books: int = 3,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the latest safely pre-event no-vig consensus per UFCStats bout.

    Event times are unavailable, so an odds row is eligible only when its UTC
    capture *date* is before the event date.  This deliberately rejects all
    same-day rows and historical prices that were added to the archive later.
    """

    required = {
        "fight_url", "fighter_1_url", "fighter_2_url", "fighter_1", "fighter_2",
        "odds_1", "odds_2", "event_date", "adding_date", "source",
    }
    missing = required - set(odds.columns)
    if missing:
        raise ValueError(f"odds are missing columns: {sorted(missing)}")
    working = odds[list(required)].copy()
    working["fight_id"] = working["fight_url"].map(_identity_token)
    working["fighter_1_id"] = working["fighter_1_url"].map(_identity_token)
    working["fighter_2_id"] = working["fighter_2_url"].map(_identity_token)
    working["event_date"] = pd.to_datetime(working["event_date"], errors="coerce")
    working["observed_at"] = pd.to_datetime(
        working["adding_date"], errors="coerce", utc=True
    )
    working["odds_1"] = pd.to_numeric(working["odds_1"], errors="coerce")
    working["odds_2"] = pd.to_numeric(working["odds_2"], errors="coerce")
    valid = (
        working["fight_id"].ne("")
        & working["fighter_1_id"].ne("")
        & working["fighter_2_id"].ne("")
        & working["event_date"].notna()
        & working["observed_at"].notna()
        & working["odds_1"].gt(1.0)
        & working["odds_2"].gt(1.0)
    )
    valid_rows = int(valid.sum())
    working = working.loc[valid].copy()
    safely_prior = working["observed_at"].dt.date < working["event_date"].dt.date
    late_rows = int((~safely_prior).sum())
    working = working.loc[safely_prior].copy()
    working = working.sort_values(
        ["fight_id", "source", "observed_at"], kind="stable"
    ).drop_duplicates(["fight_id", "source"], keep="last")
    implied_1 = 1.0 / working["odds_1"]
    implied_2 = 1.0 / working["odds_2"]
    working["fighter_1_probability"] = implied_1 / (implied_1 + implied_2)

    matchup_by_id = matchups.set_index("fight_id", drop=False)
    rows: list[dict[str, object]] = []
    identity_rejections = 0
    url_name_mismatches = 0
    for fight_id, quotes in working.groupby("fight_id", sort=False):
        if fight_id not in matchup_by_id.index or len(quotes) < minimum_books:
            continue
        matchup = matchup_by_id.loc[fight_id]
        if isinstance(matchup, pd.DataFrame):
            raise ValueError(f"duplicate matchup ID {fight_id}")
        probabilities: list[float] = []
        for quote in quotes.to_dict("records"):
            first_name = normalize_name(quote["fighter_1"])
            second_name = normalize_name(quote["fighter_2"])
            fighter_name = normalize_name(matchup["fighter"])
            opponent_name = normalize_name(matchup["opponent"])
            if first_name == fighter_name and second_name == opponent_name:
                probabilities.append(float(quote["fighter_1_probability"]))
                url_name_mismatches += int(
                    str(quote["fighter_1_id"]) != str(matchup["fighter_id"])
                    or str(quote["fighter_2_id"]) != str(matchup["opponent_id"])
                )
            elif first_name == opponent_name and second_name == fighter_name:
                probabilities.append(1.0 - float(quote["fighter_1_probability"]))
                url_name_mismatches += int(
                    str(quote["fighter_1_id"]) != str(matchup["opponent_id"])
                    or str(quote["fighter_2_id"]) != str(matchup["fighter_id"])
                )
            else:
                identity_rejections += 1
        if len(probabilities) < minimum_books:
            continue
        rows.append({
            "date": matchup["date"],
            "event_id": matchup["event_id"],
            "fight_id": fight_id,
            "fighter_id": matchup["fighter_id"],
            "opponent_id": matchup["opponent_id"],
            "fighter": matchup["fighter"],
            "opponent": matchup["opponent"],
            "target": int(matchup["target"]),
            "market_probability": float(np.mean(probabilities)),
            "book_count": len(probabilities),
            "first_observed_at": quotes["observed_at"].min().isoformat(),
            "last_observed_at": quotes["observed_at"].max().isoformat(),
        })
    result = pd.DataFrame(rows)
    audit = {
        "source_rows": int(len(odds)),
        "rows_with_parseable_ids_dates_and_prices": valid_rows,
        "late_or_same_day_rows_rejected": late_rows,
        "safely_pre_event_rows": int(len(working)),
        "paired_fights_with_minimum_books": int(len(result)),
        "minimum_books": minimum_books,
        "quote_identity_rejections": identity_rejections,
        "source_url_name_mismatches": url_name_mismatches,
        "side_orientation_rule": (
            "fighter names must exactly match the two participants within a stable "
            "UFCStats fight ID; source fighter URLs are audited but not trusted because "
            "the archive swaps them on some rows"
        ),
        "capture_rule": "UTC adding_date calendar day is before event_date",
        "warning": (
            "The archive has authentic pre-event timestamps only for a narrow "
            "2025 collection window; older backfilled rows are not causal inputs."
        ),
    }
    return result, audit


def _mma_result(winner: object, first: str, second: str) -> str | None:
    if winner is None or pd.isna(winner):
        return None
    normalized = normalize_name(winner)
    if normalized == normalize_name(first):
        return "W"
    if normalized == normalize_name(second):
        return "L"
    return None


def load_mma_archive_observations(
    database_path: str | Path,
    *,
    completed_through: date | None = None,
) -> tuple[list[ExternalBoutObservation], dict[str, object]]:
    """Load only source-ID-resolvable, unique, completed bouts from DuckDB."""

    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - exercised by CLI setup
        raise RuntimeError("duckdb is required for the public MMA archive") from error
    path = Path(database_path)
    snapshot_hash = file_sha256(path)
    connection = duckdb.connect(str(path), read_only=True)
    try:
        masters = connection.execute(
            "select fighter_id, fighter_name, dob from fighters_master"
        ).fetchdf()
        fights = connection.execute(
            """
            select fight_id, organization, event_name, event_date,
                   fighter_1, fighter_2, winner, method,
                   method_normalized, round_num, time_finish_seconds,
                   weight_class
            from fights_career_longitudinal
            """
        ).fetchdf()
    finally:
        connection.close()

    master_ids: dict[str, set[str]] = defaultdict(set)
    normalized_master_ids: dict[str, set[str]] = defaultdict(set)
    for fighter_id, fighter in masters[["fighter_id", "fighter_name"]].itertuples(
        index=False, name=None
    ):
        if pd.notna(fighter_id) and str(fighter_id).strip():
            master_ids[str(fighter)].add(str(fighter_id).strip())
            normalized_master_ids[normalize_name(fighter)].add(str(fighter_id).strip())
    unique_ids = {
        name: next(iter(values)) for name, values in master_ids.items() if len(values) == 1
    }
    ambiguous_master_names = {name for name, values in master_ids.items() if len(values) > 1}
    ambiguous_normalized_names = {
        name for name, values in normalized_master_ids.items() if len(values) > 1
    }

    through = completed_through or date.today()
    fights["event_date"] = pd.to_datetime(fights["event_date"], errors="coerce")
    future = fights["event_date"].dt.date.gt(through)
    invalid_date = fights["event_date"].isna()
    first_ids = fights["fighter_1"].map(unique_ids).fillna("")
    second_ids = fights["fighter_2"].map(unique_ids).fillna("")
    unique_identity = first_ids.ne("") & second_ids.ne("") & first_ids.ne(second_ids)
    result = [
        _mma_result(winner, first, second)
        for winner, first, second in fights[
            ["winner", "fighter_1", "fighter_2"]
        ].itertuples(index=False, name=None)
    ]
    decisive = pd.Series(result, index=fights.index).notna()
    eligible = ~future & ~invalid_date & unique_identity & decisive
    candidates = fights.loc[eligible].copy()
    candidates["fighter_1_id"] = first_ids.loc[eligible]
    candidates["fighter_2_id"] = second_ids.loc[eligible]
    candidates["result"] = pd.Series(result, index=fights.index).loc[eligible]
    candidates["pair"] = candidates.apply(
        lambda row: "\0".join(sorted((row["fighter_1_id"], row["fighter_2_id"]))),
        axis=1,
    )
    candidates["date_pair"] = (
        candidates["event_date"].dt.date.astype(str) + "\0" + candidates["pair"]
    )
    duplicated_date_pair = candidates["date_pair"].duplicated(keep=False)
    duplicated_date_pair_rows = int(duplicated_date_pair.sum())
    candidates = candidates.loc[~duplicated_date_pair].copy()
    occurrences = pd.concat(
        [
            candidates[["fight_id", "event_date", "fighter_1_id"]].rename(
                columns={"fighter_1_id": "fighter_id"}
            ),
            candidates[["fight_id", "event_date", "fighter_2_id"]].rename(
                columns={"fighter_2_id": "fighter_id"}
            ),
        ],
        ignore_index=True,
    )
    occurrences["event_day"] = occurrences["event_date"].dt.date
    repeated_keys = set(
        occurrences.groupby(["event_day", "fighter_id"])["fight_id"]
        .nunique()
        .loc[lambda values: values.gt(1)]
        .index
    )
    repeated_same_day = candidates.apply(
        lambda row: (
            (row["event_date"].date(), row["fighter_1_id"]) in repeated_keys
            or (row["event_date"].date(), row["fighter_2_id"]) in repeated_keys
        ),
        axis=1,
    )
    repeated_same_day_rows = int(repeated_same_day.sum())
    candidates = candidates.loc[~repeated_same_day].copy()

    observations: list[ExternalBoutObservation] = []
    rejected_schema = 0
    for row in candidates.to_dict("records"):
        event_date = pd.Timestamp(row["event_date"]).date().isoformat()
        event_key = sha256(
            f"{row['organization']}\0{row['event_name']}\0{event_date}".encode("utf-8")
        ).hexdigest()
        method = row.get("method_normalized") or row.get("method") or "OTHER"
        try:
            observations.append(ExternalBoutObservation.create(
                source=MMA_SOURCE_KEY,
                snapshot_sha256=snapshot_hash,
                source_bout_id=row["fight_id"],
                source_event_id=event_key,
                source_url=(
                    "https://github.com/LeandroIber/Database-complete-mma"
                    f"#fight-{row['fight_id']}"
                ),
                event_date=event_date,
                event_name=row["event_name"],
                promotion=row["organization"],
                fighter_source_id=row["fighter_1_id"],
                fighter_name=row["fighter_1"],
                opponent_source_id=row["fighter_2_id"],
                opponent_name=row["fighter_2"],
                result=row["result"],
                method=method,
                division=row.get("weight_class") or "Unknown",
                finish_round=row.get("round_num"),
                finish_clock_seconds=row.get("time_finish_seconds"),
            ))
        except (TypeError, ValueError):
            rejected_schema += 1
    audit = {
        "source_sha256": snapshot_hash,
        "source_fights": int(len(fights)),
        "source_master_fighters": int(len(masters)),
        "ambiguous_master_names": len(ambiguous_master_names),
        "ambiguous_normalized_master_names": len(ambiguous_normalized_names),
        "future_rows_rejected": int(future.fillna(False).sum()),
        "invalid_date_rows_rejected": int(invalid_date.sum()),
        "rows_without_two_unique_source_ids": int((~unique_identity).sum()),
        "nondecisive_or_unresolved_result_rows": int((~decisive).sum()),
        "duplicate_date_pair_rows_rejected": duplicated_date_pair_rows,
        "unknown_same_day_order_rows_rejected": repeated_same_day_rows,
        "schema_rows_rejected": rejected_schema,
        "accepted_physical_bouts": len(observations),
        "accepted_source_fighters": len({
            item for observation in observations
            for item in (observation.fighter_source_id, observation.opponent_source_id)
        }),
        "identity_rule": (
            "participant display name must resolve to exactly one fighters_master ID"
        ),
    }
    return observations, audit


def prepare_mma_auxiliary(
    observations: list[ExternalBoutObservation],
    raw_fights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Crosswalk source IDs via exact UFC witnesses and retain direct histories."""

    approved, review = propose_ufcstats_crosswalk(observations, raw_fights)
    identity_map = {
        (str(row.source), str(row.source_fighter_id)): str(row.canonical_fighter_id)
        for row in approved.itertuples(index=False)
    }
    mapped_source_ids = {source_id for (_source, source_id) in identity_map}
    connected = [
        observation for observation in observations
        if observation.fighter_source_id in mapped_source_ids
        or observation.opponent_source_id in mapped_source_ids
    ]
    auxiliary = build_auxiliary_doubled(connected, identity_map)
    audit = {
        "exact_ufc_witness_mappings": int(len(approved)),
        "conflicting_identity_mappings": int(len(review)),
        "connected_source_bouts_including_ufc": len(connected),
        "connected_non_ufc_physical_bouts": int(auxiliary["fight_url"].nunique()),
        "auxiliary_rows": int(len(auxiliary)),
        "ufc_results_excluded_from_auxiliary": True,
        "training_targets_emitted": False,
    }
    return auxiliary, approved, audit
