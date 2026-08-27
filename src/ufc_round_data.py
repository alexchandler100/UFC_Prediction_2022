"""Normalized, doubled UFCStats per-round data and reconciliation helpers.

The aggregate fight scraper predates UFCStats' per-round views.  This module
keeps the round contract independent of model code: it parses the two
``Per round`` tables from a fight-detail response, enriches them with durable
fight/fighter identities, and compares their sums with the aggregate table.
Missing source values remain missing; none of the helpers in this module
impute a statistic as zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from fight_semantics import (
    clock_seconds as _shared_clock_seconds,
    declared_round_lengths_seconds as _shared_round_lengths,
    scheduled_rounds_from_time_format as _shared_scheduled_rounds,
    stable_ufcstats_id,
)
from ufcstats_client import UFCStatsError


ROUND_DATA_SCHEMA_VERSION = 1

ROUND_STAT_COLUMNS = (
    "knockdowns",
    "sig_strikes_landed",
    "sig_strikes_attempts",
    "total_strikes_landed",
    "total_strikes_attempts",
    "takedowns_landed",
    "takedowns_attempts",
    "sub_attempts",
    "reversals",
    "control",
    "head_strikes_landed",
    "head_strikes_attempts",
    "body_strikes_landed",
    "body_strikes_attempts",
    "leg_strikes_landed",
    "leg_strikes_attempts",
    "distance_strikes_landed",
    "distance_strikes_attempts",
    "clinch_strikes_landed",
    "clinch_strikes_attempts",
    "ground_strikes_landed",
    "ground_strikes_attempts",
)

ROUND_DATA_COLUMNS = (
    "schema_version",
    "round_stat_id",
    "event_id",
    "event_url",
    "fight_id",
    "fight_url",
    "date",
    "source_card_index",
    "bout_order",
    "division",
    "time_format",
    "scheduled_rounds",
    "finish_round",
    "finish_time",
    "total_fight_seconds",
    "round",
    "round_seconds",
    "fighter_id",
    "fighter_url",
    "fighter",
    "opponent_id",
    "opponent_url",
    "opponent",
    "result",
    "method",
    *ROUND_STAT_COLUMNS,
    "reconciliation_status",
    "reconciliation_issue_count",
)

RECONCILIATION_COLUMNS = (
    "schema_version",
    "event_id",
    "fight_id",
    "fight_url",
    "fighter_id",
    "fighter",
    "field",
    "issue",
    "bout_value",
    "round_value",
    "delta",
    "detail",
)


def empty_round_stats_frame() -> pd.DataFrame:
    """Return an empty frame with the durable round-data column order."""

    return pd.DataFrame(columns=ROUND_DATA_COLUMNS)


def empty_reconciliation_frame() -> pd.DataFrame:
    """Return an empty frame with the durable discrepancy-report schema."""

    return pd.DataFrame(columns=RECONCILIATION_COLUMNS)


def ufcstats_identity(value: object) -> str:
    """Extract the stable lowercase identifier at the end of a source URL."""

    return stable_ufcstats_id(value)


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def scheduled_rounds_from_time_format(time_format: object) -> int | None:
    """Return UFCStats' declared number of rounds without guessing a default."""

    return _shared_scheduled_rounds(time_format)


def declared_round_lengths_seconds(time_format: object) -> tuple[int, ...]:
    """Return declared round lengths, expanding UFCStats' abbreviated suffix.

    UFCStats sometimes writes conventional schedules as ``3 Rnd (5-5)``.
    Repeating the final explicitly declared length is source-preserving for
    that notation.  A blank or malformed format returns an empty tuple rather
    than silently assuming five-minute rounds.
    """

    return _shared_round_lengths(time_format)


def _clock_seconds(value: object) -> int | None:
    return _shared_clock_seconds(value)


def round_exposure_seconds(
    round_number: object,
    finish_round: object,
    finish_time: object,
    time_format: object,
) -> int | None:
    """Return observed exposure for a round, or ``None`` when unknowable."""

    try:
        current = int(round_number)
        final = int(finish_round)
    except (TypeError, ValueError, OverflowError):
        return None
    lengths = declared_round_lengths_seconds(time_format)
    if current < 1 or final < 1 or current > final or current > len(lengths):
        return None
    if current < final:
        return lengths[current - 1]
    elapsed = _clock_seconds(finish_time)
    if elapsed is None or elapsed > lengths[current - 1]:
        return None
    return elapsed


def _integer(value: str) -> int | float:
    text = value.strip()
    return int(text) if text.isdigit() else np.nan


def _landed_attempted(value: str) -> tuple[int | float, int | float]:
    match = re.fullmatch(r"\s*(\d+)\s+of\s+(\d+)\s*", value, re.IGNORECASE)
    if match is None:
        return np.nan, np.nan
    return int(match.group(1)), int(match.group(2))


def _control_seconds(value: str) -> int | float:
    seconds = _clock_seconds(value)
    return seconds if seconds is not None else np.nan


def _two_source_values(cell: Any, *, url: str, field: str) -> tuple[str, str]:
    values = [node.get_text(" ", strip=True) for node in cell.select("p")]
    if len(values) != 2:
        raise UFCStatsError(
            f"Expected two fighter values for per-round {field} at {url}; "
            f"found {len(values)}"
        )
    return values[0], values[1]


def _round_table_bodies(soup: Any) -> list[Any]:
    bodies: list[Any] = []
    for link in soup.select("a.b-fight-details__collapse-link_rnd"):
        table = link.find_next_sibling("table")
        body = table.select_one("tbody.b-fight-details__table-body") if table else None
        if body is not None:
            bodies.append(body)
    return bodies


def _body_column_count(body: Any) -> int:
    first_row = body.select_one("tr")
    return len(first_row.select("td")) if first_row is not None else 0


def parse_ufcstats_round_stats(
    soup: Any,
    fight_url: str,
    time_format: object,
) -> pd.DataFrame:
    """Parse UFCStats' total and significant-strike per-round tables.

    The returned frame has two rows for every observed round.  It contains
    source identities and statistics only; :func:`normalize_round_stats`
    adds event/bout labels and exposure metadata.
    """

    bodies = _round_table_bodies(soup)
    totals = next(
        (body for body in bodies if _body_column_count(body) == 10),
        None,
    )
    significant = next(
        (body for body in bodies if _body_column_count(body) == 9),
        None,
    )
    if totals is None or significant is None:
        raise UFCStatsError(
            f"Missing UFCStats per-round total/significant-strike tables for {fight_url}"
        )

    total_rows = totals.select("tr.b-fight-details__table-row") or totals.select("tr")
    significant_rows = (
        significant.select("tr.b-fight-details__table-row") or significant.select("tr")
    )
    if not total_rows or len(total_rows) != len(significant_rows):
        raise UFCStatsError(
            f"Per-round table lengths disagree for {fight_url}: "
            f"totals={len(total_rows)}, significant={len(significant_rows)}"
        )

    parsed: list[dict[str, object]] = []
    fight_id = ufcstats_identity(fight_url)
    for round_index, (total_row, significant_row) in enumerate(
        zip(total_rows, significant_rows), start=1
    ):
        total_cells = total_row.select("td.b-fight-details__table-col")
        significant_cells = significant_row.select("td.b-fight-details__table-col")
        if len(total_cells) != 10 or len(significant_cells) != 9:
            raise UFCStatsError(
                f"Unexpected per-round column count for {fight_url} round {round_index}"
            )

        fighter_links = total_cells[0].select("a")
        significant_links = significant_cells[0].select("a")
        if len(fighter_links) != 2 or len(significant_links) != 2:
            raise UFCStatsError(
                f"Expected two linked fighters for {fight_url} round {round_index}"
            )
        total_identities = [
            (link.get_text(" ", strip=True), str(link.get("href") or "").strip())
            for link in fighter_links
        ]
        significant_identities = [
            (link.get_text(" ", strip=True), str(link.get("href") or "").strip())
            for link in significant_links
        ]
        if total_identities != significant_identities:
            raise UFCStatsError(
                f"Per-round fighter ordering disagrees for {fight_url} round {round_index}"
            )

        total_values = [
            _two_source_values(cell, url=fight_url, field=f"total column {index}")
            for index, cell in enumerate(total_cells[1:], start=1)
        ]
        significant_values = [
            _two_source_values(cell, url=fight_url, field=f"significant column {index}")
            for index, cell in enumerate(significant_cells[1:], start=1)
        ]
        if total_values[1] != significant_values[0]:
            raise UFCStatsError(
                f"Per-round significant-strike totals disagree between tables for "
                f"{fight_url} round {round_index}"
            )

        for side in range(2):
            fighter, fighter_url = total_identities[side]
            opponent, opponent_url = total_identities[1 - side]
            fighter_id = ufcstats_identity(fighter_url)
            opponent_id = ufcstats_identity(opponent_url)
            row: dict[str, object] = {
                "fight_id": fight_id,
                "fight_url": fight_url,
                "round": round_index,
                "time_format": _text(time_format),
                "scheduled_rounds": scheduled_rounds_from_time_format(time_format),
                "fighter_id": fighter_id,
                "fighter_url": fighter_url,
                "fighter": fighter,
                "opponent_id": opponent_id,
                "opponent_url": opponent_url,
                "opponent": opponent,
            }
            row["knockdowns"] = _integer(total_values[0][side])
            row["sig_strikes_landed"], row["sig_strikes_attempts"] = (
                _landed_attempted(total_values[1][side])
            )
            row["total_strikes_landed"], row["total_strikes_attempts"] = (
                _landed_attempted(total_values[3][side])
            )
            row["takedowns_landed"], row["takedowns_attempts"] = (
                _landed_attempted(total_values[4][side])
            )
            row["sub_attempts"] = _integer(total_values[6][side])
            row["reversals"] = _integer(total_values[7][side])
            row["control"] = _control_seconds(total_values[8][side])
            for prefix, values in zip(
                ("head", "body", "leg", "distance", "clinch", "ground"),
                significant_values[2:],
            ):
                row[f"{prefix}_strikes_landed"], row[f"{prefix}_strikes_attempts"] = (
                    _landed_attempted(values[side])
                )
            parsed.append(row)

    return pd.DataFrame(parsed)


def _side_lookup(bout_sides: pd.DataFrame) -> dict[str, Mapping[str, object]]:
    if len(bout_sides) != 2:
        raise ValueError(f"a physical fight must have exactly two sides, found {len(bout_sides)}")
    lookup: dict[str, Mapping[str, object]] = {}
    for row in bout_sides.to_dict("records"):
        fighter_id = ufcstats_identity(row.get("fighter_url"))
        fighter_url = _text(row.get("fighter_url"))
        fighter = _text(row.get("fighter"))
        for key in (fighter_id, fighter_url, fighter.casefold()):
            if key:
                lookup[key] = row
    return lookup


def _lookup_side(
    lookup: Mapping[str, Mapping[str, object]], round_row: Mapping[str, object]
) -> Mapping[str, object]:
    keys = (
        _text(round_row.get("fighter_id")),
        _text(round_row.get("fighter_url")),
        _text(round_row.get("fighter")).casefold(),
    )
    for key in keys:
        if key and key in lookup:
            return lookup[key]
    raise ValueError(
        f"per-round fighter {round_row.get('fighter')!r} could not be matched "
        "to the doubled bout rows"
    )


def _stable_round_stat_id(fight_id: str, fighter_id: str, round_number: int) -> str:
    if fight_id and fighter_id:
        return f"{fight_id}:{fighter_id}:r{round_number}"
    payload = f"{fight_id}|{fighter_id}|{round_number}".encode("utf-8")
    return "derived-" + hashlib.sha256(payload).hexdigest()[:24]


def normalize_round_stats(
    round_stats: pd.DataFrame,
    bout_sides: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich parsed round rows with stable doubled-bout metadata."""

    if round_stats.empty:
        raise ValueError("round source contained no rows")
    lookup = _side_lookup(bout_sides)
    normalized: list[dict[str, object]] = []
    for partial in round_stats.to_dict("records"):
        side = _lookup_side(lookup, partial)
        fight_url = _text(side.get("fight_url")) or _text(partial.get("fight_url"))
        fight_id = ufcstats_identity(fight_url)
        fighter_url = _text(side.get("fighter_url")) or _text(partial.get("fighter_url"))
        opponent_url = _text(side.get("opponent_url")) or _text(partial.get("opponent_url"))
        fighter_id = ufcstats_identity(fighter_url)
        opponent_id = ufcstats_identity(opponent_url)
        round_number = int(partial["round"])
        time_format = _text(partial.get("time_format")) or _text(side.get("time_format"))
        finish_round = pd.to_numeric(pd.Series([side.get("round")]), errors="coerce").iloc[0]
        finish_round_value: int | float = (
            int(finish_round) if pd.notna(finish_round) else np.nan
        )
        total_seconds = pd.to_numeric(
            pd.Series([side.get("total_fight_time")]), errors="coerce"
        ).iloc[0]
        date = pd.to_datetime(side.get("date"), errors="coerce")
        row: dict[str, object] = {
            "schema_version": ROUND_DATA_SCHEMA_VERSION,
            "round_stat_id": _stable_round_stat_id(
                fight_id, fighter_id, round_number
            ),
            "event_id": ufcstats_identity(side.get("event_url")),
            "event_url": _text(side.get("event_url")),
            "fight_id": fight_id,
            "fight_url": fight_url,
            "date": date.strftime("%Y-%m-%d") if pd.notna(date) else "",
            "source_card_index": side.get("source_card_index", np.nan),
            "bout_order": side.get("bout_order", np.nan),
            "division": _text(side.get("division")),
            "time_format": time_format,
            "scheduled_rounds": scheduled_rounds_from_time_format(time_format),
            "finish_round": finish_round_value,
            "finish_time": _text(side.get("time")),
            "total_fight_seconds": total_seconds if pd.notna(total_seconds) else np.nan,
            "round": round_number,
            "round_seconds": round_exposure_seconds(
                round_number,
                finish_round_value,
                side.get("time"),
                time_format,
            ),
            "fighter_id": fighter_id,
            "fighter_url": fighter_url,
            "fighter": _text(side.get("fighter")) or _text(partial.get("fighter")),
            "opponent_id": opponent_id,
            "opponent_url": opponent_url,
            "opponent": _text(side.get("opponent")) or _text(partial.get("opponent")),
            "result": _text(side.get("result")),
            "method": _text(side.get("method")),
            "reconciliation_status": "not_checked",
            "reconciliation_issue_count": 0,
        }
        for field in ROUND_STAT_COLUMNS:
            row[field] = partial.get(field, np.nan)
        normalized.append(row)

    result = pd.DataFrame(normalized, columns=ROUND_DATA_COLUMNS)
    validate_normalized_round_stats(result)
    return result


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _issue(
    side: Mapping[str, object],
    *,
    field: str,
    issue: str,
    bout_value: object = np.nan,
    round_value: object = np.nan,
    detail: str = "",
) -> dict[str, object]:
    bout_number = _finite_number(bout_value)
    round_number = _finite_number(round_value)
    return {
        "schema_version": ROUND_DATA_SCHEMA_VERSION,
        "event_id": str(side.get("event_id") or ""),
        "fight_id": str(side.get("fight_id") or ""),
        "fight_url": str(side.get("fight_url") or ""),
        "fighter_id": str(side.get("fighter_id") or ""),
        "fighter": str(side.get("fighter") or ""),
        "field": field,
        "issue": issue,
        "bout_value": bout_number if bout_number is not None else np.nan,
        "round_value": round_number if round_number is not None else np.nan,
        "delta": (
            round_number - bout_number
            if bout_number is not None and round_number is not None
            else np.nan
        ),
        "detail": detail,
    }


def reconcile_round_stats(
    round_stats: pd.DataFrame,
    bout_totals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate round rows and return every mismatch/unverifiable comparison.

    Round sums use ``min_count`` semantics: one missing round value makes that
    field unverifiable instead of treating the gap as zero.
    """

    validate_normalized_round_stats(round_stats)
    lookup = _side_lookup(bout_totals)
    annotated = round_stats.copy()
    issues: list[dict[str, object]] = []

    for (fight_id, fighter_id), side_rounds in annotated.groupby(
        ["fight_id", "fighter_id"], sort=False
    ):
        first = side_rounds.iloc[0].to_dict()
        source_side = _lookup_side(lookup, first)
        side_issues: list[dict[str, object]] = []
        observed_rounds = sorted(
            pd.to_numeric(side_rounds["round"], errors="coerce").dropna().astype(int)
        )
        finish_round = _finite_number(source_side.get("round"))
        if finish_round is None:
            side_issues.append(
                _issue(first, field="round", issue="finish_round_missing")
            )
        elif observed_rounds != list(range(1, int(finish_round) + 1)):
            side_issues.append(
                _issue(
                    first,
                    field="round",
                    issue="round_coverage_mismatch",
                    bout_value=finish_round,
                    round_value=len(observed_rounds),
                    detail=f"observed_rounds={observed_rounds}",
                )
            )

        exposures = pd.to_numeric(side_rounds["round_seconds"], errors="coerce")
        bout_duration = _finite_number(source_side.get("total_fight_time"))
        if exposures.isna().any():
            side_issues.append(
                _issue(
                    first,
                    field="round_seconds",
                    issue="round_exposure_missing",
                    bout_value=bout_duration,
                    detail=f"missing_rounds={side_rounds.loc[exposures.isna(), 'round'].tolist()}",
                )
            )
        else:
            exposure_sum = float(exposures.sum())
            if bout_duration is None:
                side_issues.append(
                    _issue(
                        first,
                        field="round_seconds",
                        issue="bout_duration_missing",
                        round_value=exposure_sum,
                    )
                )
            elif exposure_sum != bout_duration:
                side_issues.append(
                    _issue(
                        first,
                        field="round_seconds",
                        issue="round_duration_sum_mismatch",
                        bout_value=bout_duration,
                        round_value=exposure_sum,
                    )
                )

        for field in ROUND_STAT_COLUMNS:
            values = pd.to_numeric(side_rounds[field], errors="coerce")
            bout_value = _finite_number(source_side.get(field))
            if values.isna().any():
                side_issues.append(
                    _issue(
                        first,
                        field=field,
                        issue="round_value_missing",
                        bout_value=bout_value,
                        detail=f"missing_rounds={side_rounds.loc[values.isna(), 'round'].tolist()}",
                    )
                )
                continue
            round_value = float(values.sum())
            if bout_value is None:
                side_issues.append(
                    _issue(
                        first,
                        field=field,
                        issue="bout_value_missing",
                        round_value=round_value,
                    )
                )
            elif round_value != bout_value:
                side_issues.append(
                    _issue(
                        first,
                        field=field,
                        issue="round_sum_mismatch",
                        bout_value=bout_value,
                        round_value=round_value,
                    )
                )

        for _, round_row in side_rounds.iterrows():
            for suffix in ("landed", "attempts"):
                sig = _finite_number(round_row[f"sig_strikes_{suffix}"])
                target_parts = [
                    _finite_number(round_row[f"{target}_strikes_{suffix}"])
                    for target in ("head", "body", "leg")
                ]
                position_parts = [
                    _finite_number(round_row[f"{position}_strikes_{suffix}"])
                    for position in ("distance", "clinch", "ground")
                ]
                if sig is not None and all(value is not None for value in target_parts):
                    partition = sum(value for value in target_parts if value is not None)
                    if partition != sig:
                        side_issues.append(
                            _issue(
                                first,
                                field=f"target_sig_strikes_{suffix}",
                                issue="round_partition_mismatch",
                                bout_value=sig,
                                round_value=partition,
                                detail=f"round={int(round_row['round'])}",
                            )
                        )
                if sig is not None and all(value is not None for value in position_parts):
                    partition = sum(value for value in position_parts if value is not None)
                    if partition != sig:
                        side_issues.append(
                            _issue(
                                first,
                                field=f"position_sig_strikes_{suffix}",
                                issue="round_partition_mismatch",
                                bout_value=sig,
                                round_value=partition,
                                detail=f"round={int(round_row['round'])}",
                            )
                        )

        has_mismatch = any(
            item["issue"] in {
                "round_coverage_mismatch",
                "round_duration_sum_mismatch",
                "round_sum_mismatch",
                "round_partition_mismatch",
            }
            for item in side_issues
        )
        status = "discrepancy" if has_mismatch else ("unverifiable" if side_issues else "matched")
        mask = annotated["fight_id"].eq(fight_id) & annotated["fighter_id"].eq(
            fighter_id
        )
        annotated.loc[mask, "reconciliation_status"] = status
        annotated.loc[mask, "reconciliation_issue_count"] = len(side_issues)
        issues.extend(side_issues)

    report = pd.DataFrame(issues, columns=RECONCILIATION_COLUMNS)
    return annotated, report


def validate_normalized_round_stats(frame: pd.DataFrame) -> None:
    """Reject structural corruption while leaving source discrepancies reportable."""

    missing = set(ROUND_DATA_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"round data is missing required columns: {sorted(missing)}")
    if frame.empty:
        return
    if frame["round_stat_id"].isna().any() or frame["round_stat_id"].duplicated().any():
        raise ValueError("round_stat_id values must be present and unique")
    for identity in ("fight_id", "fighter_id", "opponent_id"):
        if frame[identity].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"{identity} must contain stable UFCStats identities")
    numeric_rounds = pd.to_numeric(frame["round"], errors="coerce")
    if numeric_rounds.isna().any() or (numeric_rounds < 1).any() or (numeric_rounds % 1 != 0).any():
        raise ValueError("round must contain positive integers")
    side_counts = frame.groupby(["fight_id", "round"], dropna=False).size()
    if not side_counts.eq(2).all():
        invalid = side_counts[~side_counts.eq(2)].head().to_dict()
        raise ValueError(f"each fight/round must have two fighter rows: {invalid}")
    if frame.duplicated(["fight_id", "fighter_id", "round"]).any():
        raise ValueError("duplicate fight/fighter/round rows were found")
    for (_fight_id, _round), pair in frame.groupby(
        ["fight_id", "round"], sort=False
    ):
        if set(pair["fighter_id"].astype(str)) != set(
            pair["opponent_id"].astype(str)
        ):
            raise ValueError("fighter/opponent identities are not mirrored within a round")
    for landed, attempted in (
        (field, field.replace("_landed", "_attempts"))
        for field in ROUND_STAT_COLUMNS
        if field.endswith("_landed")
    ):
        landed_values = pd.to_numeric(frame[landed], errors="coerce")
        attempted_values = pd.to_numeric(frame[attempted], errors="coerce")
        invalid = landed_values.notna() & attempted_values.notna() & (landed_values > attempted_values)
        if invalid.any():
            raise ValueError(f"{landed} cannot exceed {attempted}")


@dataclass(frozen=True)
class RoundBackfillSummary:
    """Bounded backfill outcome returned to callers and tests."""

    attempted_fights: int
    saved_fights: int
    failed_fights: int
    remaining_fights: int
    saved_round_rows: int
    reconciliation_issues: int
    elapsed_seconds: float = 0.0
    stopped_by_time_limit: bool = False
