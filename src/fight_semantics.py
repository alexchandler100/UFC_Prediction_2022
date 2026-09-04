"""Shared UFC bout semantics used by scraping, models, and simulation.

The repository historically inferred methods, schedules, and durations in
several places.  Keeping the rules here makes every downstream artifact expose
the same assumptions and, importantly, distinguishes source facts from an
inferred default.
"""

from __future__ import annotations

import math
import re
from typing import Mapping


SCHEDULE_CONTRACT_VERSION = "verified-pre-fight-schedule-v1"


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(value != value):  # NaN without importing pandas.
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).split())


def stable_ufcstats_id(value: object) -> str:
    """Return the durable final URL token, or an empty string."""

    text = clean_text(value).rstrip("/")
    return text.rsplit("/", 1)[-1].casefold() if text else ""


def physical_matchup_identity(
    event: object, fighter_a: object, fighter_b: object
) -> str:
    """Return an order-invariant physical-matchup identity."""

    event_id = stable_ufcstats_id(event) or clean_text(event).casefold()
    sides = sorted(
        stable_ufcstats_id(value) or clean_text(value).casefold()
        for value in (fighter_a, fighter_b)
    )
    if not event_id or not all(sides) or sides[0] == sides[1]:
        raise ValueError("event and two distinct stable fighter identities are required")
    return f"{event_id}:{sides[0]}:{sides[1]}"


def method_bucket(method: object, *, result: object | None = None) -> str:
    """Map source methods to the shared terminal-outcome vocabulary."""

    result_text = clean_text(result).upper()
    text = clean_text(method).upper()
    if result_text in {"NC", "N/C", "NO CONTEST"} or "OVERTURN" in text or text in {
        "NC",
        "N/C",
        "NO CONTEST",
    }:
        return "no_contest"
    if result_text in {"D", "DRAW"}:
        return "draw"
    if "DEC" in text:
        return "decision"
    if "KO" in text:
        return "ko_tko"
    if "SUB" in text:
        return "submission"
    return "other"


def scheduled_rounds_from_time_format(time_format: object) -> int | None:
    """Read the source-declared number of rounds without guessing."""

    text = clean_text(time_format)
    match = re.search(r"\b(\d+)\s*rnd\b", text, re.IGNORECASE)
    if match is None:
        return None
    rounds = int(match.group(1))
    overtime = re.search(r"\+\s*(?:(\d+)\s*)?ot\b", text, re.IGNORECASE)
    if overtime is not None:
        rounds += int(overtime.group(1) or 1)
    return rounds if 1 <= rounds <= 9 else None


def declared_round_lengths_seconds(time_format: object) -> tuple[int, ...]:
    """Return source-declared round lengths, expanding abbreviated suffixes."""

    scheduled = scheduled_rounds_from_time_format(time_format)
    parenthesized = re.search(r"\(([^)]*)\)", clean_text(time_format))
    if scheduled is None or parenthesized is None:
        return ()
    minutes = [int(value) for value in re.findall(r"\d+", parenthesized.group(1))]
    if not minutes or any(value <= 0 for value in minutes):
        return ()
    while len(minutes) < scheduled:
        minutes.append(minutes[-1])
    return tuple(value * 60 for value in minutes[:scheduled])


def clock_seconds(value: object) -> int | None:
    match = re.fullmatch(r"\s*(\d+):(\d{2})\s*", clean_text(value))
    if match is None or int(match.group(2)) >= 60:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def fight_duration_seconds(
    finish_round: object, finish_clock: object, time_format: object
) -> float:
    """Calculate active fight seconds from explicit round-length metadata.

    Unknown or inconsistent inputs return ``nan``.  Rest periods are never
    included in the active fight clock.
    """

    try:
        round_number = int(finish_round)
        if float(finish_round) != round_number:
            return math.nan
    except (TypeError, ValueError, OverflowError):
        return math.nan
    lengths = declared_round_lengths_seconds(time_format)
    elapsed = clock_seconds(finish_clock)
    if (
        round_number < 1
        or round_number > len(lengths)
        or elapsed is None
        or elapsed > lengths[round_number - 1]
    ):
        return math.nan
    return float(sum(lengths[: round_number - 1]) + elapsed)


def historical_schedule(
    *,
    time_format: object,
    method: object,
    total_fight_seconds: object,
    finish_round: object,
) -> tuple[int | None, str]:
    """Resolve a completed bout schedule and publish the evidence basis."""

    explicit = scheduled_rounds_from_time_format(time_format)
    if explicit is not None:
        return explicit, "explicit_time_format"
    try:
        duration = float(total_fight_seconds)
    except (TypeError, ValueError):
        duration = math.nan
    if method_bucket(method) == "decision" and math.isfinite(duration):
        inferred = duration / 300.0
        if inferred.is_integer() and 1 <= int(inferred) <= 5:
            return int(inferred), "inferred_from_decision_duration"
    try:
        final_round = int(finish_round)
    except (TypeError, ValueError, OverflowError):
        return None, "unknown"
    if final_round > 3:
        return 5, "inferred_five_round_late_finish"
    if 1 <= final_round <= 3:
        return 3, "assumed_three_round_early_finish"
    return None, "unknown"


def upcoming_schedule(bout_index: int, division: object) -> tuple[int, str]:
    """Resolve UFCStats upcoming-card schedules from currently observed data."""

    if bout_index < 0:
        raise ValueError("bout_index must be nonnegative")
    label = clean_text(division).casefold()
    if "title" in label:
        return 5, "ufcstats_title_bout_label"
    if bout_index == 0:
        return 5, "ufcstats_first_listed_main_event"
    return 3, "ufc_standard_non_main_non_title"


def schedule_from_row(row: Mapping[str, object]) -> tuple[int | None, str]:
    """Return independently declared, standard-round schedules for modeling.

    A result cannot identify the length originally scheduled for an early
    finish.  The historical display helper deliberately remains separate;
    models must not select their five-round sample using eventual results.
    """

    time_format = row.get("label_time_format", row.get("time_format"))
    rounds = scheduled_rounds_from_time_format(time_format)
    lengths = declared_round_lengths_seconds(time_format)
    if rounds is not None and 1 <= rounds <= 5 and lengths == (300,) * rounds:
        return rounds, "explicit_time_format"
    return None, "unknown"
