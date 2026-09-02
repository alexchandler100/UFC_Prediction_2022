"""Build the compact, ID-based fighter data publication used by the website."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping

import pandas as pd

from fight_semantics import method_bucket


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "content" / "data" / "processed" / "ufc_fights_reported_doubled.csv"
FIGHTER_PATH = ROOT / "content" / "data" / "processed" / "fighter_stats.csv"
OUTPUT_PATH = ROOT / "content" / "data" / "external" / "fighter_explorer.json"
VEGAS_PATH = ROOT / "content" / "data" / "external" / "vegas_odds.json"
EXTERNAL_MMA_ROOT = ROOT / "content" / "data" / "external_mma"
EXTERNAL_BOUTS_PATH = EXTERNAL_MMA_ROOT / "bouts.jsonl"
EXTERNAL_IDENTITY_PATH = EXTERNAL_MMA_ROOT / "identity_map.csv"
EXTERNAL_SUPPLEMENTS_PATH = EXTERNAL_MMA_ROOT / "fighter_history_supplements.jsonl"
SCHEMA_VERSION = 3
SIZE_LIMIT = 12 * 1024 * 1024
SHARD_SIZE_LIMIT = 4 * 1024 * 1024
SHARD_KEYS = tuple("0123456789abcdefx")

STAT_FIELDS = (
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

FIGHT_COLUMNS = (
    "date",
    "fight_id",
    "fight_url",
    "event_id",
    "event_url",
    "event_name",
    "promotion",
    "source",
    "source_label",
    "source_url",
    "stats_available",
    "result",
    "opponent_id",
    "opponent_name",
    "opponent_url",
    "division",
    "method",
    "round",
    "time",
    "total_fight_time",
    "source_card_index",
    "bout_order",
    "time_format",
    *STAT_FIELDS,
)

STAT_DEFINITIONS = {
    "knockdowns": ("Knockdowns", "Striking", "count"),
    "sig_strikes_landed": ("Significant strikes landed", "Striking", "count"),
    "sig_strikes_attempts": ("Significant strikes attempted", "Striking", "count"),
    "total_strikes_landed": ("Total strikes landed", "Striking", "count"),
    "total_strikes_attempts": ("Total strikes attempted", "Striking", "count"),
    "takedowns_landed": ("Takedowns landed", "Grappling", "count"),
    "takedowns_attempts": ("Takedowns attempted", "Grappling", "count"),
    "sub_attempts": ("Submission attempts", "Grappling", "count"),
    "reversals": ("Reversals", "Grappling", "count"),
    "control": ("Control time", "Grappling", "seconds"),
    "head_strikes_landed": ("Head strikes landed", "Targets", "count"),
    "head_strikes_attempts": ("Head strikes attempted", "Targets", "count"),
    "body_strikes_landed": ("Body strikes landed", "Targets", "count"),
    "body_strikes_attempts": ("Body strikes attempted", "Targets", "count"),
    "leg_strikes_landed": ("Leg strikes landed", "Targets", "count"),
    "leg_strikes_attempts": ("Leg strikes attempted", "Targets", "count"),
    "distance_strikes_landed": ("Distance strikes landed", "Positions", "count"),
    "distance_strikes_attempts": ("Distance strikes attempted", "Positions", "count"),
    "clinch_strikes_landed": ("Clinch strikes landed", "Positions", "count"),
    "clinch_strikes_attempts": ("Clinch strikes attempted", "Positions", "count"),
    "ground_strikes_landed": ("Ground strikes landed", "Positions", "count"),
    "ground_strikes_attempts": ("Ground strikes attempted", "Positions", "count"),
}

CAREER_DEFINITIONS = {
    "recorded_bouts": ("UFCStats bouts", "Record", "count", "higher"),
    "wins": ("Wins", "Record", "count", "higher"),
    "losses": ("Losses", "Record", "count", "lower"),
    "draws": ("Draws", "Record", "count", "context"),
    "no_contests": ("No contests", "Record", "count", "context"),
    "win_rate": ("Win rate", "Record", "percentage", "higher"),
    "finish_wins": ("Finish wins", "Record", "count", "context"),
    "finish_rate": ("Finish rate in wins", "Record", "percentage", "higher"),
    "ko_tko_wins": ("KO/TKO wins", "Record", "count", "context"),
    "submission_wins": ("Submission wins", "Record", "count", "context"),
    "decision_wins": ("Decision wins", "Record", "count", "context"),
    "other_wins": ("Other finish wins", "Record", "count", "context"),
    "total_fight_minutes": ("Total fight time", "Record", "minutes", "context"),
    "average_fight_minutes": ("Average fight time", "Record", "minutes", "context"),
    "bouts_with_duration": ("Bouts with known duration", "Data quality", "count", "higher"),
    "sig_strikes_landed_per_minute": ("Sig. strikes landed / min", "Striking", "decimal", "higher"),
    "sig_strikes_absorbed_per_minute": ("Sig. strikes absorbed / min", "Striking", "decimal", "lower"),
    "significant_strike_differential_per_minute": ("Sig. strike differential / min", "Striking", "decimal", "higher"),
    "sig_strike_accuracy": ("Sig. strike accuracy", "Striking", "percentage", "higher"),
    "sig_strike_defense": ("Sig. strike defense", "Striking", "percentage", "higher"),
    "knockdowns_per_15": ("Knockdowns / 15 min", "Striking", "decimal", "higher"),
    "knockdowns_absorbed_per_15": ("Knockdowns absorbed / 15 min", "Striking", "decimal", "lower"),
    "takedowns_landed_per_15": ("Takedowns / 15 min", "Grappling", "decimal", "higher"),
    "takedown_accuracy": ("Takedown accuracy", "Grappling", "percentage", "higher"),
    "takedown_defense": ("Takedown defense", "Grappling", "percentage", "higher"),
    "submission_attempts_per_15": ("Submission attempts / 15 min", "Grappling", "decimal", "higher"),
    "control_minutes_per_15": ("Control minutes / 15 min", "Grappling", "decimal", "higher"),
    "control_share": ("Share of recorded control time", "Grappling", "percentage", "higher"),
    "head_strike_share": ("Head strike share", "Style", "percentage", "context"),
    "body_strike_share": ("Body strike share", "Style", "percentage", "context"),
    "leg_strike_share": ("Leg strike share", "Style", "percentage", "context"),
    "distance_strike_share": ("Distance strike share", "Style", "percentage", "context"),
    "clinch_strike_share": ("Clinch strike share", "Style", "percentage", "context"),
    "ground_strike_share": ("Ground strike share", "Style", "percentage", "context"),
    "paired_opponent_stat_bouts": ("Bouts with paired opponent stats", "Data quality", "count", "higher"),
    "control_stat_bouts": ("Bouts supporting control rate", "Data quality", "count", "higher"),
    "control_share_stat_bouts": ("Bouts supporting control share", "Data quality", "count", "higher"),
}

SOURCE_LABELS = {
    "ufcstats": "UFCStats",
    "kaggle_pro_mma_fights_v1": "All Pro MMA Fights v1 (CC0)",
    "wikipedia_cc_by_sa_v4": "Wikipedia record supplement (CC BY-SA 4.0)",
}

# The CC0 source uses this fighter's older Sherdog display name. Keep the
# stable source identity while publishing the name users actually search for.
EXTERNAL_DISPLAY_NAMES = {
    ("kaggle_pro_mma_fights_v1", "/fighter/Nong-Stamp-292745"): "Stamp Fairtex",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_hash(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _stable_id(value: object) -> str:
    text = _clean_text(value).rstrip("/")
    token = text.rsplit("/", 1)[-1]
    if not token or any(character.isspace() for character in token):
        raise ValueError(f"invalid stable UFCStats identifier: {value!r}")
    return token


def _number(value: object) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return round(number, 6)


def _inches(value: object) -> int | None:
    text = _clean_text(value)
    feet_match = re.fullmatch(r"(\d+)\s*'\s*(\d+)\s*\"?", text)
    if feet_match:
        return int(feet_match.group(1)) * 12 + int(feet_match.group(2))
    inch_match = re.fullmatch(r"(\d+)\s*\"?", text)
    return int(inch_match.group(1)) if inch_match else None


def _iso_date(value: object) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _rate(total: float, seconds: float, scale_seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return round(total * scale_seconds / seconds, 6)


def _method_bucket(method: object) -> str:
    return method_bucket(method)


def _shard_key(fighter_id: str) -> str:
    normalized = fighter_id.lower()
    first = (
        normalized[len("external_"):len("external_") + 1]
        if normalized.startswith("external_")
        else normalized[:1]
    )
    return first if first in SHARD_KEYS[:-1] else "x"


def _fight_array(row: pd.Series) -> list[object]:
    fight_url = _clean_text(row.get("fight_url"))
    event_url = _clean_text(row.get("event_url"))
    source = _clean_text(row.get("source")) or "ufcstats"
    values: dict[str, object] = {
        "date": _iso_date(row.get("date")),
        "fight_id": _clean_text(row.get("fight_id")) or _stable_id(fight_url),
        "fight_url": fight_url,
        "event_id": _clean_text(row.get("event_id")) or _stable_id(event_url),
        "event_url": event_url,
        "event_name": _clean_text(row.get("event_name")),
        "promotion": _clean_text(row.get("promotion")) or "UFC",
        "source": source,
        "source_label": (
            _clean_text(row.get("source_label"))
            or SOURCE_LABELS.get(source, source)
        ),
        "source_url": _clean_text(row.get("source_url")) or fight_url,
        "stats_available": (
            bool(row.get("stats_available"))
            if row.get("stats_available") is not None
            else True
        ),
        "result": _clean_text(row.get("result")),
        "opponent_id": _stable_id(row.get("opponent_url")),
        "opponent_name": _clean_text(row.get("opponent")),
        "opponent_url": _clean_text(row.get("opponent_url")),
        "division": _clean_text(row.get("division")),
        "method": _clean_text(row.get("method")),
        "round": _number(row.get("round")),
        "time": _clean_text(row.get("time")),
        "total_fight_time": _number(row.get("total_fight_time")),
        "source_card_index": _number(row.get("source_card_index")),
        "bout_order": _number(row.get("bout_order")),
        "time_format": _clean_text(row.get("time_format")),
    }
    for field in STAT_FIELDS:
        values[field] = _number(row.get(field))
    return [values[column] for column in FIGHT_COLUMNS]


def _record_summary(rows: list[pd.Series]) -> dict[str, object]:
    """Summarize every linked result without implying detailed stat coverage."""

    ordered = sorted(
        rows,
        key=lambda row: (
            _iso_date(row.get("date")) or "",
            float(_number(row.get("bout_order")) or -1),
        ),
        reverse=True,
    )
    results = [_clean_text(row.get("result")).upper() for row in ordered]
    wins = results.count("W")
    losses = results.count("L")
    draws = results.count("D")
    no_contests = len(results) - wins - losses - draws
    win_methods = Counter(
        _method_bucket(row.get("method"))
        for row in ordered
        if _clean_text(row.get("result")).upper() == "W"
    )
    finish_wins = win_methods["ko_tko"] + win_methods["submission"] + win_methods["other"]
    promotions = Counter(_clean_text(row.get("promotion")) or "UFC" for row in ordered)
    known_seconds = [
        float(value)
        for row in ordered
        if (value := _number(row.get("total_fight_time"))) is not None
    ]
    current_streak_result = results[0] if results and results[0] in {"W", "L"} else None
    current_streak = 0
    if current_streak_result:
        for result in results:
            if result != current_streak_result:
                break
            current_streak += 1
    return {
        "recorded_bouts": len(ordered),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "no_contests": no_contests,
        "win_rate": _ratio(wins, wins + losses + draws),
        "finish_wins": finish_wins,
        "finish_rate": _ratio(finish_wins, wins),
        "ko_tko_wins": win_methods["ko_tko"],
        "submission_wins": win_methods["submission"],
        "decision_wins": win_methods["decision"],
        "other_wins": win_methods["other"],
        "recent_form": results[:5],
        "current_streak_result": current_streak_result,
        "current_streak": current_streak,
        "last_fight_date": _iso_date(ordered[0].get("date")) if ordered else None,
        "first_fight_date": _iso_date(ordered[-1].get("date")) if ordered else None,
        "promotions": [
            {"name": name, "bouts": count}
            for name, count in sorted(promotions.items(), key=lambda item: (-item[1], item[0]))
        ],
        "detailed_stat_bouts": sum(bool(row.get("stats_available", True)) for row in ordered),
        "metadata_only_bouts": sum(not bool(row.get("stats_available", True)) for row in ordered),
        "bouts_with_duration": len(known_seconds),
        "known_fight_minutes": round(sum(known_seconds) / 60.0, 4),
    }


def _external_value(observation: object, key: str) -> object:
    if isinstance(observation, Mapping):
        return observation.get(key)
    return getattr(observation, key)


def _external_fighter_id(source: str, source_fighter_id: str) -> str:
    digest = sha256(f"{source}\0{source_fighter_id}".encode("utf-8")).hexdigest()[:24]
    return f"external_{digest}"


def _external_display_name(source: str, source_fighter_id: str, name: object) -> str:
    return EXTERNAL_DISPLAY_NAMES.get(
        (source, source_fighter_id), _clean_text(name)
    )


def _is_ufc_promotion(value: object) -> bool:
    text = re.sub(r"[^a-z0-9]+", " ", _clean_text(value).casefold()).strip()
    return text == "ufc" or "ultimate fighting championship" in text


def _external_history_rows(
    observations: Iterable[object],
    identity_map: Mapping[tuple[str, str], str],
) -> tuple[list[pd.Series], int, int]:
    """Return non-UFC history, while counting rows linked to UFCStats profiles."""

    output: list[pd.Series] = []
    linked_observations = 0
    linked_perspectives = 0
    for observation in observations:
        if _is_ufc_promotion(_external_value(observation, "promotion")):
            continue
        source = _clean_text(_external_value(observation, "source"))
        first_source_id = _clean_text(_external_value(observation, "fighter_source_id"))
        second_source_id = _clean_text(_external_value(observation, "opponent_source_id"))
        first_id = identity_map.get((source, first_source_id))
        second_id = identity_map.get((source, second_source_id))
        if first_id and second_id and first_id == second_id:
            raise ValueError(
                "external identity mapping collapses both participants in "
                f"{_external_value(observation, 'observation_id')}"
            )
        if first_id or second_id:
            linked_observations += 1
        observation_id = _clean_text(_external_value(observation, "observation_id"))
        event_seed = (
            f"{source}\0{_clean_text(_external_value(observation, 'source_event_id'))}"
        )
        event_id = f"external_{sha256(event_seed.encode('utf-8')).hexdigest()[:32]}"
        fight_id = f"external_{observation_id[:32]}"
        source_url = _clean_text(_external_value(observation, "source_url"))
        finish_seconds = _number(_external_value(observation, "finish_clock_seconds"))
        finish_clock = (
            ""
            if finish_seconds is None
            else f"{int(finish_seconds) // 60}:{int(finish_seconds) % 60:02d}"
        )
        scheduled_rounds = _number(_external_value(observation, "scheduled_rounds"))
        common = {
            "date": _external_value(observation, "event_date"),
            "fight_id": fight_id,
            "fight_url": source_url,
            "event_id": event_id,
            "event_url": source_url,
            "event_name": _external_value(observation, "event_name"),
            "promotion": _external_value(observation, "promotion"),
            "source": source,
            "source_label": SOURCE_LABELS.get(source, source),
            "source_url": source_url,
            "stats_available": False,
            "division": _external_value(observation, "division"),
            "method": _external_value(observation, "method"),
            "round": _external_value(observation, "finish_round"),
            "time": finish_clock,
            "total_fight_time": None,
            "source_card_index": None,
            "bout_order": _external_value(observation, "source_bout_order"),
            "time_format": (
                f"{int(scheduled_rounds)} scheduled rounds"
                if scheduled_rounds is not None
                else ""
            ),
            **dict.fromkeys(STAT_FIELDS, None),
        }
        participants = (
            (
                first_id,
                first_source_id,
                _external_display_name(
                    source,
                    first_source_id,
                    _external_value(observation, "fighter_name"),
                ),
                second_id,
                second_source_id,
                _external_display_name(
                    source,
                    second_source_id,
                    _external_value(observation, "opponent_name"),
                ),
                _clean_text(_external_value(observation, "result")).upper(),
            ),
            (
                second_id,
                second_source_id,
                _external_display_name(
                    source,
                    second_source_id,
                    _external_value(observation, "opponent_name"),
                ),
                first_id,
                first_source_id,
                _external_display_name(
                    source,
                    first_source_id,
                    _external_value(observation, "fighter_name"),
                ),
                {"W": "L", "L": "W", "D": "D", "NC": "NC"}.get(
                    _clean_text(_external_value(observation, "result")).upper(), "NC"
                ),
            ),
        )
        for fighter_id, fighter_source_id, fighter_name, opponent_id, opponent_source_id, opponent_name, result in participants:
            mapped_fighter = bool(fighter_id)
            resolved_fighter_id = fighter_id or _external_fighter_id(
                source, fighter_source_id
            )
            resolved_opponent_id = opponent_id or _external_fighter_id(
                source, opponent_source_id
            )
            linked_perspectives += int(mapped_fighter)
            output.append(
                pd.Series(
                    {
                        **common,
                        "fighter": fighter_name,
                        "opponent": opponent_name,
                        "fighter_url": (
                            "http://ufcstats.com/fighter-details/"
                            f"{resolved_fighter_id}"
                            if mapped_fighter
                            else "https://external-mma.invalid/fighter-details/"
                            f"{resolved_fighter_id}"
                        ),
                        "opponent_url": (
                            f"http://ufcstats.com/fighter-details/{resolved_opponent_id}"
                            if opponent_id
                            else f"https://external-mma.invalid/fighter-details/{resolved_opponent_id}"
                        ),
                        "result": result,
                    }
                )
            )
    return output, linked_observations, linked_perspectives


def _supplement_history_rows(
    supplements: Iterable[object],
    identity_map: Mapping[tuple[str, str], str],
) -> list[pd.Series]:
    """Convert reviewed, source-attributed gap fills into fighter perspectives."""

    required = {
        "source",
        "source_bout_id",
        "source_event_id",
        "source_url",
        "event_date",
        "event_name",
        "promotion",
        "fighter_profile_source",
        "fighter_source_id",
        "fighter_name",
        "opponent_profile_source",
        "opponent_source_id",
        "opponent_name",
        "result",
        "method",
    }
    output: list[pd.Series] = []
    seen_bouts: set[tuple[str, str]] = set()
    for supplement in supplements:
        missing = sorted(
            field
            for field in required
            if not _clean_text(_external_value(supplement, field))
        )
        if missing:
            raise ValueError(f"external history supplement has blank fields: {missing}")
        source = _clean_text(_external_value(supplement, "source"))
        source_bout_id = _clean_text(_external_value(supplement, "source_bout_id"))
        bout_key = (source, source_bout_id)
        if bout_key in seen_bouts:
            raise ValueError(f"duplicate external history supplement bout: {bout_key}")
        seen_bouts.add(bout_key)
        event_date = _iso_date(_external_value(supplement, "event_date"))
        if event_date is None:
            raise ValueError(f"external history supplement has invalid date: {bout_key}")
        result = _clean_text(_external_value(supplement, "result")).upper()
        if result not in {"W", "L", "D", "NC"}:
            raise ValueError(f"external history supplement has invalid result: {bout_key}")

        source_url = _clean_text(_external_value(supplement, "source_url"))
        event_source_id = _clean_text(_external_value(supplement, "source_event_id"))
        promotion = _clean_text(_external_value(supplement, "promotion"))
        if re.sub(r"[^a-z0-9]+", " ", promotion.casefold()).strip() == "one championship":
            promotion = "One Championship"
        fight_seed = f"{source}\0{source_bout_id}"
        event_seed = f"{source}\0{event_source_id}"
        fight_id = f"external_{sha256(fight_seed.encode('utf-8')).hexdigest()[:32]}"
        event_id = f"external_{sha256(event_seed.encode('utf-8')).hexdigest()[:32]}"
        finish_seconds = _number(_external_value(supplement, "finish_clock_seconds"))
        finish_clock = (
            ""
            if finish_seconds is None
            else f"{int(finish_seconds) // 60}:{int(finish_seconds) % 60:02d}"
        )
        scheduled_rounds = _number(_external_value(supplement, "scheduled_rounds"))
        common = {
            "date": event_date,
            "fight_id": fight_id,
            "fight_url": source_url,
            "event_id": event_id,
            "event_url": source_url,
            "event_name": _external_value(supplement, "event_name"),
            "promotion": promotion,
            "source": source,
            "source_label": SOURCE_LABELS.get(source, source),
            "source_url": source_url,
            "stats_available": False,
            "division": _external_value(supplement, "division"),
            "method": _external_value(supplement, "method"),
            "round": _external_value(supplement, "finish_round"),
            "time": finish_clock,
            "total_fight_time": None,
            "source_card_index": None,
            "bout_order": _external_value(supplement, "source_bout_order"),
            "time_format": (
                f"{int(scheduled_rounds)} scheduled rounds"
                if scheduled_rounds is not None
                else ""
            ),
            **dict.fromkeys(STAT_FIELDS, None),
        }
        participants = (
            (
                _clean_text(_external_value(supplement, "fighter_profile_source")),
                _clean_text(_external_value(supplement, "fighter_source_id")),
                _clean_text(_external_value(supplement, "fighter_name")),
                _clean_text(_external_value(supplement, "opponent_profile_source")),
                _clean_text(_external_value(supplement, "opponent_source_id")),
                _clean_text(_external_value(supplement, "opponent_name")),
                result,
            ),
            (
                _clean_text(_external_value(supplement, "opponent_profile_source")),
                _clean_text(_external_value(supplement, "opponent_source_id")),
                _clean_text(_external_value(supplement, "opponent_name")),
                _clean_text(_external_value(supplement, "fighter_profile_source")),
                _clean_text(_external_value(supplement, "fighter_source_id")),
                _clean_text(_external_value(supplement, "fighter_name")),
                {"W": "L", "L": "W", "D": "D", "NC": "NC"}[result],
            ),
        )
        for (
            profile_source,
            fighter_source_id,
            fighter_name,
            opponent_profile_source,
            opponent_source_id,
            opponent_name,
            perspective_result,
        ) in participants:
            mapped_fighter_id = identity_map.get((profile_source, fighter_source_id))
            mapped_opponent_id = identity_map.get(
                (opponent_profile_source, opponent_source_id)
            )
            fighter_id = mapped_fighter_id or _external_fighter_id(
                profile_source, fighter_source_id
            )
            opponent_id = mapped_opponent_id or _external_fighter_id(
                opponent_profile_source, opponent_source_id
            )
            output.append(
                pd.Series(
                    {
                        **common,
                        "fighter": fighter_name,
                        "opponent": opponent_name,
                        "fighter_url": (
                            f"http://ufcstats.com/fighter-details/{fighter_id}"
                            if mapped_fighter_id
                            else f"https://external-mma.invalid/fighter-details/{fighter_id}"
                        ),
                        "opponent_url": (
                            f"http://ufcstats.com/fighter-details/{opponent_id}"
                            if mapped_opponent_id
                            else f"https://external-mma.invalid/fighter-details/{opponent_id}"
                        ),
                        "result": perspective_result,
                    }
                )
            )
    return output


def _sum_stats(rows: Iterable[pd.Series]) -> dict[str, float | None]:
    """Sum observed values without representing wholly missing fields as zero."""

    observed = {field: [] for field in STAT_FIELDS}
    for row in rows:
        for field in STAT_FIELDS:
            value = _number(row.get(field))
            if value is not None:
                observed[field].append(float(value))
    return {
        field: (sum(values) if values else None)
        for field, values in observed.items()
    }


def _field_rate(
    rows: Iterable[pd.Series],
    field: str,
    scale_seconds: float,
) -> float | None:
    """Return a rate using only bouts with both the statistic and exposure."""

    total = 0.0
    seconds = 0.0
    for row in rows:
        value = _number(row.get(field))
        duration = _number(row.get("total_fight_time"))
        if value is None or duration is None or duration <= 0:
            continue
        total += float(value)
        seconds += float(duration)
    return _rate(total, seconds, scale_seconds)


def _field_ratio(
    rows: Iterable[pd.Series],
    numerator_field: str,
    denominator_field: str,
) -> float | None:
    """Return a ratio from rows where both components were observed."""

    numerator = 0.0
    denominator = 0.0
    observed = False
    for row in rows:
        numerator_value = _number(row.get(numerator_field))
        denominator_value = _number(row.get(denominator_field))
        if numerator_value is None or denominator_value is None:
            continue
        observed = True
        numerator += float(numerator_value)
        denominator += float(denominator_value)
    return _ratio(numerator, denominator) if observed else None


def _fight_key(row: pd.Series) -> str:
    fight_id = _clean_text(row.get("fight_id"))
    return fight_id or _stable_id(row.get("fight_url"))


def _paired_rate(
    rows: Iterable[pd.Series],
    opponent_rows: Iterable[pd.Series],
    field: str,
    scale_seconds: float,
    *,
    differential: bool = False,
) -> float | None:
    """Return an absorbed or differential rate from complete fight pairs."""

    opponents = {_fight_key(row): row for row in opponent_rows}
    total = 0.0
    seconds = 0.0
    for row in rows:
        opponent = opponents.get(_fight_key(row))
        own_value = _number(row.get(field))
        opponent_value = _number(opponent.get(field)) if opponent is not None else None
        duration = _number(row.get("total_fight_time"))
        if opponent_value is None or duration is None or duration <= 0:
            continue
        if differential and own_value is None:
            continue
        total += (
            float(own_value) - float(opponent_value)
            if differential
            else float(opponent_value)
        )
        seconds += float(duration)
    return _rate(total, seconds, scale_seconds)


def _paired_control_share(
    rows: Iterable[pd.Series], opponent_rows: Iterable[pd.Series]
) -> float | None:
    opponents = {_fight_key(row): row for row in opponent_rows}
    own_total = 0.0
    combined_total = 0.0
    observed = False
    for row in rows:
        opponent = opponents.get(_fight_key(row))
        own_control = _number(row.get("control"))
        opponent_control = (
            _number(opponent.get("control")) if opponent is not None else None
        )
        if own_control is None or opponent_control is None:
            continue
        observed = True
        own_total += float(own_control)
        combined_total += float(own_control) + float(opponent_control)
    return _ratio(own_total, combined_total) if observed else None


def _published_total(value: float | None) -> int | float | None:
    if value is None:
        return None
    return int(value) if value.is_integer() else round(value, 6)


def _career(
    rows: list[pd.Series],
    opponent_rows: list[pd.Series],
) -> dict[str, object]:
    ordered = sorted(
        rows,
        key=lambda row: (
            _iso_date(row.get("date")) or "",
            float(_number(row.get("bout_order")) or -1),
        ),
        reverse=True,
    )
    results = [_clean_text(row.get("result")).upper() for row in ordered]
    wins = results.count("W")
    losses = results.count("L")
    draws = results.count("D")
    no_contests = len(results) - wins - losses - draws
    win_methods = Counter(
        _method_bucket(row.get("method"))
        for row in ordered
        if _clean_text(row.get("result")).upper() == "W"
    )
    totals = _sum_stats(ordered)
    opponent_totals = _sum_stats(opponent_rows)
    opponents_by_fight = {_fight_key(row): row for row in opponent_rows}
    known_durations = [
        float(duration)
        for row in ordered
        if (duration := _number(row.get("total_fight_time"))) is not None
        and duration > 0
    ]
    total_seconds = sum(known_durations)
    finish_wins = win_methods["ko_tko"] + win_methods["submission"] + win_methods["other"]
    divisions = Counter(_clean_text(row.get("division")) for row in ordered)
    divisions.pop("", None)
    primary_division = (
        sorted(divisions.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if divisions
        else None
    )
    current_streak_result = results[0] if results and results[0] in {"W", "L"} else None
    current_streak = 0
    if current_streak_result:
        for result in results:
            if result != current_streak_result:
                break
            current_streak += 1

    return {
        "recorded_bouts": len(ordered),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "no_contests": no_contests,
        "win_rate": _ratio(wins, wins + losses + draws),
        "finish_wins": finish_wins,
        "finish_rate": _ratio(finish_wins, wins),
        "ko_tko_wins": win_methods["ko_tko"],
        "submission_wins": win_methods["submission"],
        "decision_wins": win_methods["decision"],
        "other_wins": win_methods["other"],
        "total_fight_minutes": round(total_seconds / 60.0, 4),
        "average_fight_minutes": _ratio(
            total_seconds / 60.0, len(known_durations)
        ),
        "bouts_with_duration": len(known_durations),
        "sig_strikes_landed_per_minute": _field_rate(
            ordered, "sig_strikes_landed", 60.0
        ),
        "sig_strikes_absorbed_per_minute": _paired_rate(
            ordered, opponent_rows, "sig_strikes_landed", 60.0
        ),
        "significant_strike_differential_per_minute": _paired_rate(
            ordered,
            opponent_rows,
            "sig_strikes_landed",
            60.0,
            differential=True,
        ),
        "sig_strike_accuracy": _field_ratio(
            ordered, "sig_strikes_landed", "sig_strikes_attempts"
        ),
        "sig_strike_defense": (
            None
            if (absorbed_accuracy := _field_ratio(
                opponent_rows, "sig_strikes_landed", "sig_strikes_attempts"
            )) is None
            else round(1.0 - absorbed_accuracy, 6)
        ),
        "knockdowns_per_15": _field_rate(ordered, "knockdowns", 900.0),
        "knockdowns_absorbed_per_15": _paired_rate(
            ordered, opponent_rows, "knockdowns", 900.0
        ),
        "takedowns_landed_per_15": _field_rate(
            ordered, "takedowns_landed", 900.0
        ),
        "takedown_accuracy": _field_ratio(
            ordered, "takedowns_landed", "takedowns_attempts"
        ),
        "takedown_defense": (
            None
            if (absorbed_takedown_accuracy := _field_ratio(
                opponent_rows, "takedowns_landed", "takedowns_attempts"
            )) is None
            else round(1.0 - absorbed_takedown_accuracy, 6)
        ),
        "submission_attempts_per_15": _field_rate(
            ordered, "sub_attempts", 900.0
        ),
        "control_minutes_per_15": _field_rate(
            ordered, "control", 15.0
        ),
        "control_share": _paired_control_share(ordered, opponent_rows),
        "head_strike_share": _field_ratio(
            ordered, "head_strikes_landed", "sig_strikes_landed"
        ),
        "body_strike_share": _field_ratio(
            ordered, "body_strikes_landed", "sig_strikes_landed"
        ),
        "leg_strike_share": _field_ratio(
            ordered, "leg_strikes_landed", "sig_strikes_landed"
        ),
        "distance_strike_share": _field_ratio(
            ordered, "distance_strikes_landed", "sig_strikes_landed"
        ),
        "clinch_strike_share": _field_ratio(
            ordered, "clinch_strikes_landed", "sig_strikes_landed"
        ),
        "ground_strike_share": _field_ratio(
            ordered, "ground_strikes_landed", "sig_strikes_landed"
        ),
        "recent_form": results[:5],
        "current_streak_result": current_streak_result,
        "current_streak": current_streak,
        "last_fight_date": _iso_date(ordered[0].get("date")) if ordered else None,
        "first_fight_date": _iso_date(ordered[-1].get("date")) if ordered else None,
        "primary_division": primary_division,
        "divisions": [
            {"name": name, "bouts": count}
            for name, count in sorted(
                divisions.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "paired_opponent_stat_bouts": len(opponent_rows),
        "control_stat_bouts": sum(
            _number(row.get("control")) is not None
            and (_number(row.get("total_fight_time")) or 0) > 0
            for row in ordered
        ),
        "control_share_stat_bouts": sum(
            _number(row.get("control")) is not None
            and _number(opponent.get("control")) is not None
            for row, opponent in (
                (row, opponents_by_fight.get(_fight_key(row)))
                for row in ordered
            )
            if opponent is not None
        ),
        "totals": {
            key: _published_total(value)
            for key, value in totals.items()
        },
        "opponent_totals": {
            key: _published_total(value)
            for key, value in opponent_totals.items()
        },
    }


def build_fighter_explorer(
    raw_fights: pd.DataFrame,
    fighter_stats: pd.DataFrame,
    upcoming_fighters: pd.DataFrame | None = None,
    external_bouts: Iterable[object] | None = None,
    identity_map: Mapping[tuple[str, str], str] | None = None,
    external_supplements: Iterable[object] | None = None,
) -> dict[str, object]:
    required_raw = {
        "date",
        "fight_url",
        "event_url",
        "fighter",
        "opponent",
        "fighter_url",
        "opponent_url",
        "result",
        "division",
        "method",
        "round",
        "time",
        "total_fight_time",
        "source_card_index",
        "bout_order",
        "time_format",
        *STAT_FIELDS,
    }
    required_fighters = {"name", "height", "reach", "stance", "dob", "url"}
    missing_raw = sorted(required_raw - set(raw_fights.columns))
    missing_fighters = sorted(required_fighters - set(fighter_stats.columns))
    if missing_raw or missing_fighters:
        raise ValueError(
            f"fighter explorer inputs are missing columns; raw={missing_raw}, "
            f"fighters={missing_fighters}"
        )

    profiles: dict[str, dict[str, object]] = {}
    for _, row in fighter_stats.iterrows():
        fighter_id = _stable_id(row["url"])
        profile = {
            "id": fighter_id,
            "name": _clean_text(row["name"]),
            "url": _clean_text(row["url"]),
            "profile_scope": "ufcstats",
            "height": _clean_text(row["height"]),
            "height_inches": _inches(row["height"]),
            "reach": _clean_text(row["reach"]),
            "reach_inches": _inches(row["reach"]),
            "stance": _clean_text(row["stance"]),
            "dob": _clean_text(row["dob"]),
            "dob_iso": _iso_date(row["dob"]),
            "scheduled_division": "",
        }
        existing = profiles.get(fighter_id)
        if existing is not None and existing != profile:
            raise ValueError(f"conflicting fighter bio rows for {fighter_id}")
        profiles[fighter_id] = profile

    ufc_rows_by_fighter: dict[str, list[pd.Series]] = defaultdict(list)
    rows_by_fighter: dict[str, list[pd.Series]] = defaultdict(list)
    row_by_fight_and_fighter: dict[tuple[str, str], pd.Series] = {}
    for _, row in raw_fights.iterrows():
        fighter_id = _stable_id(row["fighter_url"])
        opponent_id = _stable_id(row["opponent_url"])
        fight_id = _stable_id(row["fight_url"])
        key = (fight_id, fighter_id)
        if key in row_by_fight_and_fighter:
            raise ValueError(f"duplicate fighter perspective for fight {fight_id}")
        row_by_fight_and_fighter[key] = row
        ufc_rows_by_fighter[fighter_id].append(row)
        rows_by_fighter[fighter_id].append(row)
        for identity, name, url in (
            (fighter_id, row["fighter"], row["fighter_url"]),
            (opponent_id, row["opponent"], row["opponent_url"]),
        ):
            profiles.setdefault(
                identity,
                {
                    "id": identity,
                    "name": _clean_text(name),
                    "url": _clean_text(url),
                    "profile_scope": "ufcstats",
                    "height": "",
                    "height_inches": None,
                    "reach": "",
                    "reach_inches": None,
                    "stance": "",
                    "dob": "",
                    "dob_iso": None,
                    "scheduled_division": "",
                },
            )

    external_rows, linked_external_fights, linked_external_fighter_rows = _external_history_rows(
        external_bouts or [], identity_map or {}
    )
    supplement_rows = _supplement_history_rows(
        external_supplements or [], identity_map or {}
    )
    external_rows.extend(supplement_rows)
    for row in external_rows:
        fighter_id = _stable_id(row["fighter_url"])
        opponent_id = _stable_id(row["opponent_url"])
        fight_id = _clean_text(row.get("fight_id"))
        key = (fight_id, fighter_id)
        if key in row_by_fight_and_fighter:
            raise ValueError(f"duplicate fighter perspective for fight {fight_id}")
        row_by_fight_and_fighter[key] = row
        rows_by_fighter[fighter_id].append(row)
        profiles.setdefault(
            fighter_id,
            {
                "id": fighter_id,
                "name": _clean_text(row.get("fighter")),
                "url": _clean_text(row.get("fighter_url")),
                "profile_scope": "external_result_metadata",
                "height": "",
                "height_inches": None,
                "reach": "",
                "reach_inches": None,
                "stance": "",
                "dob": "",
                "dob_iso": None,
                "scheduled_division": "",
            },
        )

    scheduled_ids: set[str] = set()
    if upcoming_fighters is not None and not upcoming_fighters.empty:
        required_upcoming = {
            "fighter name",
            "opponent name",
            "fighter id",
            "opponent id",
            "division",
        }
        missing_upcoming = sorted(required_upcoming - set(upcoming_fighters.columns))
        if missing_upcoming:
            raise ValueError(
                "fighter explorer upcoming rows are missing columns: "
                f"{missing_upcoming}"
            )
        for _, row in upcoming_fighters.iterrows():
            division = _clean_text(row.get("division"))
            for id_column, name_column in (
                ("fighter id", "fighter name"),
                ("opponent id", "opponent name"),
            ):
                identity = _clean_text(row.get(id_column))
                if not identity:
                    continue
                fighter_id = _stable_id(identity)
                scheduled_ids.add(fighter_id)
                profile = profiles.setdefault(
                    fighter_id,
                    {
                        "id": fighter_id,
                        "name": _clean_text(row.get(name_column)),
                        "url": (
                            "http://ufcstats.com/fighter-details/"
                            f"{fighter_id}"
                        ),
                        "profile_scope": "ufcstats",
                        "height": "",
                        "height_inches": None,
                        "reach": "",
                        "reach_inches": None,
                        "stance": "",
                        "dob": "",
                        "dob_iso": None,
                        "scheduled_division": division,
                    },
                )
                profile["scheduled_division"] = division

    fighters: list[dict[str, object]] = []
    for fighter_id, profile in profiles.items():
        ufc_rows = ufc_rows_by_fighter.get(fighter_id, [])
        ordered_ufc_rows = sorted(
            ufc_rows,
            key=lambda row: (
                _iso_date(row.get("date")) or "",
                float(_number(row.get("bout_order")) or -1),
            ),
            reverse=True,
        )
        ordered_rows = sorted(
            rows_by_fighter.get(fighter_id, []),
            key=lambda row: (
                _iso_date(row.get("date")) or "",
                float(_number(row.get("bout_order")) or -1),
            ),
            reverse=True,
        )
        opponent_rows: list[pd.Series] = []
        for row in ordered_ufc_rows:
            fight_id = _clean_text(row.get("fight_id")) or _stable_id(row["fight_url"])
            paired = row_by_fight_and_fighter.get(
                (fight_id, _stable_id(row["opponent_url"]))
            )
            if paired is not None:
                opponent_rows.append(paired)
        career = (
            _career(ordered_ufc_rows, opponent_rows)
            if ordered_ufc_rows or profile.get("profile_scope") == "ufcstats"
            else {
                "recorded_bouts": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "no_contests": 0,
                "divisions": [],
                "recent_form": [],
                "totals": {},
                "opponent_totals": {},
            }
        )
        fighter = {
            **profile,
            "career": career,
            "fights": [_fight_array(row) for row in ordered_rows],
        }
        # Most profiles are UFC-only; their existing career object is already a
        # complete record. Publish the additional all-promotion summary only
        # where linked external history changes it, keeping the index scalable.
        if len(ordered_rows) > len(ordered_ufc_rows):
            fighter["record"] = _record_summary(ordered_rows)
        fighters.append(fighter)
    fighters.sort(key=lambda item: (str(item["name"]).casefold(), str(item["id"])))

    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "data_through": (
            pd.to_datetime(raw_fights["date"], errors="coerce").max().date().isoformat()
        ),
        "identity_contract": (
            "UFCStats URL IDs for UFC profiles; deterministic source IDs for "
            "external-only historical profiles"
        ),
        "counts": {
            "fighters": len(fighters),
            "fighters_with_recorded_bouts": sum(
                int(item.get("record", item["career"])["recorded_bouts"] > 0)
                for item in fighters
            ),
            "fighters_with_ufcstats_bouts": sum(
                int(item["career"]["recorded_bouts"] > 0) for item in fighters
            ),
            "external_only_fighters": sum(
                int(item.get("profile_scope") == "external_result_metadata")
                for item in fighters
            ),
            "scheduled_fighters": len(scheduled_ids),
            "fighter_fight_rows": len(raw_fights),
            "unique_fights": int(raw_fights["fight_url"].nunique()),
            "linked_external_fights": linked_external_fights,
            "linked_external_fighter_rows": linked_external_fighter_rows,
            "external_metadata_fights": len(external_rows) // 2,
            "external_metadata_fighter_rows": len(external_rows),
            "supplement_metadata_fights": len(supplement_rows) // 2,
            "supplement_metadata_fighter_rows": len(supplement_rows),
            "published_fighter_fight_rows": len(raw_fights) + len(external_rows),
        },
        "fight_columns": list(FIGHT_COLUMNS),
        "data_dictionary": {
            "profile": {
                "height_inches": "Parsed height in inches; null when UFCStats has no value.",
                "reach_inches": "Parsed reach in inches; null when UFCStats has no value.",
                "dob_iso": "ISO birth date parsed from UFCStats; null when unavailable.",
                "scheduled_division": "Division on the currently published card; blank when not scheduled.",
            },
            "career": {
                key: {
                    "label": definition[0],
                    "group": definition[1],
                    "format": definition[2],
                    "better": definition[3],
                }
                for key, definition in CAREER_DEFINITIONS.items()
            },
            "fight_stats": {
                key: {"label": value[0], "group": value[1], "unit": value[2]}
                for key, value in STAT_DEFINITIONS.items()
            },
            "notes": [
                "The directory includes external-only historical profiles where a reusable source has a Bellator or ONE result; UFCStats career rates remain UFC-only.",
                "The broad external bootstrap is an incomplete CC0 dataset through 2021-08-11; reviewed, source-attributed supplements can close documented gaps without changing model inputs.",
                "Absorbed and defensive statistics use the paired opponent row for the same stable fight ID.",
                "Each rate uses only bouts where both its statistic and fight duration are known; bouts with unknown duration never contribute a numerator without exposure.",
                "Wholly missing statistics remain null rather than becoming zero, and coverage counts show how many bouts support duration and control-time rates.",
                "Each fight is stored as an array in fight_columns order to keep the browser download compact.",
            ],
        },
        "fighters": fighters,
    }
    body["publication_sha256"] = _canonical_hash(body)
    return body


def validate_fighter_explorer(
    publication: object,
    raw_fights: pd.DataFrame,
    fighter_stats: pd.DataFrame,
    upcoming_fighters: pd.DataFrame | None = None,
    fight_shards: dict[str, object] | None = None,
    external_bouts: Iterable[object] | None = None,
    identity_map: Mapping[tuple[str, str], str] | None = None,
    external_supplements: Iterable[object] | None = None,
) -> dict[str, object]:
    if not isinstance(publication, dict):
        raise ValueError("fighter explorer publication must be an object")
    rebuilt = build_fighter_explorer(
        raw_fights,
        fighter_stats,
        upcoming_fighters,
        external_bouts,
        identity_map,
        external_supplements,
    )
    if fight_shards is None:
        expected_publication = rebuilt
        expected_shards = None
    else:
        expected_publication, expected_shards = split_fighter_explorer(rebuilt)
    if publication != expected_publication or (
        expected_shards is not None and fight_shards != expected_shards
    ):
        raise ValueError(
            "fighter explorer publication cannot be reproduced from processed data"
        )
    return expected_publication


def split_fighter_explorer(
    publication: dict[str, object],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    index = {
        key: value
        for key, value in publication.items()
        if key not in {"fighters", "publication_sha256"}
    }
    shard_fighters: dict[str, dict[str, list[list[object]]]] = {
        key: {} for key in SHARD_KEYS
    }
    index_fighters: list[dict[str, object]] = []
    for fighter in publication["fighters"]:
        profile = {key: value for key, value in fighter.items() if key != "fights"}
        key = _shard_key(str(fighter["id"]))
        profile["fight_shard"] = key
        index_fighters.append(profile)
        shard_fighters[key][str(fighter["id"])] = fighter["fights"]

    shards: dict[str, dict[str, object]] = {}
    shard_index: dict[str, dict[str, object]] = {}
    for key in SHARD_KEYS:
        fighters = shard_fighters[key]
        shard: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "data_through": publication["data_through"],
            "fight_columns": publication["fight_columns"],
            "fighters": fighters,
        }
        shard["publication_sha256"] = _canonical_hash(shard)
        shards[key] = shard
        shard_index[key] = {
            "path": f"fighter_fights_{key}.json",
            "fighter_count": len(fighters),
            "fighter_fight_rows": sum(len(fights) for fights in fighters.values()),
            "publication_sha256": shard["publication_sha256"],
        }
    index["fighters"] = index_fighters
    index["fight_shards"] = shard_index
    index["publication_sha256"] = _canonical_hash(index)
    return index, shards


def _atomic_write_json(
    value: dict[str, object],
    destination: Path,
    size_limit: int,
) -> None:
    encoded = _canonical_json(value) + "\n"
    if len(encoded.encode("utf-8")) > size_limit:
        raise ValueError(
            f"{destination.name} exceeded {size_limit // (1024 * 1024)} MiB"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_fighter_explorer(
    publication: dict[str, object],
    output_path: str | Path = OUTPUT_PATH,
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    index, shards = split_fighter_explorer(publication)
    for key, shard in shards.items():
        _atomic_write_json(
            shard,
            destination.with_name(f"fighter_fights_{key}.json"),
            SHARD_SIZE_LIMIT,
        )
    _atomic_write_json(index, destination, SIZE_LIMIT)


def load_external_history_inputs(
    bouts_path: str | Path = EXTERNAL_BOUTS_PATH,
    identity_path: str | Path = EXTERNAL_IDENTITY_PATH,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], str]]:
    """Load the already-validated external ledger and approved identity links."""

    bouts_file = Path(bouts_path)
    identity_file = Path(identity_path)
    if not bouts_file.exists() and not identity_file.exists():
        return [], {}
    if not bouts_file.exists() or not identity_file.exists():
        raise ValueError("external fighter history requires both bouts and identity map")
    observations: list[dict[str, object]] = []
    for line_number, line in enumerate(
        bouts_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"external bout at {bouts_file}:{line_number} is not an object"
            )
        observations.append(value)
    identities = pd.read_csv(identity_file, dtype=object, keep_default_na=False)
    required = {
        "source",
        "source_fighter_id",
        "canonical_fighter_id",
        "status",
    }
    missing = sorted(required - set(identities.columns))
    if missing:
        raise ValueError(f"external identity map is missing columns: {missing}")
    approved = identities[
        identities["status"].astype(str).str.casefold().eq("approved")
    ]
    mapping: dict[tuple[str, str], str] = {}
    for row in approved.to_dict("records"):
        key = (
            _clean_text(row.get("source")),
            _clean_text(row.get("source_fighter_id")),
        )
        canonical = _clean_text(row.get("canonical_fighter_id"))
        if not all((*key, canonical)):
            raise ValueError("approved external identity rows cannot contain blank IDs")
        previous = mapping.setdefault(key, canonical)
        if previous != canonical:
            raise ValueError(f"conflicting approved external identity mapping for {key}")
    return observations, mapping


def load_fighter_history_supplements(
    supplements_path: str | Path = EXTERNAL_SUPPLEMENTS_PATH,
) -> list[dict[str, object]]:
    """Load reviewed website-only history rows; these never enter model features."""

    supplements_file = Path(supplements_path)
    if not supplements_file.exists():
        return []
    supplements: list[dict[str, object]] = []
    for line_number, line in enumerate(
        supplements_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"fighter history supplement at {supplements_file}:{line_number} "
                "is not an object"
            )
        supplements.append(value)
    return supplements


def main() -> int:
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    fighters = pd.read_csv(FIGHTER_PATH, low_memory=False)
    upcoming = None
    if VEGAS_PATH.exists():
        upcoming = pd.DataFrame(json.loads(VEGAS_PATH.read_text(encoding="utf-8")))
    external_bouts, identity_map = load_external_history_inputs()
    external_supplements = load_fighter_history_supplements()
    publication = build_fighter_explorer(
        raw,
        fighters,
        upcoming,
        external_bouts,
        identity_map,
        external_supplements,
    )
    write_fighter_explorer(publication)
    print(
        "Published fighter explorer: "
        f"{publication['counts']['fighters']:,} fighters / "
        f"{publication['counts']['unique_fights']:,} fights"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
