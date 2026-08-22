"""Adapters turn licensed source exports into canonical observations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json

import pandas as pd

from .schema import (
    ExternalBoutObservation,
    ExternalDataError,
    clean_text,
    normalize_method,
    stable_token,
)


@dataclass
class AdapterResult:
    observations: list[ExternalBoutObservation]
    rejected: list[dict[str, object]]
    total_rows: int


class KaggleProMmaAdapter:
    """Import the CC0 `binduvr/pro-mma-fights` version-1 CSV snapshot."""

    source_key = "kaggle_pro_mma_fights_v1"
    required_columns = {
        "url", "event_title", "organisation", "date", "match_nr",
        "fighter1_url", "fighter2_url", "fighter1_name", "fighter2_name",
        "fighter1_result", "fighter2_result", "win_method", "win_details",
        "round", "time",
    }

    @staticmethod
    def _bout_id(row: dict[str, object]) -> str:
        # One source event repeats match_nr=12 for two different bouts. Include
        # stable participant URLs so neither bout overwrites the other.
        participants = sorted(
            [clean_text(row["fighter1_url"]), clean_text(row["fighter2_url"])]
        )
        key = json.dumps(
            [clean_text(row["url"]), clean_text(row["match_nr"]), participants],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return sha256(key.encode("utf-8")).hexdigest()

    def convert(self, content: bytes, snapshot_sha256: str) -> AdapterResult:
        frame = pd.read_csv(BytesIO(content), dtype=object, keep_default_na=False)
        missing = self.required_columns - set(frame.columns)
        if missing:
            raise ExternalDataError(
                f"Kaggle pro-MMA CSV is missing columns: {sorted(missing)}"
            )
        observations: list[ExternalBoutObservation] = []
        rejected: list[dict[str, object]] = []
        for position, row in enumerate(frame.to_dict("records"), start=2):
            try:
                first_result = clean_text(row["fighter1_result"])
                second_result = clean_text(row["fighter2_result"])
                expected_pairs = {
                    ("win", "loss"), ("loss", "win"),
                    ("draw", "draw"), ("nc", "nc"),
                }
                pair = (first_result.casefold(), second_result.casefold())
                if pair not in expected_pairs:
                    raise ExternalDataError(
                        f"results are not complementary: {first_result!r}/{second_result!r}"
                    )
                event_path = clean_text(row["url"])
                first_url = clean_text(row["fighter1_url"])
                second_url = clean_text(row["fighter2_url"])
                if not event_path or not first_url or not second_url:
                    raise ExternalDataError("stable event/fighter URLs are required")
                raw_round = clean_text(row["round"])
                finish_round = raw_round if raw_round not in {"", "0"} else None
                observations.append(
                    ExternalBoutObservation.create(
                        source=self.source_key,
                        snapshot_sha256=snapshot_sha256,
                        source_bout_id=self._bout_id(row),
                        source_bout_order=row["match_nr"],
                        source_event_id=event_path,
                        source_url=f"https://www.sherdog.com{event_path}",
                        event_date=row["date"],
                        event_name=row["event_title"],
                        promotion=row["organisation"],
                        fighter_source_id=first_url,
                        fighter_name=row["fighter1_name"],
                        opponent_source_id=second_url,
                        opponent_name=row["fighter2_name"],
                        result=first_result,
                        method=normalize_method(row["win_method"], row["win_details"]),
                        division="Unknown",
                        finish_round=finish_round,
                        finish_clock_seconds=row["time"],
                    )
                )
            except (ExternalDataError, TypeError, ValueError) as error:
                rejected.append({"source_row": position, "reason": str(error)})
        return AdapterResult(observations, rejected, len(frame))


class CanonicalCsvAdapter:
    """Import an authorized provider export already mapped to canonical columns."""

    required_columns = {
        "source_bout_id", "source_event_id", "source_url", "event_date",
        "event_name", "promotion", "fighter_source_id", "fighter_name",
        "opponent_source_id", "opponent_name", "result", "method",
    }

    def __init__(self, source_key: str):
        self.source_key = clean_text(source_key)
        if not self.source_key:
            raise ExternalDataError("source_key is required")

    def convert(self, content: bytes, snapshot_sha256: str) -> AdapterResult:
        frame = pd.read_csv(BytesIO(content), dtype=object, keep_default_na=False)
        missing = self.required_columns - set(frame.columns)
        if missing:
            raise ExternalDataError(
                f"canonical CSV is missing columns: {sorted(missing)}"
            )
        observations: list[ExternalBoutObservation] = []
        rejected: list[dict[str, object]] = []
        for position, row in enumerate(frame.to_dict("records"), start=2):
            try:
                observations.append(
                    ExternalBoutObservation.create(
                        source=self.source_key,
                        snapshot_sha256=snapshot_sha256,
                        source_bout_id=row["source_bout_id"],
                        source_bout_order=row.get("source_bout_order"),
                        source_event_id=row["source_event_id"],
                        source_url=row["source_url"],
                        event_date=row["event_date"],
                        event_name=row["event_name"],
                        promotion=row["promotion"],
                        fighter_source_id=row["fighter_source_id"],
                        fighter_name=row["fighter_name"],
                        opponent_source_id=row["opponent_source_id"],
                        opponent_name=row["opponent_name"],
                        result=row["result"],
                        method=row["method"],
                        division=row.get("division", "Unknown"),
                        finish_round=row.get("finish_round"),
                        finish_clock_seconds=row.get("finish_clock_seconds"),
                        scheduled_rounds=row.get("scheduled_rounds"),
                        discipline=row.get("discipline", "mma"),
                        professional=str(row.get("professional", "true")).casefold()
                        not in {"false", "0", "no"},
                    )
                )
            except (ExternalDataError, TypeError, ValueError) as error:
                rejected.append({"source_row": position, "reason": str(error)})
        return AdapterResult(observations, rejected, len(frame))
