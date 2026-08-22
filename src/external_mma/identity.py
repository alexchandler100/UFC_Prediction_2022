"""Evidence-based source-identity crosswalks; never join fighters by name alone."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import re
import unicodedata

import pandas as pd

from .schema import ExternalBoutObservation, clean_text, stable_token


IDENTITY_COLUMNS = [
    "source",
    "source_fighter_id",
    "source_fighter_name",
    "canonical_fighter_id",
    "canonical_source",
    "status",
    "match_method",
    "evidence_count",
    "reviewed_at_utc",
    "notes",
]


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _ufc_physical_bouts(raw: pd.DataFrame) -> dict[tuple[str, tuple[str, str]], pd.DataFrame]:
    required = {
        "date", "fight_url", "fighter", "opponent", "fighter_url", "opponent_url"
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"UFC raw data is missing identity columns: {sorted(missing)}")
    matches: dict[tuple[str, tuple[str, str]], list[pd.DataFrame]] = defaultdict(list)
    for _fight_url, rows in raw.groupby("fight_url", sort=False):
        if len(rows) != 2:
            continue
        first = rows.iloc[0]
        names = tuple(sorted((normalize_name(first["fighter"]), normalize_name(first["opponent"]))))
        key = (pd.to_datetime(first["date"], errors="raise").date().isoformat(), names)
        matches[key].append(rows)
    return {key: groups[0] for key, groups in matches.items() if len(groups) == 1}


def propose_ufcstats_crosswalk(
    observations: list[ExternalBoutObservation],
    raw_ufc_fights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match source identities using an exact historical UFC bout/date witness.

    The external bootstrap includes UFC results as well as other promotions.
    Those UFC rows are not replayed. Instead, a unique date plus unordered
    fighter-name pair anchors the external fighter IDs to UFCStats IDs. A
    mapping is approved only when every independent witness agrees.
    """

    ufc_bouts = _ufc_physical_bouts(raw_ufc_fights)
    evidence: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    source_names: dict[tuple[str, str], str] = {}
    for observation in observations:
        promotion = normalize_name(observation.promotion)
        if "ultimate fighting championship" not in promotion and promotion != "ufc":
            continue
        key = (
            observation.event_date,
            tuple(sorted((normalize_name(observation.fighter_name), normalize_name(observation.opponent_name)))),
        )
        raw_rows = ufc_bouts.get(key)
        if raw_rows is None:
            continue
        raw_name_ids: dict[str, set[str]] = defaultdict(set)
        for row in raw_rows.to_dict("records"):
            raw_name_ids[normalize_name(row["fighter"])].add(stable_token(row["fighter_url"]))
        for source_id, source_name in (
            (observation.fighter_source_id, observation.fighter_name),
            (observation.opponent_source_id, observation.opponent_name),
        ):
            candidate_ids = raw_name_ids.get(normalize_name(source_name), set())
            if len(candidate_ids) == 1:
                source_key = (observation.source, source_id)
                evidence[source_key].append((next(iter(candidate_ids)), observation.source_bout_id))
                source_names[source_key] = source_name

    approved: list[dict[str, object]] = []
    review: list[dict[str, object]] = []
    reviewed = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for (source, source_id), witnesses in sorted(evidence.items()):
        candidates = sorted({candidate for candidate, _bout in witnesses})
        row = {
            "source": source,
            "source_fighter_id": source_id,
            "source_fighter_name": source_names[(source, source_id)],
            "canonical_fighter_id": candidates[0] if len(candidates) == 1 else "",
            "canonical_source": "ufcstats" if len(candidates) == 1 else "",
            "status": "approved" if len(candidates) == 1 else "review",
            "match_method": "historical_ufc_bout_exact",
            "evidence_count": len(witnesses),
            "reviewed_at_utc": reviewed if len(candidates) == 1 else "",
            "notes": (
                f"unique date/name-pair witnesses; source bouts: "
                + ",".join(sorted({bout for _candidate, bout in witnesses})[:5])
                if len(candidates) == 1
                else f"conflicting UFCStats candidates: {','.join(candidates)}"
            ),
        }
        (approved if len(candidates) == 1 else review).append(row)
    return (
        pd.DataFrame(approved, columns=IDENTITY_COLUMNS),
        pd.DataFrame(review, columns=IDENTITY_COLUMNS),
    )


def merge_identity_maps(existing: pd.DataFrame, proposed: pd.DataFrame) -> pd.DataFrame:
    """Merge non-conflicting proposals without overwriting a human decision."""
    frames = []
    for frame in (existing, proposed):
        working = frame.copy()
        for column in IDENTITY_COLUMNS:
            if column not in working:
                working[column] = ""
        frames.append(working[IDENTITY_COLUMNS])
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return combined
    output: list[pd.Series] = []
    for key, rows in combined.groupby(["source", "source_fighter_id"], sort=True):
        approved = rows[rows["status"].astype(str).str.casefold().eq("approved")]
        canonical_ids = set(
            approved["canonical_fighter_id"].dropna().astype(str).str.strip()
        ) - {""}
        if len(canonical_ids) > 1:
            raise ValueError(f"conflicting approved identity mappings for {key}: {canonical_ids}")
        # Existing rows come first, so preserve an explicit human selection.
        output.append((approved if not approved.empty else rows).iloc[0])
    return pd.DataFrame(output, columns=IDENTITY_COLUMNS).sort_values(
        ["source", "source_fighter_name", "source_fighter_id"], kind="stable"
    ).reset_index(drop=True)
