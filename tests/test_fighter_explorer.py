from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.build_fighter_explorer import (
    FIGHT_COLUMNS,
    STAT_FIELDS,
    build_fighter_explorer,
    load_upcoming_fighter_inputs,
    split_fighter_explorer,
    validate_fighter_explorer,
)


def _fight_row(
    *,
    fighter_id: str,
    fighter: str,
    opponent_id: str,
    opponent: str,
    result: str,
    stats: dict[str, int],
) -> dict[str, object]:
    row: dict[str, object] = {
        "date": "2026-08-01",
        "fight_url": "http://ufcstats.com/fight-details/fight-1",
        "event_url": "http://ufcstats.com/event-details/event-1",
        "fighter": fighter,
        "opponent": opponent,
        "fighter_url": f"http://ufcstats.com/fighter-details/{fighter_id}",
        "opponent_url": f"http://ufcstats.com/fighter-details/{opponent_id}",
        "result": result,
        "division": "Lightweight",
        "method": "KO/TKO" if result == "W" else "KO/TKO",
        "round": 3,
        "time": "5:00",
        "total_fight_time": 900,
        "source_card_index": 0,
        "bout_order": 0,
        "time_format": "3 Rnd (5-5)",
    }
    row.update({field: stats.get(field, 0) for field in STAT_FIELDS})
    return row


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha_stats = {
        "knockdowns": 1,
        "sig_strikes_landed": 30,
        "sig_strikes_attempts": 60,
        "total_strikes_landed": 40,
        "total_strikes_attempts": 75,
        "takedowns_landed": 2,
        "takedowns_attempts": 4,
        "sub_attempts": 1,
        "control": 180,
        "head_strikes_landed": 20,
        "head_strikes_attempts": 43,
        "body_strikes_landed": 6,
        "body_strikes_attempts": 9,
        "leg_strikes_landed": 4,
        "leg_strikes_attempts": 8,
        "distance_strikes_landed": 18,
        "distance_strikes_attempts": 41,
        "clinch_strikes_landed": 4,
        "clinch_strikes_attempts": 7,
        "ground_strikes_landed": 8,
        "ground_strikes_attempts": 12,
    }
    bravo_stats = {
        "sig_strikes_landed": 15,
        "sig_strikes_attempts": 50,
        "total_strikes_landed": 24,
        "total_strikes_attempts": 64,
        "takedowns_landed": 1,
        "takedowns_attempts": 5,
        "control": 60,
        "head_strikes_landed": 10,
        "head_strikes_attempts": 34,
        "body_strikes_landed": 3,
        "body_strikes_attempts": 8,
        "leg_strikes_landed": 2,
        "leg_strikes_attempts": 8,
        "distance_strikes_landed": 12,
        "distance_strikes_attempts": 42,
        "clinch_strikes_landed": 2,
        "clinch_strikes_attempts": 5,
        "ground_strikes_landed": 1,
        "ground_strikes_attempts": 3,
    }
    fights = pd.DataFrame(
        [
            _fight_row(
                fighter_id="alpha-id",
                fighter="Alex Alpha",
                opponent_id="bravo-id",
                opponent="Blake Bravo",
                result="W",
                stats=alpha_stats,
            ),
            _fight_row(
                fighter_id="bravo-id",
                fighter="Blake Bravo",
                opponent_id="alpha-id",
                opponent="Alex Alpha",
                result="L",
                stats=bravo_stats,
            ),
        ]
    )
    fighters = pd.DataFrame(
        [
            {
                "name": "Alex Alpha",
                "height": "5' 11\"",
                "reach": "73\"",
                "stance": "Orthodox",
                "dob": "Jan 01, 1995",
                "url": "http://ufcstats.com/fighter-details/alpha-id",
            },
            {
                "name": "Blake Bravo",
                "height": "5' 9\"",
                "reach": "70\"",
                "stance": "Southpaw",
                "dob": "Feb 02, 1996",
                "url": "http://ufcstats.com/fighter-details/bravo-id",
            },
            {
                "name": "Casey No Bouts",
                "height": "--",
                "reach": "--",
                "stance": "",
                "dob": "",
                "url": "http://ufcstats.com/fighter-details/casey-id",
            },
        ]
    )
    return fights, fighters


class FighterExplorerTests(unittest.TestCase):
    def test_builds_complete_stable_id_fighter_profiles(self) -> None:
        fights, fighters = _inputs()
        publication = build_fighter_explorer(fights, fighters)

        self.assertEqual(
            publication["counts"],
            {
                "fighters": 3,
                "fighters_with_recorded_bouts": 2,
                "fighters_with_ufcstats_bouts": 2,
                "external_only_fighters": 0,
                "scheduled_fighters": 0,
                "fighter_fight_rows": 2,
                "unique_fights": 1,
                "linked_external_fights": 0,
                "linked_external_fighter_rows": 0,
                "external_metadata_fights": 0,
                "external_metadata_fighter_rows": 0,
                "supplement_metadata_fights": 0,
                "supplement_metadata_fighter_rows": 0,
                "published_fighter_fight_rows": 2,
            },
        )
        self.assertEqual(publication["data_through"], "2026-08-01")
        self.assertEqual(len(publication["publication_sha256"]), 64)

        alpha = next(
            item for item in publication["fighters"] if item["id"] == "alpha-id"
        )
        self.assertEqual(alpha["height_inches"], 71)
        self.assertEqual(alpha["reach_inches"], 73)
        self.assertEqual(alpha["career"]["recorded_bouts"], 1)
        self.assertNotIn("record", alpha)
        self.assertEqual(alpha["career"]["sig_strikes_landed_per_minute"], 2)
        self.assertEqual(alpha["career"]["sig_strikes_absorbed_per_minute"], 1)
        self.assertEqual(alpha["career"]["sig_strike_accuracy"], 0.5)
        self.assertEqual(alpha["career"]["sig_strike_defense"], 0.7)
        self.assertEqual(alpha["career"]["takedown_defense"], 0.8)
        self.assertEqual(alpha["career"]["takedowns_landed_per_15"], 2)
        self.assertEqual(alpha["career"]["control_minutes_per_15"], 3)

        fight = dict(zip(FIGHT_COLUMNS, alpha["fights"][0], strict=True))
        self.assertEqual(fight["fight_id"], "fight-1")
        self.assertEqual(fight["opponent_id"], "bravo-id")
        self.assertEqual(fight["sig_strikes_landed"], 30)
        self.assertEqual(fight["control"], 180)

        inactive = next(
            item for item in publication["fighters"] if item["id"] == "casey-id"
        )
        self.assertEqual(inactive["career"]["recorded_bouts"], 0)
        self.assertNotIn("record", inactive)
        self.assertIsNone(inactive["career"]["sig_strikes_landed_per_minute"])
        self.assertEqual(inactive["fights"], [])

    def test_validation_rejects_an_unreproducible_publication(self) -> None:
        fights, fighters = _inputs()
        publication = build_fighter_explorer(fights, fighters)
        validate_fighter_explorer(publication, fights, fighters)

        tampered = deepcopy(publication)
        tampered["fighters"][0]["career"]["wins"] = 99
        with self.assertRaisesRegex(ValueError, "cannot be reproduced"):
            validate_fighter_explorer(tampered, fights, fighters)

    def test_rates_exclude_rows_without_the_required_stat_or_duration(self) -> None:
        fights, fighters = _inputs()
        unknown_duration = fights.copy(deep=True)
        unknown_duration["fight_url"] = (
            "http://ufcstats.com/fight-details/fight-unknown-duration"
        )
        unknown_duration["date"] = "2026-08-02"
        unknown_duration["total_fight_time"] = None
        unknown_duration.loc[0, "sig_strikes_landed"] = 300
        unknown_duration.loc[0, "sig_strikes_attempts"] = 400
        unknown_duration.loc[0, "control"] = 600

        missing_control = fights.copy(deep=True)
        missing_control["fight_url"] = (
            "http://ufcstats.com/fight-details/fight-missing-control"
        )
        missing_control["date"] = "2026-08-03"
        missing_control.loc[0, "control"] = None

        publication = build_fighter_explorer(
            pd.concat(
                [fights, unknown_duration, missing_control], ignore_index=True
            ),
            fighters,
        )
        alpha = next(
            item for item in publication["fighters"] if item["id"] == "alpha-id"
        )
        career = alpha["career"]

        self.assertEqual(publication["schema_version"], 3)
        self.assertEqual(career["recorded_bouts"], 3)
        self.assertEqual(career["bouts_with_duration"], 2)
        self.assertEqual(career["control_stat_bouts"], 1)
        self.assertEqual(career["control_share_stat_bouts"], 2)
        self.assertEqual(career["average_fight_minutes"], 15)
        self.assertEqual(career["sig_strikes_landed_per_minute"], 2)
        self.assertEqual(career["sig_strikes_absorbed_per_minute"], 1)
        self.assertEqual(career["control_minutes_per_15"], 3)
        self.assertEqual(career["totals"]["sig_strikes_landed"], 360)
        self.assertEqual(career["totals"]["control"], 780)

    def test_wholly_missing_totals_remain_null_instead_of_becoming_zero(self) -> None:
        fights, fighters = _inputs()
        fights["control"] = None
        publication = build_fighter_explorer(fights, fighters)
        alpha = next(
            item for item in publication["fighters"] if item["id"] == "alpha-id"
        )

        self.assertIsNone(alpha["career"]["totals"]["control"])
        self.assertIsNone(alpha["career"]["opponent_totals"]["control"])
        self.assertIsNone(alpha["career"]["control_minutes_per_15"])
        self.assertIsNone(alpha["career"]["control_share"])
        self.assertEqual(alpha["career"]["control_share_stat_bouts"], 0)

    def test_linked_external_history_is_visible_without_fabricated_stats(self) -> None:
        fights, fighters = _inputs()
        external = [
            {
                "observation_id": "a" * 64,
                "source": "kaggle_pro_mma_fights_v1",
                "source_bout_id": "external-bout",
                "source_bout_order": 4,
                "source_event_id": "bellator-100",
                "source_url": "https://example.test/bellator-100",
                "event_date": "2020-01-02",
                "event_name": "Bellator 100",
                "promotion": "Bellator MMA",
                "fighter_source_id": "source-alpha",
                "fighter_name": "Alex Alpha",
                "opponent_source_id": "source-charlie",
                "opponent_name": "Charlie Challenger",
                "result": "W",
                "method": "SUB",
                "division": "Unknown",
                "finish_round": 2,
                "finish_clock_seconds": 73,
                "scheduled_rounds": 3,
            }
        ]
        identities = {
            ("kaggle_pro_mma_fights_v1", "source-alpha"): "alpha-id"
        }
        publication = build_fighter_explorer(
            fights, fighters, external_bouts=external, identity_map=identities
        )
        alpha = next(
            item for item in publication["fighters"] if item["id"] == "alpha-id"
        )

        self.assertEqual(alpha["career"]["recorded_bouts"], 1)
        self.assertEqual(alpha["record"]["recorded_bouts"], 2)
        self.assertEqual(alpha["record"]["metadata_only_bouts"], 1)
        self.assertEqual(
            alpha["record"]["promotions"],
            [{"name": "Bellator MMA", "bouts": 1}, {"name": "UFC", "bouts": 1}],
        )
        external_fight = next(
            dict(zip(FIGHT_COLUMNS, values, strict=True))
            for values in alpha["fights"]
            if dict(zip(FIGHT_COLUMNS, values, strict=True))["promotion"]
            == "Bellator MMA"
        )
        self.assertFalse(external_fight["stats_available"])
        self.assertEqual(external_fight["event_name"], "Bellator 100")
        self.assertEqual(external_fight["time"], "1:13")
        self.assertIsNone(external_fight["sig_strikes_landed"])
        self.assertEqual(publication["counts"]["linked_external_fights"], 1)
        self.assertEqual(publication["counts"]["linked_external_fighter_rows"], 1)
        self.assertEqual(publication["counts"]["external_metadata_fights"], 1)
        self.assertEqual(publication["counts"]["external_metadata_fighter_rows"], 2)
        charlie = next(
            item for item in publication["fighters"]
            if item["name"] == "Charlie Challenger"
        )
        self.assertEqual(charlie["profile_scope"], "external_result_metadata")
        self.assertEqual(charlie["career"]["recorded_bouts"], 0)
        self.assertEqual(charlie["record"]["recorded_bouts"], 1)
        self.assertEqual(charlie["record"]["losses"], 1)
        self.assertEqual(publication["counts"]["external_only_fighters"], 1)
        validate_fighter_explorer(
            publication,
            fights,
            fighters,
            external_bouts=external,
            identity_map=identities,
        )

    def test_reviewed_supplement_extends_same_profile_newest_first(self) -> None:
        fights, fighters = _inputs()
        external = [
            {
                "observation_id": "a" * 64,
                "source": "kaggle_pro_mma_fights_v1",
                "source_bout_id": "old-bout",
                "source_bout_order": 0,
                "source_event_id": "old-event",
                "source_url": "https://example.test/old",
                "event_date": "2020-01-01",
                "event_name": "Old Event",
                "promotion": "One Championship",
                "fighter_source_id": "/fighter/Nong-Stamp-292745",
                "fighter_name": "Nong Stamp",
                "opponent_source_id": "/fighter/Old-Opponent-1",
                "opponent_name": "Old Opponent",
                "result": "W",
                "method": "U-DEC",
                "finish_round": 3,
                "finish_clock_seconds": 300,
                "scheduled_rounds": 3,
            }
        ]
        supplements = [
            {
                "source": "wikipedia_cc_by_sa_v4",
                "source_bout_id": "new-bout",
                "source_event_id": "new-event",
                "source_url": "https://example.test/revision",
                "event_date": "2023-09-30",
                "event_name": "New Event",
                "promotion": "ONE Championship",
                "fighter_profile_source": "kaggle_pro_mma_fights_v1",
                "fighter_source_id": "/fighter/Nong-Stamp-292745",
                "fighter_name": "Stamp Fairtex",
                "opponent_profile_source": "wikipedia_cc_by_sa_v4",
                "opponent_source_id": "new-opponent",
                "opponent_name": "New Opponent",
                "result": "W",
                "method": "TKO",
                "division": "Atomweight",
                "finish_round": 3,
                "finish_clock_seconds": 64,
                "scheduled_rounds": 5,
            }
        ]

        publication = build_fighter_explorer(
            fights,
            fighters,
            external_bouts=external,
            external_supplements=supplements,
        )
        stamp = next(
            item for item in publication["fighters"] if item["name"] == "Stamp Fairtex"
        )
        decoded = [
            dict(zip(FIGHT_COLUMNS, values, strict=True)) for values in stamp["fights"]
        ]

        self.assertEqual(stamp["record"]["recorded_bouts"], 2)
        self.assertEqual([fight["date"] for fight in decoded], ["2023-09-30", "2020-01-01"])
        self.assertEqual(
            decoded[0]["source_label"],
            "Wikipedia record supplement (CC BY-SA 4.0)",
        )
        self.assertEqual(publication["counts"]["supplement_metadata_fights"], 1)
        self.assertEqual(publication["counts"]["supplement_metadata_fighter_rows"], 2)
        self.assertEqual(publication["counts"]["external_metadata_fights"], 2)
        validate_fighter_explorer(
            publication,
            fights,
            fighters,
            external_bouts=external,
            external_supplements=supplements,
        )

    def test_sharded_publication_keeps_complete_logs_out_of_the_index(self) -> None:
        fights, fighters = _inputs()
        publication = build_fighter_explorer(fights, fighters)
        index, shards = split_fighter_explorer(publication)

        alpha = next(item for item in index["fighters"] if item["id"] == "alpha-id")
        self.assertNotIn("fights", alpha)
        self.assertEqual(alpha["fight_shard"], "a")
        self.assertEqual(len(shards["a"]["fighters"]["alpha-id"]), 1)
        self.assertEqual(
            index["fight_shards"]["a"]["publication_sha256"],
            shards["a"]["publication_sha256"],
        )
        validate_fighter_explorer(
            index,
            fights,
            fighters,
            fight_shards=shards,
        )

    def test_current_card_debutant_gets_a_searchable_empty_profile(self) -> None:
        fights, fighters = _inputs()
        upcoming = pd.DataFrame(
            [
                {
                    "fighter name": "Dana Debut",
                    "opponent name": "Alex Alpha",
                    "fighter id": "debut-id",
                    "opponent id": "alpha-id",
                    "division": "Welterweight",
                }
            ]
        )
        publication = build_fighter_explorer(fights, fighters, upcoming)
        debutant = next(
            item for item in publication["fighters"] if item["id"] == "debut-id"
        )

        self.assertEqual(publication["counts"]["scheduled_fighters"], 2)
        self.assertEqual(debutant["name"], "Dana Debut")
        self.assertEqual(debutant["scheduled_division"], "Welterweight")
        self.assertEqual(debutant["career"]["recorded_bouts"], 0)
        self.assertEqual(debutant["fights"], [])

    def test_all_announced_cards_supply_scheduled_fighters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            vegas_path = root / "vegas.json"
            upcoming_path = root / "upcoming.json"
            vegas_path.write_text(
                json.dumps(
                    {
                        "fighter name": {"0": "Current Alpha"},
                        "opponent name": {"0": "Current Beta"},
                        "fighter id": {"0": "current-alpha"},
                        "opponent id": {"0": "current-beta"},
                        "division": {"0": "Lightweight"},
                    }
                ),
                encoding="utf-8",
            )
            upcoming_path.write_text(
                json.dumps(
                    {
                        "matchups": [
                            {
                                "fighter_name": "Future Alpha",
                                "opponent_name": "Future Beta",
                                "fighter_id": "future-alpha",
                                "opponent_id": "future-beta",
                                "division": "Welterweight",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_upcoming_fighter_inputs(vegas_path, upcoming_path)

        self.assertIsNotNone(loaded)
        self.assertEqual(set(loaded["fighter name"]), {"Current Alpha", "Future Alpha"})
        self.assertEqual(set(loaded["opponent name"]), {"Current Beta", "Future Beta"})


if __name__ == "__main__":
    unittest.main()
