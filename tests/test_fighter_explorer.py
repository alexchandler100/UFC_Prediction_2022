from __future__ import annotations

from copy import deepcopy
import unittest

import pandas as pd

from src.build_fighter_explorer import (
    FIGHT_COLUMNS,
    STAT_FIELDS,
    build_fighter_explorer,
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
                "scheduled_fighters": 0,
                "fighter_fight_rows": 2,
                "unique_fights": 1,
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


if __name__ == "__main__":
    unittest.main()
