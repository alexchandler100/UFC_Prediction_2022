from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_online_data_challengers import _ranking_family_combinations
from online_data_research import (
    RANKING_FEATURES,
    add_ranking_features,
    prepare_pre_event_odds,
    prepare_rankings,
)


def _identity_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    fighters = pd.DataFrame(
        [
            {"name": "Fighter A", "url": "http://ufcstats/fighter-details/a"},
            {"name": "Fighter B", "url": "http://ufcstats/fighter-details/b"},
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "fighter": "Fighter A",
                "fighter_url": "http://ufcstats/fighter-details/a",
            },
            {
                "fighter": "Fighter B",
                "fighter_url": "http://ufcstats/fighter-details/b",
            },
        ]
    )
    return fighters, raw


class OnlineDataResearchTests(unittest.TestCase):
    def test_rankings_reject_conflicting_snapshot_and_use_strictly_prior_data(self):
        fighters, raw = _identity_inputs()
        source = pd.DataFrame(
        [
            {"date": "2020-01-01", "weightclass": "Lightweight", "fighter": "Fighter A", "rank": 1},
            {"date": "2020-01-08", "weightclass": "Lightweight", "fighter": "Fighter A", "rank": 2},
            {"date": "2020-01-08", "weightclass": "Lightweight", "fighter": "Fighter A", "rank": 3},
            {"date": "2020-01-15", "weightclass": "Lightweight", "fighter": "Fighter A", "rank": 0},
            {"date": "2020-01-15", "weightclass": "Men's Pound-for-PoundTop Rank", "fighter": "Fighter B", "rank": 0},
            {"date": "2020-01-15", "weightclass": "Men's Pound-for-PoundTop Rank", "fighter": "Fighter B", "rank": 1},
        ]
    )
        rankings, audit = prepare_rankings(source, fighters, raw)
        self.assertEqual(audit["conflicting_snapshot_dates_rejected"], 1)
        self.assertEqual(audit["synthetic_top_rank_rows_rejected"], 2)
        self.assertEqual(
            set(rankings["date"].dt.strftime("%Y-%m-%d")),
            {"2020-01-01", "2020-01-15"},
        )

        matchups = pd.DataFrame(
        [
            {"date": "2020-01-01", "fight_id": "same-day", "fighter_id": "a", "opponent_id": "b"},
            {"date": "2020-01-09", "fight_id": "after-bad-week", "fighter_id": "a", "opponent_id": "b"},
            {"date": "2020-01-16", "fight_id": "after-clean-week", "fighter_id": "a", "opponent_id": "b"},
        ]
    )
        featured, join_audit = add_ranking_features(matchups, rankings)
        self.assertEqual(featured.loc[0, "ranking_division_score_diff"], 0.0)
        self.assertEqual(featured.loc[1, "ranking_snapshot_date"], "2020-01-01")
        self.assertEqual(featured.loc[1, "ranking_division_score_diff"], 15.0)
        self.assertEqual(featured.loc[2, "ranking_division_score_diff"], 16.0)
        self.assertEqual(featured.loc[2, "ranking_champion_diff"], 1.0)
        self.assertTrue(join_audit["strictly_prior_snapshot_verified"])


    def test_appending_future_rankings_does_not_change_earlier_fight_features(self):
        fighters, raw = _identity_inputs()
        source = pd.DataFrame(
        [{"date": "2020-01-01", "weightclass": "Lightweight", "fighter": "Fighter A", "rank": 5}]
    )
        matchups = pd.DataFrame(
        [{"date": "2020-01-08", "fight_id": "old", "fighter_id": "a", "opponent_id": "b"}]
    )
        first, _ = add_ranking_features(
            matchups, prepare_rankings(source, fighters, raw)[0]
        )
        appended = pd.concat(
        [
            source,
            pd.DataFrame(
                [{"date": "2020-02-01", "weightclass": "Lightweight", "fighter": "Fighter B", "rank": 0}]
            ),
        ],
        ignore_index=True,
    )
        second, _ = add_ranking_features(
            matchups, prepare_rankings(appended, fighters, raw)[0]
        )
        self.assertTrue(
            first[list(RANKING_FEATURES)].equals(second[list(RANKING_FEATURES)])
        )


    def test_odds_use_names_within_stable_fight_when_source_urls_are_swapped(self):
        matchups = pd.DataFrame(
        [
            {
                "date": "2025-08-09",
                "event_id": "event",
                "fight_id": "fight",
                "fighter_id": "a",
                "opponent_id": "b",
                "fighter": "Fighter A",
                "opponent": "Fighter B",
                "target": 1,
            }
        ]
    )
        rows = []
        for book in ("One", "Two", "Three"):
            rows.append(
            {
                "fight_url": "http://ufcstats/fight-details/fight",
                "fighter_1_url": "http://ufcstats/fighter-details/b",
                "fighter_2_url": "http://ufcstats/fighter-details/a",
                "fighter_1": "Fighter A",
                "fighter_2": "Fighter B",
                "odds_1": 1.5,
                "odds_2": 3.0,
                "event_date": "2025-08-09",
                "adding_date": "2025-08-08T12:00:00Z",
                "source": book,
            }
        )
        rows.append({**rows[0], "source": "Late", "adding_date": "2025-08-09T01:00:00Z"})
        market, audit = prepare_pre_event_odds(pd.DataFrame(rows), matchups)
        self.assertEqual(len(market), 1)
        self.assertTrue(
            math.isclose(market.iloc[0]["market_probability"], 2.0 / 3.0)
        )
        self.assertEqual(audit["source_url_name_mismatches"], 3)
        self.assertEqual(audit["late_or_same_day_rows_rejected"], 1)


    def test_ranking_search_tests_joint_feature_groups(self):
        combinations = set(_ranking_family_combinations())
        self.assertEqual(len(combinations), 8)
        self.assertIn(
            (
                "current_division_rank",
                "current_pound_for_pound_rank",
                "ranking_history",
            ),
            combinations,
        )


if __name__ == "__main__":
    unittest.main()
