from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_market_first_challenger import (  # noqa: E402
    attach_strict_prior_ufc_counts,
    evaluate_market_first,
    prepare_features,
)


class MarketFirstChallengerTests(unittest.TestCase):
    def test_prior_counts_do_not_use_other_bouts_on_same_date_or_future_rows(self):
        raw = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "fight_url": "http://x/fight/f1",
                    "fighter_url": "http://x/fighter/a",
                    "opponent_url": "http://x/fighter/b",
                },
                {
                    "date": "2025-01-01",
                    "fight_url": "http://x/fight/f1",
                    "fighter_url": "http://x/fighter/b",
                    "opponent_url": "http://x/fighter/a",
                },
                {
                    "date": "2025-01-01",
                    "fight_url": "http://x/fight/f2",
                    "fighter_url": "http://x/fighter/a",
                    "opponent_url": "http://x/fighter/c",
                },
                {
                    "date": "2025-02-01",
                    "fight_url": "http://x/fight/f3",
                    "fighter_url": "http://x/fighter/a",
                    "opponent_url": "http://x/fighter/d",
                },
            ]
        )
        paired = pd.DataFrame(
            [
                {"fight_id": "f1", "fighter_id": "a", "opponent_id": "b"},
                {"fight_id": "f2", "fighter_id": "a", "opponent_id": "c"},
                {"fight_id": "f3", "fighter_id": "a", "opponent_id": "d"},
            ]
        )
        counts = attach_strict_prior_ufc_counts(paired, raw)
        self.assertEqual(counts["fighter_prior_ufc_fights"].tolist(), [0, 0, 2])
        self.assertEqual(counts["opponent_prior_ufc_fights"].tolist(), [0, 0, 0])

        future = raw.copy()
        future.loc[len(future)] = {
            "date": "2026-01-01",
            "fight_url": "http://x/fight/f4",
            "fighter_url": "http://x/fighter/a",
            "opponent_url": "http://x/fighter/e",
        }
        repeated = attach_strict_prior_ufc_counts(paired, future)
        self.assertEqual(
            counts["fighter_prior_ufc_fights"].tolist(),
            repeated["fighter_prior_ufc_fights"].tolist(),
        )

    def test_all_directional_features_flip_when_fighter_sides_are_swapped(self):
        paired = pd.DataFrame(
            [
                {
                    "fight_id": "normal",
                    "horizon": "opening",
                    "market_probability": 0.58,
                    "model_probability": 0.68,
                    "book_probability_range": 0.10,
                    "minimum_prior_ufc_fights": 1,
                },
                {
                    "fight_id": "normal",
                    "horizon": "safe_t24",
                    "market_probability": 0.62,
                    "model_probability": 0.72,
                    "book_probability_range": 0.12,
                    "minimum_prior_ufc_fights": 1,
                },
                {
                    "fight_id": "swapped",
                    "horizon": "opening",
                    "market_probability": 0.42,
                    "model_probability": 0.32,
                    "book_probability_range": 0.10,
                    "minimum_prior_ufc_fights": 1,
                },
                {
                    "fight_id": "swapped",
                    "horizon": "safe_t24",
                    "market_probability": 0.38,
                    "model_probability": 0.28,
                    "book_probability_range": 0.12,
                    "minimum_prior_ufc_fights": 1,
                },
            ]
        )
        features = prepare_features(paired)
        normal = features.loc[
            features["fight_id"].eq("normal") & features["horizon"].eq("safe_t24")
        ].iloc[0]
        swapped = features.loc[
            features["fight_id"].eq("swapped") & features["horizon"].eq("safe_t24")
        ].iloc[0]
        for column in (
            "market_logit",
            "model_disagreement",
            "market_movement_from_opening",
            "book_disagreement_market_strength",
            "low_history_model_disagreement",
        ):
            self.assertAlmostEqual(float(normal[column]), -float(swapped[column]))

    @staticmethod
    def _synthetic_history() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        dates = pd.date_range("2020-01-04", periods=60, freq="7D")
        for date_index, date in enumerate(dates):
            for fight_index in range(3):
                target = (date_index + fight_index) % 2
                fight_id = f"fight-{date_index}-{fight_index}"
                for horizon in ("opening", "safe_t24"):
                    rows.append(
                        {
                            "event_date": date.strftime("%Y-%m-%d"),
                            "event_id": f"event-{date_index}",
                            "fight_id": fight_id,
                            "fighter_id": f"a-{fight_id}",
                            "opponent_id": f"b-{fight_id}",
                            "fighter_name": "A",
                            "opponent_name": "B",
                            "target": target,
                            "horizon": horizon,
                            "book_count": 5,
                            "market_probability": 0.50,
                            "minimum_book_probability": 0.46,
                            "maximum_book_probability": 0.54,
                            "book_probability_range": 0.08,
                            "model_probability": 0.80 if target else 0.20,
                            "fighter_prior_ufc_fights": 5,
                            "opponent_prior_ufc_fights": 5,
                            "minimum_prior_ufc_fights": 5,
                        }
                    )
        return pd.DataFrame(rows)

    def test_later_fight_results_cannot_change_which_features_are_selected(self):
        history = self._synthetic_history()
        report, _detail = evaluate_market_first(
            history,
            minimum_event_dates=20,
            minimum_train_fights=50,
            minimum_selection_fights=15,
            minimum_test_fights=15,
            minimum_feature_support=10,
        )
        chosen = report["horizons"]["safe_t24"]["selection_winner"]["features"]
        self.assertIn("model_disagreement", chosen)

        changed = history.copy()
        final_dates = sorted(set(changed["event_date"]))[-12:]
        changed.loc[changed["event_date"].isin(final_dates), "target"] = 1 - changed.loc[
            changed["event_date"].isin(final_dates), "target"
        ]
        changed_report, _detail = evaluate_market_first(
            changed,
            minimum_event_dates=20,
            minimum_train_fights=50,
            minimum_selection_fights=15,
            minimum_test_fights=15,
            minimum_feature_support=10,
        )
        self.assertEqual(
            chosen,
            changed_report["horizons"]["safe_t24"]["selection_winner"]["features"],
        )


if __name__ == "__main__":
    unittest.main()
