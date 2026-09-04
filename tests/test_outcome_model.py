from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_predictor.outcome_model import (  # noqa: E402
    DiscreteTimeOutcomeModel,
    InsufficientVerifiedScheduleData,
    TERMINAL_OUTCOMES,
    evaluate_outcome_model,
)
from fight_predictor.outcome_publication import (  # noqa: E402
    build_outcome_forecast_publication,
    validate_outcome_forecast_publication,
)


def _training_frame() -> pd.DataFrame:
    rows = []
    outcomes = (
        (1, "KO/TKO", 120),
        (0, "KO/TKO", 280),
        (1, "SUB", 430),
        (0, "SUB", 610),
        (1, "U-DEC", 900),
        (0, "S-DEC", 900),
    )
    for repeat in range(12):
        for index, (target, method, duration) in enumerate(outcomes):
            rows.append(
                {
                    "target": target,
                    "label_method": method,
                    "label_total_fight_seconds": duration,
                    "label_time_format": "3 Rnd (5-5)",
                    "skill_diff": float((index - 2) + repeat / 20),
                }
            )
    return pd.DataFrame(rows)


class OutcomeModelTests(unittest.TestCase):
    def test_competing_risks_form_coherent_winner_method_and_totals(self):
        model = DiscreteTimeOutcomeModel(["skill_diff"], c_value=0.1).fit(
            _training_frame()
        )
        prediction = model.predict({"skill_diff": 0.75}, scheduled_rounds=3)

        self.assertEqual(set(prediction.terminal_probabilities), set(TERMINAL_OUTCOMES))
        self.assertAlmostEqual(sum(prediction.terminal_probabilities.values()), 1.0)
        self.assertAlmostEqual(sum(prediction.method_probabilities.values()), 1.0)
        self.assertGreater(prediction.fighter_win_probability, 0.0)
        self.assertLess(prediction.fighter_win_probability, 1.0)
        over_1_5 = prediction.probability_over_seconds(450)
        over_2_5 = prediction.probability_over_seconds(750)
        self.assertIsNotNone(over_1_5)
        self.assertIsNotNone(over_2_5)
        self.assertGreaterEqual(over_1_5, over_2_5)
        self.assertIsNone(prediction.probability_over_seconds(900))

    def test_unknown_schedule_is_not_silently_assumed(self):
        frame = _training_frame()
        frame.loc[0, "label_time_format"] = ""
        frame.loc[0, "label_method"] = "KO/TKO"
        frame.loc[0, "label_finish_round"] = 1
        model = DiscreteTimeOutcomeModel(["skill_diff"]).fit(frame)
        self.assertEqual(model.omitted_unknown_schedule, 1)

    def test_known_five_round_early_finish_is_retained_in_training(self):
        frame = _training_frame().iloc[:1].copy()
        frame.loc[0, "label_time_format"] = "5 Rnd (5-5-5-5-5)"
        frame.loc[0, "label_finish_round"] = 1
        model = DiscreteTimeOutcomeModel(["skill_diff"])
        risk = model._risk_rows(frame)
        self.assertEqual(model.omitted_unknown_schedule, 0)
        self.assertEqual(risk.iloc[0]["scheduled_rounds"], 5)
        self.assertEqual(risk.iloc[0]["remaining_seconds"], 1500)

    def test_legacy_fitted_model_cannot_generate_new_probabilities(self):
        model = DiscreteTimeOutcomeModel(["skill_diff"]).fit(_training_frame())
        del model.schedule_contract_version
        with self.assertRaisesRegex(ValueError, "verified schedule contract"):
            model.predict({"skill_diff": 0.75}, scheduled_rounds=5)

    def test_evaluation_stops_before_fit_when_verified_history_is_insufficient(self):
        frame = pd.concat([_training_frame()] * 14, ignore_index=True)
        frame["date"] = pd.Timestamp("2025-01-01")
        frame["event_id"] = "event-one"
        frame["bout_order"] = range(len(frame))
        frame["fight_id"] = [f"fight-{index}" for index in range(len(frame))]
        frame["label_time_format"] = ""
        frame["label_finish_round"] = 1
        with patch.object(DiscreteTimeOutcomeModel, "fit", side_effect=AssertionError("must not fit")):
            with self.assertRaises(InsufficientVerifiedScheduleData) as caught:
                evaluate_outcome_model(frame, ["skill_diff"])
        self.assertEqual(caught.exception.verified_fights, 0)
        self.assertEqual(caught.exception.excluded_fights, 1008)

    def test_upcoming_publication_freezes_method_and_total_probabilities(self):
        model = DiscreteTimeOutcomeModel(["skill_diff"], c_value=0.1).fit(
            _training_frame()
        )

        class Builder:
            @staticmethod
            def matchup_features(*_args):
                return pd.DataFrame([{"skill_diff": 0.25}])

        upcoming = pd.DataFrame(
            [
                {
                    "fighter id": "fighter-a",
                    "opponent id": "fighter-b",
                    "fighter name": "Alpha",
                    "opponent name": "Beta",
                    "division": "Lightweight",
                    "model status": "model",
                }
            ]
        )
        publication = build_outcome_forecast_publication(
            model,
            Builder(),
            upcoming,
            {
                "event_id": "event-one",
                "event_url": "http://ufcstats.com/event-details/event-one",
                "date": "August 23, 2026",
                "title": "UFC Test",
            },
            selected_c=0.1,
            training_input_sha256="a" * 64,
            model_trained_through="2026-08-15",
            forecast_issued_at_utc="2026-08-21T12:00:00+00:00",
            source_commit_sha="b" * 40,
        )
        validated = validate_outcome_forecast_publication(publication)
        matchup = validated["matchups"][0]
        self.assertEqual(matchup["scheduled_rounds"], 5)
        self.assertAlmostEqual(sum(matchup["method_probabilities"].values()), 1.0)
        self.assertIn("4.5", matchup["total_round_over_probabilities"])
        self.assertTrue(validated["candidate_only"])
        self.assertFalse(validated["execution_enabled"])


if __name__ == "__main__":
    unittest.main()
