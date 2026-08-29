import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_predictor.bayesian_logistic import BayesianLogisticConfig  # noqa: E402
from fight_predictor.bayesian_logistic_shadow import (  # noqa: E402
    POLICY_VERSION,
    SELECTED_BLEND_WEIGHT,
    BayesianLogisticShadowForecast,
    BayesianLogisticShadowStore,
    build_shadow_forecasts,
    frozen_blend,
    score_shadow_forecasts,
)
from market_tracker import MarketDataError, StoreIntegrityError  # noqa: E402


class _Builder:
    feature_columns = ("elo_slow_diff", "career_sig_accuracy_diff")

    def matchup_features(self, fighter_id, opponent_id, card_date, division):
        del fighter_id, opponent_id, card_date, division
        return pd.DataFrame(
            [{"elo_slow_diff": 0.25, "career_sig_accuracy_diff": -0.10}]
        )


class BayesianLogisticShadowTests(unittest.TestCase):
    @staticmethod
    def _forecast(*, event="event-one", fighter="fighter-a", opponent="fighter-b"):
        return BayesianLogisticShadowForecast.create(
            event_id=event,
            event_date="2026-01-10",
            timing_precision="date",
            event_start_utc=None,
            fighter_id=fighter,
            opponent_id=opponent,
            fighter_name="Fighter A",
            opponent_name="Fighter B",
            forecast_issued_at_utc="2026-01-08T12:00:00Z",
            source_commit_sha="a" * 40,
            experiment_sha256="b" * 64,
            training_start="2016-01-01",
            training_through="2025-12-13",
            training_fights=4_800,
            training_fingerprint_sha256="c" * 64,
            model_id="bayes-model",
            published_model_id="production-model",
            published_model_probability=0.55,
            bayesian_probability=0.65,
            bayesian_lower_probability=0.58,
            bayesian_upper_probability=0.71,
            frozen_blend_probability=float(
                frozen_blend(np.asarray([0.55]), np.asarray([0.65]))[0]
            ),
            calibration_slope=1.1,
            fighter_prior_fights=5,
            opponent_prior_fights=6,
            mean_chain_difference=0.002,
        )

    def test_record_is_content_addressed_and_paper_only(self):
        forecast = self._forecast()
        rebuilt = BayesianLogisticShadowForecast.from_mapping(
            forecast.to_mapping()
        )
        self.assertEqual(rebuilt, forecast)
        self.assertEqual(forecast.policy_version, POLICY_VERSION)
        self.assertEqual(forecast.bayesian_weight, SELECTED_BLEND_WEIGHT)
        self.assertTrue(forecast.paper_only)
        self.assertFalse(forecast.execution_enabled)

        tampered = forecast.to_mapping()
        tampered["frozen_blend_probability"] = 0.99
        with self.assertRaisesRegex(MarketDataError, "forecast_id"):
            BayesianLogisticShadowForecast.from_mapping(tampered)

    def test_same_day_date_only_forecast_is_rejected(self):
        with self.assertRaisesRegex(MarketDataError, "strictly before"):
            BayesianLogisticShadowForecast.create(
                **{
                    key: value
                    for key, value in self._forecast().to_mapping().items()
                    if key
                    not in {
                        "schema_version",
                        "forecast_id",
                        "policy_version",
                        "matchup_id",
                        "bayesian_weight",
                        "paper_only",
                        "candidate_only",
                        "execution_enabled",
                        "forecast_issued_at_utc",
                    }
                },
                forecast_issued_at_utc="2026-01-10T01:00:00Z",
            )

    def test_store_is_append_only_and_mirrors_are_verified(self):
        first = self._forecast()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = BayesianLogisticShadowStore(
                root / "shadow.csv", root / "shadow.jsonl"
            )
            added = store.append([first])
            duplicate = store.append([first])
            self.assertEqual(len(added.added_ids), 1)
            self.assertEqual(duplicate.duplicate_ids, (first.forecast_id,))
            self.assertEqual(store.read(), (first,))

            rewritten = self._forecast()
            mapping = rewritten.to_mapping()
            mapping["forecast_issued_at_utc"] = "2026-01-07T12:00:00.000000Z"
            mapping.pop("forecast_id")
            body = {
                key: value
                for key, value in mapping.items()
                if key
                not in {
                    "schema_version",
                    "policy_version",
                    "matchup_id",
                    "bayesian_weight",
                    "paper_only",
                    "candidate_only",
                    "execution_enabled",
                }
            }
            rewritten = BayesianLogisticShadowForecast.create(**body)
            with self.assertRaisesRegex(StoreIntegrityError, "rewritten"):
                store.append([rewritten])

            second = self._forecast(
                event="event-two", fighter="fighter-c", opponent="fighter-d"
            )
            store.append([second])
            csv_lines = (root / "shadow.csv").read_text(
                encoding="utf-8"
            ).splitlines()
            (root / "shadow.csv").write_text(
                "\n".join(csv_lines[:-1]) + "\n", encoding="utf-8"
            )
            self.assertEqual(store.read(), (first, second))
            store.append([])

            lines = (root / "shadow.jsonl").read_text(encoding="utf-8").splitlines()
            value = json.loads(lines[0])
            value["bayesian_probability"] = 0.99
            (root / "shadow.jsonl").write_text(
                json.dumps(value) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(StoreIntegrityError, "invalid Bayesian"):
                store.read()

    def test_scoring_uses_locked_orientation_and_never_enables_execution(self):
        first = self._forecast()
        second = self._forecast(
            event="event-two", fighter="fighter-d", opponent="fighter-c"
        )
        outcomes = {
            ("event-one", "fighter-a", "fighter-b"): (1, "fight-one"),
            ("event-two", "fighter-c", "fighter-d"): (1, "fight-two"),
        }
        report = score_shadow_forecasts(
            (first, second), outcomes, {"event-one", "event-two"}, set()
        )
        self.assertEqual(report["scored_fights"], 2)
        self.assertEqual(report["settled_events"], 2)
        self.assertEqual(report["metrics"]["frozen_blend"]["count"], 2)
        self.assertEqual(
            report["paired_log_loss_intervals"]["blend_minus_production"][
                "bootstrap_samples"
            ],
            10_000,
        )
        self.assertFalse(report["execution_enabled"])
        self.assertFalse(report["promotion_gate"]["count_requirements_met"])

    def test_frozen_recipe_builds_deterministic_upcoming_forecast(self):
        rng = np.random.default_rng(17)
        rows = []
        for index in range(640):
            year = 2024 if index < 520 else 2025
            elo = float(rng.normal())
            striking = float(rng.normal())
            chance = 1.0 / (1.0 + np.exp(-(0.3 * elo - 0.2 * striking)))
            rows.append(
                {
                    "date": f"{year}-06-{index % 27 + 1:02d}",
                    "event_id": f"old-event-{index // 10:03d}",
                    "bout_order": index % 10,
                    "fight_id": f"old-fight-{index:04d}",
                    "fighter_id": f"old-fighter-{index:04d}",
                    "opponent_id": f"old-opponent-{index:04d}",
                    "target": int(rng.random() < chance),
                    "elo_slow_diff": elo,
                    "career_sig_accuracy_diff": striking,
                }
            )
        training = pd.DataFrame(rows)
        upcoming = pd.DataFrame(
            [
                {
                    "event id": "future-event",
                    "date": "2026-01-10",
                    "fighter id": "fighter-a",
                    "opponent id": "fighter-b",
                    "fighter name": "Fighter A",
                    "opponent name": "Fighter B",
                    "division": "Lightweight",
                    "model id": "production-model",
                    "model probability": 0.57,
                    "fighter prior fights": 4,
                    "opponent prior fights": 5,
                }
            ]
        )
        config = BayesianLogisticConfig(
            burn_in=30,
            posterior_draws=30,
            chains=2,
            grouped_shrinkage=True,
            variance_prior_shape=3.0,
            variance_prior_scale=0.02,
            seed=23,
        )
        first = build_shadow_forecasts(
            training,
            _Builder(),
            upcoming,
            forecast_issued_at_utc="2026-01-08T12:00:00Z",
            source_commit_sha="a" * 40,
            experiment_sha256="b" * 64,
            config=config,
        )
        second = build_shadow_forecasts(
            training,
            _Builder(),
            upcoming,
            forecast_issued_at_utc="2026-01-08T12:00:00Z",
            source_commit_sha="a" * 40,
            experiment_sha256="b" * 64,
            config=config,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertGreater(first[0].bayesian_probability, 0.0)
        self.assertLess(first[0].bayesian_probability, 1.0)

    def test_weekly_workflow_publishes_only_the_paper_ledger(self):
        workflow = (REPO_ROOT / ".github/workflows/update-data.yml").read_text(
            encoding="utf-8"
        )
        updater = (REPO_ROOT / "src/update_and_rebuild_model.py").read_text(
            encoding="utf-8"
        )
        performance = (REPO_ROOT / "src/update_market_performance.py").read_text(
            encoding="utf-8"
        )
        for suffix in (".csv", ".jsonl"):
            name = f"bayesian_logistic_shadow_forecasts{suffix}"
            self.assertIn(name, workflow)
            self.assertIn(name, updater)
        self.assertIn("build_bayesian_logistic_shadow_forecasts", updater)
        self.assertIn("prospective_bayesian_logistic_blend", performance)
        self.assertNotIn("execution_enabled=True", updater)


if __name__ == "__main__":
    unittest.main()
