from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.evaluation import (  # noqa: E402
    evaluate_chronological_winner_stack,
    fit_nonnegative_logit_stack,
    stacked_win_probability,
)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -value))


def _synthetic_ledger() -> pd.DataFrame:
    rng = np.random.Generator(np.random.PCG64DXSM(9041))
    rows: list[dict[str, object]] = []
    fight_number = 0
    for year in range(2018, 2025):
        for index in range(80):
            model_signal = float(rng.normal())
            simulation_signal = float(rng.normal())
            true_probability = float(
                _sigmoid(np.asarray([1.15 * model_signal + 0.95 * simulation_signal]))[0]
            )
            red_won = bool(rng.random() < true_probability)
            rows.append(
                {
                    "date": f"{year}-{1 + index % 12:02d}-{1 + index % 27:02d}",
                    "event_id": f"event-{year}-{index // 10:02d}",
                    "fight_id": f"fight-{fight_number:04d}",
                    "actual_outcome": "red_decision" if red_won else "blue_decision",
                    "production_red_win_probability": float(
                        _sigmoid(np.asarray([model_signal]))[0]
                    ),
                    "simulation_red_win_probability": float(
                        _sigmoid(np.asarray([simulation_signal]))[0]
                    ),
                }
            )
            fight_number += 1
    return pd.DataFrame(rows)


class WinnerStackTests(unittest.TestCase):
    def test_fit_is_nonnegative_zero_intercept_and_swap_symmetric(self):
        model = [0.8, 0.7, 0.35, 0.2, 0.65, 0.4]
        simulation = [0.7, 0.45, 0.3, 0.25, 0.8, 0.35]
        truth = [1, 1, 0, 0, 1, 0]
        fitted = fit_nonnegative_logit_stack(model, simulation, truth)
        self.assertGreaterEqual(fitted["beta_model"], 0.0)
        self.assertGreaterEqual(fitted["beta_sim"], 0.0)
        self.assertEqual(fitted["intercept"], 0.0)

        forward = stacked_win_probability(
            model,
            simulation,
            beta_model=float(fitted["beta_model"]),
            beta_sim=float(fitted["beta_sim"]),
        )
        swapped = stacked_win_probability(
            [1.0 - value for value in model],
            [1.0 - value for value in simulation],
            beta_model=float(fitted["beta_model"]),
            beta_sim=float(fitted["beta_sim"]),
        )
        np.testing.assert_allclose(swapped, 1.0 - forward, atol=1e-12, rtol=0.0)

    def test_chronological_stack_adds_complementary_predictive_signal(self):
        ledger = _synthetic_ledger()
        stacked, comparison = evaluate_chronological_winner_stack(
            ledger,
            min_training_fights=100,
            card_bootstrap_replicates=300,
            random_seed=71,
        )
        self.assertEqual(comparison["status"], "evaluated")
        evaluated_folds = [
            fold for fold in comparison["folds"] if fold["status"] == "evaluated"
        ]
        self.assertGreaterEqual(len(evaluated_folds), 4)
        self.assertTrue(all(float(fold["beta_sim"]) > 0.0 for fold in evaluated_folds))
        self.assertLess(
            float(comparison["stack"]["log_loss"]),
            float(comparison["production_same_fights"]["log_loss"]),
        )
        self.assertGreater(stacked["stack_red_win_probability"].notna().sum(), 0)

    def test_current_year_labels_cannot_change_current_year_stack_predictions(self):
        ledger = _synthetic_ledger()
        original, original_comparison = evaluate_chronological_winner_stack(
            ledger,
            min_training_fights=100,
            card_bootstrap_replicates=50,
            random_seed=17,
        )
        changed = ledger.copy()
        current_year = pd.to_datetime(changed["date"]).dt.year.eq(2024)
        changed.loc[current_year, "actual_outcome"] = changed.loc[
            current_year, "actual_outcome"
        ].map({"red_decision": "blue_decision", "blue_decision": "red_decision"})
        altered, altered_comparison = evaluate_chronological_winner_stack(
            changed,
            min_training_fights=100,
            card_bootstrap_replicates=50,
            random_seed=17,
        )
        original_2024 = original.loc[current_year, "stack_red_win_probability"].to_numpy()
        altered_2024 = altered.loc[current_year, "stack_red_win_probability"].to_numpy()
        np.testing.assert_allclose(original_2024, altered_2024, atol=0.0, rtol=0.0)
        original_fold = next(
            fold for fold in original_comparison["folds"] if fold["test_year"] == 2024
        )
        altered_fold = next(
            fold for fold in altered_comparison["folds"] if fold["test_year"] == 2024
        )
        self.assertEqual(original_fold, altered_fold)

    def test_stack_fails_closed_during_warmup(self):
        ledger = _synthetic_ledger().iloc[:80]
        stacked, comparison = evaluate_chronological_winner_stack(
            ledger,
            min_training_fights=100,
            card_bootstrap_replicates=20,
        )
        self.assertEqual(comparison["status"], "insufficient_prior_out_of_fold_history")
        self.assertFalse(comparison["candidate_freeze_recommended"])
        self.assertTrue(stacked["stack_red_win_probability"].isna().all())


if __name__ == "__main__":
    unittest.main()
