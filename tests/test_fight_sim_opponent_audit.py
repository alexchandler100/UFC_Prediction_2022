from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.opponent_audit import (  # noqa: E402
    ChronologicalOpponentRidgeSelector,
    OpponentAdjustmentAuditConfig,
    fit_bout_clustered_two_way_effects,
    run_opponent_adjustment_audit,
)


def _side(
    *,
    date: str,
    event: str,
    fight: str,
    fighter: str,
    opponent: str,
    result: str,
    attempts: int,
    landed: int,
    takedown_attempts: int,
    takedowns_landed: int,
    submissions: int,
) -> dict[str, object]:
    return {
        "date": date,
        "event_url": f"https://ufcstats.test/event-details/{event}",
        "fight_url": f"https://ufcstats.test/fight-details/{fight}",
        "fighter_url": f"https://ufcstats.test/fighter-details/{fighter}",
        "opponent_url": f"https://ufcstats.test/fighter-details/{opponent}",
        "fighter": fighter,
        "opponent": opponent,
        "division": "Testweight",
        "result": result,
        "method": "U-DEC",
        "round": 3,
        "time": "5:00",
        "total_fight_time": 900,
        "knockdowns": 0,
        "sig_strikes_landed": landed,
        "sig_strikes_attempts": attempts,
        "takedowns_landed": takedowns_landed,
        "takedowns_attempts": takedown_attempts,
        "sub_attempts": submissions,
        "reversals": 0,
        "control": 60,
        "head_strikes_landed": landed,
        "body_strikes_landed": 0,
        "leg_strikes_landed": 0,
        "distance_strikes_attempts": attempts,
        "clinch_strikes_attempts": 0,
        "ground_strikes_attempts": 0,
    }


def _fight(
    event_number: int,
    fight_number: int,
    first: str,
    second: str,
) -> list[dict[str, object]]:
    date = f"2020-01-{event_number:02d}"
    event = f"event-{event_number}"
    fight = f"fight-{event_number}-{fight_number}"
    first_attempts = 80 + (event_number % 3) * 8
    second_attempts = 65 + (event_number % 2) * 7
    return [
        _side(
            date=date,
            event=event,
            fight=fight,
            fighter=first,
            opponent=second,
            result="W",
            attempts=first_attempts,
            landed=first_attempts // 2,
            takedown_attempts=4,
            takedowns_landed=2,
            submissions=event_number % 2,
        ),
        _side(
            date=date,
            event=event,
            fight=fight,
            fighter=second,
            opponent=first,
            result="L",
            attempts=second_attempts,
            landed=second_attempts // 3,
            takedown_attempts=2,
            takedowns_landed=1,
            submissions=0,
        ),
    ]


def _raw(event_count: int = 13) -> pd.DataFrame:
    rows = []
    pairings = (("a", "b"), ("c", "d"))
    for event in range(1, event_count + 1):
        if event % 2 == 0:
            pairings = (("a", "d"), ("c", "b"))
        else:
            pairings = (("a", "b"), ("c", "d"))
        for fight_number, (first, second) in enumerate(pairings, start=1):
            rows.extend(_fight(event, fight_number, first, second))
    return pd.DataFrame(rows)


def _outer() -> pd.DataFrame:
    rows = []
    for event in range(9, 13):
        for fight in range(1, 3):
            rows.append(
                {
                    "date": f"2020-01-{event:02d}",
                    "event_id": f"event-{event}",
                    "fight_id": f"fight-{event}-{fight}",
                }
            )
    return pd.DataFrame(rows)


class OpponentAdjustmentAuditTests(unittest.TestCase):
    def test_equal_bout_ridge_recovers_direction_and_shrinks(self):
        actor_ids = []
        opponent_ids = []
        residuals = []
        for _ in range(8):
            for actor, actor_value in (("strong", 0.4), ("weak", -0.4)):
                for opponent, opponent_value in (
                    ("durable", -0.3),
                    ("vulnerable", 0.3),
                ):
                    actor_ids.append(actor)
                    opponent_ids.append(opponent)
                    residuals.append(actor_value + opponent_value)
        actor_low, opponent_low = fit_bout_clustered_two_way_effects(
            actor_ids, opponent_ids, residuals, ridge=2.0
        )
        actor_high, opponent_high = fit_bout_clustered_two_way_effects(
            actor_ids, opponent_ids, residuals, ridge=20.0
        )
        self.assertGreater(actor_low["strong"], actor_low["weak"])
        self.assertGreater(opponent_low["vulnerable"], opponent_low["durable"])
        self.assertLess(abs(actor_high["strong"]), abs(actor_low["strong"]))
        self.assertLess(
            abs(opponent_high["vulnerable"]),
            abs(opponent_low["vulnerable"]),
        )

    def test_card_bootstrap_weights_count_bouts_not_actions(self):
        actors = ["a", "a", "b", "b"]
        opponents = ["c", "d", "c", "d"]
        residuals = [0.4, 0.2, -0.4, -0.2]
        weighted = fit_bout_clustered_two_way_effects(
            actors,
            opponents,
            residuals,
            ridge=2.0,
            sample_weights=[3.0, 1.0, 3.0, 1.0],
        )
        repeated = fit_bout_clustered_two_way_effects(
            [actors[0]] * 3 + [actors[1]] + [actors[2]] * 3 + [actors[3]],
            [opponents[0]] * 3
            + [opponents[1]]
            + [opponents[2]] * 3
            + [opponents[3]],
            [residuals[0]] * 3
            + [residuals[1]]
            + [residuals[2]] * 3
            + [residuals[3]],
            ridge=2.0,
        )
        for weighted_effects, repeated_effects in zip(weighted, repeated):
            self.assertEqual(set(weighted_effects), set(repeated_effects))
            for identity in weighted_effects:
                self.assertAlmostEqual(
                    weighted_effects[identity], repeated_effects[identity]
                )

    def test_audit_is_coherent_and_future_append_invariant(self):
        config = OpponentAdjustmentAuditConfig(
            min_prior_ufc_fights=1,
            inner_validation_events=3,
            minimum_training_fights=4,
            ridge_grid=(2.0, 8.0),
            bootstrap_replicates=100,
            random_seed=91,
            max_runtime_seconds=30.0,
        )
        report, predictions = run_opponent_adjustment_audit(
            _raw(12), _outer(), config=config
        )
        future_report, future_predictions = run_opponent_adjustment_audit(
            _raw(13), _outer(), config=config
        )
        self.assertEqual(report["split"]["outer_event_cards"], 4)
        self.assertEqual(report["split"]["outer_fights"], 8)
        self.assertEqual(len(predictions), 80)
        self.assertEqual(set(report["targets"]), {
            "strike_pace",
            "strike_accuracy",
            "takedown_pace",
            "takedown_accuracy",
            "submission_pace",
        })
        self.assertTrue(
            np.isfinite(
                predictions[
                    [
                        "context_loss",
                        "marginal_loss",
                        "opponent_adjusted_loss",
                    ]
                ].to_numpy(float)
            ).all()
        )
        pd.testing.assert_frame_equal(predictions, future_predictions)
        comparable = copy.deepcopy(report)
        future_comparable = copy.deepcopy(future_report)
        for value in (comparable, future_comparable):
            value.pop("report_sha256")
            value.pop("runtime")
        self.assertEqual(comparable, future_comparable)

        selector = ChronologicalOpponentRidgeSelector(_raw(13), config)
        selected, inner_ids = selector.selected_for_cutoff("2020-01-12")
        event_rows = predictions.loc[predictions["event_id"].eq("event-12")]
        for target in event_rows["target"].unique():
            self.assertEqual(
                selected[(str(target), "opponent_adjusted")],
                float(
                    event_rows.loc[
                        event_rows["target"].eq(target),
                        "opponent_adjusted_selected_ridge",
                    ].iloc[0]
                ),
            )
        self.assertEqual(len(inner_ids), config.inner_validation_events)


if __name__ == "__main__":
    unittest.main()
