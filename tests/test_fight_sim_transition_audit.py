import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.transition_audit import (
    TransitionAuditConfig,
    build_transition_opportunities,
    run_transition_audit,
)
from ufc_round_data import ROUND_DATA_COLUMNS, ROUND_STAT_COLUMNS


def _side(
    *,
    event_number,
    fight_name,
    fighter,
    opponent,
    result,
    method,
    knockdowns=0,
    takedowns=0,
    submissions=0,
    control=0,
):
    fight_id = f"event-{event_number}-{fight_name}"
    row = {column: 0 for column in ROUND_STAT_COLUMNS}
    row.update(
        {
            "schema_version": 1,
            "round_stat_id": f"{fight_id}:{fighter}:r1",
            "event_id": f"event-{event_number}",
            "event_url": f"https://ufcstats.test/event-details/event-{event_number}",
            "fight_id": fight_id,
            "fight_url": f"https://ufcstats.test/fight-details/{fight_id}",
            "date": f"2025-01-{event_number:02d}",
            "source_card_index": event_number,
            "bout_order": 0,
            "division": "Testweight",
            "time_format": "3 Rnd (5-5-5)",
            "scheduled_rounds": 3,
            "finish_round": 1,
            "finish_time": "5:00",
            "total_fight_seconds": 300,
            "round": 1,
            "round_seconds": 300,
            "fighter_id": fighter,
            "fighter_url": f"https://ufcstats.test/fighter-details/{fighter}",
            "fighter": fighter,
            "opponent_id": opponent,
            "opponent_url": f"https://ufcstats.test/fighter-details/{opponent}",
            "opponent": opponent,
            "result": result,
            "method": method,
            "knockdowns": knockdowns,
            "takedowns_landed": takedowns,
            "takedowns_attempts": takedowns,
            "sub_attempts": submissions,
            "control": control,
            "reconciliation_status": "matched",
            "reconciliation_issue_count": 0,
        }
    )
    return row


def _fight(event_number, fight_name, winner, loser, method, **winner_stats):
    return [
        _side(
            event_number=event_number,
            fight_name=fight_name,
            fighter=winner,
            opponent=loser,
            result="W",
            method=method,
            **winner_stats,
        ),
        _side(
            event_number=event_number,
            fight_name=fight_name,
            fighter=loser,
            opponent=winner,
            result="L",
            method=method,
        ),
    ]


def _rounds(event_count=12):
    rows = []
    for event in range(1, event_count + 1):
        rows.extend(
            _fight(
                event,
                "ko-high",
                "ko-high",
                "ko-fragile",
                "KO/TKO",
                knockdowns=1,
            )
        )
        rows.extend(
            _fight(
                event,
                "ko-low",
                "ko-low",
                "ko-durable",
                "U-DEC",
                knockdowns=1,
            )
        )
        rows.extend(
            _fight(
                event,
                "sub-high",
                "sub-high",
                "sub-fragile",
                "SUB",
                takedowns=1,
                submissions=1,
                control=240,
            )
        )
        rows.extend(
            _fight(
                event,
                "sub-low",
                "sub-low",
                "sub-durable",
                "U-DEC",
                takedowns=1,
                control=30,
            )
        )
    return pd.DataFrame(rows, columns=ROUND_DATA_COLUMNS)


class TransitionOpportunityTests(unittest.TestCase):
    def test_labels_same_round_associations_without_claiming_action_order(self):
        targets, metadata = build_transition_opportunities(_rounds(2))
        knockdown = targets[
            "knockdown_round_to_ko_tko_same_round_association"
        ]
        submission = targets[
            "takedown_round_to_submission_win_same_round_association"
        ]
        control = targets[
            "takedown_round_to_credited_control_share_same_round_association"
        ]
        self.assertEqual(knockdown["actual"].tolist(), [1, 0, 1, 0])
        self.assertEqual(submission["actual"].tolist(), [1, 0, 1, 0])
        self.assertEqual(control["actual"].tolist(), [0.8, 0.1, 0.8, 0.1])
        self.assertIn("does not establish action order", metadata["source_limitation"])
        self.assertIn("top versus bottom", metadata["source_limitation"])

    def test_sparse_fighter_candidate_is_evaluated_but_not_promoted(self):
        report, predictions = run_transition_audit(
            _rounds(),
            config=TransitionAuditConfig(
                holdout_latest_events=2,
                context_prior_opportunities=2,
                fighter_prior_opportunities=1,
                bootstrap_replicates=200,
                max_runtime_seconds=30,
            ),
        )
        knockdown = report["targets"][
            "knockdown_round_to_ko_tko_same_round_association"
        ]
        self.assertLess(
            knockdown["fighter_opponent_log_loss"], knockdown["context_log_loss"]
        )
        self.assertFalse(knockdown["evidence_adequate_for_mechanic"])
        self.assertFalse(knockdown["candidate_retained"])
        self.assertFalse(report["production_behavior_changed"])
        self.assertGreater(len(predictions), 0)

    def test_strict_as_of_makes_future_append_invariant(self):
        config = TransitionAuditConfig(
            holdout_latest_events=2,
            context_prior_opportunities=2,
            fighter_prior_opportunities=1,
            bootstrap_replicates=200,
            max_runtime_seconds=30,
            as_of="2025-01-09",
        )
        earlier, _ = run_transition_audit(_rounds(8), config=config)
        appended, _ = run_transition_audit(_rounds(12), config=config)
        self.assertEqual(earlier["source"], appended["source"])
        self.assertEqual(earlier["split"], appended["split"])
        self.assertEqual(earlier["targets"], appended["targets"])

    def test_holdout_is_based_on_whole_source_cards_not_target_rows(self):
        rounds = _rounds(4)
        newest = rounds["event_id"].eq("event-4")
        rounds.loc[newest, ["knockdowns", "takedowns_landed", "takedowns_attempts"]] = 0
        report, _ = run_transition_audit(
            rounds,
            config=TransitionAuditConfig(
                holdout_latest_events=1,
                bootstrap_replicates=200,
                max_runtime_seconds=30,
            ),
        )
        self.assertEqual(report["split"]["holdout_event_ids"], ["event-4"])
        for result in report["targets"].values():
            self.assertEqual(result["status"], "insufficient_opportunities")


if __name__ == "__main__":
    unittest.main()
