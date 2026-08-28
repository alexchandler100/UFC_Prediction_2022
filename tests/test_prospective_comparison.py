import sys
import unittest
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from market_tracker import (  # noqa: E402
    PROSPECTIVE_COMPARISON_FIRST_EVENT_DATE,
    prospective_comparison_report,
    symmetric_logit_blend,
)
import market_tracker.prospective_comparison as comparison  # noqa: E402


@dataclass(frozen=True)
class _Decision:
    decision_id: str
    event_id: str
    event_date: str
    market_probability: float
    model_probability: float
    model_id: str = "production-model"

    def to_mapping(self):
        return asdict(self)


@dataclass(frozen=True)
class _Settlement:
    decision_id: str
    target: int | None

    def to_mapping(self):
        return asdict(self)


class ProspectiveComparisonTests(unittest.TestCase):
    def test_scores_only_predeclared_future_cohort_and_is_order_invariant(self):
        decisions = (
            _Decision("old", "old-event", "2026-08-29", 0.90, 0.90),
            _Decision("d1", "event-1", "2026-09-05", 0.65, 0.80),
            _Decision("d2", "event-1", "2026-09-05", 0.40, 0.55),
            _Decision("d3", "event-2", "2026-09-12", 0.70, 0.60),
            _Decision("d4", "event-2", "2026-09-12", 0.30, 0.20),
            _Decision("void", "event-3", "2026-09-19", 0.60, 0.70),
        )
        settlements = (
            _Settlement("old", 0),
            _Settlement("d1", 1),
            _Settlement("d2", 1),
            _Settlement("d3", 0),
            _Settlement("d4", 0),
            _Settlement("void", None),
        )

        report = prospective_comparison_report(decisions, settlements)
        reversed_report = prospective_comparison_report(
            reversed(decisions), reversed(settlements)
        )

        self.assertEqual(report, reversed_report)
        self.assertEqual(
            report["cohort"]["first_eligible_event_date"],
            PROSPECTIVE_COMPARISON_FIRST_EVENT_DATE,
        )
        self.assertEqual(report["eligible_frozen_decisions"], 5)
        self.assertEqual(report["settled_decisions_including_voids"], 5)
        self.assertEqual(report["void_or_unscored_decisions"], 1)
        self.assertEqual(report["scored_fights"], 4)
        self.assertEqual(report["settled_events"], 2)
        self.assertEqual(report["scores"]["market"]["count"], 4)
        self.assertEqual(report["scores"]["production_model"]["count"], 4)
        expected = symmetric_logit_blend(0.65, 0.80, 0.5)
        expected_loss = -(
            math.log(expected)
            + math.log(symmetric_logit_blend(0.40, 0.55, 0.5))
            + math.log(
                1.0 - symmetric_logit_blend(0.70, 0.60, 0.5)
            )
            + math.log(
                1.0 - symmetric_logit_blend(0.30, 0.20, 0.5)
            )
        ) / 4.0
        self.assertAlmostEqual(
            report["scores"]["fixed_equal_logit_blend"]["log_loss"],
            expected_loss,
        )
        interval = report["paired_event_intervals"]["equal_blend_vs_market"]
        self.assertEqual(interval["log_loss"]["bootstrap_samples"], 10_000)
        self.assertEqual(report["status"], "collecting_results")
        self.assertFalse(report["checkpoint"]["sample_requirement_met"])
        self.assertFalse(report["execution_enabled"])

    def test_empty_cohort_and_unknown_settlement_are_explicit(self):
        old = _Decision("old", "old-event", "2026-08-29", 0.60, 0.70)
        empty = prospective_comparison_report((old,), (_Settlement("old", 1),))
        self.assertEqual(empty["scored_fights"], 0)
        self.assertIsNone(empty["scores"]["market"])

        with self.assertRaisesRegex(ValueError, "unknown decision"):
            prospective_comparison_report(
                (old,), (_Settlement("missing", 1),)
            )

    def test_requires_consistent_probability_improvement_at_checkpoint(self):
        decisions = (
            _Decision("d1", "event-1", "2026-09-05", 0.40, 0.90),
            _Decision("d2", "event-1", "2026-09-05", 0.60, 0.10),
            _Decision("d3", "event-2", "2026-09-12", 0.35, 0.85),
            _Decision("d4", "event-2", "2026-09-12", 0.65, 0.15),
        )
        settlements = (
            _Settlement("d1", 1),
            _Settlement("d2", 0),
            _Settlement("d3", 1),
            _Settlement("d4", 0),
        )
        with (
            patch.object(comparison, "PROSPECTIVE_MINIMUM_SCORED_FIGHTS", 4),
            patch.object(comparison, "PROSPECTIVE_MINIMUM_SETTLED_EVENTS", 2),
        ):
            report = prospective_comparison_report(decisions, settlements)

        self.assertTrue(report["checkpoint"]["sample_requirement_met"])
        self.assertTrue(
            report["checkpoint"]
            ["equal_blend_log_loss_better_than_market_requirement_met"]
        )
        self.assertTrue(
            report["checkpoint"]
            ["equal_blend_brier_not_worse_than_market_requirement_met"]
        )
        self.assertEqual(
            report["status"], "equal_blend_improves_probability_quality"
        )


if __name__ == "__main__":
    unittest.main()
