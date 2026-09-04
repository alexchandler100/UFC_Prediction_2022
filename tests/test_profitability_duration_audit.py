"""Check the audit's event split and evidence counts independently of saved reports."""

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_profitability_duration import (  # noqa: E402
    evaluation_time_split,
    insufficient_check_reproduction,
    schedule_evidence,
)


class ProfitabilityDurationAuditTests(unittest.TestCase):
    def test_split_keeps_every_fight_on_the_boundary_date_together(self):
        rows = []
        for event in range(91):
            for bout in range(11):
                rows.append({
                    "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=event * 7),
                    "event_id": f"event-{event}", "bout_order": bout,
                    "fight_id": f"fight-{event}-{bout}",
                })
        frame = pd.DataFrame(list(reversed(rows)))
        original = frame.copy(deep=True)
        split, cutoff = evaluation_time_split(frame)
        development = split[split["split"] == "development"]
        holdout = split[split["split"] == "holdout"]
        self.assertEqual(len(development), 792)
        self.assertEqual(len(holdout), 209)
        self.assertTrue((development["date"] < cutoff).all())
        self.assertTrue((holdout["date"] >= cutoff).all())
        self.assertFalse(set(development["event_id"]) & set(holdout["event_id"]))
        pd.testing.assert_frame_equal(frame, original)

    def test_schedule_evidence_separates_known_early_finish_from_result_inference(self):
        common = {"split": "development", "date": pd.Timestamp("2023-01-01"),
                  "event_id": "event-one", "label_time_format": ""}
        rows = [
            {**common, "fight_id": "unknown-early", "label_method": "KO/TKO",
             "label_finish_round": 1, "label_total_fight_seconds": 60},
            {**common, "fight_id": "unknown-late", "label_method": "KO/TKO",
             "label_finish_round": 4, "label_total_fight_seconds": 1100},
            {**common, "fight_id": "unknown-decision", "label_method": "U-DEC",
             "label_finish_round": 5, "label_total_fight_seconds": 1500},
            {**common, "fight_id": "known-early", "label_method": "KO/TKO",
             "label_finish_round": 1, "label_total_fight_seconds": 60,
             "label_time_format": "5 Rnd (5-5-5-5-5)"},
        ]
        evidence, grouped = schedule_evidence(pd.DataFrame(rows))
        by_id = evidence.set_index("fight_id")
        self.assertEqual(by_id.loc["unknown-early", "schedule_basis"],
                         "unknown")
        self.assertEqual(by_id.loc["known-early", "scheduled_rounds"], 5)
        self.assertTrue(by_id.loc["known-early", "explicit_schedule"])
        inferred_five = evidence[(evidence["scheduled_rounds"] == 5)
                                 & ~evidence["explicit_schedule"]]
        self.assertEqual(len(inferred_five), 0)
        self.assertEqual(int(inferred_five["finish_before_round_4"].sum()), 0)
        self.assertEqual(int(grouped["fights"].sum()), 4)
        self.assertEqual(int(grouped["over_3_5_fights"].sum()), 2)

    def test_insufficient_check_fixture_meets_overall_support_but_not_later_support(self):
        result = insufficient_check_reproduction()
        line = result["production_result"]
        self.assertEqual(line["training_fights"], 40)
        self.assertEqual(line["training_events"], 8)
        self.assertEqual(line["chronological_check"]["holdout_fights"], 2)
        self.assertEqual(result["insufficient_check_enabled"],
                         line["status"] == "available"
                         and line["chronological_check"]["status"] == "too_small")


if __name__ == "__main__":
    unittest.main()
