from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_semantics import (  # noqa: E402
    fight_duration_seconds,
    historical_schedule,
    method_bucket,
    physical_matchup_identity,
    scheduled_rounds_from_time_format,
    upcoming_schedule,
)


class FightSemanticsTests(unittest.TestCase):
    def test_method_and_identity_contracts(self):
        self.assertEqual(method_bucket("KO/TKO"), "ko_tko")
        self.assertEqual(method_bucket("U-DEC"), "decision")
        self.assertEqual(method_bucket("Overturned", result="NC"), "no_contest")
        self.assertEqual(method_bucket("U-DEC", result="D"), "draw")
        forward = physical_matchup_identity("event/e", "fighter/a", "fighter/b")
        reverse = physical_matchup_identity("event/e", "fighter/b", "fighter/a")
        self.assertEqual(forward, reverse)

    def test_schedule_and_active_duration(self):
        self.assertEqual(scheduled_rounds_from_time_format("5 Rnd (5-5-5-5-5)"), 5)
        self.assertEqual(fight_duration_seconds(3, "1:23", "3 Rnd (5-5-5)"), 683)
        self.assertTrue(math.isnan(fight_duration_seconds(4, "1:00", "3 Rnd (5-5-5)")))
        self.assertEqual(
            historical_schedule(
                time_format="", method="U-DEC", total_fight_seconds=900, finish_round=3
            ),
            (3, "inferred_from_decision_duration"),
        )
        self.assertEqual(upcoming_schedule(0, "Lightweight"), (5, "ufcstats_first_listed_main_event"))
        self.assertEqual(upcoming_schedule(4, "Women's Flyweight Title Bout"), (5, "ufcstats_title_bout_label"))


if __name__ == "__main__":
    unittest.main()
