import csv
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from repair_historical_schedules import plan_pit_schedule_repair, plan_schedule_repair


def encoded(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def fixture():
    raw, rounds = [], []
    for fighter, opponent in (("a", "b"), ("b", "a")):
        row = {"fight_url": "http://ufcstats.com/fight-details/f",
               "event_url": "http://ufcstats.com/event-details/e",
               "fighter_url": f"http://ufcstats.com/fighter-details/{fighter}",
               "opponent_url": f"http://ufcstats.com/fighter-details/{opponent}",
               "date": "2025-01-01", "time_format": "", "stat": "001.20",
               "text": "value, with comma"}
        raw.append(row)
        rounds.append({**row, "fight_id": "f", "event_id": "e", "fighter_id": fighter,
                       "opponent_id": opponent, "round": "1", "scheduled_rounds": "5",
                       "time_format": "5 Rnd (5-5-5-5-5)"})
    return raw, rounds


class HistoricalScheduleRepairTests(unittest.TestCase):
    def test_only_blank_schedule_cells_change_on_both_sides_and_repeat_is_noop(self):
        raw, rounds = fixture()
        repaired, report = plan_schedule_repair(encoded(raw), encoded(rounds))
        parsed = list(csv.DictReader(io.StringIO(repaired.decode())))
        self.assertEqual(report["changed_side_cells"], 2)
        for before, after in zip(raw, parsed):
            self.assertEqual(after["time_format"], "5 Rnd (5-5-5-5-5)")
            self.assertEqual({k: v for k, v in before.items() if k != "time_format"},
                             {k: v for k, v in after.items() if k != "time_format"})
        repeated, again = plan_schedule_repair(repaired, encoded(rounds))
        self.assertEqual(repeated, repaired)
        self.assertEqual(again["changed_side_cells"], 0)

    def test_conflicting_existing_schedule_is_never_overwritten(self):
        raw, rounds = fixture()
        raw[0]["time_format"] = "3 Rnd (5-5-5)"
        with self.assertRaisesRegex(ValueError, "Existing explicit schedule conflicts"):
            plan_schedule_repair(encoded(raw), encoded(rounds))

    def test_inconsistent_source_mirrors_are_rejected(self):
        raw, rounds = fixture()
        rounds[1]["opponent_id"] = "b"
        rounds[1]["opponent_url"] = "http://ufcstats.com/fighter-details/b"
        with self.assertRaisesRegex(ValueError, "not mirrored"):
            plan_schedule_repair(encoded(raw), encoded(rounds))

    def test_fight_and_event_identity_must_match(self):
        raw, rounds = fixture()
        raw[0]["event_url"] = "http://ufcstats.com/event-details/different"
        with self.assertRaisesRegex(ValueError, "identity"):
            plan_schedule_repair(encoded(raw), encoded(rounds))

    def test_pit_repair_changes_only_schedule_and_preserves_feature_text_and_order(self):
        raw, rounds = fixture()
        repaired_raw, _ = plan_schedule_repair(encoded(raw), encoded(rounds))
        pit = [{"fight_id": "unmatched", "label_time_format": "", "feature": "000.003"},
               {"fight_id": "f", "label_time_format": "", "feature": "1.23456789012345"}]
        repaired, changed = plan_pit_schedule_repair(encoded(pit), repaired_raw)
        rows = list(csv.DictReader(io.StringIO(repaired.decode())))
        self.assertEqual(changed, 1)
        self.assertEqual(rows[0], pit[0])
        self.assertEqual(rows[1]["feature"], pit[1]["feature"])
        self.assertEqual(rows[1]["label_time_format"], "5 Rnd (5-5-5-5-5)")
        repeated, changed = plan_pit_schedule_repair(repaired, repaired_raw)
        self.assertEqual(repeated, repaired)
        self.assertEqual(changed, 0)


if __name__ == "__main__":
    unittest.main()
