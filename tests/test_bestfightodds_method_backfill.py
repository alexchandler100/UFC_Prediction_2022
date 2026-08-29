from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backfill_bestfightodds_method_history import (  # noqa: E402
    DownloadSpec,
    _coherent_rows,
    _open_database,
    _winner_run_active,
)


class BestFightOddsMethodBackfillTests(unittest.TestCase):
    def test_prop_history_endpoint_preserves_all_source_keys(self):
        spec = DownloadSpec(
            selection_id="43928:1:8:1",
            event_url="https://www.bestfightodds.com/events/ufc-test-1",
            matchup_id=43928,
            fighter_side=1,
            prop_type_id=8,
            outcome_number=1,
            book_key="book:21",
            book_id=21,
            book_name="Book A",
        )
        self.assertEqual(
            spec.endpoint,
            "https://www.bestfightodds.com/api/ggd?b=21&m=43928&p=1&pt=8&tn=1",
        )

    def test_active_winner_run_is_detected(self):
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE runs(status TEXT NOT NULL, finished_at_utc TEXT)"
        )
        self.assertFalse(_winner_run_active(connection))
        connection.execute("INSERT INTO runs VALUES ('running', NULL)")
        self.assertTrue(_winner_run_active(connection))
        connection.execute(
            "UPDATE runs SET status='paused', finished_at_utc='2026-08-28T00:00:00Z'"
        )
        self.assertFalse(_winner_run_active(connection))

    def test_six_way_export_normalizes_only_coherent_prices(self):
        rows = []
        prices = {
            ("f1", "ko_tko"): 0.25,
            ("f1", "submission"): 0.15,
            ("f1", "decision"): 0.30,
            ("f2", "ko_tko"): 0.20,
            ("f2", "submission"): 0.10,
            ("f2", "decision"): 0.25,
        }
        for (fighter_id, method), implied in prices.items():
            rows.append(
                {
                    "ufc_event_date": "2026-08-22",
                    "ufc_event_id": "event",
                    "ufc_fight_id": "fight",
                    "source_matchup_id": 900,
                    "fighter_1_name": "Alpha",
                    "fighter_2_name": "Beta",
                    "ufc_fighter_1_id": "f1",
                    "ufc_fighter_2_id": "f2",
                    "selected_fighter_id": fighter_id,
                    "method": method,
                    "book_key": "book:21",
                    "book_name": "Book A",
                    "horizon": "safe_t24",
                    "observed_at_utc": "2026-08-20T23:59:00Z",
                    "decimal_odds": 1.0 / implied,
                    "implied_probability": implied,
                }
            )
        coherent = _coherent_rows(rows, max_skew_seconds=600)
        self.assertEqual(len(coherent), 6)
        self.assertAlmostEqual(
            sum(float(row["no_vig_probability"]) for row in coherent), 1.0
        )
        self.assertTrue(all(row["six_way_overround"] == 1.25 for row in coherent))

        rows[-1]["observed_at_utc"] = "2026-08-21T02:00:00Z"
        self.assertEqual(_coherent_rows(rows, max_skew_seconds=600), [])

    def test_database_mode_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "method.sqlite3"
            first = _open_database(path, mode="mean")
            first.close()
            with self.assertRaises(Exception):
                _open_database(path, mode="both")


if __name__ == "__main__":
    unittest.main()
