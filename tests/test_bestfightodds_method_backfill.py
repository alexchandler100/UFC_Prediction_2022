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
    _pending_downloads,
    _store_download_failure,
    _winner_run_active,
    database_summary,
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

    def test_database_can_upgrade_from_mean_to_both_without_losing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "method.sqlite3"
            first = _open_database(path, mode="mean")
            first.close()
            upgraded = _open_database(path, mode="both")
            stored_mode = upgraded.execute(
                "SELECT value FROM metadata WHERE key='download_mode'"
            ).fetchone()[0]
            upgraded.close()
            self.assertEqual(stored_mode, "both")

    def test_mode_specific_pending_downloads_stop_after_retry_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "method.sqlite3"
            connection = _open_database(path, mode="mean")
            connection.execute(
                """
                INSERT INTO events(
                    event_url, source_event_date, page_status, page_attempts,
                    updated_at_utc
                ) VALUES ('event', '2025-01-01', 'parsed', 1, 'now')
                """
            )
            connection.execute(
                """
                INSERT INTO selections(
                    selection_id, event_url, source_matchup_id,
                    source_fighter_side, source_prop_type_id,
                    source_outcome_number, market, method, raw_label,
                    fighter_1_name, fighter_2_name, ufc_event_date,
                    ufc_event_id, ufc_fight_id, ufc_fighter_1_id,
                    ufc_fighter_2_id, selected_fighter_id,
                    mean_history_available, updated_at_utc
                ) VALUES (
                    'selection', 'event', 1, 1, 8, 1, 'method', 'ko_tko',
                    'A wins by KO', 'A', 'B', '2025-01-01', 'ufc-event',
                    'fight', 'a', 'b', 'a', 1, 'now'
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO selection_books(
                    selection_id, book_key, book_id, book_name
                ) VALUES ('selection', ?, ?, ?)
                """,
                (("mean", None, "Mean"), ("book:21", 21, "Book A")),
            )
            connection.commit()

            mean_spec = _pending_downloads(
                connection, event_url="event", mode="mean"
            )[0]
            self.assertEqual(mean_spec.book_key, "mean")
            self.assertEqual(
                _pending_downloads(connection, event_url="event", mode="books")[0].book_key,
                "book:21",
            )
            _store_download_failure(
                connection, spec=mean_spec, error=ValueError("bad chart")
            )
            _store_download_failure(
                connection, spec=mean_spec, error=ValueError("bad chart")
            )
            self.assertEqual(
                _pending_downloads(connection, event_url="event", mode="mean"),
                [],
            )
            summary = database_summary(
                connection, database_path=path, mode="mean"
            )
            self.assertEqual(summary["pending_downloads"], 0)
            self.assertEqual(
                summary["source_failures_at_retry_cap"]["chart_series"], 1
            )
            books_summary = database_summary(
                connection, database_path=path, mode="books"
            )
            self.assertEqual(books_summary["pending_downloads"], 1)
            connection.close()


if __name__ == "__main__":
    unittest.main()
