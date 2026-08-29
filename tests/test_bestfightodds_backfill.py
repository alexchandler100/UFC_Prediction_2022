from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backfill_bestfightodds_history import (  # noqa: E402
    BackfillError,
    DownloadSpec,
    UFCFightIndex,
    UFCFightReference,
    _upsert_event_page,
    derive_consensus_rows,
    derive_horizon_rows,
    open_database,
    open_database_readonly,
    pending_downloads_for_event,
    store_download,
    validate_external_database_path,
)


def epoch_ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def chart(name: str, points: list[tuple[str, float]]):
    return [
        {
            "name": name,
            "data": [{"x": epoch_ms(stamp), "y": price} for stamp, price in points],
        }
    ]


class BestFightOddsBackfillTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "history.sqlite3"
        self.connection = open_database(self.database, mode="both")
        self.reference = UFCFightReference(
            event_date=date(2023, 8, 20),
            event_id="event-id",
            fight_id="fight-id",
            fighter_1_id="ufc-a",
            fighter_2_id="ufc-b",
            fighter_1_name="Fighter A",
            fighter_2_name="Fighter B",
        )
        self.index = UFCFightIndex([self.reference])

    def tearDown(self):
        self.connection.close()
        self.temporary.cleanup()

    @staticmethod
    def page(*, reversed_fighters=False):
        first, second = (
            ("Fighter B", "Fighter A")
            if reversed_fighters
            else ("Fighter A", "Fighter B")
        )
        return {
            "url": "https://www.bestfightodds.com/events/ufc-test-100",
            "event_date": "2023-08-20",
            "title": "UFC Test",
            "organizer": "UFC",
            "html_sha256": "a" * 64,
            "matchups": [
                {
                    "matchup_id": "9001",
                    "fighter_1": first,
                    "fighter_2": second,
                    "paired_books": [
                        {"book_id": 21, "book": "Book A"},
                        {"book_id": 22, "book": "Book B"},
                        {"book_id": 23, "book": "Book C"},
                        {"book_id": 28, "book": "Prediction Market"},
                    ],
                }
            ],
        }

    def test_fight_index_uses_exact_date_then_unique_adjacent_date(self):
        exact, status, offset = self.index.match(
            event_date=date(2023, 8, 20),
            fighter_1="Fighter B",
            fighter_2="Fighter A",
        )
        self.assertEqual(exact, self.reference)
        self.assertEqual(status, "exact_date_and_fighters")
        self.assertEqual(offset, 0)

        adjacent, status, offset = self.index.match(
            event_date=date(2023, 8, 19),
            fighter_1="Fighter A",
            fighter_2="Fighter B",
        )
        self.assertEqual(adjacent, self.reference)
        self.assertEqual(status, "unique_fighters_within_one_day")
        self.assertEqual(offset, 1)

        reordered, status, _ = self.index.match(
            event_date=date(2023, 8, 20),
            fighter_1="A Fighter",
            fighter_2="B Fighter",
        )
        self.assertEqual(reordered, self.reference)
        self.assertEqual(status, "exact_date_and_fighters")

    def test_fight_index_rejects_ambiguous_adjacent_match(self):
        other = UFCFightReference(
            event_date=date(2023, 8, 18),
            event_id="other-event",
            fight_id="other-fight",
            fighter_1_id="ufc-a",
            fighter_2_id="ufc-b",
            fighter_1_name="Fighter A",
            fighter_2_name="Fighter B",
        )
        index = UFCFightIndex([self.reference, other])
        matched, status, offset = index.match(
            event_date=date(2023, 8, 19),
            fighter_1="Fighter A",
            fighter_2="Fighter B",
        )
        self.assertIsNone(matched)
        self.assertEqual(status, "ambiguous_fighters_within_one_day")
        self.assertIsNone(offset)

    def test_one_character_typo_requires_the_other_fighter_to_match(self):
        matched, status, offset = self.index.match(
            event_date=date(2023, 8, 20),
            fighter_1="Fighter C",
            fighter_2="Fighter B",
        )
        self.assertEqual(matched, self.reference)
        self.assertEqual(status, "unique_near_name_within_one_day")
        self.assertEqual(offset, 0)

        rejected, status, _ = self.index.match(
            event_date=date(2023, 8, 20),
            fighter_1="Fighter C",
            fighter_2="Different Opponent",
        )
        self.assertIsNone(rejected)
        self.assertEqual(status, "unmatched")

    def test_page_mapping_aligns_fighter_ids_and_skips_prediction_markets(self):
        _upsert_event_page(
            self.connection,
            page=self.page(reversed_fighters=True),
            fight_index=self.index,
        )
        matchup = self.connection.execute(
            "SELECT * FROM matchups WHERE matchup_id=9001"
        ).fetchone()
        self.assertEqual(matchup["ufc_fight_id"], "fight-id")
        self.assertEqual(matchup["ufc_fighter_1_id"], "ufc-b")
        self.assertEqual(matchup["ufc_fighter_2_id"], "ufc-a")
        books = {
            row["book_key"]
            for row in self.connection.execute(
                "SELECT book_key FROM matchup_books WHERE matchup_id=9001"
            )
        }
        self.assertEqual(books, {"mean", "book:21", "book:22", "book:23"})

    def test_download_is_resumable_and_horizons_use_common_timestamps(self):
        _upsert_event_page(
            self.connection, page=self.page(), fight_index=self.index
        )
        pending = pending_downloads_for_event(
            self.connection,
            event_url=self.page()["url"],
            mode="mean",
        )
        self.assertEqual(len(pending), 2)
        points_1 = [
            ("2023-08-01T00:00:00Z", 1.80),
            ("2023-08-16T12:00:00Z", 1.85),
            ("2023-08-18T12:00:00Z", 1.90),
            ("2023-08-19T12:00:00Z", 1.95),
            ("2023-08-19T22:00:00Z", 2.00),
            ("2023-08-20T01:00:00Z", 2.10),
        ]
        points_2 = [(stamp, 2.10) for stamp, _ in points_1]
        for spec, points in zip(pending, (points_1, points_2)):
            store_download(
                self.connection,
                spec=spec,
                series=chart("Mean", points),
            )
        self.assertEqual(
            pending_downloads_for_event(
                self.connection,
                event_url=self.page()["url"],
                mode="mean",
            ),
            [],
        )

        rows = [
            row for row in derive_horizon_rows(self.connection) if row["book_key"] == "mean"
        ]
        by_horizon = {row["horizon"]: row for row in rows}
        self.assertEqual(set(by_horizon), {
            "opening",
            "safe_t72",
            "safe_t24",
            "safe_t6",
            "strict_latest_before_event_date",
        })
        self.assertEqual(by_horizon["opening"]["observed_at_utc"], "2023-08-01T00:00:00Z")
        self.assertEqual(by_horizon["safe_t72"]["observed_at_utc"], "2023-08-16T12:00:00Z")
        self.assertEqual(by_horizon["safe_t24"]["observed_at_utc"], "2023-08-18T12:00:00Z")
        self.assertEqual(by_horizon["safe_t6"]["observed_at_utc"], "2023-08-19T12:00:00Z")
        self.assertEqual(
            by_horizon["strict_latest_before_event_date"]["observed_at_utc"],
            "2023-08-19T22:00:00Z",
        )
        self.assertFalse(by_horizon["opening"]["actual_event_start_time_known"])

    def test_consensus_requires_three_distinct_sportsbooks(self):
        base = {
            "ufc_event_date": "2023-08-20",
            "ufc_event_id": "event-id",
            "ufc_fight_id": "fight-id",
            "ufc_fighter_1_id": "ufc-a",
            "ufc_fighter_2_id": "ufc-b",
            "fighter_1_name": "Fighter A",
            "fighter_2_name": "Fighter B",
            "source_matchup_id": 9001,
            "horizon": "safe_t24",
            "cutoff_utc": "2023-08-19T00:00:00Z",
            "cutoff_basis": "source_event_calendar_date_at_00_utc",
            "observed_at_utc": "2023-08-18T12:00:00Z",
            "book_kind": "book",
        }
        rows = [
            {
                **base,
                "book_name": book,
                "fighter_1_no_vig_probability": probability,
            }
            for book, probability in (("A", 0.55), ("B", 0.57), ("C", 0.56))
        ]
        consensus = derive_consensus_rows(rows, minimum_books=3)
        self.assertEqual(len(consensus), 1)
        self.assertAlmostEqual(consensus[0]["fighter_1_market_probability"], 0.56)
        self.assertEqual(consensus[0]["ufc_fighter_1_id"], "ufc-a")
        self.assertEqual(consensus[0]["ufc_fighter_2_id"], "ufc-b")
        self.assertEqual(
            derive_consensus_rows(rows[:2], minimum_books=3), []
        )

    def test_database_inside_repository_is_rejected(self):
        with self.assertRaisesRegex(BackfillError, "outside the repository"):
            validate_external_database_path(ROOT / "artifacts" / "odds.sqlite3", ROOT)

    def test_database_can_be_opened_readonly_for_status(self):
        self.connection.commit()
        readonly = open_database_readonly(self.database, mode="both")
        self.assertEqual(
            readonly.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0],
            "1",
        )
        readonly.close()


if __name__ == "__main__":
    unittest.main()
