import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_handler.data_handler import DataHandler
from ufc_round_data import (
    ROUND_DATA_COLUMNS,
    declared_round_lengths_seconds,
    empty_round_stats_frame,
    normalize_round_stats,
    parse_ufcstats_round_stats,
    reconcile_round_stats,
    round_exposure_seconds,
    validate_normalized_round_stats,
)


def _pair_cell(first, second):
    return (
        '<td class="b-fight-details__table-col">'
        f'<p>{first}</p><p>{second}</p></td>'
    )


def _fighter_cell(fight_id="fight-1"):
    del fight_id
    return (
        '<td class="b-fight-details__table-col">'
        '<p><a href="http://ufcstats.test/fighter-details/fighter-a">A</a></p>'
        '<p><a href="http://ufcstats.test/fighter-details/fighter-b">B</a></p>'
        '</td>'
    )


def _round_row(values):
    return (
        '<tr class="b-fight-details__table-row">'
        + _fighter_cell()
        + ''.join(_pair_cell(*value) for value in values)
        + '</tr>'
    )


def _round_fixture():
    totals = [
        [
            ("1", "0"),
            ("10 of 20", "8 of 18"),
            ("50%", "44%"),
            ("12 of 22", "9 of 19"),
            ("1 of 2", "0 of 1"),
            ("50%", "0%"),
            ("0", "0"),
            ("0", "0"),
            ("1:00", "0:20"),
        ],
        [
            ("0", "0"),
            ("5 of 10", "4 of 9"),
            ("50%", "44%"),
            ("6 of 11", "5 of 10"),
            ("0 of 1", "1 of 2"),
            ("0%", "50%"),
            ("1", "0"),
            ("0", "1"),
            ("0:30", "0:40"),
        ],
    ]
    significant = [
        [
            ("10 of 20", "8 of 18"),
            ("50%", "44%"),
            ("6 of 12", "4 of 9"),
            ("2 of 4", "2 of 4"),
            ("2 of 4", "2 of 5"),
            ("7 of 15", "6 of 14"),
            ("1 of 2", "1 of 2"),
            ("2 of 3", "1 of 2"),
        ],
        [
            ("5 of 10", "4 of 9"),
            ("50%", "44%"),
            ("3 of 6", "2 of 5"),
            ("1 of 2", "1 of 2"),
            ("1 of 2", "1 of 2"),
            ("3 of 7", "3 of 7"),
            ("1 of 1", "0 of 1"),
            ("1 of 2", "1 of 1"),
        ],
    ]
    return BeautifulSoup(
        '<a class="b-fight-details__collapse-link_rnd">Per round</a>'
        '<table><tbody class="b-fight-details__table-body">'
        + ''.join(_round_row(row) for row in totals)
        + '</tbody></table>'
        '<a class="b-fight-details__collapse-link_rnd">Per round</a>'
        '<table><tbody class="b-fight-details__table-body">'
        + ''.join(_round_row(row) for row in significant)
        + '</tbody></table>',
        'html.parser',
    )


def _raw_bout(fight_id="fight-1", event_id="event-1", date="2025-01-01"):
    base = {
        "date": date,
        "fight_url": f"http://ufcstats.test/fight-details/{fight_id}",
        "event_url": f"http://ufcstats.test/event-details/{event_id}",
        "division": "Lightweight",
        "method": "KO/TKO",
        "round": 2,
        "time": "1:30",
        "total_fight_time": 390,
        "source_card_index": 0,
        "bout_order": 0,
        "time_format": "3 Rnd (5-5-5)",
    }
    totals = {
        "knockdowns": (1, 0),
        "sig_strikes_landed": (15, 12),
        "sig_strikes_attempts": (30, 27),
        "total_strikes_landed": (18, 14),
        "total_strikes_attempts": (33, 29),
        "takedowns_landed": (1, 1),
        "takedowns_attempts": (3, 3),
        "sub_attempts": (1, 0),
        "reversals": (0, 1),
        "control": (90, 60),
        "head_strikes_landed": (9, 6),
        "head_strikes_attempts": (18, 14),
        "body_strikes_landed": (3, 3),
        "body_strikes_attempts": (6, 6),
        "leg_strikes_landed": (3, 3),
        "leg_strikes_attempts": (6, 7),
        "distance_strikes_landed": (10, 9),
        "distance_strikes_attempts": (22, 21),
        "clinch_strikes_landed": (2, 1),
        "clinch_strikes_attempts": (3, 3),
        "ground_strikes_landed": (3, 2),
        "ground_strikes_attempts": (5, 3),
    }
    rows = []
    for side, (fighter, opponent, result) in enumerate(
        (("A", "B", "W"), ("B", "A", "L"))
    ):
        row = dict(base)
        row.update(
            {
                "fighter": fighter,
                "opponent": opponent,
                "result": result,
                "fighter_url": f"http://ufcstats.test/fighter-details/fighter-{fighter.lower()}",
                "opponent_url": f"http://ufcstats.test/fighter-details/fighter-{opponent.lower()}",
            }
        )
        row.update({field: values[side] for field, values in totals.items()})
        rows.append(row)
    return pd.DataFrame(rows)


class RoundParsingTests(unittest.TestCase):
    def setUp(self):
        self.url = "http://ufcstats.test/fight-details/fight-1"
        self.partial = parse_ufcstats_round_stats(
            _round_fixture(), self.url, "3 Rnd (5-5-5)"
        )
        self.raw = _raw_bout()

    def test_parses_two_doubled_rounds_with_stable_source_ids(self):
        self.assertEqual(len(self.partial), 4)
        self.assertEqual(set(self.partial["round"]), {1, 2})
        self.assertEqual(set(self.partial["fighter_id"]), {"fighter-a", "fighter-b"})
        a_round_one = self.partial[
            self.partial["fighter_id"].eq("fighter-a")
            & self.partial["round"].eq(1)
        ].iloc[0]
        self.assertEqual(a_round_one["sig_strikes_landed"], 10)
        self.assertEqual(a_round_one["control"], 60)
        self.assertEqual(a_round_one["ground_strikes_attempts"], 3)

    def test_normalizes_metadata_and_exact_round_exposure(self):
        normalized = normalize_round_stats(self.partial, self.raw)
        self.assertEqual(tuple(normalized.columns), ROUND_DATA_COLUMNS)
        self.assertTrue(normalized["round_stat_id"].is_unique)
        a = normalized[normalized["fighter_id"].eq("fighter-a")]
        self.assertEqual(a["round_seconds"].tolist(), [300, 90])
        self.assertEqual(set(normalized["scheduled_rounds"]), {3})
        self.assertEqual(set(normalized["event_id"]), {"event-1"})

    def test_reconciles_round_sums_and_partitions_without_imputation(self):
        normalized = normalize_round_stats(self.partial, self.raw)
        annotated, report = reconcile_round_stats(normalized, self.raw)
        self.assertTrue(report.empty)
        self.assertEqual(set(annotated["reconciliation_status"]), {"matched"})

        missing = normalized.copy()
        missing.loc[
            missing["fighter_id"].eq("fighter-a") & missing["round"].eq(2),
            "knockdowns",
        ] = np.nan
        annotated, report = reconcile_round_stats(missing, self.raw)
        issue = report[
            report["fighter_id"].eq("fighter-a")
            & report["field"].eq("knockdowns")
        ].iloc[0]
        self.assertEqual(issue["issue"], "round_value_missing")
        self.assertTrue(pd.isna(issue["round_value"]))
        self.assertEqual(
            set(
                annotated.loc[
                    annotated["fighter_id"].eq("fighter-a"),
                    "reconciliation_status",
                ]
            ),
            {"unverifiable"},
        )

    def test_reports_source_mismatch_instead_of_rewriting_it(self):
        normalized = normalize_round_stats(self.partial, self.raw)
        altered = self.raw.copy()
        altered.loc[altered["fighter"].eq("A"), "knockdowns"] = 2
        annotated, report = reconcile_round_stats(normalized, altered)
        mismatch = report[
            report["fighter_id"].eq("fighter-a")
            & report["field"].eq("knockdowns")
        ].iloc[0]
        self.assertEqual(mismatch["issue"], "round_sum_mismatch")
        self.assertEqual(mismatch["bout_value"], 2)
        self.assertEqual(mismatch["round_value"], 1)
        self.assertEqual(mismatch["delta"], -1)
        self.assertEqual(
            set(
                annotated.loc[
                    annotated["fighter_id"].eq("fighter-a"),
                    "reconciliation_status",
                ]
            ),
            {"discrepancy"},
        )

    def test_schedule_helpers_do_not_assume_missing_round_lengths(self):
        self.assertEqual(declared_round_lengths_seconds("3 Rnd (5-5)"), (300, 300, 300))
        self.assertEqual(
            declared_round_lengths_seconds("1 Rnd + OT (12-3)"),
            (720, 180),
        )
        self.assertEqual(declared_round_lengths_seconds(""), ())
        self.assertIsNone(round_exposure_seconds(2, 2, "1:30", ""))

    def test_structural_validation_rejects_partial_doubled_round(self):
        normalized = normalize_round_stats(self.partial, self.raw)
        with self.assertRaisesRegex(ValueError, "two fighter rows"):
            validate_normalized_round_stats(normalized.iloc[:-1])


class RoundBackfillTests(unittest.TestCase):
    def test_backfill_rejects_runtime_budgets_over_one_hour_guardrail(self):
        handler = DataHandler.__new__(DataHandler)
        with self.assertRaisesRegex(ValueError, "max_runtime_seconds"):
            handler.backfill_ufc_fight_round_stats_doubled(
                max_fights=1, max_runtime_seconds=3301
            )

    def test_persistence_preserves_attempted_bout_without_parsed_rows(self):
        first = _raw_bout("fight-1", "event-1", "2025-02-01")
        second = _raw_bout("fight-2", "event-2", "2025-01-01")
        first_rounds = normalize_round_stats(
            parse_ufcstats_round_stats(
                _round_fixture(),
                "http://ufcstats.test/fight-details/fight-1",
                "3 Rnd (5-5-5)",
            ),
            first,
        )
        second_rounds = normalize_round_stats(
            parse_ufcstats_round_stats(
                _round_fixture(),
                "http://ufcstats.test/fight-details/fight-2",
                "3 Rnd (5-5-5)",
            ),
            second,
        )
        existing = pd.concat([first_rounds, second_rounds], ignore_index=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handler = DataHandler.__new__(DataHandler)
            handler.csv_filepaths = {
                "ufc_fight_round_stats_doubled": str(root / "rounds.csv"),
            }
            handler.round_reconciliation_filepath = str(root / "reconciliation.csv")
            handler.csv_data = {
                "ufc_fight_round_stats_doubled": existing,
            }

            # Both bouts were attempted, but only fight-1 parsed successfully.
            # fight-2 must retain its prior known-good rows.
            handler._persist_round_updates(
                first_rounds,
                pd.DataFrame(columns=[]),
                {"fight-1", "fight-2"},
            )

            stored = pd.read_csv(root / "rounds.csv")
            self.assertEqual(set(stored["fight_id"]), {"fight-1", "fight-2"})
            self.assertEqual(
                len(stored[stored["fight_id"].eq("fight-2")]),
                len(second_rounds),
            )
            validate_normalized_round_stats(stored)

    def test_production_update_does_not_request_or_persist_round_enrichment(self):
        existing = _raw_bout("fight-1", "event-1", "2025-01-01")
        incoming = _raw_bout("fight-2", "event-2", "2025-02-01")
        event_url = "http://ufcstats.test/event-details/event-2"
        page = SimpleNamespace(
            content=(
                "<table><tbody>"
                '<tr><td><a href="http://ufcstats.test/event-details/future">'
                "Future</a></td></tr>"
                f'<tr><td><a href="{event_url}">UFC Test</a></td></tr>'
                "</tbody></table>"
            ).encode("utf-8")
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handler = DataHandler.__new__(DataHandler)
            handler.csv_filepaths = {
                "ufc_fights_reported_doubled": str(root / "raw.csv"),
                "ufc_fight_round_stats_doubled": str(root / "rounds.csv"),
            }
            handler.csv_data = {
                "ufc_fights_reported_doubled": existing,
                "ufc_fight_round_stats_doubled": empty_round_stats_frame(),
            }
            handler.update_time = 0

            with (
                patch(
                    "data_handler.data_handler.ufcstats_client.get",
                    return_value=page,
                ),
                patch(
                    "data_handler.data_handler.get_event_fight_urls",
                    return_value=list(incoming["fight_url"].unique()),
                ),
                patch(
                    "data_handler.data_handler.get_fight_card",
                    return_value=incoming,
                ) as fetch_card,
                patch.object(handler, "_persist_round_updates") as persist_rounds,
            ):
                handler.update_ufc_fights_reported_doubled()

            fetch_card.assert_called_once_with(event_url)
            persist_rounds.assert_not_called()
            self.assertFalse((root / "rounds.csv").exists())
            self.assertEqual(handler.update_time, 1)

    def test_backfill_is_bounded_checkpointed_and_resumes_from_stored_ids(self):
        first = _raw_bout("fight-1", "event-1", "2025-02-01")
        second = _raw_bout("fight-2", "event-2", "2025-01-01")
        raw = pd.concat([first, second], ignore_index=True)
        partial_template = parse_ufcstats_round_stats(
            _round_fixture(),
            "http://ufcstats.test/fight-details/fight-1",
            "3 Rnd (5-5-5)",
        )

        def detail_result(url, *_args, **_kwargs):
            fight_id = url.rsplit("/", 1)[-1]
            bout = raw[raw["fight_url"].eq(url)]
            aggregate = bout[["fighter", "time_format", *[column for column in bout.columns if column.endswith(("_landed", "_attempts")) or column in {"knockdowns", "sub_attempts", "reversals", "control"}]]].copy()
            partial = partial_template.copy()
            partial["fight_id"] = fight_id
            partial["fight_url"] = url
            return aggregate, partial

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handler = DataHandler.__new__(DataHandler)
            handler.csv_filepaths = {
                "ufc_fights_reported_doubled": str(root / "raw.csv"),
                "ufc_fight_round_stats_doubled": str(root / "rounds.csv"),
            }
            handler.round_reconciliation_filepath = str(root / "reconciliation.csv")
            handler.csv_data = {
                "ufc_fights_reported_doubled": raw,
                "ufc_fight_round_stats_doubled": empty_round_stats_frame(),
            }

            with patch(
                "data_handler.data_handler.get_fight_stats", side_effect=detail_result
            ) as fetch:
                first_summary = handler.backfill_ufc_fight_round_stats_doubled(
                    max_fights=1, checkpoint_every=1
                )
                second_summary = handler.backfill_ufc_fight_round_stats_doubled(
                    max_fights=10, checkpoint_every=1
                )

            self.assertEqual(first_summary.saved_fights, 1)
            self.assertEqual(first_summary.remaining_fights, 1)
            self.assertEqual(second_summary.saved_fights, 1)
            self.assertEqual(second_summary.remaining_fights, 0)
            self.assertEqual(fetch.call_count, 2)
            stored = pd.read_csv(root / "rounds.csv")
            self.assertEqual(set(stored["fight_id"]), {"fight-1", "fight-2"})
            self.assertEqual(len(stored), 8)
            validate_normalized_round_stats(stored)


if __name__ == "__main__":
    unittest.main()
