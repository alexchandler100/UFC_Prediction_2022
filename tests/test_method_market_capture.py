from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
import tempfile
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bestfightodds_props import MethodPropSelection, PropBookPrice  # noqa: E402
from capture_market_snapshot import PublishedMatchup  # noqa: E402
from capture_method_market_snapshot import (  # noqa: E402
    _build_current_method_publication,
    _build_snapshots,
    _capture_is_due,
    _timed_horizon,
)
from market_tracker import (  # noqa: E402
    MethodMarketSnapshot,
    MethodMarketStore,
    matchup_id_for,
)


def _published() -> PublishedMatchup:
    return PublishedMatchup(
        fighter_name="Fighter Alpha",
        opponent_name="Fighter Beta",
        fighter_id="a-fighter",
        opponent_id="b-opponent",
        matchup_id=matchup_id_for("event-1", "a-fighter", "b-opponent"),
        fight_id=None,
        model_probability=0.6,
        model_status="available",
        forecast_issued_at_utc="2026-08-20T00:00:00Z",
        forecast_source_commit="a" * 40,
        bayesian_model_id="bayes",
        bayesian_status="available",
        bayesian_credible_level=0.9,
        bayesian_posterior_mean=0.6,
        bayesian_posterior_median=0.6,
        bayesian_probability_lower=0.5,
        bayesian_probability_upper=0.7,
        bayesian_calibrated_logit_location=0.0,
        bayesian_calibrated_logit_scale=1.0,
    )


def _snapshot(**overrides: object) -> MethodMarketSnapshot:
    values: dict[str, object] = {
        "capture_id": "method-capture-1",
        "event_id": "event-1",
        "fighter_id": "a-fighter",
        "opponent_id": "b-opponent",
        "fighter_name": "Fighter Alpha",
        "opponent_name": "Fighter Beta",
        "event_date": "2026-08-29",
        "timing_precision": "timestamp",
        "event_start_utc": "2026-08-29T07:00:00Z",
        "observed_at_utc": "2026-08-28T07:00:00Z",
        "source": "bestfightodds.com",
        "source_event_id": "matchup_900",
        "source_book_key": "book_21",
        "book": "Book A",
        "horizon": "t24",
        "fighter_prices": {"ko_tko": 250, "submission": 500, "decision": 200},
        "opponent_prices": {"ko_tko": 400, "submission": 700, "decision": 300},
        "source_payload_sha256": "a" * 64,
    }
    values.update(overrides)
    return MethodMarketSnapshot.create(**values)


class MethodMarketTests(unittest.TestCase):
    def test_revised_start_keeps_immutable_prices_and_uses_current_display_time(self):
        older = _snapshot()
        revised = _snapshot(capture_id="revised", event_start_utc="2026-08-29T07:10:00Z",
                            observed_at_utc="2026-08-28T07:10:00Z", fighter_prices={"ko_tko": 300})
        before = [row.to_mapping() for row in (older, revised)]
        publication = _build_current_method_publication(
            (older, revised), event_id="event-1", event_date="2026-08-29",
            event_start_utc="2026-08-29T07:00:00Z", outcome_forecasts=None,
        )
        self.assertEqual(publication["event_start_utc"], "2026-08-29T07:00:00Z")
        self.assertEqual(publication["book_market_count"], 1)
        self.assertEqual(publication["latest_observed_at_utc"], revised.observed_at_utc)
        self.assertEqual([row.to_mapping() for row in (older, revised)], before)

    def test_event_date_and_contract_conflicts_still_fail(self):
        valid = _snapshot()
        for changed in (replace(valid, event_date="2026-08-30"), replace(valid, contract_version="different-contract")):
            with self.subTest(changed=changed.contract_version):
                with self.assertRaisesRegex(Exception, "conflicting event contracts"):
                    _build_current_method_publication(
                        (valid, changed), event_id="event-1", event_date="2026-08-29",
                        event_start_utc="2026-08-29T07:00:00Z", outcome_forecasts=None,
                    )

    def test_complete_market_is_normalized_and_round_trips(self):
        snapshot = _snapshot()
        total = sum(
            getattr(snapshot, f"{side}_{method}_no_vig_probability")
            for side in ("fighter", "opponent")
            for method in ("ko_tko", "submission", "decision")
        )
        self.assertAlmostEqual(total, 1.0)
        rebuilt = MethodMarketSnapshot.from_mapping(snapshot.to_mapping())
        self.assertEqual(rebuilt, snapshot)

    def test_partial_board_preserves_missing_prices_without_imputation(self):
        snapshot = _snapshot(
            fighter_prices={"ko_tko": 250, "decision": 200},
            opponent_prices={"submission": 700},
        )
        self.assertEqual(snapshot.selection_count, 3)
        self.assertFalse(snapshot.is_complete_six_way)
        self.assertIsNone(snapshot.fighter_submission_moneyline)
        self.assertIsNone(snapshot.six_way_overround)
        self.assertIsNone(snapshot.fighter_ko_tko_no_vig_probability)
        self.assertIsNotNone(snapshot.fighter_ko_tko_implied_probability)
        self.assertEqual(
            MethodMarketSnapshot.from_mapping(snapshot.to_mapping()), snapshot
        )

    def test_store_is_append_only_by_fight_horizon_and_book(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = MethodMarketStore(root / "method.csv", root / "method.jsonl")
            snapshot = _snapshot()
            first = store.append([snapshot])
            second = store.append([snapshot])
            self.assertEqual(len(first.added_ids), 1)
            self.assertEqual(len(second.duplicate_ids), 1)
            self.assertEqual(store.read(), (snapshot,))
            with self.assertRaises(Exception):
                store.append(
                    [
                        _snapshot(
                            capture_id="different-capture",
                            fighter_prices={
                                "ko_tko": 260,
                                "submission": 500,
                                "decision": 200,
                            },
                        )
                    ]
                )

    def test_horizon_windows_are_bounded(self):
        self.assertEqual(_timed_horizon(70 * 3600), "t72")
        self.assertEqual(_timed_horizon(22 * 3600), "t24")
        self.assertEqual(_timed_horizon(4 * 3600), "t6")
        self.assertIsNone(_timed_horizon(40 * 3600))

    def test_a_stored_horizon_does_not_make_another_source_request(self):
        opening = _snapshot(horizon="opening")
        t24 = _snapshot(
            capture_id="method-capture-2",
            horizon="t24",
            observed_at_utc="2026-08-28T07:30:00Z",
        )
        self.assertFalse(_capture_is_due((opening,), None))
        self.assertTrue(_capture_is_due((opening,), "t24"))
        self.assertFalse(_capture_is_due((opening, t24), "t24"))

    def test_source_side_is_aligned_to_published_fighter(self):
        # Source order is Beta, Alpha while the published order is Alpha, Beta.
        selections = []
        prop_type = 8
        for source_side in (1, 2):
            for method, base_price in (
                ("ko_tko", 200),
                ("submission", 500),
                ("decision", 150),
            ):
                price = base_price + source_side * 50
                selections.append(
                    MethodPropSelection(
                        source_matchup_id=900,
                        source_fighter_side=source_side,
                        source_prop_type_id=prop_type,
                        source_outcome_number=1,
                        fighter_1_name="Fighter Beta",
                        fighter_2_name="Fighter Alpha",
                        market="fighter_method_of_victory",
                        method=method,
                        raw_label=f"selection {source_side} {method}",
                        mean_history_available=True,
                        book_prices=(PropBookPrice(21, "Book A", price),),
                    )
                )
                prop_type += 1
        snapshots, counters = _build_snapshots(
            selections=selections,
            published=(_published(),),
            event_day="2026-08-29",
            event_id="event-1",
            event_start_utc="2026-08-29T07:00:00Z",
            observed=datetime(2026, 8, 28, 7, tzinfo=timezone.utc),
            payload_sha="b" * 64,
            existing=(),
            timed_horizon="t24",
        )
        self.assertEqual(len(snapshots), 2)  # opening plus T-24
        self.assertEqual(counters["matched_fights_with_prices"], 1)
        # Alpha is source side 2, so its KO line is the side-2 price.
        self.assertEqual(snapshots[0].fighter_ko_tko_moneyline, 300)
        self.assertEqual(snapshots[0].opponent_ko_tko_moneyline, 250)

    def test_current_publication_aligns_prices_to_forecast_orientation(self):
        snapshot = _snapshot()
        publication = _build_current_method_publication(
            (snapshot,),
            event_id="event-1",
            event_date="2026-08-29",
            event_start_utc="2026-08-29T07:00:00Z",
            outcome_forecasts={
                "forecast_issued_at_utc": "2026-08-20T00:00:00Z",
                "matchups": [
                    {
                        "fighter_id": "b-opponent",
                        "opponent_id": "a-fighter",
                        "fighter_name": "Fighter Beta",
                        "opponent_name": "Fighter Alpha",
                        "bout_order": 2,
                        "terminal_probabilities": {
                            "fighter_ko_tko": 0.10,
                            "fighter_submission": 0.05,
                            "fighter_decision": 0.25,
                            "opponent_ko_tko": 0.20,
                            "opponent_submission": 0.10,
                            "opponent_decision": 0.30,
                        },
                    }
                ]
            },
        )
        market = publication["method_markets"][0]
        self.assertEqual(market["fighter_name"], "Fighter Beta")
        selections = market["book_quotes"][0]["selections"]
        beta_ko = next(
            row
            for row in selections
            if row["side"] == "fighter" and row["method"] == "ko_tko"
        )
        self.assertEqual(beta_ko["moneyline"], 400)
        self.assertEqual(beta_ko["candidate_model_probability"], 0.10)
        self.assertEqual(publication["expected_value_status"], "candidate_comparison_only")
        self.assertEqual(len(publication["publication_sha256"]), 64)

    def test_future_or_missing_forecast_time_keeps_prices_but_withholds_model(self):
        for issued in ("2026-08-28T07:00:01Z", None):
            with self.subTest(issued=issued):
                publication = _build_current_method_publication(
                    (_snapshot(),), event_id="event-1", event_date="2026-08-29",
                    event_start_utc="2026-08-29T07:00:00Z",
                    outcome_forecasts={"forecast_issued_at_utc": issued, "matchups": [{
                        "fighter_id": "b-opponent", "opponent_id": "a-fighter",
                        "terminal_probabilities": {"fighter_ko_tko": 0.1},
                    }]},
                )
                self.assertEqual(publication["expected_value_status"], "unavailable")
                markets = publication["method_markets"]
                self.assertEqual(len(markets), 1)
                selections = markets[0]["book_quotes"][0]["selections"]
                self.assertTrue(selections)
                self.assertTrue(all(row["candidate_model_probability"] is None for row in selections))

    def test_duplicate_source_matchups_merge_complementary_sides(self):
        selections = []
        prop_type = 20
        for source_matchup_id, side in ((899, 1), (900, 2)):
            for method in ("ko_tko", "submission", "decision"):
                selections.append(
                    MethodPropSelection(
                        source_matchup_id=source_matchup_id,
                        source_fighter_side=side,
                        source_prop_type_id=prop_type,
                        source_outcome_number=1,
                        fighter_1_name="Fighter Alpha",
                        fighter_2_name="Fighter Beta",
                        market="fighter_method_of_victory",
                        method=method,
                        raw_label=f"{side} {method}",
                        mean_history_available=True,
                        book_prices=(PropBookPrice(21, "Book A", 200 + prop_type),),
                    )
                )
                prop_type += 1
        snapshots, counters = _build_snapshots(
            selections=selections,
            published=(_published(),),
            event_day="2026-08-29",
            event_id="event-1",
            event_start_utc="2026-08-29T07:00:00Z",
            observed=datetime(2026, 8, 28, 7, tzinfo=timezone.utc),
            payload_sha="c" * 64,
            existing=(),
            timed_horizon=None,
        )
        self.assertEqual(len(snapshots), 1)
        self.assertTrue(snapshots[0].is_complete_six_way)
        self.assertEqual(snapshots[0].source_event_id, "matchups_899_900")
        self.assertEqual(counters["source_duplicate_matchups_merged"], 1)


if __name__ == "__main__":
    unittest.main()
