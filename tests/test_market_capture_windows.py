import json
from pathlib import Path
import sys
import unittest
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_capture_windows import capture_due, price_window_report, utc
from test_market_first_paper import quote, forecast


class CaptureWindowTests(unittest.TestCase):
    def setUp(self):
        self.card = {"event_id": "future-event"}
        self.previous = {"event_id": "future-event", "event_start_utc": "2026-09-05T12:00:00Z",
                         "captured_at_utc": "2026-09-03T12:00:00Z"}

    def test_event_clock_drives_day_before_and_final_capture(self):
        self.assertTrue(capture_due(self.card, self.previous, "2026-09-04T12:00:00Z")["capture"])
        self.assertTrue(capture_due(self.card, self.previous, "2026-09-05T11:00:00Z")["capture"])
        self.assertFalse(capture_due(self.card, self.previous, "2026-09-05T11:40:00Z")["capture"])
        self.assertFalse(capture_due(self.card, self.previous, "2026-09-05T12:00:00Z")["capture"])
        self.assertFalse(capture_due(self.card, {}, "2026-09-04T12:00:00Z")["capture"])
        self.assertFalse(capture_due({"event_id": "new"}, self.previous, "2026-09-04T12:00:00Z")["capture"])

    def test_existing_window_capture_avoids_repeat_api_requests(self):
        for now, captured in [("2026-09-04T12:00:00Z", "2026-09-04T11:50:00Z"),
                              ("2026-09-05T11:00:00Z", "2026-09-05T10:45:00Z")]:
            result = capture_due(self.card, dict(self.previous, captured_at_utc=captured), now)
            self.assertEqual(result["reason"], "window_already_captured")

    def capture(self, observed, *, reverse=False):
        qs = [quote(book, 150, -180, observed=observed).to_mapping() for book in ("A", "B", "C", "Target")]
        ms = []
        for q in qs:
            m = {key: q[key] for key in ("quote_id", "capture_id", "matchup_id", "event_id", "book", "source", "observed_at_utc")}
            m.update(source_quote_updated_at_utc=observed, source_commence_time_utc="2026-09-05T16:00:00Z")
            ms.append(m)
            if reverse:
                q["fighter_id"], q["opponent_id"] = q["opponent_id"], q["fighter_id"]
                q["fighter_moneyline"], q["opponent_moneyline"] = q["opponent_moneyline"], q["fighter_moneyline"]
                q["no_vig_fighter_probability"] = 1 - q["no_vig_fighter_probability"]
        return qs, ms

    def test_missing_final_never_uses_an_older_price(self):
        quotes, source = self.capture("2026-09-05T09:00:00Z")
        result = price_window_report({"moneyline": quotes}, source, [forecast().to_mapping()], [], utc("2026-09-06T12:00:00Z"))
        self.assertEqual(result["coverage"][0]["windows"]["final"]["status"], "missed")

    def test_final_reference_keeps_orientation_excludes_target_and_uses_fresh_quotes(self):
        early, em = self.capture("2026-09-04T12:00:00Z")
        final, fm = self.capture("2026-09-05T11:00:00Z", reverse=True)
        # Real captures have distinct IDs; the fixture helper uses a fixed one.
        for q, m in zip(final, fm):
            q["capture_id"] = m["capture_id"] = "final"
        target = next(q for q in early if q["book"] == "Target")
        decision = {"decision_id": "d", "event_id": target["event_id"], "matchup_id": target["matchup_id"],
            "reference_quote_id": target["quote_id"], "paper_action": "fighter", "fighter_id": "fighter-a",
            "opponent_id": "fighter-b", "action_reference_moneyline": 200,
            "decision_issued_at_utc": "2026-09-04T12:01:00Z"}
        def build(qs=final, ms=fm):
            return price_window_report({"moneyline": early + qs}, em + ms, [], [decision], utc("2026-09-06T12:00:00Z"))
        result = build()
        ref = result["moneyline_references"][0]
        self.assertEqual(ref["status"], "available")
        self.assertEqual(ref["moneyline"], 150)
        self.assertEqual(ref["other_books"], ["a", "b", "c"])
        self.assertGreater(ref["independent_probability_advantage"], 0)
        altered = deepcopy(final)
        altered[-1]["no_vig_fighter_probability"] = .99
        self.assertEqual(build(altered)["moneyline_references"][0]["independent_probability"], ref["independent_probability"])
        stale = deepcopy(fm)
        stale[-1]["source_quote_updated_at_utc"] = "2026-09-05T10:00:00Z"
        self.assertEqual(build(ms=stale)["moneyline_references"][0]["status"], "missing_final_same_book_quote")
        started = deepcopy(fm)
        for m in started: m["source_commence_time_utc"] = "2026-09-05T11:00:00Z"
        self.assertEqual(build(ms=started)["rejected_quotes"]["card_or_bout_started"], 4)
        future_update = deepcopy(fm)
        future_update[-1]["source_quote_updated_at_utc"] = "2026-09-05T11:01:00Z"
        self.assertEqual(build(ms=future_update)["moneyline_references"][0]["status"], "missing_final_same_book_quote")
        separate_capture = deepcopy(final)
        separate_source = deepcopy(fm)
        for q, m in zip(separate_capture[:3], separate_source[:3]):
            q["capture_id"] = m["capture_id"] = "another-capture"
        self.assertEqual(build(separate_capture, separate_source)["moneyline_references"][0]["status"], "fewer_than_three_other_books")
        # Publication round trip remains exactly reproducible.
        self.assertEqual(json.loads(json.dumps(result)), result)
        policies = price_window_report({'moneyline': early + final}, em + fm, [], [decision],
            utc('2026-09-06T12:00:00Z'), additional_decisions={'market_first': [decision]})
        self.assertEqual(len(policies['moneyline_references']), 2)
        self.assertEqual({r['strategy'] for r in policies['moneyline_references']}, {'locked_market', 'market_first'})
        for summary in policies['moneyline_reference_summary'].values():
            self.assertEqual(summary['recorded_bets'], 1)
            self.assertEqual(summary['independent_references'], 1)
            self.assertEqual(summary['missing_same_book_references'], 0)

    def test_totals_coverage_keeps_lines_separate_and_ignores_partial_periods(self):
        qs, ms = self.capture("2026-09-05T11:00:00Z")
        totals = []
        for i, (q, m) in enumerate(zip(qs[:3], ms[:3])):
            totals.append({**q, **m, "line": 1.5 if i == 0 else 2.5,
                           "period": "round_1" if i == 2 else "full_fight"})
        totals[1]["source_quote_updated_at_utc"] = "2026-09-05T10:00:00Z"
        result = price_window_report({"total_rounds": totals}, [], [], [], utc("2026-09-06T12:00:00Z"))
        self.assertEqual(len(result["coverage"]), 2)
        by_line = {r["line"]: r["windows"]["final"] for r in result["coverage"]}
        self.assertEqual(by_line[1.5]["status"], "captured")
        self.assertEqual(by_line[2.5]["status"], "missed")
        self.assertEqual(result["rejected_quotes"], {"stale_or_future_source_quote": 1})

    def test_absent_quotes_and_unknown_start_are_not_passes(self):
        f = forecast().to_mapping()
        f["event_start_utc"] = None
        report = price_window_report({}, [], [f], [], utc("2026-09-06T12:00:00Z"))
        self.assertEqual(report["coverage"][0]["windows"]["t24"]["status"], "unknown_start")


if __name__ == "__main__":
    unittest.main()
