from __future__ import annotations

import base64
from datetime import date
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_historical_odds_sources import (  # noqa: E402
    SitemapEvent,
    _rot47,
    decode_bestfightodds_chart,
    parse_bestfightodds_event_page,
    parse_bestfightodds_sitemap,
    select_stratified_events,
    summarize_chart,
)


class HistoricalOddsAuditTests(unittest.TestCase):
    def test_chart_decoder_preserves_absolute_timestamps_and_prices(self):
        value = [
            {
                "name": "Book A",
                "data": [
                    {"x": 1691409159000, "y": 1.91},
                    {"x": 1692484384000, "y": 2.05},
                ],
            }
        ]
        encoded = base64.b64encode(
            _rot47(json.dumps(value, separators=(",", ":"))).encode("utf-8")
        ).decode("ascii")

        decoded = decode_bestfightodds_chart(encoded)
        summary = summarize_chart(decoded, event_date=date(2023, 8, 20))

        self.assertEqual(decoded, value)
        self.assertEqual(summary["point_count"], 2)
        self.assertEqual(summary["strict_pre_event_point_count"], 2)
        self.assertEqual(summary["first_point_utc"], "2023-08-07T11:52:39Z")
        self.assertTrue(summary["finite_decimal_prices"])

    def test_event_page_extracts_paired_book_prices_and_ignores_props(self):
        html = """
        <html><head><script type="application/ld+json">
        {"@type":"SportsEvent","name":"UFC Test","startDate":"2023-08-19",
         "organizer":{"@type":"Organization","name":"UFC"}}
        </script></head><body>
        <table class="odds-table">
          <tr><th></th><th data-b="21">Book A</th><th data-b="22">Book B</th></tr>
          <tr><th scope="row"><a href="/fighters/a-1">Fighter A</a></th>
            <td class="but-sg" data-li="[21,1,31608]"><span>-125</span>▲</td>
            <td class="but-sg" data-li="[22,1,31608]"><span>-120</span></td></tr>
          <tr><th scope="row"><a href="/fighters/b-2">Fighter B</a></th>
            <td class="but-sg" data-li="[21,2,31608]"><span>+105</span>▼</td>
            <td class="but-sg" data-li="[22,2,31608]"><span>+100</span></td></tr>
          <tr class="pr"><th scope="row">Over 2.5 rounds</th>
            <td class="but-sgp" data-li="[21,31608,1,2,3]">-110</td></tr>
        </table><div>Line movement</div></body></html>
        """

        parsed = parse_bestfightodds_event_page(
            html,
            url="https://www.bestfightodds.com/events/ufc-test-1",
            sitemap_date=date(2023, 8, 19),
        )

        self.assertTrue(parsed["is_ufc"])
        self.assertEqual(parsed["matchup_count"], 1)
        self.assertEqual(parsed["book_count"], 2)
        matchup = parsed["matchups"][0]
        self.assertEqual(matchup["fighter_1"], "Fighter A")
        self.assertEqual(matchup["fighter_2"], "Fighter B")
        self.assertEqual(len(matchup["paired_books"]), 2)
        self.assertTrue(matchup["paired_books"][0]["side_1_present"])
        self.assertTrue(parsed["has_line_movement_control"])

    def test_event_page_keeps_legacy_mean_only_matchup(self):
        html = """
        <html><head><script type="application/ld+json">
        {"@type":"SportsEvent","name":"UFC Old","startDate":"2012-01-14",
         "organizer":{"name":"UFC"}}
        </script></head><body><table class="odds-table">
          <tr><th></th><th data-b="21">Current Book</th></tr>
          <tr><th scope="row"><a href="/fighters/a-1">Fighter A</a></th>
            <td></td><td class="button-cell but-si" data-li="[1,4682]">chart</td></tr>
          <tr><th scope="row"><a href="/fighters/b-2">Fighter B</a></th>
            <td></td><td class="button-cell but-si" data-li="[2,4682]">chart</td></tr>
        </table></body></html>
        """
        parsed = parse_bestfightodds_event_page(
            html,
            url="https://www.bestfightodds.com/events/ufc-old-1",
            sitemap_date=date(2012, 1, 14),
        )
        self.assertEqual(parsed["matchup_count"], 1)
        self.assertEqual(parsed["matchups"][0]["fighter_1"], "Fighter A")
        self.assertEqual(parsed["matchups"][0]["fighter_2"], "Fighter B")
        self.assertEqual(parsed["matchups"][0]["paired_books"], [])

    def test_plain_empty_chart_response_is_supported(self):
        self.assertEqual(decode_bestfightodds_chart("[]"), [])

    def test_sitemap_selection_is_deterministic_and_spans_years(self):
        rows = []
        for year in (2022, 2023, 2024):
            for number in range(5):
                rows.append(
                    f"<url><loc>https://www.bestfightodds.com/events/ufc-{year}-{number}</loc>"
                    f"<lastmod>{year}-0{number + 1}-01</lastmod></url>"
                )
        xml = "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>" + "".join(rows) + "</urlset>"
        events = parse_bestfightodds_sitemap(xml)

        first = select_stratified_events(
            events,
            sample_size=9,
            minimum_year=2022,
            maximum_date=date(2024, 12, 31),
        )
        second = select_stratified_events(
            events,
            sample_size=9,
            minimum_year=2022,
            maximum_date=date(2024, 12, 31),
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        self.assertEqual({item.event_date.year for item in first}, {2022, 2023, 2024})

    def test_invalid_chart_values_are_rejected(self):
        value = [{"name": "Book", "data": [{"x": 1, "y": float("inf")}]}]
        encoded = base64.b64encode(
            _rot47(json.dumps(value)).encode("utf-8")
        ).decode("ascii")
        with self.assertRaisesRegex(ValueError, "invalid timestamp or price"):
            decode_bestfightodds_chart(encoded)

    def test_event_dataclass_mapping_is_stable(self):
        event = SitemapEvent(
            url="https://www.bestfightodds.com/events/ufc-test-1",
            event_date=date(2020, 1, 1),
        )
        self.assertEqual(event.to_mapping()["event_date"], "2020-01-01")


if __name__ == "__main__":
    unittest.main()
