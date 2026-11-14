import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import update_event_start_times as collector


HTML = '''<article class="c-card-event--result">
<a href="/event/ufc-334"></a>
<h3 class="c-card-event--result__headline">Gane vs Hokit</h3>
<div class="c-card-event--result__date" data-main-card-timestamp="1794708000"
data-prelims-card-timestamp="1794700800" data-early-card-timestamp="1794693600">
<time datetime="2026-11-14T21:00:00Z">9 PM EST</time></div></article>'''
EVENT = {"event_id": "334", "event_date": "2026-11-14", "event_title": "UFC 334: Gane vs. Hokit"}


class EventStartTimeTests(unittest.TestCase):
    def test_uses_epoch_times_not_misleading_wall_clock_datetime(self):
        schedule = collector.parse_schedule(HTML)
        result = collector.build_publication({"events": [EVENT]}, schedule, "2026-11-14T12:00:00Z")
        self.assertEqual(result["events"][0]["starts"], {
            "early_prelims": "2026-11-14T22:00:00Z", "prelims": "2026-11-15T00:00:00Z",
            "main_card": "2026-11-15T02:00:00Z"})

    def test_requires_identity_and_date_and_rejects_ambiguous_matches(self):
        schedule = collector.parse_schedule(HTML)
        self.assertIsNone(collector.match_event({**EVENT, "event_title": "UFC 335: Other vs Fight"}, schedule))
        self.assertIsNone(collector.match_event({**EVENT, "event_date": "2026-12-14"}, schedule))
        self.assertIsNone(collector.match_event(EVENT, schedule * 2))
        schedule[0]["source_url"] = "https://www.ufc.com/event/ufc-fight-night"
        self.assertIsNotNone(collector.match_event(EVENT, schedule))

    def test_missing_prelims_are_not_guessed(self):
        schedule = collector.parse_schedule(HTML.replace('data-prelims-card-timestamp="1794700800"', 'data-prelims-card-timestamp="TBA"'))
        self.assertNotIn("prelims", schedule[0]["starts"])
        with self.assertRaises(ValueError):
            collector.parse_schedule("<html>Temporarily unavailable</html>")

    def test_network_failure_preserves_last_confirmed_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = collector.build_publication({"events": [EVENT]}, collector.parse_schedule(HTML), "2026-11-13T12:00:00Z")
            (root / "all_upcoming_forecasts.json").write_text(json.dumps({"events": [EVENT]}))
            output = root / "event_start_times.json"
            output.write_text(json.dumps(original))
            with patch.object(collector, "ROOT", root), patch.object(collector.requests, "get", side_effect=requests.Timeout):
                collector.main()
            result = json.loads(output.read_text())
            self.assertTrue(result["refresh_failed"])
            self.assertEqual(result["events"], original["events"])
