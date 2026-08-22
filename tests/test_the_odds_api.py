import os
from copy import deepcopy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from odds_getter import (  # noqa: E402
    OddsApiError,
    OddsApiResponse,
    OddsGetter,
    TheOddsApiClient,
)


class _Response:
    def __init__(self, payload, *, status=200, headers=None):
        self._payload = payload
        self.status_code = status
        self.headers = headers or {}

    def json(self):
        return self._payload


class _Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class _FailingSession:
    def get(self, url, **kwargs):
        raise requests.Timeout(
            f"timed out requesting {url}?apiKey=must-never-be-published"
        )


def _fixture_payload():
    return [
        {
            "id": "odds-event-one",
            "sport_key": "mma_mixed_martial_arts",
            "commence_time": "2026-08-22T23:00:00Z",
            "home_team": "Alpha Fighter",
            "away_team": "Beta Fighter",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": "2026-08-20T18:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-08-20T18:01:00Z",
                            "outcomes": [
                                {"name": "Beta Fighter", "price": 125},
                                {"name": "Alpha Fighter", "price": -145},
                            ],
                        },
                        {
                            "key": "totals",
                            "last_update": "2026-08-20T18:02:00Z",
                            "outcomes": [
                                {"name": "Over", "point": 2.5, "price": -110},
                                {"name": "Under", "point": 2.5, "price": -105},
                            ],
                        },
                    ],
                },
                {
                    "key": "incomplete",
                    "title": "Incomplete Book",
                    "last_update": "2026-08-20T18:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Alpha Fighter", "price": -150}
                            ],
                        }
                    ],
                },
            ],
        }
    ]


class TheOddsApiTests(unittest.TestCase):
    def test_normalizes_mma_moneylines_and_quota_without_exposing_key(self):
        session = _Session(
            _Response(
                _fixture_payload(),
                headers={
                    "x-requests-remaining": "499",
                    "x-requests-used": "1",
                    "x-requests-last": "2",
                },
            )
        )
        result = TheOddsApiClient(session).fetch(
            "super-secret-key", regions="us,us2"
        )
        self.assertEqual(len(result.frame), 1)
        row = result.frame.iloc[0]
        self.assertEqual(row["source event id"], "odds-event-one")
        self.assertEqual(row["fighter name"], "Alpha Fighter")
        self.assertEqual(row["opponent name"], "Beta Fighter")
        self.assertEqual(row["fighter DraftKings"], -145)
        self.assertEqual(row["opponent DraftKings"], 125)
        self.assertNotIn("fighter Incomplete Book", result.frame.columns)
        self.assertEqual(
            row["source DraftKings last update"], "2026-08-20T18:01:00Z"
        )
        self.assertEqual(result.quota_mapping()["requests_remaining"], 499)
        _, kwargs = session.calls[0]
        self.assertEqual(kwargs["params"]["markets"], "h2h")
        self.assertEqual(kwargs["params"]["regions"], "us,us2")
        self.assertEqual(kwargs["params"]["oddsFormat"], "american")

    def test_missing_or_rejected_key_has_clear_sanitized_error(self):
        with self.assertRaisesRegex(OddsApiError, "THE_ODDS_API_KEY is missing"):
            TheOddsApiClient(_Session(_Response([]))).fetch("")

        session = _Session(_Response({}, status=401))
        with self.assertRaises(OddsApiError) as caught:
            TheOddsApiClient(session).fetch("do-not-print-this")
        self.assertIn("authentication was rejected", str(caught.exception))
        self.assertNotIn("do-not-print-this", str(caught.exception))

    def test_optionally_normalizes_full_fight_total_rounds(self):
        session = _Session(_Response(_fixture_payload()))
        result = TheOddsApiClient(session).fetch(
            "fixture-key", include_total_rounds=True
        )

        self.assertIsNotNone(result.total_rounds_frame)
        self.assertEqual(len(result.total_rounds_frame), 1)
        row = result.total_rounds_frame.iloc[0]
        self.assertEqual(row["market"], "total_rounds")
        self.assertEqual(row["period"], "full_fight")
        self.assertEqual(row["line"], 2.5)
        self.assertEqual(row["over moneyline"], -110)
        self.assertEqual(row["under moneyline"], -105)
        self.assertEqual(row["source book key"], "draftkings")
        self.assertEqual(row["source last update"], "2026-08-20T18:02:00Z")
        _, kwargs = session.calls[0]
        self.assertEqual(kwargs["params"]["markets"], "h2h,totals")

    def test_rejects_duplicate_event_ids(self):
        duplicate = _fixture_payload() * 2
        with self.assertRaisesRegex(OddsApiError, "repeated event ID"):
            TheOddsApiClient(_Session(_Response(duplicate))).fetch("key")

    def test_rejects_wrong_sport_bad_timestamp_and_fractional_moneyline(self):
        mutations = (
            ("sport_key", "not_mma", "non-MMA"),
            ("commence_time", "not-a-time", "commence_time"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                payload = deepcopy(_fixture_payload())
                payload[0][field] = value
                with self.assertRaisesRegex(OddsApiError, message):
                    TheOddsApiClient(_Session(_Response(payload))).fetch("key")

        payload = deepcopy(_fixture_payload())
        payload[0]["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = 125.5
        with self.assertRaisesRegex(OddsApiError, "invalid.*odds"):
            TheOddsApiClient(_Session(_Response(payload))).fetch("key")

    def test_network_error_does_not_expose_key_or_chained_request_url(self):
        with self.assertRaises(OddsApiError) as caught:
            TheOddsApiClient(_FailingSession()).fetch("must-never-be-published")
        self.assertEqual(
            str(caught.exception),
            "The Odds API request failed (Timeout)",
        )
        self.assertIsNone(caught.exception.__cause__)

    def test_configured_getter_routes_api_and_builds_no_vig_consensus(self):
        api_frame = TheOddsApiClient(
            _Session(_Response(_fixture_payload()))
        ).fetch("fixture-key").frame
        api_response = OddsApiResponse(
            frame=api_frame,
            payload=_fixture_payload(),
            requests_remaining=498,
            requests_used=2,
            request_cost=2,
        )
        with patch.dict(
            os.environ,
            {
                "MARKET_ODDS_SOURCE": "the-odds-api",
                "THE_ODDS_API_KEY": "workflow-secret",
                "ODDS_API_REGIONS": "us,us2",
            },
        ), patch(
            "odds_getter.odds_getter.TheOddsApiClient.fetch",
            return_value=api_response,
        ) as fetch:
            getter = OddsGetter()
            result = getter.make_odds_df()
        self.assertEqual(getter.last_source, "the-odds-api.com")
        self.assertEqual(getter.last_request_metadata["requests_remaining"], 498)
        self.assertAlmostEqual(
            result.iloc[0]["average bookie probability"],
            (-145 / (-145 - 100))
            / ((-145 / (-145 - 100)) + (100 / (125 + 100))),
        )
        self.assertEqual(result.iloc[0]["average bookie odds"], [-133, 133])
        fetch.assert_called_once_with("workflow-secret", regions="us,us2")


if __name__ == "__main__":
    unittest.main()
