import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from odds_getter import OddsGetter
from data_handler import DataHandler
from fight_predictor import FightPredictor
import fight_stat_helpers
from fight_stat_helpers import (
    count_losses_losses_before_fight,
    count_wins_wins_before_fight,
    get_kelly_bet_from_ev_and_dk_odds,
)
from ufcstats_client import (
    RequestTimeout,
    UFCStatsClient,
    UFCStatsError,
    UFCStatsEventNotComplete,
)
from validate_data import validate_raw_fights


def make_response(body: str, url: str = "http://ufcstats.test/events", status: int = 200):
    response = requests.Response()
    response.status_code = status
    response.url = url
    response.encoding = "utf-8"
    response._content = body.encode("utf-8")
    return response


class FakeSession:
    def __init__(self, get_responses):
        self.headers = {}
        self.get_responses = list(get_responses)
        self.post_calls = []

    def mount(self, *_args, **_kwargs):
        return None

    def get(self, _url, **_kwargs):
        return self.get_responses.pop(0)

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return make_response("", url=url, status=204)


class UFCStatsClientTests(unittest.TestCase):
    def test_solves_browser_check_and_retries_original_page(self):
        challenge = make_response(
            '<title>Loading…</title><p>Checking your browser…</p>'
            '<script>var nonce="fixture", target=new Array(1+1).join("0");</script>'
        )
        page = make_response('<table class="expected-table"></table>')
        session = FakeSession([challenge, page])
        client = UFCStatsClient(
            session=session,
            timeout=RequestTimeout(1, 1),
            max_pow_attempts=10_000,
            min_request_interval=0,
        )

        result = client.get("http://ufcstats.test/events", expected_text="expected-table")

        self.assertIs(result, page)
        self.assertEqual(len(session.post_calls), 1)
        self.assertEqual(session.post_calls[0][0], "http://ufcstats.test/__c")
        self.assertEqual(session.post_calls[0][1]["data"]["nonce"], "fixture")

    def test_rejects_status_200_wrong_page(self):
        session = FakeSession([make_response("<title>Error</title>")])
        client = UFCStatsClient(session=session, min_request_interval=0)
        with self.assertRaisesRegex(UFCStatsError, "did not contain"):
            client.get("http://ufcstats.test/events", expected_text="expected-table")

    def test_defers_an_incomplete_future_event_instead_of_ingesting_it(self):
        future_date = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%B %d, %Y")
        response = make_response(
            '<table class="b-fight-details__table"></table>'
            f'<li class="b-list__box-list-item">{future_date}</li>'
        )

        class StaticClient:
            def get(self, *_args, **_kwargs):
                return response

        with patch.object(fight_stat_helpers, "ufcstats_client", StaticClient()):
            with self.assertRaises(UFCStatsEventNotComplete):
                fight_stat_helpers.get_fight_card("http://ufcstats.test/event")


class OddsTests(unittest.TestCase):
    def test_consensus_removes_vig_in_probability_space(self):
        row = pd.Series(
            {
                "fighter BookA": "-110",
                "opponent BookA": "-110",
                "fighter BookB": "+120",
                "opponent BookB": "-140",
            }
        )
        consensus = OddsGetter().get_consensus_odds(row, ["BookA", "BookB"])
        fighter_probability = OddsGetter.odds_to_probability(consensus[0])
        self.assertGreater(fighter_probability, 0.46)
        self.assertLess(fighter_probability, 0.48)

    def test_parses_common_moneyline_formats(self):
        self.assertEqual(OddsGetter.parse_american_odds("EVEN"), 100)
        self.assertEqual(OddsGetter.parse_american_odds("−125"), -125)
        self.assertIsNone(OddsGetter.parse_american_odds(""))

    def test_bet_sizing_is_fractional_nonnegative_and_capped(self):
        fighter, opponent = get_kelly_bet_from_ev_and_dk_odds(-1000, 300, -350)
        self.assertGreaterEqual(fighter, 0)
        self.assertGreaterEqual(opponent, 0)
        self.assertLessEqual(fighter, 5.0)
        self.assertLessEqual(opponent, 5.0)


class RawValidationTests(unittest.TestCase):
    def make_raw(self):
        return pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "fight_url": "fight-1",
                    "event_url": "event-1",
                    "result": "W",
                    "fighter": "A",
                    "opponent": "B",
                    "fighter_url": "fighter-a",
                    "opponent_url": "fighter-b",
                    "division": "Lightweight",
                    "method": "U-DEC",
                    "round": 3,
                    "time": "5:00",
                    "total_fight_time": 900,
                    "sig_strikes_landed": 20,
                    "sig_strikes_attempts": 30,
                },
                {
                    "date": "2025-01-01",
                    "fight_url": "fight-1",
                    "event_url": "event-1",
                    "result": "L",
                    "fighter": "B",
                    "opponent": "A",
                    "fighter_url": "fighter-b",
                    "opponent_url": "fighter-a",
                    "division": "Lightweight",
                    "method": "U-DEC",
                    "round": 3,
                    "time": "5:00",
                    "total_fight_time": 900,
                    "sig_strikes_landed": 10,
                    "sig_strikes_attempts": 25,
                },
            ]
        )

    def test_accepts_valid_doubled_fight(self):
        self.assertEqual(validate_raw_fights(self.make_raw()).errors, [])

    def test_rejects_broken_pair_and_impossible_stats(self):
        raw = self.make_raw()
        raw.loc[1, "fighter_url"] = "fighter-a"
        raw.loc[0, "sig_strikes_landed"] = 31
        errors = validate_raw_fights(raw).errors
        self.assertTrue(any("fighter IDs" in error for error in errors))
        self.assertTrue(any("exceeds" in error for error in errors))


class PointInTimeFeatureTests(unittest.TestCase):
    def test_future_rows_do_not_change_prior_second_degree_features(self):
        base = pd.DataFrame(
            [
                {"fighter": "Opponent", "opponent": "X", "result": "W"},
                {"fighter": "Fighter", "opponent": "Opponent", "result": "W"},
                {"fighter": "Fighter", "opponent": "Y", "result": "W"},
            ]
        ).set_index(pd.to_datetime(["2020-01-01", "2020-02-01", "2020-06-01"]))
        future = pd.DataFrame(
            [
                {"fighter": "Opponent", "opponent": "Z", "result": "W"},
                {"fighter": "Fighter", "opponent": "Q", "result": "W"},
            ]
        ).set_index(pd.to_datetime(["2021-01-01", "2021-02-01"]))
        extended = pd.concat([base, future])

        before = count_wins_wins_before_fight(base, "Fighter", "l1y")
        after = count_wins_wins_before_fight(extended, "Fighter", "l1y")

        self.assertEqual(before.tolist(), after[: len(before)].tolist())
        self.assertEqual(before[-1], 1)

    def test_losses_losses_uses_each_fight_date(self):
        history = pd.DataFrame(
            [
                {"fighter": "Opponent", "opponent": "X", "result": "L"},
                {"fighter": "Fighter", "opponent": "Opponent", "result": "L"},
                {"fighter": "Fighter", "opponent": "Y", "result": "L"},
            ],
            index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-06-01"]),
        )

        values = count_losses_losses_before_fight(history, "Fighter", "l1y")

        self.assertEqual(values[-1], 1)

    def test_matchup_probability_is_antisymmetric(self):
        predictor = FightPredictor.__new__(FightPredictor)
        predictor.imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        reference = pd.DataFrame([[1.0, 2.0], [-1.0, -2.0]], columns=["a", "b"])
        imputed = predictor.imputer.fit_transform(reference)
        predictor.scaler = StandardScaler(with_mean=False).fit(imputed)
        predictor.theta = [0.25, -0.5]
        predictor.b = 0.0

        forward = pd.DataFrame([[3.0, -1.0]], columns=["a", "b"])
        reverse = -forward

        self.assertAlmostEqual(
            predictor.probability(forward) + predictor.probability(reverse),
            1.0,
            places=12,
        )


class PredictionHistoryTests(unittest.TestCase):
    def test_draws_and_missing_predictions_are_retained_as_not_scored(self):
        history = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "fighter name": "A",
                    "opponent name": "B",
                    "predicted fighter odds": "-150",
                    "best fighter bookie": "",
                    "best opponent bookie": "",
                    "fighter bet bankroll percentage": 0.0,
                    "opponent bet bankroll percentage": 0.0,
                },
                {
                    "date": "2025-01-01",
                    "fighter name": "C",
                    "opponent name": "D",
                    "predicted fighter odds": "",
                    "best fighter bookie": "",
                    "best opponent bookie": "",
                    "fighter bet bankroll percentage": 0.0,
                    "opponent bet bankroll percentage": 0.0,
                },
            ]
        )
        completed = pd.DataFrame(
            [
                {"date": "2025-01-01", "fighter": "A", "opponent": "B", "result": "D"},
                {"date": "2025-01-01", "fighter": "B", "opponent": "A", "result": "D"},
            ]
        )
        handler = DataHandler.__new__(DataHandler)
        handler.odds_getter = OddsGetter()

        result = handler.update_prediction_correctness(history, completed, 300.0)

        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[0, "forecast status"], "draw")
        self.assertEqual(result.loc[1, "forecast status"], "no_prediction")
        self.assertEqual(result.loc[0, "correct?"], "N/A")
        self.assertEqual(result.loc[1, "correct?"], "N/A")

    def test_odds_outage_does_not_abort_core_predictions(self):
        class BrokenOddsGetter:
            def make_odds_df(self):
                raise RuntimeError("fixture outage")

        predictions = pd.DataFrame(
            [{"fighter name": "A", "opponent name": "B"}]
        )
        handler = DataHandler.__new__(DataHandler)
        handler.odds_getter = BrokenOddsGetter()

        result = handler.save_fightoddsio_to_vegas_odds_json_and_merge_with_predictions_df(
            predictions
        )

        self.assertEqual(result.loc[0, "odds source status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
