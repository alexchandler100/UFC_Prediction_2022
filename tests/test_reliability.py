import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from odds_getter import OddsGetter
from data_handler import DataHandler
from data_handler.data_handler import validate_scraped_event_integrity
from fight_predictor import (
    FightPredictor,
    PointInTimeDatasetBuilder,
    TemporalFightPredictor,
)
from fight_predictor.point_in_time import _metrics, training_fingerprint
import fight_stat_helpers
from fight_stat_helpers import (
    calculate_total_fight_time,
    count_losses_losses_before_fight,
    count_wins_wins_before_fight,
    extract_time_format,
    get_kelly_bet_from_ev_and_dk_odds,
)
from ufcstats_client import (
    RequestTimeout,
    UFCStatsClient,
    UFCStatsError,
    UFCStatsEventNotComplete,
)
from validate_data import validate_point_in_time, validate_raw_fights


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

    def test_extracts_real_ufcstats_time_format_markup(self):
        soup = fight_stat_helpers.BeautifulSoup(
            '<i class="b-fight-details__text-item">'
            '<i class="b-fight-details__label">Time format:</i>'
            '5 Rnd (5-5-5-5-5)</i>',
            'html.parser',
        )
        self.assertEqual(extract_time_format(soup), '5 Rnd (5-5-5-5-5)')

    def test_rejects_new_non_nc_fight_with_unknown_duration(self):
        response = make_response(
            '<li class="b-list__box-list-item">Date:\n August 14, 2026</li>'
            '<table class="b-fight-details__table">'
            '<tr class="b-fight-details__table-row"></tr>'
            '<tr class="b-fight-details__table-row">'
            '<td><a href="http://ufcstats.test/fight/f1"></a><p>win</p></td>'
            '<td><a href="http://ufcstats.test/fighter/a"></a>'
            '<a href="http://ufcstats.test/fighter/b"></a><p>A</p><p>B</p></td>'
            '<td></td><td></td><td></td><td></td>'
            '<td><p>Lightweight</p></td><td><p>U-DEC</p></td>'
            '<td><p>1</p></td><td><p>N/A</p></td>'
            '</tr></table>'
        )

        class StaticClient:
            def get(self, *_args, **_kwargs):
                return response

        details = pd.DataFrame(
            {'fighter': ['A', 'B'], 'time_format': ['3 Rnd (5-5-5)'] * 2}
        )
        with (
            patch.object(fight_stat_helpers, 'ufcstats_client', StaticClient()),
            patch.object(fight_stat_helpers, 'get_fight_stats', return_value=details),
        ):
            with self.assertRaisesRegex(UFCStatsError, 'calculate elapsed time'):
                fight_stat_helpers.get_fight_card('http://ufcstats.test/event')


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

    def test_partial_recent_event_refresh_is_quarantined(self):
        scraped = pd.DataFrame({'fight_url': ['fight-1', 'fight-1']})
        with self.assertRaisesRegex(UFCStatsError, 'Refusing to shrink stored event'):
            validate_scraped_event_integrity(
                'event-1',
                scraped,
                ['fight-1'],
                ['fight-1', 'fight-2'],
            )
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
    @staticmethod
    def make_fight(
        fight_id,
        event_id,
        fight_date,
        fighter_id,
        opponent_id,
        fighter_result,
        *,
        bout_order=0,
        source_card_index=0,
        fighter_sig=0,
        opponent_sig=0,
    ):
        common = {
            "date": fight_date,
            "fight_url": f"http://ufcstats.test/fight-details/{fight_id}",
            "event_url": f"http://ufcstats.test/event-details/{event_id}",
            "division": "Lightweight",
            "method": "U-DEC",
            "round": 3,
            "time": "5:00",
            "total_fight_time": 900,
            "source_card_index": source_card_index,
            "bout_order": bout_order,
            "knockdowns": 0,
            "sig_strikes_attempts": 40,
            "total_strikes_landed": 0,
            "total_strikes_attempts": 40,
            "takedowns_landed": 0,
            "takedowns_attempts": 0,
            "sub_attempts": 0,
            "reversals": 0,
            "control": 0,
        }
        first = {
            **common,
            "fighter": fighter_id.upper(),
            "opponent": opponent_id.upper(),
            "fighter_url": f"http://ufcstats.test/fighter-details/{fighter_id}",
            "opponent_url": f"http://ufcstats.test/fighter-details/{opponent_id}",
            "result": fighter_result,
            "sig_strikes_landed": fighter_sig,
        }
        second = {
            **common,
            "fighter": opponent_id.upper(),
            "opponent": fighter_id.upper(),
            "fighter_url": f"http://ufcstats.test/fighter-details/{opponent_id}",
            "opponent_url": f"http://ufcstats.test/fighter-details/{fighter_id}",
            "result": "L" if fighter_result == "W" else "W",
            "sig_strikes_landed": opponent_sig,
        }
        return [first, second]

    @staticmethod
    def make_profiles(*fighter_ids):
        return pd.DataFrame(
            [
                {
                    "name": fighter_id.upper(),
                    "height": "5' 10\"",
                    "reach": '70"',
                    "stance": "Orthodox",
                    "dob": "Jan 01, 1990",
                    "url": f"http://ufcstats.test/fighter-details/{fighter_id}",
                }
                for fighter_id in fighter_ids
            ]
        )

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

    def test_stable_ids_make_features_order_independent_and_point_in_time(self):
        raw = pd.DataFrame(
            self.make_fight("f1", "e1", "2020-01-01", "a", "b", "W", fighter_sig=35)
            + self.make_fight("f2", "e2", "2021-01-01", "a", "c", "L", fighter_sig=10)
        )
        profiles = self.make_profiles("a", "b", "c")
        original_builder = PointInTimeDatasetBuilder(raw, profiles)
        original = original_builder.build()
        shuffled_builder = PointInTimeDatasetBuilder(
            raw.sample(frac=1.0, random_state=9).reset_index(drop=True), profiles
        )
        shuffled = shuffled_builder.build()

        pd.testing.assert_frame_equal(original, shuffled, check_exact=True)
        first = original.loc[original["fight_id"] == "f1"].iloc[0]
        second = original.loc[original["fight_id"] == "f2"].iloc[0]
        self.assertEqual(first["career_sig_landed_per15_diff"], 0.0)
        self.assertNotEqual(second["career_sig_landed_per15_diff"], 0.0)
        self.assertEqual(validate_point_in_time(raw, original).errors, [])

    def test_appending_future_fight_cannot_change_existing_features(self):
        base_raw = pd.DataFrame(
            self.make_fight("f1", "e1", "2020-01-01", "a", "b", "W", fighter_sig=20)
            + self.make_fight("f2", "e2", "2021-01-01", "a", "c", "L", fighter_sig=10)
        )
        future = pd.DataFrame(
            self.make_fight("f3", "e3", "2022-01-01", "b", "c", "W", fighter_sig=25)
        )
        profiles = self.make_profiles("a", "b", "c")
        before = PointInTimeDatasetBuilder(base_raw, profiles).build()
        after = PointInTimeDatasetBuilder(
            pd.concat([base_raw, future], ignore_index=True), profiles
        ).build()

        pd.testing.assert_frame_equal(
            before,
            after[after["fight_id"].isin(before["fight_id"])].reset_index(drop=True),
            check_exact=True,
        )

    def test_same_event_disjoint_matchups_do_not_share_results(self):
        raw = pd.DataFrame(
            self.make_fight(
                "f1", "event", "2020-01-01", "a", "b", "W",
                bout_order=0, source_card_index=1,
            )
            + self.make_fight(
                "f2", "event", "2020-01-01", "c", "d", "W",
                bout_order=1, source_card_index=0,
            )
        )
        point = PointInTimeDatasetBuilder(
            raw, self.make_profiles("a", "b", "c", "d")
        ).build()
        self.assertTrue((point["elo_medium_diff"] == 0.0).all())

    def test_later_tournament_round_sees_earlier_round(self):
        raw = pd.DataFrame(
            self.make_fight(
                "f1", "event", "2020-01-01", "a", "b", "W",
                bout_order=0, source_card_index=1,
            )
            + self.make_fight(
                "f2", "event", "2020-01-01", "a", "c", "W",
                bout_order=1, source_card_index=0,
            )
        )
        point = PointInTimeDatasetBuilder(
            raw, self.make_profiles("a", "b", "c")
        ).build()
        later = point.loc[point["fight_id"] == "f2"].iloc[0]
        self.assertGreater(later["elo_medium_diff"], 0.0)

    def test_no_contest_advances_rating_decay_clock(self):
        first = self.make_fight("f1", "e1", "2019-01-01", "a", "b", "W")
        no_contest = self.make_fight(
            "f2", "e2", "2024-01-01", "a", "c", "W"
        )
        for side in no_contest:
            side["result"] = "NC"
            side["method"] = "Overturned"
        builder = PointInTimeDatasetBuilder(
            pd.DataFrame(first + no_contest), self.make_profiles("a", "b", "c")
        )
        builder.build()

        expected = builder._decayed_rating(
            1532.0, pd.Timestamp("2019-01-01"), pd.Timestamp("2024-01-01")
        )
        self.assertAlmostEqual(
            builder.states["a"].ratings["elo_medium"], expected, places=12
        )

    def test_no_contest_may_have_unknown_round_and_duration(self):
        no_contest = self.make_fight(
            "f1", "e1", "2024-01-01", "a", "b", "W"
        )
        for side in no_contest:
            side["result"] = "NC"
            side["method"] = "CNC"
            side["round"] = np.nan
            side["total_fight_time"] = np.nan
        builder = PointInTimeDatasetBuilder(
            pd.DataFrame(no_contest), self.make_profiles("a", "b")
        )
        with self.assertRaisesRegex(ValueError, "No terminal W/L"):
            builder.build()

    def test_historical_round_formats_drive_elapsed_seconds(self):
        self.assertEqual(
            calculate_total_fight_time(2, "5:00", "2 Rnd (15-3)"), 1200
        )
        self.assertEqual(
            calculate_total_fight_time(3, "5:00", "3 Rnd (5-5)"), 900
        )
        self.assertEqual(
            calculate_total_fight_time(5, "5:00", "5 Rnd (5-5)"), 1500
        )
        self.assertEqual(
            calculate_total_fight_time(1, "17:00", "No Time Limit"), 1020
        )
        self.assertTrue(
            np.isnan(calculate_total_fight_time(np.nan, "N/A", "1 Rnd (5)"))
        )

    def test_historical_ad_hoc_matchup_is_rejected_after_replay(self):
        raw = pd.DataFrame(
            self.make_fight("f1", "e1", "2020-01-01", "a", "b", "W")
        )
        builder = PointInTimeDatasetBuilder(raw, self.make_profiles("a", "b"))
        builder.build()
        with self.assertRaisesRegex(ValueError, "Historical matchup"):
            builder.matchup_features("a", "b", "2020-01-01", "Lightweight")

    def test_missing_stat_does_not_add_zero_exposure(self):
        first = self.make_fight("f1", "e1", "2020-01-01", "a", "b", "W")
        first[0]["control"] = np.nan
        second = self.make_fight("f2", "e2", "2021-01-01", "a", "c", "W")
        second[0]["control"] = 90
        third = self.make_fight("f3", "e3", "2022-01-01", "a", "d", "W")
        point = PointInTimeDatasetBuilder(
            pd.DataFrame(first + second + third),
            self.make_profiles("a", "b", "c", "d"),
        ).build()
        third_row = point.loc[point["fight_id"] == "f3"].iloc[0]
        self.assertAlmostEqual(third_row["career_control_per15_diff"], 45.0)

    def test_builder_rejects_nonfinite_stats(self):
        raw = pd.DataFrame(
            self.make_fight("f1", "e1", "2020-01-01", "a", "b", "W")
        )
        raw["control"] = raw["control"].astype(object)
        raw.loc[0, "control"] = "inf"
        with self.assertRaisesRegex(ValueError, "non-finite"):
            PointInTimeDatasetBuilder(raw, self.make_profiles("a", "b")).build()

    def test_builder_rejects_fractional_card_order(self):
        raw = pd.DataFrame(
            self.make_fight(
                "f1", "e1", "2020-01-01", "a", "b", "W",
                bout_order=0.2, source_card_index=0.8,
            )
        )
        with self.assertRaisesRegex(ValueError, "bout_order.*integers"):
            PointInTimeDatasetBuilder(raw, self.make_profiles("a", "b")).build()

    def test_point_in_time_validator_rejects_missing_or_corrupt_lineage(self):
        raw = pd.DataFrame(
            self.make_fight("f1", "e1", "2020-01-01", "a", "b", "W")
            + self.make_fight("f2", "e2", "2021-01-01", "a", "c", "L")
            + self.make_fight("f3", "e3", "2022-01-01", "b", "c", "W")
        )
        point = PointInTimeDatasetBuilder(
            raw, self.make_profiles("a", "b", "c")
        ).build()
        corrupt = point[point["fight_id"] != "f2"].copy().reset_index(drop=True)
        corrupt.loc[0, "target"] = 1 - int(corrupt.loc[0, "target"])
        corrupt.loc[0, "fighter_url"] = (
            "http://ufcstats.test/fighter-details/not-the-fighter"
        )
        errors = validate_point_in_time(raw, corrupt).errors
        self.assertTrue(any("exact set" in error for error in errors))
        self.assertTrue(any("target disagrees" in error for error in errors))
        self.assertTrue(any("fighter_url disagrees" in error for error in errors))

    def test_duplicate_display_names_are_ambiguous_without_a_url(self):
        profiles = self.make_profiles("bruno-one", "bruno-two")
        profiles["name"] = "Bruno Silva"
        raw = pd.DataFrame(
            self.make_fight(
                "f1", "e1", "2020-01-01", "bruno-one", "bruno-two", "W"
            )
        )
        builder = PointInTimeDatasetBuilder(raw, profiles)
        self.assertIsNone(builder.resolve_fighter_id("Bruno Silva"))
        self.assertEqual(
            builder.resolve_fighter_id(
                "Bruno Silva", "http://ufcstats.test/fighter-details/bruno-two"
            ),
            "bruno-two",
        )

    def test_portable_artifact_round_trip_preserves_probability_and_symmetry(self):
        raw = pd.DataFrame(
            self.make_fight("f1", "e1", "2020-01-01", "a", "b", "W")
            + self.make_fight("f2", "e2", "2021-01-01", "a", "b", "L")
        )
        builder = PointInTimeDatasetBuilder(raw, self.make_profiles("a", "b"))
        template = builder.build()
        rows = []
        rng = np.random.default_rng(48)
        for index in range(120):
            row = template.iloc[index % len(template)].copy()
            row["date"] = pd.Timestamp("2020-01-01") + pd.Timedelta(days=index)
            row["fight_id"] = f"synthetic-{index}"
            row["target"] = index % 2
            row[list(builder.feature_columns)] = rng.normal(size=len(builder.feature_columns))
            rows.append(row)
        training = pd.DataFrame(rows).reset_index(drop=True)
        builder.training_data = training.copy()
        predictor = TemporalFightPredictor(training, builder)
        predictor.imputer, predictor.scaler, predictor.model = predictor._fit_pipeline(
            training[list(builder.feature_columns)], training["target"], 0.1
        )
        predictor._artifact_scale = predictor.scaler.scale_.copy()
        predictor._artifact_coefficients = predictor.model.coef_[0].copy()
        predictor.best_c = 0.1
        predictor.calibration_slope = 1.1
        predictor.evaluation = {"fixture": True}
        sample = training.loc[[0], list(builder.feature_columns)]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "winner_model.json"
            matrix_path = Path(directory) / "point_in_time.csv"
            training.to_csv(matrix_path, index=False)
            round_tripped = pd.read_csv(matrix_path, low_memory=False)
            self.assertEqual(
                training_fingerprint(training, builder.feature_columns),
                training_fingerprint(round_tripped, builder.feature_columns),
            )
            predictor.save_artifact(path)
            loaded = TemporalFightPredictor.load_artifact(path, builder)
            self.assertAlmostEqual(predictor.probability(sample), loaded.probability(sample), 14)
            self.assertAlmostEqual(
                loaded.probability(sample) + loaded.probability(-sample), 1.0, 12
            )

            original_training = builder.training_data.copy()
            builder.training_data.loc[0, builder.feature_columns[0]] += 1.0
            with self.assertRaisesRegex(ValueError, "training fingerprint"):
                TemporalFightPredictor.load_artifact(path, builder)
            builder.training_data = original_training

            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["coefficients"][0] += 1.0
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_id"):
                TemporalFightPredictor.load_artifact(path, builder)

    def test_one_class_metric_fold_serializes_null_auc(self):
        metrics = _metrics(np.ones(3, dtype=int), np.array([0.6, 0.7, 0.8]))
        self.assertIsNone(metrics["auc"])
        json.dumps(metrics, allow_nan=False)


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

    def test_primary_forecast_and_independent_model_are_scored_separately(self):
        history = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "fighter name": "A",
                    "opponent name": "B",
                    "forecast fighter odds": "-150",
                    "predicted fighter odds": "+130",
                    "best fighter bookie": "",
                    "best opponent bookie": "",
                    "fighter bet bankroll percentage": 0.0,
                    "opponent bet bankroll percentage": 0.0,
                }
            ]
        )
        completed = pd.DataFrame(
            [
                {"date": "2025-01-01", "fighter": "A", "opponent": "B", "result": "W"},
                {"date": "2025-01-01", "fighter": "B", "opponent": "A", "result": "L"},
            ]
        )
        handler = DataHandler.__new__(DataHandler)
        handler.odds_getter = OddsGetter()

        result = handler.update_prediction_correctness(history, completed, 300.0)

        self.assertEqual(result.loc[0, "correct?"], 1)
        self.assertEqual(result.loc[0, "model correct?"], 0)

    def test_new_history_uses_stable_ids_and_records_result_lineage(self):
        history = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "fighter name": "A",
                    "opponent name": "B",
                    "fighter id": "fighter-a",
                    "opponent id": "fighter-b",
                    "forecast fighter odds": "-150",
                    "predicted fighter odds": "-150",
                    "best fighter bookie": "",
                    "best opponent bookie": "",
                    "fighter bet bankroll percentage": 0.0,
                    "opponent bet bankroll percentage": 0.0,
                }
            ]
        )
        completed = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "fighter": "A",
                    "opponent": "B",
                    "fighter_url": "http://ufcstats.com/fighter-details/not-a",
                    "opponent_url": "http://ufcstats.com/fighter-details/not-b",
                    "fight_url": "http://ufcstats.com/fight-details/wrong-fight",
                    "result": "L",
                },
                {
                    "date": "2025-01-01",
                    "fighter": "Renamed A",
                    "opponent": "Renamed B",
                    "fighter_url": "http://ufcstats.com/fighter-details/fighter-a",
                    "opponent_url": "http://ufcstats.com/fighter-details/fighter-b",
                    "fight_url": "http://ufcstats.com/fight-details/right-fight",
                    "result": "W",
                },
            ]
        )
        handler = DataHandler.__new__(DataHandler)
        handler.odds_getter = OddsGetter()

        result = handler.update_prediction_correctness(history, completed, 300.0)

        self.assertEqual(result.loc[0, "correct?"], 1)
        self.assertEqual(result.loc[0, "fight id"], "right-fight")
        self.assertEqual(result.loc[0, "actual result"], "W")

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

    def test_ambiguous_odds_match_does_not_abort_core_predictions(self):
        class AmbiguousOddsGetter:
            def make_odds_df(self):
                return pd.DataFrame(
                    [
                        {
                            "fighter name": "A",
                            "opponent name": "B",
                            "average bookie odds": [-110, -110],
                        },
                        {
                            "fighter name": "A",
                            "opponent name": "B",
                            "average bookie odds": [-105, -115],
                        },
                    ]
                )

        predictions = pd.DataFrame(
            [{"fighter name": "A", "opponent name": "B"}]
        )
        handler = DataHandler.__new__(DataHandler)
        handler.odds_getter = AmbiguousOddsGetter()
        handler.bookies = []
        handler.json_data = {"vegas_odds": pd.DataFrame()}

        result = handler.save_fightoddsio_to_vegas_odds_json_and_merge_with_predictions_df(
            predictions
        )

        self.assertEqual(result.loc[0, "odds source status"], "ambiguous")

    def test_malformed_odds_schema_does_not_abort_core_predictions(self):
        class MalformedOddsGetter:
            def make_odds_df(self):
                return pd.DataFrame([{"unexpected": "field"}])

        predictions = pd.DataFrame(
            [{"fighter name": "A", "opponent name": "B"}]
        )
        handler = DataHandler.__new__(DataHandler)
        handler.odds_getter = MalformedOddsGetter()
        handler.bookies = []
        handler.json_data = {"vegas_odds": pd.DataFrame()}

        result = handler.save_fightoddsio_to_vegas_odds_json_and_merge_with_predictions_df(
            predictions
        )

        self.assertEqual(result.loc[0, "odds source status"], "unavailable")

    def test_timestamped_market_consensus_is_primary_without_enabling_bets(self):
        class StaticOddsGetter(OddsGetter):
            def make_odds_df(self):
                return pd.DataFrame(
                    [
                        {
                            "fighter name": "A",
                            "opponent name": "B",
                            "fighter BookA": "+120",
                            "opponent BookA": "-140",
                            "average bookie probability": 0.47,
                            "average bookie odds": [113, -113],
                        }
                    ]
                )

        predictions = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "fighter name": "A",
                    "opponent name": "B",
                    "model id": "model-fixture",
                    "model probability": 0.6,
                    "predicted fighter odds": "-150",
                    "predicted opponent odds": "+150",
                    "forecast probability": 0.6,
                    "forecast source": "stats_model",
                }
            ]
        )
        handler = DataHandler.__new__(DataHandler)
        handler.odds_getter = StaticOddsGetter()
        handler.bookies = ["BookA"]
        handler.json_data = {"vegas_odds": pd.DataFrame()}

        result = handler.save_fightoddsio_to_vegas_odds_json_and_merge_with_predictions_df(
            predictions
        )

        self.assertEqual(result.loc[0, "forecast source"], "market_no_vig_consensus")
        self.assertAlmostEqual(result.loc[0, "forecast probability"], 0.47)
        self.assertEqual(
            result.loc[0, "betting status"],
            "disabled_pending_market_relative_validation",
        )
        self.assertTrue(result.loc[0, "odds observed at"])
        self.assertTrue(pd.isna(result.loc[0, "fighter bet bankroll percentage"]))


if __name__ == "__main__":
    unittest.main()
