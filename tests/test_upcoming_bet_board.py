import copy
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_handler.data_handler as data_handler_module  # noqa: E402
from data_handler import DataHandler  # noqa: E402
from market_tracker import EarlyMarketObservation  # noqa: E402
from market_tracker._common import canonical_hash  # noqa: E402
from upcoming_bet_board import (  # noqa: E402
    _calibrated_offer,
    allocate_paper_offers,
    build_upcoming_bet_board,
    build_upcoming_forecast_publication,
    validate_upcoming_bet_board,
)


OBSERVED = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


def _forecast_frame() -> pd.DataFrame:
    common = {
        "division": "Lightweight",
        "model id": "winner-model",
        "model version": "winner-v1",
        "model trained through": "2026-08-22",
        "model probability": 0.6,
        "model status": "available",
        "forecast issued at": "2026-08-29T10:00:00Z",
        "forecast source commit": "a" * 40,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "date": "2026-08-30",
                "event id": "event-one",
                "event url": "http://ufcstats.com/event-details/event-one",
                "event title": "UFC Event One",
                "bout order": 0,
                "fighter id": "fighter-alpha",
                "opponent id": "fighter-beta",
                "fighter name": "Alpha One",
                "opponent name": "Beta Two",
            },
            {
                **common,
                "date": "2026-09-05",
                "event id": "event-two",
                "event url": "http://ufcstats.com/event-details/event-two",
                "event title": "UFC Event Two",
                "bout order": 0,
                "fighter id": "fighter-gamma",
                "opponent id": "fighter-delta",
                "fighter name": "Gamma Three",
                "opponent name": "Delta Four",
            },
        ]
    )


def _observation(
    *,
    event_id: str,
    commence: str,
    fighter: str,
    opponent: str,
    book: str,
    fighter_line: int,
    opponent_line: int,
) -> EarlyMarketObservation:
    return EarlyMarketObservation.create(
        first_capture_id="capture-one",
        first_observed_at_utc=OBSERVED,
        source="the-odds-api.com",
        source_payload_sha256="b" * 64,
        source_event_id=event_id,
        source_commence_time_utc=commence,
        source_fighter_name=fighter,
        source_opponent_name=opponent,
        book=book,
        source_book_key=book.casefold().replace(" ", "-"),
        source_quote_updated_at_utc="2026-08-29T11:59:30Z",
        market="h2h",
        outcome_a=fighter,
        outcome_b=opponent,
        outcome_a_moneyline=fighter_line,
        outcome_b_moneyline=opponent_line,
    )


def _event_prices(
    event_id: str,
    commence: str,
    fighter: str,
    opponent: str,
    *,
    reverse: bool = False,
) -> list[EarlyMarketObservation]:
    source_fighter, source_opponent = (
        (opponent, fighter) if reverse else (fighter, opponent)
    )
    observations = []
    for book, direct_lines in (
        ("Consensus A", (-200, 170)),
        ("Consensus B", (-200, 170)),
        ("Consensus C", (-200, 170)),
        ("Target Book", (100, -120)),
    ):
        lines = tuple(reversed(direct_lines)) if reverse else direct_lines
        observations.append(
            _observation(
                event_id=event_id,
                commence=commence,
                fighter=source_fighter,
                opponent=source_opponent,
                book=book,
                fighter_line=lines[0],
                opponent_line=lines[1],
            )
        )
    return observations


class UpcomingBetBoardTests(unittest.TestCase):
    def setUp(self):
        self.forecasts = build_upcoming_forecast_publication(
            _forecast_frame(), generated_at_utc="2026-08-29T10:00:00Z"
        )

    def test_model_refresh_keeps_saved_market_inputs_instead_of_emptying_board(self):
        updater = (ROOT / "src" / "update_and_rebuild_model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("EarlyMarketObservationStore", updater)
        self.assertIn("early_market_observations,", updater)
        self.assertIn("current_opportunities=current_opportunities", updater)
        self.assertNotIn("all_upcoming_forecasts,\n        (),", updater)

    def test_board_matches_all_announced_cards_and_sorts_only_qualified_prices(self):
        observations = [
            *_event_prices(
                "source-one",
                "2026-08-30T20:00:00Z",
                "Alpha One",
                "Beta Two",
            ),
            *_event_prices(
                "source-two",
                "2026-09-05T20:00:00Z",
                "Gamma Three",
                "Delta Four",
                reverse=True,
            ),
        ]
        board = build_upcoming_bet_board(
            self.forecasts,
            observations,
            observed_at_utc=OBSERVED,
            source="the-odds-api.com",
        )

        self.assertEqual(board["announced_event_count"], 2)
        self.assertEqual(board["market_matched_matchup_count"], 2)
        self.assertEqual(len(board["market_matchups"]), 2)
        self.assertTrue(all(item["book_count"] == 4 for item in board["market_matchups"]))
        self.assertTrue(all(len(item["book_quotes"]) == 4 for item in board["market_matchups"]))
        self.assertTrue(
            all(0.0 < item["consensus_fighter_probability"] < 1.0 for item in board["market_matchups"])
        )
        reverse_quotes = board["market_matchups"][1]["book_quotes"]
        reverse_target = next(item for item in reverse_quotes if item["book"] == "Target Book")
        self.assertEqual(reverse_target["fighter_moneyline"], 100)
        self.assertEqual(reverse_target["opponent_moneyline"], -120)
        self.assertEqual(board["qualified_bet_count"], 2)
        self.assertEqual(
            [bet["estimated_expected_return"] for bet in board["bets"]],
            sorted(
                (bet["estimated_expected_return"] for bet in board["bets"]),
                reverse=True,
            ),
        )
        self.assertTrue(all(bet["threshold_met"] for bet in board["bets"]))
        self.assertTrue(
            all(bet["bayesian_kelly"]["status"] == "available" for bet in board["bets"])
        )
        self.assertTrue(
            all(
                0.0 <= bet["bayesian_kelly"]["recommended_fraction"] <= 0.05
                for bet in board["bets"]
            )
        )
        self.assertEqual(
            {bet["event_id"] for bet in board["bets"]},
            {"event-one", "event-two"},
        )
        self.assertTrue(all(bet["target_book"] == "Target Book" for bet in board["bets"]))

    def test_totals_without_supported_calibration_are_withheld_from_funded_board(self):
        matchup = self.forecasts["matchups"][0]
        current = {
            "event_id": matchup["event_id"],
            "observed_at_utc": "2026-08-29T12:00:00Z",
            "source": "the-odds-api.com",
            "matchups": [],
            "prop_markets": {
                "total_rounds": {
                    "positive_candidates": [
                        {
                            "matchup_id": matchup["matchup_id"],
                            "selection": "Over 1.5 rounds",
                            "side": "over",
                            "line": 1.5,
                            "target_book": "Book A",
                            "offered_moneyline": -110,
                            "model_probability": 0.6,
                            "estimated_expected_return": 0.145,
                            "scheduled_rounds": 3,
                            "schedule_basis": "ufc_standard_non_main_non_title",
                            "model_id": "outcome-model-one",
                            "model_version": "candidate-discrete-time-competing-risks-v1",
                            "model_trained_through": "2026-08-22",
                            "forecast_issued_at_utc": "2026-08-29T10:00:00Z",
                            "break_even_probability": 0.5238095238,
                            "paper_threshold_met": True,
                        },
                        {
                            "matchup_id": matchup["matchup_id"],
                            "selection": "Under 2.5 rounds",
                            "side": "under",
                            "target_book": "Book B",
                            "offered_moneyline": 130,
                            "model_probability": 0.5,
                            "estimated_expected_return": 0.15,
                            "paper_threshold_met": True,
                        },
                    ]
                }
            },
        }
        board = build_upcoming_bet_board(
            self.forecasts,
            (),
            observed_at_utc=OBSERVED,
            source="the-odds-api.com",
            current_opportunities=current,
            enable_total_bayesian_calibration=False,
        )

        self.assertEqual(board["qualified_bet_count"], 0)
        self.assertEqual(board["offers"], [])
        self.assertEqual(len(current["prop_markets"]["total_rounds"]["positive_candidates"]), 2)

    def test_all_eligible_books_survive_until_accessible_book_selection(self):
        observations = _event_prices("source-one", "2026-08-30T20:00:00Z", "Alpha One", "Beta Two")
        observations.append(_observation(event_id="source-one", commence="2026-08-30T20:00:00Z",
                            fighter="Alpha One", opponent="Beta Two", book="Second Target",
                            fighter_line=100, opponent_line=-120))
        board = build_upcoming_bet_board(self.forecasts, observations, observed_at_utc=OBSERVED,
                                        source="the-odds-api.com")
        self.assertEqual(board["schema_version"], 2)
        self.assertTrue({"Target Book", "Second Target"}.issubset({offer["target_book"] for offer in board["offers"]}))
        self.assertEqual(len(board["bets"]), 1)
        self.assertAlmostEqual(board["allocated_fraction"], .01)
        other = [offer for offer in board["offers"] if offer["target_book"] == "Target Book"]
        reallocated = allocate_paper_offers(other)
        self.assertAlmostEqual(reallocated[0]["allocated_fraction"], .01)
        self.assertEqual(reallocated[0]["offered_moneyline"], 100)
        self.assertEqual(reallocated[0]["raw_estimated_win_probability"], other[0]["raw_estimated_win_probability"])
        self.assertEqual(reallocated[0]["bayesian_kelly"], other[0]["bayesian_kelly"])

    def test_calibration_can_remove_nominal_edge_and_missing_timing_never_funds(self):
        bet = {"estimated_win_probability": .7, "offered_moneyline": 100,
               "source_quote_updated_at_utc": "2026-08-29T11:59:30Z",
               "event_start_utc": "2026-08-30T20:00:00Z",
               "bayesian_kelly": {"status": "available", "posterior_mean_probability": .49,
                                  "posterior_lower_probability": .48, "recommended_fraction": .01}}
        self.assertIsNone(_calibrated_offer(bet, OBSERVED))
        bet["bayesian_kelly"].update(posterior_mean_probability=.6, posterior_lower_probability=.55)
        offer = _calibrated_offer(bet, OBSERVED)
        self.assertAlmostEqual(offer["estimated_expected_return"], .2)
        self.assertAlmostEqual(offer["raw_estimated_expected_return"], .4)
        bet["category"] = "Total rounds"
        self.assertIsNone(_calibrated_offer(bet, OBSERVED))
        bet["category"] = "Moneyline"
        bet["source_quote_updated_at_utc"] = None
        self.assertIsNone(_calibrated_offer(bet, OBSERVED))
        bet["source_quote_updated_at_utc"] = "2026-08-29T11:00:00Z"
        self.assertIsNone(_calibrated_offer(bet, OBSERVED))
        bet["source_quote_updated_at_utc"] = "2026-08-29T12:00:01Z"
        self.assertIsNone(_calibrated_offer(bet, OBSERVED))
        bet["source_quote_updated_at_utc"] = "2026-08-29T11:59:30Z"
        bet["event_start_utc"] = "2026-08-29T12:00:00Z"
        self.assertIsNone(_calibrated_offer(bet, OBSERVED))

    def test_allocation_caps_cards_outstanding_and_reversed_fighter_duplicates(self):
        offers = []
        for event in ("a", "b", "c"):
            for index in range(7):
                offers.append({"event_id": event, "event_date": "2026-08-30", "matchup_id": str(index),
                               "fighter_id": f"fighter-{index}", "opponent_id": f"opponent-{index}",
                               "selection": f"fighter-{index}", "side": "fighter", "target_book": "A",
                               "offered_moneyline": 150, "estimated_expected_return": .2,
                               "bayesian_kelly": {"recommended_fraction": .05}})
        reverse = dict(offers[0], fighter_id=offers[0]["opponent_id"], opponent_id=offers[0]["fighter_id"],
                       matchup_id="reversed", estimated_expected_return=.3)
        offers.append(reverse)
        allocated = allocate_paper_offers(offers)
        self.assertEqual(allocated, allocate_paper_offers(reversed(offers)))
        self.assertAlmostEqual(sum(x["allocated_fraction"] for x in allocated), .10)
        for event in ("a", "b", "c"):
            self.assertLessEqual(sum(x["allocated_fraction"] for x in allocated if x["event_id"] == event), .05 + 1e-12)
        self.assertEqual(sum(x["allocated_fraction"] > 0 for x in allocated if x["event_id"] == "a" and x["matchup_id"] in {"0", "reversed"}), 1)
        for offer in allocated:
            unhashed = dict(offer)
            supplied = unhashed.pop("bet_id")
            self.assertEqual(supplied, canonical_hash(unhashed))

    def test_legacy_board_remains_readable_for_historical_evidence(self):
        board = build_upcoming_bet_board(self.forecasts, (), observed_at_utc=OBSERVED, source="test")
        board["schema_version"] = 1
        board["policy_version"] = "all-upcoming-qualified-bets-v1"
        board.pop("publication_sha256")
        board["publication_sha256"] = canonical_hash(board)
        self.assertEqual(validate_upcoming_bet_board(board), board)

    def test_validator_rejects_a_below_threshold_row_even_with_updated_hashes(self):
        board = build_upcoming_bet_board(
            self.forecasts,
            _event_prices(
                "source-one",
                "2026-08-30T20:00:00Z",
                "Alpha One",
                "Beta Two",
            ),
            observed_at_utc=OBSERVED,
            source="the-odds-api.com",
        )
        tampered = copy.deepcopy(board)
        tampered["bets"][0]["estimated_expected_return"] = 0.01
        tampered["bets"][0].pop("bet_id")
        tampered["bets"][0]["bet_id"] = canonical_hash(tampered["bets"][0])
        tampered.pop("publication_sha256")
        tampered["publication_sha256"] = canonical_hash(tampered)

        with self.assertRaisesRegex(ValueError, "below-policy"):
            validate_upcoming_bet_board(tampered)

    def test_validator_rejects_an_inconsistent_stored_price_count(self):
        board = build_upcoming_bet_board(
            self.forecasts,
            _event_prices(
                "source-one",
                "2026-08-30T20:00:00Z",
                "Alpha One",
                "Beta Two",
            ),
            observed_at_utc=OBSERVED,
            source="the-odds-api.com",
        )
        tampered = copy.deepcopy(board)
        tampered["market_matchups"][0]["book_count"] += 1
        tampered.pop("publication_sha256")
        tampered["publication_sha256"] = canonical_hash(tampered)

        with self.assertRaisesRegex(ValueError, "stored-price count"):
            validate_upcoming_bet_board(tampered)


class UpcomingCardScrapeTests(unittest.TestCase):
    def test_all_announced_cards_with_fights_are_returned_in_date_order(self):
        events_html = b"""
        <table class='b-statistics__table-events'>
          <a class='b-link b-link_style_black' href='http://ufcstats.com/event-details/later'>Later Card</a>
          <span class='b-statistics__date'>September 05, 2026</span>
          <a class='b-link b-link_style_black' href='http://ufcstats.com/event-details/next'>Next Card</a>
          <span class='b-statistics__date'>August 30, 2026</span>
        </table>
        """

        def detail(first, second):
            return f"""
            <table class='b-fight-details__table'><tr class='b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click'>
              <p>{first}</p><p>{second}</p><p>0-0-0</p><p>Lightweight</p>
              <a href='http://ufcstats.com/fighter-details/{first.casefold()}'></a>
              <a href='http://ufcstats.com/fighter-details/{second.casefold()}'></a>
            </tr></table>
            """.encode()

        pages = {
            "http://ufcstats.com/statistics/events/upcoming": events_html,
            "http://ufcstats.com/event-details/later": detail("Gamma", "Delta"),
            "http://ufcstats.com/event-details/next": detail("Alpha", "Beta"),
        }

        class Client:
            @staticmethod
            def get(url, **_kwargs):
                return SimpleNamespace(content=pages[url])

        handler = DataHandler.__new__(DataHandler)
        with patch.object(data_handler_module, "ufcstats_client", Client()):
            cards = handler.get_upcoming_fight_cards()

        self.assertEqual([card[1] for card in cards], ["Next Card", "Later Card"])
        self.assertEqual(cards[0][2][0][0:3], ["Alpha", "Beta", "Lightweight"])


if __name__ == "__main__":
    unittest.main()
