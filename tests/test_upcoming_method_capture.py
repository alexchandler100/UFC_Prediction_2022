from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from types import SimpleNamespace

import pandas as pd

from test_method_market_capture import _published, _snapshot
from test_upcoming_bet_board import _forecast_frame, _event_prices, OBSERVED
from test_outcome_model import _training_frame
import capture_method_market_snapshot as collector
from bestfightodds_props import MethodPropSelection, PropBookPrice
from fight_predictor.outcome_model import DiscreteTimeOutcomeModel
from fight_predictor.upcoming_outcomes import publish_upcoming_outcomes
from market_tracker import matchup_id_for
from upcoming_bet_board import build_upcoming_forecast_publication, build_upcoming_bet_board


class UpcomingMethodCaptureTests(unittest.TestCase):
    def test_missing_fight_keeps_opening_and_timed_collection_running(self):
        first = _published()
        second = replace(first, fighter_id='c-fighter', fighter_name='Fighter Gamma',
                         matchup_id=matchup_id_for('event-1', 'c-fighter', first.opponent_id))
        first_quote = _snapshot(horizon='opening')
        second_quote = _snapshot(horizon='opening', fighter_id='c-fighter', fighter_name='Fighter Gamma')
        self.assertTrue(collector._capture_is_due([first_quote], None, [first, second]))
        self.assertFalse(collector._capture_is_due([first_quote, second_quote], None, [first, second]))
        self.assertTrue(collector._capture_is_due(
            [first_quote, second_quote, _snapshot()], 't24', [first, second]))

    def test_late_opening_is_saved_without_rewriting_first_fight(self):
        first = _published()
        second = replace(first, fighter_id='c-fighter', fighter_name='Fighter Gamma',
                         matchup_id=matchup_id_for('event-1', 'c-fighter', first.opponent_id))
        existing = _snapshot(horizon='opening')
        selections = [MethodPropSelection(index, 1, 8, 1, row.fighter_name, row.opponent_name,
                      'fighter_method_of_victory', 'ko_tko', 'wins by KO/TKO', True,
                      (PropBookPrice(21, 'Book A', 900),)) for index, row in enumerate([first, second], 1)]
        args = dict(selections=selections, published=[first, second], event_day='2026-08-29',
                    event_id='event-1', event_start_utc='2026-08-29T07:00:00Z',
                    observed=datetime(2026, 8, 28, 8, tzinfo=timezone.utc), payload_sha='b' * 64,
                    timed_horizon=None)
        rows, _ = collector._build_snapshots(existing=[existing], **args)
        self.assertEqual([row.matchup_id for row in rows], [second.matchup_id])
        self.assertEqual(existing.fighter_ko_tko_moneyline, 250)
        repeated, _ = collector._build_snapshots(existing=[existing, *rows], **args)
        self.assertEqual(repeated, [])

    def test_one_fetch_records_current_and_future_cards_and_validates(self):
        now = datetime(2026, 8, 28, 8, tzinfo=timezone.utc)
        first = _published()
        future = collector.MethodMatchup('Gamma Three', 'Delta Four', 'gamma', 'delta',
                                         matchup_id_for('event-2', 'gamma', 'delta'))
        card = collector.MethodCard('2026-09-05', 'event-2', (future,), '2026-09-05T18:00:00Z', None)
        def selections(_html, day):
            row = first if day == '2026-08-29' else future
            return (MethodPropSelection(10 if row is first else 20, 1, 8, 1,
                row.fighter_name, row.opponent_name, 'fighter_method_of_victory', 'ko_tko',
                'wins by KO/TKO', True, (PropBookPrice(21, 'Book A', 250),)),)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            for name in ['METHOD_CSV_PATH', 'METHOD_JSONL_PATH', 'METHOD_FORECAST_CSV_PATH',
                         'METHOD_FORECAST_JSONL_PATH', 'REPORT_PATH', 'CURRENT_METHOD_PATH']:
                old = getattr(collector, name)
                stack.enter_context(patch.object(collector, name, root / old.name))
            stack.enter_context(patch.dict('os.environ', {collector.SOURCE_POLICY_ENV: '1'}))
            clock = stack.enter_context(patch.object(collector, 'datetime', wraps=datetime))
            clock.now.return_value = now
            stack.enter_context(patch.object(collector, '_load_publication', return_value=(
                '2026-08-29', 'event-1', (first,), '2026-08-29T07:00:00Z', 23 * 3600)))
            stack.enter_context(patch.object(collector, '_future_cards', return_value=([card], [])))
            fetch = stack.enter_context(patch.object(collector, '_fetch', return_value=SimpleNamespace(text='', content=b'page')))
            stack.enter_context(patch.object(collector, '_robots_allows_public_paths', return_value=True))
            stack.enter_context(patch.object(collector, '_event_selections', side_effect=selections))
            stack.enter_context(patch.object(collector, '_outcome_forecasts', return_value=None))
            report = collector.capture_method_snapshot()
            self.assertEqual(report['records_added'], 3)  # current opening/T24 + future opening
            self.assertEqual(report['future_records_built'], 1)
            self.assertEqual(len(report['events']), 2)
            self.assertEqual(fetch.call_count, 2)  # robots and one shared page
            self.assertEqual(collector.validate_generated_capture(), report)
            before = collector.METHOD_JSONL_PATH.read_bytes()
            with self.assertRaises(collector.MethodCaptureSkipped):
                collector.capture_method_snapshot()
            self.assertEqual(collector.METHOD_JSONL_PATH.read_bytes(), before)

    def test_future_card_uses_published_start_and_removes_cancelled_identities(self):
        announced = build_upcoming_forecast_publication(_forecast_frame(), generated_at_utc=OBSERVED)
        observations = _event_prices('source-two', '2026-09-05T18:00:00Z', 'Gamma Three', 'Delta Four')
        board = build_upcoming_bet_board(announced, observations, observed_at_utc=OBSERVED, source='test')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecast, prices = root / 'forecast.json', root / 'board.json'
            forecast.write_text(json.dumps(announced))
            prices.write_text(json.dumps(board))
            with patch.object(collector, 'UPCOMING_FORECAST_PATH', forecast), patch.object(collector, 'UPCOMING_BOARD_PATH', prices):
                cards, missing = collector._future_cards(OBSERVED, 'event-one')
                self.assertEqual([card.event_id for card in cards], ['event-two'])
                self.assertIsNone(cards[0].timed_horizon)
                self.assertEqual(cards[0].matchups[0].fighter_name, 'Gamma Three')
                self.assertEqual(missing, [])
                cards, missing = collector._future_cards(datetime(2026, 9, 4, tzinfo=timezone.utc), 'event-one')
                self.assertEqual(cards, [])
                self.assertEqual(missing[0]['status'], 'awaiting_current_card_timing')
                # A changed announcement must not reuse the old board's identities.
                changed = _forecast_frame()
                changed.loc[1, 'opponent id'] = 'replacement'
                forecast.write_text(json.dumps(build_upcoming_forecast_publication(changed, generated_at_utc=OBSERVED)))
                cards, missing = collector._future_cards(OBSERVED, 'event-one')
                self.assertEqual(cards, [])
                self.assertEqual(missing[0]['status'], 'awaiting_source_start_time')

    def test_future_card_without_start_is_reported_and_not_assumed_midnight(self):
        announced = build_upcoming_forecast_publication(_forecast_frame(), generated_at_utc=OBSERVED)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'forecast.json'
            path.write_text(json.dumps(announced))
            with patch.object(collector, 'UPCOMING_FORECAST_PATH', path), patch.object(collector, 'UPCOMING_BOARD_PATH', Path(directory) / 'missing'):
                cards, missing = collector._future_cards(OBSERVED, 'event-one')
                self.assertEqual(cards, [])
                self.assertEqual(missing[0]['event_id'], 'event-two')

    def test_source_event_filter_rejects_old_rematch_and_accepts_utc_next_day(self):
        html = ''.join(f'<div class="table-div"><div class="table-header"><h1>{title}</h1>'
                       f'<span class="table-header-date">{date}</span></div>{marker}</div>'
                       for title, date, marker in [('UFC A', 'August 22nd', 'old'),
                                                   ('UFC B', 'August 30th', 'current'),
                                                   ('PFL', 'August 29th', 'other')])
        with patch.object(collector, 'parse_bestfightodds_method_props', side_effect=lambda value: [value]):
            rows = collector._event_selections(html, '2026-08-29')
        self.assertEqual(len(rows), 1)
        self.assertIn('current', rows[0])

    def test_every_card_gets_same_model_and_its_own_main_event_schedule(self):
        frame = _forecast_frame()
        frame['event id'] = ['a' * 16, 'b' * 16]
        frame['event url'] = ['http://ufcstats.com/event-details/' + value for value in frame['event id']]
        announced = build_upcoming_forecast_publication(frame, generated_at_utc=OBSERVED)
        class Builder:
            def matchup_features(self, *args):
                return pd.DataFrame([{'skill_diff': 0.25}])
        model = DiscreteTimeOutcomeModel(['skill_diff'], c_value=0.1).fit(_training_frame())
        with tempfile.TemporaryDirectory() as directory:
            publications = publish_upcoming_outcomes(model, Builder(), announced, directory,
                selected_c=0.1, training_input_sha256='c' * 64, model_trained_through='2026-08-22',
                forecast_issued_at_utc=OBSERVED.isoformat(), source_commit_sha='d' * 40)
            self.assertEqual(len(list(Path(directory).glob('*.json'))), 2)
        self.assertEqual(len({row['model_id'] for row in publications}), 1)
        self.assertEqual([row['matchups'][0]['scheduled_rounds'] for row in publications], [5, 5])
        self.assertEqual([row['event_date'] for row in publications], ['2026-08-30', '2026-09-05'])


if __name__ == '__main__':
    unittest.main()
