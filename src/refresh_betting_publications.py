"""Rebuild current betting views offline without inventing a new odds capture.

The existing immutable price/decision/forecast ledgers are read only. Original
quote times remain unchanged, so the browser can expire old prices correctly.
Newly computed views are not appended as if published at the old capture time.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import capture_market_snapshot as capture
import capture_method_market_snapshot as methods
from market_tracker._common import canonical_hash
from market_tracker.opportunities import build_current_opportunities
from upcoming_bet_board import build_upcoming_bet_board, write_upcoming_bet_board
from update_bet_performance import update_bet_performance


ROOT = Path(__file__).resolve().parent
MARKET = ROOT / 'content/data/market'
EXTERNAL = ROOT / 'content/data/external'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def read_store(store: type, stem: str) -> tuple:
    return store(MARKET / f'{stem}.csv', MARKET / f'{stem}.jsonl').read()


def write_json(path: Path, data: dict) -> None:
    # Reuse the publication writer's atomic file replacement.
    from market_tracker.bankroll import _atomic_json
    _atomic_json(path, data)


def refresh() -> dict:
    report = read_json(MARKET / 'capture_report.json')
    previous = read_json(MARKET / 'current_opportunities.json')
    opportunities = build_current_opportunities(
        read_store(capture.QuoteSnapshotStore, 'quote_snapshots'),
        read_store(capture.ForecastCaptureStore, 'forecast_captures'),
        read_store(capture.QuoteSourceMetadataStore, 'quote_source_metadata'),
        read_store(capture.PaperDecisionStore, 'paper_decisions'),
        capture_id=report['capture_id'],
        total_round_quotes=read_store(capture.TotalRoundsQuoteStore, 'total_round_quote_snapshots'),
        total_round_forecasts=read_store(capture.TotalRoundsForecastStore, 'total_round_forecast_captures'),
        total_round_decisions=read_store(capture.TotalRoundsPaperDecisionStore, 'total_round_paper_decisions'),
        bayesian_filtered_decisions=read_store(capture.BayesianFilteredDecisionStore, 'bayesian_filtered_paper_decisions'),
        method_price_status=previous.get('prop_markets', {}).get('method_of_victory', {}).get('price_status', 'unavailable'),
    )
    board = build_upcoming_bet_board(
        read_json(EXTERNAL / 'all_upcoming_forecasts.json'),
        read_store(capture.EarlyMarketObservationStore, 'early_market_observations'),
        observed_at_utc=opportunities['observed_at_utc'],
        source=opportunities['source'], current_opportunities=opportunities,
    )
    write_json(MARKET / 'current_opportunities.json', opportunities)
    write_upcoming_bet_board(board)
    report.update({
        'opportunity_publication_sha256': opportunities['publication_sha256'],
        'upcoming_bet_board_sha256': board['publication_sha256'],
        'upcoming_bet_board_qualified_bets': board['qualified_bet_count'],
        'upcoming_bet_board_announced_events': board['announced_event_count'],
        'view_rebuilt_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    report.pop('report_sha256', None)
    report['report_sha256'] = canonical_hash(report)
    write_json(MARKET / 'capture_report.json', report)
    method_report = read_json(MARKET / 'method_capture_report.json')
    method_records = read_store(methods.MethodMarketStore, 'method_market_snapshots')
    method_forecasts = read_store(methods.MethodForecastStore, 'method_forecast_captures')
    method_view = methods._build_current_method_publication(
        method_records,
        event_id=method_report['event_id'], event_date=method_report['event_date'],
        event_start_utc=method_report['event_start_utc'],
        outcome_forecasts=methods._outcome_forecasts(method_report['event_id']),
    )
    methods._write_current_method_publication(method_view)
    method_report['current_publication_sha256'] = method_view['publication_sha256']
    method_report['records_total'] = len(method_records)
    method_report['dataset_sha256'] = methods.MethodMarketStore.dataset_sha256(method_records)
    method_report['method_forecasts_total'] = len(method_forecasts)
    method_report['method_forecast_dataset_sha256'] = methods.MethodForecastStore.dataset_sha256(method_forecasts)
    method_report['view_rebuilt_at_utc'] = report['view_rebuilt_at_utc']
    # Method report schema currently has no report hash; support one if added.
    if 'report_sha256' in method_report:
        method_report.pop('report_sha256')
        method_report['report_sha256'] = canonical_hash(method_report)
    write_json(MARKET / 'method_capture_report.json', method_report)
    update_bet_performance()
    return {'qualified_bets': board['qualified_bet_count'],
            'odds_observed_at_utc': board['observed_at_utc'],
            'ledgers_appended': False, 'network_requests': 0}


if __name__ == '__main__':
    print(json.dumps(refresh(), sort_keys=True))
