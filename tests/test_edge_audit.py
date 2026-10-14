import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from market_tracker.edge_audit import build_edge_audit, normalize_picks
from market_capture_windows import utc


def pick(**extra):
    return dict(strategy='Model', pick_id='p', market='moneyline', event_id='e', event_date='2026-10-10',
        matchup_id='m', fighter_id='a', opponent_id='b', selected_fighter_id='a', side='fighter', line=None,
        event_start_utc='2026-10-10T14:00:00Z', issued_at_utc='2026-10-09T14:00:00Z',
        book='Target', probability=.7, moneyline=100, settlement_status='win', profit_units=1, risk_units=1, **extra)


def quote(book, probability=.55, **extra):
    body = dict(book=book, quote_id=book, event_id='e', capture_id='final', matchup_id='m',
        fighter_id='a', opponent_id='b', event_start_utc='2026-10-10T14:00:00Z', source='provider',
        timing_precision='timestamp', observed_at_utc='2026-10-10T13:00:00Z',
        source_quote_updated_at_utc='2026-10-10T12:59:00Z', source_commence_time_utc='2026-10-10T14:00:00Z',
        no_vig_fighter_probability=probability, fighter_moneyline=-120, opponent_moneyline=100)
    body.update(extra)
    return body


class EdgeAuditTests(unittest.TestCase):
    def run_audit(self, picks=None, quotes=None, market='moneyline', metadata=None, now='2026-10-10T15:00:00Z'):
        rows = quotes if quotes is not None else [quote('Target', .99), quote('B'), quote('C'), quote('D')]
        return build_edge_audit(picks or [pick()], {market: rows}, metadata if metadata is not None else rows, utc(now))

    def test_target_excluded_and_returns_use_original_price(self):
        r = self.run_audit()['records'][0]
        self.assertAlmostEqual(r['final_fair_probability'], .55)
        self.assertAlmostEqual(r['claimed_expected_return'], .4)
        self.assertAlmostEqual(r['final_fair_expected_return'], .1)
        self.assertEqual(r['other_books'], ['b', 'c', 'd'])
        self.assertEqual(r['moneyline'], 100)

    def test_totals_exact_line_period_and_no_book_double_count(self):
        p=pick();p.update(market='total_rounds', side='under', line=2.5, selected_fighter_id=None)
        qs=[quote(book, line=2.5, period='full_fight', no_vig_over_probability=.7, over_moneyline=-200, under_moneyline=160)
            for book in ['Target','B','C','D']]
        qs += [quote('Other',line=1.5,period='full_fight',no_vig_over_probability=.01)]
        r=self.run_audit([p], qs, 'total_rounds')['records'][0]
        self.assertAlmostEqual(r['final_fair_probability'], .3)
        bad=copy.deepcopy(qs);bad[2]['period']='round_1';bad[3]['line']=1.5
        self.assertEqual(self.run_audit([p],bad,'total_rounds')['records'][0]['final_status'],'fewer_than_three_other_books')

    def test_reverse_fighters_keeps_selected_identity(self):
        qs=[quote('Target'),quote('B'),quote('C'),quote('D',.45,fighter_id='b',opponent_id='a')]
        r=self.run_audit(quotes=qs)['records'][0]
        self.assertAlmostEqual(r['final_fair_probability'], .55)

    def test_stale_future_started_or_asynchronous_peers_cannot_supply_reference(self):
        for changes in [dict(source_quote_updated_at_utc='2026-10-10T12:00:00Z'),
                        dict(source_quote_updated_at_utc='2026-10-10T13:01:00Z'),
                        dict(source_commence_time_utc='2026-10-10T12:00:00Z'),
                        dict(observed_at_utc='2026-10-10T13:01:00Z'), dict(capture_id='other'),
                        dict(opponent_id='wrong')]:
            qs=[quote('Target'),quote('B'),quote('C'),quote('D',**changes)]
            self.assertEqual(self.run_audit(quotes=qs)['records'][0]['final_status'],'fewer_than_three_other_books')
        self.assertEqual(self.run_audit(metadata=[])['records'][0]['final_status'],'missing_final_same_book_quote')

    def test_missing_final_not_due_methods_and_pre_entry_quotes_stay_missing(self):
        r=self.run_audit(now='2026-10-10T12:00:00Z')['records'][0]
        self.assertEqual(r['final_status'],'not_due')
        p=pick();p['issued_at_utc']='2026-10-10T13:00:00Z'
        self.assertEqual(self.run_audit([p])['records'][0]['final_status'],'missing_final_same_book_quote')
        p['market']='method'
        self.assertEqual(self.run_audit([p])['records'][0]['final_status'],'method_final_source_timing_unavailable')

    def test_chooses_last_same_book_capture_without_falling_back_to_better_older_market(self):
        qs=[quote('Target'),quote('B'),quote('C'),quote('D')]
        qs.append(quote('Target',.9,quote_id='late',capture_id='late',observed_at_utc='2026-10-10T13:30:00Z',source_quote_updated_at_utc='2026-10-10T13:29:00Z'))
        r=self.run_audit(quotes=qs)['records'][0]
        self.assertEqual(r['final_quote_id'],'late')
        self.assertIsNone(r['final_fair_expected_return'])

    def test_return_summary_excludes_voids_and_pending_and_removes_biggest_winner(self):
        ps=[pick()]
        for identifier, status, profit, risk in [('big','win',12.48,1),('loss','loss',-1,1),('void','void',0,0),('pending',None,None,None)]:
            p=pick();p.update(pick_id=identifier,settlement_status=status,profit_units=profit,risk_units=risk)
            ps.append(p)
        s=next(r for r in self.run_audit(ps)['breakdowns'] if r['dimension']=='strategy')
        self.assertEqual(s['settled_bets'],3)
        self.assertAlmostEqual(s['profit_units'],12.48)
        self.assertAlmostEqual(s['profit_without_largest_win_units'],0)
        self.assertEqual(s['void_bets'],1)
        self.assertEqual(s['pending_bets'],1)

    def test_adapters_use_actual_settlement_statuses_and_do_not_change_ledgers(self):
        d=dict(event_id='e',event_date='2026-10-10',matchup_id='m',fighter_id='a',opponent_id='b',
            event_start_utc='2026-10-10T14:00:00Z',decision_id='p',paper_action='fighter',
            action_probability=.7,action_reference_moneyline=100,decision_issued_at_utc='2026-10-09T14:00:00Z',reference_quote_id='Target')
        files={'paper_decisions.jsonl':[d], 'paper_settlements.jsonl':[dict(decision_id='p',settlement_status='paper_win',hypothetical_profit_units=1,hypothetical_risk_units=1)]}
        before=copy.deepcopy(files)
        ps=normalize_picks(lambda name,default:files.get(name,default), {'moneyline':[quote('Target')]})
        self.assertEqual(ps[0]['settlement_status'],'win')
        self.assertEqual(self.run_audit(ps)['breakdowns'][-1]['settled_bets'],1)
        self.assertEqual(files,before)


if __name__ == '__main__':
    unittest.main()
