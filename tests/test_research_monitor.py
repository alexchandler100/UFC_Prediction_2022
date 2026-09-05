import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from update_research_monitor import histories, experiment, update, utc


class ResearchMonitorTests(unittest.TestCase):
    def quote(self, **extra):
        return dict(matchup_id='m', event_id='e', fighter_id='a', opponent_id='b', fighter_name='A',
            opponent_name='B', book='Book', observed_at_utc='2026-09-01T12:00:00Z', quote_id='q',
            fighter_moneyline=-110, opponent_moneyline=100, **extra)

    def test_fighter_identity_survives_reversed_orientation(self):
        a = self.quote()
        b = {**a, 'fighter_id': 'b', 'opponent_id': 'a', 'fighter_name': 'B', 'opponent_name': 'A',
             'fighter_moneyline': 200, 'opponent_moneyline': -250, 'observed_at_utc': '2026-09-02T12:00:00Z'}
        result = histories({'moneyline': [b, a]}, [{'quote_id':'q','source_quote_updated_at_utc':'2026-09-01T11:59:00Z'}])
        series = next(s for s in result['series'] if s['selection_id'] == 'a')
        self.assertEqual([p['moneyline'] for p in series['points']], [-110, -250])
        self.assertEqual(len(result['series']), 2)

    def test_totals_lines_methods_and_duplicate_observations(self):
        total = self.quote(line=1.5, period='full_fight', over_moneyline=120, under_moneyline=-140)
        method = self.quote(fighter_ko_tko_moneyline=400, opponent_decision_moneyline=200)
        result = histories({'total_rounds':[total,{**total,'line':2.5}], 'method':[method,method]}, [])
        self.assertEqual(len(result['series']), 6)
        self.assertTrue(all(len(s['points']) == 1 for s in result['series']))
        self.assertTrue(all(s['points'][0]['source_quote_updated_at_utc'] is None for s in result['series']))

    def test_passes_future_and_overdue_results_are_distinct(self):
        decision = {'decision_id':'one','decision_issued_at_utc':'2026-09-01T12:00:00Z',
                    'event_start_utc':'2026-09-02T12:00:00Z','hypothetical_risk_units':0}
        second = {**decision,'decision_id':'two','event_start_utc':'2026-09-10T12:00:00Z','hypothetical_risk_units':1}
        result=experiment('test',[decision,second],[],utc('2026-09-05T12:00:00Z'))
        self.assertEqual((result['recorded'],result['recommendations'],result['pending'],result['overdue_review']),(2,1,2,1))
        result=experiment('test',[decision],[{'decision_id':'one','hypothetical_profit_units':0}],utc('2026-09-05T12:00:00Z'))
        self.assertEqual(result['overdue_review'],0)

    def test_publication_reproduces_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);update(root);update(root,validate_only=True)
            path=root/'research_monitor.json'; data=json.loads(path.read_text());data['version']=9;path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): update(root,validate_only=True)


if __name__ == '__main__': unittest.main()
