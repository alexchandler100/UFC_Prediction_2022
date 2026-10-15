import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from external_mma.wiki_results import parse_page, discover_event_pages, WikiResultsAdapter


def payload(division='Lightweight', when='January 1, 2024', extra='', linked=True):
    name='<a href="/wiki/A_Fighter">A Fighter</a>' if linked else 'A Fighter'
    return {'parse':{'title':'Event season','pageid':1,'revid':100,'text':{'*':f'''
      <table class="infobox"><tr><th>Date</th><td>{when}</td></tr></table>
      <h3>Results</h3><table><tr><th>Fight card</th></tr>
      <tr><td>{division}</td><td>{name}</td><td>def.</td><td>B Fighter</td>
      <td>TKO (punches)</td><td>1</td><td>1:23</td><td>{extra}</td></tr></table>'''}}}


class WikiResultTests(unittest.TestCase):
    def test_imported_metadata_has_revision_attribution_without_inventing_statistics(self):
        r=parse_page(payload(),'Road to UFC','2024-01-02').observations[0]
        self.assertEqual(r.event_date,'2024-01-01')
        self.assertEqual(r.source_url,'https://en.wikipedia.org/w/index.php?oldid=100')
        self.assertEqual(r.finish_clock_seconds,83)
        self.assertIsNone(r.scheduled_rounds)
        self.assertIn('A_Fighter',(r.fighter_source_id,r.opponent_source_id))
        self.assertTrue(any(i.startswith('unresolved:Road to UFC:2024-01-01:') for i in (r.fighter_source_id,r.opponent_source_id)))

    def test_future_and_undated_results_stay_out(self):
        self.assertEqual(parse_page(payload(),'Road to UFC','2023-12-31').observations,[])
        self.assertEqual(parse_page(payload(when='TBA'),'Road to UFC','2024-01-02').observations,[])

    def test_flag_links_cannot_become_fighter_identities(self):
        p=payload(division='Flyweight MMA')
        p['parse']['text']['*']=p['parse']['text']['*'].replace('<td><a href="/wiki/A_Fighter">', '<td><a href="/wiki/Japan"><img src="flag.svg"></a><a href="/wiki/A_Fighter">')
        row=parse_page(p,'ONE','2024-01-02').observations[0]
        self.assertNotIn('Japan',(row.fighter_source_id,row.opponent_source_id))
        self.assertIn('A_Fighter',(row.fighter_source_id,row.opponent_source_id))

    def test_mixed_sport_and_exhibition_filtering(self):
        self.assertEqual(len(parse_page(payload(division='Flyweight MMA'),'ONE','2024-01-02').observations),1)
        for division in ['Flyweight Muay Thai','Lightweight kickboxing','Submission grappling','Flyweight']:
            self.assertEqual(parse_page(payload(division=division),'ONE','2024-01-02').observations,[])
        self.assertEqual(parse_page(payload(extra='exhibition'),'Rizin','2024-01-02').observations,[])

    def test_yearly_summary_is_not_a_second_event_and_duplicates_keep_same_bout_id(self):
        p=payload();p['parse']['text']['*']+='<h2>Title fights</h2>'+p['parse']['text']['*'].split('<h3>Results</h3>')[1]
        self.assertEqual(len(parse_page(p,'KSW','2024-01-02').observations),1)
        a=parse_page(payload(),'PFL','2024-01-02').observations[0]
        other=payload();other['parse']['title']='Different article';other['parse']['revid']=101
        b=parse_page(other,'PFL','2024-01-02').observations[0]
        self.assertEqual(a.observation_id,b.observation_id)

    def test_colon_event_titles_are_allowed_but_namespaces_and_future_cards_are_not(self):
        p=payload();p['parse']['text']['*']='''<table><tr><th>#</th><th>Event</th><th>Date</th></tr>
        <tr><td>1</td><td><a href="/wiki/PFL_Dubai:_A_vs._B">Event</a></td><td>January 1, 2024</td></tr>
        <tr><td>2</td><td><a href="/wiki/File:picture">File</a></td><td>January 1, 2024</td></tr>
        <tr><td>3</td><td><a href="/wiki/Future">Future</a></td><td>January 1, 2025</td></tr></table>'''
        self.assertEqual(discover_event_pages(p,'2024-01-02'),['PFL Dubai: A vs. B'])

    def test_changed_results_go_to_review_instead_of_overwriting(self):
        p=payload();old=parse_page(p,'PFL','2024-01-02').observations
        p['parse']['text']['*']=p['parse']['text']['*'].replace('TKO (punches)','Submission')
        p['parse']['revid']=101
        r=WikiResultsAdapter('PFL','2024-01-02',old).convert(json.dumps(p).encode(),'a'*64)
        self.assertEqual(r.observations,[])
        self.assertEqual(r.rejected[-1]['reason'],'source_revision_changed_result_or_identity')


if __name__=='__main__':unittest.main()
