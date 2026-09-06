import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import capture_market_snapshot as collector
from market_tracker._common import canonical_hash


class OpportunityPublicationSizeTests(unittest.TestCase):
    def test_book_rich_card_exceeding_old_limit_preserves_every_price_and_hash(self):
        book = {'book': 'Example', 'fighter_moneyline': 110, 'opponent_moneyline': -130,
                'source_quote_updated_at_utc': '2026-09-05T12:00:00Z',
                'bayesian_kelly': {'posterior_mean_probability': .51, 'posterior_lower_probability': .42,
                                   'recommended_fraction': 0, 'status': 'available'}}
        publication = {'matchups': [{'matchup_id': str(i), 'fighter_name': 'José',
            'book_quotes': [{**book, 'book': f'Book {j}'} for j in range(40)]} for i in range(20)]}
        publication['publication_sha256'] = canonical_hash(publication)
        self.assertGreater(len(json.dumps(publication, indent=2).encode()), 256 * 1024)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'current_opportunities.json'
            with patch.object(collector, 'CURRENT_OPPORTUNITIES_PATH', path):
                collector._atomic_write_current_opportunities(publication)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), publication)
            self.assertLess(path.stat().st_size, collector.CURRENT_OPPORTUNITIES_SIZE_LIMIT)
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_oversized_utf8_payload_does_not_replace_existing_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'current_opportunities.json'
            path.write_text('{"existing":true}', encoding='utf-8')
            with patch.object(collector, 'CURRENT_OPPORTUNITIES_PATH', path):
                with self.assertRaisesRegex(collector.CaptureError, r'bytes > 1,048,576 bytes'):
                    collector._atomic_write_current_opportunities({'text': 'é' * (600 * 1024)})
            self.assertEqual(path.read_text(), '{"existing":true}')
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])


if __name__ == '__main__': unittest.main()
