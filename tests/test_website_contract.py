import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class WebsiteForecastContractTests(unittest.TestCase):
    def test_upcoming_table_separates_model_market_and_published_forecast(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        self.assertIn("Independent<br>Model Fair Odds", page)
        self.assertIn("Vegas Consensus<br>Fair Odds", page)
        self.assertIn("Published<br>Forecast", page)
        self.assertIn("'predicted fighter odds'", script)
        self.assertIn("'market no-vig fighter probability'", script)
        self.assertIn("'forecast fighter odds'", script)
        self.assertIn("probabilityToAmericanOdds", script)

    def test_paper_market_view_names_the_book_line_consensus_and_ev(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "collect-market-snapshot.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("Paper Betting Opportunities", page)
        self.assertIn("Locked T-24", page)
        self.assertIn("Target Price vs", page)
        self.assertIn("All Captured", page)
        self.assertIn("current_opportunities.json", script)
        self.assertIn("target_book", script)
        self.assertIn("offered_moneyline", script)
        self.assertIn("estimated_expected_return", script)
        self.assertIn("consensus_books", script)
        self.assertIn("target excluded", script)
        self.assertIn("execution_enabled", script)
        self.assertIn("current_opportunities.json", workflow)


if __name__ == "__main__":
    unittest.main()
