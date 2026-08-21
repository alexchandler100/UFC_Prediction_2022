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


if __name__ == "__main__":
    unittest.main()
