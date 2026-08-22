import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class WebsiteExplorerContractTests(unittest.TestCase):
    def test_site_is_structured_around_matchup_and_fighter_research(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")

        self.assertIn("Research the fight, not just the pick", page)
        self.assertIn("Fighter database", page)
        self.assertIn("Build a matchup", page)
        self.assertIn("Model &amp; data", page)
        self.assertIn("fighter_explorer.json", script)
        self.assertIn("ensureFighterFights", script)
        self.assertIn("fight_shards", script)
        self.assertIn("function renderFighterProfile", script)
        self.assertIn("function renderMatchup", script)
        self.assertIn("function renderFightHistory", script)
        self.assertIn("data_dictionary.fight_stats", script)
        self.assertIn("Linked Bellator and ONE bouts", script)
        self.assertIn("All recorded MMA bouts", script)
        self.assertIn("promotionFilter", script)
        self.assertNotIn("jquery", page.lower())
        self.assertNotIn("fighterPictures", script)

    def test_complete_data_is_reachable_without_rendering_it_all_up_front(self):
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")

        self.assertIn("fight_columns", script)
        self.assertIn("pairedFight", script)
        self.assertIn("Career raw totals and opponent totals", script)
        self.assertIn("Career dates, divisions, form, and streak metadata", script)
        self.assertIn("Open official UFCStats fight page", script)
        self.assertIn("Result metadata only", script)
        self.assertIn("Open upstream event page", script)
        self.assertIn('details.addEventListener("toggle"', script)

    def test_market_view_names_book_price_consensus_and_expected_return(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "collect-market-snapshot.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("Consensus, best price, and paper decisions", page)
        self.assertIn("Potentially profitable prices", page)
        self.assertIn("Total-round prices and model probabilities", page)
        self.assertIn("automatic betting is intentionally off", script)
        self.assertIn("current_opportunities.json", script)
        self.assertIn("target_book", script)
        self.assertIn("offered_moneyline", script)
        self.assertIn("estimated_expected_return", script)
        self.assertIn("book_quotes", script)
        self.assertIn("locked_t24_decision", script)
        self.assertIn("Leave-one-book-out fair line", script)
        self.assertIn("outcome_forecasts.json", script)
        self.assertIn("positive_candidates", script)
        self.assertIn("Candidate duration-model probability", script)
        self.assertIn("method EV is unavailable", script)
        self.assertIn("Totals profitability and closing-line value", page)
        self.assertIn("renderProfitabilityEvidence", script)
        self.assertIn("Compare predeclared EV thresholds", script)
        self.assertIn("Locked T-24 residual paper decision", script)
        self.assertIn("current_opportunities.json", workflow)

    def test_layout_has_explicit_mobile_breakpoints(self):
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")
        self.assertIn("@media (max-width: 900px)", style)
        self.assertIn("@media (max-width: 600px)", style)
        self.assertIn(".matchup-selectors", style)
        self.assertIn(".fight-summary", style)


if __name__ == "__main__":
    unittest.main()
