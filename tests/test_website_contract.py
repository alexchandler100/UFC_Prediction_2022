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
        update_workflow = (
            REPO_ROOT / ".github" / "workflows" / "update-data.yml"
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
        self.assertIn("Profitability and closing-line value", page)
        self.assertIn("renderProfitabilityEvidence", script)
        self.assertIn("Compare predeclared EV thresholds", script)
        self.assertIn("Locked T-24 residual paper decision", script)
        self.assertIn("bayesian_winner_challenger.json", script)
        self.assertIn("Bayesian-filtered moneyline", script)
        self.assertIn("bayesianFilteredCandidate", script)
        self.assertIn("filtered minus base roi", script.lower())
        self.assertIn("Probability EV is positive", script)
        self.assertIn("Bayesian model and expected-return uncertainty", script)
        self.assertIn("bayesian_winner_challenger.json", update_workflow)
        self.assertIn("current_opportunities.json", workflow)
        self.assertIn("bayesian_filtered_paper_decisions.jsonl", workflow)

    def test_layout_has_explicit_mobile_breakpoints(self):
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")
        self.assertIn("@media (max-width: 900px)", style)
        self.assertIn("@media (max-width: 600px)", style)
        self.assertIn(".matchup-selectors", style)
        self.assertIn(".fight-summary", style)

    def test_current_card_actions_open_the_selected_research_content(self):
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn("function focusMarketMatchup", script)
        self.assertIn(
            'setRoute(`market/${matchup.fighter_id}/${matchup.opponent_id}`)',
            script,
        )
        self.assertIn('details[data-book-lines="moneyline"]', script)
        self.assertIn("if (prices) prices.open = true", script)
        self.assertIn("marketButton.disabled = !hasCurrentPrices", script)
        self.assertIn("renderMatchup(fighterA, fighterB);\n      return;", script)
        self.assertIn(".market-card.is-route-target", style)
        self.assertIn("scroll-margin-top: 96px", style)

    def test_stale_market_capture_cannot_replace_the_current_card(self):
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")

        self.assertIn("function publicationMatchesCurrentCard", script)
        self.assertIn("function vegasMatchesCurrentCard", script)
        self.assertIn("function currentMarket()", script)
        self.assertIn("if (!vegasMatchesCurrentCard()", script)
        self.assertIn(
            "const market = currentMarket();\n  return market?.matchups?.length ? market.matchups : legacyRows();",
            script,
        )
        self.assertIn("not the current ${state.card?.date", script)
        self.assertNotIn(
            "return state.market?.matchups?.length ? state.market.matchups : legacyRows();",
            script,
        )

    def test_fight_graph_has_zoom_and_pan_navigation(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn('id="graph-zoom-in"', page)
        self.assertIn('id="graph-zoom-out"', page)
        self.assertIn('id="graph-zoom-fit"', page)
        self.assertIn("function configureFightGraphViewport", script)
        self.assertIn('svg.addEventListener("wheel"', script)
        self.assertIn('svg.addEventListener("pointerdown"', script)
        self.assertIn("ArrowRight", script)
        self.assertIn("touch-action: none", style)

    def test_fight_graph_lists_edges_with_expandable_statistics(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn('id="fight-graph-edge-rows"', page)
        self.assertIn("Winner / result", page)
        self.assertIn("function renderFightGraphEdgeTable", script)
        self.assertIn("View stats", script)
        self.assertIn("renderFightDetails(body, edge, opponentFight", script)
        self.assertIn(".fight-graph-edge-table", style)

    def test_fight_graph_has_simple_and_advanced_fighter_filtering(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn('id="graph-mode-simple"', page)
        self.assertIn('id="graph-mode-advanced"', page)
        self.assertIn('id="graph-rule-list"', page)
        self.assertIn("Five-round wins (title/main event)", script)
        self.assertIn("function aggregateGraphFighter", script)
        self.assertIn("function graphRuleMatches", script)
        self.assertIn("GRAPH_FILTER_PRESETS", script)
        self.assertIn("Matching fighters and their opponents", page)
        self.assertIn('id="graph-rule-stance"', page)
        self.assertIn('id="graph-fight-method"', page)
        self.assertIn('id="graph-fight-round"', page)
        self.assertIn('id="graph-fight-detail"', page)
        self.assertIn("function fightMatchesAdvancedConstraints", script)
        self.assertIn("avg_opponent_win_rate", script)
        self.assertIn(".graph-rule-row", style)

    def test_dark_mode_is_the_default_theme(self):
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn("color-scheme: dark", style)
        self.assertIn("--cream: #0b1118", style)
        self.assertIn("Dark is the default presentation", style)

    def test_fight_graph_has_quick_time_ranges_and_custom_years(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn('data-graph-years="all"', page)
        self.assertIn('data-graph-years="1"', page)
        self.assertIn('data-graph-years="3"', page)
        self.assertIn('data-graph-years="5"', page)
        self.assertIn('data-graph-years="10"', page)
        self.assertIn('id="graph-custom-years"', page)
        self.assertIn("function applyGraphQuickRange", script)
        self.assertIn("setUTCFullYear", script)
        self.assertIn(".graph-date-shortcuts", style)

    def test_fight_table_shows_values_for_advanced_numeric_rules(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn('id="fight-graph-edge-headings"', page)
        self.assertIn("function fightGraphTableMetrics", script)
        self.assertIn("function renderFightGraphEdgeHeadings", script)
        self.assertIn("function formatGraphTableMetric", script)
        self.assertIn("aggregates.get(edge.winnerId)", script)
        self.assertIn("aggregates.get(edge.loserId)", script)
        self.assertIn('"Winner / loser"', script)
        self.assertIn(".fight-graph-edge-metric", style)

    def test_matchup_graph_expands_two_fighters_by_depth(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn('id="graph-mode-matchup"', page)
        self.assertIn('id="graph-matchup-fighter-a"', page)
        self.assertIn('id="graph-matchup-fighter-b"', page)
        self.assertIn('id="graph-matchup-depth"', page)
        self.assertIn("function filteredMatchupFightGraph", script)
        self.assertIn("for (let level = 0; level < depth", script)
        self.assertIn("function layoutMatchupFightGraph", script)
        self.assertIn('setRoute(`graph/${fighter.id}/${opponent.id}`)', script)
        self.assertIn('actionButton("View fight graph"', script)
        self.assertIn("configureGraphMatchup(parts[1], parts[2], 1)", script)
        self.assertIn(".fight-graph-node.is-seed", style)


if __name__ == "__main__":
    unittest.main()
