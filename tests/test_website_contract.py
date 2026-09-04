import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class WebsiteExplorerContractTests(unittest.TestCase):
    def test_site_is_structured_around_matchup_and_fighter_research(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")

        self.assertIn("Fighter database", page)
        self.assertIn("Build a matchup", page)
        self.assertNotIn("Research the fight, not just the pick", page)
        self.assertNotIn("Model &amp; data", page)
        self.assertNotIn('id="view-data"', page)
        self.assertIn("fighter_explorer.json", script)
        self.assertIn("ensureFighterFights", script)
        self.assertIn("fight_shards", script)
        self.assertIn("function renderFighterProfile", script)
        self.assertIn("function renderMatchup", script)
        self.assertIn("function renderFightHistory", script)
        self.assertIn("data_dictionary.fight_stats", script)
        self.assertIn("Linked Bellator and ONE bouts", script)
        self.assertIn("All recorded MMA bouts", script)
        self.assertIn("recordWithBoutCount", script)
        self.assertIn("compareFightsNewestFirst", script)
        self.assertIn("details.dataset.fightDate", script)
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
        self.assertIn("Qualified upcoming paper bets", page)
        self.assertIn("All announced UFC cards", page)
        self.assertNotIn("Potentially profitable prices", page)
        self.assertIn("All current total and method prices", page)
        self.assertIn('id="market-book-filter"', page)
        self.assertIn("automatic betting is intentionally off", script)
        self.assertIn("current_opportunities.json", script)
        self.assertIn("odds_history.json", script)
        self.assertIn("upcoming_bet_board.json", script)
        self.assertIn("bet_performance.json", script)
        self.assertIn("function renderQualifiedUpcomingBets", script)
        self.assertIn("ranked by estimated return", script)
        self.assertIn("function renderMarketBookFilter", script)
        self.assertIn("function appendQualifiedBetExplanation", script)
        self.assertIn("combined sizing is not implemented yet", script)
        self.assertIn("target_book", script)
        self.assertIn("offered_moneyline", script)
        self.assertIn("estimated_expected_return", script)
        self.assertIn("book_quotes", script)
        self.assertIn("locked_t24_decision", script)
        self.assertIn("Leave-one-book-out fair line", script)
        self.assertIn("outcome_forecasts.json", script)
        self.assertIn("current_method_markets.json", script)
        self.assertIn("Missing outcomes remain unavailable", script)
        self.assertIn("method-price-table", script)
        self.assertIn("Profitability and closing-line value", page)
        self.assertIn("renderProfitabilityEvidence", script)
        self.assertIn("Compare predeclared EV thresholds", script)
        self.assertIn("Locked T-24 residual paper decision", script)
        self.assertIn("bayesian_winner_challenger.json", script)
        self.assertIn("bayesianFilteredCandidate", script)
        self.assertIn("filtered minus base roi", script.lower())
        self.assertIn("Probability EV is positive", script)
        self.assertIn("Bayesian model and expected-return uncertainty", script)
        self.assertIn("bayesian_winner_challenger.json", update_workflow)
        self.assertIn("current_opportunities.json", workflow)
        self.assertIn("odds_history.json", workflow)
        self.assertIn("odds_history.json", update_workflow)
        self.assertIn("upcoming_bet_board.json", workflow)
        self.assertIn("published_bet_snapshots.json", workflow)
        self.assertIn("bet_performance.json", workflow)
        self.assertIn("all_upcoming_forecasts.json", update_workflow)
        self.assertIn("current_method_markets.json", workflow)
        self.assertIn("bayesian_filtered_paper_decisions.jsonl", workflow)

    def test_each_moneyline_fight_has_consensus_and_book_odds_history(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn("consensus moved over time", page)
        self.assertIn("function renderOddsHistory", script)
        self.assertIn("function oddsHistoryChart", script)
        self.assertIn("const uncapturedMatchups = legacyRows()", script)
        self.assertIn("No timestamped book capture is available for this fight yet", script)
        self.assertIn("Odds movement over time", script)
        self.assertIn("Show source", script)
        self.assertIn("series.kind === \"book\"", script)
        self.assertIn("pointerdown", script)
        self.assertIn(".odds-history-svg", style)
        self.assertIn("width: 100%", style)
        self.assertIn(".odds-history-controls { align-items: stretch; flex-direction: column; }", style)

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
        self.assertIn("renderMatchup(fighterA, fighterB);", script)
        self.assertIn('focusRouteTarget("#matchup-workbench", expectedHash)', script)
        self.assertIn(".market-card.is-route-target", style)
        self.assertIn("--route-scroll-offset: 96px", style)
        self.assertIn("--route-scroll-offset: 140px", style)

    def test_matchups_lists_every_announced_event_as_compact_expandable_rows(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")
        publication = json.loads(
            (REPO_ROOT / "src/content/data/external/all_upcoming_forecasts.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertGreater(len(publication["events"]), 1)
        self.assertIn("All upcoming UFC fights", page)
        self.assertIn("all_upcoming_forecasts.json", script)
        self.assertIn("function allUpcomingEventGroups", script)
        self.assertIn("state.upcomingBetBoard?.market_matchups", script)
        self.assertIn('element("details", "upcoming-bout")', script)
        self.assertIn("upcoming-bout-summary", script)
        self.assertIn("upcoming-bout-details", script)
        self.assertIn("const boutNumber = eventBoutCount - fallbackIndex", script)
        self.assertIn("Bookie odds available from", script)
        self.assertIn("function upcomingBookPriceDetails", script)
        self.assertIn("stored book price", script)
        self.assertIn("Consensus unavailable", script)
        self.assertIn("main event first within each card", script)
        self.assertIn(".upcoming-event-group", style)
        self.assertIn(".upcoming-bout-summary", style)
        self.assertIn(".upcoming-bout-details", style)
        self.assertIn(".upcoming-odds-indicator.has-odds", style)
        self.assertIn(".upcoming-price-grid", style)

    def test_routes_focus_the_requested_data_instead_of_the_tab_hero(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")

        for target_id in (
            "current-card-results",
            "fighter-directory-controls",
            "fighter-profile",
            "matchup-workbench",
            "fight-graph-controls",
            "fight-graph-results",
            "simulation-picker",
            "simulation-results",
            "qualified-upcoming-bets",
            "market-research-results",
        ):
            self.assertIn(f'id="{target_id}"', page)
        self.assertIn("function focusRouteTarget", script)
        self.assertIn('matchupConfigured ? "#fight-graph-results" : "#fight-graph-controls"', script)
        self.assertIn('focusRouteTarget(requestedMatchupId ? "#simulation-results"', script)
        self.assertIn('focusRouteTarget("#qualified-upcoming-bets", expectedHash)', script)
        self.assertIn(
            'setRoute(`market/${matchup.fighter_id}/${matchup.opponent_id}`)',
            script,
        )

    def test_stale_market_capture_cannot_replace_the_current_card(self):
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")

        self.assertIn("function publicationMatchesCurrentCard", script)
        self.assertIn("function vegasMatchesCurrentCard", script)
        self.assertIn("function currentMarket()", script)
        self.assertIn("publicationMatchesCurrentCard(publication.event_date, publication.event_id)", script)
        self.assertIn("latest stored simulation", script)
        self.assertIn("if (!vegasMatchesCurrentCard()", script)
        self.assertIn(
            "const market = currentMarket();\n  return orderedCardMatchups(market?.matchups?.length ? market.matchups : legacyRows());",
            script,
        )
        self.assertIn("function authoritativeBoutOrderMap", script)
        self.assertIn("function orderedCardMatchups", script)
        self.assertIn("not the current ${state.card?.date", script)
        self.assertNotIn(
            "return state.market?.matchups?.length ? state.market.matchups : legacyRows();",
            script,
        )

    def test_fight_views_use_ordered_rows_distributions_and_comparison_histories(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn('id="simulation-card-list"', page)
        self.assertIn("function boutOrderLabel", script)
        self.assertIn("function renderSimulationCardList", script)
        self.assertIn("function simulationMiniDuration", script)
        self.assertIn("function simulationStatisticGrid", script)
        self.assertIn("matchup-history-columns", script)
        self.assertIn("ensureFighterFights(fighter).then", script)
        self.assertIn("fight-summary-primary", script)
        self.assertIn("fight-summary-secondary", script)
        self.assertIn("grid-template-columns: minmax(0, 1fr);", style)
        self.assertIn(".bar { display: block;", style)
        self.assertIn(".simulation-card-row", style)
        self.assertIn(".matchup-history-columns", style)

    def test_all_upcoming_bet_board_is_compact_and_mobile_safe(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn('id="qualified-upcoming-list"', page)
        self.assertIn('bet?.threshold_met === true', script)
        self.assertIn('Number(right.estimated_expected_return)', script)
        self.assertIn('const item = document.createElement("details")', script)
        self.assertNotIn("bestTotalByMatchup", script)
        self.assertIn("They remain visible", script)
        self.assertIn('.qualified-bet-row', style)
        self.assertIn('.qualified-bet-item', style)
        self.assertIn('.qualified-bet-explanation', style)

    def test_published_bet_performance_has_bankroll_and_timing_controls(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn('data-nav="performance"', page)
        self.assertIn('id="performance-bankroll"', page)
        self.assertIn('id="performance-staking"', page)
        self.assertIn('value="half_kelly"', page)
        self.assertIn('value="half_kelly_model_blend"', page)
        self.assertIn('value="half_kelly_sim_blend"', page)
        self.assertIn('value="half_kelly_model_sim_blend"', page)
        self.assertIn('value="favorite_early_underdog_late"', page)
        self.assertIn("function simulatePaperBankroll", script)
        self.assertIn("function performanceStakePlan", script)
        self.assertIn("function equalLogitPool", script)
        self.assertIn("function renderBetPerformance", script)
        self.assertIn("kellyFraction", script)
        self.assertIn("½ Kelly stake", script)
        self.assertIn("activeNav.offsetLeft", script)
        self.assertIn(".performance-table", style)
        self.assertIn(".performance-filter-grid", style)

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

    def test_fight_graph_viewport_is_responsive_and_touch_enabled(self):
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn("function fightGraphFitBox", script)
        self.assertIn("function resizeFightGraphViewport", script)
        self.assertIn("const pointers = new Map()", script)
        self.assertIn("const beginPinch", script)
        self.assertIn('event.pointerType === "mouse"', script)
        self.assertIn('canvas.addEventListener("touchmove"', script)
        self.assertIn("new ResizeObserver", script)
        self.assertIn("overscroll-behavior: contain", style)
        self.assertIn("height: clamp(440px, 68svh, 620px)", style)
        self.assertIn("min-width: 0", style)
        self.assertNotIn("min-width: 760px", style)

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

    def test_matchup_graph_colors_each_fighter_branch_by_result(self):
        page = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPO_ROOT / "script.js").read_text(encoding="utf-8")
        style = (REPO_ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn('id="fight-graph-color-legend"', page)
        self.assertIn("function renderFightGraphColorLegend", script)
        self.assertIn("const contextCandidates = new Map()", script)
        self.assertIn("const seedBranches = new Map", script)
        self.assertIn("branch: seedBranches.get(edge.winnerId)", script)
        self.assertIn('result: edge.winnerId === fighterId ? "win" : "loss"', script)
        self.assertIn('group.classList.add(`is-branch-${context.branch}`, `is-branch-${context.result}`)', script)
        self.assertIn('arrow.setAttribute("fill", "context-stroke")', script)
        self.assertIn("group.dataset.fighterId = node.id", script)
        for color in ("--graph-a-win", "--graph-b-win", "--graph-a-loss", "--graph-b-loss"):
            self.assertIn(color, style)
        for selector in (
            ".fight-graph-edge.is-branch-a.is-branch-win",
            ".fight-graph-edge.is-branch-b.is-branch-win",
            ".fight-graph-edge.is-branch-a.is-branch-loss",
            ".fight-graph-edge.is-branch-b.is-branch-loss",
        ):
            self.assertIn(selector, style)


if __name__ == "__main__":
    unittest.main()
