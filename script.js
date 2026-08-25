"use strict";

const DATA_PATHS = {
  explorer: "src/content/data/external/fighter_explorer.json",
  vegas: "src/content/data/external/vegas_odds.json",
  card: "src/content/data/external/card_info.json",
  model: "src/content/data/external/winner_model.json",
  bayesian: "src/content/data/external/bayesian_winner_challenger.json",
  market: "src/content/data/market/current_opportunities.json",
  performance: "src/content/data/market/performance_report.json",
  outcomes: "src/content/data/external/outcome_forecasts.json",
};

const state = {
  explorer: null,
  vegas: null,
  card: null,
  model: null,
  bayesian: null,
  market: null,
  performance: null,
  outcomes: null,
  fighters: [],
  fighterById: new Map(),
  bayesianByPair: new Map(),
  fightColumn: new Map(),
  shardCache: new Map(),
  fightGraphPromise: null,
  fightGraphEdges: [],
  fightGraphFighterRows: new Map(),
  fightGraphFilterMode: "simple",
  fightGraphRuleSequence: 0,
  fightGraphPinnedId: null,
  fightGraphRenderToken: 0,
  fightGraphViewport: null,
  selected: { a: null, b: null },
  directoryLimit: 48,
};

const FIGHT_GRAPH_MAX_FIGHTERS = 140;
const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const GRAPH_RULE_METRICS = [
  { key: "fights", label: "Fights", group: "Results", description: "All recorded bouts in the selected period." },
  { key: "wins", label: "Wins", group: "Results", description: "Recorded wins in the selected period." },
  { key: "losses", label: "Losses", group: "Results", description: "Recorded losses in the selected period." },
  { key: "draws_nc", label: "Draws / no contests", group: "Results", description: "Draws and no contests in the selected period." },
  { key: "win_rate", label: "Win rate (%)", group: "Results", description: "Wins divided by wins plus losses." },
  { key: "win_streak", label: "Latest win streak", group: "Results", description: "Consecutive wins from the fighter's latest bout in the selected period." },
  { key: "unique_opponents", label: "Unique opponents", group: "Results", description: "Distinct opponents faced in the selected period." },
  { key: "finish_wins", label: "Finish wins", group: "Wins by type", description: "Wins that did not end by decision." },
  { key: "ko_wins", label: "KO / TKO wins", group: "Wins by type", description: "Wins recorded as KO or TKO." },
  { key: "submission_wins", label: "Submission wins", group: "Wins by type", description: "Wins recorded as submissions." },
  { key: "decision_wins", label: "Decision wins", group: "Wins by type", description: "Unanimous, split, majority, or other decision wins." },
  { key: "five_round_wins", label: "Five-round wins (title/main event)", group: "Wins by type", description: "Known five-round wins using published schedules and modern UFC headliner position; title status is not separate." },
  { key: "total_knockdowns", label: "Knockdowns", group: "Striking", description: "Total knockdowns recorded for the fighter." },
  { key: "avg_sig_landed", label: "Avg. significant strikes landed", group: "Striking", description: "Average per bout with detailed statistics.", step: "0.1" },
  { key: "sig_landed_per_minute", label: "Significant strikes landed / minute", group: "Striking", description: "Significant strikes landed divided by recorded fight time.", step: "0.1" },
  { key: "sig_accuracy", label: "Significant-strike accuracy (%)", group: "Striking", description: "Total significant strikes landed divided by attempted.", step: "0.1" },
  { key: "head_landed", label: "Head strikes landed", group: "Striking", description: "Total significant head strikes landed." },
  { key: "body_landed", label: "Body strikes landed", group: "Striking", description: "Total significant body strikes landed." },
  { key: "leg_landed", label: "Leg strikes landed", group: "Striking", description: "Total significant leg strikes landed." },
  { key: "total_takedowns", label: "Takedowns landed", group: "Grappling", description: "Total takedowns landed." },
  { key: "takedown_accuracy", label: "Takedown accuracy (%)", group: "Grappling", description: "Total takedowns landed divided by attempted.", step: "0.1" },
  { key: "submission_attempts", label: "Submission attempts", group: "Grappling", description: "Total submission attempts." },
  { key: "control_minutes", label: "Control time (minutes)", group: "Grappling", description: "Total recorded control time.", step: "0.1" },
  { key: "avg_fight_minutes", label: "Avg. fight duration (minutes)", group: "Pace & duration", description: "Average duration of bouts with a known duration.", step: "0.1" },
  { key: "total_fight_minutes", label: "Total fight time (minutes)", group: "Pace & duration", description: "Combined known fight duration.", step: "0.1" },
  { key: "fastest_win_minutes", label: "Fastest win (minutes)", group: "Pace & duration", description: "Elapsed time of the fighter's fastest recorded win.", step: "0.1" },
  { key: "stats_coverage", label: "Detailed-stat coverage (%)", group: "Data quality", description: "Share of selected bouts with detailed statistics.", step: "0.1" },
  { key: "avg_opponent_wins", label: "Avg. opponent career wins", group: "Opponent quality", description: "Average published career wins of opponents faced in the period.", step: "0.1" },
  { key: "avg_opponent_win_rate", label: "Avg. opponent career win rate (%)", group: "Opponent quality", description: "Average published career win rate of opponents faced in the period.", step: "0.1" },
  { key: "age", label: "Age at period end", group: "Profile", description: "Age at the selected end date or dataset date." },
  { key: "height", label: "Height (inches)", group: "Profile", description: "Published fighter height." },
  { key: "reach", label: "Reach (inches)", group: "Profile", description: "Published fighter reach." },
  { key: "career_wins", label: "Career wins (all data)", group: "Career profile", description: "Career wins across the complete published history." },
  { key: "career_fights", label: "Career fights (all data)", group: "Career profile", description: "Career recorded bouts across the complete published history." },
];
const GRAPH_RULE_OPERATORS = [
  ["gt", "more than"], ["gte", "at least"], ["eq", "exactly"], ["lte", "at most"], ["lt", "less than"], ["between", "between"],
];
const GRAPH_FILTER_PRESETS = {
  "proven-winners": [{ metric: "wins", operator: "gt", value: 5 }],
  "five-round-winners": [{ metric: "five_round_wins", operator: "gte", value: 3 }],
  finishers: [{ metric: "finish_wins", operator: "gte", value: 3 }, { metric: "win_rate", operator: "gte", value: 60 }],
  "volume-strikers": [{ metric: "sig_landed_per_minute", operator: "gte", value: 4 }, { metric: "sig_accuracy", operator: "gte", value: 45 }],
  wrestlers: [{ metric: "total_takedowns", operator: "gte", value: 8 }, { metric: "takedown_accuracy", operator: "gte", value: 40 }],
};

const $ = (selector, parent = document) => parent.querySelector(selector);

function element(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined && text !== null) item.textContent = String(text);
  return item;
}

function actionButton(label, className, handler) {
  const button = element("button", className, label);
  button.type = "button";
  button.addEventListener("click", handler);
  return button;
}

function appendText(parent, tag, className, text) {
  const item = element(tag, className, text);
  parent.append(item);
  return item;
}

function normalize(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatNumber(value, digits = 2) {
  const number = finite(value);
  return number === null ? "—" : number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatPercent(value, digits = 1) {
  const number = finite(value);
  return number === null ? "—" : `${(number * 100).toFixed(digits)}%`;
}

function formatOdds(value) {
  const number = finite(value);
  if (number === null) return "—";
  return number > 0 ? `+${Math.round(number)}` : `${Math.round(number)}`;
}

function decimalOdds(value) {
  const odds = finite(value);
  if (odds === null || odds === 0 || Math.abs(odds) < 100) return null;
  return odds > 0 ? 1 + odds / 100 : 1 + 100 / Math.abs(odds);
}

function normalCdf(value) {
  const x = Number(value);
  if (!Number.isFinite(x)) return x > 0 ? 1 : 0;
  const sign = x < 0 ? -1 : 1;
  const absolute = Math.abs(x) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * absolute);
  const erf = sign * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-absolute * absolute));
  return (1 + erf) / 2;
}

function probabilityLogit(value) {
  const probability = finite(value);
  if (probability === null || probability <= 0 || probability >= 1) return null;
  return Math.log(probability / (1 - probability));
}

function formatDate(value, options = { year: "numeric", month: "short", day: "numeric" }) {
  if (!value) return "Unknown";
  const date = new Date(`${String(value).slice(0, 10)}T12:00:00Z`);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleDateString(undefined, { ...options, timeZone: "UTC" });
}

function dateKey(value) {
  if (value === null || value === undefined || value === "") return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return [
    date.getUTCFullYear(),
    String(date.getUTCMonth() + 1).padStart(2, "0"),
    String(date.getUTCDate()).padStart(2, "0"),
  ].join("-");
}

function formatTimestamp(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function formatDuration(seconds) {
  const number = finite(seconds);
  if (number === null) return "—";
  const minutes = Math.floor(number / 60);
  return `${minutes}:${String(Math.round(number - minutes * 60)).padStart(2, "0")}`;
}

function record(fighter) {
  if (!fighter) return "No recorded bouts";
  const results = fighter.record || fighter.career;
  let value = `${results.wins}-${results.losses}-${results.draws}`;
  if (results.no_contests) value += ` (${results.no_contests} NC)`;
  return value;
}

function fullRecord(fighter) {
  const results = fighter?.record || fighter?.career;
  if (!results) return { recorded_bouts: 0, promotions: [], metadata_only_bouts: 0 };
  if (results.promotions) return results;
  return { ...results, promotions: results.recorded_bouts ? [{ name: "UFC", bouts: results.recorded_bouts }] : [], metadata_only_bouts: 0, detailed_stat_bouts: results.recorded_bouts || 0 };
}

function promotionBouts(fighter, promotionPattern) {
  return fullRecord(fighter).promotions?.filter((item) => promotionPattern.test(item.name)).reduce((sum, item) => sum + item.bouts, 0) || 0;
}

function fighterDivision(fighter) {
  return fighter?.scheduled_division || fighter?.career.primary_division || "";
}

function ageOn(fighter, dateValue = null) {
  if (!fighter?.dob_iso) return null;
  const born = new Date(`${fighter.dob_iso}T12:00:00Z`);
  const onDate = dateValue ? new Date(dateValue) : new Date();
  if (Number.isNaN(born.getTime()) || Number.isNaN(onDate.getTime())) return null;
  let age = onDate.getUTCFullYear() - born.getUTCFullYear();
  const month = onDate.getUTCMonth() - born.getUTCMonth();
  if (month < 0 || (month === 0 && onDate.getUTCDate() < born.getUTCDate())) age -= 1;
  return age;
}

function daysSince(value) {
  if (!value) return null;
  const then = new Date(`${value}T12:00:00Z`);
  const asOf = state.card?.date ? new Date(state.card.date) : new Date();
  if (Number.isNaN(then.getTime()) || Number.isNaN(asOf.getTime())) return null;
  return Math.max(0, Math.round((asOf - then) / 86400000));
}

function decodeFight(values) {
  const fight = {};
  state.explorer.fight_columns.forEach((column, index) => { fight[column] = values[index]; });
  return fight;
}

async function ensureFighterFights(fighter) {
  if (Array.isArray(fighter.fights)) return fighter.fights;
  const shardKey = fighter.fight_shard;
  const metadata = state.explorer.fight_shards?.[shardKey];
  if (!metadata?.path) throw new Error(`No fight-log shard is published for ${fighter.name}`);
  let shardPromise = state.shardCache.get(shardKey);
  if (!shardPromise) {
    shardPromise = fetchJson(`src/content/data/external/${metadata.path}`).then((shard) => {
      if (shard.publication_sha256 !== metadata.publication_sha256) {
        throw new Error(`Fight-log shard ${shardKey} does not match the explorer index`);
      }
      Object.entries(shard.fighters || {}).forEach(([fighterId, fights]) => {
        const profile = state.fighterById.get(fighterId);
        if (profile) profile.fights = Array.isArray(fights) ? fights : [];
      });
      return shard;
    });
    state.shardCache.set(shardKey, shardPromise);
  }
  const shard = await shardPromise;
  fighter.fights = Array.isArray(shard.fighters?.[fighter.id]) ? shard.fighters[fighter.id] : [];
  return fighter.fights;
}

function graphFightId(fight, winnerId, loserId) {
  return [fight.source || fight.promotion || "fight", fight.fight_id || `${fight.date}-${winnerId}-${loserId}`].join(":");
}

async function ensureFightGraphData() {
  if (state.fightGraphPromise) return state.fightGraphPromise;
  state.fightGraphPromise = (async () => {
    const shardEntries = Object.entries(state.explorer.fight_shards || {});
    const shards = await Promise.all(shardEntries.map(async ([shardKey, metadata]) => {
      let shardPromise = state.shardCache.get(shardKey);
      if (!shardPromise) {
        shardPromise = fetchJson(`src/content/data/external/${metadata.path}`).then((shard) => {
          if (shard.publication_sha256 !== metadata.publication_sha256) throw new Error(`Fight-log shard ${shardKey} does not match the explorer index`);
          Object.entries(shard.fighters || {}).forEach(([fighterId, fights]) => {
            const profile = state.fighterById.get(fighterId);
            if (profile) profile.fights = Array.isArray(fights) ? fights : [];
          });
          return shard;
        });
        state.shardCache.set(shardKey, shardPromise);
      }
      return shardPromise;
    }));
    const seen = new Set();
    const edges = [];
    const fighterRows = new Map();
    shards.forEach((shard) => Object.entries(shard.fighters || {}).forEach(([fighterId, rows]) => {
      const fighter = state.fighterById.get(fighterId);
      if (!fighter || !Array.isArray(rows)) return;
      const decodedRows = rows.map(decodeFight);
      fighterRows.set(fighterId, decodedRows);
      decodedRows.forEach((fight) => {
        if (String(fight.result || "").toUpperCase() !== "W" || !fight.opponent_id) return;
        const loser = state.fighterById.get(String(fight.opponent_id));
        if (!loser) return;
        const id = graphFightId(fight, fighter.id, loser.id);
        if (seen.has(id)) return;
        seen.add(id);
        edges.push({ id, winnerId: fighter.id, winnerName: fighter.name, loserId: loser.id, loserName: loser.name, ...fight });
      });
    }));
    state.fightGraphFighterRows = fighterRows;
    state.fightGraphEdges = edges.sort((left, right) => String(right.date).localeCompare(String(left.date)) || left.id.localeCompare(right.id));
    return state.fightGraphEdges;
  })().catch((error) => { state.fightGraphPromise = null; throw error; });
  return state.fightGraphPromise;
}

function fightGraphEmpty(message) {
  const canvas = $("#fight-graph-canvas");
  canvas.replaceChildren(element("div", "empty-state", message));
  state.fightGraphViewport = null;
  updateFightGraphViewportControls();
}

function updateFightGraphViewportControls() {
  const viewport = state.fightGraphViewport;
  const enabled = Boolean(viewport?.svg?.isConnected);
  ["#graph-zoom-out", "#graph-zoom-in", "#graph-zoom-fit"].forEach((selector) => { $(selector).disabled = !enabled; });
  $("#graph-zoom-level").value = enabled ? `${Math.round(viewport.fullWidth / viewport.box.width * 100)}%` : "100%";
}

function applyFightGraphViewport() {
  const viewport = state.fightGraphViewport;
  if (!viewport?.svg?.isConnected) return;
  const box = viewport.box;
  if (box.width >= viewport.fullWidth) box.x = (viewport.fullWidth - box.width) / 2;
  else box.x = Math.max(-box.width * 0.06, Math.min(viewport.fullWidth - box.width * 0.94, box.x));
  if (box.height >= viewport.fullHeight) box.y = (viewport.fullHeight - box.height) / 2;
  else box.y = Math.max(-box.height * 0.06, Math.min(viewport.fullHeight - box.height * 0.94, box.y));
  viewport.svg.setAttribute("viewBox", `${box.x.toFixed(2)} ${box.y.toFixed(2)} ${box.width.toFixed(2)} ${box.height.toFixed(2)}`);
  updateFightGraphViewportControls();
}

function zoomFightGraph(factor, anchorX = 0.5, anchorY = 0.5) {
  const viewport = state.fightGraphViewport;
  if (!viewport) return;
  const current = viewport.box;
  const minimumWidth = viewport.fullWidth / 8;
  const maximumWidth = viewport.fullWidth * 2;
  const nextWidth = Math.max(minimumWidth, Math.min(maximumWidth, current.width * factor));
  const nextHeight = nextWidth * viewport.fullHeight / viewport.fullWidth;
  const focusX = current.x + current.width * anchorX;
  const focusY = current.y + current.height * anchorY;
  current.x = focusX - nextWidth * anchorX;
  current.y = focusY - nextHeight * anchorY;
  current.width = nextWidth;
  current.height = nextHeight;
  applyFightGraphViewport();
}

function panFightGraph(horizontal, vertical) {
  const viewport = state.fightGraphViewport;
  if (!viewport) return;
  viewport.box.x += horizontal;
  viewport.box.y += vertical;
  applyFightGraphViewport();
}

function fitFightGraph() {
  const viewport = state.fightGraphViewport;
  if (!viewport) return;
  viewport.box = { x: 0, y: 0, width: viewport.fullWidth, height: viewport.fullHeight };
  applyFightGraphViewport();
}

function configureFightGraphViewport(svg, width, height) {
  state.fightGraphViewport = { svg, fullWidth: width, fullHeight: height, box: { x: 0, y: 0, width, height } };
  svg.setAttribute("tabindex", "0");
  svg.setAttribute("aria-description", "Use the mouse wheel or zoom buttons to zoom. Drag the empty background or use arrow keys to pan. Press 0 to fit the graph.");
  let drag = null;
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const bounds = svg.getBoundingClientRect();
    const anchorX = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
    const anchorY = Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height));
    zoomFightGraph(event.deltaY < 0 ? 0.82 : 1.22, anchorX, anchorY);
  }, { passive: false });
  svg.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest?.(".fight-graph-edge-hit")) return;
    drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, boxX: state.fightGraphViewport.box.x, boxY: state.fightGraphViewport.box.y };
    try { svg.setPointerCapture?.(event.pointerId); } catch {}
    svg.classList.add("is-panning");
    event.preventDefault();
  });
  svg.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const bounds = svg.getBoundingClientRect();
    const viewport = state.fightGraphViewport;
    viewport.box.x = drag.boxX - (event.clientX - drag.x) / bounds.width * viewport.box.width;
    viewport.box.y = drag.boxY - (event.clientY - drag.y) / bounds.height * viewport.box.height;
    applyFightGraphViewport();
  });
  const endPan = (event) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    try { svg.releasePointerCapture?.(event.pointerId); } catch {}
    drag = null;
    svg.classList.remove("is-panning");
  };
  svg.addEventListener("pointerup", endPan);
  svg.addEventListener("pointercancel", endPan);
  svg.addEventListener("keydown", (event) => {
    if (event.target !== svg) return;
    const stepX = state.fightGraphViewport.box.width * 0.08;
    const stepY = state.fightGraphViewport.box.height * 0.08;
    const actions = {
      ArrowLeft: () => panFightGraph(-stepX, 0),
      ArrowRight: () => panFightGraph(stepX, 0),
      ArrowUp: () => panFightGraph(0, -stepY),
      ArrowDown: () => panFightGraph(0, stepY),
      "+": () => zoomFightGraph(0.8),
      "=": () => zoomFightGraph(0.8),
      "-": () => zoomFightGraph(1.25),
      "0": fitFightGraph,
    };
    if (!actions[event.key]) return;
    event.preventDefault();
    actions[event.key]();
  });
  applyFightGraphViewport();
}

function resetFightGraphDetails() {
  const details = $("#fight-graph-details");
  details.replaceChildren();
  appendText(details, "p", "eyebrow", "Fight details");
  appendText(details, "h3", "", "Select an arrow");
  appendText(details, "p", "", "Hover, focus, or click any arrow to inspect its event, date, weight class, method, round, and time.");
}

function fightGraphTableMetrics(rules = []) {
  const seen = new Set();
  return rules.map((rule) => graphMetricDefinition(rule.metric)).filter((metric) => {
    if (seen.has(metric.key)) return false;
    seen.add(metric.key); return true;
  });
}

function renderFightGraphEdgeHeadings(metrics = []) {
  const headings = $("#fight-graph-edge-headings");
  headings.replaceChildren();
  ["Date", "Fighters", "Winner / result", "Finish"].forEach((label) => appendText(headings, "th", "", label));
  metrics.forEach((metric) => {
    const heading = appendText(headings, "th", "fight-graph-edge-metric-heading", metric.label);
    heading.title = metric.description;
    appendText(heading, "small", "", "Winner / loser");
  });
  const action = element("th"); appendText(action, "span", "visually-hidden", "Actions"); headings.append(action);
}

function formatGraphTableMetric(metric, value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (["win_rate", "sig_accuracy", "takedown_accuracy", "stats_coverage", "avg_opponent_win_rate"].includes(metric.key)) return `${formatNumber(value, 1)}%`;
  if (metric.key.includes("minutes") || metric.key.startsWith("avg_") || metric.key.endsWith("per_minute")) return formatNumber(value, 1);
  return formatNumber(value, Number.isInteger(value) ? 0 : 2);
}

function resetFightGraphEdgeTable(message = "Draw a graph to list its fights.") {
  const rows = $("#fight-graph-edge-rows");
  renderFightGraphEdgeHeadings();
  rows.replaceChildren();
  const row = document.createElement("tr");
  const cell = appendText(row, "td", "fight-graph-edge-table-empty", message);
  cell.colSpan = $("#fight-graph-edge-headings").childElementCount;
  rows.append(row);
  $("#fight-graph-edge-count").textContent = "0 fights";
}

function pinFightGraphEdge(edge) {
  state.fightGraphPinnedId = edge.id;
  document.querySelectorAll(".fight-graph-edge.is-pinned").forEach((item) => item.classList.remove("is-pinned"));
  document.querySelectorAll(".fight-graph-edge").forEach((item) => {
    if (item.dataset.edgeId === edge.id) item.classList.add("is-pinned");
  });
  renderFightGraphDetails(edge);
}

function renderFightGraphEdgeTable(edges, rules = [], aggregates = new Map()) {
  const rows = $("#fight-graph-edge-rows");
  const metrics = fightGraphTableMetrics(rules);
  renderFightGraphEdgeHeadings(metrics);
  rows.replaceChildren();
  $("#fight-graph-edge-count").textContent = `${edges.length.toLocaleString()} fight${edges.length === 1 ? "" : "s"}`;
  edges.forEach((edge, index) => {
    const summaryRow = document.createElement("tr");
    summaryRow.className = "fight-graph-edge-summary-row";
    appendText(summaryRow, "td", "fight-graph-edge-date", formatDate(edge.date, { year: "numeric", month: "short", day: "numeric" }));
    const matchup = element("td", "fight-graph-edge-matchup");
    appendText(matchup, "strong", "", edge.winnerName);
    matchup.append(document.createTextNode(" vs "));
    appendText(matchup, "span", "", edge.loserName);
    summaryRow.append(matchup);
    const result = element("td", "fight-graph-edge-result");
    appendText(result, "span", "pill win", "W");
    appendText(result, "strong", "", edge.winnerName);
    summaryRow.append(result);
    appendText(summaryRow, "td", "fight-graph-edge-finish", `${edge.method || "Method unavailable"} · R${edge.round || "—"} ${edge.time || ""}`);
    metrics.forEach((metric) => {
      const winnerValue = formatGraphTableMetric(metric, aggregates.get(edge.winnerId)?.[metric.key]);
      const loserValue = formatGraphTableMetric(metric, aggregates.get(edge.loserId)?.[metric.key]);
      const cell = element("td", "fight-graph-edge-metric");
      appendText(cell, "strong", "", winnerValue); appendText(cell, "span", "", "/"); appendText(cell, "span", "", loserValue);
      cell.title = `${edge.winnerName}: ${winnerValue}; ${edge.loserName}: ${loserValue}`;
      cell.setAttribute("aria-label", `${metric.label}. ${edge.winnerName}: ${winnerValue}. ${edge.loserName}: ${loserValue}.`);
      summaryRow.append(cell);
    });
    const action = element("td", "fight-graph-edge-action");
    const detailId = `fight-graph-row-detail-${index}`;
    const button = element("button", "text-button small-button", "View stats");
    button.type = "button";
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-controls", detailId);
    action.append(button); summaryRow.append(action);

    const detailRow = document.createElement("tr");
    detailRow.className = "fight-graph-edge-detail-row";
    detailRow.hidden = true;
    const detailCell = element("td"); detailCell.colSpan = 5 + metrics.length;
    const body = element("div", "fight-graph-row-details", "Open to load complete fight statistics.");
    body.id = detailId; detailCell.append(body); detailRow.append(detailCell);
    let rendered = false;
    button.addEventListener("click", async () => {
      const opening = detailRow.hidden;
      detailRow.hidden = !opening;
      summaryRow.classList.toggle("is-open", opening);
      button.textContent = opening ? "Hide stats" : "View stats";
      button.setAttribute("aria-expanded", String(opening));
      if (!opening) return;
      pinFightGraphEdge(edge);
      if (rendered) return;
      rendered = true;
      body.textContent = edge.stats_available ? "Loading paired fight statistics…" : "Loading source details…";
      try {
        const opponentFight = await pairedFight(edge);
        renderFightDetails(body, edge, opponentFight, edge.winnerName);
      } catch (error) {
        console.error(error);
        body.textContent = "Detailed fight statistics could not be loaded.";
      }
    });
    rows.append(summaryRow, detailRow);
  });
}

function renderFightGraphDetails(edge) {
  const details = $("#fight-graph-details");
  details.replaceChildren();
  appendText(details, "p", "eyebrow", edge.promotion || edge.source_label || "Recorded fight");
  appendText(details, "h3", "", `${edge.winnerName} defeated ${edge.loserName}`);
  appendText(details, "p", "fight-graph-detail-result", `${edge.method || "Method unavailable"} · Round ${edge.round || "—"}, ${edge.time || "clock unavailable"}`);
  const list = element("dl", "fight-graph-detail-list");
  [
    ["Date", formatDate(edge.date)],
    ["Event", edge.event_name || (edge.event_id ? `${edge.promotion || edge.source_label || "Recorded"} event` : "Event unavailable")],
    ["Weight class", edge.division || "Unknown"],
    ["Promotion", edge.promotion || "Unknown"],
    ["Source", edge.source_label || edge.source || "Unknown"],
  ].forEach(([label, value]) => { appendText(list, "dt", "", label); appendText(list, "dd", "", value); });
  details.append(list);
  const links = element("div", "fight-graph-links");
  const sourceUrl = edge.fight_url || edge.source_url || edge.event_url;
  if (sourceUrl) {
    const link = element("a", "fight-graph-source", edge.stats_available ? "Open official fight page" : "Open source page");
    link.href = sourceUrl; link.target = "_blank"; link.rel = "noreferrer"; links.append(link);
  }
  if (edge.event_url && edge.event_url !== sourceUrl) {
    const eventLink = element("a", "fight-graph-source", "Open official event page");
    eventLink.href = edge.event_url; eventLink.target = "_blank"; eventLink.rel = "noreferrer"; links.append(eventLink);
  }
  if (links.childElementCount) details.append(links);
}

function graphPath(edge, nodesById, pairIndexes) {
  const source = nodesById.get(edge.winnerId); const target = nodesById.get(edge.loserId);
  const deltaX = target.x - source.x; const deltaY = target.y - source.y;
  const distance = Math.max(1, Math.hypot(deltaX, deltaY)); const unitX = deltaX / distance; const unitY = deltaY / distance;
  const startX = source.x + unitX * (source.radius + 3); const startY = source.y + unitY * (source.radius + 3);
  const endX = target.x - unitX * (target.radius + 9); const endY = target.y - unitY * (target.radius + 9);
  const pair = [edge.winnerId, edge.loserId].sort().join("|"); const siblings = pairIndexes.get(pair) || [];
  const index = siblings.indexOf(edge.id); const offset = (index - (siblings.length - 1) / 2) * 22;
  if (Math.abs(offset) < 1) return `M ${startX.toFixed(1)} ${startY.toFixed(1)} L ${endX.toFixed(1)} ${endY.toFixed(1)}`;
  const direction = edge.winnerId < edge.loserId ? 1 : -1;
  const controlX = (startX + endX) / 2 - unitY * offset * direction; const controlY = (startY + endY) / 2 + unitX * offset * direction;
  return `M ${startX.toFixed(1)} ${startY.toFixed(1)} Q ${controlX.toFixed(1)} ${controlY.toFixed(1)} ${endX.toFixed(1)} ${endY.toFixed(1)}`;
}

function layoutFightGraph(nodeList, edges, width, height) {
  const goldenAngle = Math.PI * (3 - Math.sqrt(5)); const centerX = width / 2; const centerY = height / 2;
  nodeList.forEach((node, index) => {
    const radius = Math.sqrt((index + 0.5) / nodeList.length) * Math.min(width, height) * 0.42; const angle = index * goldenAngle;
    node.x = centerX + Math.cos(angle) * radius; node.y = centerY + Math.sin(angle) * radius; node.vx = 0; node.vy = 0;
  });
  const nodesById = new Map(nodeList.map((node) => [node.id, node]));
  for (let iteration = 0; iteration < 110; iteration += 1) {
    const cooling = 1 - iteration / 130;
    for (let leftIndex = 0; leftIndex < nodeList.length; leftIndex += 1) {
      const left = nodeList[leftIndex];
      for (let rightIndex = leftIndex + 1; rightIndex < nodeList.length; rightIndex += 1) {
        const right = nodeList[rightIndex]; let deltaX = right.x - left.x; let deltaY = right.y - left.y; let distanceSquared = deltaX * deltaX + deltaY * deltaY;
        if (distanceSquared < 1) { deltaX = 1; deltaY = 0; distanceSquared = 1; }
        const distance = Math.sqrt(distanceSquared); const force = Math.min(7, 4200 / distanceSquared) * cooling;
        const forceX = deltaX / distance * force; const forceY = deltaY / distance * force;
        left.vx -= forceX; left.vy -= forceY; right.vx += forceX; right.vy += forceY;
      }
    }
    edges.forEach((edge) => {
      const source = nodesById.get(edge.winnerId); const target = nodesById.get(edge.loserId); if (!source || !target) return;
      const deltaX = target.x - source.x; const deltaY = target.y - source.y; const distance = Math.max(1, Math.hypot(deltaX, deltaY));
      const force = (distance - 125) * 0.014 * cooling; const forceX = deltaX / distance * force; const forceY = deltaY / distance * force;
      source.vx += forceX; source.vy += forceY; target.vx -= forceX; target.vy -= forceY;
    });
    nodeList.forEach((node) => {
      node.vx += (centerX - node.x) * 0.0025; node.vy += (centerY - node.y) * 0.0025; node.vx *= 0.72; node.vy *= 0.72;
      node.x = Math.max(46, Math.min(width - 46, node.x + node.vx)); node.y = Math.max(46, Math.min(height - 58, node.y + node.vy));
    });
  }
  return nodesById;
}

function layoutMatchupFightGraph(nodeList, fighterDepths) {
  const layers = new Map();
  nodeList.forEach((node) => {
    node.depth = fighterDepths.get(node.id) ?? 0;
    if (!layers.has(node.depth)) layers.set(node.depth, []);
    layers.get(node.depth).push(node);
  });
  layers.forEach((nodes) => nodes.sort((left, right) => right.appearances - left.appearances || left.name.localeCompare(right.name)));
  let outerRadius = 270;
  layers.forEach((nodes, depth) => {
    if (depth > 0) outerRadius = Math.max(outerRadius, depth * 250, nodes.length * 42 / (Math.PI * 2));
  });
  const width = Math.max(1120, Math.ceil(outerRadius * 2 + 260));
  const height = Math.max(680, width);
  const centerX = width / 2; const centerY = height / 2;
  (layers.get(0) || []).forEach((node, index, seeds) => {
    node.x = centerX + (index - (seeds.length - 1) / 2) * 145;
    node.y = centerY;
  });
  const maximumDepth = Math.max(...layers.keys());
  for (let depth = 1; depth <= maximumDepth; depth += 1) {
    const nodes = layers.get(depth) || [];
    if (!nodes.length) continue;
    const radius = Math.max(depth * 250, nodes.length * 42 / (Math.PI * 2));
    const offset = depth % 2 ? -Math.PI / 2 : -Math.PI / 2 + Math.PI / Math.max(nodes.length, 2);
    nodes.forEach((node, index) => {
      const angle = offset + index * Math.PI * 2 / nodes.length;
      node.x = centerX + Math.cos(angle) * radius;
      node.y = centerY + Math.sin(angle) * radius;
    });
  }
  return { nodesById: new Map(nodeList.map((node) => [node.id, node])), width, height };
}

function renderFightGraph(edges, counts, { fighterDepths = null, seedIds = new Set() } = {}) {
  const canvas = $("#fight-graph-canvas"); canvas.replaceChildren();
  const nodeIds = [...new Set([...seedIds, ...edges.flatMap((edge) => [edge.winnerId, edge.loserId])])]; const wins = new Map();
  edges.forEach((edge) => wins.set(edge.winnerId, (wins.get(edge.winnerId) || 0) + 1));
  const nodeList = nodeIds.map((id) => {
    const fighter = state.fighterById.get(id); const appearances = counts.get(id) || 0;
    return { id, name: fighter?.name || id, appearances, wins: wins.get(id) || 0, radius: Math.min(18, 8 + Math.sqrt(Math.max(appearances, 1)) * 2.1) };
  }).sort((left, right) => right.appearances - left.appearances || left.name.localeCompare(right.name));
  let width = 1120; let height = Math.max(640, Math.min(940, 560 + nodeList.length * 2.6)); let nodesById;
  if (fighterDepths) {
    const layout = layoutMatchupFightGraph(nodeList, fighterDepths); width = layout.width; height = layout.height; nodesById = layout.nodesById;
  } else nodesById = layoutFightGraph(nodeList, edges, width, height);
  const pairIndexes = new Map();
  edges.forEach((edge) => { const pair = [edge.winnerId, edge.loserId].sort().join("|"); if (!pairIndexes.has(pair)) pairIndexes.set(pair, []); pairIndexes.get(pair).push(edge.id); });
  const svg = document.createElementNS(SVG_NAMESPACE, "svg"); svg.classList.add("fight-graph-svg"); svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img"); svg.setAttribute("aria-label", `Directed graph of ${nodeList.length} fighters and ${edges.length} decisive fights`);
  const defs = document.createElementNS(SVG_NAMESPACE, "defs"); const marker = document.createElementNS(SVG_NAMESPACE, "marker");
  marker.setAttribute("id", "fight-graph-arrow"); marker.setAttribute("viewBox", "0 0 10 10"); marker.setAttribute("refX", "8"); marker.setAttribute("refY", "5"); marker.setAttribute("markerWidth", "7"); marker.setAttribute("markerHeight", "7"); marker.setAttribute("orient", "auto-start-reverse");
  const arrow = document.createElementNS(SVG_NAMESPACE, "path"); arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z"); marker.append(arrow); defs.append(marker); svg.append(defs);
  const edgeLayer = document.createElementNS(SVG_NAMESPACE, "g"); edgeLayer.classList.add("fight-graph-edges");
  edges.forEach((edge) => {
    const group = document.createElementNS(SVG_NAMESPACE, "g"); group.classList.add("fight-graph-edge"); group.dataset.edgeId = edge.id;
    const pathValue = graphPath(edge, nodesById, pairIndexes); const visible = document.createElementNS(SVG_NAMESPACE, "path");
    visible.classList.add("fight-graph-edge-line"); visible.setAttribute("d", pathValue); visible.setAttribute("marker-end", "url(#fight-graph-arrow)");
    const hit = document.createElementNS(SVG_NAMESPACE, "path"); hit.classList.add("fight-graph-edge-hit"); hit.setAttribute("d", pathValue); hit.setAttribute("tabindex", "0"); hit.setAttribute("role", "button");
    hit.setAttribute("aria-label", `${edge.winnerName} defeated ${edge.loserName} on ${formatDate(edge.date)} by ${edge.method || "unknown method"}`);
    const preview = () => { document.querySelectorAll(".fight-graph-edge.is-preview").forEach((item) => item.classList.remove("is-preview")); group.classList.add("is-preview"); renderFightGraphDetails(edge); };
    const restore = () => { group.classList.remove("is-preview"); const pinned = edges.find((item) => item.id === state.fightGraphPinnedId); if (pinned) renderFightGraphDetails(pinned); else resetFightGraphDetails(); };
    hit.addEventListener("mouseenter", preview); hit.addEventListener("focus", preview); hit.addEventListener("mouseleave", restore); hit.addEventListener("blur", restore);
    hit.addEventListener("click", () => { state.fightGraphPinnedId = state.fightGraphPinnedId === edge.id ? null : edge.id; document.querySelectorAll(".fight-graph-edge.is-pinned").forEach((item) => item.classList.remove("is-pinned")); if (state.fightGraphPinnedId) group.classList.add("is-pinned"); renderFightGraphDetails(edge); });
    group.append(visible, hit); edgeLayer.append(group);
  });
  svg.append(edgeLayer); const nodeLayer = document.createElementNS(SVG_NAMESPACE, "g"); nodeLayer.classList.add("fight-graph-nodes");
  nodeList.forEach((node) => {
    const group = document.createElementNS(SVG_NAMESPACE, "g"); group.classList.add("fight-graph-node"); if (seedIds.has(node.id)) group.classList.add("is-seed"); group.dataset.depth = String(node.depth ?? ""); group.setAttribute("transform", `translate(${node.x.toFixed(1)} ${node.y.toFixed(1)})`);
    const title = document.createElementNS(SVG_NAMESPACE, "title"); title.textContent = `${node.name}${seedIds.has(node.id) ? " (seed fighter)" : node.depth ? ` (depth ${node.depth})` : ""}: ${node.wins} wins shown in ${node.appearances} fights`;
    const circle = document.createElementNS(SVG_NAMESPACE, "circle"); circle.setAttribute("r", (node.radius + (seedIds.has(node.id) ? 4 : 0)).toFixed(1)); const label = document.createElementNS(SVG_NAMESPACE, "text");
    label.setAttribute("y", String(node.radius + 14)); label.textContent = node.name.length > 22 ? `${node.name.slice(0, 20)}…` : node.name; group.append(title, circle, label); nodeLayer.append(group);
  });
  svg.append(nodeLayer); canvas.append(svg); configureFightGraphViewport(svg, width, height);
}

function graphMetricDefinition(key) {
  return GRAPH_RULE_METRICS.find((metric) => metric.key === key) || GRAPH_RULE_METRICS[0];
}

function markGraphFiltersDirty() {
  const status = $("#fight-graph-status");
  if (status) status.textContent = "Filters changed — select Draw graph to apply them.";
}

function selectGraphQuickRange(value) {
  document.querySelectorAll("[data-graph-years]").forEach((button) => button.classList.toggle("is-active", button.dataset.graphYears === String(value)));
  $("#graph-apply-custom-years").classList.toggle("is-active", value === "custom");
}

function applyGraphQuickRange(value, custom = false) {
  const start = $("#graph-start-date");
  const end = $("#graph-end-date");
  if (value === "all") {
    start.value = ""; end.value = ""; selectGraphQuickRange("all"); markGraphFiltersDirty(); return;
  }
  const years = Math.round(Number(value));
  if (!Number.isFinite(years) || years < 1 || years > 100) {
    $("#fight-graph-status").textContent = "Choose a number of years from 1 to 100.";
    $("#graph-custom-years").focus(); return;
  }
  const latest = end.max || state.explorer?.data_through || dateKey(new Date());
  const startDate = new Date(`${latest}T12:00:00Z`);
  startDate.setUTCFullYear(startDate.getUTCFullYear() - years);
  start.value = dateKey(startDate); end.value = latest;
  selectGraphQuickRange(custom ? "custom" : String(years));
  markGraphFiltersDirty();
}

function updateGraphRuleRow(row) {
  const metric = graphMetricDefinition($(".graph-rule-metric", row).value);
  const operator = $(".graph-rule-operator", row).value;
  const firstValue = $(".graph-rule-value", row);
  const secondValue = $(".graph-rule-value-max", row);
  firstValue.step = metric.step || "1";
  secondValue.step = metric.step || "1";
  secondValue.hidden = operator !== "between";
  $(".graph-rule-and", row).hidden = operator !== "between";
  $(".graph-rule-description", row).textContent = metric.description;
}

function addGraphRule(values = {}) {
  const list = $("#graph-rule-list");
  const row = element("div", "graph-rule-row");
  row.dataset.ruleId = String(++state.fightGraphRuleSequence);
  const metricSelect = element("select", "graph-rule-metric");
  metricSelect.setAttribute("aria-label", "Fighter metric");
  const groups = new Map();
  GRAPH_RULE_METRICS.forEach((metric) => {
    if (!groups.has(metric.group)) {
      const group = document.createElement("optgroup"); group.label = metric.group; metricSelect.append(group); groups.set(metric.group, group);
    }
    const option = element("option", "", metric.label); option.value = metric.key; groups.get(metric.group).append(option);
  });
  metricSelect.value = values.metric || "wins";
  const operatorSelect = element("select", "graph-rule-operator");
  operatorSelect.setAttribute("aria-label", "Comparison");
  GRAPH_RULE_OPERATORS.forEach(([value, label]) => { const option = element("option", "", label); option.value = value; operatorSelect.append(option); });
  operatorSelect.value = values.operator || "gte";
  const firstValue = element("input", "graph-rule-value"); firstValue.type = "number"; firstValue.value = values.value ?? 1; firstValue.setAttribute("aria-label", "Rule value");
  const and = element("span", "graph-rule-and", "and");
  const secondValue = element("input", "graph-rule-value-max"); secondValue.type = "number"; secondValue.value = values.max ?? values.value ?? 1; secondValue.setAttribute("aria-label", "Maximum rule value");
  const remove = actionButton("Remove", "text-button small-button graph-remove-rule", () => { row.remove(); markGraphFiltersDirty(); }); remove.setAttribute("aria-label", "Remove condition");
  const description = element("small", "graph-rule-description");
  row.append(metricSelect, operatorSelect, firstValue, and, secondValue, remove, description);
  [metricSelect, operatorSelect].forEach((input) => input.addEventListener("change", () => { updateGraphRuleRow(row); markGraphFiltersDirty(); }));
  [firstValue, secondValue].forEach((input) => input.addEventListener("input", markGraphFiltersDirty));
  list.append(row); updateGraphRuleRow(row);
  return row;
}

function clearGraphRules() {
  $("#graph-rule-list").replaceChildren();
  markGraphFiltersDirty();
}

function applyGraphFilterPreset(name) {
  clearGraphRules();
  (GRAPH_FILTER_PRESETS[name] || []).forEach(addGraphRule);
}

function setGraphFilterMode(mode) {
  const advanced = mode === "advanced";
  const matchup = mode === "matchup";
  state.fightGraphFilterMode = matchup ? "matchup" : advanced ? "advanced" : "simple";
  $("#graph-advanced-filters").hidden = !advanced;
  $("#graph-matchup-filters").hidden = !matchup;
  $(".fight-graph-controls").classList.toggle("is-matchup-mode", matchup);
  $("#graph-filter-title").textContent = matchup ? "Choose two fighters and a depth" : "Choose who appears";
  [["#graph-mode-simple", !advanced && !matchup], ["#graph-mode-advanced", advanced], ["#graph-mode-matchup", matchup]].forEach(([selector, active]) => {
    const button = $(selector); button.classList.toggle("is-active", active); button.setAttribute("aria-pressed", String(active));
  });
  if (matchup && !$("#graph-division").value) $("#graph-division").value = "*";
  markGraphFiltersDirty();
}

function readGraphRules() {
  if (state.fightGraphFilterMode !== "advanced") return [];
  return [...document.querySelectorAll(".graph-rule-row")].map((row) => {
    const value = finite($(".graph-rule-value", row).value);
    const maximum = finite($(".graph-rule-value-max", row).value);
    return { metric: $(".graph-rule-metric", row).value, operator: $(".graph-rule-operator", row).value, value, maximum };
  }).filter((rule) => rule.value !== null && (rule.operator !== "between" || rule.maximum !== null));
}

function graphWindowFilters() {
  return {
    division: $("#graph-division").value,
    promotion: $("#graph-promotion").value,
    startDate: $("#graph-start-date").value,
    endDate: $("#graph-end-date").value,
  };
}

function fightMatchesGraphWindow(fight, filters) {
  return !(filters.division && filters.division !== "*" && fight.division !== filters.division)
    && !(filters.promotion && fight.promotion !== filters.promotion)
    && !(filters.startDate && fight.date < filters.startDate)
    && !(filters.endDate && fight.date > filters.endDate);
}

function isFiveRoundFight(fight) {
  const format = normalize(fight.time_format);
  const modernUfcHeadliner = fight.promotion === "UFC" && Number(fight.source_card_index) === 0 && String(fight.date || "") >= "2011-11-05";
  return /(^| )5 (rnd|scheduled rounds?)( |$)/.test(format) || Number(fight.round) > 3 || modernUfcHeadliner;
}

function sumFightMetric(rows, key) {
  return rows.reduce((total, fight) => total + (finite(fight[key]) ?? 0), 0);
}

function aggregateGraphFighter(fighterId, rows, filters) {
  const fighter = state.fighterById.get(fighterId);
  const ordered = rows.filter((fight) => fightMatchesGraphWindow(fight, filters)).sort((left, right) => String(right.date).localeCompare(String(left.date)));
  const wins = ordered.filter((fight) => String(fight.result).toUpperCase() === "W");
  const losses = ordered.filter((fight) => String(fight.result).toUpperCase() === "L");
  const detailed = ordered.filter((fight) => fight.stats_available);
  const knownDuration = ordered.filter((fight) => finite(fight.total_fight_time) !== null);
  const knownWins = wins.map((fight) => finite(fight.total_fight_time)).filter((value) => value !== null);
  let winStreak = 0;
  for (const fight of ordered) { if (String(fight.result).toUpperCase() !== "W") break; winStreak += 1; }
  const sigAttempts = sumFightMetric(detailed, "sig_strikes_attempts");
  const sigLanded = sumFightMetric(detailed, "sig_strikes_landed");
  const takedownAttempts = sumFightMetric(detailed, "takedowns_attempts");
  const takedowns = sumFightMetric(detailed, "takedowns_landed");
  const totalSeconds = sumFightMetric(knownDuration, "total_fight_time");
  const finishWins = wins.filter((fight) => !/DEC/i.test(String(fight.method || "")));
  const opponents = ordered.map((fight) => state.fighterById.get(String(fight.opponent_id))).filter(Boolean);
  const opponentWins = opponents.map((opponent) => finite(opponent.career?.wins)).filter((value) => value !== null);
  const opponentWinRates = opponents.map((opponent) => finite(opponent.career?.win_rate)).filter((value) => value !== null);
  const results = wins.length + losses.length;
  const asOf = filters.endDate || state.explorer.data_through;
  return {
    fights: ordered.length,
    wins: wins.length,
    losses: losses.length,
    draws_nc: ordered.length - results,
    win_rate: results ? wins.length / results * 100 : null,
    win_streak: winStreak,
    unique_opponents: new Set(ordered.map((fight) => fight.opponent_id).filter(Boolean)).size,
    finish_wins: finishWins.length,
    ko_wins: wins.filter((fight) => /KO|TKO/i.test(String(fight.method || ""))).length,
    submission_wins: wins.filter((fight) => /SUB/i.test(String(fight.method || ""))).length,
    decision_wins: wins.filter((fight) => /DEC/i.test(String(fight.method || ""))).length,
    five_round_wins: wins.filter(isFiveRoundFight).length,
    total_knockdowns: sumFightMetric(detailed, "knockdowns"),
    avg_sig_landed: detailed.length ? sigLanded / detailed.length : null,
    sig_landed_per_minute: totalSeconds ? sigLanded / (totalSeconds / 60) : null,
    sig_accuracy: sigAttempts ? sigLanded / sigAttempts * 100 : null,
    head_landed: sumFightMetric(detailed, "head_strikes_landed"),
    body_landed: sumFightMetric(detailed, "body_strikes_landed"),
    leg_landed: sumFightMetric(detailed, "leg_strikes_landed"),
    total_takedowns: takedowns,
    takedown_accuracy: takedownAttempts ? takedowns / takedownAttempts * 100 : null,
    submission_attempts: sumFightMetric(detailed, "sub_attempts"),
    control_minutes: sumFightMetric(detailed, "control") / 60,
    avg_fight_minutes: knownDuration.length ? totalSeconds / knownDuration.length / 60 : null,
    total_fight_minutes: totalSeconds / 60,
    fastest_win_minutes: knownWins.length ? Math.min(...knownWins) / 60 : null,
    stats_coverage: ordered.length ? detailed.length / ordered.length * 100 : null,
    avg_opponent_wins: opponentWins.length ? opponentWins.reduce((sum, value) => sum + value, 0) / opponentWins.length : null,
    avg_opponent_win_rate: opponentWinRates.length ? opponentWinRates.reduce((sum, value) => sum + value, 0) / opponentWinRates.length * 100 : null,
    age: ageOn(fighter, asOf),
    height: finite(fighter?.height_inches),
    reach: finite(fighter?.reach_inches),
    career_wins: finite(fighter?.career?.wins),
    career_fights: finite(fighter?.career?.recorded_bouts),
  };
}

function graphRuleMatches(value, rule) {
  if (value === null || value === undefined || !Number.isFinite(value)) return false;
  if (rule.operator === "gt") return value > rule.value;
  if (rule.operator === "gte") return value >= rule.value;
  if (rule.operator === "eq") return Math.abs(value - rule.value) < 0.000001;
  if (rule.operator === "lte") return value <= rule.value;
  if (rule.operator === "lt") return value < rule.value;
  if (rule.operator === "between") return value >= Math.min(rule.value, rule.maximum) && value <= Math.max(rule.value, rule.maximum);
  return false;
}

function fightMatchesAdvancedConstraints(edge) {
  const method = $("#graph-fight-method").value;
  const round = $("#graph-fight-round").value;
  const detail = $("#graph-fight-detail").value;
  const methodText = String(edge.method || "");
  if (method === "ko" && !/KO|TKO/i.test(methodText)) return false;
  if (method === "submission" && !/SUB/i.test(methodText)) return false;
  if (method === "decision" && !/DEC/i.test(methodText)) return false;
  if (method === "other" && /KO|TKO|SUB|DEC/i.test(methodText)) return false;
  if (round && (round === "5" ? Number(edge.round) < 5 : Number(edge.round) !== Number(round))) return false;
  if (detail === "detailed" && !edge.stats_available) return false;
  if (detail === "metadata" && edge.stats_available) return false;
  return true;
}

function filteredFightGraph() {
  const filters = graphWindowFilters();
  const queryTerms = normalize($("#graph-fighter-search").value).split(" ").filter(Boolean); const minimum = Number($("#graph-min-fights").value) || 1;
  const baseEdges = state.fightGraphEdges.filter((edge) => fightMatchesGraphWindow(edge, filters));
  const baseCounts = new Map(); baseEdges.forEach((edge) => { baseCounts.set(edge.winnerId, (baseCounts.get(edge.winnerId) || 0) + 1); baseCounts.set(edge.loserId, (baseCounts.get(edge.loserId) || 0) + 1); });
  const eligible = new Set([...baseCounts].filter(([, count]) => count >= minimum).map(([id]) => id));
  const rules = readGraphRules();
  const join = $("#graph-rule-join").value;
  const scope = $("#graph-rule-scope").value;
  const stance = $("#graph-rule-stance").value;
  const matching = new Set();
  const aggregates = new Map();
  eligible.forEach((fighterId) => {
    const fighter = state.fighterById.get(fighterId);
    const values = rules.length ? aggregateGraphFighter(fighterId, state.fightGraphFighterRows.get(fighterId) || [], filters) : null;
    if (values) aggregates.set(fighterId, values);
    if (stance && fighter?.stance !== stance) return;
    if (rules.length) {
      const results = rules.map((rule) => graphRuleMatches(values[rule.metric], rule));
      if ((join === "any" && results.some(Boolean)) || (join !== "any" && results.every(Boolean))) matching.add(fighterId);
    } else matching.add(fighterId);
  });
  const advancedMode = state.fightGraphFilterMode === "advanced";
  const advancedFighterFilter = advancedMode && (rules.length || stance);
  const constraintValues = [$("#graph-fight-method").value, $("#graph-fight-round").value, $("#graph-fight-detail").value];
  const advancedFilterCount = rules.length + (stance ? 1 : 0) + constraintValues.filter(Boolean).length;
  const edges = baseEdges.filter((edge) => {
    if (!eligible.has(edge.winnerId) || !eligible.has(edge.loserId)) return false;
    if (queryTerms.length && !queryTerms.every((term) => normalize(`${edge.winnerName} ${edge.loserName}`).includes(term))) return false;
    if (advancedMode && !fightMatchesAdvancedConstraints(edge)) return false;
    if (!advancedFighterFilter) return true;
    return scope === "either" ? matching.has(edge.winnerId) || matching.has(edge.loserId) : matching.has(edge.winnerId) && matching.has(edge.loserId);
  });
  const counts = new Map(); edges.forEach((edge) => { counts.set(edge.winnerId, (counts.get(edge.winnerId) || 0) + 1); counts.set(edge.loserId, (counts.get(edge.loserId) || 0) + 1); });
  return { edges, counts, matchingFighterCount: matching.size, rules, aggregates, advancedFilterCount };
}

function filteredMatchupFightGraph() {
  const fighterAId = $("#graph-matchup-fighter-a").value;
  const fighterBId = $("#graph-matchup-fighter-b").value;
  const depth = Math.max(1, Math.min(2, Number($("#graph-matchup-depth").value) || 1));
  const filters = graphWindowFilters();
  const availableEdges = state.fightGraphEdges.filter((edge) => fightMatchesGraphWindow(edge, filters));
  const adjacency = new Map();
  availableEdges.forEach((edge) => {
    [edge.winnerId, edge.loserId].forEach((fighterId) => {
      if (!adjacency.has(fighterId)) adjacency.set(fighterId, []);
      adjacency.get(fighterId).push(edge);
    });
  });
  const fighterDepths = new Map([[fighterAId, 0], [fighterBId, 0]]);
  const selectedEdgeIds = new Set();
  let frontier = new Set([fighterAId, fighterBId]);
  for (let level = 0; level < depth && frontier.size; level += 1) {
    const next = new Set();
    frontier.forEach((fighterId) => (adjacency.get(fighterId) || []).forEach((edge) => {
      selectedEdgeIds.add(edge.id);
      [edge.winnerId, edge.loserId].forEach((connectedId) => {
        if (!fighterDepths.has(connectedId)) {
          fighterDepths.set(connectedId, level + 1);
          next.add(connectedId);
        }
      });
    }));
    frontier = next;
  }
  const edges = availableEdges.filter((edge) => selectedEdgeIds.has(edge.id));
  const counts = new Map();
  edges.forEach((edge) => {
    counts.set(edge.winnerId, (counts.get(edge.winnerId) || 0) + 1);
    counts.set(edge.loserId, (counts.get(edge.loserId) || 0) + 1);
  });
  return { edges, counts, fighterDepths, depth, seedIds: new Set([fighterAId, fighterBId]) };
}

async function drawFightGraph() {
  const status = $("#fight-graph-status"); const button = $("#graph-apply"); const division = $("#graph-division").value;
  const startDate = $("#graph-start-date").value; const endDate = $("#graph-end-date").value;
  const matchupMode = state.fightGraphFilterMode === "matchup";
  const fighterAId = $("#graph-matchup-fighter-a").value; const fighterBId = $("#graph-matchup-fighter-b").value;
  if (matchupMode && (!fighterAId || !fighterBId)) { status.textContent = "Choose two fighters before drawing the graph."; $(fighterAId ? "#graph-matchup-fighter-b" : "#graph-matchup-fighter-a").focus(); return; }
  if (matchupMode && fighterAId === fighterBId) { status.textContent = "Choose two different fighters."; $("#graph-matchup-fighter-b").focus(); return; }
  if (!matchupMode && !division) { status.textContent = "Choose a weight class before drawing the graph."; $("#graph-division").focus(); return; }
  if (startDate && endDate && startDate > endDate) { status.textContent = "Start date must be on or before the end date."; return; }
  button.disabled = true; status.textContent = state.fightGraphEdges.length ? "Filtering recorded fights…" : "Loading historical fight data…"; fightGraphEmpty("Loading the fight network…"); resetFightGraphEdgeTable("Loading matching fights…");
  try {
    await ensureFightGraphData();
    const result = matchupMode ? filteredMatchupFightGraph() : filteredFightGraph();
    const { edges, counts, matchingFighterCount = 0, rules = [], aggregates = new Map(), advancedFilterCount = 0, fighterDepths = null, depth = null, seedIds = new Set() } = result;
    const fighterCount = new Set([...seedIds, ...edges.flatMap((edge) => [edge.winnerId, edge.loserId])]).size;
    state.fightGraphPinnedId = null; resetFightGraphDetails();
    if (!edges.length) { fightGraphEmpty(matchupMode ? "No decisive fights involving this pair were found in the selected weight, promotion, and date window." : "No decisive fights connect fighters matching these filters. Try a wider date range, a lower minimum, fewer advanced rules, or include matching fighters' opponents."); resetFightGraphEdgeTable("No decisive fights match these filters."); status.textContent = advancedFilterCount ? `${matchingFighterCount.toLocaleString()} fighters satisfy the fighter rules, but no connections match every advanced constraint.` : "No matching decisive fights."; return; }
    renderFightGraphEdgeTable(edges, rules, aggregates);
    if (!matchupMode && fighterCount > FIGHT_GRAPH_MAX_FIGHTERS) { fightGraphEmpty(`${fighterCount.toLocaleString()} fighters match. Narrow the dates or fighter name, or increase the minimum fights to draw a readable graph.`); status.textContent = `${edges.length.toLocaleString()} fights connect ${fighterCount.toLocaleString()} fighters; the drawing limit is ${FIGHT_GRAPH_MAX_FIGHTERS}.`; return; }
    renderFightGraph(edges, counts, { fighterDepths, seedIds });
    status.textContent = matchupMode
      ? `Depth ${depth} includes ${edges.length.toLocaleString()} decisive fight${edges.length === 1 ? "" : "s"} connecting ${fighterCount.toLocaleString()} fighter${fighterCount === 1 ? "" : "s"} from ${state.fighterById.get(fighterAId)?.name} and ${state.fighterById.get(fighterBId)?.name}. Arrows point from winner to loser.`
      : `${edges.length.toLocaleString()} decisive fight${edges.length === 1 ? "" : "s"} connect ${fighterCount.toLocaleString()} fighter${fighterCount === 1 ? "" : "s"}.${advancedFilterCount ? ` ${matchingFighterCount.toLocaleString()} fighters satisfy the fighter rules; ${advancedFilterCount} advanced filter${advancedFilterCount === 1 ? "" : "s"} active.` : ""} Arrows point from winner to loser.`;
  } catch (error) { console.error(error); fightGraphEmpty("The historical fight data could not be loaded. Try again."); status.textContent = error.message; }
  finally { button.disabled = false; }
}

async function prepareFightGraph() {
  if (state.fightGraphEdges.length || $("#graph-apply").disabled) return;
  const button = $("#graph-apply"); const status = $("#fight-graph-status"); button.disabled = true; status.textContent = "Loading historical fight filters…";
  try {
    const edges = await ensureFightGraphData(); const promotions = [...new Set(edges.map((edge) => edge.promotion).filter(Boolean))].sort(); const promotionSelect = $("#graph-promotion");
    if (promotionSelect.options.length === 1) promotions.forEach((value) => { const option = element("option", "", value); option.value = value; promotionSelect.append(option); });
    const dates = edges.map((edge) => edge.date).filter(Boolean).sort(); [$("#graph-start-date"), $("#graph-end-date")].forEach((input) => { input.min = dates[0] || ""; input.max = dates[dates.length - 1] || state.explorer.data_through; });
    status.textContent = state.fightGraphFilterMode === "matchup" ? "Choose two fighters and a depth, then draw the graph." : "Choose a weight class, then draw the graph.";
  } catch (error) { console.error(error); status.textContent = `Fight graph unavailable: ${error.message}`; }
  finally { button.disabled = false; }
}

function resetFightGraph() {
  $("#graph-division").value = ""; $("#graph-promotion").value = ""; $("#graph-start-date").value = ""; $("#graph-end-date").value = ""; $("#graph-fighter-search").value = ""; $("#graph-min-fights").value = "1";
  $("#graph-matchup-fighter-a").value = ""; $("#graph-matchup-fighter-b").value = ""; $("#graph-matchup-depth").value = "1";
  selectGraphQuickRange("all");
  $("#graph-rule-join").value = "all"; $("#graph-rule-scope").value = "both"; $("#graph-rule-stance").value = "";
  $("#graph-fight-method").value = ""; $("#graph-fight-round").value = ""; $("#graph-fight-detail").value = ""; clearGraphRules(); setGraphFilterMode("simple");
  state.fightGraphPinnedId = null; fightGraphEmpty("Choose filters to draw the fight network."); resetFightGraphDetails(); resetFightGraphEdgeTable(); $("#fight-graph-status").textContent = "Choose a weight class, then draw the graph.";
}

async function fetchJson(path, required = true) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return await response.json();
  } catch (error) {
    if (required) throw new Error(`Could not load ${path}: ${error.message}`);
    return null;
  }
}

async function loadData() {
  const [explorer, vegas, card, model, bayesian, market, performance, outcomes] = await Promise.all([
    fetchJson(DATA_PATHS.explorer),
    fetchJson(DATA_PATHS.vegas, false),
    fetchJson(DATA_PATHS.card, false),
    fetchJson(DATA_PATHS.model, false),
    fetchJson(DATA_PATHS.bayesian, false),
    fetchJson(DATA_PATHS.market, false),
    fetchJson(DATA_PATHS.performance, false),
    fetchJson(DATA_PATHS.outcomes, false),
  ]);
  state.explorer = explorer;
  state.vegas = vegas;
  state.card = card;
  state.model = model;
  state.bayesian = bayesian;
  state.market = market;
  state.performance = performance;
  state.outcomes = outcomes;
  state.fighters = explorer.fighters;
  state.fighterById = new Map(state.fighters.map((fighter) => [fighter.id, fighter]));
  state.fightColumn = new Map(explorer.fight_columns.map((column, index) => [column, index]));
  state.bayesianByPair = new Map();
  if (vegas?.["fighter id"]) Object.keys(vegas["fighter id"]).forEach((index) => {
    const fighterId = String(vegas["fighter id"]?.[index] || "");
    const opponentId = String(vegas["opponent id"]?.[index] || "");
    if (!fighterId || !opponentId || !vegas["bayesian model id"]?.[index]) return;
    const key = [fighterId, opponentId].sort().join("|");
    state.bayesianByPair.set(key, {
      fighter_id: fighterId,
      opponent_id: opponentId,
      model_id: vegas["bayesian model id"]?.[index],
      mean: finite(vegas["bayesian posterior mean"]?.[index]),
      median: finite(vegas["bayesian posterior median"]?.[index]),
      lower: finite(vegas["bayesian probability lower"]?.[index]),
      upper: finite(vegas["bayesian probability upper"]?.[index]),
      credible_level: finite(vegas["bayesian credible level"]?.[index]),
      logit_location: finite(vegas["bayesian calibrated logit location"]?.[index]),
      logit_scale: finite(vegas["bayesian calibrated logit scale"]?.[index]),
      status: vegas["bayesian status"]?.[index],
    });
  });
}

function fighterByName(name) {
  const key = normalize(name);
  return state.fighters.find((fighter) => normalize(fighter.name) === key) || null;
}

function findFighters(query, limit = 24) {
  const terms = normalize(query).split(" ").filter(Boolean);
  const candidates = terms.length ? state.fighters.filter((fighter) => {
    const haystack = normalize(`${fighter.name} ${fighterDivision(fighter)} ${fighter.stance || ""}`);
    return terms.every((term) => haystack.includes(term));
  }) : [...state.fighters];
  return candidates
    .sort((left, right) => {
      const leftStarts = normalize(left.name).startsWith(terms.join(" ")) ? 1 : 0;
      const rightStarts = normalize(right.name).startsWith(terms.join(" ")) ? 1 : 0;
      return rightStarts - leftStarts || fullRecord(right).recorded_bouts - fullRecord(left).recorded_bouts || left.name.localeCompare(right.name);
    })
    .slice(0, limit);
}

function setRoute(route) {
  const hash = `#${route}`;
  if (window.location.hash === hash) applyRoute();
  else window.location.hash = hash;
}

function configureGraphMatchup(fighterAId, fighterBId, depth = 1) {
  if (!state.fighterById.has(fighterAId) || !state.fighterById.has(fighterBId) || fighterAId === fighterBId) return false;
  setGraphFilterMode("matchup");
  $("#graph-matchup-fighter-a").value = fighterAId;
  $("#graph-matchup-fighter-b").value = fighterBId;
  $("#graph-matchup-depth").value = String(Math.max(1, Math.min(2, Number(depth) || 1)));
  $("#graph-division").value = "*";
  $("#graph-promotion").value = "";
  $("#graph-start-date").value = "";
  $("#graph-end-date").value = "";
  selectGraphQuickRange("all");
  return true;
}

function clearMarketMatchupFocus() {
  document.querySelectorAll("#market-matchups .is-route-target").forEach((card) => card.classList.remove("is-route-target"));
}

function focusMarketMatchup(fighterId, opponentId) {
  clearMarketMatchupFocus();
  const target = Array.from(document.querySelectorAll("#market-matchups [data-market-fighter-id][data-market-opponent-id]")).find((card) => {
    const cardFighter = card.dataset.marketFighterId;
    const cardOpponent = card.dataset.marketOpponentId;
    return (cardFighter === fighterId && cardOpponent === opponentId)
      || (cardFighter === opponentId && cardOpponent === fighterId);
  });
  if (!target) return false;
  target.classList.add("is-route-target");
  const prices = target.querySelector('details[data-book-lines="moneyline"]');
  if (prices) prices.open = true;
  target.focus({ preventScroll: true });
  target.scrollIntoView({ behavior: "smooth", block: "start" });
  return true;
}

function showView(name) {
  document.querySelectorAll("[data-view]").forEach((view) => view.classList.toggle("is-active", view.dataset.view === name));
  document.querySelectorAll("[data-nav]").forEach((button) => button.classList.toggle("is-active", button.dataset.nav === name));
  document.title = `${name === "matchups" ? "Matchups" : name === "fighters" ? "Fighters" : name === "graph" ? "Fight graph" : name === "market" ? "Market" : "Model & data"} · UFC Data Lab`;
}

function applyRoute() {
  if (!state.explorer) return;
  const parts = window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const view = ["matchups", "fighters", "graph", "market", "data"].includes(parts[0]) ? parts[0] : "matchups";
  showView(view);

  if (view === "graph") {
    const preparation = prepareFightGraph();
    if (parts[1] && parts[2] && configureGraphMatchup(parts[1], parts[2], 1)) {
      const expectedHash = `#graph/${parts[1]}/${parts[2]}`;
      Promise.resolve(preparation).then(() => {
        if (window.location.hash === expectedHash && state.fightGraphFilterMode === "matchup") drawFightGraph();
      });
    }
  }

  if (view === "fighters" && parts[1]) renderFighterProfile(parts[1]);
  else if (view === "fighters") showFighterDirectory();

  if (view === "matchups" && parts[1] && parts[2]) {
    const fighterA = state.fighterById.get(parts[1]);
    const fighterB = state.fighterById.get(parts[2]);
    if (fighterA && fighterB) {
      selectMatchupFighter("a", fighterA);
      selectMatchupFighter("b", fighterB);
      renderMatchup(fighterA, fighterB);
      return;
    }
  }
  if (view === "market") {
    if (parts[1] && parts[2] && focusMarketMatchup(parts[1], parts[2])) return;
    clearMarketMatchupFocus();
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderCoverage() {
  const container = $("#coverage-summary");
  container.replaceChildren();
  const values = [
    [state.explorer.counts.fighters.toLocaleString(), "fighter profiles"],
    [state.explorer.counts.unique_fights.toLocaleString(), "UFCStats fights"],
    [(state.explorer.counts.linked_external_fights || 0).toLocaleString(), "linked Bellator / ONE fights"],
    [formatDate(state.explorer.data_through), "data through"],
  ];
  values.forEach(([value, label]) => {
    const item = element("div", "coverage-stat");
    appendText(item, "strong", "", value);
    appendText(item, "span", "", label);
    container.append(item);
  });
}

function makeAutocomplete(input, results, side) {
  const render = () => {
    results.replaceChildren();
    const matches = findFighters(input.value, 28).filter((fighter) => fighter.id !== state.selected[side === "a" ? "b" : "a"]?.id);
    matches.forEach((fighter) => {
      const button = element("button", "search-result");
      button.type = "button";
      button.setAttribute("role", "option");
      const identity = element("span");
      appendText(identity, "strong", "", fighter.name);
      appendText(identity, "small", "", [fighterDivision(fighter), fighter.stance].filter(Boolean).join(" · ") || "Profile only");
      button.append(identity, element("span", "record", record(fighter)));
      button.addEventListener("click", () => selectMatchupFighter(side, fighter));
      results.append(button);
    });
    if (!matches.length) results.append(element("div", "empty-state", "No matching fighters."));
    results.classList.add("is-open");
    input.setAttribute("aria-expanded", "true");
  };
  input.addEventListener("input", () => { state.selected[side] = null; updateAnalyzeButton(); render(); });
  input.addEventListener("focus", render);
  input.addEventListener("keydown", (event) => { if (event.key === "Escape") closeAutocomplete(input, results); });
}

function closeAutocomplete(input, results) {
  results.classList.remove("is-open");
  input.setAttribute("aria-expanded", "false");
}

function selectMatchupFighter(side, fighter) {
  state.selected[side] = fighter;
  const input = $(`#matchup-fighter-${side}`);
  input.value = fighter.name;
  closeAutocomplete(input, $(`#matchup-results-${side}`));
  updateAnalyzeButton();
}

function updateAnalyzeButton() {
  $("#analyze-matchup").disabled = !state.selected.a || !state.selected.b || state.selected.a.id === state.selected.b.id;
}

function clearMatchup() {
  state.selected = { a: null, b: null };
  ["a", "b"].forEach((side) => { $(`#matchup-fighter-${side}`).value = ""; closeAutocomplete($(`#matchup-fighter-${side}`), $(`#matchup-results-${side}`)); });
  $("#matchup-workbench").replaceChildren();
  updateAnalyzeButton();
  if (window.location.hash.startsWith("#matchups/")) setRoute("matchups");
}

function legacyRows() {
  if (!vegasMatchesCurrentCard() || !state.vegas?.["fighter name"]) return [];
  return Object.keys(state.vegas["fighter name"]).map((index) => ({
    fighter_id: state.vegas["fighter id"]?.[index],
    fighter_name: state.vegas["fighter name"]?.[index],
    opponent_id: state.vegas["opponent id"]?.[index],
    opponent_name: state.vegas["opponent name"]?.[index],
    model_probability_for_fighter: state.vegas["model probability"]?.[index],
    full_market_consensus: {
      fighter_probability: state.vegas["market no-vig fighter probability"]?.[index],
      opponent_probability: finite(state.vegas["market no-vig fighter probability"]?.[index]) === null ? null : 1 - Number(state.vegas["market no-vig fighter probability"]?.[index]),
    },
    current_signal: null,
  }));
}

function publicationMatchesCurrentCard(eventDate, eventId = "") {
  const cardDate = dateKey(state.card?.date);
  if (!cardDate || dateKey(eventDate) !== cardDate) return false;
  const cardEventId = String(state.card?.event_id || "").trim();
  const publicationEventId = String(eventId || "").trim();
  return !cardEventId || !publicationEventId || cardEventId === publicationEventId;
}

function vegasMatchesCurrentCard() {
  const dates = Object.values(state.vegas?.date || {}).map(dateKey).filter(Boolean);
  if (!dates.length || dates.some((date) => date !== dateKey(state.card?.date))) return false;
  const eventIds = Object.values(state.vegas?.["event id"] || {}).map((value) => String(value || "").trim()).filter(Boolean);
  const cardEventId = String(state.card?.event_id || "").trim();
  return !cardEventId || !eventIds.length || eventIds.every((eventId) => eventId === cardEventId);
}

function currentMarket() {
  return publicationMatchesCurrentCard(state.market?.event_date, state.market?.event_id) ? state.market : null;
}

function currentOutcomes() {
  return publicationMatchesCurrentCard(state.outcomes?.event_date, state.outcomes?.event_id) ? state.outcomes : null;
}

function currentMatchups() {
  const market = currentMarket();
  return market?.matchups?.length ? market.matchups : legacyRows();
}

function bayesianForMatchup(matchup) {
  const fighterId = String(matchup?.fighter_id || "");
  const opponentId = String(matchup?.opponent_id || "");
  if (!fighterId || !opponentId) return null;
  const source = state.bayesianByPair.get([fighterId, opponentId].sort().join("|"));
  if (!source || source.mean === null || source.lower === null || source.upper === null || source.logit_location === null || source.logit_scale === null) return null;
  if (source.fighter_id === fighterId) return source;
  return {
    ...source,
    fighter_id: fighterId,
    opponent_id: opponentId,
    mean: 1 - source.mean,
    median: source.median === null ? null : 1 - source.median,
    lower: 1 - source.upper,
    upper: 1 - source.lower,
    logit_location: -source.logit_location,
  };
}

function bayesianSide(forecast, side) {
  if (!forecast) return null;
  if (side === "fighter") return forecast;
  return {
    ...forecast,
    mean: 1 - forecast.mean,
    median: forecast.median === null ? null : 1 - forecast.median,
    lower: 1 - forecast.upper,
    upper: 1 - forecast.lower,
    logit_location: -forecast.logit_location,
  };
}

function bayesianExpectedReturn(distribution, odds) {
  const decimal = decimalOdds(odds);
  if (!distribution || decimal === null) return null;
  const breakEven = 1 / decimal;
  const thresholdLogit = probabilityLogit(breakEven);
  let probabilityPositive = null;
  if (thresholdLogit !== null) {
    probabilityPositive = distribution.logit_scale === 0
      ? Number(distribution.mean > breakEven)
      : 1 - normalCdf((thresholdLogit - distribution.logit_location) / distribution.logit_scale);
  }
  return {
    mean: decimal * distribution.mean - 1,
    lower: decimal * distribution.lower - 1,
    upper: decimal * distribution.upper - 1,
    probability_positive: probabilityPositive,
    break_even_probability: breakEven,
  };
}

function bestBayesianCandidate(matchup) {
  const forecast = bayesianForMatchup(matchup);
  if (!forecast || forecast.status !== "paper_only_challenger") return null;
  const candidates = [];
  (matchup.book_quotes || []).filter((quote) => quote.eligible_for_consensus !== false).forEach((quote) => {
    [["fighter", matchup.fighter_name, quote.fighter_moneyline], ["opponent", matchup.opponent_name, quote.opponent_moneyline]].forEach(([side, name, odds]) => {
      const distribution = bayesianSide(forecast, side);
      const expected = bayesianExpectedReturn(distribution, odds);
      if (!expected) return;
      candidates.push({ side, name, book: quote.book, odds, distribution, ...expected });
    });
  });
  candidates.sort((left, right) => right.mean - left.mean || right.probability_positive - left.probability_positive || left.book.localeCompare(right.book));
  return candidates[0] || null;
}

function bayesianFilteredCandidate(matchup) {
  const signal = matchup?.current_signal;
  if (!signal || !["fighter", "opponent"].includes(signal.paper_action)) return null;
  const forecast = bayesianForMatchup(matchup);
  if (!forecast) return null;
  const distribution = bayesianSide(forecast, signal.paper_action);
  const expected = bayesianExpectedReturn(distribution, signal.offered_moneyline);
  if (!expected) return null;
  const policy = state.bayesian?.decision_policy || {};
  const minimumMean = Number(policy.minimum_mean_expected_return ?? 0.05);
  const minimumProbability = Number(policy.minimum_probability_positive_expected_return ?? 0.80);
  let status = "qualified";
  if (forecast.status !== "paper_only_challenger") status = "bayesian_status_veto";
  else if (expected.mean < minimumMean) status = "bayesian_mean_ev_veto";
  else if (expected.probability_positive < minimumProbability) status = "bayesian_probability_veto";
  return {
    side: signal.paper_action,
    name: signal.action_name || signal.best_candidate_name,
    book: signal.target_book,
    odds: signal.offered_moneyline,
    distribution,
    status,
    qualified: status === "qualified",
    ...expected,
  };
}

function renderCurrentCard() {
  const container = $("#upcoming-matchups");
  container.replaceChildren();
  $("#current-card-title").textContent = state.card?.title || "Current fight card";
  $("#current-card-meta").textContent = state.card ? `${state.card.date} · ${currentMatchups().length} scheduled matchups` : "Current card metadata is unavailable.";
  const matchups = currentMatchups();
  if (!matchups.length) {
    container.append(element("div", "empty-state", "No current-card matchups are published yet. Use the matchup builder below to research any two fighters."));
    return;
  }
  matchups.forEach((matchup, index) => {
    const fighter = state.fighterById.get(matchup.fighter_id) || fighterByName(matchup.fighter_name);
    const opponent = state.fighterById.get(matchup.opponent_id) || fighterByName(matchup.opponent_name);
    const card = element("article", "matchup-card");
    const top = element("div", "matchup-card-top");
    appendText(top, "span", "", fighterDivision(fighter) || fighterDivision(opponent) || `Bout ${index + 1}`);
    appendText(top, "span", "", `${Math.min(fullRecord(fighter).recorded_bouts || 0, fullRecord(opponent).recorded_bouts || 0)}-bout minimum history`);
    card.append(top);

    const pair = element("div", "matchup-pair");
    [fighter, opponent].forEach((person, personIndex) => {
      if (personIndex) pair.append(element("span", "vs", "VS"));
      const side = element("div", "matchup-side");
      const name = person?.name || (personIndex ? matchup.opponent_name : matchup.fighter_name);
      if (person) side.append(actionButton(name, "", () => setRoute(`fighters/${person.id}`)));
      else appendText(side, "strong", "", name);
      appendText(side, "small", "", person ? `${record(person)} · ${formatNumber(person.career.sig_strikes_landed_per_minute)} SLpM` : "No profile match");
      pair.append(side);
    });
    card.append(pair);

    const market = element("div", "matchup-market");
    const probability = finite(matchup.full_market_consensus?.fighter_probability);
    const modelProbability = finite(matchup.model_probability_for_fighter);
    const left = element("span");
    appendText(left, "strong", "", probability === null ? "Market unavailable" : `${formatPercent(probability)} / ${formatPercent(1 - probability)}`);
    appendText(left, "span", "", " no-vig market");
    const right = element("span");
    appendText(right, "strong", "", modelProbability === null ? "Model unavailable" : `${formatPercent(modelProbability)} / ${formatPercent(1 - modelProbability)}`);
    appendText(right, "span", "", " model");
    market.append(left, right);
    card.append(market);

    const bayesian = bayesianForMatchup(matchup);
    if (bayesian) {
      const posterior = element("div", "matchup-market");
      const posteriorValue = element("span");
      appendText(posteriorValue, "strong", "", formatPercent(bayesian.mean));
      appendText(posteriorValue, "span", "", ` posterior mean for ${matchup.fighter_name}`);
      const posteriorRange = element("span");
      appendText(posteriorRange, "strong", "", `${formatPercent(bayesian.lower)}-${formatPercent(bayesian.upper)}`);
      appendText(posteriorRange, "span", "", bayesian.status === "paper_only_challenger" ? ` ${formatPercent(bayesian.credible_level, 0)} credible interval` : " parameter interval · EV abstains for low history");
      posterior.append(posteriorValue, posteriorRange);
      card.append(posterior);
    }

    const actions = element("div", "card-actions");
    const analyze = actionButton("Research matchup", "primary-button small-button", () => {
      if (fighter && opponent) setRoute(`matchups/${fighter.id}/${opponent.id}`);
    });
    analyze.disabled = !fighter || !opponent;
    const graphButton = actionButton("View fight graph", "secondary-button small-button", () => {
      if (fighter && opponent) setRoute(`graph/${fighter.id}/${opponent.id}`);
    });
    graphButton.disabled = !fighter || !opponent;
    if (!fighter || !opponent) graphButton.title = "Both fighters need linked profiles to build their fight graph.";
    const quoteCount = Array.isArray(matchup.book_quotes) ? matchup.book_quotes.length : 0;
    const hasCurrentPrices = Boolean(matchup.fighter_id && matchup.opponent_id && quoteCount);
    const marketButton = actionButton(
      hasCurrentPrices ? `View ${quoteCount} book prices` : "Prices unavailable",
      "text-button small-button",
      () => {
        if (hasCurrentPrices) setRoute(`market/${matchup.fighter_id}/${matchup.opponent_id}`);
      },
    );
    marketButton.disabled = !hasCurrentPrices;
    if (!hasCurrentPrices) marketButton.title = "No current book-price capture is published for this matchup.";
    actions.append(analyze, graphButton, marketButton);
    card.append(actions);
    container.append(card);
  });
}

function marketMatchupFor(fighterA, fighterB) {
  return currentMatchups().find((matchup) => {
    const ids = new Set([matchup.fighter_id, matchup.opponent_id]);
    return ids.has(fighterA.id) && ids.has(fighterB.id);
  }) || null;
}

const COMPARISON_GROUPS = [
  ["Tale of the tape", [
    ["Height", (f) => f.height_inches, "inches", "context"],
    ["Reach", (f) => f.reach_inches, "inches", "context"],
    ["Age", (f) => ageOn(f, state.card?.date), "years", "context"],
    ["Recorded MMA bouts", (f) => fullRecord(f).recorded_bouts, "integer", "higher"],
    ["UFCStats bouts", (f) => f.career.recorded_bouts, "integer", "higher"],
    ["Average UFC fight time", (f) => f.career.average_fight_minutes, "minutes", "context"],
  ]],
  ["Striking", [
    ["Sig. strikes landed / min", (f) => f.career.sig_strikes_landed_per_minute, "decimal", "higher"],
    ["Sig. strikes absorbed / min", (f) => f.career.sig_strikes_absorbed_per_minute, "decimal", "lower"],
    ["Sig. strike differential / min", (f) => f.career.significant_strike_differential_per_minute, "signed", "higher"],
    ["Sig. strike accuracy", (f) => f.career.sig_strike_accuracy, "percentage", "higher"],
    ["Sig. strike defense", (f) => f.career.sig_strike_defense, "percentage", "higher"],
    ["Knockdowns / 15 min", (f) => f.career.knockdowns_per_15, "decimal", "higher"],
    ["Knockdowns absorbed / 15 min", (f) => f.career.knockdowns_absorbed_per_15, "decimal", "lower"],
  ]],
  ["Grappling", [
    ["Takedowns / 15 min", (f) => f.career.takedowns_landed_per_15, "decimal", "higher"],
    ["Takedown accuracy", (f) => f.career.takedown_accuracy, "percentage", "higher"],
    ["Takedown defense", (f) => f.career.takedown_defense, "percentage", "higher"],
    ["Submission attempts / 15 min", (f) => f.career.submission_attempts_per_15, "decimal", "higher"],
    ["Control minutes / 15 min", (f) => f.career.control_minutes_per_15, "decimal", "higher"],
    ["Share of recorded control", (f) => f.career.control_share, "percentage", "higher"],
  ]],
  ["Results & style", [
    ["All-promotion win rate", (f) => fullRecord(f).win_rate, "percentage", "higher"],
    ["All-promotion finish rate", (f) => fullRecord(f).finish_rate, "percentage", "context"],
    ["Head strike share", (f) => f.career.head_strike_share, "percentage", "context"],
    ["Body strike share", (f) => f.career.body_strike_share, "percentage", "context"],
    ["Leg strike share", (f) => f.career.leg_strike_share, "percentage", "context"],
    ["Distance strike share", (f) => f.career.distance_strike_share, "percentage", "context"],
    ["Ground strike share", (f) => f.career.ground_strike_share, "percentage", "context"],
  ]],
];

function formatMetric(value, format) {
  if (finite(value) === null) return "—";
  if (format === "percentage") return formatPercent(value);
  if (format === "inches") return `${formatNumber(value, 0)} in`;
  if (format === "years") return `${formatNumber(value, 0)}`;
  if (format === "minutes") return `${formatNumber(value, 1)} min`;
  if (format === "integer") return formatNumber(value, 0);
  if (format === "signed") return `${Number(value) > 0 ? "+" : ""}${formatNumber(value)}`;
  return formatNumber(value);
}

function comparisonEdge(valueA, valueB, better) {
  const a = finite(valueA); const b = finite(valueB);
  if (a === null || b === null || better === "context" || a === b) return [false, false];
  return better === "lower" ? [a < b, b < a] : [a > b, b > a];
}

function renderComparisonTable(fighterA, fighterB) {
  const panel = element("section", "panel comparison-panel");
  const heading = element("div", "section-heading");
  const headingCopy = element("div");
  appendText(headingCopy, "p", "eyebrow", "Side-by-side");
  appendText(headingCopy, "h2", "", "Complete matchup comparison");
  appendText(headingCopy, "p", "section-note", "Highlighted values are directionally favorable for that metric; descriptive style metrics are not ranked.");
  heading.append(headingCopy);
  panel.append(heading);

  COMPARISON_GROUPS.forEach(([title, metrics]) => {
    const section = element("div", "comparison-section");
    appendText(section, "h3", "", title);
    const table = element("table", "comparison-table");
    const thead = document.createElement("thead");
    const header = document.createElement("tr");
    [fighterA.name, "", "Metric", "", fighterB.name].forEach((value) => appendText(header, "th", "", value));
    thead.append(header); table.append(thead);
    const body = document.createElement("tbody");
    metrics.forEach(([label, getter, format, better]) => {
      const valueA = getter(fighterA); const valueB = getter(fighterB);
      const [edgeA, edgeB] = comparisonEdge(valueA, valueB, better);
      const row = document.createElement("tr");
      const aCell = appendText(row, "td", `value-a${edgeA ? " is-edge" : ""}`, formatMetric(valueA, format));
      aCell.setAttribute("data-label", fighterA.name);
      const bars = element("td");
      const barsWrap = element("div", "comparison-bars");
      const halfA = element("span", "bar-half"); const halfB = element("span", "bar-half");
      const max = Math.max(Math.abs(finite(valueA) || 0), Math.abs(finite(valueB) || 0), 0.0001);
      const barA = element("span", "bar"); const barB = element("span", "bar bar-b");
      barA.style.width = `${Math.max(4, Math.abs(finite(valueA) || 0) / max * 100)}%`;
      barB.style.width = `${Math.max(4, Math.abs(finite(valueB) || 0) / max * 100)}%`;
      halfA.append(barA); halfB.append(barB); barsWrap.append(halfA, halfB); bars.append(barsWrap);
      row.append(bars);
      appendText(row, "td", "metric-label", label);
      row.append(element("td"));
      const bCell = appendText(row, "td", `value-b${edgeB ? " is-edge" : ""}`, formatMetric(valueB, format));
      bCell.setAttribute("data-label", fighterB.name);
      body.append(row);
    });
    table.append(body); section.append(table); panel.append(section);
  });
  return panel;
}

function strongerName(fighterA, valueA, fighterB, valueB, lower = false) {
  if (finite(valueA) === null || finite(valueB) === null || valueA === valueB) return null;
  return lower ? (valueA < valueB ? fighterA.name : fighterB.name) : (valueA > valueB ? fighterA.name : fighterB.name);
}

function matchupInsights(fighterA, fighterB) {
  const a = fighterA.career; const b = fighterB.career;
  const allA = fullRecord(fighterA); const allB = fullRecord(fighterB);
  const insights = [];
  const minimumBouts = Math.min(a.recorded_bouts, b.recorded_bouts);
  if (minimumBouts < 3) {
    insights.push(["Sample warning", "Treat differences cautiously", `At least one fighter has only ${minimumBouts} recorded UFC bout${minimumBouts === 1 ? "" : "s"}. Rates can move sharply with so little exposure.`, true]);
  } else {
    insights.push(["Data confidence", `${a.recorded_bouts} vs ${b.recorded_bouts} recorded bouts`, `Paired opponent statistics are available for ${a.paired_opponent_stat_bouts} and ${b.paired_opponent_stat_bouts} bouts respectively.`, false]);
  }

  const reachDiff = finite(fighterA.reach_inches) !== null && finite(fighterB.reach_inches) !== null ? fighterA.reach_inches - fighterB.reach_inches : null;
  const heightDiff = finite(fighterA.height_inches) !== null && finite(fighterB.height_inches) !== null ? fighterA.height_inches - fighterB.height_inches : null;
  if (reachDiff !== null) {
    const longer = reachDiff === 0 ? "Neither fighter" : reachDiff > 0 ? fighterA.name : fighterB.name;
    insights.push(["Physical", reachDiff === 0 ? "Equal recorded reach" : `${longer} has ${Math.abs(reachDiff)} in of reach`, heightDiff === null || heightDiff === 0 ? "Height is even or unavailable." : `${heightDiff > 0 ? fighterA.name : fighterB.name} is ${Math.abs(heightDiff)} in taller.`, false]);
  }

  const netLeader = strongerName(fighterA, a.significant_strike_differential_per_minute, fighterB, b.significant_strike_differential_per_minute);
  insights.push(["Striking exchange", netLeader ? `${netLeader} owns the better recorded differential` : "Recorded differentials are even or incomplete", `${fighterA.name}: ${formatMetric(a.sig_strikes_landed_per_minute, "decimal")} landed / ${formatMetric(a.sig_strikes_absorbed_per_minute, "decimal")} absorbed per minute. ${fighterB.name}: ${formatMetric(b.sig_strikes_landed_per_minute, "decimal")} / ${formatMetric(b.sig_strikes_absorbed_per_minute, "decimal")}.`, false]);

  const aTakedownLook = finite(a.takedowns_landed_per_15) !== null && finite(b.takedown_defense) !== null ? a.takedowns_landed_per_15 * (1 - b.takedown_defense) : null;
  const bTakedownLook = finite(b.takedowns_landed_per_15) !== null && finite(a.takedown_defense) !== null ? b.takedowns_landed_per_15 * (1 - a.takedown_defense) : null;
  const pressure = strongerName(fighterA, aTakedownLook, fighterB, bTakedownLook);
  insights.push(["Grappling interaction", pressure ? `${pressure} has the stronger takedown-pressure marker` : "No clear takedown-pressure marker", `${fighterA.name}'s ${formatNumber(a.takedowns_landed_per_15)} TD/15 meets ${formatPercent(b.takedown_defense)} defense; ${fighterB.name}'s ${formatNumber(b.takedowns_landed_per_15)} meets ${formatPercent(a.takedown_defense)}. This is descriptive, not a forecast.`, false]);

  const finishLeader = strongerName(fighterA, allA.finish_rate, fighterB, allB.finish_rate);
  insights.push(["Finishing profile", finishLeader ? `${finishLeader} finishes a larger share of recorded wins` : "Similar or incomplete finish rates", `${fighterA.name}: ${allA.ko_tko_wins} KO/TKO, ${allA.submission_wins} submissions. ${fighterB.name}: ${allB.ko_tko_wins} KO/TKO, ${allB.submission_wins} submissions across all linked promotions.`, false]);

  const lastA = daysSince(allA.last_fight_date); const lastB = daysSince(allB.last_fight_date);
  const activityLeader = strongerName(fighterA, lastA, fighterB, lastB, true);
  insights.push(["Recency", activityLeader ? `${activityLeader} fought more recently` : "Similar or incomplete activity data", `${fighterA.name}: ${allA.last_fight_date ? formatDate(allA.last_fight_date) : "unknown"}. ${fighterB.name}: ${allB.last_fight_date ? formatDate(allB.last_fight_date) : "unknown"}.`, false]);
  return insights;
}

function renderMatchupMarketContext(fighterA, fighterB) {
  const matchup = marketMatchupFor(fighterA, fighterB);
  if (!matchup) return null;
  const panel = element("section", "panel");
  const heading = element("div", "section-heading");
  const copy = element("div"); appendText(copy, "p", "eyebrow", "Current-card context"); appendText(copy, "h2", "", "Model and market");
  appendText(copy, "p", "section-note", "Displayed separately from historical performance so consensus price is never mistaken for the statistical profile.");
  heading.append(copy, actionButton("Open market research", "secondary-button", () => setRoute("market"))); panel.append(heading);
  const modelFighter = finite(matchup.model_probability_for_fighter);
  const marketFighter = finite(matchup.full_market_consensus?.fighter_probability);
  const aIsFighter = matchup.fighter_id === fighterA.id;
  const probabilityA = modelFighter === null ? null : aIsFighter ? modelFighter : 1 - modelFighter;
  const marketA = marketFighter === null ? null : aIsFighter ? marketFighter : 1 - marketFighter;
  const tiles = element("div", "stats-grid");
  [[formatPercent(probabilityA), `${fighterA.name} model`], [formatPercent(probabilityA === null ? null : 1 - probabilityA), `${fighterB.name} model`], [formatPercent(marketA), `${fighterA.name} market`], [formatPercent(marketA === null ? null : 1 - marketA), `${fighterB.name} market`]].forEach(([value, label]) => {
    const tile = element("div", "stat-tile"); appendText(tile, "strong", "", value); appendText(tile, "span", "", label); tiles.append(tile);
  });
  panel.append(tiles); return panel;
}

function renderMatchup(fighterA, fighterB) {
  const container = $("#matchup-workbench"); container.replaceChildren();
  const header = element("section", "matchup-header");
  [fighterA, fighterB].forEach((fighter, index) => {
    if (index) header.append(element("span", "versus-mark", "VS"));
    const side = element("div", "matchup-fighter");
    appendText(side, "p", "eyebrow", fighterDivision(fighter) || "Fighter profile");
    appendText(side, "h2", "", fighter.name);
    appendText(side, "p", "", `${record(fighter)} · ${fighter.stance || "Stance unknown"} · ${fighter.reach || "Reach unknown"}`);
    side.append(actionButton("Open full profile →", "text-button", () => setRoute(`fighters/${fighter.id}`)));
    header.append(side);
  });
  container.append(header);
  const insightGrid = element("section", "insight-grid");
  matchupInsights(fighterA, fighterB).forEach(([label, title, copy, caution]) => {
    const card = element("article", `insight-card${caution ? " caution-card" : ""}`);
    appendText(card, "span", "insight-label", label); appendText(card, "h3", "", title); appendText(card, "p", "", copy); insightGrid.append(card);
  });
  container.append(insightGrid, renderComparisonTable(fighterA, fighterB));
  const context = renderMatchupMarketContext(fighterA, fighterB); if (context) container.append(context);
  container.scrollIntoView({ behavior: "smooth", block: "start" });
}

function populateFilters() {
  const divisions = [...new Set(state.fighters.flatMap((fighter) => [fighter.scheduled_division, ...fighter.career.divisions.map((division) => division.name)]).filter(Boolean))].sort();
  const stances = [...new Set(state.fighters.map((fighter) => fighter.stance).filter(Boolean))].sort();
  divisions.forEach((value) => { const option = element("option", "", value); option.value = value; $("#division-filter").append(option); });
  divisions.forEach((value) => { const option = element("option", "", value); option.value = value; $("#graph-division").append(option); });
  stances.forEach((value) => { const option = element("option", "", value); option.value = value; $("#stance-filter").append(option); });
  const graphFighters = [...state.fighters].sort((left, right) => left.name.localeCompare(right.name));
  [$("#graph-matchup-fighter-a"), $("#graph-matchup-fighter-b")].forEach((select) => {
    const fragment = document.createDocumentFragment();
    graphFighters.forEach((fighter) => { const option = element("option", "", fighter.name); option.value = fighter.id; fragment.append(option); });
    select.append(fragment);
  });
}

function filteredDirectoryFighters() {
  const query = normalize($("#fighter-directory-search").value);
  const terms = query.split(" ").filter(Boolean);
  const division = $("#division-filter").value;
  const stance = $("#stance-filter").value;
  const recordedOnly = $("#recorded-only").checked;
  return state.fighters.filter((fighter) => {
    const haystack = normalize(`${fighter.name} ${fighterDivision(fighter)} ${fighter.stance || ""}`);
    return (!terms.length || terms.every((term) => haystack.includes(term)))
      && (!division || fighter.scheduled_division === division || fighter.career.divisions.some((item) => item.name === division))
      && (!stance || fighter.stance === stance)
      && (!recordedOnly || fullRecord(fighter).recorded_bouts > 0);
  }).sort((a, b) => fullRecord(b).recorded_bouts - fullRecord(a).recorded_bouts || a.name.localeCompare(b.name));
}

function showFighterDirectory() {
  $(".directory-controls").hidden = false; $("#fighter-directory").hidden = false; $("#fighter-profile").replaceChildren();
  renderFighterDirectory();
}

function renderFighterDirectory() {
  const container = $("#fighter-directory"); container.replaceChildren();
  const fighters = filteredDirectoryFighters();
  const summary = element("div", "directory-summary");
  appendText(summary, "span", "", `${fighters.length.toLocaleString()} matching fighter${fighters.length === 1 ? "" : "s"}`);
  appendText(summary, "span", "", `Showing ${Math.min(fighters.length, state.directoryLimit).toLocaleString()}`); container.append(summary);
  if (!fighters.length) { container.append(element("div", "empty-state", "No fighters match those filters.")); return; }
  const grid = element("div", "fighter-grid");
  fighters.slice(0, state.directoryLimit).forEach((fighter) => {
    const card = element("article", "fighter-card");
    const title = element("div"); appendText(title, "h3", "", fighter.name);
    appendText(title, "div", "fighter-card-sub", [fighterDivision(fighter), fighter.stance, fighter.reach ? `${fighter.reach} reach` : ""].filter(Boolean).join(" · ") || "Profile information only"); card.append(title);
    const stats = element("div", "mini-stats");
    [[record(fighter), "record"], [formatNumber(fighter.career.sig_strikes_landed_per_minute), "SLpM"], [formatNumber(fighter.career.takedowns_landed_per_15), "TD / 15"]].forEach(([value, label]) => {
      const stat = element("div", "mini-stat"); appendText(stat, "strong", "", value); appendText(stat, "span", "", label); stats.append(stat);
    }); card.append(stats, actionButton("Open full profile", "secondary-button", () => setRoute(`fighters/${fighter.id}`))); grid.append(card);
  });
  container.append(grid);
  if (fighters.length > state.directoryLimit) {
    const row = element("div", "load-more-row"); row.append(actionButton(`Show ${Math.min(48, fighters.length - state.directoryLimit)} more`, "secondary-button", () => { state.directoryLimit += 48; renderFighterDirectory(); })); container.append(row);
  }
}

function statTile(value, label, note = "") {
  const tile = element("div", "stat-tile"); appendText(tile, "strong", "", value); appendText(tile, "span", "", label); if (note) appendText(tile, "small", "", note); return tile;
}

function bioItem(value, label) {
  const item = element("div", "bio-item"); appendText(item, "strong", "", value || "—"); appendText(item, "span", "", label); return item;
}

function renderCareerTable(fighter, group) {
  const wrapper = element("div", "explain-card"); appendText(wrapper, "h3", "", group);
  const table = element("table", "data-table"); const body = document.createElement("tbody");
  Object.entries(state.explorer.data_dictionary.career).filter(([, definition]) => definition.group === group).forEach(([key, definition]) => {
    const row = document.createElement("tr"); appendText(row, "td", "", definition.label); appendText(row, "td", "", formatMetric(fighter.career[key], definition.format)); body.append(row);
  });
  table.append(body); wrapper.append(table); return wrapper;
}

async function pairedFight(fight) {
  const opponent = state.fighterById.get(fight.opponent_id);
  if (!opponent) return null;
  await ensureFighterFights(opponent);
  const array = opponent.fights.find((values) => values[state.fightColumn.get("fight_id")] === fight.fight_id);
  return array ? decodeFight(array) : null;
}

function renderFightDetails(body, fight, opponentFight, fighterName) {
  body.replaceChildren();
  if (!fight.stats_available) {
    const notice = element("div", "coverage-notice");
    appendText(notice, "strong", "", "Result metadata only");
    appendText(notice, "p", "", "This source provides the result, method, round, and clock, but not reliable bout-level striking or grappling statistics. Missing detail is intentionally shown as unavailable, not zero.");
    body.append(notice);
    const metadata = element("table", "data-table"); const rows = document.createElement("tbody");
    [["Promotion", fight.promotion || "—"], ["Event", fight.event_name || "—"], ["Method", fight.method || "—"], ["Finish", `Round ${fight.round || "—"} · ${fight.time || "clock unavailable"}`], ["Scheduled format", fight.time_format || "—"], ["Dataset", fight.source_label || fight.source || "—"]].forEach(([label, value]) => { const row = document.createElement("tr"); appendText(row, "td", "", label); appendText(row, "td", "", value); rows.append(row); });
    metadata.append(rows); body.append(metadata);
    if (fight.source_url) {
      const source = element("p", "fight-source", `${fight.event_name || fight.promotion || "Source event"}. `);
      const link = element("a", "", "Open upstream event page"); link.href = fight.source_url; link.target = "_blank"; link.rel = "noreferrer"; source.append(link); body.append(source);
    }
    return;
  }
  const grid = element("div", "fight-detail-grid");
  const groups = [...new Set(Object.values(state.explorer.data_dictionary.fight_stats).map((definition) => definition.group))];
  groups.forEach((group) => {
    const section = element("div"); appendText(section, "h3", "", group);
    const table = element("table", "data-table");
    const head = document.createElement("thead"); const headRow = document.createElement("tr");
    ["Statistic", fighterName, fight.opponent_name].forEach((value) => appendText(headRow, "th", "", value)); head.append(headRow); table.append(head);
    const tableBody = document.createElement("tbody");
    Object.entries(state.explorer.data_dictionary.fight_stats).filter(([, definition]) => definition.group === group).forEach(([key, definition]) => {
      const row = document.createElement("tr"); appendText(row, "td", "", definition.label);
      appendText(row, "td", "", definition.unit === "seconds" ? formatDuration(fight[key]) : formatNumber(fight[key], 0));
      appendText(row, "td", "", opponentFight ? (definition.unit === "seconds" ? formatDuration(opponentFight[key]) : formatNumber(opponentFight[key], 0)) : "—"); tableBody.append(row);
    });
    table.append(tableBody); section.append(table); grid.append(section);
  });
  body.append(grid);
  const source = element("p", "fight-source");
  appendText(source, "span", "", `Fight ID ${fight.fight_id} · Event ID ${fight.event_id} · source card index ${fight.source_card_index ?? "—"} · bout order ${fight.bout_order ?? "—"} · ${fight.time_format || "round format unknown"}. `);
  const link = element("a", "", "Open official UFCStats fight page"); link.href = fight.fight_url; link.target = "_blank"; link.rel = "noreferrer"; source.append(link); body.append(source);
  appendText(source, "span", "", " · ");
  const eventLink = element("a", "", "Open official event page"); eventLink.href = fight.event_url; eventLink.target = "_blank"; eventLink.rel = "noreferrer"; source.append(eventLink);
}

function renderFightHistory(fighter) {
  const panel = element("section", "panel");
  const heading = element("div", "section-heading"); const copy = element("div"); appendText(copy, "p", "eyebrow", "Bout-level data"); appendText(copy, "h2", "", "Recorded fight history");
  appendText(copy, "p", "section-note", "UFCStats bouts include detailed performance data. Linked Bellator and ONE bouts include the result metadata actually available from the external source."); heading.append(copy);
  const decoded = fighter.fights.map(decodeFight);
  const promotions = [...new Set(decoded.map((fight) => fight.promotion).filter(Boolean))].sort();
  const controls = element("div", "history-controls"); appendText(controls, "label", "", "Promotion");
  const promotionFilter = document.createElement("select"); const allOption = element("option", "", "All promotions"); allOption.value = ""; promotionFilter.append(allOption);
  promotions.forEach((promotion) => { const option = element("option", "", promotion); option.value = promotion; promotionFilter.append(option); }); controls.append(promotionFilter); heading.append(controls); panel.append(heading);
  const history = element("div", "fight-history");
  const renderRows = () => {
    history.replaceChildren();
    const visible = decoded.filter((fight) => !promotionFilter.value || fight.promotion === promotionFilter.value);
    if (!visible.length) history.append(element("div", "empty-state", "No recorded fights match this promotion filter."));
    visible.forEach((fight) => {
    const details = document.createElement("details");
    const summary = element("summary", "fight-summary"); appendText(summary, "time", "", formatDate(fight.date, { year: "numeric", month: "short", day: "numeric" }));
    const resultKey = String(fight.result).toLowerCase(); appendText(summary, "span", `result ${resultKey === "w" ? "win" : resultKey === "l" ? "loss" : "neutral"}`, fight.result || "—");
    const opponentName = element("strong", "", fight.opponent_name); summary.append(opponentName);
    appendText(summary, "span", "", `${fight.method || "Method unavailable"} · R${fight.round || "—"} ${fight.time || ""}`);
    const context = element("small", "fight-context"); appendText(context, "span", "promotion-pill", fight.promotion || "Unknown promotion"); context.append(document.createTextNode(` ${fight.event_name || fight.division || "Event unavailable"}`)); summary.append(context); details.append(summary);
    const body = element("div", "details-body", fight.stats_available ? "Open to load complete bout statistics." : "Open to inspect result metadata and source coverage."); details.append(body);
    let rendered = false; details.addEventListener("toggle", async () => {
      if (!details.open || rendered) return;
      rendered = true; body.textContent = fight.stats_available ? "Loading paired opponent statistics…" : "Loading source metadata…";
      try {
        const opponentFight = fight.stats_available ? await pairedFight(fight) : null;
        renderFightDetails(body, fight, opponentFight, fighter.name);
      } catch (error) {
        body.textContent = `Could not load paired opponent statistics: ${error.message}`;
      }
    }); history.append(details);
    });
  };
  promotionFilter.addEventListener("change", renderRows); renderRows();
  panel.append(history); return panel;
}

function renderRawTotals(fighter) {
  const details = document.createElement("details"); const summary = element("summary", "", "Career raw totals and opponent totals"); details.append(summary);
  const body = element("div", "details-body"); appendText(body, "p", "section-note", "Totals are sums of the bout-level rows above. Opponent totals power absorbed and defensive rates.");
  const columns = element("div", "data-columns");
  [["Fighter totals", fighter.career.totals], ["Opponent totals", fighter.career.opponent_totals]].forEach(([title, totals]) => {
    const section = element("div"); appendText(section, "h3", "", title); const table = element("table", "data-table"); const tableBody = document.createElement("tbody");
    Object.entries(state.explorer.data_dictionary.fight_stats).forEach(([key, definition]) => { const row = document.createElement("tr"); appendText(row, "td", "", definition.label); appendText(row, "td", "", definition.unit === "seconds" ? formatDuration(totals[key]) : formatNumber(totals[key], 0)); tableBody.append(row); });
    table.append(tableBody); section.append(table); columns.append(section);
  }); body.append(columns); details.append(body); return details;
}

async function renderFighterProfile(fighterId) {
  const fighter = state.fighterById.get(fighterId);
  if (!fighter) { setRoute("fighters"); return; }
  $(".directory-controls").hidden = true; $("#fighter-directory").hidden = true;
  const container = $("#fighter-profile"); container.replaceChildren();
  const allResults = fullRecord(fighter);
  const header = element("section", "profile-header"); const identity = element("div"); identity.append(actionButton("← Back to fighter directory", "back-button", () => setRoute("fighters")));
  appendText(identity, "p", "eyebrow", fighterDivision(fighter) || "Fighter profile"); appendText(identity, "h2", "", fighter.name);
  appendText(identity, "p", "", `${record(fighter)} across linked promotions · ${allResults.recorded_bouts} bouts · ${fighter.career.recorded_bouts} with UFCStats detail`); header.append(identity);
  const sourceLink = element("a", "", "Open official UFCStats profile ↗"); sourceLink.href = fighter.url; sourceLink.target = "_blank"; sourceLink.rel = "noreferrer"; identity.append(sourceLink);
  const bio = element("div", "profile-bio");
  [[fighter.height || "—", "Height"], [fighter.reach || "—", "Reach"], [fighter.stance || "—", "Stance"], [ageOn(fighter, state.card?.date) ?? "—", "Age at current card"], [fighter.dob_iso ? formatDate(fighter.dob_iso) : fighter.dob || "—", "Date of birth"], [fighter.id, "Stable fighter ID"]].forEach(([value, label]) => bio.append(bioItem(value, label))); header.append(bio); container.append(header);

  const coverage = element("section", "panel coverage-panel"); const coverageHeading = element("div", "section-heading"); const coverageCopy = element("div"); appendText(coverageCopy, "p", "eyebrow", "Source coverage"); appendText(coverageCopy, "h2", "", "What is actually recorded"); appendText(coverageCopy, "p", "section-note", "All-promotion results and UFCStats performance detail are kept separate so missing external statistics cannot silently become zero."); coverageHeading.append(coverageCopy); coverage.append(coverageHeading);
  const coverageStats = element("div", "stats-grid"); [[allResults.recorded_bouts, "All recorded MMA bouts", record(fighter)], [fighter.career.recorded_bouts, "UFCStats detail", "Striking and grappling available"], [promotionBouts(fighter, /bellator/i), "Bellator history", "Result metadata"], [promotionBouts(fighter, /one championship/i), "ONE history", "Result metadata"], [allResults.metadata_only_bouts || 0, "Metadata-only bouts", "Never included in UFC performance rates"]].forEach(([value, label, note]) => coverageStats.append(statTile(formatNumber(value, 0), label, note))); coverage.append(coverageStats); container.append(coverage);

  const keyStats = element("section", "stats-grid");
  [[formatNumber(fighter.career.sig_strikes_landed_per_minute), "Sig. strikes landed / min", `${formatNumber(fighter.career.sig_strikes_absorbed_per_minute)} absorbed · UFC only`], [formatMetric(fighter.career.significant_strike_differential_per_minute, "signed"), "Sig. strike differential / min", `${formatPercent(fighter.career.sig_strike_defense)} defense · UFC only`], [formatNumber(fighter.career.takedowns_landed_per_15), "Takedowns / 15 min", `${formatPercent(fighter.career.takedown_accuracy)} accuracy · UFC only`], [formatNumber(fighter.career.control_minutes_per_15), "Control min / 15", `${formatPercent(fighter.career.control_share)} control share · UFC only`], [formatNumber(fighter.career.submission_attempts_per_15), "Sub attempts / 15", "UFCStats detail"], [formatNumber(fighter.career.knockdowns_per_15), "Knockdowns / 15", `${formatNumber(fighter.career.knockdowns_absorbed_per_15)} absorbed · UFC only`], [formatPercent(allResults.finish_rate), "Finish rate in wins", `${allResults.ko_tko_wins} KO · ${allResults.submission_wins} SUB · all promotions`], [allResults.recent_form.join(" · ") || "—", "Last five results", allResults.last_fight_date ? `Last fought ${formatDate(allResults.last_fight_date)}` : "No fight date"]].forEach(([value, label, note]) => keyStats.append(statTile(value, label, note))); container.append(keyStats);

  const careerPanel = element("section", "panel"); const heading = element("div", "section-heading"); const copy = element("div"); appendText(copy, "p", "eyebrow", "UFCStats performance"); appendText(copy, "h2", "", "Detailed fighter statistics"); appendText(copy, "p", "section-note", "These rates use UFCStats fight time only. Bellator and ONE metadata affects the all-promotion record above, never these detailed rates."); heading.append(copy); careerPanel.append(heading);
  const careerColumns = element("div", "explain-grid"); ["Record", "Striking", "Grappling", "Style", "Data quality"].forEach((group) => careerColumns.append(renderCareerTable(fighter, group))); careerPanel.append(careerColumns);
  const metadata = document.createElement("details"); metadata.append(element("summary", "", "Career dates, divisions, form, and streak metadata")); const metadataBody = element("div", "details-body");
  const metadataTable = element("table", "data-table"); const metadataRows = document.createElement("tbody");
  [["First linked fight", allResults.first_fight_date ? formatDate(allResults.first_fight_date) : "—"], ["Most recent linked fight", allResults.last_fight_date ? formatDate(allResults.last_fight_date) : "—"], ["Recent form across promotions", allResults.recent_form.join(" · ") || "—"], ["Current W/L streak", allResults.current_streak_result ? `${allResults.current_streak} ${allResults.current_streak_result}` : "—"], ["Promotions", allResults.promotions.map((item) => `${item.name} (${item.bouts})`).join(", ") || "—"], ["UFC divisions", fighter.career.divisions.map((item) => `${item.name} (${item.bouts})`).join(", ") || "—"]].forEach(([label, value]) => { const row = document.createElement("tr"); appendText(row, "td", "", label); appendText(row, "td", "", value); metadataRows.append(row); });
  metadataTable.append(metadataRows); metadataBody.append(metadataTable); metadata.append(metadataBody); careerPanel.append(metadata, renderRawTotals(fighter)); container.append(careerPanel);
  const historyLoading = element("div", "empty-state", "Loading complete fight log…"); container.append(historyLoading);
  try {
    await ensureFighterFights(fighter);
    if (container.isConnected && window.location.hash === `#fighters/${fighter.id}`) historyLoading.replaceWith(renderFightHistory(fighter));
  } catch (error) {
    historyLoading.textContent = `The fight log could not be loaded: ${error.message}`;
  }
}

function renderProfitabilityEvidence() {
  const container = $("#profitability-evidence"); container.replaceChildren();
  const totals = state.performance?.total_rounds;
  if (!totals) {
    container.append(element("div", "empty-state", "The totals decision ledger has not published a performance report yet. It will populate after a successful totals capture reaches the T-24 window."));
    return;
  }
  const official = totals.official_strategy || {};
  const comparison = totals.forecast_comparators || {};
  const clv = totals.latest_available_price_clv || {};
  const residual = totals.next_residual_weight_selection || {};
  const gate = totals.promotion_gate || {};
  const metrics = element("div", "stats-grid");
  [
    [formatNumber(totals.scored_forecasts, 0), "Settled forecasts", `${formatNumber(gate.minimum_scored_lines, 0)} required before promotion`],
    [formatNumber(official.selections, 0), "Official paper selections", "5% residual-EV threshold"],
    [formatPercent(official.hypothetical_roi), "Hypothetical ROI", official.selections ? `${official.wins}-${official.losses} paper record` : "Awaiting selections"],
    [formatPercent(clv.mean_probability_edge), "Mean closing-line value", clv.count ? `${clv.count} same-book comparisons` : "Awaiting a later same-book price"],
    [formatNumber(comparison.residual_minus_market_log_loss, 4), "Residual minus market log loss", "Negative is better than market consensus"],
    [formatPercent(residual.selected_weight), "Independent-model weight", String(residual.selection_status || "market_only_insufficient_history").replaceAll("_", " ")],
  ].forEach(([value, label, note]) => metrics.append(statTile(value, label, note)));
  container.append(metrics);
  appendText(container, "p", "section-note", gate.count_requirements_met
    ? "The sample-count gate is met; calibration, return, and closing-line confidence requirements must also pass before policy promotion. Execution remains disabled."
    : `Evidence collection remains active: ${formatNumber(gate.settled_events, 0)} / ${formatNumber(gate.minimum_settled_events, 0)} events and ${formatNumber(gate.paper_selections, 0)} / ${formatNumber(gate.minimum_paper_selections, 0)} paper selections.`);
  const filtered = state.performance?.bayesian_filtered_moneyline_policy;
  if (filtered) {
    const base = filtered.base_policy_on_same_cohort || {};
    const bayes = filtered.bayesian_filtered_policy || {};
    const paired = filtered.paired_roi_difference || {};
    appendText(container, "h3", "", "Bayesian filter versus the existing moneyline policy");
    const filterMetrics = element("div", "stats-grid");
    [
      [formatNumber(filtered.paired_settled_decisions, 0), "Paired settled decisions", "Same post-deployment cohort"],
      [formatNumber(base.selections, 0), "Existing-policy selections", `${formatPercent(base.hypothetical_roi)} ROI`],
      [formatNumber(bayes.selections, 0), "Selections surviving filter", `${formatPercent(bayes.hypothetical_roi)} ROI`],
      [formatPercent(paired.point_difference), "Filtered minus base ROI", paired.ci_95_lower === null || paired.ci_95_lower === undefined ? "Awaiting multiple settled cards" : `95% interval ${formatPercent(paired.ci_95_lower)} to ${formatPercent(paired.ci_95_upper)}`],
    ].forEach(([value, label, note]) => filterMetrics.append(statTile(value, label, note)));
    container.append(filterMetrics);
    appendText(container, "p", "section-note", "This comparison begins only with immutable T-24 decisions captured after deployment. The filter may veto an existing selection, but it never changes to the opposite fighter. Execution remains disabled.");
  }
  const strategies = totals.shadow_threshold_strategies || {};
  const thresholds = Object.keys(strategies.market_residual || {});
  if (!thresholds.length) return;
  const details = document.createElement("details"); details.append(element("summary", "", "Compare predeclared EV thresholds"));
  const body = element("div", "details-body book-table-wrap"); const table = element("table", "data-table");
  const head = document.createElement("thead"); const header = document.createElement("tr");
  ["Probability source", "EV threshold", "Selections", "Record", "Paper ROI", "Max drawdown"].forEach((value) => appendText(header, "th", "", value)); head.append(header); table.append(head);
  const tbody = document.createElement("tbody");
  [["Independent model", strategies.independent_model], ["Market residual", strategies.market_residual]].forEach(([label, rows]) => {
    thresholds.forEach((threshold) => {
      const values = rows?.[threshold] || {}; const row = document.createElement("tr");
      [label, formatPercent(values.threshold), formatNumber(values.selections, 0), `${values.wins || 0}-${values.losses || 0}`, formatPercent(values.hypothetical_roi), formatNumber(values.hypothetical_max_drawdown_units, 2)].forEach((value) => appendText(row, "td", "", value)); tbody.append(row);
    });
  });
  table.append(tbody); body.append(table); details.append(body); container.append(details);
}

function renderMarket() {
  const notice = $("#market-notice"); const container = $("#market-matchups"); const opportunityContainer = $("#market-opportunities"); const propContainer = $("#prop-market-details"); notice.replaceChildren(); container.replaceChildren(); opportunityContainer.replaceChildren(); propContainer.replaceChildren(); renderProfitabilityEvidence();
  const market = currentMarket();
  const outcomes = currentOutcomes();
  const copy = element("div"); appendText(copy, "h2", "", "Paper research only—automatic betting is intentionally off.");
  const marketNotice = market
    ? `Quotes captured ${formatTimestamp(market.observed_at_utc)}. The policy compares the best available price with a consensus that excludes that target book.`
    : state.market
      ? `The latest stored quotes are for ${formatDate(state.market.event_date)}, not the current ${state.card?.date || "fight card"}. Current-card prices will appear after the next synchronized market snapshot.`
      : "No current book-by-book market capture is published. Fighter and matchup research remains available.";
  appendText(copy, "p", "", marketNotice);
  notice.append(copy, element("span", "pill orange", market ? "Execution disabled" : "Current prices unavailable"));
  const matchups = market?.matchups || [];
  const propMarkets = market?.prop_markets;
  const totalRounds = propMarkets?.total_rounds;
  const methodMarket = propMarkets?.method_of_victory;
  const ranked = [];
  matchups.forEach((matchup) => {
    const signal = matchup.current_signal;
    const expectedReturn = finite(signal?.estimated_expected_return);
    if (signal && expectedReturn !== null && expectedReturn > 0) ranked.push({
      category: "Moneyline",
      matchup: `${matchup.fighter_name} vs ${matchup.opponent_name}`,
      selection: signal.best_candidate_name,
      book: signal.target_book,
      odds: signal.offered_moneyline,
      probability: signal.market_probability,
      expectedReturn,
      thresholdMet: signal.paper_action !== "pass",
      probabilityLabel: "Leave-one-book-out fair probability",
      warning: "Market-relative estimate; the target book is excluded from consensus.",
    });
    const bayesianCandidate = bayesianFilteredCandidate(matchup);
    if (bayesianCandidate && bayesianCandidate.mean > 0) {
      ranked.push({
        category: "Bayesian-filtered moneyline",
        matchup: `${matchup.fighter_name} vs ${matchup.opponent_name}`,
        selection: bayesianCandidate.name,
        book: bayesianCandidate.book,
        odds: bayesianCandidate.odds,
        probability: bayesianCandidate.distribution.mean,
        expectedReturn: bayesianCandidate.mean,
        thresholdMet: bayesianCandidate.qualified,
        decisionLabel: bayesianCandidate.qualified ? "Filter keeps bet" : "Filter vetoes bet",
        filterStatus: bayesianCandidate.status,
        probabilityPositive: bayesianCandidate.probability_positive,
        expectedReturnLower: bayesianCandidate.lower,
        expectedReturnUpper: bayesianCandidate.upper,
        probabilityLabel: "Posterior-mean model probability",
        warning: `${formatPercent(bayesianCandidate.probability_positive)} posterior probability of positive EV; ${formatPercent(bayesianCandidate.lower)} to ${formatPercent(bayesianCandidate.upper)} credible EV interval. Challenger only—no execution.`,
      });
    }
  });
  (totalRounds?.positive_candidates || []).forEach((candidate) => ranked.push({
    category: "Total rounds",
    matchup: `${candidate.fighter_name} vs ${candidate.opponent_name}`,
    selection: candidate.selection,
    book: candidate.target_book,
    odds: candidate.offered_moneyline,
    probability: candidate.model_probability,
    expectedReturn: candidate.estimated_expected_return,
    thresholdMet: candidate.paper_threshold_met,
    probabilityLabel: "Candidate duration-model probability",
    warning: `Candidate model only - ${candidate.scheduled_rounds} scheduled rounds - ${String(candidate.schedule_basis).replaceAll("_", " ")}.`,
  }));
  ranked.sort((left, right) => right.expectedReturn - left.expectedReturn || left.matchup.localeCompare(right.matchup));
  $("#market-opportunity-status").textContent = ranked.length
    ? `${ranked.length} positive-EV price${ranked.length === 1 ? "" : "s"} in the latest synchronized capture. Bayesian-filtered rows begin with an existing-policy selection and may only keep or veto it; all rows remain paper-only.`
    : "No positive-EV price is currently published. This is a valid result, not a data failure.";
  if (!ranked.length) opportunityContainer.append(element("div", "empty-state", totalRounds ? "No current moneyline or total-round price has positive estimated value." : "Finish-time EV is awaiting the next successful totals capture; no current moneyline price is positive EV."));
  ranked.forEach((candidate) => {
    const card = element("article", "opportunity-card");
    const meta = element("div", "opportunity-meta"); meta.append(element("span", "pill neutral", candidate.category), element("span", `pill ${candidate.thresholdMet ? "win" : "orange"}`, candidate.decisionLabel || (candidate.thresholdMet ? "Paper threshold met" : "+EV below threshold"))); card.append(meta);
    appendText(card, "h3", "", candidate.matchup); appendText(card, "p", "signal-reason", `${candidate.selection} at ${candidate.book}`);
    const stats = element("div", "signal-line"); const candidateStats = [[formatOdds(candidate.odds), "Offered price"], [formatPercent(candidate.probability), candidate.probabilityLabel], [formatPercent(candidate.expectedReturn), "Estimated return"]];
    if (finite(candidate.probabilityPositive) !== null) candidateStats.push([formatPercent(candidate.probabilityPositive), "Probability EV is positive"]);
    candidateStats.forEach(([value, label]) => { const item = element("div", "signal-stat"); appendText(item, "strong", "", value); appendText(item, "span", "", label); stats.append(item); });
    const warning = candidate.filterStatus ? `The existing policy selected this same side and price. Bayesian filter: ${String(candidate.filterStatus).replaceAll("_", " ")}. ${candidate.warning}` : candidate.warning;
    card.append(stats, element("div", "candidate-warning", warning)); opportunityContainer.append(card);
  });

  const totalStatus = totalRounds?.price_status === "available" ? `${totalRounds.quote_count} book/line quotes and ${totalRounds.forecast_count} frozen model probabilities.` : "Awaiting the next successful total-round odds capture.";
  const methodStatus = methodMarket?.expected_value_status === "available" ? "Method-of-victory EV is available." : "Method probabilities exist, but method EV is unavailable because the configured provider supplies no method prices.";
  $("#prop-coverage-status").textContent = `${totalStatus} ${methodStatus}`;
  if (!(totalRounds?.markets || []).length) propContainer.append(element("div", "empty-state", "No synchronized total-round lines are published yet. The next market snapshot will populate this section when books expose totals."));
  (totalRounds?.markets || []).forEach((market) => {
    const card = element("article", "market-card"); const best = market.best_candidate;
    appendText(card, "h3", "", `${market.fighter_name} vs ${market.opponent_name}`);
    appendText(card, "p", "signal-reason", `Full fight total: ${formatNumber(market.line, 1)} rounds - ${market.eligible_quote_count}/${market.quote_count} fresh book lines`);
    if (best) {
      const stats = element("div", "signal-line"); [[best.selection, "Best side"], [`${formatOdds(best.offered_moneyline)} - ${best.target_book}`, "Best price"], [formatPercent(best.estimated_expected_return), "Candidate model EV"]].forEach(([value, label]) => { const item = element("div", "signal-stat"); appendText(item, "strong", "", value); appendText(item, "span", "", label); stats.append(item); }); card.append(stats);
    } else appendText(card, "p", "signal-reason", market.forecast_unavailable_reason || "No valid candidate price.");
    if (market.locked_t24_decision) {
      const locked = market.locked_t24_decision; const lockedDetails = document.createElement("details"); lockedDetails.append(element("summary", "", "Locked T-24 residual paper decision"));
      const lockedBody = element("div", "details-body"); const lockedTable = element("table", "data-table"); const lockedRows = document.createElement("tbody");
      [["Captured", formatTimestamp(locked.captured_at_utc)], ["Paper action", locked.paper_action], ["Selection", locked.selection || "Pass"], ["Target book", locked.target_book], ["Offered price", formatOdds(locked.offered_moneyline)], ["Market over probability", formatPercent(locked.market_over_probability)], ["Independent-model over probability", formatPercent(locked.model_over_probability)], ["Residual over probability", formatPercent(locked.residual_over_probability)], ["Independent-model weight", formatPercent(locked.selected_residual_weight)], ["Estimated return", formatPercent(locked.estimated_expected_return)], ["Residual status", String(locked.residual_selection_status).replaceAll("_", " ")]].forEach(([label, value]) => { const row = document.createElement("tr"); appendText(row, "td", "", label); appendText(row, "td", "", value); lockedRows.append(row); });
      lockedTable.append(lockedRows); lockedBody.append(lockedTable); lockedDetails.append(lockedBody); card.append(lockedDetails);
    }
    const details = document.createElement("details"); details.append(element("summary", "", `All ${market.book_quotes.length} total prices`)); const body = element("div", "details-body book-table-wrap"); const table = element("table", "data-table"); const head = document.createElement("thead"); const header = document.createElement("tr"); ["Book", "Over", "Under", "No-vig over", "Quote age"].forEach((value) => appendText(header, "th", "", value)); head.append(header); table.append(head); const tbody = document.createElement("tbody"); market.book_quotes.forEach((quote) => { const row = document.createElement("tr"); [quote.book, formatOdds(quote.over_moneyline), formatOdds(quote.under_moneyline), formatPercent(quote.no_vig_over_probability), `${formatNumber(quote.source_quote_age_seconds, 0)}s`].forEach((value) => appendText(row, "td", "", value)); tbody.append(row); }); table.append(tbody); body.append(table); details.append(body); card.append(details); propContainer.append(card);
  });
  const outcomeMatchups = (outcomes?.matchups || []).filter((item) => item.matchup_id && item.terminal_probabilities);
  if (outcomeMatchups.length) {
    const methodCard = element("article", "market-card wide-card"); appendText(methodCard, "h3", "", "Candidate method-of-victory probabilities"); appendText(methodCard, "p", "signal-reason", "These are model probabilities, not expected values. A book-specific method price is required before any row can enter the positive-EV list.");
    const details = document.createElement("details"); details.append(element("summary", "", `View ${outcomeMatchups.length} matchup forecasts`)); const body = element("div", "details-body book-table-wrap"); const table = element("table", "data-table"); const head = document.createElement("thead"); const header = document.createElement("tr"); ["Matchup", "Exact outcome", "Model probability"].forEach((value) => appendText(header, "th", "", value)); head.append(header); table.append(head); const tbody = document.createElement("tbody");
    outcomeMatchups.forEach((matchup) => Object.entries(matchup.terminal_probabilities).sort((left, right) => right[1] - left[1]).forEach(([outcome, probability]) => { const side = outcome.startsWith("fighter_") ? matchup.fighter_name : matchup.opponent_name; const method = outcome.replace(/^fighter_|^opponent_/, "").replaceAll("_", " ").replace("ko tko", "KO/TKO"); const row = document.createElement("tr"); [ `${matchup.fighter_name} vs ${matchup.opponent_name}`, `${side} by ${method}`, formatPercent(probability) ].forEach((value) => appendText(row, "td", "", value)); tbody.append(row); }));
    table.append(tbody); body.append(table); details.append(body); methodCard.append(details); propContainer.append(methodCard);
  }
  if (!matchups.length) { container.append(element("div", "empty-state", "Run a successful market snapshot to publish current book lines and paper decisions.")); return; }
  matchups.forEach((matchup) => {
    const signal = matchup.current_signal; const card = element("article", "market-card");
    card.dataset.marketFighterId = String(matchup.fighter_id || "");
    card.dataset.marketOpponentId = String(matchup.opponent_id || "");
    card.tabIndex = -1;
    const row = element("div", "fighter-row"); appendText(row, "h3", "", `${matchup.fighter_name} vs ${matchup.opponent_name}`);
    const hasSelection = signal && signal.paper_action !== "pass";
    row.append(element("span", `pill ${hasSelection ? "win" : "neutral"}`, hasSelection ? "Paper bet" : "Pass")); card.append(row);
    if (signal) {
      const stats = element("div", "signal-line");
      [[formatOdds(signal.offered_moneyline), `${signal.best_candidate_name} at ${signal.target_book}`], [formatOdds(signal.market_fair_moneyline), "Leave-one-book-out fair line"], [formatPercent(signal.estimated_expected_return), "Estimated return"]].forEach(([value, label]) => { const item = element("div", "signal-stat"); appendText(item, "strong", "", value); appendText(item, "span", "", label); stats.append(item); }); card.append(stats);
      appendText(card, "p", "signal-reason", signal.reason);
      appendText(card, "p", "signal-reason", `Consensus: ${signal.consensus_book_count} books · ${formatPercent(signal.market_probability)} fair probability · model weight ${formatPercent(signal.model_weight)}.`);
    } else appendText(card, "p", "signal-reason", matchup.current_signal_unavailable_reason || "No evaluable paper signal for this matchup.");
    const bayesianForecast = bayesianForMatchup(matchup); const bayesianCandidate = bestBayesianCandidate(matchup); const bayesianFiltered = bayesianFilteredCandidate(matchup);
    if (bayesianForecast) {
      const bayesianDetails = document.createElement("details"); bayesianDetails.append(element("summary", "", "Bayesian model and expected-return uncertainty"));
      const bayesianBody = element("div", "details-body"); const bayesianTable = element("table", "data-table"); const bayesianRows = document.createElement("tbody");
      const posteriorRows = [
        ["Posterior mean", `${matchup.fighter_name} ${formatPercent(bayesianForecast.mean)} / ${matchup.opponent_name} ${formatPercent(1 - bayesianForecast.mean)}`],
        [`${formatPercent(bayesianForecast.credible_level, 0)} credible interval`, `${matchup.fighter_name} ${formatPercent(bayesianForecast.lower)}-${formatPercent(bayesianForecast.upper)}`],
        ["Model status", bayesianForecast.status === "paper_only_challenger" ? "Paper-only Laplace challenger; calibration uncertainty is not included" : "EV abstention: insufficient fighter history to represent input uncertainty"],
      ];
      if (bayesianCandidate) posteriorRows.push(
        ["Best current model-priced side", `${bayesianCandidate.name} ${formatOdds(bayesianCandidate.odds)} at ${bayesianCandidate.book}`],
        ["Posterior mean EV", formatPercent(bayesianCandidate.mean)],
        ["Credible EV interval", `${formatPercent(bayesianCandidate.lower)} to ${formatPercent(bayesianCandidate.upper)}`],
        ["Probability EV is positive", formatPercent(bayesianCandidate.probability_positive)],
      );
      if (bayesianFiltered) posteriorRows.push(
        ["Existing-policy candidate", `${bayesianFiltered.name} ${formatOdds(bayesianFiltered.odds)} at ${bayesianFiltered.book}`],
        ["Bayesian-filtered action", bayesianFiltered.qualified ? `Keep ${bayesianFiltered.name}` : `Veto · ${String(bayesianFiltered.status).replaceAll("_", " ")}`],
        ["Bayesian EV at existing price", formatPercent(bayesianFiltered.mean)],
        ["Probability EV is positive at existing price", formatPercent(bayesianFiltered.probability_positive)],
      );
      posteriorRows.forEach(([label, value]) => { const posteriorRow = document.createElement("tr"); appendText(posteriorRow, "td", "", label); appendText(posteriorRow, "td", "", value); bayesianRows.append(posteriorRow); });
      bayesianTable.append(bayesianRows); bayesianBody.append(bayesianTable); bayesianDetails.append(bayesianBody); card.append(bayesianDetails);
    }
    const details = document.createElement("details"); details.dataset.bookLines = "moneyline"; details.append(element("summary", "", `All ${matchup.book_quotes.length} book lines`));
    const body = element("div", "details-body book-table-wrap"); const table = element("table", "data-table");
    const head = document.createElement("thead"); const header = document.createElement("tr"); ["Book", matchup.fighter_name, matchup.opponent_name, "Quote age", "Consensus"].forEach((value) => appendText(header, "th", "", value)); head.append(header); table.append(head);
    const tbody = document.createElement("tbody"); matchup.book_quotes.forEach((quote) => { const quoteRow = document.createElement("tr"); appendText(quoteRow, "td", "", quote.book); appendText(quoteRow, "td", "", formatOdds(quote.fighter_moneyline)); appendText(quoteRow, "td", "", formatOdds(quote.opponent_moneyline)); appendText(quoteRow, "td", "", `${formatNumber(quote.source_quote_age_seconds, 0)}s`); appendText(quoteRow, "td", "", quote.eligible_for_consensus ? "Yes" : "No"); tbody.append(quoteRow); }); table.append(tbody); body.append(table); details.append(body); card.append(details);
    if (matchup.locked_t24_decision) {
      const locked = matchup.locked_t24_decision; const lockedDetails = document.createElement("details"); lockedDetails.append(element("summary", "", "Locked T-24 paper decision"));
      const lockedBody = element("div", "details-body"); const lockedTable = element("table", "data-table"); const lockedRows = document.createElement("tbody");
      [["Captured", formatTimestamp(locked.observed_at_utc)], ["Candidate", locked.best_candidate_name || "—"], ["Target book", locked.target_book || "—"], ["Offered line", formatOdds(locked.offered_moneyline)], ["Leave-one-book-out fair line", formatOdds(locked.market_fair_moneyline)], ["Estimated return", formatPercent(locked.estimated_expected_return)], ["Paper action", locked.paper_action], ["Reason", locked.reason]].forEach(([label, value]) => { const lockedRow = document.createElement("tr"); appendText(lockedRow, "td", "", label); appendText(lockedRow, "td", "", value); lockedRows.append(lockedRow); });
      lockedTable.append(lockedRows); lockedBody.append(lockedTable); lockedDetails.append(lockedBody); card.append(lockedDetails);
    }
    if (matchup.locked_t24_bayesian_filter) {
      const filtered = matchup.locked_t24_bayesian_filter; const filteredDetails = document.createElement("details"); filteredDetails.append(element("summary", "", "Locked T-24 Bayesian filter"));
      const filteredBody = element("div", "details-body"); const filteredTable = element("table", "data-table"); const filteredRows = document.createElement("tbody");
      [["Base paper action", filtered.base_paper_action], ["Filtered action", filtered.filtered_paper_action], ["Filter result", String(filtered.filter_status).replaceAll("_", " ")], ["Same frozen price", formatOdds(filtered.candidate_moneyline)], ["Posterior mean probability", formatPercent(filtered.candidate_posterior_mean_probability)], ["Posterior mean EV", formatPercent(filtered.candidate_posterior_mean_expected_return)], ["Credible EV interval", `${formatPercent(filtered.candidate_expected_return_lower)} to ${formatPercent(filtered.candidate_expected_return_upper)}`], ["Probability EV is positive", formatPercent(filtered.candidate_probability_positive_expected_return)]].forEach(([label, value]) => { const filteredRow = document.createElement("tr"); appendText(filteredRow, "td", "", label); appendText(filteredRow, "td", "", value); filteredRows.append(filteredRow); });
      filteredTable.append(filteredRows); filteredBody.append(filteredTable); filteredDetails.append(filteredBody); card.append(filteredDetails);
    }
    const fighter = state.fighterById.get(matchup.fighter_id); const opponent = state.fighterById.get(matchup.opponent_id);
    if (fighter && opponent) card.append(actionButton("Research fighter matchup", "secondary-button", () => setRoute(`matchups/${fighter.id}/${opponent.id}`))); container.append(card);
  });
}

function explainCard(title, copy) {
  const card = element("article", "explain-card"); appendText(card, "h2", "", title); appendText(card, "p", "", copy); return card;
}

function renderModelData() {
  const container = $("#model-data-content"); container.replaceChildren(); const grid = element("section", "explain-grid");
  const model = state.model; const evaluation = model?.temporal_evaluation?.calibrated_model;
  const modelCard = explainCard("The prediction model", model ? `Yes—the current predictor is ${String(model.model_type || "logistic regression").replaceAll("_", " ")}. It uses ${model.feature_columns?.length || 0} point-in-time features trained only on information available before each fight. Calibration adjusts the raw probabilities before publication.` : "The model artifact is not currently published.");
  if (evaluation) { const metrics = element("div", "metric-row"); [[formatPercent(evaluation.accuracy), "holdout accuracy"], [formatNumber(evaluation.log_loss, 3), "holdout log loss"], [formatNumber(evaluation.auc, 3), "holdout AUC"]].forEach(([value, label]) => { const stat = element("div", "mini-stat"); appendText(stat, "strong", "", value); appendText(stat, "span", "", label); metrics.append(stat); }); modelCard.append(metrics); }
  grid.append(modelCard);
  const bayesianEvaluation = state.bayesian?.temporal_evaluation; const bayesianWalk = bayesianEvaluation?.walk_forward?.aggregate; const bayesianComparison = bayesianEvaluation?.comparison_to_point_model; const bayesianGate = bayesianEvaluation?.evidence_gate; const bayesianProspective = state.performance?.bayesian_moneyline_challenger; const bayesianFilter = state.performance?.bayesian_filtered_moneyline_policy;
  const bayesianCard = explainCard("Bayesian logistic challenger", state.bayesian ? "The challenger places a Gaussian posterior around the regularized logistic coefficients using a Laplace approximation. Each fight receives a posterior probability interval and a distribution of expected return at the offered price. It remains paper-only." : "The Bayesian challenger artifact is not currently published.");
  if (bayesianWalk) { const bayesianMetrics = element("div", "metric-row"); [[formatNumber(bayesianWalk.log_loss, 3), "walk-forward log loss"], [formatNumber(bayesianWalk.brier, 3), "walk-forward Brier"], [formatPercent(bayesianWalk.mean_90_probability_interval_width), "mean 90% interval width"], [formatNumber(bayesianComparison?.walk_forward_log_loss_delta_vs_point, 4), "log-loss delta vs point model"]].forEach(([value, label]) => { const stat = element("div", "mini-stat"); appendText(stat, "strong", "", value); appendText(stat, "span", "", label); bayesianMetrics.append(stat); }); bayesianCard.append(bayesianMetrics); }
  if (bayesianProspective) { const prospectiveMetrics = element("div", "metric-row"); [[formatNumber(bayesianProspective.scored_forecasts, 0), "prospectively scored fights"], [formatNumber(bayesianProspective.settled_shadow_selections, 0), "settled shadow selections"], [formatPercent(bayesianProspective.hypothetical_roi), "shadow ROI"], [`${bayesianProspective.wins || 0}-${bayesianProspective.losses || 0}`, "shadow record"]].forEach(([value, label]) => { const stat = element("div", "mini-stat"); appendText(stat, "strong", "", value); appendText(stat, "span", "", label); prospectiveMetrics.append(stat); }); bayesianCard.append(prospectiveMetrics); appendText(bayesianCard, "p", "section-note", bayesianProspective.source_limit); }
  if (bayesianFilter) { const filteredMetrics = element("div", "metric-row"); [[formatNumber(bayesianFilter.paired_settled_decisions, 0), "immutable paired decisions"], [formatNumber(bayesianFilter.bayesian_filtered_policy?.selections, 0), "selections surviving veto"], [formatPercent(bayesianFilter.bayesian_filtered_policy?.hypothetical_roi), "filtered ROI"], [formatPercent(bayesianFilter.paired_roi_difference?.point_difference), "ROI delta vs existing policy"]].forEach(([value, label]) => { const stat = element("div", "mini-stat"); appendText(stat, "strong", "", value); appendText(stat, "span", "", label); filteredMetrics.append(stat); }); bayesianCard.append(filteredMetrics); appendText(bayesianCard, "p", "section-note", "The immutable T-24 filter starts with an existing-policy selection and only keeps or vetoes that same side and price."); }
  if (bayesianGate) appendText(bayesianCard, "p", "section-note", `Evidence gate: ${String(bayesianGate.status).replaceAll("_", " ")}. Prospective CLV and return requirements are not met; execution is disabled.`);
  grid.append(bayesianCard);
  const market = currentMarket();
  const marketWeight = finite(market?.model_weight);
  const marketCard = explainCard("How bets are informed", market ? `The price policy finds the best offered line, then estimates fair probability from the other eligible books. The target book is excluded to avoid grading its price against itself. The model currently receives ${formatPercent(marketWeight)} weight${marketWeight === 0 ? " because prospective market-relative evidence is still being collected" : " in the blended estimate"}.` : "Current-card market policy output is unavailable, so no live paper signal is shown.");
  if (state.performance?.promotion_gate) appendText(marketCard, "p", "section-note", `Promotion gate: ${String(state.performance.promotion_gate.status).replaceAll("_", " ")} · ${state.performance.promotion_gate.paper_selections} / ${state.performance.promotion_gate.minimum_paper_selections} minimum paper selections.`); grid.append(marketCard);
  const dataCard = explainCard("Dataset coverage", `${state.explorer.counts.fighters.toLocaleString()} profiles and ${state.explorer.counts.unique_fights.toLocaleString()} unique fights are published through ${formatDate(state.explorer.data_through)}. Stable UFCStats URL IDs join fighters, opponents, fights, and events; display names are not used as identity keys.`);
  const dataMetrics = element("div", "metric-row"); [[state.explorer.counts.fighters_with_recorded_bouts.toLocaleString(), "profiles with bouts"], [state.explorer.counts.fighter_fight_rows.toLocaleString(), "fighter stat rows"], [state.explorer.fight_columns.length.toLocaleString(), "fields per fight"]].forEach(([value, label]) => { const stat = element("div", "mini-stat"); appendText(stat, "strong", "", value); appendText(stat, "span", "", label); dataMetrics.append(stat); }); dataCard.append(dataMetrics); grid.append(dataCard);
  const rulesCard = explainCard("Sanitization rules", "The publication is rebuilt deterministically from processed CSV files and rejected if it cannot be reproduced. Duplicate fighter perspectives, conflicting IDs, invalid numbers, and oversized output fail validation.");
  const list = element("ul", "provenance-list"); state.explorer.data_dictionary.notes.forEach((note) => appendText(list, "li", "", note)); rulesCard.append(list); grid.append(rulesCard); container.append(grid);

  const dictionary = element("section", "panel"); const heading = element("div", "section-heading"); const headingCopy = element("div"); appendText(headingCopy, "p", "eyebrow", "Definitions"); appendText(headingCopy, "h2", "", "Published data dictionary"); appendText(headingCopy, "p", "section-note", "Every career and bout-level statistic exposed by the explorer."); heading.append(headingCopy); dictionary.append(heading);
  const columns = element("div", "data-columns");
  [["Career statistics", state.explorer.data_dictionary.career], ["Per-fight statistics", state.explorer.data_dictionary.fight_stats]].forEach(([title, definitions]) => { const section = element("div"); appendText(section, "h3", "", title); const table = element("table", "data-table"); const tbody = document.createElement("tbody"); Object.entries(definitions).forEach(([key, definition]) => { const row = document.createElement("tr"); appendText(row, "td", "", definition.label); appendText(row, "td", "", definition.group); appendText(row, "td", "", definition.format || definition.unit); tbody.append(row); }); table.append(tbody); section.append(table); columns.append(section); }); dictionary.append(columns);
  const provenance = document.createElement("details"); provenance.append(element("summary", "", "Publication identity and integrity metadata")); const body = element("div", "details-body"); appendText(body, "p", "", `Identity contract: ${state.explorer.identity_contract}`); appendText(body, "p", "hash", `SHA-256: ${state.explorer.publication_sha256}`); if (model?.model_id) appendText(body, "p", "hash", `Model ID: ${model.model_id} · trained through ${model.training_labels_through || model.data_through}`); if (state.bayesian?.model_id) appendText(body, "p", "hash", `Bayesian challenger ID: ${state.bayesian.model_id} · base model ${state.bayesian.base_model_id}`); provenance.append(body); dictionary.append(provenance); container.append(dictionary);
}

function bindEvents() {
  document.querySelectorAll("[data-nav]").forEach((button) => button.addEventListener("click", () => setRoute(button.dataset.nav)));
  makeAutocomplete($("#matchup-fighter-a"), $("#matchup-results-a"), "a"); makeAutocomplete($("#matchup-fighter-b"), $("#matchup-results-b"), "b");
  $("#analyze-matchup").addEventListener("click", () => { if (state.selected.a && state.selected.b) setRoute(`matchups/${state.selected.a.id}/${state.selected.b.id}`); });
  $("#clear-matchup").addEventListener("click", clearMatchup);
  $("#graph-apply").addEventListener("click", drawFightGraph);
  $("#graph-reset").addEventListener("click", resetFightGraph);
  $("#graph-mode-simple").addEventListener("click", () => setGraphFilterMode("simple"));
  $("#graph-mode-advanced").addEventListener("click", () => setGraphFilterMode("advanced"));
  $("#graph-mode-matchup").addEventListener("click", () => setGraphFilterMode("matchup"));
  document.querySelectorAll("[data-graph-years]").forEach((button) => button.addEventListener("click", () => applyGraphQuickRange(button.dataset.graphYears)));
  $("#graph-apply-custom-years").addEventListener("click", () => applyGraphQuickRange($("#graph-custom-years").value, true));
  $("#graph-custom-years").addEventListener("keydown", (event) => { if (event.key === "Enter") applyGraphQuickRange(event.currentTarget.value, true); });
  ["#graph-start-date", "#graph-end-date"].forEach((selector) => $(selector).addEventListener("change", () => selectGraphQuickRange("")));
  $("#graph-add-rule").addEventListener("click", () => { addGraphRule(); markGraphFiltersDirty(); });
  $("#graph-clear-rules").addEventListener("click", clearGraphRules);
  document.querySelectorAll("[data-graph-preset]").forEach((button) => button.addEventListener("click", () => applyGraphFilterPreset(button.dataset.graphPreset)));
  ["#graph-division", "#graph-promotion", "#graph-start-date", "#graph-end-date", "#graph-min-fights", "#graph-rule-join", "#graph-rule-scope", "#graph-rule-stance", "#graph-fight-method", "#graph-fight-round", "#graph-fight-detail", "#graph-matchup-fighter-a", "#graph-matchup-fighter-b", "#graph-matchup-depth"].forEach((selector) => $(selector).addEventListener("change", markGraphFiltersDirty));
  $("#graph-fighter-search").addEventListener("input", markGraphFiltersDirty);
  $("#graph-zoom-out").addEventListener("click", () => zoomFightGraph(1.25));
  $("#graph-zoom-in").addEventListener("click", () => zoomFightGraph(0.8));
  $("#graph-zoom-fit").addEventListener("click", fitFightGraph);
  $("#graph-fighter-search").addEventListener("keydown", (event) => { if (event.key === "Enter") drawFightGraph(); });
  ["#fighter-directory-search", "#division-filter", "#stance-filter", "#recorded-only"].forEach((selector) => { const input = $(selector); input.addEventListener(input.tagName === "INPUT" && input.type === "search" ? "input" : "change", () => { state.directoryLimit = 48; renderFighterDirectory(); }); });
  document.addEventListener("click", (event) => { ["a", "b"].forEach((side) => { const picker = $(`[data-picker="${side}"]`); if (!picker.contains(event.target)) closeAutocomplete($(`#matchup-fighter-${side}`), $(`#matchup-results-${side}`)); }); });
  window.addEventListener("hashchange", applyRoute);
}

async function start() {
  try {
    await loadData();
    renderCoverage(); populateFilters(); renderCurrentCard(); renderFighterDirectory(); renderMarket(); renderModelData(); bindEvents();
    $("#publication-stamp").textContent = `Dataset through ${formatDate(state.explorer.data_through)} · schema v${state.explorer.schema_version}`;
    const status = $("#header-status"); status.classList.add("is-ready"); status.lastChild.textContent = " Data ready";
    $("#load-message").hidden = true; applyRoute();
  } catch (error) {
    console.error(error);
    const message = $("#load-message"); message.classList.add("is-error"); message.replaceChildren(element("span", "", `The data explorer could not start: ${error.message}`));
    const status = $("#header-status"); status.classList.add("is-error"); status.lastChild.textContent = " Data error";
  }
}

start();
