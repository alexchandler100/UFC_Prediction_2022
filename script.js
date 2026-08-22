"use strict";

const DATA_PATHS = {
  explorer: "src/content/data/external/fighter_explorer.json",
  vegas: "src/content/data/external/vegas_odds.json",
  card: "src/content/data/external/card_info.json",
  model: "src/content/data/external/winner_model.json",
  market: "src/content/data/market/current_opportunities.json",
  performance: "src/content/data/market/performance_report.json",
};

const state = {
  explorer: null,
  vegas: null,
  card: null,
  model: null,
  market: null,
  performance: null,
  fighters: [],
  fighterById: new Map(),
  fightColumn: new Map(),
  shardCache: new Map(),
  selected: { a: null, b: null },
  directoryLimit: 48,
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

function formatDate(value, options = { year: "numeric", month: "short", day: "numeric" }) {
  if (!value) return "Unknown";
  const date = new Date(`${String(value).slice(0, 10)}T12:00:00Z`);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleDateString(undefined, { ...options, timeZone: "UTC" });
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
  const career = fighter.career;
  let value = `${career.wins}-${career.losses}-${career.draws}`;
  if (career.no_contests) value += ` (${career.no_contests} NC)`;
  return value;
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
  const [explorer, vegas, card, model, market, performance] = await Promise.all([
    fetchJson(DATA_PATHS.explorer),
    fetchJson(DATA_PATHS.vegas, false),
    fetchJson(DATA_PATHS.card, false),
    fetchJson(DATA_PATHS.model, false),
    fetchJson(DATA_PATHS.market, false),
    fetchJson(DATA_PATHS.performance, false),
  ]);
  state.explorer = explorer;
  state.vegas = vegas;
  state.card = card;
  state.model = model;
  state.market = market;
  state.performance = performance;
  state.fighters = explorer.fighters;
  state.fighterById = new Map(state.fighters.map((fighter) => [fighter.id, fighter]));
  state.fightColumn = new Map(explorer.fight_columns.map((column, index) => [column, index]));
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
      return rightStarts - leftStarts || right.career.recorded_bouts - left.career.recorded_bouts || left.name.localeCompare(right.name);
    })
    .slice(0, limit);
}

function setRoute(route) {
  const hash = `#${route}`;
  if (window.location.hash === hash) applyRoute();
  else window.location.hash = hash;
}

function showView(name) {
  document.querySelectorAll("[data-view]").forEach((view) => view.classList.toggle("is-active", view.dataset.view === name));
  document.querySelectorAll("[data-nav]").forEach((button) => button.classList.toggle("is-active", button.dataset.nav === name));
  document.title = `${name === "matchups" ? "Matchups" : name === "fighters" ? "Fighters" : name === "market" ? "Market" : "Model & data"} · UFC Data Lab`;
}

function applyRoute() {
  if (!state.explorer) return;
  const parts = window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const view = ["matchups", "fighters", "market", "data"].includes(parts[0]) ? parts[0] : "matchups";
  showView(view);

  if (view === "fighters" && parts[1]) renderFighterProfile(parts[1]);
  else if (view === "fighters") showFighterDirectory();

  if (view === "matchups" && parts[1] && parts[2]) {
    const fighterA = state.fighterById.get(parts[1]);
    const fighterB = state.fighterById.get(parts[2]);
    if (fighterA && fighterB) {
      selectMatchupFighter("a", fighterA);
      selectMatchupFighter("b", fighterB);
      renderMatchup(fighterA, fighterB);
    }
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderCoverage() {
  const container = $("#coverage-summary");
  container.replaceChildren();
  const values = [
    [state.explorer.counts.fighters.toLocaleString(), "fighter profiles"],
    [state.explorer.counts.unique_fights.toLocaleString(), "unique fights"],
    [state.explorer.counts.fighter_fight_rows.toLocaleString(), "fighter stat lines"],
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
  if (!state.vegas?.["fighter name"]) return [];
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

function currentMatchups() {
  return state.market?.matchups?.length ? state.market.matchups : legacyRows();
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
    appendText(top, "span", "", `${Math.min(fighter?.career.recorded_bouts || 0, opponent?.career.recorded_bouts || 0)}-bout minimum sample`);
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

    const actions = element("div", "card-actions");
    const analyze = actionButton("Research matchup", "primary-button small-button", () => {
      if (fighter && opponent) setRoute(`matchups/${fighter.id}/${opponent.id}`);
    });
    analyze.disabled = !fighter || !opponent;
    const marketButton = actionButton("View prices", "text-button small-button", () => setRoute("market"));
    actions.append(analyze, marketButton);
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
    ["Recorded UFC bouts", (f) => f.career.recorded_bouts, "integer", "higher"],
    ["Average fight time", (f) => f.career.average_fight_minutes, "minutes", "context"],
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
    ["Win rate", (f) => f.career.win_rate, "percentage", "higher"],
    ["Finish rate in wins", (f) => f.career.finish_rate, "percentage", "context"],
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

  const finishLeader = strongerName(fighterA, a.finish_rate, fighterB, b.finish_rate);
  insights.push(["Finishing profile", finishLeader ? `${finishLeader} finishes a larger share of wins` : "Similar or incomplete finish rates", `${fighterA.name}: ${a.ko_tko_wins} KO/TKO, ${a.submission_wins} submissions. ${fighterB.name}: ${b.ko_tko_wins} KO/TKO, ${b.submission_wins} submissions.`, false]);

  const lastA = daysSince(a.last_fight_date); const lastB = daysSince(b.last_fight_date);
  const activityLeader = strongerName(fighterA, lastA, fighterB, lastB, true);
  insights.push(["Recency", activityLeader ? `${activityLeader} fought more recently` : "Similar or incomplete activity data", `${fighterA.name}: ${a.last_fight_date ? formatDate(a.last_fight_date) : "unknown"}. ${fighterB.name}: ${b.last_fight_date ? formatDate(b.last_fight_date) : "unknown"}.`, false]);
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
  stances.forEach((value) => { const option = element("option", "", value); option.value = value; $("#stance-filter").append(option); });
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
      && (!recordedOnly || fighter.career.recorded_bouts > 0);
  }).sort((a, b) => b.career.recorded_bouts - a.career.recorded_bouts || a.name.localeCompare(b.name));
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
  const heading = element("div", "section-heading"); const copy = element("div"); appendText(copy, "p", "eyebrow", "Bout-level data"); appendText(copy, "h2", "", "Complete UFC fight log");
  appendText(copy, "p", "section-note", "Expand any bout for all striking, target, position, grappling, timing, and source fields for both fighters."); heading.append(copy); panel.append(heading);
  const history = element("div", "fight-history");
  if (!fighter.fights.length) history.append(element("div", "empty-state", "No UFC fight statistics are recorded for this fighter."));
  fighter.fights.forEach((values) => {
    const fight = decodeFight(values); const details = document.createElement("details");
    const summary = element("summary", "fight-summary"); appendText(summary, "time", "", formatDate(fight.date, { year: "numeric", month: "short", day: "numeric" }));
    appendText(summary, "span", `result ${String(fight.result).toLowerCase() === "w" ? "win" : "loss"}`, fight.result || "—");
    const opponentName = element("strong", "", fight.opponent_name); summary.append(opponentName);
    appendText(summary, "span", "", `${fight.method || "Method unavailable"} · R${fight.round || "—"} ${fight.time || ""}`);
    appendText(summary, "small", "", fight.division || "Division unavailable"); details.append(summary);
    const body = element("div", "details-body", "Open to load complete bout statistics."); details.append(body);
    let rendered = false; details.addEventListener("toggle", async () => {
      if (!details.open || rendered) return;
      rendered = true; body.textContent = "Loading paired opponent statistics…";
      try {
        const opponentFight = await pairedFight(fight);
        renderFightDetails(body, fight, opponentFight, fighter.name);
      } catch (error) {
        body.textContent = `Could not load paired opponent statistics: ${error.message}`;
      }
    }); history.append(details);
  });
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
  const header = element("section", "profile-header"); const identity = element("div"); identity.append(actionButton("← Back to fighter directory", "back-button", () => setRoute("fighters")));
  appendText(identity, "p", "eyebrow", fighterDivision(fighter) || "Fighter profile"); appendText(identity, "h2", "", fighter.name);
  appendText(identity, "p", "", `${record(fighter)} recorded UFC record · ${fighter.career.recorded_bouts} bouts · ${formatNumber(fighter.career.total_fight_minutes, 1)} recorded minutes`); header.append(identity);
  const sourceLink = element("a", "", "Open official UFCStats profile ↗"); sourceLink.href = fighter.url; sourceLink.target = "_blank"; sourceLink.rel = "noreferrer"; identity.append(sourceLink);
  const bio = element("div", "profile-bio");
  [[fighter.height || "—", "Height"], [fighter.reach || "—", "Reach"], [fighter.stance || "—", "Stance"], [ageOn(fighter, state.card?.date) ?? "—", "Age at current card"], [fighter.dob_iso ? formatDate(fighter.dob_iso) : fighter.dob || "—", "Date of birth"], [fighter.id, "Stable fighter ID"]].forEach(([value, label]) => bio.append(bioItem(value, label))); header.append(bio); container.append(header);

  const keyStats = element("section", "stats-grid");
  [[formatNumber(fighter.career.sig_strikes_landed_per_minute), "Sig. strikes landed / min", `${formatNumber(fighter.career.sig_strikes_absorbed_per_minute)} absorbed`], [formatMetric(fighter.career.significant_strike_differential_per_minute, "signed"), "Sig. strike differential / min", `${formatPercent(fighter.career.sig_strike_defense)} defense`], [formatNumber(fighter.career.takedowns_landed_per_15), "Takedowns / 15 min", `${formatPercent(fighter.career.takedown_accuracy)} accuracy`], [formatNumber(fighter.career.control_minutes_per_15), "Control min / 15", `${formatPercent(fighter.career.control_share)} control share`], [formatNumber(fighter.career.submission_attempts_per_15), "Sub attempts / 15", `${fighter.career.submission_wins} submission wins`], [formatNumber(fighter.career.knockdowns_per_15), "Knockdowns / 15", `${formatNumber(fighter.career.knockdowns_absorbed_per_15)} absorbed`], [formatPercent(fighter.career.finish_rate), "Finish rate in wins", `${fighter.career.ko_tko_wins} KO · ${fighter.career.submission_wins} SUB`], [fighter.career.recent_form.join(" · ") || "—", "Last five results", fighter.career.last_fight_date ? `Last fought ${formatDate(fighter.career.last_fight_date)}` : "No fight date"]].forEach(([value, label, note]) => keyStats.append(statTile(value, label, note))); container.append(keyStats);

  const careerPanel = element("section", "panel"); const heading = element("div", "section-heading"); const copy = element("div"); appendText(copy, "p", "eyebrow", "Career rates"); appendText(copy, "h2", "", "All derived fighter statistics"); appendText(copy, "p", "section-note", "Rates use recorded UFC fight time. Missing denominators remain missing rather than becoming zero."); heading.append(copy); careerPanel.append(heading);
  const careerColumns = element("div", "explain-grid"); ["Record", "Striking", "Grappling", "Style", "Data quality"].forEach((group) => careerColumns.append(renderCareerTable(fighter, group))); careerPanel.append(careerColumns);
  const metadata = document.createElement("details"); metadata.append(element("summary", "", "Career dates, divisions, form, and streak metadata")); const metadataBody = element("div", "details-body");
  const metadataTable = element("table", "data-table"); const metadataRows = document.createElement("tbody");
  [["First recorded UFC fight", fighter.career.first_fight_date ? formatDate(fighter.career.first_fight_date) : "—"], ["Most recent UFC fight", fighter.career.last_fight_date ? formatDate(fighter.career.last_fight_date) : "—"], ["Recent form", fighter.career.recent_form.join(" · ") || "—"], ["Current W/L streak", fighter.career.current_streak_result ? `${fighter.career.current_streak} ${fighter.career.current_streak_result}` : "—"], ["Divisions", fighter.career.divisions.map((item) => `${item.name} (${item.bouts})`).join(", ") || "—"]].forEach(([label, value]) => { const row = document.createElement("tr"); appendText(row, "td", "", label); appendText(row, "td", "", value); metadataRows.append(row); });
  metadataTable.append(metadataRows); metadataBody.append(metadataTable); metadata.append(metadataBody); careerPanel.append(metadata, renderRawTotals(fighter)); container.append(careerPanel);
  const historyLoading = element("div", "empty-state", "Loading complete fight log…"); container.append(historyLoading);
  try {
    await ensureFighterFights(fighter);
    if (container.isConnected && window.location.hash === `#fighters/${fighter.id}`) historyLoading.replaceWith(renderFightHistory(fighter));
  } catch (error) {
    historyLoading.textContent = `The fight log could not be loaded: ${error.message}`;
  }
}

function renderMarket() {
  const notice = $("#market-notice"); const container = $("#market-matchups"); notice.replaceChildren(); container.replaceChildren();
  const copy = element("div"); appendText(copy, "h2", "", "Paper research only—automatic betting is intentionally off.");
  appendText(copy, "p", "", state.market ? `Quotes captured ${formatTimestamp(state.market.observed_at_utc)}. The policy compares the best available price with a consensus that excludes that target book.` : "No current book-by-book market capture is published. Fighter and matchup research remains available.");
  notice.append(copy, element("span", "pill orange", state.market ? "Execution disabled" : "Capture unavailable"));
  const matchups = state.market?.matchups || [];
  if (!matchups.length) { container.append(element("div", "empty-state", "Run a successful market snapshot to publish current book lines and paper decisions.")); return; }
  matchups.forEach((matchup) => {
    const signal = matchup.current_signal; const card = element("article", "market-card");
    const row = element("div", "fighter-row"); appendText(row, "h3", "", `${matchup.fighter_name} vs ${matchup.opponent_name}`);
    const hasSelection = signal && signal.paper_action !== "pass";
    row.append(element("span", `pill ${hasSelection ? "win" : "neutral"}`, hasSelection ? "Paper bet" : "Pass")); card.append(row);
    if (signal) {
      const stats = element("div", "signal-line");
      [[formatOdds(signal.offered_moneyline), `${signal.best_candidate_name} at ${signal.target_book}`], [formatOdds(signal.market_fair_moneyline), "Leave-one-book-out fair line"], [formatPercent(signal.estimated_expected_return), "Estimated return"]].forEach(([value, label]) => { const item = element("div", "signal-stat"); appendText(item, "strong", "", value); appendText(item, "span", "", label); stats.append(item); }); card.append(stats);
      appendText(card, "p", "signal-reason", signal.reason);
      appendText(card, "p", "signal-reason", `Consensus: ${signal.consensus_book_count} books · ${formatPercent(signal.market_probability)} fair probability · model weight ${formatPercent(signal.model_weight)}.`);
    } else appendText(card, "p", "signal-reason", matchup.current_signal_unavailable_reason || "No evaluable paper signal for this matchup.");
    const details = document.createElement("details"); details.append(element("summary", "", `All ${matchup.book_quotes.length} book lines`));
    const body = element("div", "details-body book-table-wrap"); const table = element("table", "data-table");
    const head = document.createElement("thead"); const header = document.createElement("tr"); ["Book", matchup.fighter_name, matchup.opponent_name, "Quote age", "Consensus"].forEach((value) => appendText(header, "th", "", value)); head.append(header); table.append(head);
    const tbody = document.createElement("tbody"); matchup.book_quotes.forEach((quote) => { const quoteRow = document.createElement("tr"); appendText(quoteRow, "td", "", quote.book); appendText(quoteRow, "td", "", formatOdds(quote.fighter_moneyline)); appendText(quoteRow, "td", "", formatOdds(quote.opponent_moneyline)); appendText(quoteRow, "td", "", `${formatNumber(quote.source_quote_age_seconds, 0)}s`); appendText(quoteRow, "td", "", quote.eligible_for_consensus ? "Yes" : "No"); tbody.append(quoteRow); }); table.append(tbody); body.append(table); details.append(body); card.append(details);
    if (matchup.locked_t24_decision) {
      const locked = matchup.locked_t24_decision; const lockedDetails = document.createElement("details"); lockedDetails.append(element("summary", "", "Locked T-24 paper decision"));
      const lockedBody = element("div", "details-body"); const lockedTable = element("table", "data-table"); const lockedRows = document.createElement("tbody");
      [["Captured", formatTimestamp(locked.observed_at_utc)], ["Candidate", locked.best_candidate_name || "—"], ["Target book", locked.target_book || "—"], ["Offered line", formatOdds(locked.offered_moneyline)], ["Leave-one-book-out fair line", formatOdds(locked.market_fair_moneyline)], ["Estimated return", formatPercent(locked.estimated_expected_return)], ["Paper action", locked.paper_action], ["Reason", locked.reason]].forEach(([label, value]) => { const lockedRow = document.createElement("tr"); appendText(lockedRow, "td", "", label); appendText(lockedRow, "td", "", value); lockedRows.append(lockedRow); });
      lockedTable.append(lockedRows); lockedBody.append(lockedTable); lockedDetails.append(lockedBody); card.append(lockedDetails);
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
  const marketWeight = finite(state.market?.model_weight);
  const marketCard = explainCard("How bets are informed", state.market ? `The price policy finds the best offered line, then estimates fair probability from the other eligible books. The target book is excluded to avoid grading its price against itself. The model currently receives ${formatPercent(marketWeight)} weight${marketWeight === 0 ? " because prospective market-relative evidence is still being collected" : " in the blended estimate"}.` : "Current market policy output is unavailable, so no live paper signal is shown.");
  if (state.performance?.promotion_gate) appendText(marketCard, "p", "section-note", `Promotion gate: ${String(state.performance.promotion_gate.status).replaceAll("_", " ")} · ${state.performance.promotion_gate.paper_selections} / ${state.performance.promotion_gate.minimum_paper_selections} minimum paper selections.`); grid.append(marketCard);
  const dataCard = explainCard("Dataset coverage", `${state.explorer.counts.fighters.toLocaleString()} profiles and ${state.explorer.counts.unique_fights.toLocaleString()} unique fights are published through ${formatDate(state.explorer.data_through)}. Stable UFCStats URL IDs join fighters, opponents, fights, and events; display names are not used as identity keys.`);
  const dataMetrics = element("div", "metric-row"); [[state.explorer.counts.fighters_with_recorded_bouts.toLocaleString(), "profiles with bouts"], [state.explorer.counts.fighter_fight_rows.toLocaleString(), "fighter stat rows"], [state.explorer.fight_columns.length.toLocaleString(), "fields per fight"]].forEach(([value, label]) => { const stat = element("div", "mini-stat"); appendText(stat, "strong", "", value); appendText(stat, "span", "", label); dataMetrics.append(stat); }); dataCard.append(dataMetrics); grid.append(dataCard);
  const rulesCard = explainCard("Sanitization rules", "The publication is rebuilt deterministically from processed CSV files and rejected if it cannot be reproduced. Duplicate fighter perspectives, conflicting IDs, invalid numbers, and oversized output fail validation.");
  const list = element("ul", "provenance-list"); state.explorer.data_dictionary.notes.forEach((note) => appendText(list, "li", "", note)); rulesCard.append(list); grid.append(rulesCard); container.append(grid);

  const dictionary = element("section", "panel"); const heading = element("div", "section-heading"); const headingCopy = element("div"); appendText(headingCopy, "p", "eyebrow", "Definitions"); appendText(headingCopy, "h2", "", "Published data dictionary"); appendText(headingCopy, "p", "section-note", "Every career and bout-level statistic exposed by the explorer."); heading.append(headingCopy); dictionary.append(heading);
  const columns = element("div", "data-columns");
  [["Career statistics", state.explorer.data_dictionary.career], ["Per-fight statistics", state.explorer.data_dictionary.fight_stats]].forEach(([title, definitions]) => { const section = element("div"); appendText(section, "h3", "", title); const table = element("table", "data-table"); const tbody = document.createElement("tbody"); Object.entries(definitions).forEach(([key, definition]) => { const row = document.createElement("tr"); appendText(row, "td", "", definition.label); appendText(row, "td", "", definition.group); appendText(row, "td", "", definition.format || definition.unit); tbody.append(row); }); table.append(tbody); section.append(table); columns.append(section); }); dictionary.append(columns);
  const provenance = document.createElement("details"); provenance.append(element("summary", "", "Publication identity and integrity metadata")); const body = element("div", "details-body"); appendText(body, "p", "", `Identity contract: ${state.explorer.identity_contract}`); appendText(body, "p", "hash", `SHA-256: ${state.explorer.publication_sha256}`); if (model?.model_id) appendText(body, "p", "hash", `Model ID: ${model.model_id} · trained through ${model.training_labels_through || model.data_through}`); provenance.append(body); dictionary.append(provenance); container.append(dictionary);
}

function bindEvents() {
  document.querySelectorAll("[data-nav]").forEach((button) => button.addEventListener("click", () => setRoute(button.dataset.nav)));
  makeAutocomplete($("#matchup-fighter-a"), $("#matchup-results-a"), "a"); makeAutocomplete($("#matchup-fighter-b"), $("#matchup-results-b"), "b");
  $("#analyze-matchup").addEventListener("click", () => { if (state.selected.a && state.selected.b) setRoute(`matchups/${state.selected.a.id}/${state.selected.b.id}`); });
  $("#clear-matchup").addEventListener("click", clearMatchup);
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
