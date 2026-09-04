# Historical profitability audit

The saved broad historical test does not establish a profitable betting rule. These are hypothetical one-unit bets using recorded prices, not executed bets or the exact website policy.

## Reproduced results

| Strategy | Bets | Profit | ROI | 2% payout reduction ROI | 5% payout reduction ROI |
|---|---:|---:|---:|---:|---:|
| current_model | 748 | -42.59u | -5.69% | -6.80% | -8.46% |
| fixed_50_50_logit_blend | 633 | -50.87u | -8.04% | -9.16% | -10.84% |
| leave_one_out_market | 27 | -15.00u | -55.56% | -56.30% | -57.41% |
| market_first_candidate | 113 | -6.66u | -5.90% | -6.72% | -7.95% |

The shared cohort contains 904 fights across 189 events. Making no bets produces zero profit and zero exposure; ROI is undefined because nothing is risked.

Payout reductions are deterministic stress scenarios: subtract 2% or 5% of net winnings from winning bets only. Losing stakes stay at -1 unit. Picks and thresholds remain fixed. These percentages are assumptions, not measured execution costs.

## What limits the result

- **No profitable broad strategy** All four saved strategies lose money before the added payout stresses. The narrower positive 35-bet snapshot was superseded by broader rolling evidence. Source: HISTORICAL_ODDS_BACKFILL.md:235, HISTORICAL_ODDS_BACKFILL.md:263.
- **Older threshold selection used fitted outcomes** The narrow evaluator loads an adjustment refitted on both training and selection outcomes, then selects betting thresholds from that same selection period. Its later test was separate at the time, but threshold-selection returns are optimistic training evidence. The broad rolling evaluator correctly uses a development-only fit for threshold selection. Source: src/evaluate_market_first_challenger.py:405, src/evaluate_historical_moneyline_profitability.py:555, src/evaluate_rolling_moneyline_profitability.py:311.
- **Abstention was not the historical fallback** The historical evaluator uses a 5% cutoff even when no earlier cutoff was profitable or earlier examples were insufficient. This audit adds a zero-exposure comparison without rewriting old policies. Source: src/evaluate_historical_moneyline_profitability.py:492.
- **Selected long shots need calibration checks** The saved market-only rule selected 27 bets, including 27 underdogs, with median decimal odds 6.75. See summary.csv for estimated and realized returns. These hindsight diagnostics motivate a new independent calibration test, not a favorite-only rule selected from these results. Source: rolling_moneyline_profitability_2021_2026.csv, breakdowns.csv.
- **Book access and timing were assumed** The historical test chooses among seven books and excludes limits, rejected bets, fees, and latency. Historical T-24 is measured from midnight UTC of the source event date; prices can be up to 24 hours old. This is not an exact replay of the deployed website policy. Source: src/evaluate_rolling_moneyline_profitability.py:478, HISTORICAL_ODDS_BACKFILL.md:230.
- **Closing movement is not verified closing fair value** The source closing metric subtracts raw same-book implied probabilities. It does not remove the later bookmaker margin or prove that the entry quote was executable. Positive movement alone did not ensure profit in this ledger. Source: src/evaluate_historical_moneyline_profitability.py:359.
- **Historical research was reused** Feature choices and model families were explored using some of these years. Chronological fitting prevents direct use of future training outcomes, but repeated design decisions make these retrospective development evidence rather than fresh confirmation. New comparisons must freeze rules before collecting later fights. Source: src/evaluate_rolling_moneyline_profitability.py:476, MODEL_FAMILY_RESEARCH.md:181.
- **Prediction provenance can be missing** The precomputed prediction loader checks training cutoffs only if supplied; its output still describes the file as causal. Require traceable training provenance or label imported predictions unverified in future evaluations. Source: src/evaluate_bestfightodds_history.py:117.
- **Method prices cannot establish executable returns** Method research covers 7,755 selections over 2,586 fights using historical mean prices. The outcome model and the earlier-data-selected blend did not beat the market. Mean prices are not known executable offers, so no method profit replay is inferred here. Source: HISTORICAL_METHOD_PRICE_EVALUATION_2026-08-30.md:26, HISTORICAL_METHOD_PRICE_EVALUATION_2026-08-30.md:36.

## Reproduction and interpretation

Run `python scripts/audit_profitability_history.py`; override `--analysis-dir`, `--database`, or `--output-dir` when needed. The database argument records availability only; this bounded audit never derives raw odds, fits a model, changes thresholds, or accesses the network.

`summary.csv` includes event-level 95% return intervals; one-event samples receive no interval. Cards are resampled as whole units so bets on one card stay together. These intervals do not correct for all previous experiments or guarantee future results. `breakdowns.csv` contains hindsight diagnostics by year, book, odds, quote age, and threshold-selection status. Book rows only attribute original best-book bets; they are not single-book strategies.

The 'enough_prior_threshold_examples' cohort reproduces the source report: it excludes insufficient-history fallbacks, but still includes 5% fallbacks when earlier cutoffs lost money. Both cohorts are therefore descriptive, not an automatically selected deployment policy.

Event-close drawdown avoids inventing settlement order within a card. The separate fight-ID-order drawdown reproduces the historical deterministic ordering and is not a claim about intraday bankroll exposure. This ledger lacks execution and outstanding-bet timing, so capped bankroll, one-funded-bet-per-fight across market types, and 10% outstanding-exposure policies cannot be honestly replayed from it.

All input fingerprints, source period contracts, and unavailable comparisons are recorded in `audit_report.json`. No production recommendation changed.
