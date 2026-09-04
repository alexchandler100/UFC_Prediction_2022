# Betting profitability audit — September 4, 2026

**The project does not yet establish a repeatable profitable betting strategy. The first improvement should be correcting the duration data and recommendation calculations, followed by controlling combined stakes.** More aggressive staking or another complex winner model is not supported by the available results.

This audit covers the checked-in website publication, historical research, and saved paper decisions. It changes no website behavior, production model, or existing ledger. No new data was collected and no models were retrained. All returns below are hypothetical, not records of accepted wagers.

## What the evidence says

The broad historical test used **904 fights across 189 events**. Each year's predictions used earlier data, although these years have also influenced previous research decisions. Betting one unit on each selection gives:

| Strategy | Bets | Profit | Return per unit risked |
| --- | ---: | ---: | ---: |
| UFC winner model | 748 | −42.59 units | −5.69% |
| Equal model/market blend | 633 | −50.87 units | −8.04% |
| Other sportsbooks' consensus | 27 | −15.00 units | −55.56% |
| Small adjustment to market consensus | 113 | −6.66 units | −5.90% |
| Make no bets | 0 | 0 | Undefined: nothing risked |

The adjustment's 95% return interval runs from approximately **−24.83% to +13.34%**. Resampling whole UFC cards keeps related bets together. This range does not account for every previous experiment and does not prove future profitability. The earlier positive 35-bet result is superseded by this broader study; its threshold selection also reused outcomes already used to fit the adjustment.

Official prospective tracking—the rule frozen before the fights—contains **two settled bets, both losses, from one event**. The website's wider archive has 15 settled records across two cards, including experimental totals. A first-qualifying replay at 1% of bankroll per bet grows 100 units to **100.9792**; full Kelly ends at **95.3305**. Those two cards cannot establish either staking rule as superior. The default robust sizing replay funds **zero** of its four supported settled records; its unchanged bankroll is abstention, not a profitable betting track record.

See [historical results and payout stresses](history/findings.md), [historical summary table](history/summary.csv), and [exact website replay](website_replay_summary.csv).

## Ranked improvement plan

| Priority | Finding and consequence | Proposed correction | Acceptance check |
| --- | --- | --- | --- |
| 1 | Missing schedules are inferred from results. The 206 inferred five-round training fights contain **zero early finishes**; 200 lasted over 3.5 rounds. This inflates the duration model's five-round examples. | Recover scheduled lengths independently from fight-page metadata; omit unknown schedules from duration fitting/evaluation. Rebuild duration forecasts and their calibration before assessing totals value. | A missing schedule plus a round-one finish stays unknown. Known five-round early finishes remain in the five-round group. Publish coverage lost/recovered and later-fight comparisons against market prices. |
| 2 | The current totals assessment cannot pass the performance replay's moneyline validator. A temporary archive reproduction confirms the failure after current board assessments are included. | Dispatch validation by market and policy; preserve the exact assessment originally published. | Current totals survive publication, archiving, settlement, and bankroll replay without replacing their probability or price. |
| 3 | Totals calibration can be marked available when its later-fight check is too small. Five-round calibration has only **14 later test fights**. | Require a completed, adequately supported later check. Keep profitability approval separate from probability adjustment. | The 40-fight synthetic example with only two later test fights cannot authorize sizing. Corrected totals must beat relevant market comparisons before funding. |
| 4 | Tai Tuivasa +505 advertises **+14.10%** expected return, while its calibrated mean implies **−3.25%** and its suggested stake is zero. | Use one explicit decision probability for qualification, displayed return, and sizing; retain the raw estimate as research context. | A raw-positive, calibrated-negative bet is not presented as a funded recommendation. Count funded bets separately from evaluated records. |
| 5 | Prices are checked when collected but do not expire in the browser. Totals omit exact source-update and event-start fields. | Carry quote/update/start timestamps through publication; expire unusable quotes and started events, including when collection fails. | Clock-controlled checks at publication, 30 minutes later, six hours later, and event start; unknown timing cannot imply freshness. |
| 6 | Eleven current total-round stakes sum to **55% of bankroll on one card**; two fights each carry 10%. | Add a shared allocation step. Evaluate the illustrative 1% per fight, 5% per card, 10% outstanding limits, with one funded selection per fight. | All limits hold across simultaneous cards, duplicate snapshots, and related markets. Limits are illustrative risk preferences, not proven optimal percentages. |
| 7 | Choosing accessible books filters the single globally best offer; it does not reconsider alternative qualifying prices. | Retain eligible offers and recompute selection using the user's chosen books, excluding the target book from its probability estimate. | Removing the best book substitutes a qualifying accessible alternative or clearly records no available bet. |
| 8 | Historical research, official decisions, and archive replays answer different questions; apparent value clusters in long shots. | Label evidence sources and denominator counts; add a no-bet option to future threshold selection. Freeze a small number of proposed policies before new events. | Forecast/model/calibration versions and training cutoffs are traceable; no later prices or outcomes enter selection; no duplicate recommendation becomes another independent result. |

Details and affected rows: [duration evidence](duration/findings.md), [board evidence](board/findings.md), [all current bet calculations](board/board_trace.csv), and [illustrative allocation](board/illustrative_allocation.csv). These are proposed corrections, not implemented product changes.

All 12 current prices and forecasts trace to saved source data. Their exact current bet IDs are absent from the archived snapshots, however, so the current assessments must not be treated as already archived. The book-access check identifies 32 qualifying alternative book/selection offers hidden by the current filter. The duration repair requires reevaluating the shared method/duration output for 14 published matchups, not just the two current five-round total bets.

## Price realism and unavailable comparisons

- Historical stress tables reduce **net winning payouts by 2% and 5%**, leaving losing stakes, selected bets, and original thresholds unchanged. They are fixed scenarios, not measured fees or observed execution costs. None of the four broad strategies becomes profitable.
- Book/year/odds/price-age tables are diagnostic contributions to the original policy. The saved historical ledger contains only its selected best offers; it cannot reproduce a true single-book policy without reconstructing alternatives and fits. That work is explicitly unavailable in this bounded audit.
- Historical quotes can be up to 24 hours old, and their nominal T-24 cutoff is anchored to the event date rather than a verified card start. The website's capture rule instead permits 30-minute-old quotes. These samples do not reproduce identical entry conditions.
- Existing closing-price comparisons measure later same-book implied-probability movement. They are **not** verified, complete, margin-adjusted closing-market value. The provider distinguishes snapshot and quote-update timestamps in its [API documentation](https://the-odds-api.com/liveapi/guides/v4/).
- Historical method research covers 7,755 selections over 2,586 fights using averaged prices. The market beat the outcome model; these averages cannot establish executable returns. No method-profit claim is inferred.
- Archived totals lack usable historical robust sizing assessments and exact settlement times. A verified capped outstanding-capital replay is unavailable. The current-board allocation is an exposure illustration, not a historical profit estimate.

## Reproduce and review

From the repository root, using the existing Python environment and Node:

```powershell
.venv/Scripts/python.exe -B scripts/audit_profitability.py
.venv/Scripts/python.exe -B -m unittest discover -s tests -p 'test_profitability_*audit.py' -v
```

The historical component reads existing files under `~/.ufc-data-lab/historical-odds/bestfightodds/analysis`; `--analysis-dir` overrides that location. It does not download missing data. Component scripts also accept `--output-dir` for isolated reproduction. The runner has per-component time limits and fingerprints production inputs before and after execution.

[Manifest](manifest.json) records input hashes, completed components, runtime, and whether protected production files remained unchanged. [Validation](validation.json) records focused test results. This narrative describes the September 4 snapshot; if inputs change, regenerate the tables and review the narrative before using it as a new audit.

Validation completed: **68 tests passed**, including 16 new audit checks. The full audit runner completed in **14.235 seconds** on this machine and verified that all 223 protected production inputs remained unchanged. Test execution took approximately 26 seconds separately. The website replay directly executes its existing pure calculation functions; no browser session or fresh odds request was needed.

After corrections, keep proposed rules fixed while collecting new pre-fight decisions. Preserve the existing minimum review counts (main moneyline policy: 500 scored fights, 100 selections, 40 settled events; totals: 300 scored lines, 100 selections, 30 events), plus positive return and price-quality evidence. Reaching a count alone is not approval. Rank future experiments by whether they add information beyond obtainable sportsbook prices; do not choose a live strategy from the small positive archive replay.
