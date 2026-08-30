# Historical method-price evaluation — 2026-08-30

## Plain result

Historical method prices predict fighter-by-method outcomes better than the
current logistic outcome model. A market/model blend chosen using only earlier
years did not improve on the market. Do not use the outcome model or simulator
to adjust method prices for production betting.

## What was actually collected

BestFightOdds historical pages usually expose three separate yes/no props for
one fighter: that fighter by KO/TKO, submission, or decision. They do not
usually provide one simultaneous six-outcome board for both fighters. The old
evaluator expected the latter and correctly produced no results rather than
misrepresenting the data.

The corrected evaluator scores each available prop as a binary prediction. The
collection contains:

- 358 parsed events and no failed event pages;
- 2,729 matched fights;
- 8,265 method selections and 133,708 historical price points; and
- seven selection histories still pending, less than 0.1% of the selections.

The prices are BestFightOdds historical means. They are useful for probability
research but are not prices at which a bet is known to have been available.

## Test design

For every test year, the fight model was refitted using only fights before that
year. Its ten-year training samples ranged from 4,310 fights for the 2021 test
to 4,882 fights for the 2026 test. No later fight result was used to predict an
earlier fight.

The main comparison contains 7,755 individual method selections covering 2,586
fights and 237 events from January 2021 through August 2026. It uses the safe
T-24 historical observation and requires that observation to have been updated
within the preceding 72 hours. Historical event start timestamps are not
available, so the cutoff is based conservatively on midnight UTC of the source
event date.

Raw price probabilities include the source's average markup. Starting in 2022,
that average effect is estimated from earlier price years only. The 2021 prices
remain raw because no earlier method-price year is available.

Log loss measures the quality of the full probabilities, not just which choice
was largest. Lower is better.

| Forecast | Binary log loss | Brier score |
|---|---:|---:|
| Calibrated historical market | **0.39450** | **0.12252** |
| Logistic outcome model | 0.41618 | 0.12918 |
| Fixed 75% market / 25% model | 0.39322 | 0.12208 |
| Fixed 50% market / 50% model | 0.39641 | 0.12310 |
| Blend selected using earlier years | 0.39481 | 0.12264 |

The model-minus-market log-loss difference was `+0.02168`; its whole-event 95%
range was `[+0.01541, +0.02790]`. The positive range means the model was
reliably worse on this sample.

The fixed 25% model blend had a small hindsight improvement of `-0.00128`, but
its 95% range was `[-0.00285, +0.00029]`, which includes no improvement. The
year-by-year blend that made every choice using earlier data was slightly worse
than market alone: `+0.00031`, with a 95% range of
`[-0.00141, +0.00204]`.

## Method breakdown

The market beat the model in all three prop types.

| Prop | Selections | Market log loss | Model log loss |
|---|---:|---:|---:|
| Fighter by KO/TKO | 2,582 | **0.39797** | 0.42224 |
| Fighter by submission | 2,583 | **0.27237** | 0.29267 |
| Fighter by decision | 2,590 | **0.51284** | 0.53332 |

## Year-by-year result

The market also beat the model in every individual test year.

| Test year | Selections | Market log loss | Model log loss |
|---|---:|---:|---:|
| 2021 | 1,338 | **0.41967** | 0.42638 |
| 2022 | 1,364 | **0.38721** | 0.40965 |
| 2023 | 1,349 | **0.41898** | 0.43990 |
| 2024 | 1,425 | **0.37529** | 0.39183 |
| 2025 | 1,317 | **0.38248** | 0.40904 |
| 2026 | 962 | **0.38040** | 0.42384 |

## Simulator limitation

The simulator is not part of this full comparison. Matching causal simulator
forecasts do not exist for these 2,586 fights. The available 229-fight
simulation experiment was also used to choose simulator changes, so adding it
here would be an unfair test and would still leave most prices unmatched.

Before testing a market/model/simulator blend, generate frozen simulator
forecasts for a separate historical period without tuning on those outcomes.
The existing simulator evidence is not strong enough to justify spending the
compute required for a full high-precision rerun yet.

## Decision

- Keep the historical market as the method-probability benchmark.
- Do not promote the logistic method model, the fixed blend, or the rolling
  blend.
- Do not infer profitability from averaged prices.
- Finish the seven pending mean histories when the other source collector is
  idle.
- Collect individual-book method histories only if the source contract and
  settlement rules can be verified; those are required for an honest profit
  test.
- Future method-model work should target information not already present in
  the market and must beat this market benchmark on later untouched events.

The generated detail and JSON report remain outside Git under
`~/.ufc-data-lab/historical-odds/bestfightodds/analysis/`.
