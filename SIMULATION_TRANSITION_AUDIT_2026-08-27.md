# Fighter-specific transition audit — 2026-08-27

## Question and boundary

This bounded experiment tested whether fighter- and opponent-specific histories
add held-out information for conditional fight transitions that the simulator
currently treats only marginally or at context/global level. It did **not**
change simulator parameters, upcoming forecasts, production probabilities, or
betting decisions.

UFCStats supplies round totals rather than timestamped actions. The targets are
therefore explicitly same-round associations, not observed causal transitions:

- KO/TKO win in a round where the winner recorded a knockdown;
- submission attempt in a round where the fighter recorded a takedown;
- submission win in a round where the winner recorded a takedown; and
- credited control share in a round where the fighter recorded a takedown.

The last target is not observed top-position time. UFCStats `CTRL` does not
identify top versus bottom position, and a same-round takedown need not precede
all credited control.

## Data and runtime

The resumable local backfill collected the newest 1,000 physical fights:

- 4,798 doubled fighter-round rows;
- 81 event cards;
- zero fetch failures;
- zero reconciliation issues; and
- 2.30 MB for the normalized round CSV plus a header-only reconciliation CSV.

The first 100 fights took 75 seconds. The remaining 900 took 599 seconds. The
audit itself took under four seconds. Each command had an independent hard
wall-clock budget below one hour; the backfill now accepts
`--max-runtime-seconds` and checkpoints complete bouts before stopping.

The oldest 66 cards were development data and the newest 15 whole cards were a
locked holdout. Every candidate used a fixed strongly pooled estimator:
division/round context had 25 pseudo-opportunities, fighter and opponent effects
had 12 pseudo-opportunities, and their shrunken log-odds (or means for control)
were averaged. No pooling strength was selected on the holdout. Uncertainty
used 5,000 event-card block-bootstrap replicas.

## Results

| Same-round target | Development / holdout opportunities | Context score | Fighter + opponent score | Paired candidate-minus-context 95% interval | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Knockdown → KO/TKO | 318 / 99 | log loss 0.65517 | log loss 0.65284 | −0.00954 to +0.00409 | Not retained |
| Takedown → submission attempt | 1,197 / 221 | log loss 0.55997 | log loss 0.55766 | −0.00931 to +0.00573 | Not retained |
| Takedown → submission win | 1,197 / 221 | log loss 0.39593 | log loss 0.39428 | −0.01417 to +0.01042 | Not retained |
| Takedown → credited control share | 1,197 / 221 | MAE 0.23690 | MAE 0.23094 | **−0.00976 to −0.00233** | Retain only for separate mechanics validation |

The three binary candidates moved in the hypothesized direction, but their
event-card intervals crossed zero. They are not evidence for replacing the
current simulator logic. Knockdown history was especially sparse in holdout:
only 10/99 actor rows had at least three prior knockdown-round opportunities,
only 2/99 opponent rows did, and none had three on both sides.

Control had materially better coverage: 101/221 actor rows and 66/221 opponent
rows had at least three prior takedown-round opportunities; 42 rows had at
least three on both sides. Its MAE improvement was 0.00596 control-share units,
and 99.98% of event-block bootstrap samples favored the pooled fighter/opponent
candidate. RMSE also improved from 0.27398 to 0.26773.

## Interpretation and frozen next step

The experiment supports the general idea, but only for coarse conditional
control so far. It does not establish that fighter-specific KD→finish or
TD→submission conversion improves future prediction, and it cannot identify
exact transition order.

Before changing the simulator, implement conditional control behind an
explicit candidate-only parameter-fit flag. Map own takedown-round credited
control history to top retention and control conceded after an opponent's
takedown to escape resistance, retaining the same causal cutoffs, event-card
bootstrap member, and strong pooling. First use the opened 15-card cohort only
as a development mechanics screen: reject the mapping if it harms joint
side/method, winner, duration, ground-strike, or control predictive scores.
Because this audit already opened those cards, final validation must use later
prospective cards or another genuinely untouched chronological cohort. Every
development screen and validation command must retain a hard runtime budget of
at most 3,300 seconds.

Reproduction:

```bash
export PYTHONPATH="$PWD/src"

python -m fight_sim backfill \
  --max-fights 1000 \
  --checkpoint-every 25 \
  --max-runtime-seconds 1800 \
  --summary-output artifacts/simulations/transition-audit/round-backfill-summary.json

python -m fight_sim transition-audit \
  --holdout-latest-events 15 \
  --bootstrap-replicates 5000 \
  --max-runtime-seconds 300 \
  --output artifacts/simulations/transition-audit/report-1000-fights.json \
  --predictions-output artifacts/simulations/transition-audit/predictions-1000-fights.csv
```

Detailed reports and holdout rows remain under ignored `artifacts/`; the
normalized/reconciled source CSV is the durable input.

## Follow-up completed

The explicit candidate fit, paired five-card simulator screen, and broad
100-path chronological audit are complete. The candidate improved all headline
five-card point estimates, including total-control CRPS from 148.66 to 145.13
seconds, but the broader simulator remained a poor standalone predictor and
substantially underpredicted UFCStats control. See
`SIMULATION_CONDITIONAL_CONTROL_AND_BREADTH_REPORT_2026-08-27.md` for the full
229-fight result. The fit remains candidate-only and has not changed production
or website forecasts.
