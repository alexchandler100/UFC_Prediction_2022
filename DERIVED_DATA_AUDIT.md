# Derived data correctness audit

Audit date: 2026-08-25

## Scope and result

The repository has three distinct derived products. They must not be treated as
one dataset:

1. `ufc_fights_point_in_time.csv` is the production winner-model matrix. It has
   one row per terminal W/L fight and 82 pre-fight difference features.
2. `fighter_explorer.json` and `fighter_fights_*.json` are the website's career
   summaries and fight logs.
3. `ufc_fights_reported_derived_doubled.csv` is a notebook-era, 326-column
   artifact. It is not part of the weekly production model or website.

The production point-in-time matrix was reproduced from current raw inputs:
8,669 fight rows, 82 features, and 710,858 feature cells matched at a maximum
absolute floating-point difference of `4.55e-13`. Strict repository validation
now performs this complete source replay automatically and rejects any changed
feature cell above `1e-12`.

One real website-summary defect was confirmed and fixed. Some early UFCStats
bouts have detailed counts but no recoverable duration. Those counts previously
entered rate numerators while contributing no denominator time, inflating some
per-minute statistics. Every website rate now uses only bouts with both the
required statistic and a known positive duration. A wholly unavailable statistic
stays `null`, and duration/control coverage counts are published beside the
career data.

No formula error was found in the current 82-feature production builder. Its
stored matrix exactly matched a clean chronological replay. This does not mean
that every feature is necessarily predictive or that the model has a betting
edge; it means the implemented values match the documented formulas and causal
data contract.

## Raw-data invariants

`validate_data.py` enforces the following before derived data is trusted:

- exactly two mirrored rows per stable UFCStats fight ID;
- opposite fighter/opponent IDs and compatible results on each pair;
- nonnegative finite counts, valid elapsed fight time, and landed counts no
  greater than attempts;
- head + body + leg equals significant strikes, separately for landed and
  attempted counts;
- distance + clinch + ground equals significant strikes, separately for landed
  and attempted counts;
- significant strikes do not exceed total strikes; and
- the two fighters' recorded control time does not exceed elapsed fight time.

The current snapshot passes these arithmetic checks. There are 71 older fights
with unknown duration and 181 with unknown control time. Missing source values
are retained as unknown rather than silently converted to zero.

## Production feature formulas

Every side feature below is calculated immediately before a bout. The model
stores `fighter value - opponent value`, so reversing a matchup negates all 82
features.

### Elo and activity

- Three overall and three same-division Elo ratings start at 1500, use
  K-factors 32, 64, and 128, and calculate expected score as
  `1 / (1 + 10 ** ((opponent - fighter) / 400))`.
- Split and majority decisions use 75% of the normal Elo update. Draws use a
  score of 0.5. No-contests do not change rating but do advance its as-of date.
- Inactivity regresses rating toward 1500 by `0.92 ** inactive_years`.
- Reliable Elo shrinks medium Elo toward 1500 by
  `prior_fights / (prior_fights + 5)`.
- Rating uncertainty is `1 / sqrt(prior_fights + 1)`.
- Average opponent Elo has two virtual 1500-rated opponents.
- Layoff days are capped at five years and transformed with `log1p`; a fighter
  without history receives a two-year default.

### Record features

The same eight formulas are calculated for career, last year, last three years,
and current division:

- fights, wins, and losses are `log1p(count)`;
- win rate is `(wins + 0.5 * draws + 1.5) / (wins + losses + draws + 3)`;
- finish-win and finish-loss rates are `(count + 1) / (wins + losses + 4)`; and
- KO and submission win rates are `(count + 0.75) / (wins + losses + 4)`.

These are deliberately smoothed state features. They are not literal displayed
career percentages.

### Performance features

Sixteen formulas are calculated for career and the last three years:

- count rates per 15 minutes are `sum(observed count) /
  (sum(observed seconds) / 900 + 1)`;
- the extra `+1` is one zero-count 15-minute prior bout;
- significant-strike accuracy is
  `(landed + 0.45 * 40) / (attempted + 40)`;
- significant-strike defense is one minus the opponent version of that formula;
- takedown accuracy is `(landed + 0.35 * 8) / (attempted + 8)`; and
- takedown defense is one minus the opponent version.

Only observations where the required source fields are present enter a formula.
Unknown counts add neither zero counts nor exposure. The exact feature-name set,
formula examples, chronological behavior, no-contest behavior, and missing-stat
behavior are regression-tested in `tests/test_reliability.py`.

### Fighter profile features

- Age is calculated at the bout date from date of birth. Unknown age is imputed
  to 29 and accompanied by `age_known = 0`; age squared is also included.
- Height is parsed in inches, defaults to 69 when missing, and has a known flag.
- Reach is parsed in inches, defaults to 70 when missing, and has a known flag.

The known flags let the model distinguish a real average measurement from an
imputed value.

## Website career formulas

Website statistics are descriptive, unsmoothed career summaries and intentionally
differ from model features:

- per-minute or per-15 rates divide observed totals by the duration of precisely
  the bouts contributing those totals;
- absorbed and differential rates require a matched opponent row for the same
  stable fight ID;
- accuracy and target/position shares include only rows where both numerator and
  denominator are known;
- control share requires both fighters' control values; and
- average fight time divides known total duration by the number of bouts with a
  known duration, not by every recorded bout.

The publication is rebuilt from processed sources and compared as a complete
object, including every fighter and shard, during strict validation.

## Unsupported legacy table

The 326-column `ufc_fights_reported_derived_doubled.csv` is stale relative to
the raw table (318 fighter rows / 159 fights behind at audit time). Its old
name-based builder also turns some undefined accuracies into zero and contains
hand-weighted composite scores that add overlapping strike partitions. For
example, significant-strike totals, target partitions, and position partitions
can represent the same strikes more than once inside one composite. Those
scores therefore do not have a defensible unit or interpretation.

The artifact remains readable only so old exploratory notebooks do not break.
It must not be used for current training, forecasts, website statistics, or
correctness claims. New work should use the stable-ID point-in-time matrix.

## Source and coverage limits

UFCStats defines SLpM, SApM, significant-strike accuracy/defense, takedown
average/accuracy/defense, and submission average on its fighter pages; see the
[UFCStats fighter statistics definitions](http://ufcstats.com/fighter-details/45f7cb591c3ab00b).
The raw table's mirrored detailed counts can be checked against individual
[UFCStats fight pages](http://ufcstats.com/fight-details/ebf7cea27b83c432).

This audit verified recent and historical source-page samples plus every
internal row/pair/arithmetic invariant. It was not an independent page-by-page
rescrape of all 8,824 fights. The updater intentionally refreshes the latest 12
already-known events because UFCStats may correct old pages without changing
URLs. It also intentionally excludes events whose name contains `Road to UFC`;
that is a coverage policy, not evidence that such bouts do not exist.
