# Free historical odds feasibility audit

Generated 2026-08-28T23:53:39Z. This is a bounded, read-only research
audit. It did not change production predictions and did not store raw pages.

## Bottom line

BestFightOdds is **technically usable** for a historical research
backfill. Its visible event tables do not contain quote times, but the public
line-movement chart used by the site returns bookmaker-specific prices with
absolute timestamps. The sample recovered 358
strictly pre-event price points.

Coverage is not uniform. Mean market history was verified back to
2012; a strict three-book history was
verified only from 2021 in this
sample. Older mean or single-book prices can benchmark the model, but they must
not be presented as equally strong as a multi-book consensus.

A large backfill is **not started automatically**. The public robots policy
allows the sampled paths, while the short published terms neither grant nor
prohibit bulk automated reuse. Ask the source for permission or clarification
before copying the archive at scale. If permission is received, store raw
history outside Git and commit only compact derived consensus data and audits.

FightOdds.io was not crawled: `robots_path_returns_generic_html_application; automated sampling skipped`.

## What was checked

- 50 of 50 BestFightOdds event pages succeeded, spread across 2012-2026.
- 13 pages exposed at least one matchup with paired prices from three books.
- Detailed line history was tested on 10 events.
- 10 detailed events had timestamped, pre-event prices for both fighters from all three sampled books.
- Mean history was checked across 15 events; 15 had pre-event data for both fighters.
- 436 chart points decoded; 358 occurred strictly before the event calendar date.
- The audit made 175 requests and downloaded 24.69 MiB.

## Data rule for any future backfill

Accept a price only when it has a bookmaker, matchup, both fighter sides, and
an absolute timestamp before the event. Build separate opening, T-72, T-24,
T-6, and closing datasets; never mix those horizons. Require at least three
books for a market consensus. With date-only historical event times, the
strict safe cutoff is before 00:00 UTC on the event date until an authoritative
event start time is added.

Historical results may be used to develop and freeze a model/market blend.
They do not replace the already scheduled future-only confirmation over at
least 200 fights and 20 events.

## Next action

Request source permission, then run a low-rate resumable backfill outside git. The backfill should be
resumable, rate-limited, capped by requests and disk space, and retain source
timestamps. If permission is not available, continue the repository's T-24
prospective capture instead of using weakly dated closing lines.
