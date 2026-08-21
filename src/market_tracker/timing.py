"""Prospective, outcome-blind tests of UFC market-entry timing.

The rules in this module deliberately operate on immutable capture ledgers.
They do not execute wagers.  A favorite is classified once, from the first
eligible capture, and is never reclassified after a later line flip.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from statistics import median
from typing import Iterable, Mapping

from ._common import canonical_hash, implied_probability, utc_datetime
from .quotes import QuoteSnapshot, consensus_as_of
from .source_metadata import QuoteSourceMetadata


TIMING_POLICY_VERSION = "favorite-early-dog-late-v1"
MIN_CONSENSUS_BOOKS = 3
MIN_EXPECTED_RETURN = 0.05
MAX_SOURCE_QUOTE_AGE_SECONDS = 30.0 * 60.0
EARLIEST_MIN_LEAD_SECONDS = 32.0 * 60.0 * 60.0
EARLIEST_MAX_LEAD_SECONDS = 144.0 * 60.0 * 60.0
BASELINE_MIN_LEAD_SECONDS = 20.0 * 60.0 * 60.0
BASELINE_MAX_LEAD_SECONDS = 28.0 * 60.0 * 60.0
LATE_MIN_LEAD_SECONDS = 1.0 * 60.0 * 60.0
LATE_MAX_LEAD_SECONDS = 5.0 * 60.0 * 60.0


@dataclass(frozen=True)
class _Capture:
    capture_id: str
    matchup_id: str
    event_id: str
    fighter_id: str
    opponent_id: str
    observed_at_utc: str
    event_start_utc: str
    lead_seconds: float
    quotes: tuple[QuoteSnapshot, ...]


@dataclass(frozen=True)
class _Candidate:
    event_id: str
    matchup_id: str
    capture_id: str
    observed_at_utc: str
    action: str
    book: str
    quote_id: str
    moneyline: int
    expected_return: float
    break_even_probability: float


def _profit_multiple(moneyline: int) -> float:
    return moneyline / 100.0 if moneyline > 0 else 100.0 / abs(moneyline)


def _expected_return(win_probability: float, moneyline: int) -> float:
    return win_probability * _profit_multiple(moneyline) - (1.0 - win_probability)


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _event_block_interval(
    observations: Iterable[tuple[str, float]], *, label: str
) -> dict[str, object]:
    grouped: dict[str, tuple[float, int]] = {}
    for event_id, value in observations:
        total, count = grouped.get(event_id, (0.0, 0))
        grouped[event_id] = (total + float(value), count + 1)
    count = sum(item[1] for item in grouped.values())
    point = sum(item[0] for item in grouped.values()) / count if count else None
    result: dict[str, object] = {
        "definition": label,
        "observation_count": count,
        "event_count": len(grouped),
        "point_estimate": point,
        "bootstrap_samples": 0,
        "ci_95_lower": None,
        "ci_95_upper": None,
    }
    if len(grouped) < 2 or not count:
        return result
    blocks = [grouped[key] for key in sorted(grouped)]
    generator = random.Random(
        int(canonical_hash({"label": label, "blocks": blocks})[:16], 16)
    )
    samples: list[float] = []
    for _ in range(10_000):
        selected = [generator.choice(blocks) for _ in blocks]
        selected_count = sum(item[1] for item in selected)
        samples.append(sum(item[0] for item in selected) / selected_count)
    samples.sort()
    result.update(
        {
            "bootstrap_samples": len(samples),
            "ci_95_lower": _quantile(samples, 0.025),
            "ci_95_upper": _quantile(samples, 0.975),
        }
    )
    return result


def _eligible_captures(
    quotes: tuple[QuoteSnapshot, ...],
    metadata: tuple[QuoteSourceMetadata, ...],
) -> tuple[_Capture, ...]:
    metadata_by_quote = {item.quote_id: item for item in metadata}
    grouped: dict[tuple[str, str], list[QuoteSnapshot]] = defaultdict(list)
    for quote in quotes:
        grouped[(quote.capture_id, quote.matchup_id)].append(quote)
    captures: list[_Capture] = []
    for (capture_id, matchup_id), records in grouped.items():
        fresh: list[QuoteSnapshot] = []
        for quote in records:
            source = metadata_by_quote.get(quote.quote_id)
            if source is None:
                continue
            if (
                source.capture_id != quote.capture_id
                or source.matchup_id != quote.matchup_id
                or source.event_id != quote.event_id
                or source.book != quote.book
                or source.observed_at_utc != quote.observed_at_utc
            ):
                continue
            if -300.0 <= float(source.source_quote_age_seconds) <= MAX_SOURCE_QUOTE_AGE_SECONDS:
                fresh.append(quote)
        books = {item.book.casefold() for item in fresh}
        if len(books) < MIN_CONSENSUS_BOOKS + 1 or len(books) != len(fresh):
            continue
        contracts = {
            (
                item.event_id,
                item.fighter_id,
                item.opponent_id,
                item.timing_precision,
                item.event_start_utc,
                item.observed_at_utc,
            )
            for item in fresh
        }
        if len(contracts) != 1:
            continue
        event_id, fighter_id, opponent_id, precision, event_start, observed = next(
            iter(contracts)
        )
        if precision != "timestamp" or not event_start:
            continue
        lead = (
            utc_datetime(event_start, "event_start_utc")
            - utc_datetime(observed, "observed_at_utc")
        ).total_seconds()
        if lead < LATE_MIN_LEAD_SECONDS or lead > EARLIEST_MAX_LEAD_SECONDS:
            continue
        captures.append(
            _Capture(
                capture_id=capture_id,
                matchup_id=matchup_id,
                event_id=event_id,
                fighter_id=fighter_id,
                opponent_id=opponent_id,
                observed_at_utc=observed,
                event_start_utc=event_start,
                lead_seconds=lead,
                quotes=tuple(sorted(fresh, key=lambda item: (item.book.casefold(), item.quote_id))),
            )
        )
    return tuple(
        sorted(captures, key=lambda item: (item.observed_at_utc, item.capture_id, item.matchup_id))
    )


def _market_probability(capture: _Capture) -> float:
    return consensus_as_of(
        capture.quotes,
        capture_id=capture.capture_id,
        matchup_id=capture.matchup_id,
        as_of_utc=capture.observed_at_utc,
        min_books=MIN_CONSENSUS_BOOKS,
    ).no_vig_fighter_probability


def _candidate(capture: _Capture | None, allowed_sides: tuple[str, ...]) -> _Candidate | None:
    if capture is None:
        return None
    candidates: list[_Candidate] = []
    for target in capture.quotes:
        market = consensus_as_of(
            capture.quotes,
            capture_id=capture.capture_id,
            matchup_id=capture.matchup_id,
            as_of_utc=capture.observed_at_utc,
            min_books=MIN_CONSENSUS_BOOKS,
            exclude_books=(target.book,),
        )
        probabilities = {
            "fighter": market.no_vig_fighter_probability,
            "opponent": 1.0 - market.no_vig_fighter_probability,
        }
        lines = {
            "fighter": target.fighter_moneyline,
            "opponent": target.opponent_moneyline,
        }
        side_values = [
            (side, _expected_return(probabilities[side], lines[side]))
            for side in allowed_sides
        ]
        side, expected = min(side_values, key=lambda item: (-item[1], item[0]))
        if expected < MIN_EXPECTED_RETURN:
            continue
        candidates.append(
            _Candidate(
                event_id=capture.event_id,
                matchup_id=capture.matchup_id,
                capture_id=capture.capture_id,
                observed_at_utc=capture.observed_at_utc,
                action=side,
                book=target.book,
                quote_id=target.quote_id,
                moneyline=lines[side],
                expected_return=expected,
                break_even_probability=implied_probability(lines[side]),
            )
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -item.expected_return,
            item.book.casefold(),
            item.quote_id,
            item.action,
        ),
    )


def _first_in_window(
    captures: tuple[_Capture, ...], minimum: float, maximum: float
) -> _Capture | None:
    eligible = [item for item in captures if minimum <= item.lead_seconds <= maximum]
    # The first scheduled observation to enter a window has the greatest lead.
    return min(
        eligible,
        key=lambda item: (-item.lead_seconds, item.observed_at_utc, item.capture_id),
    ) if eligible else None


def _best_break_even(capture: _Capture, side: str) -> float:
    lines = [
        item.fighter_moneyline if side == "fighter" else item.opponent_moneyline
        for item in capture.quotes
    ]
    return min(implied_probability(line) for line in lines)


def _price_timing_summary(
    paired: list[tuple[str, _Capture, _Capture, str, str]]
) -> dict[str, object]:
    favorite_advantages: list[tuple[str, float]] = []
    underdog_advantages: list[tuple[str, float]] = []
    common_favorite: list[tuple[str, float]] = []
    common_underdog: list[tuple[str, float]] = []
    for event_id, early, late, favorite, underdog in paired:
        favorite_advantages.append(
            (
                event_id,
                _best_break_even(late, favorite) - _best_break_even(early, favorite),
            )
        )
        underdog_advantages.append(
            (
                event_id,
                _best_break_even(early, underdog) - _best_break_even(late, underdog),
            )
        )
        early_by_book = {item.book.casefold(): item for item in early.quotes}
        late_by_book = {item.book.casefold(): item for item in late.quotes}
        common = sorted(set(early_by_book) & set(late_by_book))
        if common:
            fav_values = []
            dog_values = []
            for book in common:
                early_quote = early_by_book[book]
                late_quote = late_by_book[book]
                early_fav_line = (
                    early_quote.fighter_moneyline
                    if favorite == "fighter"
                    else early_quote.opponent_moneyline
                )
                late_fav_line = (
                    late_quote.fighter_moneyline
                    if favorite == "fighter"
                    else late_quote.opponent_moneyline
                )
                early_dog_line = (
                    early_quote.fighter_moneyline
                    if underdog == "fighter"
                    else early_quote.opponent_moneyline
                )
                late_dog_line = (
                    late_quote.fighter_moneyline
                    if underdog == "fighter"
                    else late_quote.opponent_moneyline
                )
                fav_values.append(
                    implied_probability(late_fav_line) - implied_probability(early_fav_line)
                )
                dog_values.append(
                    implied_probability(early_dog_line) - implied_probability(late_dog_line)
                )
            common_favorite.append((event_id, sum(fav_values) / len(fav_values)))
            common_underdog.append((event_id, sum(dog_values) / len(dog_values)))

    def describe(values: list[tuple[str, float]], definition: str) -> dict[str, object]:
        numbers = [item[1] for item in values]
        return {
            "definition": definition,
            "count": len(numbers),
            "mean_probability_price_advantage": (
                sum(numbers) / len(numbers) if numbers else None
            ),
            "median_probability_price_advantage": median(numbers) if numbers else None,
            "heuristic_better_rate": (
                sum(value > 0.0 for value in numbers) / len(numbers) if numbers else None
            ),
            "event_block_interval": _event_block_interval(values, label=definition),
        }

    return {
        "paired_matchups": len(paired),
        "interpretation": "positive values favor early favorites or late underdogs",
        "best_available_price": {
            "favorite_early_minus_late": describe(
                favorite_advantages,
                "late favorite break-even probability minus early; positive favors early",
            ),
            "underdog_late_minus_early": describe(
                underdog_advantages,
                "early underdog break-even probability minus late; positive favors late",
            ),
        },
        "same_book_price": {
            "favorite_early_minus_late": describe(
                common_favorite,
                "same-book late favorite break-even minus early; positive favors early",
            ),
            "underdog_late_minus_early": describe(
                common_underdog,
                "same-book early underdog break-even minus late; positive favors late",
            ),
        },
    }


def _policy_summary(
    selections: list[_Candidate],
    outcomes: Mapping[tuple[str, str, str], tuple[int | None, str]],
    capture_by_id: Mapping[tuple[str, str], _Capture],
    all_captures: Mapping[str, tuple[_Capture, ...]],
) -> dict[str, object]:
    profits: list[tuple[str, float]] = []
    clv: list[tuple[str, float]] = []
    wins = losses = unscored = 0
    action_counts = {"fighter": 0, "opponent": 0}
    entry_window_counts = {"early": 0, "fixed_t24": 0, "late": 0, "other": 0}
    for selection in selections:
        capture = capture_by_id[(selection.capture_id, selection.matchup_id)]
        action_counts[selection.action] += 1
        if EARLIEST_MIN_LEAD_SECONDS <= capture.lead_seconds <= EARLIEST_MAX_LEAD_SECONDS:
            entry_window_counts["early"] += 1
        elif BASELINE_MIN_LEAD_SECONDS <= capture.lead_seconds <= BASELINE_MAX_LEAD_SECONDS:
            entry_window_counts["fixed_t24"] += 1
        elif LATE_MIN_LEAD_SECONDS <= capture.lead_seconds <= LATE_MAX_LEAD_SECONDS:
            entry_window_counts["late"] += 1
        else:
            entry_window_counts["other"] += 1
        outcome = outcomes.get(
            (capture.event_id, capture.fighter_id, capture.opponent_id)
        )
        if outcome is None or outcome[0] is None:
            unscored += 1
        else:
            target = int(outcome[0])
            won = (selection.action == "fighter" and target == 1) or (
                selection.action == "opponent" and target == 0
            )
            profit = _profit_multiple(selection.moneyline) if won else -1.0
            profits.append((selection.event_id, profit))
            if won:
                wins += 1
            else:
                losses += 1
        later = []
        for later_capture in all_captures.get(selection.matchup_id, ()):
            if later_capture.observed_at_utc <= selection.observed_at_utc:
                continue
            for quote in later_capture.quotes:
                if quote.book.casefold() == selection.book.casefold():
                    later.append(quote)
        if later:
            closing = max(later, key=lambda item: (item.observed_at_utc, item.quote_id))
            closing_line = (
                closing.fighter_moneyline
                if selection.action == "fighter"
                else closing.opponent_moneyline
            )
            clv.append(
                (
                    selection.event_id,
                    implied_probability(closing_line) - selection.break_even_probability,
                )
            )
    expected_values = [item.expected_return for item in selections]
    clv_values = [item[1] for item in clv]
    return {
        "paper_only": True,
        "selection_count": len(selections),
        "scored_selection_count": len(profits),
        "unscored_selection_count": unscored,
        "action_counts": action_counts,
        "entry_window_counts": entry_window_counts,
        "wins": wins,
        "losses": losses,
        "hypothetical_profit_units": sum(item[1] for item in profits),
        "hypothetical_roi": (
            sum(item[1] for item in profits) / len(profits) if profits else None
        ),
        "mean_locked_expected_return": (
            sum(expected_values) / len(expected_values) if expected_values else None
        ),
        "return_event_block_interval": _event_block_interval(
            profits,
            label="whole-card bootstrap of flat one-unit hypothetical return per selection",
        ),
        "clv": {
            "definition": (
                "latest fresh same-book implied probability minus locked break-even; "
                "positive favors the selected price"
            ),
            "count": len(clv_values),
            "mean_probability_edge": (
                sum(clv_values) / len(clv_values) if clv_values else None
            ),
            "positive_rate": (
                sum(value > 0.0 for value in clv_values) / len(clv_values)
                if clv_values
                else None
            ),
            "event_block_interval": _event_block_interval(
                clv,
                label="whole-card bootstrap of latest same-book CLV probability edge",
            ),
        },
    }


def evaluate_timing_policies(
    quotes: Iterable[QuoteSnapshot],
    source_metadata: Iterable[QuoteSourceMetadata],
    outcomes: Mapping[tuple[str, str, str], tuple[int | None, str]],
) -> dict[str, object]:
    """Evaluate three frozen timing rules without looking at outcomes to select prices."""

    quote_records = tuple(quotes)
    metadata_records = tuple(source_metadata)
    captures = _eligible_captures(quote_records, metadata_records)
    by_matchup: dict[str, list[_Capture]] = defaultdict(list)
    for capture in captures:
        by_matchup[capture.matchup_id].append(capture)
    frozen_classifications: dict[str, tuple[str, str]] = {}
    paired_price_rows: list[tuple[str, _Capture, _Capture, str, str]] = []
    policy_selections: dict[str, list[_Candidate]] = {
        "fixed_t24": [],
        "favorite_early_underdog_late": [],
        "earliest_available_both": [],
    }
    coverage = {
        "matchups_with_eligible_capture": len(by_matchup),
        "matchups_with_early_capture": 0,
        "matchups_with_t24_capture": 0,
        "matchups_with_late_capture": 0,
        "matchups_classified_pickem": 0,
    }
    normalized_by_matchup: dict[str, tuple[_Capture, ...]] = {}
    for matchup_id in sorted(by_matchup):
        matchup_captures = tuple(
            sorted(by_matchup[matchup_id], key=lambda item: (item.observed_at_utc, item.capture_id))
        )
        normalized_by_matchup[matchup_id] = matchup_captures
        early = _first_in_window(
            matchup_captures, EARLIEST_MIN_LEAD_SECONDS, EARLIEST_MAX_LEAD_SECONDS
        )
        baseline = _first_in_window(
            matchup_captures, BASELINE_MIN_LEAD_SECONDS, BASELINE_MAX_LEAD_SECONDS
        )
        late = _first_in_window(
            matchup_captures, LATE_MIN_LEAD_SECONDS, LATE_MAX_LEAD_SECONDS
        )
        if early is not None:
            coverage["matchups_with_early_capture"] += 1
        if baseline is not None:
            coverage["matchups_with_t24_capture"] += 1
        if late is not None:
            coverage["matchups_with_late_capture"] += 1
        first = _first_in_window(
            matchup_captures, LATE_MIN_LEAD_SECONDS, EARLIEST_MAX_LEAD_SECONDS
        )
        if first is None:
            continue
        earliest_candidate = _candidate(first, ("fighter", "opponent"))
        if earliest_candidate is not None:
            policy_selections["earliest_available_both"].append(earliest_candidate)
        baseline_candidate = _candidate(baseline, ("fighter", "opponent"))
        if baseline_candidate is not None:
            policy_selections["fixed_t24"].append(baseline_candidate)

        probability = _market_probability(first)
        if abs(probability - 0.5) <= 1e-12:
            coverage["matchups_classified_pickem"] += 1
            continue
        favorite = "fighter" if probability > 0.5 else "opponent"
        underdog = "opponent" if favorite == "fighter" else "fighter"
        frozen_classifications[matchup_id] = (favorite, underdog)

        # This branch is causal: lock a qualifying favorite at the early
        # capture; only if none exists does the policy wait for a late dog.
        combined_candidate = _candidate(early, (favorite,))
        if combined_candidate is None:
            combined_candidate = _candidate(late, (underdog,))
        if combined_candidate is not None:
            policy_selections["favorite_early_underdog_late"].append(combined_candidate)
        if early is not None and late is not None:
            paired_price_rows.append((early.event_id, early, late, favorite, underdog))

    capture_by_id = {
        (item.capture_id, item.matchup_id): item for item in captures
    }
    policies = {
        name: _policy_summary(
            selections,
            outcomes,
            capture_by_id,
            normalized_by_matchup,
        )
        for name, selections in policy_selections.items()
    }
    return {
        "policy_version": TIMING_POLICY_VERSION,
        "status": "collecting_prospective_evidence",
        "paper_only": True,
        "execution_enabled": False,
        "favorite_definition": (
            "side above 50% in the full no-vig consensus at the first eligible "
            "capture; classification never changes after a line flip"
        ),
        "selection_contract": {
            "minimum_leave_one_out_consensus_books": MIN_CONSENSUS_BOOKS,
            "minimum_expected_return": MIN_EXPECTED_RETURN,
            "maximum_source_quote_age_seconds": MAX_SOURCE_QUOTE_AGE_SECONDS,
            "early_lead_hours": [
                EARLIEST_MIN_LEAD_SECONDS / 3600.0,
                EARLIEST_MAX_LEAD_SECONDS / 3600.0,
            ],
            "fixed_t24_lead_hours": [
                BASELINE_MIN_LEAD_SECONDS / 3600.0,
                BASELINE_MAX_LEAD_SECONDS / 3600.0,
            ],
            "late_lead_hours": [
                LATE_MIN_LEAD_SECONDS / 3600.0,
                LATE_MAX_LEAD_SECONDS / 3600.0,
            ],
            "window_selection": "first successful scheduled capture to enter each window",
            "one_selection_per_matchup_per_policy": True,
        },
        "ledger_counts": {
            "quotes": len(quote_records),
            "source_metadata": len(metadata_records),
            "eligible_captures": len(captures),
            "frozen_classifications": len(frozen_classifications),
        },
        "coverage": coverage,
        "price_timing_test": _price_timing_summary(paired_price_rows),
        "shadow_policies": policies,
    }
