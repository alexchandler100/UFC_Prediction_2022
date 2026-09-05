"""Prospective moneyline comparison; fixed one-unit paper stakes, no execution."""
from __future__ import annotations

from collections import defaultdict
import random

from ._common import MarketDataError, canonical_hash, utc_datetime
from .paper import PaperDecision, _profit_for_one_unit_risk
from .quotes import consensus_as_of

VERSION = "prospective-equal-stake-moneylines-v1"
STRATEGIES = ("market", "adjusted_market", "production_model")


def seal(value):
    return {**value, "record_sha256": canonical_hash(value)}


def verify(value):
    body = dict(value)
    digest = body.pop("record_sha256", None)
    if digest != canonical_hash(body):
        raise ValueError("experiment record hash mismatch")
    return value


def build_records(quotes, forecasts, metadata, existing, policy, calibrator, now):
    """Freeze the first usable capture per physical matchup after activation."""
    now = utc_datetime(now, "now")
    start = utc_datetime(policy["activated_at_utc"], "activation")
    grouped = defaultdict(list)
    for quote in quotes:
        observed = utc_datetime(quote.observed_at_utc, "observed")
        if observed >= start and 0 <= (now - observed).total_seconds() <= 300:
            grouped[(quote.observed_at_utc, quote.capture_id, quote.matchup_id)].append(quote)
    forecast_index = {}
    for forecast in forecasts:
        key = (forecast.capture_id, forecast.matchup_id)
        if key in forecast_index:
            raise ValueError("duplicate matchup forecast in capture")
        forecast_index[key] = forecast
    meta = {item.quote_id: item for item in metadata}
    if len(meta) != len(metadata):
        raise ValueError("duplicate quote metadata")
    seen = {item["matchup_id"] for item in existing}
    pending = []
    for (_, capture, matchup), group in sorted(grouped.items()):
        if matchup in seen:
            continue
        forecast = forecast_index.get((capture, matchup))
        if forecast is None or forecast.probability_provenance != "native_probability":
            continue
        if utc_datetime(forecast.forecast_issued_at_utc, "forecast") > utc_datetime(group[0].observed_at_utc, "observed"):
            continue
        fresh = []
        for quote in group:
            source = meta.get(quote.quote_id)
            if source is None or not quote.event_start_utc or quote.timing_precision != "timestamp":
                continue
            if any(getattr(source, key) != getattr(quote, key) for key in
                   ("capture_id", "matchup_id", "event_id", "book", "source", "observed_at_utc")):
                raise ValueError("quote metadata identity mismatch")
            updated = utc_datetime(source.source_quote_updated_at_utc, "provider quote")
            observed = utc_datetime(quote.observed_at_utc, "observed")
            event = utc_datetime(quote.event_start_utc, "event")
            if utc_datetime(source.source_commence_time_utc, "provider start") != event:
                continue
            if not (0 <= (observed - updated).total_seconds() <= 1800
                    and 0 <= (now - updated).total_seconds() <= 1800
                    and 20 * 3600 <= (event - observed).total_seconds() <= 28 * 3600
                    and event > now):
                continue
            fresh.append(quote)
        if len({q.book.casefold() for q in fresh}) < 4:
            continue
        offers = []
        for quote in sorted(fresh, key=lambda q: (q.book.casefold(), q.quote_id)):
            try:
                market = consensus_as_of(fresh, capture_id=capture, matchup_id=matchup,
                    as_of_utc=quote.observed_at_utc, min_books=3, exclude_books=(quote.book,))
                checked = PaperDecision.create(market, quote, forecast, selected_gamma=0,
                    decision_issued_at_utc=now, maximum_quote_age_seconds=300)
            except MarketDataError:
                continue
            p = checked.market_probability
            adjusted = calibrator.assessment(p, quote.fighter_moneyline)["posterior_mean_probability"]
            for side, line, complement in (("fighter", quote.fighter_moneyline, False),
                                           ("opponent", quote.opponent_moneyline, True)):
                probabilities = dict(zip(STRATEGIES, (p, adjusted, checked.model_probability)))
                if complement:
                    probabilities = {key: 1 - value for key, value in probabilities.items()}
                offers.append({"book": quote.book, "side": side, "moneyline": line,
                    "probabilities": probabilities, "quote_id": quote.quote_id,
                    "market_consensus_id": market.consensus_id,
                    "consensus_quote_ids": list(market.quote_ids),
                    "source_metadata": meta[quote.quote_id].to_mapping(),
                    "expected_returns": {key: value * (1 + _profit_for_one_unit_risk(line)) - 1
                                         for key, value in probabilities.items()}})
        if offers:
            row = seal({"policy_sha256": policy["record_sha256"], "matchup_id": matchup,
                "event_id": forecast.event_id, "event_date": forecast.event_date,
                "event_start_utc": forecast.event_start_utc,
                "fighter_id": forecast.fighter_id, "opponent_id": forecast.opponent_id,
                "recorded_at_utc": now.isoformat(), "capture_id": capture,
                "forecast": forecast.to_mapping(), "offers": offers})
            pending.append(row)
            seen.add(matchup)
    return pending


def choose(record, strategy, book, minimum_ev):
    if strategy == "no_bet":
        return None
    candidates = [offer for offer in record["offers"]
                  if (book is None or offer["book"].casefold() == book.casefold())
                  and offer["expected_returns"][strategy] >= minimum_ev]
    return min(candidates, key=lambda offer: (-offer["expected_returns"][strategy],
               offer["book"].casefold(), offer["side"], offer["quote_id"]), default=None)


def report(records, settlements, policy):
    """Card-level accounting; pending bets never enter settled returns."""
    settled = {item["matchup_id"]: item for item in settlements}
    books = sorted({offer["book"] for row in records for offer in row["offers"]})
    results = []
    for book in [None, *books]:
        available = [row for row in records if book is None or
                     any(offer["book"] == book for offer in row["offers"])]
        for strategy in ("no_bet", *STRATEGIES):
            for haircut in (0, .02, .05):
                cards = defaultdict(lambda: [0., 0.])
                bets = pending = voids = 0
                for row in available:
                    selection = choose(row, strategy, book, policy["minimum_expected_return"])
                    outcome = settled.get(row["matchup_id"])
                    if outcome is None:
                        pending += int(selection is not None)
                        continue
                    card = cards[(row["event_start_utc"], row["event_id"])]
                    if selection is None:
                        continue
                    if outcome["target"] is None:
                        voids += 1
                        continue
                    bets += 1
                    won = outcome["target"] == int(selection["side"] == "fighter")
                    card[0] += 1
                    card[1] += _profit_for_one_unit_risk(selection["moneyline"]) * (1 - haircut) if won else -1
                risk = sum(card[0] for card in cards.values())
                profit = sum(card[1] for card in cards.values())
                balance = peak = 100.
                drawdown = 0.
                for key in sorted(cards):
                    balance += cards[key][1]
                    peak = max(peak, balance)
                    drawdown = max(drawdown, (peak - balance) / peak)
                interval = None
                if len(cards) >= 2 and risk:
                    generator = random.Random(20260904)
                    blocks = list(cards.values())
                    samples = []
                    for _ in range(2000):
                        sample = generator.choices(blocks, k=len(blocks))
                        sampled_risk = sum(block[0] for block in sample)
                        if sampled_risk:
                            samples.append(sum(block[1] for block in sample) / sampled_risk)
                    samples.sort()
                    if samples:
                        interval = [samples[int(.025 * (len(samples) - 1))],
                                    samples[int(.975 * (len(samples) - 1))]]
                results.append({"book": book or "all_books_hypothetical", "strategy": strategy,
                    "winning_payout_reduction": haircut, "available_fights": len(available),
                    "settled_cards": len(cards), "settled_bets": bets, "pending_bets": pending,
                    "void_bets": voids, "risk_units": risk, "profit_units": profit,
                    "return_per_unit": profit / risk if risk else None,
                    "ending_normalized_bankroll": balance, "bankroll_growth": profit / 100,
                    "max_card_end_drawdown": drawdown, "card_bootstrap_roi_95": interval})
    return {"policy": policy, "paper_only": True, "execution_enabled": False,
            "status": "collecting_results", "frozen_fights": len(records),
            "settled_fights": len(settlements), "results": results,
            "model_ids": sorted({row["forecast"]["model_id"] for row in records}),
            "records_sha256": canonical_hash(records), "settlements_sha256": canonical_hash(settlements)}
