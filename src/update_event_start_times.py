"""Publish official UFC broadcast start times without changing betting cutoffs."""
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent / "content/data/external"
SOURCE_URL = "https://www.ufc.com/events"
STAGES = {"early_prelims": "early", "prelims": "prelims", "main_card": "main"}


def normalized(value):
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def parse_schedule(html):
    events = []
    for card in BeautifulSoup(html, "html.parser").select(".c-card-event--result"):
        dates = card.select_one(".c-card-event--result__date")
        link = card.select_one('a[href^="/event/"]')
        headline = card.select_one(".c-card-event--result__headline")
        if dates is None or link is None or headline is None:
            continue
        starts = {}
        for key, attribute in STAGES.items():
            raw = dates.get(f"data-{attribute}-card-timestamp", "")
            try:
                stamp = datetime.fromtimestamp(int(raw), timezone.utc)
                if 2000 <= stamp.year <= 2100:
                    starts[key] = stamp.isoformat().replace("+00:00", "Z")
            except (ValueError, OverflowError, OSError):
                pass
        if "main_card" not in starts:
            continue
        events.append({"headline": headline.get_text(" ", strip=True),
                       "source_url": "https://www.ufc.com" + link["href"],
                       "starts": starts})
    if not events:
        raise ValueError("No official event start times found")
    return events


def match_event(event, schedules):
    matches = []
    for schedule in schedules:
        main_date = datetime.fromisoformat(schedule["starts"]["main_card"]).date()
        event_date = datetime.fromisoformat(event["event_date"]).date()
        # Main cards can begin after midnight UTC. Require the event identity too.
        if abs((main_date - event_date).days) > 1:
            continue
        title = normalized(event["event_title"])
        headline = normalized(schedule["headline"])
        numbered = re.search(r"/ufc-(\d+)$", schedule["source_url"])
        same_number = numbered and re.search(rf"\bufc {numbered[1]}\b", title)
        if same_number or (headline != "tbd vs tbd" and title.endswith(headline)):
            matches.append(schedule)
    return matches[0] if len(matches) == 1 else None


def build_publication(upcoming, schedules, observed_at):
    events = []
    for event in upcoming["events"]:
        matched = match_event(event, schedules)
        events.append({"event_id": event["event_id"], "event_date": event["event_date"],
                       "event_title": event["event_title"],
                       "starts": matched["starts"] if matched else {},
                       "source_url": matched["source_url"] if matched else SOURCE_URL,
                       "observed_at_utc": observed_at})
    return {"schema_version": 1, "generated_at_utc": observed_at,
            "source_url": SOURCE_URL, "refresh_failed": False, "events": events}


def main():
    output = ROOT / "event_start_times.json"
    upcoming = json.loads((ROOT / "all_upcoming_forecasts.json").read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        response = requests.get(SOURCE_URL, timeout=30)
        response.raise_for_status()
        publication = build_publication(upcoming, parse_schedule(response.text), now)
    except (requests.RequestException, ValueError) as error:
        # A schedule outage must not block model/market publication or erase known times.
        publication = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"schema_version": 1, "events": []}
        publication.update(refresh_failed=True, last_attempt_at_utc=now)
        print(f"Official start-time refresh unavailable; retaining last confirmed times: {error}")
    output.write_text(json.dumps(publication, indent=2) + "\n", encoding="utf-8")
    print(f"Official start times: {sum(bool(e['starts']) for e in publication['events'])} cards")


if __name__ == "__main__":
    main()
