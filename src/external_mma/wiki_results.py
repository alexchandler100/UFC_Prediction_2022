"""Strict professional-MMA result tables from revision-pinned MediaWiki exports."""
from datetime import date
from hashlib import sha256
import json
import re
from urllib.parse import unquote

from bs4 import BeautifulSoup
import pandas as pd

from .adapters import AdapterResult
from .identity import normalize_name
from .schema import ExternalBoutObservation, ExternalDataError, clean_text

SOURCE = 'wikipedia_event_results_v1'
BLOCKED_NAMESPACES = ('file:', 'category:', 'help:', 'template:', 'wikipedia:', 'special:', 'portal:', 'draft:')
MONTH = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
DATE = re.compile(r'\d{4}-\d{2}-\d{2}|' + MONTH + r'\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+' + MONTH + r'\s+\d{4}', re.I)


def explicit_date(text):
    match = DATE.search(text)
    if not match:
        return None
    try:
        return pd.to_datetime(match.group(), errors='raise').date().isoformat()
    except (ValueError, TypeError):
        return None


def fighter(cell, event_id):
    name = clean_text(cell.get_text(' ', strip=True))
    link = next((a for a in cell.find_all('a', href=True)
                 if a['href'].startswith('/wiki/') and not unquote(a['href'][6:]).casefold().startswith(BLOCKED_NAMESPACES)
                 and 'redlink=1' not in a['href'] and a.get_text(' ',strip=True)
                 and a.find('img') is None), None)
    # Names without a source profile are event-scoped, never silently pooled.
    identity = unquote(link['href'][6:]) if link else 'unresolved:' + event_id + ':' + normalize_name(name)
    return identity, name


def parse_page(payload, promotion, through):
    parsed = payload['parse']
    soup = BeautifulSoup(parsed['text']['*'], 'html.parser')
    for tag in soup.select('sup.reference'):
        tag.decompose()
    revision = int(parsed['revid'])
    digest = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    accepted, rejected = {}, []
    infobox = None
    for table in soup.find_all('table'):
        if 'infobox' in table.get('class', []):
            values = {}
            for tr in table.find_all('tr'):
                th, td = tr.find('th'), tr.find('td')
                if th and td:
                    values[th.get_text(' ', strip=True)] = td.get_text(' ', strip=True)
            when = explicit_date(values.get('Date', ''))
            infobox = (when, table.find('caption').get_text(' ', strip=True) if table.find('caption') else parsed['title']) if when else None
            continue
        rows = table.find_all('tr')
        candidates = []
        for tr in rows:
            cells = tr.find_all(['td','th'], recursive=False)
            if len(cells) < 7:
                continue
            connector = cells[2].get_text(' ', strip=True).casefold()
            if connector not in ('def.', 'def', 'draw', 'draws', 'nc', 'n/c', 'no contest'):
                continue
            candidates.append((cells, connector))
        if not candidates:
            continue
        heading = table.find_previous(['h2', 'h3'])
        # Championship summary tables repeat bouts and are not event cards.
        if heading and any(t in heading.get_text(' ', strip=True).casefold() for t in ('title fights','championship history')):
            continue
        if infobox is None:
            rejected.append({'reason':'result_table_without_explicit_event_date','table_heading':heading.get_text(' ',strip=True) if heading else ''})
            continue
        when, event_name = infobox
        if when > through:
            rejected.append({'reason':'future_event','event_date':when})
            continue
        event_id = promotion + ':' + when
        for cells, connector in candidates:
            division = clean_text(cells[0].get_text(' ', strip=True))
            combined = ' '.join(c.get_text(' ',strip=True) for c in cells).casefold()
            # ONE and Rizin mix disciplines; ONE must explicitly identify MMA.
            if (any(word in combined for word in ('muay thai','kickboxing','submission grappling','exhibition','amateur','bare knuckle','shoot boxing','special rules'))
                    or (promotion == 'ONE' and not re.search(r'\bmma\b', division, re.I))
                    or re.search(r'\bboxing\b', division, re.I)):
                rejected.append({'reason':'non_mma_or_unverified_discipline','event_date':when,'division':division})
                continue
            first_id, first = fighter(cells[1], event_id)
            second_id, second = fighter(cells[3], event_id)
            method = clean_text(cells[4].get_text(' ',strip=True))
            result = 'D' if connector in ('draw','draws') else 'NC' if connector in ('nc','n/c','no contest') or 'no contest' in method.casefold() else 'W'
            key = sha256(json.dumps([promotion,when,sorted([normalize_name(first),normalize_name(second)])],sort_keys=True).encode()).hexdigest()
            try:
                observation = ExternalBoutObservation.create(source=SOURCE,snapshot_sha256=digest,
                    source_bout_id=key,source_event_id=event_id,source_url=f'https://en.wikipedia.org/w/index.php?oldid={revision}',
                    event_date=when,event_name=event_name,promotion=promotion,
                    fighter_source_id=first_id,fighter_name=first,opponent_source_id=second_id,opponent_name=second,
                    result=result,method=method,division=re.sub(r'\s+MMA\b','',division,flags=re.I),
                    finish_round=cells[5].get_text(' ',strip=True) or None,
                    finish_clock_seconds=cells[6].get_text(' ',strip=True) or None)
                if key in accepted and accepted[key] != observation:
                    raise ExternalDataError('conflicting duplicate event result')
                accepted[key] = observation
            except (ValueError, TypeError) as error:
                rejected.append({'reason':str(error),'event_date':when,'fighter':first,'opponent':second})
    return AdapterResult(list(accepted.values()), rejected, len(accepted)+len(rejected))


class WikiResultsAdapter:
    source_key = SOURCE

    def __init__(self, promotion, through, existing=()):
        self.promotion, self.through = promotion, through
        self.existing = {row.observation_id:row for row in existing}

    def convert(self, content, snapshot_sha256):
        result = parse_page(json.loads(content), self.promotion, self.through)
        kept = []
        for row in result.observations:
            # Source hash is over the exact imported bytes, not reserialized JSON.
            value = row.to_dict(); value['snapshot_sha256'] = snapshot_sha256
            row = ExternalBoutObservation.from_mapping(value)
            old = self.existing.get(row.observation_id)
            if old:
                a,b = old.to_dict(),row.to_dict()
                for k in ('snapshot_sha256','source_url','event_name'):
                    a.pop(k);b.pop(k)
                if a != b:
                    result.rejected.append({'reason':'source_revision_changed_result_or_identity','observation_id':row.observation_id})
                # Never rewrite old observations, including their attribution.
                continue
            kept.append(row)
        result.observations = kept
        return result


def discover_event_pages(payload, through):
    soup = BeautifulSoup(payload['parse']['text']['*'],'html.parser')
    pages = set()
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        if not rows:
            continue
        headers = [c.get_text(' ',strip=True).casefold() for c in rows[0].find_all(['th','td'],recursive=False)]
        if 'event' not in headers or 'date' not in headers:
            continue
        for tr in rows[1:]:
            cells = tr.find_all(['th','td'],recursive=False)
            if len(cells) <= max(headers.index('event'),headers.index('date')):
                continue
            when = explicit_date(cells[headers.index('date')].get_text(' ',strip=True))
            if not when or when > through:
                continue
            for a in cells[headers.index('event')].find_all('a',href=True):
                if a['href'].startswith('/wiki/') and not unquote(a['href'][6:]).casefold().startswith(BLOCKED_NAMESPACES):
                    pages.add(unquote(a['href'][6:]).replace('_',' '))
    return sorted(pages)
