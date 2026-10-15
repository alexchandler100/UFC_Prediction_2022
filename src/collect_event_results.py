"""Collect attributed, revision-pinned MMA event results into a separate research ledger."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd
import requests

from external_mma import ExternalMmaStore, propose_ufcstats_crosswalk
from external_mma.wiki_results import SOURCE, WikiResultsAdapter, discover_event_pages
from market_tracker._storage import atomic_write_text

ROOT = Path(__file__).resolve().parent / 'content/data/external_mma/event_results'
SERIES = {'PFL':'Professional Fighters League','Bellator':'Bellator MMA','ONE':'ONE Championship',
          'Rizin':'Rizin Fighting Federation','KSW':'Konfrontacja Sztuk Walki','ACA':'Absolute Championship Akhmat',
          'OKTAGON':'Oktagon MMA','Cage Warriors':'Cage Warriors','LFA':'Legacy Fighting Alliance',
          'Invicta':'Invicta FC','BRAVE':'Brave Combat Federation','UFC':'UFC'}
API = 'https://en.wikipedia.org/w/api.php'


def seeds(years):
    result = [(f'Road to UFC Season {n}', 'Road to UFC') for n in range(1,max(datetime.now(timezone.utc).year-2021,1)+1)]
    result += [(f"Dana White's Contender Series season {n}", 'Contender Series') for n in range(1,max(datetime.now(timezone.utc).year-2016,1)+1)]
    result += [(f'{year} in {name}', promotion) for year in sorted(years,reverse=True) for promotion,name in SERIES.items()]
    return result


def write(path,value):
    atomic_write_text(path,json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')


def coverage(root, manifest):
    store=ExternalMmaStore(root); validation=store.validate(); rows=store.observations()
    promotions=[]
    for promotion in ['Contender Series','Road to UFC',*SERIES]:
        subset=[r for r in rows if r.promotion==promotion]
        promotions.append({'promotion':promotion,'bouts':len(subset),
            'events':len({r.source_event_id for r in subset}),
            'first_event_date':min((r.event_date for r in subset),default=None),
            'last_event_date':max((r.event_date for r in subset),default=None),
            'unresolved_participant_occurrences':sum(fid.startswith('unresolved:') for r in subset for fid in (r.fighter_source_id,r.opponent_source_id)),
            'status':'collected_research_only' if subset else 'no_usable_results_collected'})
    report={'version':'revision-pinned-professional-mma-results-v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'paper_only':True,'production_enabled':False,'validation':validation,'promotions':promotions,
        'sources':manifest,'license':'CC BY-SA 4.0; source revision attribution retained on every bout',
        'limitations':['This is a secondary-source research ledger; coverage is not claimed complete.',
            'ONE requires explicit MMA designation; other mixed-discipline rows are excluded.',
            'No detailed strike, takedown, control statistics or scheduled rounds are invented.',
            'Unlinked fighter names are event-scoped and cannot silently merge careers.',
            'Only unique dated UFC bout witnesses approve UFCStats identity links.',
            'Changed source results or identities are quarantined rather than silently rewriting observations.',
            'Historical result metadata does not establish when it became available before later fights. No automatic production replay.']}
    write(root/'coverage.json',report)
    return report


def official_crosschecks(root=ROOT):
    """Sample primary-source checks; never claim the whole secondary ledger is verified."""
    from bs4 import BeautifulSoup
    from hashlib import sha256
    from external_mma.identity import normalize_name
    import re
    rows=ExternalMmaStore(root).observations()
    sources=[('2026-10-13','https://www.ufc.com/news/dana-whites-contender-series-season-10-week-10-results'),
             ('2026-05-28','https://www.ufc.com/news/road-to-ufc-season-5-results-scorecards-opening-rounds-day-1')]
    checks=[];seen=set()
    for when,url in sources:
        response=requests.get(url,timeout=(10,45))
        if not response.ok:
            checks.append({'source_url':url,'status':'unavailable','http_status':response.status_code});continue
        soup=BeautifulSoup(response.content,'html.parser')
        normalized=lambda s:normalize_name(s).replace(' ','')
        for heading in soup.find_all(['h2','h3','h4','strong']):
            text=heading.get_text(' ',strip=True)
            match=re.search(r'^(.+?) (?:defeats|defeated) (.+?) (?:via|by) (.+)$',text,re.I)
            if not match:
                continue
            winner,loser=match.group(1),match.group(2)
            key=(url,normalized(winner),normalized(loser))
            if key in seen:continue
            seen.add(key)
            found=[r for r in rows if r.event_date==when and
                {normalized(r.fighter_name),normalized(r.opponent_name)}=={normalized(winner),normalized(loser)}]
            matched=[r for r in found if (r.fighter_name if r.result=='W' else r.opponent_name if r.result=='L' else '') and
                normalized(r.fighter_name if r.result=='W' else r.opponent_name)==normalized(winner)]
            checks.append({'source_url':url,'source_payload_sha256':sha256(response.content).hexdigest(),
                'event_date':when,'winner':winner,'opponent':loser,
                'status':'winner_confirmed' if len(matched)==1 else 'not_matched' if not found else 'conflict_or_ambiguous',
                'observation_ids':[r.observation_id for r in found]})
    report={'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Sample checks of winner and dated opponent pair only; not full validation of methods, clocks or every collected bout.',
        'checks':checks}
    write(root/'official_crosschecks.json',report)
    print(json.dumps({'official_sample_checks':len(checks),'winner_confirmed':sum(r['status']=='winner_confirmed' for r in checks)}))
    return report


def collect(root=ROOT, *, years=range(2021,2027), max_pages=180, refresh_all=False, validate_only=False, promotions=None):
    if validate_only:
        actual=ExternalMmaStore(root).validate()
        saved=json.loads((root/'coverage.json').read_text(encoding='utf-8'))
        if saved['validation'] != actual:
            raise ValueError('Research coverage does not match the result ledgers')
        print(json.dumps(actual,sort_keys=True));return
    root.mkdir(parents=True,exist_ok=True)
    registry={'schema_version':1,'policy':'Revision-pinned factual MMA results under attributed CC BY-SA 4.0; research only.',
        'sources':[{'key':SOURCE,'collection_status':'licensed_export','license':'Creative Commons Attribution-ShareAlike 4.0',
            'source_page':'https://en.wikipedia.org/wiki/Wikipedia:Copyrights','adapter':'mediawiki_event_results',
            'notes':'Public MediaWiki revision exports. Keep attribution; event-scoped unlinked names are not resolved identities.'}]}
    write(root/'source_registry.json',registry)
    atomic_write_text(root/'.gitignore','raw/\n')
    manifest_path=root/'pages.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    session=requests.Session();session.headers['User-Agent']='UFCDataResearch/1.0 (bounded research event collection)'
    queue=[(title,promotion) for title,promotion in seeds(years) if promotions is None or promotion in promotions]
    seen=set(); fetched=0; current_year=datetime.now(timezone.utc).year
    through=datetime.now(timezone.utc).date().isoformat();store=ExternalMmaStore(root)
    while queue and fetched < max_pages:
        title,promotion=queue.pop(0)
        if title in seen:
            continue
        seen.add(title)
        old=manifest.get(title,{})
        is_current=(str(current_year) in title or title.endswith(f'season {current_year-2016}') or title.endswith(f'Season {current_year-2021}'))
        recent = old.get('retrieved_at_utc') and (datetime.now(timezone.utc)-datetime.fromisoformat(old['retrieved_at_utc'])).total_seconds()<86400
        if old.get('status')=='imported' and not refresh_all and (not is_current or recent):
            queue.extend((t,promotion) for t in old.get('event_pages',[]));continue
        # Previously missing annual pages are retried; never reinterpret a 403 as no results.
        try:
            time.sleep(5)
            fetched+=1
            response=session.get(API,params={'action':'parse','page':title,'prop':'text|revid','format':'json','redirects':1},timeout=(10,45))
            if response.status_code in (403,429):
                manifest[title]={'status':'deferred','promotion':promotion,
                    'reason':f'HTTP {response.status_code}; stopped all requests; retry_after={response.headers.get("Retry-After", "unspecified")}' }
                print(json.dumps(manifest[title]),flush=True)
                break
            response.raise_for_status();payload=response.json()
            if 'error' in payload:
                manifest[title]={'status':'unavailable','promotion':promotion,'reason':payload['error'].get('code','api_error')}
                continue
            if 'parse' not in payload or 'text' not in payload['parse']:
                raise ValueError('unexpected MediaWiki response')
            # Match the exact canonical bytes hashed by the store.
            content=json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()
            result=store.import_bytes(WikiResultsAdapter(promotion,through,store.observations()),content)
            event_pages=discover_event_pages(payload,through)
            queue.extend((t,promotion) for t in event_pages)
            manifest[title]={'status':'imported','promotion':promotion,'revision':payload['parse']['revid'],
                'page_id':payload['parse']['pageid'],'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
                'snapshot_sha256':result['snapshot_sha256'],'added':result.get('added',0),
                'rejected':result['rejected'],'event_pages':event_pages}
            print(json.dumps({'page':title,'added':result.get('added',0),'rejected':result['rejected']}),flush=True)
        except (requests.RequestException,ValueError) as error:
            manifest[title]={'status':'failed','promotion':promotion,'reason':str(error)}
            print(json.dumps({'page':title,'error':str(error)}),flush=True)
        write(manifest_path,manifest)
    write(manifest_path,manifest)
    report=coverage(root,manifest)
    raw_path=Path(__file__).resolve().parent/'content/data/processed/ufc_fights_reported_doubled.csv'
    if raw_path.exists():
        approved,review=propose_ufcstats_crosswalk(store.observations(),pd.read_csv(raw_path,low_memory=False))
        # This separate evidence file never overwrites the production identity map.
        from data_handler.data_handler import atomic_to_csv
        atomic_to_csv(approved,root/'identity_map.csv',index=False)
        atomic_to_csv(review,root/'identity_review_queue.csv',index=False)
        report['approved_ufcstats_identity_links']=len(approved)
        report['conflicting_identity_links']=len(review)
        write(root/'coverage.json',report)
    print(json.dumps({'observations':report['validation']['observations'],'pages_requested':fetched,'remaining_pages':len(queue)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--start-year',type=int,default=2021)
    parser.add_argument('--end-year',type=int,default=datetime.now(timezone.utc).year)
    parser.add_argument('--max-pages',type=int,default=180)
    parser.add_argument('--refresh-all',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--cross-check',action='store_true')
    parser.add_argument('--promotions',help='Comma-separated promotion names; defaults to every configured source')
    args=parser.parse_args()
    if args.cross_check:
        official_crosschecks(args.root)
    else:
        collect(args.root,years=range(args.start_year,args.end_year+1),max_pages=args.max_pages,refresh_all=args.refresh_all,validate_only=args.validate_only,
            promotions=set(args.promotions.split(',')) if args.promotions else None)
