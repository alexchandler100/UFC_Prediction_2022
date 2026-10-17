"""Bridge series identities with a dated series bout and a separate UFC witness."""
from collections import defaultdict, Counter
from datetime import date, datetime, timezone
import json
import re
from pathlib import Path

import pandas as pd

from external_mma.identity import IDENTITY_COLUMNS, normalize_name
from external_mma.integration import load_identity_map, is_ufc_promotion
from external_mma.storage import ExternalMmaStore
from market_tracker._storage import atomic_write_text
from online_data_research import MMA_SOURCE_KEY, _mma_result, file_sha256
from external_mma.schema import ExternalBoutObservation

DATA=Path(__file__).resolve().parent/'content/data'


def identity_name(name):
    # Discovery still requires a unique date + BOTH participants and a separate
    # UFC witness. This accommodates Korean name order and HyunSung/Hyun-sung.
    return ' '.join(sorted(normalize_name(re.sub(r'(?<=[a-z])(?=[A-Z])',' ',name)).split()))


def match_key(date,a,b):
    return (date,tuple(sorted((identity_name(a),identity_name(b)))))


def identity_witnesses(fights, masters, digest, through):
    """Identity evidence only: an unknown opponent must not erase a known fighter.

    Unknown/ambiguous directory names get bout-local IDs, never a shared name ID.
    These observations are not exported into the career-results archive.
    """
    names=defaultdict(set);exact_names=defaultdict(set)
    for row in masters.itertuples(index=False):
        if pd.notna(row.fighter_name) and pd.notna(row.fighter_id):
            names[identity_name(row.fighter_name)].add(str(row.fighter_id))
            exact_names[row.fighter_name].add(str(row.fighter_id))
    output=[]; counts=Counter()
    for row in fights.itertuples(index=False):
        day=pd.to_datetime(row.event_date,errors='coerce')
        if pd.isna(day) or day.date()>through:continue
        if not isinstance(row.fighter_1,str) or not isinstance(row.fighter_2,str):continue
        result=_mma_result(row.winner,row.fighter_1,row.fighter_2)
        if result is None:continue
        ids=[]
        for side,name in enumerate((row.fighter_1,row.fighter_2)):
            # Preserve the archive's exact directory lookup. Only fall back to
            # formatting normalization when there is no exact directory entry.
            candidates=exact_names.get(name,names[identity_name(name)])
            ids.append(next(iter(candidates)) if len(candidates)==1 else f'unresolved:{row.fight_id}:{side}')
            counts['directory_resolved' if len(candidates)==1 else 'bout_local_participant']+=1
        if ids[0]==ids[1]:continue
        output.append(ExternalBoutObservation.create(source=MMA_SOURCE_KEY,snapshot_sha256=digest,
            source_bout_id=row.fight_id,source_event_id=str(row.event_name)+' '+str(day.date()),
            source_url=f'https://github.com/LeandroIber/Database-complete-mma#fight-{row.fight_id}',
            event_date=str(day.date()),event_name=row.event_name,promotion=row.organization,
            fighter_source_id=ids[0],fighter_name=row.fighter_1,opponent_source_id=ids[1],
            opponent_name=row.fighter_2,result=result,method=row.method or 'OTHER'))
    return output,dict(counts)


def load_identity_witnesses(path):
    import duckdb
    with duckdb.connect(str(path),read_only=True) as connection:
        fights=connection.execute("""select * from fights_career_longitudinal
            where lower(organization) = 'ufc'
               or lower(event_name) like '%contender%'
               or lower(event_name) like '%road to ufc%'""").fetchdf()
        masters=connection.execute('select fighter_id, fighter_name from fighters_master').fetchdf()
    digest=file_sha256(path)
    rows,audit=identity_witnesses(fights,masters,digest,date.today())
    return rows,{'source_sha256':digest,'source_fights':len(fights),
        'identity_evidence_bouts':len(rows),**audit}


def extend_archive_crosswalk(archive, raw, crosswalk, now):
    index=defaultdict(list)
    for _,group in raw.groupby('fight_url',sort=False):
        if len(group)!=2:continue
        r=group.iloc[0]
        index[match_key(str(r.date)[:10],r.fighter,r.opponent)].append(group)
    witnesses=defaultdict(list)
    for r in archive:
        if not is_ufc_promotion(r.promotion):continue
        matches=index.get(match_key(r.event_date,r.fighter_name,r.opponent_name),[])
        if len(matches)!=1:continue
        group=matches[0]
        for fid,name in ((r.fighter_source_id,r.fighter_name),(r.opponent_source_id,r.opponent_name)):
            if fid.startswith('unresolved:'):continue
            rows=group.loc[group.fighter.map(identity_name).eq(identity_name(name))]
            if len(rows)!=1:continue
            actual=rows.iloc[0]
            expected=r.result if fid==r.fighter_source_id else {'W':'L','L':'W','D':'D','NC':'NC'}[r.result]
            if actual.result!=expected:continue
            witnesses[(r.source,fid)].append((str(actual.fighter_url).rstrip('/').rsplit('/',1)[-1],name,r.source_bout_id,str(actual.fight_url)))
    existing={(r.source,r.source_fighter_id):r.canonical_fighter_id for r in crosswalk.itertuples(index=False)}
    additions=[]
    for (source,fid),proof in sorted(witnesses.items()):
        ids={p[0] for p in proof}
        if len(ids)!=1:continue
        canonical=next(iter(ids))
        if (source,fid) in existing:
            if existing[(source,fid)]!=canonical:raise ValueError('Name-order witness conflicts with existing identity')
            continue
        additions.append({'source':source,'source_fighter_id':fid,'source_fighter_name':proof[0][1],
            'canonical_fighter_id':canonical,'canonical_source':'ufcstats','status':'approved',
            'match_method':'dated_ufc_pair_name_token_order','evidence_count':len(proof),
            'reviewed_at_utc':now.isoformat(),'notes':json.dumps([{'archive_bout_id':p[2],'ufc_fight_url':p[3]} for p in proof],sort_keys=True)})
    return pd.concat([crosswalk,pd.DataFrame(additions,columns=IDENTITY_COLUMNS)],ignore_index=True)


def propose(series, archive, crosswalk, existing, now):
    index=defaultdict(list)
    for bout in archive:index[match_key(bout.event_date,bout.fighter_name,bout.opponent_name)].append(bout)
    approved={(r.source,r.source_fighter_id):r for r in crosswalk.itertuples(index=False)}
    candidates=defaultdict(list)
    for bout in series:
        if bout.promotion not in ('Contender Series','Road to UFC'):continue
        matches=index.get(match_key(bout.event_date,bout.fighter_name,bout.opponent_name),[])
        if len(matches)!=1:continue
        witness=matches[0]
        same=identity_name(bout.fighter_name)==identity_name(witness.fighter_name)
        expected=witness.result if same else {'W':'L','L':'W','D':'D','NC':'NC'}[witness.result]
        if expected!=bout.result:continue
        for source_id,name in ((bout.fighter_source_id,bout.fighter_name),(bout.opponent_source_id,bout.opponent_name)):
            archive_id=witness.fighter_source_id if identity_name(name)==identity_name(witness.fighter_name) else witness.opponent_source_id
            link=approved.get((witness.source,archive_id))
            if link is None:continue
            if getattr(link,'match_method','')=='dated_ufc_pair_name_token_order':
                proofs=json.loads(link.notes)
                if not any(p['archive_bout_id']!=witness.source_bout_id for p in proofs):continue
            candidates[(bout.source,source_id)].append({'source_name':name,
                'canonical_fighter_id':link.canonical_fighter_id,'promotion':bout.promotion,
                'series_observation_id':bout.observation_id,'series_source_url':bout.source_url,
                'name_match_rule':'case/accent/punctuation/camel-case normalization and name-token order; exact dated opponent pair required',
                'archive_observation':witness.to_dict(),'ufc_crosswalk_evidence':link._asdict()})
    mappings=[];evidence=[];conflicts=[]
    for (source,source_id),witnesses in sorted(candidates.items()):
        ids={w['canonical_fighter_id'] for w in witnesses}
        previous=existing.get((source,source_id))
        if len(ids)!=1 or (previous and previous not in ids):
            conflicts.append({'source':source,'source_fighter_id':source_id,'candidate_ids':sorted(ids),'existing':previous})
            continue
        canonical=next(iter(ids))
        if previous:continue
        mappings.append({'source':source,'source_fighter_id':source_id,'source_fighter_name':witnesses[0]['source_name'],
            'canonical_fighter_id':canonical,'canonical_source':'ufcstats','status':'approved',
            'match_method':'dated_series_bout_plus_archive_ufc_witness','evidence_count':len(witnesses),
            'reviewed_at_utc':now.isoformat(),'notes':'Exact date/opponent pair and matching result; archive source ID independently anchored by a dated UFC bout. See series_identity_evidence.json.'})
        evidence.append({'source':source,'source_fighter_id':source_id,'canonical_fighter_id':canonical,'witnesses':witnesses})
    return pd.DataFrame(mappings,columns=IDENTITY_COLUMNS),evidence,conflicts


def coverage_audit(series, archive, crosswalk, links):
    index=defaultdict(list)
    for row in archive:index[match_key(row.event_date,row.fighter_name,row.opponent_name)].append(row)
    anchored={(r.source,r.source_fighter_id) for r in crosswalk.itertuples(index=False)}
    counts=defaultdict(Counter);missing=[]
    for row in series:
        if row.promotion not in ('Road to UFC','Contender Series'):continue
        matches=index[match_key(row.event_date,row.fighter_name,row.opponent_name)]
        for fid,name in ((row.fighter_source_id,row.fighter_name),(row.opponent_source_id,row.opponent_name)):
            if (row.source,fid) in links:reason='linked'
            elif not matches:reason='no_dated_opponent_pair_in_archive'
            elif len(matches)>1:reason='ambiguous_archive_bout'
            else:
                witness=matches[0]
                same=identity_name(name)==identity_name(witness.fighter_name)
                wid=witness.fighter_source_id if same else witness.opponent_source_id
                if wid.startswith('unresolved:'):reason='missing_or_ambiguous_directory_identity'
                elif (witness.source,wid) not in anchored:reason='no_verified_ufc_bout_for_archive_identity'
                else:reason='result_or_identity_conflict'
            counts[row.promotion][reason]+=1
            if reason!='linked':missing.append({'promotion':row.promotion,'date':row.event_date,
                'fighter':name,'opponent':row.opponent_name if fid==row.fighter_source_id else row.fighter_name,
                'source_fighter_id':fid,'source_url':row.source_url,'reason':reason})
    return {'participant_appearances':dict(counts),'unlinked':missing}


def main():
    root=DATA/'external_mma/event_results'
    archive,audit=load_identity_witnesses(Path(__file__).resolve().parents[1]/'artifacts/free_data_audit/mmastats/dataset_global_v3.duckdb')
    raw=pd.read_csv(DATA/'processed/ufc_fights_reported_doubled.csv',low_memory=False)
    crosswalk=pd.DataFrame(columns=IDENTITY_COLUMNS)
    crosswalk=extend_archive_crosswalk(archive,raw,crosswalk,datetime.now(timezone.utc))
    series=ExternalMmaStore(root).observations()
    mapping,evidence,conflicts=propose(series,archive,crosswalk,load_identity_map(root/'identity_map.csv'),datetime.now(timezone.utc))
    # A repeat may add evidence; it must never silently remove or change an accepted link.
    path=root/'series_identity_map.csv'
    if path.exists():
        previous=load_identity_map(path)
        current={(r.source,r.source_fighter_id):r.canonical_fighter_id for r in mapping.itertuples(index=False)}
        if any(current.get(k)!=v for k,v in previous.items()):raise ValueError('Previously accepted series link changed; review required')
    atomic_write_text(path,mapping.to_csv(index=False,lineterminator='\n'))
    report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'archive_audit':audit,
        'new_identity_links':len(mapping),'conflicts':conflicts,'evidence':evidence,
        'limitations':['Both archives are secondary sources, not independent primary verification.',
            'Ambiguous or conflicting date/name-pair witnesses are excluded. Name alone never links a career.']}
    atomic_write_text(root/'series_identity_evidence.json',json.dumps(report,sort_keys=True,indent=2)+'\n')
    links=load_identity_map(root/'identity_map.csv')|load_identity_map(path)
    gaps=coverage_audit(series,archive,crosswalk,links)
    atomic_write_text(root/'series_identity_gaps.json',json.dumps(gaps,sort_keys=True,indent=2)+'\n')
    print(json.dumps(gaps['participant_appearances']),flush=True)
    counts=Counter()
    new={(r.source,r.source_fighter_id) for r in mapping.itertuples(index=False)}
    for r in series:
        for fid in (r.fighter_source_id,r.opponent_source_id):
            if (r.source,fid) in new:counts[r.promotion]+=1
    print(json.dumps({'new_links':len(mapping),'new_linked_appearances':counts,'conflicts':len(conflicts)}),flush=True)


if __name__=='__main__':main()
