"""Rebind unchanged winner models after the documented schedule-only repair.

Refuses any other source or training change. Issues current forecasts anew;
historical forecast/price/decision ledgers and model parameters are untouched.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pandas as pd

from external_mma import load_approved_auxiliary
from fight_predictor.point_in_time import PointInTimeDatasetBuilder
from market_tracker._common import canonical_hash
from market_tracker._storage import atomic_write_text
from upcoming_bet_board import validate_upcoming_forecast_publication
from validate_data import validate_model_artifact, validate_bayesian_artifact

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'src/content/data'
AUDIT = ROOT / 'audit/profitability/improvements'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def model_id(artifact):
    body = {key: value for key, value in artifact.items() if key != 'model_id'}
    return sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:20]


def reverse_recorded_schedules(raw, changes):
    """Recover the exact prior source state, failing on unexpected repairs."""
    old = raw.copy(deep=True)
    ids = raw['fight_url'].astype(str).str.rstrip('/').str.rsplit('/').str[-1]
    seen = set()
    for change in changes:
        fight = change['fight_id']
        mask = ids.eq(fight)
        if fight in seen or change['changed_side_cells'] != 2 or int(mask.sum()) != 2:
            raise ValueError('schedule repair must identify exactly two unique fight sides')
        if not raw.loc[mask, 'time_format'].eq(change['time_format']).all():
            raise ValueError('current schedule differs from the recorded repair')
        old.loc[mask, 'time_format'] = ''
        seen.add(fight)
    return old


def refresh():
    raw_path = DATA / 'processed/ufc_fights_reported_doubled.csv'
    repair = read(AUDIT / 'schedule_repair.json')
    if sha256(raw_path.read_bytes()).hexdigest() != repair['proposed_raw_sha256']:
        raise ValueError('raw source has changed since the schedule repair; use the normal model rebuild')
    raw = pd.read_csv(raw_path, low_memory=False)
    fighters = pd.read_csv(DATA / 'processed/fighter_stats.csv', low_memory=False)
    pit = pd.read_csv(DATA / 'processed/ufc_fights_point_in_time.csv', low_memory=False)
    auxiliary = load_approved_auxiliary(DATA / 'processed/external_mma_auxiliary_doubled.csv',
                                       DATA / 'external_mma/model_policy.json')
    external = DATA / 'external'
    winner = read(external / 'winner_model.json')
    bayesian = read(external / 'bayesian_winner_challenger.json')
    check = validate_model_artifact(winner, raw, fighters, pit, auxiliary)
    if not check.errors:
        return {'status': 'already_consistent'}
    if check.errors != ['winner model state fingerprint differs from raw/profile source data']:
        raise ValueError(check.errors)
    previous = reverse_recorded_schedules(raw, repair['changes'])
    previous_check = validate_model_artifact(winner, previous, fighters, pit, auxiliary)
    bayesian_check = validate_bayesian_artifact(bayesian, winner, pit)
    if previous_check.errors or bayesian_check.errors:
        raise ValueError(previous_check.errors + bayesian_check.errors)
    builder = PointInTimeDatasetBuilder(raw, fighters, auxiliary_fights=auxiliary)
    fingerprint = builder._state_source_fingerprint(builder._validate_and_prepare_raw())
    new_winner = deepcopy(winner)
    new_winner['state_fingerprint_sha256'] = fingerprint
    new_winner['model_id'] = model_id(new_winner)
    new_bayesian = deepcopy(bayesian)
    new_bayesian['state_fingerprint_sha256'] = fingerprint
    new_bayesian['base_model_id'] = new_winner['model_id']
    new_bayesian['model_id'] = model_id(new_bayesian)
    errors = validate_model_artifact(new_winner, raw, fighters, pit, auxiliary).errors
    errors += validate_bayesian_artifact(new_bayesian, new_winner, pit).errors
    if errors:
        raise ValueError(errors)
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    revision = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
        'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    capture = read(DATA / 'market/capture_report.json')
    if now >= datetime.fromisoformat(capture['event_start_utc'].replace('Z', '+00:00')):
        raise ValueError('current card has started; use the normal upcoming-card rebuild')
    vegas = read(external / 'vegas_odds.json')
    upcoming = read(external / 'all_upcoming_forecasts.json')
    validate_upcoming_forecast_publication(upcoming)
    for column, old_id, new_id in (
        ('model id', winner['model_id'], new_winner['model_id']),
        ('bayesian model id', bayesian['model_id'], new_bayesian['model_id']),
    ):
        if set(vegas[column].values()) != {old_id}:
            raise ValueError('current forecast references a different model')
        vegas[column] = {key: new_id for key in vegas[column]}
    for column, value in [('forecast issued at', timestamp), ('forecast source commit', revision)]:
        vegas[column] = {key: value for key in vegas[column]}
    for row in upcoming['matchups']:
        if row['model_id'] != winner['model_id'] or row['event_date'] < now.date().isoformat():
            raise ValueError('upcoming forecast has a different model or a past event')
        row.update(model_id=new_winner['model_id'], forecast_issued_at_utc=timestamp,
                   forecast_source_commit=revision)
    upcoming['generated_at_utc'] = timestamp
    upcoming.pop('publication_sha256')
    upcoming['publication_sha256'] = canonical_hash(upcoming)
    validate_upcoming_forecast_publication(upcoming)
    # Every check above completes before the first write. No historic file is edited.
    for name, value in [('winner_model', new_winner), ('bayesian_winner_challenger', new_bayesian),
                        ('vegas_odds', vegas), ('all_upcoming_forecasts', upcoming)]:
        write(external / f'{name}.json', value)
    from refresh_betting_publications import refresh as refresh_views
    refresh_views()
    result = {'status': 'refreshed_without_training', 'issued_at_utc': timestamp,
        'previous_model_id': winner['model_id'], 'new_model_id': new_winner['model_id'],
        'previous_state_fingerprint': winner['state_fingerprint_sha256'],
        'new_state_fingerprint': fingerprint, 'verified_schedule_repairs': len(repair['changes']),
        'training_fingerprint_unchanged': winner['training_fingerprint_sha256'],
        'parameters_and_probabilities_unchanged': True, 'historical_ledgers_modified': False}
    write(AUDIT / 'model_provenance_refresh.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(refresh(), sort_keys=True))
