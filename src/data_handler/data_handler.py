import os 
import pandas as pd
from bs4 import BeautifulSoup
import urllib.request
import requests
import csv
import json
from datetime import date, datetime, timezone
import re
import numpy as np
from pathlib import Path
import tempfile
import time

# local imports
from fight_stat_helpers import (
                       same_name, 
                       same_name_vect,
                       get_kelly_bet_from_ev_and_dk_odds,
                       bet_payout,
                       clean_method_for_winner_predictions,
                       clean_method_for_method_predictions,
                       make_cumsum_before_current_fight,
                       make_avg_before_current_fight,
                       get_fighter_stats,
                       count_wins_wins_before_fight,
                       count_losses_losses_before_fight,
                       fight_math,
                       get_fight_card,
                       get_fight_stats,
                       get_event_fight_urls,
            )

# replace downcasting behavior deprecated
pd.set_option('future.no_silent_downcasting', True)

from odds_getter import OddsGetter
from ufcstats_client import UFCStatsError, UFCStatsEventNotComplete, ufcstats_client
from ufc_round_data import (
    RECONCILIATION_COLUMNS,
    ROUND_DATA_COLUMNS,
    RoundBackfillSummary,
    empty_reconciliation_frame,
    empty_round_stats_frame,
    normalize_round_stats,
    reconcile_round_stats,
    ufcstats_identity as round_ufcstats_identity,
    validate_normalized_round_stats,
)

git_root = str(Path(__file__).resolve().parents[2])

pd.options.mode.chained_assignment = None # default='warn' (disables SettingWithCopyWarning)


def atomic_write_text(path, text, encoding='utf-8'):
    """Write text beside its destination and atomically replace on success."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, 'w', encoding=encoding, newline='') as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_to_csv(dataframe, path, *, index=False):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent
    )
    os.close(descriptor)
    try:
        dataframe.to_csv(temporary_name, index=index)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def ufcstats_identity(value):
    """Return the stable identifier at the end of a UFCStats URL."""
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ''
    return str(value).strip().rstrip('/').rsplit('/', 1)[-1].lower()


def validate_scraped_event_integrity(
    event_url, scraped_stats, manifest_fight_urls, stored_fight_urls=()
):
    """Reject partial event parses and quarantine unexpected source deletions."""
    parsed_fights = set(scraped_stats['fight_url'].dropna().astype(str))
    advertised_fights = set(str(value) for value in manifest_fight_urls)
    stored_fights = set(str(value) for value in stored_fight_urls)
    if parsed_fights != advertised_fights:
        missing = sorted(advertised_fights - parsed_fights)
        unexpected = sorted(parsed_fights - advertised_fights)
        raise UFCStatsError(
            f'Parsed fight IDs did not match the event manifest for {event_url}; '
            f'missing={missing[:3]}, unexpected={unexpected[:3]}'
        )
    removed = stored_fights - parsed_fights
    if removed:
        raise UFCStatsError(
            f'Refusing to shrink stored event {event_url}; source omitted '
            f'{len(removed)} existing fight(s): {sorted(removed)[:3]}'
        )

class DataHandler:
    def __init__(self):
        # updated scraped fight data (after running ufc_fights_reported_doubled_updated function from UFC_data_scraping file)
        self.csv_filepaths = {
            'fighter_stats': f'{git_root}/src/content/data/processed/fighter_stats.csv',
            'ufc_fights_reported_derived_doubled': f'{git_root}/src/content/data/processed/ufc_fights_reported_derived_doubled.csv',
            'ufc_fights_reported_doubled': f'{git_root}/src/content/data/processed/ufc_fights_reported_doubled.csv',
            'ufc_fight_round_stats_doubled': f'{git_root}/src/content/data/processed/ufc_fight_round_stats_doubled.csv',
            'ufc_fight_data_for_website': f'{git_root}/src/content/data/processed/ufc_fight_data_for_website.csv', # not really needed...
        }
        self.round_reconciliation_filepath = (
            f'{git_root}/src/content/data/processed/'
            'ufc_fight_round_stats_reconciliation.csv'
        )

        self.json_filepaths = {
            'card_info': f'{git_root}/src/content/data/external/card_info.json',
            'fighter_stats': f'{git_root}/src/content/data/external/fighter_stats.json',
            'interesting_stats': f'{git_root}/src/content/data/external/interesting_stats.json',
            'prediction_history': f'{git_root}/src/content/data/external/prediction_history.json',
            'theta': f'{git_root}/src/content/data/external/theta.json',
            'intercept': f'{git_root}/src/content/data/external/intercept.json',
            'ufc_fight_data_for_website': f'{git_root}/src/content/data/external/ufc_fight_data_for_website.json',
            'vegas_odds': f'{git_root}/src/content/data/external/vegas_odds.json',
        }
        # The 68 MB legacy derived table is no longer part of the weekly model
        # or website publication path.  Load it lazily only for old notebooks
        # and manual tools that explicitly request it.
        self.csv_data = {
            key: pd.read_csv(path, sep=',')
            for key, path in self.csv_filepaths.items()
            if key != 'ufc_fights_reported_derived_doubled'
            and Path(path).exists()
        }
        prediction_history = pd.read_json(self.json_filepaths['prediction_history'])
        vegas_odds = pd.read_json(self.json_filepaths['vegas_odds'])

        self.json_data = {
            'prediction_history': prediction_history,
            'vegas_odds': vegas_odds,
        }
        
        self.odds_getter = OddsGetter()
        self.update_time = 0
        
        self.bookies = ['DraftKings', 'BetMGM', 'Caesars', 'BetRivers', 'FanDuel', 'PointsBet', 'Unibet', 'Bet365', 'BetWay', '5D', 'Ref','BetOnline','MyBookie']

    def get(self, key, filetype='csv'):
        if filetype == 'json':
            assert key in list(self.json_data.keys()), "Invalid key provided"
            return self.json_data[key].copy()
        assert key in self.csv_filepaths, "Invalid key provided"
        if key not in self.csv_data:
            path = Path(self.csv_filepaths[key])
            if key == 'ufc_fight_round_stats_doubled' and not path.exists():
                self.csv_data[key] = empty_round_stats_frame()
            else:
                self.csv_data[key] = pd.read_csv(path, sep=',')
        df = self.csv_data[key].copy()
        return df
    
    def set(self, key, value):
        assert key in self.csv_filepaths, "Invalid key provided"
        self.csv_data[key] = value
    
    def save_csv(self, key):
        assert key in list(self.csv_filepaths.keys()), "Invalid key provided"
        atomic_to_csv(self.csv_data[key], self.csv_filepaths[key], index=False)

    def _round_reconciliation_path(self):
        configured = getattr(self, 'round_reconciliation_filepath', None)
        if configured:
            return Path(configured)
        round_path = Path(self.csv_filepaths['ufc_fight_round_stats_doubled'])
        return round_path.with_name('ufc_fight_round_stats_reconciliation.csv')

    def _read_round_reconciliation(self):
        path = self._round_reconciliation_path()
        if not path.exists():
            return empty_reconciliation_frame()
        report = pd.read_csv(path, low_memory=False)
        missing = set(RECONCILIATION_COLUMNS) - set(report.columns)
        if missing:
            raise ValueError(
                f'round reconciliation report is missing columns: {sorted(missing)}'
            )
        return report.loc[:, RECONCILIATION_COLUMNS].copy()

    def _persist_round_updates(self, new_rows, new_issues, replace_fight_ids):
        """Atomically replace only successfully parsed physical fights.

        ``replace_fight_ids`` can describe every bout a caller attempted, but
        a transient detail-page/parser failure may leave no replacement rows
        for one of those bouts.  Derive the destructive replacement set from
        structurally valid parsed rows so an attempted refresh can never erase
        the last known-good copy.
        """
        validate_normalized_round_stats(new_rows)
        requested_ids = {
            str(value).strip()
            for value in replace_fight_ids
            if str(value).strip()
        }
        parsed_ids = set(new_rows['fight_id'].dropna().astype(str))
        replace_ids = requested_ids & parsed_ids
        if not replace_ids:
            return

        new_rows = new_rows[
            new_rows['fight_id'].astype(str).isin(replace_ids)
        ].copy()
        if new_issues.empty:
            new_issues = empty_reconciliation_frame()
        else:
            new_issues = new_issues[
                new_issues['fight_id'].astype(str).isin(replace_ids)
            ].copy()
        existing = self.get('ufc_fight_round_stats_doubled')
        if not existing.empty:
            validate_normalized_round_stats(existing)
        kept = existing[~existing['fight_id'].astype(str).isin(replace_ids)]
        combined = pd.concat([new_rows, kept], ignore_index=True)
        if combined.empty:
            combined = empty_round_stats_frame()
        else:
            combined = combined.loc[:, ROUND_DATA_COLUMNS]
            combined = combined.sort_values(
                ['date', 'bout_order', 'fight_id', 'round', 'fighter_id'],
                ascending=[False, True, True, True, True],
                kind='stable',
            ).reset_index(drop=True)
        validate_normalized_round_stats(combined)

        existing_issues = self._read_round_reconciliation()
        kept_issues = existing_issues[
            ~existing_issues['fight_id'].astype(str).isin(replace_ids)
        ]
        combined_issues = pd.concat([new_issues, kept_issues], ignore_index=True)
        if combined_issues.empty:
            combined_issues = empty_reconciliation_frame()
        else:
            combined_issues = combined_issues.loc[:, RECONCILIATION_COLUMNS]
            combined_issues = combined_issues.sort_values(
                ['fight_id', 'fighter_id', 'field', 'issue', 'detail'],
                kind='stable',
            ).reset_index(drop=True)

        self.set('ufc_fight_round_stats_doubled', combined)
        self.save_csv('ufc_fight_round_stats_doubled')
        atomic_to_csv(
            combined_issues, self._round_reconciliation_path(), index=False
        )
            
    def save_json(self, key, column):
        assert key in list(self.json_filepaths.keys()), "Invalid key provided"
        print(f'sending updated {key}.csv to {key}.json')
        self.make_json(self.csv_filepaths[key], self.json_filepaths[key], column)
        
    def set_regression_coeffs_and_intercept(self, theta, b):
        # NOTE we don't even do this anymore. We just have an old theta and b in the json files that we use for the website
        # TODO this is a bit clunky, should be able to just set the theta and b directly using the set method
        # these need to be dictionaries to use json.dump
        self.theta_dict = {i:theta[i] for i in range(len(theta))}
        self.intercept_dict = {0:b}
        
        atomic_write_text(self.json_filepaths['theta'], json.dumps(theta))
        atomic_write_text(self.json_filepaths['intercept'], json.dumps(b))
        
    def update_data_csvs_and_jsons(self, key='all'):
        assert key in list(self.csv_filepaths.keys()) + ['all'], "Invalid key provided"
        if key == 'ufc_fights_reported_doubled':
            self.update_ufc_fights_reported_doubled()
        elif key == 'fighter_stats':
            self.update_fighter_stats()
        elif key == 'ufc_fights_reported_derived_doubled':
            self.update_ufc_fights_reported_derived_doubled()
        elif key == 'ufc_fight_round_stats_doubled':
            self.backfill_ufc_fight_round_stats_doubled()
        elif key == 'prediction_history':
            self.update_prediction_history()
        elif key == 'all':
            self.update_ufc_fights_reported_doubled()
            self.update_fighter_stats()
            self.update_ufc_fight_data_for_website()
            # Image search is optional presentation work and depends on a
            # third-party Google HTML layout.  It must not make the core weekly
            # data/model transaction slower or less reliable.
        else:
            raise ValueError("No update function implemented for this key")

        
    def get_most_recent_fight_date(self, key):
        # find the most recent fight date in the specified key's dataframe
        assert key in ['ufc_fights_reported_doubled', 'ufc_fights_reported_derived_doubled'], "Invalid key provided"
        dates = self.csv_data[key]['date']
        # convert dates to datetime objects if they are not already
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates, errors='coerce')
        # find the most recent date 
        most_recent_date = dates.max()
        return most_recent_date # TODO make sure this is ordered with earliest date first (might not be)
                
    def update_ufc_fights_reported_doubled(self):  # takes dataframe of fight stats as input
        old_ufc_fights_reported_doubled = self.get('ufc_fights_reported_doubled')
        raw_schema_migrated = False
        if (
            'source_card_index' not in old_ufc_fights_reported_doubled
            or 'bout_order' not in old_ufc_fights_reported_doubled
            or 'time_format' not in old_ufc_fights_reported_doubled
            or old_ufc_fights_reported_doubled[
                [
                    column for column in ('source_card_index', 'bout_order')
                    if column in old_ufc_fights_reported_doubled
                ]
            ].isna().any().any()
        ):
            # The stored row order within each event is the UFCStats display
            # order (main event first).  Materialize it once as durable source
            # metadata; future scrapes supply these columns directly.
            positions = np.arange(len(old_ufc_fights_reported_doubled))
            temporary = old_ufc_fights_reported_doubled[
                ['event_url', 'fight_url']
            ].copy()
            temporary['_position'] = positions
            first_positions = (
                temporary.groupby(['event_url', 'fight_url'], sort=False)['_position']
                .min()
                .rename('first_position')
                .reset_index()
            )
            first_positions['source_card_index'] = first_positions.groupby(
                'event_url', sort=False
            )['first_position'].rank(method='dense').astype(int) - 1
            event_sizes = first_positions.groupby('event_url')['fight_url'].transform('size')
            first_positions['bout_order'] = (
                event_sizes - 1 - first_positions['source_card_index']
            )
            order_lookup = first_positions.set_index(
                ['event_url', 'fight_url']
            )[['source_card_index', 'bout_order']]
            keys = list(old_ufc_fights_reported_doubled[
                ['event_url', 'fight_url']
            ].itertuples(index=False, name=None))
            old_ufc_fights_reported_doubled['source_card_index'] = [
                int(order_lookup.loc[key, 'source_card_index']) for key in keys
            ]
            old_ufc_fights_reported_doubled['bout_order'] = [
                int(order_lookup.loc[key, 'bout_order']) for key in keys
            ]
            if 'time_format' not in old_ufc_fights_reported_doubled:
                old_ufc_fights_reported_doubled['time_format'] = ''
                legacy_dates = pd.to_datetime(
                    old_ufc_fights_reported_doubled['date'], errors='coerce'
                )
                legacy_rounds = pd.to_numeric(
                    old_ufc_fights_reported_doubled['round'], errors='coerce'
                )
                uncertain_legacy_duration = (
                    (legacy_dates < pd.Timestamp('2001-01-01'))
                    & (legacy_rounds > 1)
                )
                old_ufc_fights_reported_doubled.loc[
                    uncertain_legacy_duration, 'total_fight_time'
                ] = np.nan
                print(
                    'Marked '
                    f'{int(uncertain_legacy_duration.sum())} legacy fight sides '
                    'with unknown nonstandard round duration'
                )
            raw_schema_migrated = True
        # Older parser versions collapsed no-contests into draws.  UFCStats'
        # explicit CNC/overturned methods let us repair those labels safely.
        mislabeled_no_contests = old_ufc_fights_reported_doubled['method'].isin(
            ['CNC', 'Overturned']
        ) & (old_ufc_fights_reported_doubled['result'] == 'D')
        old_ufc_fights_reported_doubled.loc[mislabeled_no_contests, 'result'] = 'NC'
        normalized_existing_results = int(mislabeled_no_contests.sum())
        url = 'http://ufcstats.com/statistics/events/completed?page=all'
        page = ufcstats_client.get(url, expected_text='b-statistics__table-events')
        soup = BeautifulSoup(page.content, "html.parser")
        events_table = soup.select_one('tbody')
        if events_table is None:
            raise UFCStatsError(
                f'Completed-events table was not found at {url}; refusing to update data'
            )
        ufc_fights_reported_doubled_new_rows = pd.DataFrame()
            
        events = list(events_table.select('a')[1:]) #omit first event, future event
        saved_event_hrefs = set(old_ufc_fights_reported_doubled.event_url.unique())
        new_events = [
            event for event in events
            if event['href'] not in saved_event_hrefs
            and "road to ufc" not in event.text.strip().lower()
        ]

        # Re-scrape a bounded recent window, not just changed URL manifests.
        # UFCStats can correct a result, method, identity, time, or detailed
        # statistics without changing any fight URL.
        recent_saved_events = [
            event for event in events
            if event['href'] in saved_event_hrefs
            and "road to ufc" not in event.text.strip().lower()
        ][:12]
        events_to_refresh = recent_saved_events
        if events_to_refresh:
            print(
                f'Reconciling source contents for {len(events_to_refresh)} recent events'
            )

        events_to_scrape = new_events + events_to_refresh
        if not events_to_scrape:
            if normalized_existing_results or raw_schema_migrated:
                old_ufc_fights_reported_doubled['date'] = pd.to_datetime(
                    old_ufc_fights_reported_doubled['date'], errors='raise'
                )
                old_ufc_fights_reported_doubled = (
                    old_ufc_fights_reported_doubled.sort_values(
                        'date', ascending=False, kind='stable'
                    ).reset_index(drop=True)
                )
                old_ufc_fights_reported_doubled['date'] = (
                    old_ufc_fights_reported_doubled['date'].dt.strftime('%Y-%m-%d')
                )
                self.set(
                    'ufc_fights_reported_doubled', old_ufc_fights_reported_doubled
                )
                self.save_csv('ufc_fights_reported_doubled')
                print(
                    f'Normalized {normalized_existing_results} historical '
                    'no-contest fight sides'
                )
                if raw_schema_migrated:
                    print('Added durable source card order to historical fights')
            print('No new events to scrape for ufc_fights_reported_doubled')
            return
        
        refreshed_event_hrefs = set()
        new_event_hrefs = {event['href'] for event in new_events}
        completed_new_event_hrefs = set()
        for event in events_to_scrape:
            name = event.text.strip()
            href = event['href']
            if "road to ufc" in name.lower():
                continue  # skip Road to UFC events
            try:
                manifest_fight_urls = get_event_fight_urls(href)
                # Per-round tables are research enrichment and are fetched by
                # the bounded ``fight_sim backfill`` command.  The production
                # updater collects authoritative bout totals only.
                stats = get_fight_card(href)
                stored_fight_urls = old_ufc_fights_reported_doubled.loc[
                    old_ufc_fights_reported_doubled['event_url'].eq(href),
                    'fight_url',
                ].unique()
                validate_scraped_event_integrity(
                    href,
                    stats,
                    manifest_fight_urls,
                    stored_fight_urls,
                )
            except UFCStatsEventNotComplete as error:
                print(f'Deferring incomplete same-day event {name}: {error}')
                continue
            except UFCStatsError as error:
                if href in new_event_hrefs:
                    print(f'Deferring new event {name} after a source error: {error}')
                else:
                    print(
                        f'Keeping stored event {name} after refresh error: {error}'
                    )
                continue
            refreshed_event_hrefs.add(href)
            if href in new_event_hrefs:
                completed_new_event_hrefs.add(href)
            ufc_fights_reported_doubled_new_rows = pd.concat([stats, ufc_fights_reported_doubled_new_rows], axis=0)
            
        # convert date column to string format YYYY-MM-DD
        if ufc_fights_reported_doubled_new_rows.empty:
            if normalized_existing_results or raw_schema_migrated:
                old_ufc_fights_reported_doubled['date'] = pd.to_datetime(
                    old_ufc_fights_reported_doubled['date'], errors='raise'
                )
                old_ufc_fights_reported_doubled = (
                    old_ufc_fights_reported_doubled.sort_values(
                        'date', ascending=False, kind='stable'
                    ).reset_index(drop=True)
                )
                old_ufc_fights_reported_doubled['date'] = (
                    old_ufc_fights_reported_doubled['date'].dt.strftime('%Y-%m-%d')
                )
                self.set(
                    'ufc_fights_reported_doubled',
                    old_ufc_fights_reported_doubled,
                )
                self.save_csv('ufc_fights_reported_doubled')
                print('Saved raw schema/result normalization before deferring events')
            print(
                'No fully completed new event rows were available; deferred '
                'events will be retried on the next run'
            )
            return
        
        ufc_fights_reported_doubled_new_rows['date'] = pd.to_datetime(ufc_fights_reported_doubled_new_rows['date'], errors='coerce')
        ufc_fights_reported_doubled_new_rows['date'] = ufc_fights_reported_doubled_new_rows['date'].dt.strftime('%Y-%m-%d')

        old_rows_to_keep = old_ufc_fights_reported_doubled[
            ~old_ufc_fights_reported_doubled['event_url'].isin(refreshed_event_hrefs)
        ]
        updated_stats = pd.concat(
            [ufc_fights_reported_doubled_new_rows, old_rows_to_keep], axis=0
        )
        # Source URLs are the durable identities.  A retry or a corrected
        # event must replace a side rather than silently duplicating it.
        updated_stats = updated_stats.drop_duplicates(
            subset=['fight_url', 'fighter_url'], keep='first'
        )
        updated_stats['date'] = pd.to_datetime(updated_stats['date'], errors='raise')
        updated_stats = updated_stats.sort_values('date', ascending=False, kind='stable')
        updated_stats['date'] = updated_stats['date'].dt.strftime('%Y-%m-%d')
        updated_stats = updated_stats.reset_index(drop=True)
        fight_side_counts = updated_stats.groupby('fight_url', dropna=False).size()
        if not (fight_side_counts == 2).all():
            invalid = fight_side_counts[fight_side_counts != 2].head().to_dict()
            raise ValueError(
                f'Refusing to save incomplete doubled fights; invalid fight_url counts: {invalid}'
            )
        # set ufc_fights_reported_doubled and save it to csv
        self.update_time = int(
            ufc_fights_reported_doubled_new_rows.loc[
                ufc_fights_reported_doubled_new_rows['event_url'].isin(
                    completed_new_event_hrefs
                ),
                'fight_url',
            ].nunique()
        )
        self.set('ufc_fights_reported_doubled', updated_stats)
        self.save_csv('ufc_fights_reported_doubled')

    def backfill_ufc_fight_round_stats_doubled(
        self,
        max_fights=100,
        *,
        checkpoint_every=10,
        refresh_existing=False,
        max_runtime_seconds=3000.0,
    ):
        """Fetch a bounded, resumable set of historical fight-detail pages.

        A fight is considered complete only when structurally valid doubled
        round rows are already stored.  Failed pages are reported and remain
        eligible for the next run.  Successful checkpoints replace whole
        physical fights, so an interruption cannot leave half a bout in the
        persisted dataset.
        """
        if not isinstance(max_fights, int) or isinstance(max_fights, bool) or max_fights < 0:
            raise ValueError('max_fights must be a nonnegative integer')
        if (
            not isinstance(checkpoint_every, int)
            or isinstance(checkpoint_every, bool)
            or checkpoint_every < 1
        ):
            raise ValueError('checkpoint_every must be a positive integer')
        if (
            isinstance(max_runtime_seconds, bool)
            or not isinstance(max_runtime_seconds, (int, float))
            or not 0 < float(max_runtime_seconds) <= 3300
        ):
            raise ValueError('max_runtime_seconds must be in (0, 3300]')
        started_at = time.monotonic()
        deadline = started_at + float(max_runtime_seconds)

        raw = self.get('ufc_fights_reported_doubled')
        required_raw = {'fight_url', 'fighter_url', 'fighter', 'opponent'}
        missing_raw = required_raw - set(raw.columns)
        if missing_raw:
            raise ValueError(
                f'raw fights are missing round-backfill columns: {sorted(missing_raw)}'
            )
        physical_counts = raw.groupby('fight_url', dropna=False).size()
        invalid = physical_counts[physical_counts != 2]
        if not invalid.empty:
            raise ValueError(
                'round backfill requires exactly two raw sides per fight; '
                f'invalid={invalid.head().to_dict()}'
            )

        existing = self.get('ufc_fight_round_stats_doubled')
        if not existing.empty:
            validate_normalized_round_stats(existing)
        completed_ids = set(existing['fight_id'].dropna().astype(str))
        physical = (
            raw.assign(
                _fight_id=raw['fight_url'].map(round_ufcstats_identity),
                _date=pd.to_datetime(raw.get('date'), errors='coerce'),
            )
            .sort_values(
                ['_date', 'fight_url'], ascending=[False, True], kind='stable'
            )
            .drop_duplicates('fight_url', keep='first')
        )
        if not refresh_existing:
            physical = physical[~physical['_fight_id'].isin(completed_ids)]
        candidates = physical.head(max_fights)

        attempted = 0
        saved = 0
        failed = 0
        saved_rows = 0
        issue_count = 0
        pending_rows = []
        pending_issues = []
        pending_ids = set()
        stopped_by_time_limit = False

        def checkpoint():
            if not pending_ids:
                return
            rows = pd.concat(pending_rows, ignore_index=True)
            issues = (
                pd.concat(pending_issues, ignore_index=True)
                if pending_issues
                else empty_reconciliation_frame()
            )
            self._persist_round_updates(rows, issues, pending_ids)
            pending_rows.clear()
            pending_issues.clear()
            pending_ids.clear()

        for candidate in candidates.to_dict('records'):
            if time.monotonic() >= deadline:
                stopped_by_time_limit = True
                break
            attempted += 1
            fight_url = str(candidate['fight_url'])
            fight_id = round_ufcstats_identity(fight_url)
            bout_sides = raw[raw['fight_url'].astype(str).eq(fight_url)].copy()
            try:
                aggregate, parsed_rounds = get_fight_stats(
                    fight_url,
                    str(candidate['fighter']),
                    str(candidate['opponent']),
                    include_round_stats=True,
                )
                time_formats = aggregate['time_format'].dropna().astype(str).str.strip()
                time_formats = time_formats[time_formats.ne('')].unique()
                if len(time_formats) != 1:
                    raise UFCStatsError(
                        f'fight detail page did not provide one time format for {fight_url}'
                    )
                bout_sides['time_format'] = time_formats[0]
                normalized = normalize_round_stats(parsed_rounds, bout_sides)

                # Reconcile against aggregate values from the same response,
                # while retaining the raw rows for stable event/side identity.
                source_totals = bout_sides.copy()
                for source in aggregate.to_dict('records'):
                    source_name = str(source.get('fighter') or '').strip()
                    parsed_identity = parsed_rounds[
                        parsed_rounds['fighter'].astype(str).eq(source_name)
                    ]['fighter_id'].dropna().astype(str).unique()
                    if len(parsed_identity) == 1:
                        matching = source_totals['fighter_url'].map(
                            round_ufcstats_identity
                        ).eq(parsed_identity[0])
                    else:
                        matching = source_totals['fighter'].astype(str).eq(source_name)
                    if matching.sum() != 1:
                        raise UFCStatsError(
                            f'aggregate fighter {source_name!r} did not map uniquely '
                            f'to stored sides for {fight_url}'
                        )
                    for field in ROUND_DATA_COLUMNS:
                        if field in source:
                            source_totals.loc[matching, field] = source[field]
                normalized, issues = reconcile_round_stats(
                    normalized, source_totals
                )
            except (UFCStatsError, ValueError, IndexError, TypeError) as error:
                failed += 1
                print(f'Round backfill failed for {fight_url}: {error}')
                continue

            pending_rows.append(normalized)
            if not issues.empty:
                pending_issues.append(issues)
            pending_ids.add(fight_id)
            saved += 1
            saved_rows += len(normalized)
            issue_count += len(issues)
            if saved % checkpoint_every == 0:
                checkpoint()

        checkpoint()
        now_existing = self.get('ufc_fight_round_stats_doubled')
        now_complete_ids = set(now_existing['fight_id'].dropna().astype(str))
        all_ids = {
            round_ufcstats_identity(value)
            for value in raw['fight_url'].dropna().astype(str).unique()
        }
        remaining = len(all_ids - now_complete_ids)
        summary = RoundBackfillSummary(
            attempted_fights=attempted,
            saved_fights=saved,
            failed_fights=failed,
            remaining_fights=remaining,
            saved_round_rows=saved_rows,
            reconciliation_issues=issue_count,
            elapsed_seconds=round(time.monotonic() - started_at, 3),
            stopped_by_time_limit=stopped_by_time_limit,
        )
        print(
            'Round backfill: '
            f'attempted={attempted}, saved={saved}, failed={failed}, '
            f'remaining={remaining}, issues={issue_count}'
        )
        return summary
    
        
    # updates fighter attributes with new fighters not yet saved yet
    def update_fighter_stats(self):
        ufc_fights_reported_doubled = self.get('ufc_fights_reported_doubled')
        fighter_stats = self.get('fighter_stats')
        known_fighter_urls = set(fighter_stats['url'].dropna().astype(str))
        all_source_urls = set(
            ufc_fights_reported_doubled['fighter_url'].dropna().astype(str)
        )
        raw_dates = pd.to_datetime(
            ufc_fights_reported_doubled['date'], errors='coerce'
        )
        recent_event_urls = (
            ufc_fights_reported_doubled.assign(_date=raw_dates)
            .groupby('event_url', as_index=False)['_date'].max()
            .sort_values('_date', ascending=False, kind='stable')
            .head(3)['event_url']
        )
        active_urls = set(
            ufc_fights_reported_doubled.loc[
                ufc_fights_reported_doubled['event_url'].isin(recent_event_urls),
                'fighter_url',
            ].dropna().astype(str)
        )
        refresh_urls = sorted((all_source_urls - known_fighter_urls) | active_urls)
        refreshed_profiles = []
        for f_url in refresh_urls:
            action = 'refreshing active fighter' if f_url in known_fighter_urls else 'adding new fighter'
            print(f'{action}: {f_url}')
            try:
                page = ufcstats_client.get(
                    f_url, expected_text='b-list__info-box'
                )
                soup = BeautifulSoup(page.content, "html.parser")
                name_element = soup.find(
                    'span', class_='b-content__title-highlight'
                )
                info_box = soup.select_one(
                    'div.b-list__info-box.b-list__info-box_style_small-width.js-guide'
                )
                if name_element is None or info_box is None:
                    raise UFCStatsError(
                        f'fighter profile fields were missing at {f_url}'
                    )
                attributes = {}
                for item in info_box.select('li'):
                    label = item.find('i')
                    if label is None:
                        continue
                    key = label.get_text(' ', strip=True).rstrip(':').casefold()
                    full_text = item.get_text(' ', strip=True)
                    value = full_text[len(label.get_text(' ', strip=True)):].strip()
                    attributes[key] = value
                refreshed_profiles.append(
                    {
                        'name': name_element.get_text(' ', strip=True),
                        'height': attributes.get('height', '--'),
                        'reach': attributes.get('reach', '--'),
                        'stance': attributes.get('stance', '--'),
                        'dob': attributes.get('dob', '--'),
                        'url': f_url,
                    }
                )
            except UFCStatsError as error:
                print(f'Keeping existing/unknown fighter profile after source error: {error}')

        refreshed_fighters = pd.DataFrame(
            refreshed_profiles,
            columns=['name', 'height', 'reach', 'stance', 'dob', 'url'],
        )
        refreshed_urls = set(refreshed_fighters['url'])
        updated_fighters = pd.concat(
            [
                refreshed_fighters,
                fighter_stats[~fighter_stats['url'].isin(refreshed_urls)],
            ],
            ignore_index=True,
        )
        updated_fighters = updated_fighters.drop_duplicates('url', keep='first')
        self.set('fighter_stats', updated_fighters)
        self.save_csv('fighter_stats')
        self.save_json('fighter_stats', 'name')
                        
        
    def clean_ufc_fights_for_winner_prediction(self, ufc_fights_predictive_flattened_diffs, prediction_type='winner'):
        #importing csv fight data and saving as dataframes
        ufc_fights_winner = ufc_fights_predictive_flattened_diffs.copy()
        #cleaning the methods column for winner prediction
        #changing anything other than 'U-DEC','M-DEC', 'KO/TKO', 'SUB', to 'bullshit'
        #changing 'U-DEC','M-DEC', to 'DEC'
        if prediction_type == 'winner':
            ufc_fights_winner['method'] = clean_method_for_winner_predictions(ufc_fights_winner['method'])
        elif prediction_type == 'method':
            ufc_fights_winner['method'] = clean_method_for_method_predictions(ufc_fights_winner['method'])
        #getting rid of rows with incomplete or useless data
        # Winner prediction must use every terminal W/L outcome.  Filtering on
        # a post-fight method (especially split decisions) trains on easier
        # fights than the model will see at deployment time.
        terminal_result_mask = ufc_fights_winner['result'].isin(['W', 'L'])
        ufc_fights_winner = ufc_fights_winner[terminal_result_mask]
        # Do not discard a fight because an unused feature is missing.  The
        # selected model pipeline handles missing predictive values explicitly.
        required_metadata = ['fighter', 'opponent', 'date', 'result']
        available_required = [
            column for column in required_metadata if column in ufc_fights_winner
        ]
        ufc_fights_winner = ufc_fights_winner.dropna(subset=available_required)
        ufc_fights_winner['result'] = (ufc_fights_winner['result'] == 'W').values.astype(int)
        
        return ufc_fights_winner
    
    
    def update_ufc_fights_reported_derived_doubled(self):
        raw = self.get('ufc_fights_reported_doubled')
        derived = self.get('ufc_fights_reported_derived_doubled')
        metadata_columns = ['date', 'fighter', 'opponent', 'result', 'method', 'division']

        # Reconcile actual rows instead of selecting date > max(date).  The old
        # rule permanently missed late-added fights, corrections, and a second
        # event on the same day.  An occurrence suffix preserves legitimate
        # repeated matchups with otherwise identical metadata.
        def add_row_identity(frame):
            identified = frame.copy()
            identified['date'] = pd.to_datetime(identified['date'], errors='raise').dt.strftime('%Y-%m-%d')
            identified['_occurrence'] = identified.groupby(
                metadata_columns, dropna=False, sort=False
            ).cumcount()
            identified['_row_identity'] = list(map(
                tuple,
                identified[metadata_columns + ['_occurrence']].itertuples(index=False, name=None),
            ))
            return identified

        raw_identified = add_row_identity(raw)
        derived_identified = add_row_identity(derived)
        raw_identities = set(raw_identified['_row_identity'])
        derived_identities = set(derived_identified['_row_identity'])
        missing_identities = raw_identities - derived_identities
        stale_identities = derived_identities - raw_identities

        new_rows = raw_identified[raw_identified['_row_identity'].isin(missing_identities)].drop(
            columns=['_occurrence', '_row_identity']
        )
        if stale_identities:
            derived = derived_identified[
                ~derived_identified['_row_identity'].isin(stale_identities)
            ].drop(columns=['_occurrence', '_row_identity'])
            self.set('ufc_fights_reported_derived_doubled', derived)

        self.update_time = max(len(new_rows), len(stale_identities)) // 2
        print(
            f'derived reconciliation: {len(new_rows)} new/corrected sides, '
            f'{len(stale_identities)} stale sides'
        )
        if missing_identities or stale_identities:
            if not new_rows.empty:
                ufc_fights_reported_derived_doubled = self.populate_new_fights_with_statistics(new_rows)
            else:
                ufc_fights_reported_derived_doubled = self.get('ufc_fights_reported_derived_doubled')
                ufc_fights_reported_derived_doubled['date'] = pd.to_datetime(
                    ufc_fights_reported_derived_doubled['date'], errors='raise'
                )
                ufc_fights_reported_derived_doubled.set_index('date', inplace=True)
            ufc_fights_reported_derived_doubled = ufc_fights_reported_derived_doubled.sort_index(kind='stable')
            # save the results to a csv file 
            ufc_fights_reported_derived_doubled_path = f'{git_root}/src/content/data/processed/ufc_fights_reported_derived_doubled.csv'
            atomic_to_csv(
                ufc_fights_reported_derived_doubled,
                ufc_fights_reported_derived_doubled_path,
                index=True,
            )
            ufc_fights_reported_derived_doubled.reset_index(inplace=True, drop=False)
            # set the new dataframe in the data manager
            self.set('ufc_fights_reported_derived_doubled', ufc_fights_reported_derived_doubled)
            print(f'Saved ufc_fights_reported_derived_doubled to {ufc_fights_reported_derived_doubled_path}, shape {ufc_fights_reported_derived_doubled.shape}')
        else:
            print('nothing to update')
            
    def make_ufc_fights_predictive_flattened(self, derived_doubled_df, shuffle=True, random_state=48):

        ufc_fights_reported_derived_doubled = derived_doubled_df.copy()

        non_predictive_columns = [
            'date',
            'fighter',
            'opponent',
            # 'method', # TODO predict method too
            # 'division', # TODO filter by division
            'stance', # TODO incorporate stance
        ]

        predictive_columns = [col for col in ufc_fights_reported_derived_doubled.columns if col not in non_predictive_columns]

        # shuffle pairs to avoid bias
        assert len(ufc_fights_reported_derived_doubled) % 2 == 0, "DataFrame length must be even to create pairs"
        shuffled_rows = []
        rng = np.random.default_rng(random_state)
        for i in range(0, len(ufc_fights_reported_derived_doubled), 2):
            pair = ufc_fights_reported_derived_doubled.iloc[i:i+2]
            if shuffle and rng.integers(0, 2):
                pair = pair.iloc[::-1]  # deterministic, outcome-independent orientation
            shuffled_rows.append(pair)
        # Concatenate back into a single DataFrame
        ufc_fights_reported_derived_doubled = pd.concat(shuffled_rows).reset_index(drop=True)

        # drop non-predictive columns
        ufc_fights_predictive = ufc_fights_reported_derived_doubled[predictive_columns]

        # grab fighter and opponent columns for diffing in flattened dataframe
        fighter_col = ufc_fights_reported_derived_doubled['fighter'].loc[::2]
        opponent_col = ufc_fights_reported_derived_doubled['opponent'].loc[::2]
        date_col = ufc_fights_reported_derived_doubled['date'].loc[::2]
        result_col = ufc_fights_reported_derived_doubled['result'].loc[::2]
        method_col = ufc_fights_reported_derived_doubled['method'].loc[::2]
        division_col = ufc_fights_reported_derived_doubled['division'].loc[::2]

        # flatten into a dataframe with fighter and opponent columns
        ufc_fights_predictive_even = ufc_fights_predictive.loc[::2].copy()
        ufc_fights_predictive_odd = ufc_fights_predictive.loc[1::2].copy()
        ufc_fights_predictive_even = ufc_fights_predictive_even[predictive_columns].reset_index(drop=True)
        ufc_fights_predictive_odd = ufc_fights_predictive_odd[predictive_columns].reset_index(drop=True)

        # make diff columns 
        ufc_fights_predictive_flattened_dict = {}
        ufc_fights_predictive_flattened_dict['fighter'] = fighter_col.to_numpy()
        ufc_fights_predictive_flattened_dict['opponent'] = opponent_col.to_numpy()
        ufc_fights_predictive_flattened_dict['date'] = date_col.to_numpy()
        ufc_fights_predictive_flattened_dict['result'] = result_col.to_numpy()
        ufc_fights_predictive_flattened_dict['method'] = method_col.to_numpy()
        ufc_fights_predictive_flattened_dict['division'] = division_col.to_numpy()

        for col in predictive_columns:
            if col not in ['fighter', 'opponent', 'result', 'method', 'division']:
                ufc_fights_predictive_flattened_dict[f'fighter_{col}'] = ufc_fights_predictive_even[col].values
                ufc_fights_predictive_flattened_dict[f'opponent_{col}'] = ufc_fights_predictive_odd[col].values
                
        ufc_fights_predictive_flattened = pd.DataFrame(ufc_fights_predictive_flattened_dict)
        return ufc_fights_predictive_flattened
            
            
    def make_ufc_fights_predictive_flattened_diffs(self, derived_doubled_df, shuffle=True, random_state=48):

        ufc_fights_reported_derived_doubled = derived_doubled_df.copy()

        non_predictive_columns = [
            'date',
            'fighter',
            'opponent',
            # 'method', # TODO predict method too
            # 'division', # TODO filter by division
            'stance', # TODO incorporate stance
        ]

        predictive_columns = [col for col in ufc_fights_reported_derived_doubled.columns if col not in non_predictive_columns]

        # shuffle pairs to avoid bias
        assert len(ufc_fights_reported_derived_doubled) % 2 == 0, "DataFrame length must be even to create pairs"
        shuffled_rows = []
        rng = np.random.default_rng(random_state)
        for i in range(0, len(ufc_fights_reported_derived_doubled), 2):
            pair = ufc_fights_reported_derived_doubled.iloc[i:i+2]
            if shuffle and rng.integers(0, 2):
                pair = pair.iloc[::-1]  # deterministic, outcome-independent orientation
            shuffled_rows.append(pair)
        # Concatenate back into a single DataFrame
        ufc_fights_reported_derived_doubled = pd.concat(shuffled_rows).reset_index(drop=True)

        # drop non-predictive columns
        ufc_fights_predictive = ufc_fights_reported_derived_doubled[predictive_columns]

        # grab fighter and opponent columns for diffing in flattened dataframe
        fighter_col = ufc_fights_reported_derived_doubled['fighter'].loc[::2]
        opponent_col = ufc_fights_reported_derived_doubled['opponent'].loc[::2]
        date_col = ufc_fights_reported_derived_doubled['date'].loc[::2]
        result_col = ufc_fights_reported_derived_doubled['result'].loc[::2]
        method_col = ufc_fights_reported_derived_doubled['method'].loc[::2]
        division_col = ufc_fights_reported_derived_doubled['division'].loc[::2]

        # flatten into a dataframe with fighter and opponent columns
        ufc_fights_predictive_even = ufc_fights_predictive.loc[::2].copy()
        ufc_fights_predictive_odd = ufc_fights_predictive.loc[1::2].copy()
        ufc_fights_predictive_even = ufc_fights_predictive_even[predictive_columns].reset_index(drop=True)
        ufc_fights_predictive_odd = ufc_fights_predictive_odd[predictive_columns].reset_index(drop=True)

        # make diff columns 
        ufc_fights_predictive_diffs_dict = {}
        ufc_fights_predictive_diffs_dict['fighter'] = fighter_col.to_numpy()
        ufc_fights_predictive_diffs_dict['opponent'] = opponent_col.to_numpy()
        ufc_fights_predictive_diffs_dict['date'] = date_col.to_numpy()
        ufc_fights_predictive_diffs_dict['result'] = result_col.to_numpy()
        ufc_fights_predictive_diffs_dict['method'] = method_col.to_numpy()
        ufc_fights_predictive_diffs_dict['division'] = division_col.to_numpy()

        for col in predictive_columns:
            if col not in ['fighter', 'opponent', 'result', 'method', 'division']:
                ufc_fights_predictive_diffs_dict[f'{col}_diff'] = ufc_fights_predictive_even[col].values - ufc_fights_predictive_odd[col].values
        # add a select few sum columns / higher order too
        # so we can determine absolute age and not just relative age
        ufc_fights_predictive_diffs_dict['age_sum'] = ufc_fights_predictive_even['age'].values + ufc_fights_predictive_odd['age'].values
        # I am thinking these are causing over fitting. test this, maybe include just sq diff?
        # ufc_fights_predictive_diffs_dict['age_sq_diff'] = ufc_fights_predictive_even['age'].values ** 2 - ufc_fights_predictive_odd['age'].values ** 2
        # ufc_fights_predictive_diffs_dict['age_sq_sum'] = ufc_fights_predictive_even['age'].values ** 2 + ufc_fights_predictive_odd['age'].values ** 2
                
        ufc_fights_predictive_diffs = pd.DataFrame(ufc_fights_predictive_diffs_dict)
        return ufc_fights_predictive_diffs
    
        
    def update_ufc_fight_data_for_website(self):
        updated_ufc_fights_reported_doubled = self.get('ufc_fights_reported_doubled')
        updated_ufc_fights_reported_doubled['index'] = list(range(updated_ufc_fights_reported_doubled.shape[0])) # add index column to dataframe

        json_columns = ['date', 'result', 'fighter', 'opponent', 'division', 'method', 'round', 'time', 'knockdowns', 'sub_attempts', 'reversals', 'takedowns_landed', 
                        'takedowns_attempts', 'sig_strikes_landed', 'sig_strikes_attempts', 'total_strikes_landed', 'total_strikes_attempts', 'head_strikes_landed',
                        'head_strikes_attempts', 'body_strikes_landed', 'body_strikes_attempts', 'leg_strikes_landed', 'leg_strikes_attempts', 'distance_strikes_landed', 
                        'distance_strikes_attempts', 'clinch_strikes_landed', 'clinch_strikes_attempts', 'ground_strikes_landed', 'ground_strikes_attempts', 
                        'index',]

        ufc_fight_data_for_website = updated_ufc_fights_reported_doubled[json_columns]

        # make new csv just to send it to json
        # this is inefficient and wastes space... but its just because its the only way I know to make a json file
        # of the correct format (fix needed but not super important)
        print('exporting updated ufc_fights_reported_derived_doubled.json for use in javascript portion of website')
        atomic_to_csv(
            ufc_fight_data_for_website,
            self.csv_filepaths['ufc_fight_data_for_website'],
            index=False,
        )

        # convert ufc_fights_reported_derived_doubled.csv to json files to read via javascript in website
        csvFilePath = self.csv_filepaths['ufc_fight_data_for_website']
        jsonFilePath = self.json_filepaths['ufc_fight_data_for_website']
        self.make_json(csvFilePath, jsonFilePath, 'index')
        
    def update_pictures(self):
        # updating the picture scrape
        # updated scraped fighter data (after running ufc_fights_reported_doubled_updated function from UFC_data_scraping file)
        fighter_stats = self.get('fighter_stats')
        names = list(fighter_stats['name'])

        print('Scraping pictures of newly added fighters from Google image search')
        # run this to update the image scrape
        for name in names:
            name_reduced = name.replace(" ", "")
            image_file_path = "content/images/" + str(1) + name_reduced + ".jpg"
            if os.path.isfile(image_file_path): # skip names that already have images
                continue
            self.scrape_pictures(name)
            
    def save_fightoddsio_to_vegas_odds_json_and_merge_with_predictions_df(self, predictions_df):
        # Third-party odds are optional enrichment. Keep an untouched model
        # forecast fallback so *any* scrape, schema, or merge failure cannot
        # discard a valid UFCStats/model publication.
        fallback = predictions_df.copy(deep=True)
        try:
            return self._merge_fightodds_with_predictions(
                predictions_df.copy(deep=True)
            )
        except Exception as error:
            fallback['odds source status'] = 'unavailable'
            fallback['odds source'] = ''
            fallback['odds observed at'] = ''
            fallback['market no-vig fighter probability'] = np.nan
            fallback['betting status'] = (
                'disabled_pending_market_relative_validation'
            )
            fallback['fighter bet bankroll percentage'] = np.nan
            fallback['opponent bet bankroll percentage'] = np.nan
            fallback['best fighter bookie'] = ''
            fallback['best opponent bookie'] = ''
            if 'average bookie odds' not in fallback:
                fallback['average bookie odds'] = pd.Series(
                    [None] * len(fallback), dtype=object
                )
            print(
                'WARNING: sportsbook enrichment failed after retrieval; '
                'publishing the independent model forecasts without book lines '
                f'({type(error).__name__}: {error})'
            )
            return fallback

    def _merge_fightodds_with_predictions(self, predictions_df):
        print('getting sportsbook odds from the configured market source')
        predictions_df['odds source status'] = 'unmatched'
        predictions_df['odds source'] = ''
        predictions_df['odds observed at'] = ''
        predictions_df['market no-vig fighter probability'] = np.nan
        predictions_df['betting status'] = 'disabled_pending_market_relative_validation'
        predictions_df['fighter bet bankroll percentage'] = np.nan
        predictions_df['opponent bet bankroll percentage'] = np.nan
        predictions_df['best fighter bookie'] = ''
        predictions_df['best opponent bookie'] = ''
        if 'average bookie odds' not in predictions_df:
            predictions_df['average bookie odds'] = pd.Series(
                [None] * len(predictions_df), dtype=object
            )
        try:
            odds_df = self.odds_getter.make_odds_df()
        except Exception as error:
            # Odds are enrichment, not the source of fight statistics or model
            # predictions.  Preserve a truthful no-odds card when this
            # independent site or Chrome is unavailable instead of aborting
            # the entire weekly publication.
            predictions_df['odds source status'] = 'unavailable'
            print(
                'WARNING: sportsbook odds are unavailable; publishing '
                f'predictions without book lines ({type(error).__name__}: {error})'
            )
            return predictions_df
        odds_source = getattr(self.odds_getter, 'last_source', '') or 'unknown'
        predictions_df['odds source'] = odds_source
        observed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        previous_vegas = self.get('vegas_odds', filetype='json')
        odds_df['fighter bet bankroll percentage'] = np.nan
        odds_df['opponent bet bankroll percentage'] = np.nan
        odds_df['best fighter bookie'] = ''
        odds_df['best opponent bookie'] = ''
        
        # TODO go through and figure out if any parlays have better expectation values
        # than the individual bets (2 leg and 3 leg parlays only probably worth it)
        
        # merge into predictions_df 
        matched_fights = 0
        for i in range(len(predictions_df)):
            fighter = predictions_df['fighter name'][i]
            opponent = predictions_df['opponent name'][i]
            # find row in odds_df where fighter and opponent match (could be in either order in the other df)
            odds_row = odds_df[same_name_vect(odds_df['fighter name'], fighter) & same_name_vect(odds_df['opponent name'], opponent)]
            fighter_a = 'fighter'
            fighter_b = 'opponent'
            if odds_row.empty:
                # try switching fighter and opponent
                opponent = predictions_df['fighter name'][i]
                fighter = predictions_df['opponent name'][i]
                fighter_a = 'opponent'
                fighter_b = 'fighter'
                odds_row = odds_df[same_name_vect(odds_df['fighter name'], fighter) & same_name_vect(odds_df['opponent name'], opponent)]
            if not odds_row.empty and 'source commence time' in odds_row:
                expected_day = pd.to_datetime(
                    predictions_df.at[i, 'date'], errors='coerce', utc=True
                )
                source_days = pd.to_datetime(
                    odds_row['source commence time'], errors='coerce', utc=True
                )
                within_card_window = source_days.notna()
                if pd.isna(expected_day):
                    within_card_window &= False
                else:
                    within_card_window &= (
                        (source_days.dt.normalize() - expected_day.normalize())
                        .dt.days.abs()
                        .le(1)
                    )
                odds_row = odds_row[within_card_window]
            if odds_row.empty:
                print(f'No odds found for {fighter} vs {opponent} from {odds_source}, skipping...')
                continue
            if len(odds_row) != 1:
                predictions_df.at[i, 'odds source status'] = 'ambiguous'
                print(
                    f'Ambiguous {odds_source} matchup for {fighter} vs {opponent}: '
                    f'{len(odds_row)} rows; skipping...'
                )
                continue
            matched_fights += 1
            predictions_df.at[i, 'odds source status'] = 'matched'
            # TODO update these with the actual bookies we are getting odds from (or just those I can actually use)
            source_columns = {column.casefold(): column for column in odds_row.columns}
            for bookie in self.bookies:
                fighter_source = source_columns.get(f'{fighter_a} {bookie}'.casefold())
                opponent_source = source_columns.get(f'{fighter_b} {bookie}'.casefold())
                if fighter_source and opponent_source:
                    predictions_df.at[i, f'fighter {bookie}'] = odds_row[fighter_source].iloc[0]
                    predictions_df.at[i, f'opponent {bookie}'] = odds_row[opponent_source].iloc[0]
            # add average odds for fighter and opponent
            consensus_odds = odds_row['average bookie odds'].iloc[0]
            consensus_probability = odds_row.get(
                'average bookie probability', pd.Series([np.nan], index=odds_row.index)
            ).iloc[0]
            if fighter_a == 'opponent' and isinstance(consensus_odds, (list, tuple)):
                consensus_odds = list(reversed(consensus_odds))
                if pd.notna(consensus_probability):
                    consensus_probability = 1.0 - float(consensus_probability)
            predictions_df.at[i, f'average bookie odds'] = consensus_odds
            predictions_df.at[i, 'market no-vig fighter probability'] = consensus_probability
            if pd.notna(consensus_probability):
                market_fighter_odds = self.odds_getter.probability_to_odds(
                    float(consensus_probability)
                )
                market_opponent_odds = self.odds_getter.probability_to_odds(
                    1.0 - float(consensus_probability)
                )
                predictions_df.at[i, 'forecast probability'] = float(
                    consensus_probability
                )
                predictions_df.at[i, 'forecast source'] = 'market_no_vig_consensus'
                predictions_df.at[i, 'forecast fighter odds'] = (
                    f'+{market_fighter_odds}'
                    if market_fighter_odds > 0 else str(market_fighter_odds)
                )
                predictions_df.at[i, 'forecast opponent odds'] = (
                    f'+{market_opponent_odds}'
                    if market_opponent_odds > 0 else str(market_opponent_odds)
                )
            predictions_df.at[i, 'odds observed at'] = observed_at
            # Preserve the original observation time on an identical retry.
            # This keeps no-change workflow runs byte-stable and prevents a
            # manual rerun from pretending unchanged lines were first seen
            # later than they really were.
            if not previous_vegas.empty:
                prediction_fighter_id = str(
                    predictions_df.at[i, 'fighter id']
                    if 'fighter id' in predictions_df else ''
                ).strip()
                prediction_opponent_id = str(
                    predictions_df.at[i, 'opponent id']
                    if 'opponent id' in predictions_df else ''
                ).strip()
                if (
                    prediction_fighter_id
                    and prediction_opponent_id
                    and {'fighter id', 'opponent id'}.issubset(previous_vegas.columns)
                ):
                    identity_match = (
                        previous_vegas['fighter id'].astype(str).eq(prediction_fighter_id)
                        & previous_vegas['opponent id'].astype(str).eq(prediction_opponent_id)
                    )
                else:
                    # Compatibility path for forecasts created before stable
                    # UFCStats fighter IDs were published.
                    identity_match = (
                        same_name_vect(
                            previous_vegas['fighter name'],
                            predictions_df.at[i, 'fighter name'],
                        )
                        & same_name_vect(
                            previous_vegas['opponent name'],
                            predictions_df.at[i, 'opponent name'],
                        )
                    )
                date_match = (
                    pd.to_datetime(previous_vegas['date'], errors='coerce').dt.normalize()
                    == pd.to_datetime(
                        predictions_df.at[i, 'date'], errors='coerce'
                    ).normalize()
                )
                previous_match = previous_vegas[identity_match & date_match]
                if len(previous_match) == 1:
                    previous_row = previous_match.iloc[0]
                    comparison_columns = [
                        'model id', 'predicted fighter odds', 'predicted opponent odds',
                        'average bookie odds', 'market no-vig fighter probability',
                        *[f'fighter {bookie}' for bookie in self.bookies],
                        *[f'opponent {bookie}' for bookie in self.bookies],
                    ]

                    def comparable(value):
                        if isinstance(value, (list, tuple, np.ndarray)):
                            return tuple(comparable(item) for item in value)
                        if pd.isna(value):
                            return None
                        return str(value)

                    quotes_unchanged = all(
                        comparable(previous_row.get(column))
                        == comparable(predictions_df.at[i, column])
                        for column in comparison_columns
                        if column in predictions_df
                    )
                    previous_observed_at = previous_row.get('odds observed at')
                    if quotes_unchanged and pd.notna(previous_observed_at) and str(previous_observed_at):
                        predictions_df.at[i, 'odds observed at'] = previous_observed_at
            
            # add expected values for fighter and opponent
            fighter_predicted_odds = predictions_df.at[i, 'predicted fighter odds']
            if pd.isna(fighter_predicted_odds) or fighter_predicted_odds == '':
                continue
            # The available history does not yet demonstrate incremental edge
            # over timestamped no-vig market probabilities.  Collect the data
            # now, but do not turn an unvalidated model residual into a wager.
            model_id = predictions_df.at[i, 'model id'] if 'model id' in predictions_df else ''
            if pd.notna(model_id) and str(model_id).strip():
                continue
            # search over all bookies for the best odds
            fighter_bookie_odds_dict = {}
            opponent_bookie_odds_dict = {}
            fighter_bookie_kelly_dict = {}
            opponent_bookie_kelly_dict = {}
            for bookie in self.bookies:
                bookie_fighter_odds = predictions_df.at[i, f'fighter {bookie}']
                bookie_opponent_odds = predictions_df.at[i, f'opponent {bookie}']
                # check if these have integer values or empty values (default is an empty string)
                parsed_fighter_odds = self.odds_getter.parse_american_odds(bookie_fighter_odds)
                parsed_opponent_odds = self.odds_getter.parse_american_odds(bookie_opponent_odds)
                if parsed_fighter_odds is not None and parsed_opponent_odds is not None:
                    fighter_bookie_odds_dict[bookie] = parsed_fighter_odds
                    opponent_bookie_odds_dict[bookie] = parsed_opponent_odds
                    predicted_fighter_odds = int(fighter_predicted_odds)
                    fighter_kelly, opponent_kelly = get_kelly_bet_from_ev_and_dk_odds(
                        predicted_fighter_odds, parsed_fighter_odds, parsed_opponent_odds
                    )
                    fighter_bookie_kelly_dict[bookie] = fighter_kelly
                    opponent_bookie_kelly_dict[bookie] = opponent_kelly
            # fight highest kelly percentage for fighter and opponent
            if fighter_bookie_kelly_dict:
                best_fighter_bookie = max(fighter_bookie_kelly_dict, key=fighter_bookie_kelly_dict.get)
                best_opponent_bookie = max(opponent_bookie_kelly_dict, key=opponent_bookie_kelly_dict.get)
                predictions_df.at[i, 'best fighter bookie'] = best_fighter_bookie
                predictions_df.at[i, 'best opponent bookie'] = best_opponent_bookie
                predictions_df.at[i, 'fighter bet bankroll percentage'] = fighter_bookie_kelly_dict[best_fighter_bookie]
                predictions_df.at[i, 'opponent bet bankroll percentage'] = opponent_bookie_kelly_dict[best_opponent_bookie]

        print(
            f'Matched {odds_source} lines for {matched_fights}/'
            f'{len(predictions_df)} UFCStats fights'
        )
        if matched_fights < len(predictions_df):
            print(f'{odds_source} matchups available for comparison:')
            print(odds_df[['fighter name', 'opponent name']].to_string(index=False))
        return predictions_df
    
    # TODO vegas_odds is really not the right name for this data as it contains predictions, not just vegas odds
    def update_vegas_odds(self, vegas_odds):
        #save to json
        result = vegas_odds.to_json()
        parsed = json.loads(result)
        jsonFilePath = self.json_filepaths['vegas_odds']
        atomic_write_text(jsonFilePath, json.dumps(parsed, indent=4))
        print('saved to '+jsonFilePath)
    
    def update_prediction_history(self):
        if self.update_time == 0:
            print('No new fights have occurred since last update, skipping prediction history update')
            return

        vegas_odds_old=self.get('vegas_odds', filetype='json') # this is the old vegas odds dataframe (from last week)
        ufc_fights_reported_doubled = self.get('ufc_fights_reported_doubled') # THIS SHOULD HAVE BEEN UPDATED AT THIS POINT! WE SHOULD ADD A CHECK TO CHECK THIS
        prediction_history=self.get('prediction_history', filetype='json')
        
        currentBankroll = prediction_history['current bankroll after'].iloc[0] if 'current bankroll after' in prediction_history.columns else 300.0; # default bankroll if not present in prediction history

        # getting rid of fights that didn't actually happen and adding correctness results of those that did
        vegas_odds_old = self.update_prediction_correctness(vegas_odds_old, ufc_fights_reported_doubled, currentBankroll)

        #making a copy of vegas_odds
        vegas_odds_copy=vegas_odds_old.copy()

        #add the newly scraped fights and predicted fights to the history of prediction list (idea: might be better to wait to join until after the fights happen)
        prediction_history = pd.concat([vegas_odds_copy, prediction_history], axis = 0).reset_index(drop=True)

        #saving the new prediction_history dataframe to json
        result = prediction_history.to_json()
        parsed = json.loads(result)
        prediction_history_filtpath = self.json_filepaths['prediction_history']
        
        # TODO USE THE SAVE FUNCTION OF THE DATA HANDLER
        atomic_write_text(prediction_history_filtpath, json.dumps(parsed, indent=4))
            
        print(f'saved to {prediction_history_filtpath}')
        
    def update_card_info(self, card=None):
        card_date, card_title, fights_list = card or self.get_next_fight_card()
        card_date = self.convert_scraped_date_to_standard_date(card_date)

        # New scrapes carry the immutable UFCStats event URL alongside each
        # matchup.  Keep accepting older/manual three-field card fixtures, but
        # publish stable event lineage whenever it is available so market
        # snapshots never have to identify a card from its display title.
        event_url = (
            str(fights_list[0][5]).strip()
            if fights_list and len(fights_list[0]) > 5 and fights_list[0][5]
            else ''
        )
        card_info_dict = {
            "date": card_date,
            "title": card_title,
            "event_url": event_url,
            "event_id": ufcstats_identity(event_url),
        }

        print('Writing upcoming card info to content/data/external/card_info.json')
        atomic_write_text(
            self.json_filepaths['card_info'], json.dumps(card_info_dict)
        )
        return card_date, card_title, fights_list
        
    def scrape_pictures(self, name):
        try:
            URL = "https://www.google.com/search?q="+name+" ufc fighting" + \
                "&sxsrf=ALeKk03xBalIZi7BAzyIRw8R4_KrIEYONg:1620885765119&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjv44CC_sXwAhUZyjgGHSgdAQ8Q_AUoAXoECAEQAw&cshid=1620885828054361"
            page = requests.get(URL)
            soup = BeautifulSoup(page.content, 'html.parser')
            # ... or ... image_tags = soup.find_all('img', class_='t0fcAb')
            image_tags = soup.find_all('img')
            links = []
            for image_tag in image_tags:
                links.append(image_tag['src'])
                name_reduced = name.replace(" ", "")
            for i in range(1, 5):
                urllib.request.urlretrieve(links[i], f"{git_root}/src/content/images/"+str(i)+name_reduced+".jpg")
            print('scraped 5 random pictures of '+name+' from Google search')

        except:
            print('The scrape did not work for '+name)
        
    # Function to convert a CSV to JSON
    # Takes the file paths as arguments
    def make_json(self, csvFilePath, jsonFilePath, column):

        # create a dictionary
        data = {}

        # Open a csv reader called DictReader
        with open(csvFilePath, encoding='utf-8') as csvf:
            csvReader = csv.DictReader(csvf)

            # Convert each row into a dictionary
            # and add it to data
            for rows in csvReader:

                # primary key given by column variable
                key = rows[column]
                data[key] = rows

        # Open a json writer, and use the json.dumps()
        # function to dump data
        atomic_write_text(jsonFilePath, json.dumps(data, indent=4))
            
    # thresh is the number of bookies we allow to not have odds on the books
    # TODO name should better indicate the context
    def drop_irrelevant_fights(self, df, thresh):
        irr = []
        for i in df.index:
            count = 0
            row = list(df.loc[i])
            for j in row:
                if j == '':
                    count += 1
            if count > 2*thresh:
                irr.append(i)
        df = df.drop(irr)
        return df

    # TODO name should better indicate the context
    def drop_repeats(self, df):
        irr = []
        ufc_fights_predictive_flattened_diffs = self.get('ufc_fights_predictive_flattened_diffs')
        for i in df.index:
            fname = df['fighter name'][i]
            oname = df['opponent name'][i]
            for j in range(200):
                fname_old = ufc_fights_predictive_flattened_diffs['fighter'][j]
                oname_old = ufc_fights_predictive_flattened_diffs['opponent'][j]
                if (same_name(fname, fname_old) and same_name(oname, oname_old)) or (same_name(oname, fname_old) and same_name(fname, oname_old)):
                    irr.append(i)
        df = df.drop(irr)
        return df
    
    # TODO name should better indicate the context
    def update_prediction_correctness(self, vegas_odds_old, ufc_fights_reported_doubled, currentBankroll):
        r"""
        This function checks the vegas odds dataframe against the ufc fights dataframe to find fights that didn't happen
        and to add correctness results for those that did happen. It returns a list of indices of fights that didn't happen.
        It also updates the vegas odds dataframe with correctness results for the fights that did happen.
        """
        # Retain every attempted forecast.  Coverage failures, cancellations,
        # draws, and no-contests are part of model performance and must not be
        # silently removed from history.
        vegas_odds_old['fighter bet'] = 0.0
        vegas_odds_old['opponent bet'] = 0.0
        vegas_odds_old['current bankroll after'] = 0.0
        vegas_odds_old['bet result'] = 'N/A'
        vegas_odds_old['forecast status'] = 'unmatched_or_canceled'
        for score_column in ('correct?', 'model correct?'):
            if score_column in vegas_odds_old:
                vegas_odds_old[score_column] = vegas_odds_old[
                    score_column
                ].astype(object)
            else:
                vegas_odds_old[score_column] = pd.Series(
                    'N/A', index=vegas_odds_old.index, dtype=object
                )
        for index1, row1 in vegas_odds_old.iloc[::-1].iterrows(): # iterate backwards in the order the fights actually happened
            card_date = row1['date']
            
            forecast_prediction_value = row1.get('forecast fighter odds')
            prediction_value = (
                forecast_prediction_value
                if pd.notna(forecast_prediction_value)
                and str(forecast_prediction_value).strip()
                else row1.get('predicted fighter odds')
            )
            model_prediction_value = row1.get('predicted fighter odds')
            if pd.isna(prediction_value) or str(prediction_value).strip() == '':
                vegas_odds_old.at[index1, 'correct?'] = 'N/A'
                vegas_odds_old.at[index1, 'model correct?'] = 'N/A'
                vegas_odds_old.at[index1, 'forecast status'] = 'no_prediction'
                vegas_odds_old.at[index1, 'current bankroll after'] = currentBankroll
                print('no prediction made for fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'])
                continue
            
            fighter_odds = self.odds_getter.parse_american_odds(prediction_value)
            if fighter_odds is None:
                raise ValueError(
                    f'Invalid forecast fighter odds {prediction_value!r} for '
                    f'{row1["fighter name"]} vs {row1["opponent name"]}'
                )
            model_fighter_odds = self.odds_getter.parse_american_odds(
                model_prediction_value
            )
            best_fighter_bookie = row1['best fighter bookie']
            best_opponent_bookie = row1['best opponent bookie']
            
            # check if we ever found odds for the fighter and opponent
            best_fighter_bookie_odds = row1.get(f'fighter {best_fighter_bookie}')
            if not best_fighter_bookie_odds:
                print(f'No odds found for fighter {row1["fighter name"]} from bookie {best_fighter_bookie}, skipping...')
            best_opponent_bookie_odds = row1.get(f'opponent {best_opponent_bookie}')
            if not best_opponent_bookie_odds:
                print(f'No odds found for opponent {row1["opponent name"]} from bookie {best_opponent_bookie}, skipping...')
            
            best_fighter_bookie_odds = self.odds_getter.parse_american_odds(
                best_fighter_bookie_odds
            )
            best_opponent_bookie_odds = self.odds_getter.parse_american_odds(
                best_opponent_bookie_odds
            )
                
            fighter_bankroll_percentage = row1.get('fighter bet bankroll percentage', 0.0)
            if not fighter_bankroll_percentage:
                print(f'No bankroll percentage found for fighter {row1["fighter name"]}, skipping...')
                fighter_bankroll_percentage = 0.0
                
            opponent_bankroll_percentage = row1.get('opponent bet bankroll percentage', 0.0)
            if not opponent_bankroll_percentage:
                print(f'No bankroll percentage found for opponent {row1["opponent name"]}, skipping...')
                opponent_bankroll_percentage = 0.0
                
            fighter_bankroll_percentage = float(fighter_bankroll_percentage)
            opponent_bankroll_percentage = float(opponent_bankroll_percentage)
            if best_fighter_bookie_odds is None:
                fighter_bankroll_percentage = 0.0
            if best_opponent_bookie_odds is None:
                opponent_bankroll_percentage = 0.0
            
            # if a prediction was made, check if the fight actually happened and then check if the prediction / bet was correct / won
            # TODO this is slow but sort of necessary if we need to add multiple cards at the same time
            # import ipdb;ipdb.set_trace(context=10)
            card_timestamp = pd.to_datetime(card_date, errors='raise').normalize()
            fight_dates = pd.to_datetime(
                ufc_fights_reported_doubled['date'], errors='coerce'
            ).dt.normalize()
            relevant_fights = ufc_fights_reported_doubled[fight_dates == card_timestamp]
            print(f'searching through {relevant_fights.shape[0]//2} confirmed fights on {str(card_date).split(" ")[0]} for {row1["fighter name"]} vs {row1["opponent name"]}')

            forecast_fighter_id = str(row1.get('fighter id', '')).strip().lower()
            forecast_opponent_id = str(row1.get('opponent id', '')).strip().lower()
            
            match_found = False
            for index2, row2 in relevant_fights.iterrows():
                has_stable_ids = bool(forecast_fighter_id and forecast_opponent_id)
                if has_stable_ids:
                    is_match = (
                        ufcstats_identity(row2.get('fighter_url')) == forecast_fighter_id
                        and ufcstats_identity(row2.get('opponent_url')) == forecast_opponent_id
                    )
                else:
                    # Compatibility path for historical forecasts that predate
                    # the stable-ID schema.
                    is_match = (
                        same_name(row1['fighter name'], row2['fighter'])
                        and same_name(row1['opponent name'], row2['opponent'])
                    )
                if is_match:
                    match_found = True
                    print('adding fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'])
                    actual_result = row2['result']
                    vegas_odds_old.at[index1, 'fight id'] = ufcstats_identity(
                        row2.get('fight_url')
                    )
                    vegas_odds_old.at[index1, 'fight url'] = row2.get('fight_url', '')
                    vegas_odds_old.at[index1, 'actual result'] = actual_result

                    def score_pick(american_odds):
                        if actual_result in ['D', 'NC'] or american_odds is None:
                            return 'N/A'
                        if abs(int(american_odds)) == 100:
                            return 'N/A'
                        picked_fighter = int(american_odds) < 0
                        fighter_won = actual_result == 'W'
                        return int(picked_fighter == fighter_won)

                    vegas_odds_old.at[index1, 'correct?'] = score_pick(fighter_odds)
                    vegas_odds_old.at[index1, 'model correct?'] = score_pick(
                        model_fighter_odds
                    )
                    if actual_result in ['D', 'NC']:
                        result_status = {'D': 'draw', 'NC': 'no_contest'}[actual_result]
                        vegas_odds_old.at[index1, 'forecast status'] = result_status
                    elif abs(int(fighter_odds)) == 100:
                        vegas_odds_old.at[index1, 'forecast status'] = 'completed'
                    else:
                        vegas_odds_old.at[index1, 'forecast status'] = 'completed'
                    # update the bankroll based on the bet made
                    fighter_bet = 0.0
                    opponent_bet = 0.0
                    fighter_payout = 0.0
                    opponent_payout = 0.0
                    bet_result = 'N/A'
                    if fighter_bankroll_percentage > 0: # check if we even made a bet on the fighter
                        fighter_bet = fighter_bankroll_percentage / 100 * currentBankroll
                        vegas_odds_old.at[index1, 'fighter bet'] = fighter_bet
                        bet_result = actual_result
                        fighter_payout = bet_payout(best_fighter_bookie_odds, fighter_bet, bet_result)
                    if opponent_bankroll_percentage > 0: # check if we even made a bet on the opponent
                        opponent_bet = opponent_bankroll_percentage / 100 * currentBankroll
                        vegas_odds_old.at[index1, 'opponent bet'] = opponent_bet
                        # win the bet if the opponent wins (the result column is the result of the fighter, so if the fighter wins, the opponent loses)
                        if actual_result in ['D', 'NC']:
                            bet_result = actual_result
                        else:
                            bet_result = 'L' if actual_result == 'W' else 'W'
                        opponent_payout = bet_payout(best_opponent_bookie_odds, opponent_bet, bet_result)
                    currentBankroll = currentBankroll + fighter_payout + opponent_payout - fighter_bet - opponent_bet
                    # TODO why is this set to dtype int?
                    vegas_odds_old.at[index1, 'current bankroll after'] = currentBankroll
                    vegas_odds_old.at[index1, 'bet result'] = bet_result
                    break
            if not match_found:
                vegas_odds_old.at[index1, 'correct?'] = 'N/A'
                vegas_odds_old.at[index1, 'model correct?'] = 'N/A'
                vegas_odds_old.at[index1, 'current bankroll after'] = currentBankroll
                print('fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'] + ' not found in ufc_fights_reported_derived_doubled.csv')
        return vegas_odds_old
    
    def make_derived_doubled_vector_for_fight(self, new_rows):
        ufc_fights_reported_doubled = self.get('ufc_fights_reported_doubled')
        ufc_fights_reported_doubled['date'] = pd.to_datetime(ufc_fights_reported_doubled['date'], errors='coerce')
        ufc_fights_reported_doubled.set_index('date', inplace=True)
        ufc_fights_reported_doubled = ufc_fights_reported_doubled.sort_index(kind='stable')

        ufc_fights_reported_derived_doubled = self.get('ufc_fights_reported_derived_doubled')
        # import ipdb; ipdb.set_trace(context=10)  # uncomment to debug
        ufc_fights_reported_derived_doubled['date'] = pd.to_datetime(ufc_fights_reported_derived_doubled['date'], errors='coerce')
        ufc_fights_reported_derived_doubled.set_index('date', inplace=True)
        ufc_fights_reported_derived_doubled = ufc_fights_reported_derived_doubled.sort_index(kind='stable')
        
        # add new rows to bottom of derived dataframe in reverse order
        new_rows_derived = new_rows[['fighter', 'opponent', 'date', 'result', 'method', 'division']].copy()
        new_rows_derived['date'] = pd.to_datetime(new_rows_derived['date'], errors='coerce')
        new_rows_derived.set_index('date', inplace=True)
        new_rows_derived = new_rows_derived.sort_index(kind='stable')
        ufc_fights_reported_derived_doubled = pd.concat(
            [ufc_fights_reported_derived_doubled, new_rows_derived], axis=0
        ).sort_index(kind='stable')
        
        # add new rows to bottom of doubled reported dataframe in reverse order
        ufc_fights_reported_doubled = pd.concat(
            [ufc_fights_reported_doubled, new_rows_derived], axis=0
        ).sort_index(kind='stable')
        # replace all nan values with zeros in just the rows we added (otherwise rolling averages will all be nan since we subtract off the current row to not include current fight)
        ufc_fights_reported_doubled.iloc[-len(new_rows_derived):] = ufc_fights_reported_doubled.iloc[-len(new_rows_derived):].replace(np.nan, 0.0)
        
        names_to_update = new_rows.fighter.drop_duplicates()
        
        ufc_fights_reported_derived_doubled = self.fill_in_statistics_for_fights(ufc_fights_reported_derived_doubled, ufc_fights_reported_doubled, names_to_update)
                
        # Upcoming fights are later than the completed-history cutoff, so the
        # newly built pair is at the end after the explicit temporal sort.
        last_two_rows = ufc_fights_reported_derived_doubled.iloc[-len(new_rows_derived):]
        ufc_fights_reported_derived_doubled = last_two_rows
        return ufc_fights_reported_derived_doubled.reset_index()
    
                
    def populate_new_fights_with_statistics(self, new_rows):
        ufc_fights_reported_doubled = self.get('ufc_fights_reported_doubled')
        ufc_fights_reported_doubled['date'] = pd.to_datetime(ufc_fights_reported_doubled['date'], errors='coerce')
        ufc_fights_reported_doubled.set_index('date', inplace=True)
        ufc_fights_reported_doubled = ufc_fights_reported_doubled.sort_index(kind='stable')

        ufc_fights_reported_derived_doubled = self.get('ufc_fights_reported_derived_doubled')
        ufc_fights_reported_derived_doubled['date'] = pd.to_datetime(ufc_fights_reported_derived_doubled['date'], errors='coerce')
        ufc_fights_reported_derived_doubled.set_index('date', inplace=True)
        ufc_fights_reported_derived_doubled = ufc_fights_reported_derived_doubled.sort_index(kind='stable')
        
        # add new rows to bottom of derived dataframe in reverse order
        new_rows_derived = new_rows[['fighter', 'opponent', 'date', 'result', 'method', 'division']].copy()
        new_rows_derived['date'] = pd.to_datetime(new_rows_derived['date'], errors='coerce')
        new_rows_derived.set_index('date', inplace=True)
        new_rows_derived = new_rows_derived.sort_index(kind='stable')
        ufc_fights_reported_derived_doubled = pd.concat(
            [ufc_fights_reported_derived_doubled, new_rows_derived], axis=0
        ).sort_index(kind='stable')
        
        names_to_update = new_rows.fighter.drop_duplicates()
        
        ufc_fights_reported_derived_doubled = self.fill_in_statistics_for_fights(ufc_fights_reported_derived_doubled, ufc_fights_reported_doubled, names_to_update)
                
        return ufc_fights_reported_derived_doubled
    
    ########### FUNCTIONS USED IN update_data_csvs_and_jsons.py ###########
    
    def fill_in_statistics_for_fights(self, ufc_fights_reported_derived_doubled, ufc_fights_reported_doubled, names_to_update=None):
        fighter_stats = self.get('fighter_stats')
        if names_to_update is None:
            names_to_update = list(fighter_stats['name'].unique())
        # GOAL reproduce these statistics 
        
        # SOMETIMES ROAD TO UFC OR OTHER SIMILAR EVENTS SCREW THIS UP... careful
        ufc_fights_reported_derived_doubled = ufc_fights_reported_derived_doubled.copy()
        ufc_fights_reported_doubled = ufc_fights_reported_doubled.copy()
        ufc_fights_reported_derived_doubled = (
            ufc_fights_reported_derived_doubled.sort_index(kind='stable')
        )
        ufc_fights_reported_doubled = (
            ufc_fights_reported_doubled.sort_index(kind='stable')
        )

        physical_stats = [u'age', u'height', u'reach', u'stance']
        # the rest will have total, l2y and l5y versions
        record_stats = [u'wins', u'losses', u'wins_ko', u'wins_sub', u'wins_dec', u'losses_ko', u'losses_sub' u'losses_dec']
        # the following will also have inflicted (inf) and absorbed (abs) versions
        grappling_event_stats = [u'reversals', u'control', u'sub_attempts'] # does not include landed/attempted (hence event)
        striking_event_stats = [u'knockdowns'] # does not include landed/attempted (hence event)
        # the following will also have attempts and landed versions
        grappling_stats = [u'takedowns']
        striking_stats = [u'sig_strikes', u'total_strikes', u'head_strikes', u'body_strikes', u'leg_strikes', u'distance_strikes', u'clinch_strikes', u'ground_strikes']

        for idx, name in enumerate(names_to_update):
            print(f'Processing fighter {idx+1}/{len(names_to_update)}: {name}')
            fighter_inf_mask = same_name_vect(ufc_fights_reported_doubled['fighter'], name)
            fighter_mask = fighter_inf_mask # mask where the fighter is the given name (choosing to make this name without the word inf to avoid confusion later)
            fighter_abs_mask = same_name_vect(ufc_fights_reported_doubled['opponent'], name)
            localized_df     = ufc_fights_reported_doubled[fighter_inf_mask].copy() # to store results of computations for this fighter
            localized_df_inf = ufc_fights_reported_doubled[fighter_inf_mask].copy() # to compute inflicted stats for this fighter
            localized_df_abs = ufc_fights_reported_doubled[fighter_abs_mask].copy() # to compute absorbed stats for this fighter
            
            fighter_1_wins_mask = ufc_fights_reported_doubled['result'] == 'W'
            # Use these to do stuff like fight math and fighter score
            global_inf_wins_mask = fighter_inf_mask & fighter_1_wins_mask
            global_inf_losses_mask = fighter_inf_mask & ~fighter_1_wins_mask
            
            # make localized versions of the masks above     
            localized_inf_wins_mask = localized_df_inf['result'] == 'W'
            localized_inf_losses_mask = localized_df_inf['result'] == 'L'
            
            # make corresponding dataframes
            localized_inf_wins_df = localized_df_inf[localized_inf_wins_mask].copy()
            localized_inf_losses_df = localized_df_inf[localized_inf_losses_mask].copy()
            
            # find all people who this fighter has beaten
            fighter_has_beaten = localized_inf_wins_df['opponent'].unique()
            # find all people who this fighter has lost to
            fighter_has_lost_to = localized_inf_losses_df['opponent'].unique()
            # make dataframe of all fights where fighter won or someone they beat won
            fighter_2deg_of_sep_wins_df = ufc_fights_reported_doubled.loc[fighter_inf_mask | (ufc_fights_reported_doubled['fighter'].isin(fighter_has_beaten) & (ufc_fights_reported_doubled['result'] == 'W'))].copy()
            # make dataframe of all fights where fighter lost or someone they lost to lost
            fighter_2deg_of_sep_loss_df = ufc_fights_reported_doubled.loc[fighter_inf_mask | (ufc_fights_reported_doubled['fighter'].isin(fighter_has_lost_to) & (ufc_fights_reported_doubled['result'] == 'L'))].copy()
            # maybe include cross terms too, e.g. losses of people you beat or wins of people you lost to
            
            # compute record stats first
            record_indicator_df = pd.DataFrame(index=localized_df.index)  # will hold indicators for wins, losses, etc for cumsum calculations
            # do some cumsum computations to make rolling averages for the fighter
            record_indicator_df['won'] = (localized_df['result'] == 'W').astype(int)
            record_indicator_df['won_ko'] = ((localized_df['result'] == 'W') & (localized_df['method'].str.contains('KO|TKO', na=False))).astype(int)
            record_indicator_df['won_sub'] = ((localized_df['result'] == 'W') & (localized_df['method'].str.contains('SUB', na=False))).astype(int)
            record_indicator_df['won_dec'] = ((localized_df['result'] == 'W') & (localized_df['method'].str.contains('DEC', na=False))).astype(int)
            record_indicator_df['lost'] = (localized_df['result'] == 'L').astype(int)
            record_indicator_df['lost_ko'] = ((localized_df['result'] == 'L') & (localized_df['method'].str.contains('KO|TKO', na=False))).astype(int)
            record_indicator_df['lost_sub'] = ((localized_df['result'] == 'L') & (localized_df['method'].str.contains('SUB', na=False))).astype(int)
            record_indicator_df['lost_dec'] = ((localized_df['result'] == 'L') & (localized_df['method'].str.contains('DEC', na=False))).astype(int)
            record_indicator_df['num_fights'] = np.arange(0, len(localized_df))  # cumulative fights
            # column of all ones to use for cumsum calculations
            record_indicator_df['ones'] = 1
            
            stats_to_add_to_main_df = [] # keep track of new columns we are adding to the main df (to avoid highly fragmented df warning)
            new_columns_dict = {}
            
            # add physical stats (age, height, reach, stance) which don't need rolling averages
            physical_stats = [u'age', u'height', u'reach', u'stance']

            fighter_stats_results = get_fighter_stats(name, fighter_stats)
            if fighter_stats_results is None:
                print(f'Warning: Either no stats or too many stats found for fighter {name}, not populating fighters statistics')
                continue
            height, reach, dob, stance_ = fighter_stats_results
            # use dob to compute age at time of fight
            if pd.isna(dob):
                age_series = pd.Series([np.nan] * len(localized_df), index=localized_df.index)
            else:
                dob_date = pd.to_datetime(dob, errors='coerce')
                age_series = (localized_df.index - dob_date).days / 365.25 # ends up in weird format so we need to convert to a numpy array first
                age_series = pd.Series(np.array(age_series), index=localized_df.index)
            # add stats to new_columns_dict
            new_columns_dict['age'] = age_series
            new_columns_dict['height'] = pd.Series([height] * len(localized_df), index=localized_df.index)
            new_columns_dict['reach'] = pd.Series([reach] * len(localized_df), index=localized_df.index)
            new_columns_dict['stance'] = pd.Series([stance_] * len(localized_df), index=localized_df.index)
            stats_to_add_to_main_df.extend(['age', 'height', 'reach', 'stance'])
            
            
            # record statistic columns 
            for stat_name, stat_indicator in zip(['wins', 'wins_ko', 'wins_sub', 'wins_dec', 'losses', 'losses_ko', 'losses_sub', 'losses_dec', 'num_fights'],
                                                ['won', 'won_ko', 'won_sub', 'won_dec', 'lost', 'lost_ko', 'lost_sub', 'lost_dec', 'ones']):
                for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                    new_col_name = f'{timeframe}_{stat_name}'
                    stats_to_add_to_main_df.append(new_col_name)
                    new_columns_dict[new_col_name] = make_cumsum_before_current_fight(record_indicator_df, stat_indicator, timeframe=timeframe)
                    
            fighter_2deg_wins_mask = same_name_vect(fighter_2deg_of_sep_wins_df['fighter'], name)
            fighter_2deg_losses_mask = same_name_vect(fighter_2deg_of_sep_loss_df['fighter'], name)
            for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                new_col_name = f'{timeframe}_wins_wins'
                stats_to_add_to_main_df.append(new_col_name)
                wins_wins_extended = count_wins_wins_before_fight(fighter_2deg_of_sep_wins_df, name, timeframe=timeframe)
                # get the sub series that has the fighter as the fighter (not opponent)
                wins_wins = wins_wins_extended[fighter_2deg_wins_mask]
                new_columns_dict[new_col_name] = wins_wins
                
                new_col_name = f'{timeframe}_losses_losses'
                stats_to_add_to_main_df.append(new_col_name)
                losses_losses_extended = count_losses_losses_before_fight(fighter_2deg_of_sep_loss_df, name, timeframe=timeframe)
                # get the sub series that has the fighter as the fighter (not opponent)
                losses_losses = losses_losses_extended[fighter_2deg_losses_mask]
                new_columns_dict[new_col_name] = losses_losses
                
                new_col_name = f'{timeframe}_fight_math'
                stats_to_add_to_main_df.append(new_col_name)
                fight_math_extended = fight_math(name, fighter_2deg_of_sep_wins_df, timeframe=timeframe)
                fight_math_col = fight_math_extended[fighter_2deg_wins_mask]
                new_columns_dict[new_col_name] = fight_math_col
                
            # TODO ADD DOMINANCE SCORES PER FIGHT
        
            # compute grappling stats
            for stat in grappling_event_stats:
                col_name = f'{stat}'
                for inf_abs in ['inf', 'abs']:
                    for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                        new_col_name = f'{timeframe}_{inf_abs}_{col_name}_per_min'
                        stats_to_add_to_main_df.append(new_col_name)
                        if inf_abs == 'inf':
                            new_columns_dict[new_col_name] = make_avg_before_current_fight(
                                localized_df_inf, # use inflicted stats df to find inflicted stats
                                col_name, 
                                timeframe=timeframe, 
                                landed_attempts=None
                                )
                        elif inf_abs == 'abs':
                            new_columns_dict[new_col_name] = make_avg_before_current_fight(
                                localized_df_abs, # use absorbed stats df to find absorbed stats
                                col_name, 
                                timeframe=timeframe, 
                                landed_attempts=None
                                )
                            
            for stat in striking_event_stats:
                col_name = f'{stat}'
                for inf_abs in ['inf', 'abs']:
                    for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                        new_col_name = f'{timeframe}_{inf_abs}_{col_name}_per_min'
                        stats_to_add_to_main_df.append(new_col_name)
                        if inf_abs == 'inf':
                            new_columns_dict[new_col_name] = make_avg_before_current_fight(
                                localized_df_inf, # use inflicted stats df to find inflicted stats
                                col_name, 
                                timeframe=timeframe, 
                                landed_attempts=None
                                )
                        elif inf_abs == 'abs':
                            new_columns_dict[new_col_name] = make_avg_before_current_fight(
                                localized_df_abs, # use absorbed stats df to find absorbed stats
                                col_name, 
                                timeframe=timeframe, 
                                landed_attempts=None
                                )
                            
            # adding grappling stats
            for stat in grappling_stats:
                col_name = f'{stat}'
                for inf_abs in ['inf', 'abs']:
                    for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                        for landed_attempts in ['landed', 'attempts']:
                            new_col_name_per_min = f'{timeframe}_{inf_abs}_{col_name}_{landed_attempts}_per_min'
                            stats_to_add_to_main_df.append(new_col_name_per_min)
                            if inf_abs == 'inf':
                                new_columns_dict[new_col_name_per_min] = make_avg_before_current_fight(
                                    localized_df_inf, # use inflicted stats df to find inflicted stats
                                    col_name, 
                                    timeframe=timeframe, 
                                    landed_attempts=landed_attempts
                                    )
                            elif inf_abs == 'abs':
                                new_columns_dict[new_col_name_per_min] = make_avg_before_current_fight(
                                    localized_df_abs, # use absorbed stats df to find absorbed stats
                                    col_name, 
                                    timeframe=timeframe, 
                                    landed_attempts=landed_attempts
                                    )
                        # division by number of minutes cancels out, so accuracy is just landed / attempts
                        new_col_name_accuracy = f'{timeframe}_{inf_abs}_{col_name}_accuracy'
                        stats_to_add_to_main_df.append(new_col_name_accuracy)
                        accuracy = new_columns_dict[f'{timeframe}_{inf_abs}_{col_name}_landed_per_min'] / new_columns_dict[f'{timeframe}_{inf_abs}_{col_name}_attempts_per_min'].replace(0, np.nan) # avoid division by zero
                        accuracy.replace(np.nan, 0, inplace=True)  # replace NaN with 0
                        new_columns_dict[new_col_name_accuracy] = accuracy
                                
            # adding striking stats
            for stat in striking_stats:
                col_name = f'{stat}'
                for inf_abs in ['inf', 'abs']:
                    for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                        for landed_attempts in ['landed', 'attempts']:
                            new_col_name = f'{timeframe}_{inf_abs}_{col_name}_{landed_attempts}_per_min'
                            stats_to_add_to_main_df.append(new_col_name)
                            if inf_abs == 'inf':
                                new_columns_dict[new_col_name] = make_avg_before_current_fight(
                                    localized_df_inf, # use inflicted stats df to find inflicted stats
                                    col_name, 
                                    timeframe=timeframe, 
                                    landed_attempts=landed_attempts
                                    )
                            elif inf_abs == 'abs':
                                new_columns_dict[new_col_name] = make_avg_before_current_fight(
                                    localized_df_abs, # use absorbed stats df to find absorbed stats
                                    col_name, 
                                    timeframe=timeframe, 
                                    landed_attempts=landed_attempts
                                    )
                        # division by number of minutes cancels out, so accuracy is just landed / attempts
                        new_col_name_accuracy = f'{timeframe}_{inf_abs}_{col_name}_accuracy'
                        stats_to_add_to_main_df.append(new_col_name_accuracy)
                        accuracy = new_columns_dict[f'{timeframe}_{inf_abs}_{col_name}_landed_per_min'] / new_columns_dict[f'{timeframe}_{inf_abs}_{col_name}_attempts_per_min'].replace(0, np.nan) # avoid division by zero
                        accuracy.replace(np.nan, 0, inplace=True)  # replace NaN with 0
                        new_columns_dict[new_col_name_accuracy] = accuracy
                        
            ## SOMETHING BETWEEN HERE AND LINE 990 IS SCREWING UP TAKEDOWNS PER MIN... AND PROBABLY OTHER THINGS TOO
            # STATS COMPUTED FROM THE new_columns_dict
            # add an offensive striking score
            standup_striking_score_stats = [u'sig_strikes', u'total_strikes', u'head_strikes', u'body_strikes', u'leg_strikes', u'distance_strikes', u'clinch_strikes']
            # say a strike landed is 3 times more valuable than a strike attempted
            # say a knockdown is worth 10 times a landed strike
            # add up all inf attempts
            inf_abs = 'inf'  # we only care about inflicted stats for the striking score
            for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                new_col_name = f'{timeframe}_offensive_standing_striking_score'
                stats_to_add_to_main_df.append(new_col_name)
                offensive_standup_striking_score = (new_columns_dict[f'{timeframe}_{inf_abs}_knockdowns_per_min'] * 10).copy() # 10 times more valuable than a attempted strike
                for stat in standup_striking_score_stats:
                    offensive_standup_striking_score += new_columns_dict[f'{timeframe}_{inf_abs}_{stat}_landed_per_min'] * 3  # 3 times more valuable than a strike attempted
                    offensive_standup_striking_score += new_columns_dict[f'{timeframe}_{inf_abs}_{stat}_attempts_per_min']  # add attempts
                # knockout win is worth 3 times more than a knockdown per minute
                offensive_standup_striking_score += new_columns_dict[f'{timeframe}_wins_ko'] * 3  # 3 times more valuable than a knockdown per minute
                new_columns_dict[new_col_name] = offensive_standup_striking_score
                
            # add an defensive striking loss (smaller is better)
            inf_abs = 'abs'  # we only care about absorbed stats for the striking loss
            for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                new_col_name = f'{timeframe}_defensive_standing_striking_loss'
                stats_to_add_to_main_df.append(new_col_name)
                defensive_standup_striking_loss = (new_columns_dict[f'{timeframe}_{inf_abs}_knockdowns_per_min'] * 10).copy() # MAKE SURE TO COPY!!!
                for stat in standup_striking_score_stats:
                    defensive_standup_striking_loss += new_columns_dict[f'{timeframe}_{inf_abs}_{stat}_landed_per_min'] * 3 # 3 times more costly than a strike attempted
                    defensive_standup_striking_loss += new_columns_dict[f'{timeframe}_{inf_abs}_{stat}_attempts_per_min']  # add attempts
                # knockout loss is worth 3 times more than a knockdown per minute
                defensive_standup_striking_loss += new_columns_dict[f'{timeframe}_losses_ko'] * 3  # 3 times more costly than a knockdown per minute
                # add the score to the new columns dict
                new_columns_dict[new_col_name] = defensive_standup_striking_loss
                
            # add an offensive grappling score
            offensive_grappling_score_stats = [u'takedowns', u'sub_attempts', u'reversals', u'control']
            # takedowns and sub attempts and reversals are equally weighted. 30 seconds of control is worth 1 takedown or sub attempt
            inf_abs = 'inf'  # we only care about inflicted stats for the grappling score
            for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                new_col_name = f'{timeframe}_offensive_grappling_score'
                stats_to_add_to_main_df.append(new_col_name)
                offensive_grappling_score = new_columns_dict[f'{timeframe}_{inf_abs}_takedowns_landed_per_min'].copy() # MAKE SURE TO COPY!!!
                offensive_grappling_score += new_columns_dict[f'{timeframe}_{inf_abs}_takedowns_attempts_per_min'] / 5 # 5 takedown attempts are worth 1 takedown landed
                offensive_grappling_score += new_columns_dict[f'{timeframe}_{inf_abs}_sub_attempts_per_min'] 
                offensive_grappling_score += new_columns_dict[f'{timeframe}_{inf_abs}_reversals_per_min']
                offensive_grappling_score += new_columns_dict[f'{timeframe}_{inf_abs}_control_per_min'] / 30  # 30 seconds of control is worth 1 takedown or sub attempt
                # ground strikes are included. Say 5 ground strikes are worth 1 takedown or sub attempt
                offensive_grappling_score += new_columns_dict[f'{timeframe}_{inf_abs}_ground_strikes_landed_per_min'] / 5  # 5 ground strikes are worth 1 takedown or sub attempt
                offensive_grappling_score += new_columns_dict[f'{timeframe}_{inf_abs}_ground_strikes_attempts_per_min'] / 15  # 5 ground strikes are worth 1 takedown or sub attempt

                # add the score to the new columns dict
                # submission win is worth 3 times more than a takedown per minute
                offensive_grappling_score += new_columns_dict[f'{timeframe}_wins_sub'] * 3  # 3 times more valuable than a takedown per minute
                # add the score to the new columns dict
                new_columns_dict[new_col_name] = offensive_grappling_score
                
            # add an defensive grappling loss (smaller is better)
            defensive_grappling_loss_stats = [u'takedowns', u'sub_attempts', u'reversals', u'control']
            # takedowns and sub attempts and reversals are equally weighted. 30 seconds of control is worth 1 takedown or sub attempt
            inf_abs = 'abs'  # we only care about absorbed stats for the grappling
            for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                new_col_name = f'{timeframe}_defensive_grappling_loss'
                stats_to_add_to_main_df.append(new_col_name)
                defensive_grappling_loss = new_columns_dict[f'{timeframe}_{inf_abs}_takedowns_landed_per_min'].copy() # MAKE SURE TO COPY!!!
                defensive_grappling_loss += new_columns_dict[f'{timeframe}_{inf_abs}_takedowns_attempts_per_min'] / 5 # 5 takedown attempts are worth 1 takedown landed
                defensive_grappling_loss += new_columns_dict[f'{timeframe}_{inf_abs}_sub_attempts_per_min'] 
                defensive_grappling_loss += new_columns_dict[f'{timeframe}_{inf_abs}_reversals_per_min']
                defensive_grappling_loss += new_columns_dict[f'{timeframe}_{inf_abs}_control_per_min'] / 30  # 30 seconds of control is worth 1 takedown or sub attempt
                # ground strikes are included. Say 5 ground strikes are worth 1 takedown or sub attempt
                defensive_grappling_loss += new_columns_dict[f'{timeframe}_{inf_abs}_ground_strikes_landed_per_min'] / 5  # 5 ground strikes are worth 1 takedown or sub attempt
                defensive_grappling_loss += new_columns_dict[f'{timeframe}_{inf_abs}_ground_strikes_attempts_per_min'] / 15  # 5 ground strikes are worth 1 takedown or sub attempt
                # submission loss is worth 3 times more than a takedown per minute
                defensive_grappling_loss += new_columns_dict[f'{timeframe}_losses_sub'] * 3  # 3 times more costly than a takedown per minute
                # add the score to the new columns dict
                new_columns_dict[new_col_name] = defensive_grappling_loss
                
                # make overall fighter scores
            for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                new_col_name = f'{timeframe}_overall_fighter_score'
                stats_to_add_to_main_df.append(new_col_name)
                overall_fighter_score = new_columns_dict[f'{timeframe}_offensive_standing_striking_score'] - new_columns_dict[f'{timeframe}_defensive_standing_striking_loss']
                overall_fighter_score += new_columns_dict[f'{timeframe}_offensive_grappling_score'] - new_columns_dict[f'{timeframe}_defensive_grappling_loss']
                # add a bonus for winning fights
                overall_fighter_score += new_columns_dict[f'{timeframe}_wins'] * 2  # each win is worth 2 points
                overall_fighter_score -= new_columns_dict[f'{timeframe}_losses'] * 2  # each loss is worth -2 points
                new_columns_dict[new_col_name] = overall_fighter_score
                                        
            # add all new columns to localized df at once to avoid highly fragmented df warning
            new_columns_df = pd.DataFrame(new_columns_dict, index=localized_df.index)
            localized_df = pd.concat([localized_df, new_columns_df], axis=1)
            # add all new stats to main df at once to avoid highly fragmented df warning
            localized_df = localized_df[stats_to_add_to_main_df].copy()  # keep only the new stats we computed
                        
            ufc_fights_reported_derived_doubled.loc[fighter_mask, stats_to_add_to_main_df] = localized_df[stats_to_add_to_main_df]
        return ufc_fights_reported_derived_doubled

    # UFCStats normally includes a year, but retain a safe fallback for its
    # older month/day-only format and handle December-to-January rollover.
    def convert_scraped_date_to_standard_date(self, input_date) -> str:
        input_date = str(input_date).strip()
        has_year = re.search(r'\b\d{4}\b', input_date) is not None
        value = input_date if has_year else f'{input_date}, {date.today().year}'
        parsed = pd.to_datetime(value, errors='raise')
        if not has_year and parsed.date() < date.today():
            parsed = parsed + pd.DateOffset(years=1)
        return parsed.strftime('%B %d, %Y')

    def _upcoming_card_fights(self, card_link, card_title):
        page = ufcstats_client.get(
            card_link, expected_text='b-fight-details__table'
        )
        soup = BeautifulSoup(page.content, "html.parser")
        fights = soup.find_all(
            "tr",
            {
                "class": (
                    "b-fight-details__table-row "
                    "b-fight-details__table-row__hover js-fight-details-click"
                )
            },
        )
        if not fights:
            return []
        fights_list = []
        for fight in fights:
            entries = [
                entry.get_text().strip()
                for entry in fight.find_all('p')
                if entry.get_text().strip()
            ]
            if len(entries) != 4:
                raise UFCStatsError(
                    f'Expected four fields for an upcoming fight at '
                    f'{card_link}, got {entries!r}'
                )
            fighter, opponent, _, weight_class = entries
            fighter_links = [
                anchor.get('href') for anchor in fight.find_all('a')
                if 'fighter-details' in str(anchor.get('href', ''))
            ]
            if len(fighter_links) != 2:
                raise UFCStatsError(
                    f'Expected two fighter IDs for {fighter} vs {opponent} at '
                    f'{card_link}, got {fighter_links!r}'
                )
            fights_list.append(
                [
                    fighter,
                    opponent,
                    weight_class,
                    fighter_links[0],
                    fighter_links[1],
                    card_link,
                ]
            )
        return fights_list

    def get_upcoming_fight_cards(self):
        """Return every announced UFCStats card that has listed matchups."""

        url = 'http://ufcstats.com/statistics/events/upcoming'
        page = ufcstats_client.get(url, expected_text='b-statistics__table-events')
        soup = BeautifulSoup(page.content, "html.parser")
        mycards = soup.find_all("a", {"class": "b-link b-link_style_black"})
        mydates = soup.find_all("span", {"class":"b-statistics__date"})
        if not mycards or not mydates:
            raise UFCStatsError(
                f'No upcoming UFC card/date was found at {url}; page layout may have changed'
            )
        if len(mycards) != len(mydates):
            raise UFCStatsError(
                'Upcoming UFCStats event titles and dates have different counts'
            )
        cards = []
        for date_element, card_element in zip(mydates, mycards):
            card_date = date_element.get_text().strip()
            card_title = card_element.get_text().strip()
            card_link = str(card_element.attrs.get('href', '')).strip()
            if not card_date or not card_title or not card_link:
                raise UFCStatsError('Upcoming UFCStats event metadata is incomplete')
            fights_list = self._upcoming_card_fights(card_link, card_title)
            if not fights_list:
                print(
                    f'Skipping announced card without listed fights: '
                    f'{card_title} ({card_date})'
                )
                continue
            cards.append(
                (card_date, card_title, fights_list)
            )
        if not cards:
            raise UFCStatsError('No announced UFCStats card contains listed fights')
        cards.sort(
            key=lambda item: pd.to_datetime(
                self.convert_scraped_date_to_standard_date(item[0]),
                errors='raise',
            )
        )
        return cards

    def get_next_fight_card(self):
        return self.get_upcoming_fight_cards()[0]
