import os 
import pandas as pd
from bs4 import BeautifulSoup
import urllib.request
import requests
import csv
import json
from datetime import date
import re
import numpy as np
from pathlib import Path
import tempfile

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
                       get_event_fight_urls,
            )

# replace downcasting behavior deprecated
pd.set_option('future.no_silent_downcasting', True)

from odds_getter import OddsGetter
from ufcstats_client import UFCStatsError, UFCStatsEventNotComplete, ufcstats_client

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

class DataHandler:
    def __init__(self):
        # updated scraped fight data (after running ufc_fights_reported_doubled_updated function from UFC_data_scraping file)
        self.csv_filepaths = {
            'fighter_stats': f'{git_root}/src/content/data/processed/fighter_stats.csv',
            'ufc_fights_reported_derived_doubled': f'{git_root}/src/content/data/processed/ufc_fights_reported_derived_doubled.csv',
            'ufc_fights_reported_doubled': f'{git_root}/src/content/data/processed/ufc_fights_reported_doubled.csv',
            'ufc_fight_data_for_website': f'{git_root}/src/content/data/processed/ufc_fight_data_for_website.csv', # not really needed...
        }

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
        self.csv_data = {key : pd.read_csv(self.csv_filepaths[key], sep=',') for key in self.csv_filepaths.keys()}
        prediction_history = pd.read_json(self.json_filepaths['prediction_history'])
        vegas_odds = pd.read_json(self.json_filepaths['vegas_odds'])

        self.json_data = {
            'prediction_history': prediction_history,
            'vegas_odds': vegas_odds,
        }
        
        self.odds_getter = OddsGetter()
        
        self.bookies = ['DraftKings', 'BetMGM', 'Caesars', 'BetRivers', 'FanDuel', 'PointsBet', 'Unibet', 'Bet365', 'BetWay', '5D', 'Ref','BetOnline','MyBookie']

    def get(self, key, filetype='csv'):
        if filetype == 'json':
            assert key in list(self.json_data.keys()), "Invalid key provided"
            return self.json_data[key].copy()
        assert key in list(self.csv_data.keys()), "Invalid key provided"
        df = self.csv_data[key].copy()
        return df
    
    def set(self, key, value):
        assert key in list(self.csv_data.keys()), "Invalid key provided"
        self.csv_data[key] = value
    
    def save_csv(self, key):
        assert key in list(self.csv_filepaths.keys()), "Invalid key provided"
        atomic_to_csv(self.csv_data[key], self.csv_filepaths[key], index=False)
            
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
        assert key in list(self.csv_data.keys()) + ['all'], "Invalid key provided"
        if key == 'ufc_fights_reported_doubled':
            self.update_ufc_fights_reported_doubled()
        elif key == 'fighter_stats':
            self.update_fighter_stats()
        elif key == 'ufc_fights_reported_derived_doubled':
            self.update_ufc_fights_reported_derived_doubled()
        elif key == 'prediction_history':
            self.update_prediction_history()
        elif key == 'all':
            self.update_ufc_fights_reported_doubled()
            self.update_fighter_stats()
            self.update_ufc_fights_reported_derived_doubled()
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

        # Cheaply reconcile the manifests for recent stored events.  This
        # repairs late-added or formerly partial bouts without downloading the
        # full historical site every week.
        recent_saved_events = [
            event for event in events
            if event['href'] in saved_event_hrefs
            and "road to ufc" not in event.text.strip().lower()
        ][:12]
        events_to_refresh = []
        for event in recent_saved_events:
            source_fights = set(get_event_fight_urls(event['href']))
            stored_fights = set(
                old_ufc_fights_reported_doubled.loc[
                    old_ufc_fights_reported_doubled['event_url'] == event['href'],
                    'fight_url',
                ]
            )
            if source_fights != stored_fights:
                print(
                    f'Reconciling changed event manifest {event.text.strip()}: '
                    f'{len(stored_fights)} stored vs {len(source_fights)} source fights'
                )
                events_to_refresh.append(event)

        events_to_scrape = new_events + events_to_refresh
        if not events_to_scrape:
            if normalized_existing_results:
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
            print('No new events to scrape for ufc_fights_reported_doubled')
            return
        
        refreshed_event_hrefs = set()
        for event in events_to_scrape:
            name = event.text.strip()
            href = event['href']
            if "road to ufc" in name.lower():
                continue  # skip Road to UFC events
            try:
                stats = get_fight_card(href)
            except UFCStatsEventNotComplete as error:
                print(f'Deferring incomplete same-day event {name}: {error}')
                continue
            refreshed_event_hrefs.add(href)
            ufc_fights_reported_doubled_new_rows = pd.concat([stats, ufc_fights_reported_doubled_new_rows], axis=0)
            
        # convert date column to string format YYYY-MM-DD
        if ufc_fights_reported_doubled_new_rows.empty:
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
        self.set('ufc_fights_reported_doubled', updated_stats)
        self.save_csv('ufc_fights_reported_doubled')
    
        
    # updates fighter attributes with new fighters not yet saved yet
    def update_fighter_stats(self):
        ufc_fights_reported_doubled = self.get('ufc_fights_reported_doubled')
        fighter_stats = self.get('fighter_stats')
        fighter_stats_urls = fighter_stats.url.unique()
        ufc_fights_reported_doubled_urls = ufc_fights_reported_doubled.fighter_url.unique()
        
        fighter_details = {'name': [], 'height': [],
                        'reach': [], 'stance': [], 'dob': [], 'url': []}
        known_fighter_urls = set(fighter_stats_urls)

        for f_url in ufc_fights_reported_doubled_urls:
            if f_url in known_fighter_urls:
                continue # if we already have the fighter in our stats, skip it
            
            print('adding new fighter:', f_url)
            page = ufcstats_client.get(f_url, expected_text='b-list__info-box')
            soup = BeautifulSoup(page.content, "html.parser")

            fighter_name = soup.find(
                'span', class_='b-content__title-highlight').text.strip()
            fighter_details['name'].append(fighter_name)

            fighter_details['url'].append(f_url)

            fighter_attr = soup.find(
                'div', class_='b-list__info-box b-list__info-box_style_small-width js-guide').select('li')
            for i in range(len(fighter_attr)):
                attr = fighter_attr[i].text.split(':')[-1].strip()
                if i == 0:
                    fighter_details['height'].append(attr)
                elif i == 1:
                    pass  # weight is always just whatever weightclass they were fighting at
                elif i == 2:
                    fighter_details['reach'].append(attr)
                elif i == 3:
                    fighter_details['stance'].append(attr)
                else:
                    fighter_details['dob'].append(attr)
        new_fighters = pd.DataFrame(fighter_details)
        updated_fighters = pd.concat([new_fighters, fighter_stats])
        updated_fighters = updated_fighters.reset_index(drop=True)
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
        print('getting bookie odds from fightodds.io')
        predictions_df['odds source status'] = 'unmatched'
        try:
            odds_df = self.odds_getter.make_odds_df()
        except Exception as error:
            # Odds are enrichment, not the source of fight statistics or model
            # predictions.  Preserve a truthful no-odds card when this
            # independent site or Chrome is unavailable instead of aborting
            # the entire weekly publication.
            predictions_df['odds source status'] = 'unavailable'
            print(
                'WARNING: fightodds.io odds are unavailable; publishing '
                f'predictions without book lines ({type(error).__name__}: {error})'
            )
            return predictions_df
        odds_df['fighter bet bankroll percentage'] = np.nan
        odds_df['opponent bet bankroll percentage'] = np.nan
        odds_df['best fighter bookie'] = ''
        odds_df['best opponent bookie'] = ''
        
        # TODO go through and figure out if any parlays have better expectation values
        # than the individual bets (2 leg and 3 leg parlays only probably worth it)
        
        # merge into predictions_df 
        # TODO IF WE COME TO TRUST THIS fightodds.io website we can use this as our source of upcoming fights instead of ufcstats.com and avoid the merge 
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
            if odds_row.empty:
                print(f'No odds found for {fighter} vs {opponent} on fightodds.io, skipping...')
                continue
            if len(odds_row) != 1:
                raise ValueError(
                    f'Ambiguous fightodds.io matchup for {fighter} vs {opponent}: '
                    f'{len(odds_row)} rows'
                )
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
            if fighter_a == 'opponent' and isinstance(consensus_odds, (list, tuple)):
                consensus_odds = list(reversed(consensus_odds))
            predictions_df.at[i, f'average bookie odds'] = consensus_odds
            
            # add expected values for fighter and opponent
            fighter_predicted_odds = predictions_df.at[i, 'predicted fighter odds']
            if pd.isna(fighter_predicted_odds) or fighter_predicted_odds == '':
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

        print(f'Matched fightodds.io lines for {matched_fights}/{len(predictions_df)} UFCStats fights')
        if matched_fights < len(predictions_df):
            print('FightOdds matchups available for comparison:')
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

        card_info_dict = {"date":card_date, "title":card_title}

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
        for index1, row1 in vegas_odds_old.iloc[::-1].iterrows(): # iterate backwards in the order the fights actually happened
            card_date = row1['date']
            
            prediction_value = row1.get('predicted fighter odds')
            if pd.isna(prediction_value) or str(prediction_value).strip() == '':
                vegas_odds_old.at[index1, 'correct?'] = 'N/A'
                vegas_odds_old.at[index1, 'forecast status'] = 'no_prediction'
                vegas_odds_old.at[index1, 'current bankroll after'] = currentBankroll
                print('no prediction made for fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'])
                continue
            
            fighter_odds = self.odds_getter.parse_american_odds(prediction_value)
            if fighter_odds is None:
                raise ValueError(
                    f'Invalid predicted fighter odds {prediction_value!r} for '
                    f'{row1["fighter name"]} vs {row1["opponent name"]}'
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
            
            match_found = False
            for index2, row2 in relevant_fights.iterrows():
                if same_name(row1['fighter name'], row2['fighter']) and same_name(row1['opponent name'], row2['opponent']):
                    match_found = True
                    print('adding fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'])
                    actual_result = row2['result']
                    if actual_result in ['D', 'NC']:
                        vegas_odds_old.at[index1, 'correct?'] = 'N/A'
                        result_status = {'D': 'draw', 'NC': 'no_contest'}[actual_result]
                        vegas_odds_old.at[index1, 'forecast status'] = result_status
                    elif abs(int(fighter_odds)) == 100:
                        vegas_odds_old.at[index1,'correct?'] = 'N/A' # model did not predict a winner, called it dead even
                        vegas_odds_old.at[index1, 'forecast status'] = 'completed'
                    elif (int(fighter_odds) < 0 and actual_result == 'W') or (int(fighter_odds) > 0 and actual_result == 'L'):
                        vegas_odds_old.at[index1,'correct?'] = 1
                        vegas_odds_old.at[index1, 'forecast status'] = 'completed'
                    else:
                        vegas_odds_old.at[index1,'correct?'] = 0
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

    def get_next_fight_card(self):
        url = 'http://ufcstats.com/statistics/events/upcoming'
        page = ufcstats_client.get(url, expected_text='b-statistics__table-events')
        soup = BeautifulSoup(page.content, "html.parser") 
        mycards = soup.find_all("a", {"class": "b-link b-link_style_black"})
        mydates = soup.find_all("span", {"class":"b-statistics__date"})
        if not mycards or not mydates:
            raise UFCStatsError(
                f'No upcoming UFC card/date was found at {url}; page layout may have changed'
            )
        date = mydates[0]
        card = mycards[0] 
        card_date = date.get_text().strip()
        card_title = card.get_text().strip()
        card_link = card.attrs['href']
        page = ufcstats_client.get(card_link, expected_text='b-fight-details__table')
        soup = BeautifulSoup(page.content, "html.parser")
        fights = soup.find_all("tr",{"class": "b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click"})
        if not fights:
            raise UFCStatsError(
                f'Upcoming card {card_title!r} contained no parseable fights at {card_link}'
            )
        fights_list = []
        for fight in fights:
            entries = [entry.get_text().strip() for entry in fight.find_all('p') if entry.get_text().strip()]
            if len(entries) != 4:
                raise UFCStatsError(
                    f'Expected four fields for an upcoming fight at {card_link}, got {entries!r}'
                )
            fighter, opponent, _, weight_class = entries
            fights_list.append([fighter,opponent,weight_class])
        return card_date, card_title, fights_list
