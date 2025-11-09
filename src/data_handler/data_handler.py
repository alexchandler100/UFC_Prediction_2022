import git
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
            )

# replace downcasting behavior deprecated
pd.set_option('future.no_silent_downcasting', True)

from odds_getter import OddsGetter

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")

pd.options.mode.chained_assignment = None # default='warn' (disables SettingWithCopyWarning)

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
        self.csv_data[key].to_csv(self.csv_filepaths[key], index=False)
            
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
        
        with open(self.json_filepaths['theta'], 'w') as outfile:
            json.dump(theta, outfile)

        with open(self.json_filepaths['intercept'], 'w') as outfile:
            json.dump(b, outfile)
        
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
            self.update_pictures()
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
        url = 'http://ufcstats.com/statistics/events/completed?page=all'
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        events_table = soup.select_one('tbody')
        ufc_fights_reported_doubled_new_rows = pd.DataFrame()
        
        events = [event['href'] for event in events_table.select( 'a')[1:]] # omit first event (future event) # TODO WE MAY AS WELL USE THIS TO POPULATE THE FUTURE EVENT INSTEAD OF GETTING IT FROM ANOTHER WEBSITE LATER...
        saved_events = set(old_ufc_fights_reported_doubled.event_url.unique())
        new_events = [event for event in events if event not in saved_events]  # get only new events
        if not new_events:
            print('No new events to scrape for ufc_fights_reported_doubled')
            return
        for event in new_events: # skip events that are already in the old_ufc_fights_reported_doubled
            print(event)
            stats = get_fight_card(event)
            ufc_fights_reported_doubled_new_rows = pd.concat([stats, ufc_fights_reported_doubled_new_rows], axis=0)
            
        # convert date column to string format YYYY-MM-DD
        ufc_fights_reported_doubled_new_rows['date'] = pd.to_datetime(ufc_fights_reported_doubled_new_rows['date'], errors='coerce')
        ufc_fights_reported_doubled_new_rows['date'] = ufc_fights_reported_doubled_new_rows['date'].dt.strftime('%Y-%m-%d')

        updated_stats = pd.concat([ufc_fights_reported_doubled_new_rows, old_ufc_fights_reported_doubled], axis=0)
        updated_stats = updated_stats.reset_index(drop=True)
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
            page = requests.get(f_url)
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
        #fights with outcome "Win" or "Loss" (no "Draw")
        draw_mask=ufc_fights_winner['result'] != 'D'
        #fights where the method of victory is TKO/SUB/DEC (no split decision or DQ or Overturned or anything else like that)
        method_mask_winner=(ufc_fights_winner['method']!='bullshit')
        # drop any rows with nan in any column
        ufc_fights_winner=ufc_fights_winner[draw_mask & method_mask_winner]
        ufc_fights_winner = ufc_fights_winner.dropna(axis=0, how='any')
        ufc_fights_winner['result'] = (ufc_fights_winner['result'] == 'W').values.astype(int)
        
        return ufc_fights_winner
    
    
    def update_ufc_fights_reported_derived_doubled(self):
        # most recent fight in ufc_fights_reported_doubled_updated versus most recent fight in ufc_fights_reported_derived_doubled
        most_recent_date_in_updated_ufc_fights_reported_doubled = self.get_most_recent_fight_date('ufc_fights_reported_doubled')
        most_recent_date_in_old_ufc_fights_reported_derived_doubled = self.get_most_recent_fight_date('ufc_fights_reported_derived_doubled')
        most_recent_date_in_updated_ufc_fights_reported_doubled = pd.to_datetime(most_recent_date_in_updated_ufc_fights_reported_doubled)
        most_recent_date_in_old_ufc_fights_reported_derived_doubled = pd.to_datetime(most_recent_date_in_old_ufc_fights_reported_derived_doubled)
        
        self.update_time = (most_recent_date_in_updated_ufc_fights_reported_doubled - most_recent_date_in_old_ufc_fights_reported_derived_doubled).days
        print('days since last update: '+str(self.update_time))

        # this gives the new rows in ufc_fights_reported_doubled_updated which do not appear in ufc_fights_reported_derived_doubled
        ufc_fights_reported_doubled_updated = self.get('ufc_fights_reported_doubled')
        # date format is "2025-07-12"
        ufc_fights_reported_doubled_updated['date'] = pd.to_datetime(ufc_fights_reported_doubled_updated['date'], format='%Y-%m-%d')
        # get only the fights in ufc_fights_reported_doubled_updated whose dates are more recent than the most recent date in ufc_fights_reported_derived_doubled
        new_rows = ufc_fights_reported_doubled_updated[ufc_fights_reported_doubled_updated['date'] > most_recent_date_in_old_ufc_fights_reported_derived_doubled]
        if self.update_time > 0: # should just stop the script here. Can do this later once we do everything inside a function call
            ufc_fights_reported_derived_doubled = self.populate_new_fights_with_statistics(new_rows)
            # save the results to a csv file 
            ufc_fights_reported_derived_doubled_path = f'{git_root}/src/content/data/processed/ufc_fights_reported_derived_doubled.csv'
            ufc_fights_reported_derived_doubled.to_csv(ufc_fights_reported_derived_doubled_path, index=True)
            ufc_fights_reported_derived_doubled.reset_index(inplace=True, drop=False)
            # set the new dataframe in the data manager
            self.set('ufc_fights_reported_derived_doubled', ufc_fights_reported_derived_doubled)
            print(f'Saved ufc_fights_reported_derived_doubled to {ufc_fights_reported_derived_doubled_path}, shape {ufc_fights_reported_derived_doubled.shape}')
        else:
            print('nothing to update')
            
    def make_ufc_fights_predictive_flattened(self, derived_doubled_df, shuffle=True):

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
        for i in range(0, len(ufc_fights_reported_derived_doubled), 2):
            pair = ufc_fights_reported_derived_doubled.iloc[i:i+2]
            if shuffle:
                pair = pair.sample(frac=1).reset_index(drop=True)  # shuffle within the pair
            shuffled_rows.append(pair)
        # Concatenate back into a single DataFrame
        ufc_fights_reported_derived_doubled = pd.concat(shuffled_rows).reset_index(drop=True)

        # drop non-predictive columns
        ufc_fights_predictive = ufc_fights_reported_derived_doubled[predictive_columns]

        # grab fighter and opponent columns for diffing in flattened dataframe
        fighter_col = ufc_fights_reported_derived_doubled['fighter'].loc[::2]
        opponent_col = ufc_fights_reported_derived_doubled['opponent'].loc[::2]
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
        ufc_fights_predictive_flattened_dict['fighter'] = fighter_col
        ufc_fights_predictive_flattened_dict['opponent'] = opponent_col
        ufc_fights_predictive_flattened_dict['result'] = result_col
        ufc_fights_predictive_flattened_dict['method'] = method_col
        ufc_fights_predictive_flattened_dict['division'] = division_col

        for col in predictive_columns:
            if col not in ['fighter', 'opponent', 'result', 'method', 'division']:
                ufc_fights_predictive_flattened_dict[f'fighter_{col}'] = ufc_fights_predictive_even[col].values
                ufc_fights_predictive_flattened_dict[f'opponent_{col}'] = ufc_fights_predictive_odd[col].values
                
        ufc_fights_predictive_flattened = pd.DataFrame(ufc_fights_predictive_flattened_dict)
        return ufc_fights_predictive_flattened
            
            
    def make_ufc_fights_predictive_flattened_diffs(self, derived_doubled_df, shuffle=True):

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
        for i in range(0, len(ufc_fights_reported_derived_doubled), 2):
            pair = ufc_fights_reported_derived_doubled.iloc[i:i+2]
            if shuffle:
                pair = pair.sample(frac=1).reset_index(drop=True)  # shuffle within the pair
            shuffled_rows.append(pair)
        # Concatenate back into a single DataFrame
        ufc_fights_reported_derived_doubled = pd.concat(shuffled_rows).reset_index(drop=True)

        # drop non-predictive columns
        ufc_fights_predictive = ufc_fights_reported_derived_doubled[predictive_columns]

        # grab fighter and opponent columns for diffing in flattened dataframe
        fighter_col = ufc_fights_reported_derived_doubled['fighter'].loc[::2]
        opponent_col = ufc_fights_reported_derived_doubled['opponent'].loc[::2]
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
        ufc_fights_predictive_diffs_dict['fighter'] = fighter_col
        ufc_fights_predictive_diffs_dict['opponent'] = opponent_col
        ufc_fights_predictive_diffs_dict['result'] = result_col
        ufc_fights_predictive_diffs_dict['method'] = method_col
        ufc_fights_predictive_diffs_dict['division'] = division_col

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
        ufc_fight_data_for_website.to_csv('content/data/processed/ufc_fight_data_for_website.csv', index=False)

        # convert ufc_fights_reported_derived_doubled.csv to json files to read via javascript in website
        csvFilePath = r'content/data/processed/ufc_fight_data_for_website.csv'
        jsonFilePath = r'content/data/external/ufc_fight_data_for_website.json'
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
        odds_df = self.odds_getter.make_odds_df()
        odds_df['fighter bet bankroll percentage'] = np.nan
        odds_df['opponent bet bankroll percentage'] = np.nan
        odds_df['best fighter bookie'] = ''
        odds_df['best opponent bookie'] = ''
        
        # TODO go through and figure out if any parlays have better expectation values
        # than the individual bets (2 leg and 3 leg parlays only probably worth it)
        
        # merge into predictions_df 
        # TODO IF WE COME TO TRUST THIS fightodds.io website we can use this as our source of upcoming fights instead of ufcstats.com and avoid the merge 
        for i in range(len(predictions_df)):
            fighter = predictions_df['fighter name'][i]
            opponent = predictions_df['opponent name'][i]
            # find row in odds_df where fighter and opponent match (could be in either order in the other df)
            odds_row = odds_df[same_name_vect(odds_df['fighter name'], fighter) & same_name_vect(odds_df['opponent name'], opponent)]
            fighter_a = 'fighter'
            fighter_b = 'opponent'
            if odds_row.empty:
                opponent = predictions_df['fighter name'][i]
                fighter = predictions_df['opponent name'][i]
                fighter_a = 'opponent'
                fighter_b = 'fighter'
                odds_row = odds_df[same_name_vect(odds_df['fighter name'], fighter) & same_name_vect(odds_df['opponent name'], opponent)]
            if odds_row.empty:
                print(f'No odds found for {fighter} vs {opponent} on fightodds.io, skipping...')
                continue
            # TODO update these with the actual bookies we are getting odds from (or just those I can actually use)
            for bookie in self.bookies:
                if f'fighter {bookie}' in odds_row.columns:
                    predictions_df.at[i, f'fighter {bookie}'] = odds_row[f'{fighter_a} {bookie}'].values[0]
                    predictions_df.at[i, f'opponent {bookie}'] = odds_row[f'{fighter_b} {bookie}'].values[0]
            # add average odds for fighter and opponent
            predictions_df.at[i, f'average bookie odds'] = odds_row['average bookie odds'].values[0]
            
            # add expected values for fighter and opponent
            fighter_predicted_odds = predictions_df.at[i, 'predicted fighter odds']
            if not fighter_predicted_odds:
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
                if bookie_fighter_odds and bookie_opponent_odds:
                    fighter_bookie_odds_dict[bookie] = int(bookie_fighter_odds)
                    opponent_bookie_odds_dict[bookie] = int(bookie_opponent_odds)
                    predicted_fighter_odds = int(fighter_predicted_odds)
                    fighter_kelly, opponent_kelly = get_kelly_bet_from_ev_and_dk_odds(predicted_fighter_odds, int(bookie_fighter_odds), int(bookie_opponent_odds))
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

        return predictions_df
    
    # TODO vegas_odds is really not the right name for this data as it contains predictions, not just vegas odds
    def update_vegas_odds(self, vegas_odds):
        #save to json
        result = vegas_odds.to_json()
        parsed = json.loads(result)
        jsonFilePath='content/data/external/vegas_odds.json'
        with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(parsed, indent=4))
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
        with open(prediction_history_filtpath, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(parsed, indent=4))
            
        print(f'saved to {prediction_history_filtpath}')
        
    def update_card_info(self):
        card_date, card_title, fights_list = self.get_next_fight_card()
        card_date = self.convert_scraped_date_to_standard_date(card_date)

        card_info_dict = {"date":card_date, "title":card_title}

        print('Writing upcoming card info to content/data/external/card_info.json')
        with open('content/data/external/card_info.json', 'w') as outfile:
            json.dump(card_info_dict, outfile)            
        
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
        with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(data, indent=4))
            
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
        # getting rid of fights that didn't actually happen and adding correctness results of those that did
        bad_indices = []
        vegas_odds_old['fighter bet'] = 0.0
        vegas_odds_old['opponent bet'] = 0.0
        vegas_odds_old['current bankroll after'] = 0.0
        vegas_odds_old['bet result'] = 'N/A'
        for index1, row1 in vegas_odds_old.iloc[::-1].iterrows(): # iterate backwards in the order the fights actually happened
            card_date = row1['date']
            
            # if no prediction was made, throw it away
            if not row1['predicted fighter odds']:
                bad_indices.append(index1)
                print('no prediction made for fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'])
                continue
            
            fighter_odds = int(row1['predicted fighter odds'])
            best_fighter_bookie = row1['best fighter bookie']
            best_opponent_bookie = row1['best opponent bookie']
            
            # check if we ever found odds for the fighter and opponent
            best_fighter_bookie_odds = row1.get(f'fighter {best_fighter_bookie}')
            if not best_fighter_bookie_odds:
                print(f'No odds found for fighter {row1["fighter name"]} from bookie {best_fighter_bookie}, skipping...')
            best_opponent_bookie_odds = row1.get(f'opponent {best_opponent_bookie}')
            if not best_opponent_bookie_odds:
                print(f'No odds found for opponent {row1["opponent name"]} from bookie {best_opponent_bookie}, skipping...')
            
            if best_fighter_bookie_odds:
                best_fighter_bookie_odds = int(row1.get(f'fighter {best_fighter_bookie}'))
            if best_opponent_bookie_odds:
                best_opponent_bookie_odds = int(row1.get(f'opponent {best_opponent_bookie}'))
                
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
            
            # if a prediction was made, check if the fight actually happened and then check if the prediction / bet was correct / won
            # TODO this is slow but sort of necessary if we need to add multiple cards at the same time
            # import ipdb;ipdb.set_trace(context=10)
            relevant_fights = ufc_fights_reported_doubled[pd.to_datetime(ufc_fights_reported_doubled['date']) == card_date]
            print(f'searching through {relevant_fights.shape[0]//2} confirmed fights on {str(card_date).split(" ")[0]} for {row1["fighter name"]} vs {row1["opponent name"]}')
            
            match_found = False
            for index2, row2 in relevant_fights.iterrows():
                if same_name(row1['fighter name'], row2['fighter']) and same_name(row1['opponent name'], row2['opponent']):
                    match_found = True
                    print('adding fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'])
                    if abs(int(fighter_odds)) == 100:
                        vegas_odds_old.at[index1,'correct?'] = 'N/A' # model did not predict a winner, called it dead even
                    elif (int(fighter_odds) < 0 and row2['result'] == 'W') or (int(fighter_odds) > 0 and row2['result'] == 'L'):
                        vegas_odds_old.at[index1,'correct?'] = 1
                    else:
                        vegas_odds_old.at[index1,'correct?'] = 0
                    # update the bankroll based on the bet made
                    fighter_bet = 0.0
                    opponent_bet = 0.0
                    fighter_payout = 0.0
                    opponent_payout = 0.0
                    bet_result = 'N/A'
                    if fighter_bankroll_percentage > 0: # check if we even made a bet on the fighter
                        fighter_bet = fighter_bankroll_percentage / 100 * currentBankroll
                        vegas_odds_old.at[index1, 'fighter bet'] = fighter_bet
                        bet_result = row2['result']
                        fighter_payout = bet_payout(best_fighter_bookie_odds, fighter_bet, bet_result)
                    if opponent_bankroll_percentage > 0: # check if we even made a bet on the opponent
                        opponent_bet = opponent_bankroll_percentage / 100 * currentBankroll
                        vegas_odds_old.at[index1, 'opponent bet'] = opponent_bet
                        # win the bet if the opponent wins (the result column is the result of the fighter, so if the fighter wins, the opponent loses)
                        bet_result = 'L' if row2['result'] == 'W' else 'W'
                        opponent_payout = bet_payout(best_opponent_bookie_odds, opponent_bet, bet_result)
                    currentBankroll = currentBankroll + fighter_payout + opponent_payout - fighter_bet - opponent_bet
                    # TODO why is this set to dtype int?
                    vegas_odds_old.at[index1, 'current bankroll after'] = currentBankroll
                    vegas_odds_old.at[index1, 'bet result'] = bet_result
                    # TODO add case for draw
                    break
            if not match_found: # if the fight didn't happen, throw it away
                bad_indices.append(index1)
                print('fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'] + ' not found in ufc_fights_reported_derived_doubled.csv')
        vegas_odds_old = vegas_odds_old.drop(bad_indices)
        return vegas_odds_old
    
    def make_derived_doubled_vector_for_fight(self, new_rows):
        ufc_fights_reported_doubled = self.get('ufc_fights_reported_doubled')
        ufc_fights_reported_doubled['date'] = pd.to_datetime(ufc_fights_reported_doubled['date'], errors='coerce')
        ufc_fights_reported_doubled = ufc_fights_reported_doubled.loc[::-1]
        ufc_fights_reported_doubled.set_index('date', inplace=True)

        ufc_fights_reported_derived_doubled = self.get('ufc_fights_reported_derived_doubled')
        # import ipdb; ipdb.set_trace(context=10)  # uncomment to debug
        ufc_fights_reported_derived_doubled['date'] = pd.to_datetime(ufc_fights_reported_derived_doubled['date'], errors='coerce')
        ufc_fights_reported_derived_doubled.set_index('date', inplace=True)
        
        # add new rows to bottom of derived dataframe in reverse order
        new_rows_derived = new_rows[['fighter', 'opponent', 'date', 'result', 'method', 'division']].copy()
        new_rows_derived['date'] = pd.to_datetime(new_rows_derived['date'], errors='coerce')
        new_rows_derived.set_index('date', inplace=True)
        ufc_fights_reported_derived_doubled = pd.concat([ufc_fights_reported_derived_doubled, new_rows_derived[::-1]], axis=0)
        
        # add new rows to bottom of doubled reported dataframe in reverse order
        ufc_fights_reported_doubled = pd.concat([ufc_fights_reported_doubled, new_rows_derived[::-1]], axis=0)
        # replace all nan values with zeros in just the rows we added (otherwise rolling averages will all be nan since we subtract off the current row to not include current fight)
        ufc_fights_reported_doubled.iloc[-len(new_rows_derived):] = ufc_fights_reported_doubled.iloc[-len(new_rows_derived):].replace(np.nan, 0.0)
        
        names_to_update = new_rows.fighter
        
        ufc_fights_reported_derived_doubled = self.fill_in_statistics_for_fights(ufc_fights_reported_derived_doubled, ufc_fights_reported_doubled, names_to_update)
                
        # grab just the most recent two rows
        last_two_rows = ufc_fights_reported_derived_doubled.iloc[-len(new_rows_derived):]
        ufc_fights_reported_derived_doubled = last_two_rows[::-1] # reverse back to original order
        return ufc_fights_reported_derived_doubled.reset_index()
    
                
    def populate_new_fights_with_statistics(self, new_rows):
        ufc_fights_reported_doubled = self.get('ufc_fights_reported_doubled')
        ufc_fights_reported_doubled['date'] = pd.to_datetime(ufc_fights_reported_doubled['date'], errors='coerce')
        ufc_fights_reported_doubled = ufc_fights_reported_doubled.loc[::-1]
        ufc_fights_reported_doubled.set_index('date', inplace=True)

        ufc_fights_reported_derived_doubled = self.get('ufc_fights_reported_derived_doubled')
        ufc_fights_reported_derived_doubled['date'] = pd.to_datetime(ufc_fights_reported_derived_doubled['date'], errors='coerce')
        ufc_fights_reported_derived_doubled.set_index('date', inplace=True)
        
        # add new rows to bottom of derived dataframe in reverse order
        new_rows_derived = new_rows[['fighter', 'opponent', 'date', 'result', 'method', 'division']].copy()
        new_rows_derived['date'] = pd.to_datetime(new_rows_derived['date'], errors='coerce')
        new_rows_derived.set_index('date', inplace=True)
        ufc_fights_reported_derived_doubled = pd.concat([ufc_fights_reported_derived_doubled, new_rows_derived[::-1]], axis=0)
        
        names_to_update = new_rows.fighter
        
        ufc_fights_reported_derived_doubled = self.fill_in_statistics_for_fights(ufc_fights_reported_derived_doubled, ufc_fights_reported_doubled, names_to_update)
                
        return ufc_fights_reported_derived_doubled
    
    ########### FUNCTIONS USED IN update_data_csvs_and_jsons.py ###########
    
    def fill_in_statistics_for_fights(self, ufc_fights_reported_derived_doubled, ufc_fights_reported_doubled, names_to_update=None):
        fighter_stats = self.get('fighter_stats')
        if names_to_update is None:
            names_to_update = list(fighter_stats['name'].unique())
        # GOAL reproduce these statistics 
        
        ufc_fights_reported_derived_doubled = ufc_fights_reported_derived_doubled.copy()
        ufc_fights_reported_doubled = ufc_fights_reported_doubled.copy()

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

    # input looks like 'July 15th'. Need to add year to it
    def convert_scraped_date_to_standard_date(self, input_date) -> str:
        year = date.today().strftime('%B %d, %Y')[-4:]
        month = input_date.split(' ')[0]
        day = input_date.split(' ')[1]
        i = 0
        while day[i].isdigit():
            i+=1
        day = day[:i]
        return f'{month} {day}, {year}'

    def get_next_fight_card(self):
        url = 'http://ufcstats.com/statistics/events/upcoming'
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser") 
        mycards = soup.find_all("a", {"class": "b-link b-link_style_black"})
        mydates = soup.find_all("span", {"class":"b-statistics__date"})
        date = mydates[0]
        card = mycards[0] 
        card_date = date.get_text().strip()
        card_title = card.get_text().strip()
        card_link = card.attrs['href']
        page = requests.get(card_link)
        soup = BeautifulSoup(page.content, "html.parser")
        fights = soup.find_all("tr",{"class": "b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click"})
        fights_list = []
        for fight in fights:
            fighter, opponent, _, weight_class = [entry.get_text().strip() for entry in fight.find_all('p') if entry.get_text().strip()!= '']
            fights_list.append([fighter,opponent,weight_class])
        return card_date, card_title, fights_list