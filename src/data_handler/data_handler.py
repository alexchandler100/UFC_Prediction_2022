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
                       fight_math
            )

from odds_getter import OddsGetter

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")

pd.options.mode.chained_assignment = None # default='warn' (disables SettingWithCopyWarning)

class DataHandler:
    def __init__(self):
        # updated scraped fight data (after running ufc_fights_reported_doubled_updated function from UFC_data_scraping file)
        self.csv_filepaths = {
            'fighter_stats': f'{git_root}/src/content/data/processed/fighter_stats.csv',
            'ufc_fights_predictive_flattened_diffs': f'{git_root}/src/content/data/processed/ufc_fights_predictive_flattened_diffs.csv',
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
        
        # card_info = pd.read_json(self.json_filepaths['card_info'])
        # fighter_stats = pd.read_json(self.json_filepaths['fighter_stats'])
        # interesting_stats = pd.read_json(self.json_filepaths['interesting_stats'])
        prediction_history = pd.read_json(self.json_filepaths['prediction_history'])
        # theta = pd.read_json(self.json_filepaths['theta'])
        # intercept = pd.read_json(self.json_filepaths['intercept'])
        # ufc_fight_data_for_website = pd.read_json(self.json_filepaths['ufc_fight_data_for_website'])
        vegas_odds = pd.read_json(self.json_filepaths['vegas_odds'])

        self.json_data = {
            # 'card_info': card_info,
            # 'fighter_stats': fighter_stats,
            # 'interesting_stats': interesting_stats,
            'prediction_history': prediction_history,
            # 'theta': theta,
            # 'intercept': intercept,
            # 'ufc_fight_data_for_website': ufc_fight_data_for_website,
            'vegas_odds': vegas_odds,
        }
        # {key : pd.read_json(self.json_filepaths[key]) for key in self.json_filepaths.keys()}
        
        self.odds_getter = OddsGetter()
        
        # TODO update this with the actual bookies we are getting odds from (or just those I can actually use)
        self.bookies = ['DraftKings', 'BetMGM', 'Caesars', 'BetRivers', 'FanDuel', 'PointsBet', 'Unibet', 'Bet365', 'BetWay', '5D', 'Ref']

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
            # import ipdb;ipdb.set_trace(context=10)  # uncomment to debug
            self.update_ufc_fights_reported_derived_doubled()
            self.update_ufc_fight_data_for_website()
            self.update_pictures()
        else:
            raise ValueError("No update function implemented for this key")

        
    def get_most_recent_fight_date(self, key):
        # find the most recent fight date in the specified key's dataframe
        assert key in ['ufc_fights_reported_doubled', 'ufc_fights_reported_derived_doubled', 'ufc_fights_predictive_flattened_diffs'], "Invalid key provided"
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
            stats = self.get_fight_card(event)
            ufc_fights_reported_doubled_new_rows = pd.concat([stats, ufc_fights_reported_doubled_new_rows], axis=0)
            
        # convert date column to string format YYYY-MM-DD
        ufc_fights_reported_doubled_new_rows['date'] = pd.to_datetime(ufc_fights_reported_doubled_new_rows['date'], errors='coerce')
        ufc_fights_reported_doubled_new_rows['date'] = ufc_fights_reported_doubled_new_rows['date'].dt.strftime('%Y-%m-%d')

        updated_stats = pd.concat([ufc_fights_reported_doubled_new_rows, old_ufc_fights_reported_doubled], axis=0)
        updated_stats = updated_stats.reset_index(drop=True)
        # set ufc_fights_reported_doubled and save it to csv
        self.set('ufc_fights_reported_doubled', updated_stats)
        self.save_csv('ufc_fights_reported_doubled')
    
    def get_fight_card(self, url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        fight_card = pd.DataFrame()
        date = soup.select_one('li.b-list__box-list-item').text.strip().split('\n')[-1].strip()
        date = pd.to_datetime(date, format='%B %d, %Y')
        # can use df.date.dt.to_pydatetime() to get an array of datetime.datetime objects if needed
        rows = soup.select('tr.b-fight-details__table-row')[1:]
        
        print(f'date: {date}, url: {url}, number of fights on card: {len(rows)}')
        
        for row in rows:
            fight_data_dict = {'date': [], 'fight_url': [], 'event_url': [], 'result': [], 'fighter': [], 'opponent': [], 'division': [], 'method': [],
                        'round': [], 'time': [], 'total_fight_time': [], 'fighter_url': [], 'opponent_url': []}
            fight_data_dict['date'] += [date, date]  # add date of fight # TODO consider changing to datetime object
            fight_data_dict['event_url'] += [url, url]  # add event url
            cols = row.select('td')
            
            if not cols:
                print(f'A fight card for {url} is probably in progress so we are skipping this event.')
                return
            
            cols0_a = cols[0].select_one('a')
            
            if not cols0_a:
                print(f'B fight card for {url} is probably in progress so we are skipping this event.')
                return 
            
            # get fight url and results
            fight_url = cols0_a['href']  # get fight url
            fight_data_dict['fight_url'] += [fight_url, fight_url]
            results = cols[0].select('p')
            # pick 0 or 1 randomly (use this to determine the ordering of fighter and opponent) (winner always listed first and we dont want that order all the time)
            fighter = cols[1].select('p')[0].text.strip()
            opponent = cols[1].select('p')[1].text.strip()
            fighter_url = cols[1].select('a')[0]['href']
            opponent_url = cols[1].select('a')[1]['href']
            
            result = ['D', 'D'] if len(results) == 2 else ['W','L']
            fight_data_dict['result'] += result
            
            # get fighter names and fighter urls
            fight_data_dict['fighter'] += [fighter, opponent]
            fight_data_dict['opponent'] += [opponent, fighter]
            fight_data_dict['fighter_url'] += [fighter_url, opponent_url]
            fight_data_dict['opponent_url'] += [opponent_url, fighter_url]
                        
            # get division
            division = cols[6].select_one('p').text.strip()
            fight_data_dict['division'] += [division, division]
            
            # get method
            method = cols[7].select_one('p').text.strip()
            fight_data_dict['method'] += [method, method]
            
            # get round
            rd = cols[8].select_one('p').text.strip()
            rd = int(rd) if rd.isdigit() else np.nan  # sometimes it says 'N/A' for no contest, so convert to 0
            fight_data_dict['round'] += [rd, rd]
            
            # get time
            time = cols[9].select_one('p').text.strip()
            fight_data_dict['time'] += [time, time]
            
            # calculate the number of seconds in the last round
            time_tuple = time.split(':')
            assert len(time_tuple) == 2, f"Time format error: {time} in fight {fight_url}"
            minutes, seconds = time_tuple
            minutes = int(minutes) if minutes.isdigit() else 0
            seconds = int(seconds) if seconds.isdigit() else 0
            total_seconds = minutes * 60 + seconds
            
            total_fight_time = (rd - 1) * 60 * 5 + total_seconds
            fight_data_dict['total_fight_time'] += [total_fight_time, total_fight_time]

            fight_data_dict = pd.DataFrame(fight_data_dict)
            # get striking details
            strike_data_dict = self.get_fight_stats(fight_url, fighter, opponent)
                    
            if strike_data_dict is None:
                continue  # skip this fight if no fight details available
            
            # join to fight details
            fight_data_dict = pd.merge(fight_data_dict, strike_data_dict, on='fighter', how='left', copy=False)
            
            # add fight details to fight card
            fight_card = pd.concat([fight_card, fight_data_dict], axis=0)
            
        fight_card = fight_card.reset_index(drop=True)
        return fight_card

        
    def get_fight_stats(self, url, fighter, opponent):
        r"""
        Makes dataframe with two rows per fight, one for each fighter
        """
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        
        fd_columns = {}
        
        statistics1 = [
            'knockdowns', 
            'sig_strikes_landed', 'sig_strikes_attempts', 
            'total_strikes_landed', 'total_strikes_attempts', 
            'takedowns_landed', 'takedowns_attempts', 
            'sub_attempts', 
            'reversals', 
            'control'
            ]
        
        statistics2 = [
            'head_strikes_landed', 'head_strikes_attempts',
            'body_strikes_landed', 'body_strikes_attempts',
            'leg_strikes_landed', 'leg_strikes_attempts',
            'distance_strikes_landed', 'distance_strikes_attempts',
            'clinch_strikes_landed', 'clinch_strikes_attempts',
            'ground_strikes_landed', 'ground_strikes_attempts'
            ]
        
        statistics = statistics1 + statistics2
        fd_columns['fighter'] = []
        for stat in statistics:
            fd_columns[stat] = []
            
        # gets overall fight details
        fight_data_details = soup.select_one('tbody.b-fight-details__table-body')
        if fight_data_details == None:
            print('missing fight details for:', url)
            return None
        fd_cols = fight_data_details.select('td.b-fight-details__table-col')
        
        def get_attempts_and_landed_from_str(stat_str):
            if ' of ' in stat_str:
                landed, attempts = stat_str.split(' of ')
                landed = int(landed) if landed.isdigit() else np.nan
                attempts = int(attempts) if attempts.isdigit() else np.nan
                return landed, attempts
            else:
                return np.nan, np.nan
            
        def get_seconds_from_minutes_seconds(time_str):
            if ':' in time_str:
                mins, secs = time_str.split(':')
                total_secs = int(mins) * 60 + int(secs)
                return total_secs
            else:
                print('Unexpected time format:', time_str)
                return np.nan
                
        fighter_col_index = 0
        knockdowns_col_index = 1
        sig_strikes_col_index = 2
        total_strikes_col_index = 4
        takedowns_col_index = 5
        sub_attempts_col_index = 7
        reversals_col_index = 8
        control_col_index = 9
        
        fighter_col = fd_cols[fighter_col_index].select('p')
        knockdown_col = fd_cols[knockdowns_col_index].select('p')
        sig_strike_col = fd_cols[sig_strikes_col_index].select('p')
        total_strike_col = fd_cols[total_strikes_col_index].select('p')
        takedown_col = fd_cols[takedowns_col_index].select('p')
        sub_attempt_col = fd_cols[sub_attempts_col_index].select('p')
        reversal_col = fd_cols[reversals_col_index].select('p')
        control_col = fd_cols[control_col_index].select('p')
        
        fighter_index = 0 
        opponent_index = 1
        
        fighter = fighter_col[fighter_index].text.strip()
        opponent = fighter_col[opponent_index].text.strip()
        fd_columns['fighter'] += [fighter, opponent]
            
        fighter_knockdowns = knockdown_col[fighter_index].text.strip() # looks like an integer 
        opponent_knockdowns = knockdown_col[opponent_index].text.strip() # looks like an integer
        fighter_knockdowns = int(fighter_knockdowns) if fighter_knockdowns.isdigit() else np.nan
        opponent_knockdowns = int(opponent_knockdowns) if opponent_knockdowns.isdigit() else np.nan
        fd_columns['knockdowns'] += [fighter_knockdowns, opponent_knockdowns]
        
        fighter_sig_strikes = sig_strike_col[fighter_index].text.strip() # looks like '10 of 20' so we split into attempts and landed
        opponent_sig_strikes = sig_strike_col[opponent_index].text.strip()
        fighter_sig_strikes_landed, fighter_sig_strikes_attempts = get_attempts_and_landed_from_str(fighter_sig_strikes)
        opponent_sig_strikes_landed, opponent_sig_strikes_attempts = get_attempts_and_landed_from_str(opponent_sig_strikes)
        fd_columns['sig_strikes_landed'] += [fighter_sig_strikes_landed, opponent_sig_strikes_landed]
        fd_columns['sig_strikes_attempts'] += [fighter_sig_strikes_attempts, opponent_sig_strikes_attempts]
        
        fighter_total_strikes = total_strike_col[fighter_index].text.strip() # looks like '10 of 20' so we split into attempts and landed
        fighter_total_strikes_landed, fighter_total_strikes_attempts = get_attempts_and_landed_from_str(fighter_total_strikes)
        opponent_total_strikes = total_strike_col[opponent_index].text.strip() # looks like '10 of 20' so we split into attempts and landed
        opponent_total_strikes_landed, opponent_total_strikes_attempts = get_attempts_and_landed_from_str(opponent_total_strikes)
        fd_columns['total_strikes_landed'] += [fighter_total_strikes_landed, opponent_total_strikes_landed]
        fd_columns['total_strikes_attempts'] += [fighter_total_strikes_attempts, opponent_total_strikes_attempts]
        
        fighter_takedowns = takedown_col[fighter_index].text.strip() # looks like '10 of 20' so we split into attempts and landed
        fighter_takedowns_landed, fighter_takedowns_attempts = get_attempts_and_landed_from_str(fighter_takedowns)
        opponent_takedowns = takedown_col[opponent_index].text.strip() # looks like '10 of 20' so we split into attempts and landed
        opponent_takedowns_landed, opponent_takedowns_attempts = get_attempts_and_landed_from_str(opponent_takedowns)
        fd_columns['takedowns_landed'] += [fighter_takedowns_landed, opponent_takedowns_landed]
        fd_columns['takedowns_attempts'] += [fighter_takedowns_attempts, opponent_takedowns_attempts]
        
        fighter_sub_attempts = sub_attempt_col[fighter_index].text.strip() # looks like an integer
        fighter_sub_attempts = int(fighter_sub_attempts) if fighter_sub_attempts.isdigit() else np.nan
        opponent_sub_attempts = sub_attempt_col[opponent_index].text.strip() # looks like an integer
        opponent_sub_attempts = int(opponent_sub_attempts) if opponent_sub_attempts.isdigit() else np.nan
        fd_columns['sub_attempts'] += [fighter_sub_attempts, opponent_sub_attempts]
        
        fighter_reversals = reversal_col[fighter_index].text.strip() # looks like an integer
        fighter_reversals = int(fighter_reversals) if fighter_reversals.isdigit() else np.nan
        opponent_reversals = reversal_col[opponent_index].text.strip() # looks like an integer
        opponent_reversals = int(opponent_reversals) if opponent_reversals.isdigit() else np.nan
        fd_columns['reversals'] += [fighter_reversals, opponent_reversals]
        
        fighter_control = control_col[fighter_index].text.strip() # looks like '1:30' for 1 minute 30 seconds
        fighter_control = get_seconds_from_minutes_seconds(fighter_control)
        opponent_control = control_col[opponent_index].text.strip() # looks like '1:30' for 1 minute 30 seconds
        opponent_control = get_seconds_from_minutes_seconds(opponent_control)
        fd_columns['control'] += [fighter_control, opponent_control]

        # get sig strike details
        sig_strike_details = soup.find('p', class_='b-fight-details__collapse-link_tot', string=re.compile('Significant Strikes')).find_next('tbody', class_='b-fight-details__table-body')
        fd_cols = sig_strike_details.select('td.b-fight-details__table-col')
        
        head_strikes_col_index = 3
        body_strikes_col_index = 4
        leg_strikes_col_index = 5
        distance_strikes_col_index = 6
        clinch_strikes_col_index = 7
        ground_strikes_col_index = 8
        
        head_strikes_col = fd_cols[head_strikes_col_index].select('p')
        body_strikes_col = fd_cols[body_strikes_col_index].select('p')
        leg_strikes_col = fd_cols[leg_strikes_col_index].select('p')
        distance_strikes_col = fd_cols[distance_strikes_col_index].select('p')
        clinch_strikes_col = fd_cols[clinch_strikes_col_index].select('p')
        ground_strikes_col = fd_cols[ground_strikes_col_index].select('p')
        
        fighter_head_strikes = head_strikes_col[fighter_index].text.strip()
        fighter_head_strikes_landed, fighter_head_strikes_attempts = get_attempts_and_landed_from_str(fighter_head_strikes)
        opponent_head_strikes = head_strikes_col[opponent_index].text.strip()
        opponent_head_strikes_landed, opponent_head_strikes_attempts = get_attempts_and_landed_from_str(opponent_head_strikes)
        fd_columns['head_strikes_landed'] += [fighter_head_strikes_landed, opponent_head_strikes_landed]
        fd_columns['head_strikes_attempts'] += [fighter_head_strikes_attempts, opponent_head_strikes_attempts]
        
        fighter_body_strikes = body_strikes_col[fighter_index].text.strip()
        fighter_body_strikes_landed, fighter_body_strikes_attempts = get_attempts_and_landed_from_str(fighter_body_strikes)
        opponent_body_strikes = body_strikes_col[opponent_index].text.strip()
        opponent_body_strikes_landed, opponent_body_strikes_attempts = get_attempts_and_landed_from_str(opponent_body_strikes)
        fd_columns['body_strikes_landed'] += [fighter_body_strikes_landed, opponent_body_strikes_landed]
        fd_columns['body_strikes_attempts'] += [fighter_body_strikes_attempts, opponent_body_strikes_attempts]
        
        fighter_leg_strikes = leg_strikes_col[fighter_index].text.strip()
        fighter_leg_strikes_landed, fighter_leg_strikes_attempts = get_attempts_and_landed_from_str(fighter_leg_strikes)
        opponent_leg_strikes = leg_strikes_col[opponent_index].text.strip()
        opponent_leg_strikes_landed, opponent_leg_strikes_attempts = get_attempts_and_landed_from_str(opponent_leg_strikes)
        fd_columns['leg_strikes_landed'] += [fighter_leg_strikes_landed, opponent_leg_strikes_landed]
        fd_columns['leg_strikes_attempts'] += [fighter_leg_strikes_attempts, opponent_leg_strikes_attempts]
        
        fighter_distance_strikes = distance_strikes_col[fighter_index].text.strip()
        fighter_distance_strikes_landed, fighter_distance_strikes_attempts = get_attempts_and_landed_from_str(fighter_distance_strikes)
        opponent_distance_strikes = distance_strikes_col[opponent_index].text.strip()
        opponent_distance_strikes_landed, opponent_distance_strikes_attempts = get_attempts_and_landed_from_str(opponent_distance_strikes)
        fd_columns['distance_strikes_landed'] += [fighter_distance_strikes_landed, opponent_distance_strikes_landed]
        fd_columns['distance_strikes_attempts'] += [fighter_distance_strikes_attempts, opponent_distance_strikes_attempts]
        
        fighter_clinch_strikes = clinch_strikes_col[fighter_index].text.strip()
        fighter_clinch_strikes_landed, fighter_clinch_strikes_attempts = get_attempts_and_landed_from_str(fighter_clinch_strikes)
        opponent_clinch_strikes = clinch_strikes_col[opponent_index].text.strip()
        opponent_clinch_strikes_landed, opponent_clinch_strikes_attempts = get_attempts_and_landed_from_str(opponent_clinch_strikes)
        fd_columns['clinch_strikes_landed'] += [fighter_clinch_strikes_landed, opponent_clinch_strikes_landed]
        fd_columns['clinch_strikes_attempts'] += [fighter_clinch_strikes_attempts, opponent_clinch_strikes_attempts]
        
        fighter_ground_strikes = ground_strikes_col[fighter_index].text.strip()
        fighter_ground_strikes_landed, fighter_ground_strikes_attempts = get_attempts_and_landed_from_str(fighter_ground_strikes)
        opponent_ground_strikes = ground_strikes_col[opponent_index].text.strip()
        opponent_ground_strikes_landed, opponent_ground_strikes_attempts = get_attempts_and_landed_from_str(opponent_ground_strikes)
        fd_columns['ground_strikes_landed'] += [fighter_ground_strikes_landed, opponent_ground_strikes_landed]
        fd_columns['ground_strikes_attempts'] += [fighter_ground_strikes_attempts, opponent_ground_strikes_attempts]
        
        fight_stat_df = pd.DataFrame(fd_columns)
        
        return fight_stat_df
        
    # updates fighter attributes with new fighters not yet saved yet
    def update_fighter_stats(self):
        # TODO find a way to avoid using the old version of fighter_stats.csv ... this is clunky
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
                        
        
    def clean_ufc_fights_for_winner_prediction(self, ufc_fights_predictive_flattened_diffs):
        #importing csv fight data and saving as dataframes
        ufc_fights_winner = ufc_fights_predictive_flattened_diffs.copy()
        #cleaning the methods column for winner prediction
        #changing anything other than 'U-DEC','M-DEC', 'KO/TKO', 'SUB', to 'bullshit'
        #changing 'U-DEC','M-DEC', to 'DEC'
        ufc_fights_winner['method'] = clean_method_for_winner_predictions(ufc_fights_winner['method'])
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
        
    
    def clean_ufc_fights_for_method_prediction(self, ufc_fights_predictive_flattened_diffs):
        ufc_fights_method = ufc_fights_predictive_flattened_diffs.copy()
        #cleaning the methods column for method prediction
        #changing anything other than 'U-DEC','M-DEC', 'S-DEC', 'KO/TKO', 'SUB', to 'bullshit'
        #changing 'U-DEC','M-DEC', 'S-DEC', to 'DEC'
        ufc_fights_method['method'] = clean_method_for_method_predictions(ufc_fights_method['method'])
        #fights with outcome "Win" or "Loss" (no "Draw")
        draw_mask=ufc_fights_method['result'] != 'D'
        #fights where the method of victory is TKO/SUB/DEC (no split decision or DQ or Overturned or anything else like that)
        method_mask_method=(ufc_fights_method['method']!='bullshit')
        #fights where age is known
        age_mask=(ufc_fights_method['fighter_age']!='unknown')&(ufc_fights_method['opponent_age']!='unknown')&(ufc_fights_method['fighter_age']!=0)&(ufc_fights_method['opponent_age']!=0)
        #fights where height/reach is known
        height_mask=(ufc_fights_method['fighter_height']!='unknown')&(ufc_fights_method['opponent_height']!='unknown')
        reach_mask=(ufc_fights_method['fighter_reach']!='unknown')&(ufc_fights_method['opponent_reach']!='unknown')
        #fights where number of wins is known
        wins_mask=(ufc_fights_method['fighter_wins'] != 'unknown' )&(ufc_fights_method['opponent_wins'] != 'unknown')
        #fights where both fighters have strike statistics (gets rid of UFC debuts)
        strikes_mask=(ufc_fights_method['fighter_inf_sig_strikes_attempts_avg'] != 0)&(ufc_fights_method['opponent_inf_sig_strikes_attempts_avg'] != 0)
        #includes only the fights satisfying these conditions
        ufc_fights_method=ufc_fights_method[draw_mask&method_mask_method&age_mask&height_mask&reach_mask&wins_mask&strikes_mask]
        # ufc_fights_predictive_flattened_diffs=ufc_fights_predictive_flattened_diffs[draw_mask&method_mask_winner&age_mask&height_mask&reach_mask&wins_mask&strikes_mask]

        #listing all stats and making some new stats from them (differences often score higher in the learning models)
        record_statistics=[u'fighter_wins',u'fighter_losses',u'fighter_L5Y_wins',u'fighter_L5Y_losses',u'fighter_L2Y_wins',u'fighter_L2Y_losses',u'fighter_ko_wins',u'fighter_ko_losses',u'fighter_L5Y_ko_wins',u'fighter_L5Y_ko_losses',u'fighter_L2Y_ko_wins',u'fighter_L2Y_ko_losses',u'fighter_sub_wins',u'fighter_sub_losses',u'fighter_L5Y_sub_wins',u'fighter_L5Y_sub_losses',u'fighter_L2Y_sub_wins',u'fighter_L2Y_sub_losses',u'opponent_wins',u'opponent_losses',u'opponent_L5Y_wins',u'opponent_L5Y_losses',u'opponent_L2Y_wins',u'opponent_L2Y_losses',u'opponent_ko_wins',u'opponent_ko_losses',u'opponent_L5Y_ko_wins',u'opponent_L5Y_ko_losses',u'opponent_L2Y_ko_wins',u'opponent_L2Y_ko_losses',u'opponent_sub_wins',u'opponent_sub_losses',u'opponent_L5Y_sub_wins',u'opponent_L5Y_sub_losses',u'opponent_L2Y_sub_wins',u'opponent_L2Y_sub_losses']
        physical_stats=[ u'fighter_age',u'fighter_height',u'fighter_reach',u'opponent_age',u'opponent_height',u'opponent_reach']
        #THERE MAY BE A PROBLEM IN AGE HEIGHT REACH TO DO WITH STRING VS FLOAT. MAKE SURE THESE ARE ALL THE CORRECT TYPE
        #MAYBE WE ARE LOSING PREDICTABILITY HERE (but we apply float later so may it is ok)
        #here is the list of all stats available (besides stance), does not include names or result
        punch_statistics=[  u'fighter_inf_knockdowns_avg',u'fighter_inf_pass_avg',u'fighter_inf_reversals_avg',u'fighter_inf_sub_attempts_avg',u'fighter_inf_takedowns_landed_avg',u'fighter_inf_takedowns_attempts_avg',u'fighter_inf_sig_strikes_landed_avg',u'fighter_inf_sig_strikes_attempts_avg',u'fighter_inf_total_strikes_landed_avg',u'fighter_inf_total_strikes_attempts_avg',u'fighter_inf_head_strikes_landed_avg',u'fighter_inf_head_strikes_attempts_avg',u'fighter_inf_body_strikes_landed_avg',u'fighter_inf_body_strikes_attempts_avg',u'fighter_inf_leg_strikes_landed_avg',u'fighter_inf_leg_strikes_attempts_avg',u'fighter_inf_distance_strikes_landed_avg',u'fighter_inf_distance_strikes_attempts_avg',u'fighter_inf_clinch_strikes_landed_avg',u'fighter_inf_clinch_strikes_attempts_avg',u'fighter_inf_ground_strikes_landed_avg',u'fighter_inf_ground_strikes_attempts_avg',u'fighter_abs_knockdowns_avg',u'fighter_abs_pass_avg',u'fighter_abs_reversals_avg',u'fighter_abs_sub_attempts_avg',u'fighter_abs_takedowns_landed_avg',u'fighter_abs_takedowns_attempts_avg',u'fighter_abs_sig_strikes_landed_avg',u'fighter_abs_sig_strikes_attempts_avg',u'fighter_abs_total_strikes_landed_avg',u'fighter_abs_total_strikes_attempts_avg',u'fighter_abs_head_strikes_landed_avg',u'fighter_abs_head_strikes_attempts_avg',u'fighter_abs_body_strikes_landed_avg',u'fighter_abs_body_strikes_attempts_avg',u'fighter_abs_leg_strikes_landed_avg',u'fighter_abs_leg_strikes_attempts_avg',u'fighter_abs_distance_strikes_landed_avg',u'fighter_abs_distance_strikes_attempts_avg',u'fighter_abs_clinch_strikes_landed_avg',u'fighter_abs_clinch_strikes_attempts_avg',u'fighter_abs_ground_strikes_landed_avg',u'fighter_abs_ground_strikes_attempts_avg',u'opponent_inf_knockdowns_avg',u'opponent_inf_pass_avg',u'opponent_inf_reversals_avg',u'opponent_inf_sub_attempts_avg',u'opponent_inf_takedowns_landed_avg',u'opponent_inf_takedowns_attempts_avg',u'opponent_inf_sig_strikes_landed_avg',u'opponent_inf_sig_strikes_attempts_avg',u'opponent_inf_total_strikes_landed_avg',u'opponent_inf_total_strikes_attempts_avg',u'opponent_inf_head_strikes_landed_avg',u'opponent_inf_head_strikes_attempts_avg',u'opponent_inf_body_strikes_landed_avg',u'opponent_inf_body_strikes_attempts_avg',u'opponent_inf_leg_strikes_landed_avg',u'opponent_inf_leg_strikes_attempts_avg',u'opponent_inf_distance_strikes_landed_avg',u'opponent_inf_distance_strikes_attempts_avg',u'opponent_inf_clinch_strikes_landed_avg',u'opponent_inf_clinch_strikes_attempts_avg',u'opponent_inf_ground_strikes_landed_avg',u'opponent_inf_ground_strikes_attempts_avg',u'opponent_abs_knockdowns_avg',u'opponent_abs_pass_avg',u'opponent_abs_reversals_avg',u'opponent_abs_sub_attempts_avg',u'opponent_abs_takedowns_landed_avg',u'opponent_abs_takedowns_attempts_avg',u'opponent_abs_sig_strikes_landed_avg',u'opponent_abs_sig_strikes_attempts_avg',u'opponent_abs_total_strikes_landed_avg',u'opponent_abs_total_strikes_attempts_avg',u'opponent_abs_head_strikes_landed_avg',u'opponent_abs_head_strikes_attempts_avg',u'opponent_abs_body_strikes_landed_avg',u'opponent_abs_body_strikes_attempts_avg',u'opponent_abs_leg_strikes_landed_avg',u'opponent_abs_leg_strikes_attempts_avg',u'opponent_abs_distance_strikes_landed_avg',u'opponent_abs_distance_strikes_attempts_avg',u'opponent_abs_clinch_strikes_landed_avg',u'opponent_abs_clinch_strikes_attempts_avg',u'opponent_abs_ground_strikes_landed_avg',u'opponent_abs_ground_strikes_attempts_avg']
        #adding record differences to ufc_fights_predictive_flattened_diffs
        record_statistics_diff = []
        half_length=int(len(record_statistics)/2)
        for i in range(half_length):
            ufc_fights_method[record_statistics[i]+'_diff_2']=ufc_fights_method[record_statistics[i]]-ufc_fights_method[record_statistics[i+half_length]]
            record_statistics_diff.append(record_statistics[i]+'_diff_2')
        #lets try and improve the greedy algorithm by considering differences. Lets start by replacing height and reach by their differences
        ufc_fights_method['height_diff']=ufc_fights_method['fighter_height'].apply(float)-ufc_fights_method['opponent_height'].apply(float)
        ufc_fights_method['reach_diff']=ufc_fights_method['fighter_reach'].apply(float)-ufc_fights_method['opponent_reach'].apply(float)

        physical_stats_diff = ['fighter_age_diff', 'height_diff', 'reach_diff']

        #adding punch differences to ufc_fights_predictive_flattened_diffs
        punch_statistics_diff = []
        half_length=int(len(punch_statistics)/2)
        for i in range(half_length):
            ufc_fights_method[punch_statistics[i]+'_diff_2']=ufc_fights_method[punch_statistics[i]]-ufc_fights_method[punch_statistics[i+half_length]]
            punch_statistics_diff.append(punch_statistics[i]+'_diff_2')

        possible_stats = record_statistics_diff + physical_stats_diff + punch_statistics_diff

        #setting
        ufc_fights_method['fighter_age'] = ufc_fights_method['fighter_age'].apply(float)
        ufc_fights_method['opponent_age'] = ufc_fights_method['opponent_age'].apply(float)
        ufc_fights_method['fighter_age_diff'] = ufc_fights_method['fighter_age']-ufc_fights_method['opponent_age']
    
    def update_ufc_fights_reported_derived_doubled(self):
        # most recent fight in ufc_fights_reported_doubled_updated versus most recent fight in ufc_fights_reported_derived_doubled
        most_recent_date_in_updated_ufc_fights_reported_doubled = self.get_most_recent_fight_date('ufc_fights_reported_doubled')
        most_recent_date_in_old_ufc_fights_reported_derived_doubled = self.get_most_recent_fight_date('ufc_fights_reported_derived_doubled')
        most_recent_date_in_updated_ufc_fights_reported_doubled = pd.to_datetime(most_recent_date_in_updated_ufc_fights_reported_doubled)
        most_recent_date_in_old_ufc_fights_reported_derived_doubled = pd.to_datetime(most_recent_date_in_old_ufc_fights_reported_derived_doubled)
        
        update_time = (most_recent_date_in_updated_ufc_fights_reported_doubled - most_recent_date_in_old_ufc_fights_reported_derived_doubled).days
        print('days since last update: '+str(update_time))

        # this gives the new rows in ufc_fights_reported_doubled_updated which do not appear in ufc_fights_reported_derived_doubled
        ufc_fights_reported_doubled_updated = self.get('ufc_fights_reported_doubled')
        # date format is "2025-07-12"
        ufc_fights_reported_doubled_updated['date'] = pd.to_datetime(ufc_fights_reported_doubled_updated['date'], format='%Y-%m-%d')
        # get only the fights in ufc_fights_reported_doubled_updated whose dates are more recent than the most recent date in ufc_fights_reported_derived_doubled
        new_rows = ufc_fights_reported_doubled_updated[ufc_fights_reported_doubled_updated['date'] > most_recent_date_in_old_ufc_fights_reported_derived_doubled]
        # import ipdb;ipdb.set_trace(context=10)
        # TODO SOMETHING IS GOING VERY WRONG WITH UPDATING HERE. IT MAKES OUR PREDICTION ACCURACY 100% SO THE RESULT IS GETTING MIXED IN SOMEHOW
        if update_time > 0: # should just stop the script here. Can do this later once we do everything inside a function call
            ufc_fights_reported_derived_doubled = self.populate_new_fights_with_statistics(new_rows)
            # save the results to a csv file 
            ufc_fights_reported_derived_doubled_path = f'{git_root}/src/content/data/processed/ufc_fights_reported_derived_doubled.csv'
            ufc_fights_reported_derived_doubled.to_csv(ufc_fights_reported_derived_doubled_path, index=False)
            # set the new dataframe in the data manager
            ufc_fights_reported_derived_doubled.reset_index(inplace=True)
            self.set('ufc_fights_reported_derived_doubled', ufc_fights_reported_derived_doubled)
            print(f'Saved predictive fight history with stats to {ufc_fights_reported_derived_doubled_path}, shape {ufc_fights_reported_derived_doubled.shape}')
        else:
            print('nothing to update')
            
            
    def make_ufc_fights_predictive_flattened_diffs(self, derived_doubled_df):

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
        
        # import ipdb; ipdb.set_trace(context=10)  # uncomment to debug
        
        # cannot ipdb before oddsgetter makes the selenium request
        # import ipdb; ipdb.set_trace(context=10)  # uncomment to debug
        
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
                    fighter_kelly, opponent_kelly = get_kelly_bet_from_ev_and_dk_odds(int(fighter_predicted_odds), int(bookie_fighter_odds), int(bookie_opponent_odds))
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
                    
            # dk_fighter_vegas_odds = predictions_df.at[i, f'fighter DraftKings']
            # dk_opponent_vegas_odds = predictions_df.at[i, f'opponent DraftKings']
            # if not fighter_predicted_odds or not dk_fighter_vegas_odds or not dk_opponent_vegas_odds:
            #     continue  # skip if any of these values are missing
            # fighter_predicted_odds = int(fighter_predicted_odds)
            # dk_fighter_vegas_odds = int(dk_fighter_vegas_odds)
            # dk_opponent_vegas_odds = int(dk_opponent_vegas_odds)
            # fighter_bankroll_percentage, opponent_bankroll_percentage = get_kelly_bet_from_ev_and_dk_odds(fighter_predicted_odds, dk_fighter_vegas_odds, dk_opponent_vegas_odds)
            # predictions_df.at[i, 'fighter bet bankroll percentage'] = fighter_bankroll_percentage
            # predictions_df.at[i, 'opponent bet bankroll percentage'] = opponent_bankroll_percentage

            
        # save to vegas_oddsjson
        # odds_df= self.drop_irrelevant_fights(odds_df,3) #allows 3 bookies to have missing odds. can increase this to 2 or 3 as needed
        # odds_df = self.drop_non_ufc_fights(odds_df)
        #odds_df=drop_repeats(odds_df)
        
        # NOTE COMMENTED THIS OUT BECAUSE WE ARE OVERWRITING IT JUST AFTER WE CALL THIS FUNCTION. SHOULD BE FINE...
        # print('saving odds to content/data/external/vegas_odds.json')
        # result = odds_df.to_json()
        # parsed = json.loads(result)
        # jsonFilePath='content/data/external/vegas_odds.json'
        # with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        #    jsonf.write(json.dumps(parsed, indent=4))
        # print('saved to '+jsonFilePath)
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
            if row1['predicted fighter odds'] == '':
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
            relevant_fights = ufc_fights_reported_doubled[pd.to_datetime(ufc_fights_reported_doubled.index) == card_date]
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
        fighter_stats = self.get('fighter_stats')
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
        
        ufc_fights_reported_derived_doubled = self.fill_in_statistics_for_fights(ufc_fights_reported_derived_doubled, ufc_fights_reported_doubled, names_to_update, fighter_stats)
                
        # grab just the most recent two rows
        last_two_rows = ufc_fights_reported_derived_doubled.iloc[-len(new_rows_derived):]
        ufc_fights_reported_derived_doubled = last_two_rows
        return ufc_fights_reported_derived_doubled.reset_index()
    
                
    def populate_new_fights_with_statistics(self, new_rows):
        fighter_stats = self.get('fighter_stats')
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
        
        ufc_fights_reported_derived_doubled = self.fill_in_statistics_for_fights(ufc_fights_reported_derived_doubled, ufc_fights_reported_doubled, names_to_update, fighter_stats)
                
        return ufc_fights_reported_derived_doubled
    
    ########### FUNCTIONS USED IN update_data_csvs_and_jsons.py ###########
    
    def fill_in_statistics_for_fights(self, ufc_fights_reported_derived_doubled, ufc_fights_reported_doubled, names_to_update, fighter_stats):
        # GOAL reproduce these statistics 
        
        ufc_fights_reported_derived_doubled = ufc_fights_reported_derived_doubled.copy()
        ufc_fights_reported_doubled = ufc_fights_reported_doubled.copy()

        # TODO MAKE THIS GRABBABLE WITH A FUNCTION CALL
        physical_stats = [u'age', u'height', u'reach', u'stance']
        # the rest will have total, l2y and l5y versions
        record_stats = [u'wins', u'losses', u'wins_ko', u'wins_sub', u'wins_dec', u'losses_ko', u'losses_sub' u'losses_dec']
        # the following will also have inflicted (inf) and absorbed (abs) versions
        grappling_event_stats = [u'reversals', u'control', u'sub_attempts']
        striking_event_stats = [u'knockdowns']
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
            
            # import ipdb; ipdb.set_trace(context=10)  # uncomment to debug
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
                # import ipdb; ipdb.set_trace(context=10)  # uncomment to debug
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
                    

            for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                new_col_name = f'{timeframe}_wins_wins'
                stats_to_add_to_main_df.append(new_col_name)
                wins_wins_extended = count_wins_wins_before_fight(fighter_2deg_of_sep_wins_df, name, timeframe=timeframe)
                # get the sub series that has the fighter as the fighter (not opponent)
                wins_wins = wins_wins_extended[same_name_vect(fighter_2deg_of_sep_wins_df['fighter'], name)]
                new_columns_dict[new_col_name] = wins_wins
                
                new_col_name = f'{timeframe}_losses_losses'
                stats_to_add_to_main_df.append(new_col_name)
                losses_losses_extended = count_losses_losses_before_fight(fighter_2deg_of_sep_loss_df, name, timeframe=timeframe)
                # get the sub series that has the fighter as the fighter (not opponent)
                losses_losses = losses_losses_extended[same_name_vect(fighter_2deg_of_sep_loss_df['fighter'], name)]
                new_columns_dict[new_col_name] = losses_losses
                
                new_col_name = f'{timeframe}_fight_math'
                stats_to_add_to_main_df.append(new_col_name)
                fight_math_extended = fight_math(name, fighter_2deg_of_sep_wins_df, timeframe)
                fight_math_col = fight_math_extended[fighter_2deg_of_sep_wins_df]
                new_columns_dict[new_col_name] = fight_math_col
        
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
                # import ipdb; ipdb.set_trace(context=10)  # uncomment to debug
                offensive_standup_striking_score = new_columns_dict[f'{timeframe}_{inf_abs}_knockdowns_per_min'] * 10 # 10 times more valuable than a attempted strike
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
                defensive_standup_striking_loss = new_columns_dict[f'{timeframe}_{inf_abs}_knockdowns_per_min'] * 10
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
                offensive_grappling_score = new_columns_dict[f'{timeframe}_{inf_abs}_takedowns_landed_per_min'] 
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
                defensive_grappling_loss = new_columns_dict[f'{timeframe}_{inf_abs}_takedowns_landed_per_min'] 
                defensive_grappling_loss = new_columns_dict[f'{timeframe}_{inf_abs}_takedowns_landed_per_min'] / 5 # 5 takedown attempts are worth 1 takedown landed
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
        
            # TODO make fight_math stats for all timeframes
                                
            # add all new columns to localized df at once to avoid highly fragmented df warning
            new_columns_df = pd.DataFrame(new_columns_dict, index=localized_df.index)
            localized_df = pd.concat([localized_df, new_columns_df], axis=1)
            # add all new stats to main df at once to avoid highly fragmented df warning
            localized_df = localized_df[stats_to_add_to_main_df].copy()  # keep only the new stats we computed
                        
            # for stat in stats_to_add_to_main_df:
            #     ufc_fights_reported_derived_doubled.loc[fighter_mask, stat] = localized_df[stat]
            # THIS IS ABSOLUTE BULLSHIT THAT THIS WORKS BUT THE FOLLOWING DOES NOT BECAUSE IT CREATES A COPY
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


    # thresh is the number of bookies we allow to not have odds on the books