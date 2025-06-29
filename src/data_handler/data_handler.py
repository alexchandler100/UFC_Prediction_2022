import git
import os 
import pandas as pd
from bs4 import BeautifulSoup
import urllib.request
import requests
import csv
import json
from datetime import datetime
from datetime import date
import random
import re
import numpy as np

# local imports
from fight_stat_helpers import (in_ufc, 
                       same_name, 
                       same_name_vect,
                       wins_before_vect, 
                       losses_before_vect, 
                       fighter_age_vect, 
                       fighter_height, 
                       L5Y_wins_vect, 
                       L5Y_losses_vect, 
                       L2Y_wins_vect, 
                       L2Y_losses_vect, 
                       ko_wins_vect, 
                       ko_losses_vect, 
                       L5Y_ko_wins_vect, 
                       L5Y_ko_losses_vect, 
                       L2Y_ko_wins_vect, 
                       L2Y_ko_losses_vect, 
                       sub_wins_vect, 
                       sub_losses_vect, 
                       L5Y_sub_wins_vect, 
                       L5Y_sub_losses_vect, 
                       L2Y_sub_wins_vect, 
                       L2Y_sub_losses_vect, 
                       avg_count_vect, 
                       zero_vect, 
                       opponent_column,
                       stance_vect,
                       time_diff,
                       time_diff_vect,
                       fight_math_diff_vect,
                       fighter_score_diff_vect,
                       get_kelly_bet_from_ev_and_dk_odds,
                       bet_payout
            )

from odds_getter import OddsGetter

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")

pd.options.mode.chained_assignment = None # default='warn' (disables SettingWithCopyWarning)

class DataHandler:
    def __init__(self):
        # updated scraped fight data (after running fight_hist_updated function from UFC_data_scraping file)
        self.csv_filepaths = {
            'fight_hist': f'{git_root}/src/content/data/processed/fight_hist.csv',
            'fighter_stats': f'{git_root}/src/content/data/processed/fighter_stats.csv',
            'ufc_fight_data_for_website': f'{git_root}/src/content/data/processed/ufc_fight_data_for_website.csv',
            'ufc_fights_crap': f'{git_root}/src/content/data/processed/ufc_fights_crap.csv',
            'ufc_fights': f'{git_root}/src/content/data/processed/ufc_fights.csv',
        }
        # NOTES ON fight_hist.csv:
        # - Columns (8,73) have mixed types.
        
        # NOTES ON ufc_fights.csv:
        # - column 15 (reversals) is clearly wrong. It contains a time like 1:23 and if you go back it used to be an integer (number of reversals)
        # Columns (15,37,38,102,103) have mixed types.
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

    def get(self, key, filetype='csv'):
        if filetype == 'json':
            assert key in list(self.json_data.keys()), "Invalid key provided"
            return self.json_data[key].copy()
        assert key in list(self.csv_data.keys()), "Invalid key provided"
        return self.csv_data[key].copy()
    
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
        if key == 'fight_hist':
            self.update_fight_hist()
        elif key == 'fighter_stats':
            self.update_fighter_stats()
            pass
        elif key == 'ufc_fights_crap':
            self.update_ufc_fights_crap()
            pass
        elif key == 'ufc_fights':
            self.update_ufc_fights()
        elif key == 'prediction_history':
            self.update_prediction_history()
        elif key == 'all':
            self.update_fight_hist()
            self.update_fighter_stats()
            self.update_ufc_fights_crap()
            self.update_ufc_fights()
            self.update_ufc_fight_data_for_website()
            self.update_pictures()
        else:
            raise ValueError("No update function implemented for this key")

        
    def get_most_recent_fight_date(self, key):
        # find the most recent fight date in the specified key's dataframe
        assert key in ['fight_hist', 'ufc_fights_crap', 'ufc_fights'], "Invalid key provided"
        return self.csv_data[key]['date'][0]
                
    def update_fight_hist(self):  # takes dataframe of fight stats as input
        old_fight_hist = self.get('fight_hist')
        url = 'http://ufcstats.com/statistics/events/completed?page=all'
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        events_table = soup.select_one('tbody')
        fight_hist_new_rows = pd.DataFrame()
        
        events = [event['href'] for event in events_table.select( 'a')[1:]] # omit first event (future event) # TODO WE MAY AS WELL USE THIS TO POPULATE THE FUTURE EVENT INSTEAD OF GETTING IT FROM ANOTHER WEBSITE LATER...
        saved_events = set(old_fight_hist.event_url.unique())
        new_events = [event for event in events if event not in saved_events]  # get only new events
        for event in new_events: # skip events that are already in the old_fight_hist
            print(event)
            stats = self.get_fight_card(event)
            fight_hist_new_rows = pd.concat([fight_hist_new_rows, stats], axis=0)

        updated_stats = pd.concat([fight_hist_new_rows, old_fight_hist], axis=0)
        updated_stats = updated_stats.reset_index(drop=True)
        # set fight_hist and save it to csv
        self.set('fight_hist', updated_stats)
        self.save_csv('fight_hist')
    

    def get_fight_card(self, url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        fight_card = pd.DataFrame()
        date = soup.select_one(
            'li.b-list__box-list-item').text.strip().split('\n')[-1].strip()
        rows = soup.select('tr.b-fight-details__table-row')[1:]
        for row in rows:
            fight_det = {'date': [], 'fight_url': [], 'event_url': [], 'result': [], 'fighter': [], 'opponent': [], 'division': [], 'method': [],
                        'round': [], 'time': [], 'fighter_url': [], 'opponent_url': []}
            fight_det['date'] += [date, date]  # add date of fight
            fight_det['event_url'] += [url, url]  # add event url
            cols = row.select('td')
            for i in range(len(cols)):
                if i in set([2, 3, 4, 5]):  # skip sub, td, pass, strikes
                    pass
                elif i == 0:  # get fight url and results
                    fight_url = cols[i].select_one('a')['href']  # get fight url
                    fight_det['fight_url'] += [fight_url, fight_url]

                    results = cols[i].select('p')
                    if len(results) == 2:  # was a draw, table shows two draws
                        fight_det['result'] += ['D', 'D']
                    else:  # first fighter won, second lost
                        fight_det['result'] += ['W', 'L']

                elif i == 1:  # get fighter names and fighter urls
                    fighter_1 = cols[i].select('p')[0].text.strip()
                    fighter_2 = cols[i].select('p')[1].text.strip()

                    fighter_1_url = cols[i].select('a')[0]['href']
                    fighter_2_url = cols[i].select('a')[1]['href']

                    fight_det['fighter'] += [fighter_1, fighter_2]
                    fight_det['opponent'] += [fighter_2, fighter_1]

                    fight_det['fighter_url'] += [fighter_1_url, fighter_2_url]
                    fight_det['opponent_url'] += [fighter_2_url, fighter_1_url]
                elif i == 6:  # get division
                    division = cols[i].select_one('p').text.strip()
                    fight_det['division'] += [division, division]
                elif i == 7:  # get method
                    method = cols[i].select_one('p').text.strip()
                    fight_det['method'] += [method, method]
                elif i == 8:  # get round
                    rd = cols[i].select_one('p').text.strip()
                    fight_det['round'] += [rd, rd]
                elif i == 9:  # get time
                    time = cols[i].select_one('p').text.strip()
                    fight_det['time'] += [time, time]

            fight_det = pd.DataFrame(fight_det)
            # get striking details
            str_det = self.get_fight_stats(fight_url)
            if str_det is None:
                pass
            else:
                # join to fight details
                fight_det = pd.merge(fight_det, str_det,
                                    on='fighter', how='left', copy=False)
                # add fight details to fight card
                fight_card = pd.concat([fight_card, fight_det], axis=0)
        fight_card = fight_card.reset_index(drop=True)
        return fight_card
    
        
    # there is a problem for collecting reversals (fix needed) seems like it now collect riding time since sept 2020
    # function for getting individual fight stats
    def get_fight_stats(self, url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        fd_columns = {'fighter': [], 'knockdowns': [], 'sig_strikes': [], 'total_strikes': [], 'takedowns': [], 'sub_attempts': [], 'pass': [],
                    'reversals': []}

        # gets overall fight details
        fight_details = soup.select_one('tbody.b-fight-details__table-body')
        if fight_details == None:
            print('missing fight details for:', url)
            return None
        else:
            fd_cols = fight_details.select('td.b-fight-details__table-col')
            for i in range(len(fd_cols)):
                # skip 3 and 6: strike % and takedown %, will calculate these later
                if i == 3 or i == 6:
                    pass
                else:
                    col = fd_cols[i].select('p')
                    for row in col:
                        data = row.text.strip()
                        if i == 0:  # add to fighter
                            fd_columns['fighter'].append(data)
                        elif i == 1:  # add to sig strikes
                            fd_columns['knockdowns'].append(data)
                        elif i == 2:  # add to total strikes
                            fd_columns['sig_strikes'].append(data)
                        elif i == 4:  # add to total strikes
                            fd_columns['total_strikes'].append(data)
                        elif i == 5:  # add to takedowns
                            fd_columns['takedowns'].append(data)
                        elif i == 7:  # add to sub attempts
                            fd_columns['sub_attempts'].append(data)
                        elif i == 8:  # add to passes
                            fd_columns['pass'].append(data)
                        elif i == 9:  # add to reversals
                            fd_columns['reversals'].append(data)
            ov_details = pd.DataFrame(fd_columns)

            # get sig strike details
            sig_strike_details = soup.find('p', class_='b-fight-details__collapse-link_tot', text=re.compile(
                'Significant Strikes')).find_next('tbody', class_='b-fight-details__table-body')
            sig_columns = {'fighter': [], 'head_strikes': [], 'body_strikes': [], 'leg_strikes': [], 'distance_strikes': [],
                        'clinch_strikes': [], 'ground_strikes': []}
            fd_cols = sig_strike_details.select('td.b-fight-details__table-col')
            for i in range(len(fd_cols)):
                # skip 1, 2 (sig strikes, sig %)
                if i == 1 or i == 2:
                    pass
                else:
                    col = fd_cols[i].select('p')
                    for row in col:
                        data = row.text.strip()
                        if i == 0:  # add to fighter
                            sig_columns['fighter'].append(data)
                        elif i == 3:  # add to head strikes
                            sig_columns['head_strikes'].append(data)
                        elif i == 4:  # add to body strikes
                            sig_columns['body_strikes'].append(data)
                        elif i == 5:  # add to leg strikes
                            sig_columns['leg_strikes'].append(data)
                        elif i == 6:  # add to distance strikes
                            sig_columns['distance_strikes'].append(data)
                        elif i == 7:  # add to clinch strikes
                            sig_columns['clinch_strikes'].append(data)
                        elif i == 8:  # add to ground strikes
                            sig_columns['ground_strikes'].append(data)
            sig_details = pd.DataFrame(sig_columns)

            cfd = pd.merge(ov_details, sig_details, on='fighter', how='left', copy=False)

            cfd['takedowns_landed'] = cfd.takedowns.str.split( ' of ').str[0].astype(int)
            cfd['takedowns_attempts'] = cfd.takedowns.str.split( ' of ').str[-1].astype(int)
            cfd['sig_strikes_landed'] = cfd.sig_strikes.str.split( ' of ').str[0].astype(int)
            cfd['sig_strikes_attempts'] = cfd.sig_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['total_strikes_landed'] = cfd.total_strikes.str.split( ' of ').str[0].astype(int)
            cfd['total_strikes_attempts'] = cfd.total_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['head_strikes_landed'] = cfd.head_strikes.str.split( ' of ').str[0].astype(int)
            cfd['head_strikes_attempts'] = cfd.head_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['body_strikes_landed'] = cfd.body_strikes.str.split( ' of ').str[0].astype(int)
            cfd['body_strikes_attempts'] = cfd.body_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['leg_strikes_landed'] = cfd.leg_strikes.str.split( ' of ').str[0].astype(int)
            cfd['leg_strikes_attempts'] = cfd.leg_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['distance_strikes_landed'] = cfd.distance_strikes.str.split( ' of ').str[0].astype(int)
            cfd['distance_strikes_attempts'] = cfd.distance_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['clinch_strikes_landed'] = cfd.clinch_strikes.str.split( ' of ').str[0].astype(int)
            cfd['clinch_strikes_attempts'] = cfd.clinch_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['ground_strikes_landed'] = cfd.ground_strikes.str.split( ' of ').str[0].astype(int)
            cfd['ground_strikes_attempts'] = cfd.ground_strikes.str.split( ' of ').str[-1].astype(int)

            cfd = cfd.drop(['takedowns', 'sig_strikes', 'total_strikes', 'head_strikes', 'body_strikes', 'leg_strikes', 'distance_strikes',
                            'clinch_strikes', 'ground_strikes'], axis=1)
            return (cfd)
        
    # updates fighter attributes with new fighters not yet saved yet
    def update_fighter_stats(self):
        # TODO find a way to avoid using the old version of fighter_stats.csv ... this is clunky
        fight_hist = self.get('fight_hist')
        fighter_stats = self.get('fighter_stats')
        fighter_stats_urls = fighter_stats.url.unique()
        fight_hist_urls = fight_hist.fighter_url.unique()
        
        fighter_details = {'name': [], 'height': [],
                        'reach': [], 'stance': [], 'dob': [], 'url': []}
        known_fighter_urls = set(fighter_stats_urls)

        for f_url in fight_hist_urls:
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
    
    def update_ufc_fights_crap(self):
        # most recent fight in fight_hist_updated versus most recent fight in ufc_fights_crap
        most_recent_date_in_updated_fight_hist = self.get_most_recent_fight_date('fight_hist')
        most_recent_date_in_old_ufc_fights_crap = self.get_most_recent_fight_date('ufc_fights_crap')
        update_time = time_diff(most_recent_date_in_old_ufc_fights_crap, most_recent_date_in_updated_fight_hist)
        print('days since last update: '+str(update_time))

        # this gives the new rows in fight_hist_updated which do not appear in ufc_fights_crap
        fight_hist_updated = self.get('fight_hist')
        new_rows = fight_hist_updated.loc[time_diff_vect(fight_hist_updated['date'], most_recent_date_in_updated_fight_hist) < update_time].copy()

        ufc_fights_crap = self.get('ufc_fights_crap')
        if update_time > 0: # should just stop the script here. Can do this later once we do everything inside a function call
            new_rows = self.populate_new_fights_with_statistics(new_rows)
            
            # making sure new columns coincide with old columns
            crapcolumns = list(ufc_fights_crap.columns)
            new_rows = new_rows[crapcolumns]
            print('New columns coincide with old columns: ' + str(all(ufc_fights_crap.columns == new_rows.columns)))
            print('joining new data to ufc_fights_crap.csv')

        else:
            print('nothing to update')
            
        frames = [new_rows, ufc_fights_crap]
        updated_ufc_fights_crap = pd.concat(frames, ignore_index=True)

        # saving the updated ufc_fights_crap file
        self.set('ufc_fights_crap', updated_ufc_fights_crap)
        self.save_csv('ufc_fights_crap')
        
    def update_ufc_fights(self):
        # here is the list of all stats available (besides stance), does not include names or result
        computed_statistics = [u'fighter_wins', u'fighter_losses', u'fighter_age', u'fighter_height', u'fighter_reach', u'fighter_L5Y_wins', u'fighter_L5Y_losses', 
                            u'fighter_L2Y_wins', u'fighter_L2Y_losses', u'fighter_ko_wins', u'fighter_ko_losses', u'fighter_L5Y_ko_wins', u'fighter_L5Y_ko_losses', 
                            u'fighter_L2Y_ko_wins', u'fighter_L2Y_ko_losses', u'fighter_sub_wins', u'fighter_sub_losses', u'fighter_L5Y_sub_wins', u'fighter_L5Y_sub_losses', 
                            u'fighter_L2Y_sub_wins', u'fighter_L2Y_sub_losses', u'fighter_inf_knockdowns_avg', u'fighter_inf_pass_avg', u'fighter_inf_reversals_avg', 
                            u'fighter_inf_sub_attempts_avg', u'fighter_inf_takedowns_landed_avg', u'fighter_inf_takedowns_attempts_avg', u'fighter_inf_sig_strikes_landed_avg', 
                            u'fighter_inf_sig_strikes_attempts_avg', u'fighter_inf_total_strikes_landed_avg', u'fighter_inf_total_strikes_attempts_avg', 
                            u'fighter_inf_head_strikes_landed_avg', u'fighter_inf_head_strikes_attempts_avg', u'fighter_inf_body_strikes_landed_avg', 
                            u'fighter_inf_body_strikes_attempts_avg', u'fighter_inf_leg_strikes_landed_avg', u'fighter_inf_leg_strikes_attempts_avg', 
                            u'fighter_inf_distance_strikes_landed_avg', u'fighter_inf_distance_strikes_attempts_avg', u'fighter_inf_clinch_strikes_landed_avg', 
                            u'fighter_inf_clinch_strikes_attempts_avg', u'fighter_inf_ground_strikes_landed_avg', u'fighter_inf_ground_strikes_attempts_avg', 
                            u'fighter_abs_knockdowns_avg', u'fighter_abs_pass_avg', u'fighter_abs_reversals_avg', u'fighter_abs_sub_attempts_avg', 
                            u'fighter_abs_takedowns_landed_avg', u'fighter_abs_takedowns_attempts_avg', u'fighter_abs_sig_strikes_landed_avg', 
                            u'fighter_abs_sig_strikes_attempts_avg', u'fighter_abs_total_strikes_landed_avg', u'fighter_abs_total_strikes_attempts_avg', 
                            u'fighter_abs_head_strikes_landed_avg', u'fighter_abs_head_strikes_attempts_avg', u'fighter_abs_body_strikes_landed_avg', 
                            u'fighter_abs_body_strikes_attempts_avg', u'fighter_abs_leg_strikes_landed_avg', u'fighter_abs_leg_strikes_attempts_avg', 
                            u'fighter_abs_distance_strikes_landed_avg', u'fighter_abs_distance_strikes_attempts_avg', u'fighter_abs_clinch_strikes_landed_avg', 
                            u'fighter_abs_clinch_strikes_attempts_avg', u'fighter_abs_ground_strikes_landed_avg', u'fighter_abs_ground_strikes_attempts_avg', u'opponent_wins', 
                            u'opponent_losses', u'opponent_age',  u'opponent_height', u'opponent_reach', u'opponent_L5Y_wins', u'opponent_L5Y_losses', u'opponent_L2Y_wins', 
                            u'opponent_L2Y_losses', u'opponent_ko_wins', u'opponent_ko_losses', u'opponent_L5Y_ko_wins', u'opponent_L5Y_ko_losses', u'opponent_L2Y_ko_wins',
                            u'opponent_L2Y_ko_losses', u'opponent_sub_wins', u'opponent_sub_losses', u'opponent_L5Y_sub_wins', u'opponent_L5Y_sub_losses', 
                            u'opponent_L2Y_sub_wins', u'opponent_L2Y_sub_losses', u'opponent_inf_knockdowns_avg', u'opponent_inf_pass_avg', u'opponent_inf_reversals_avg', 
                            u'opponent_inf_sub_attempts_avg', u'opponent_inf_takedowns_landed_avg', u'opponent_inf_takedowns_attempts_avg', u'opponent_inf_sig_strikes_landed_avg',
                            u'opponent_inf_sig_strikes_attempts_avg', u'opponent_inf_total_strikes_landed_avg', u'opponent_inf_total_strikes_attempts_avg', 
                            u'opponent_inf_head_strikes_landed_avg', u'opponent_inf_head_strikes_attempts_avg', u'opponent_inf_body_strikes_landed_avg',
                            u'opponent_inf_body_strikes_attempts_avg', u'opponent_inf_leg_strikes_landed_avg', u'opponent_inf_leg_strikes_attempts_avg', 
                            u'opponent_inf_distance_strikes_landed_avg', u'opponent_inf_distance_strikes_attempts_avg', u'opponent_inf_clinch_strikes_landed_avg', 
                            u'opponent_inf_clinch_strikes_attempts_avg', u'opponent_inf_ground_strikes_landed_avg', u'opponent_inf_ground_strikes_attempts_avg', 
                            u'opponent_abs_knockdowns_avg', u'opponent_abs_pass_avg', u'opponent_abs_reversals_avg', u'opponent_abs_sub_attempts_avg', 
                            u'opponent_abs_takedowns_landed_avg', u'opponent_abs_takedowns_attempts_avg', u'opponent_abs_sig_strikes_landed_avg', 
                            u'opponent_abs_sig_strikes_attempts_avg', u'opponent_abs_total_strikes_landed_avg', u'opponent_abs_total_strikes_attempts_avg', 
                            u'opponent_abs_head_strikes_landed_avg', u'opponent_abs_head_strikes_attempts_avg', u'opponent_abs_body_strikes_landed_avg', 
                            u'opponent_abs_body_strikes_attempts_avg', u'opponent_abs_leg_strikes_landed_avg', u'opponent_abs_leg_strikes_attempts_avg', 
                            u'opponent_abs_distance_strikes_landed_avg', u'opponent_abs_distance_strikes_attempts_avg', u'opponent_abs_clinch_strikes_landed_avg', 
                            u'opponent_abs_clinch_strikes_attempts_avg', u'opponent_abs_ground_strikes_landed_avg', u'opponent_abs_ground_strikes_attempts_avg', 
                            u'fighter_stance', u'opponent_stance', '1-fight_math', '6-fight_math', '4-fighter_score_diff', '9-fighter_score_diff', '15-fighter_score_diff',]

        # list containing all columns of any interest
        relevant_list = ['date', 'division', 'fighter', 'opponent', 'result', 'method']
        relevant_list.extend(computed_statistics)

        # creates a clean file with only columns which are relevant to predicting
        updated_ufc_fights_crap = self.get('ufc_fights_crap')
        # TODO would be better to grab existing ufc_fights, and add columns corresponding to new fights 
        updated_ufc_fights = updated_ufc_fights_crap[relevant_list]

        # lets randomly remove one of every two copied fights
        random_indices = [random.choice([i, i+1]) for i in range(0, len(updated_ufc_fights['fighter']), 2)]
        updated_ufc_fights = updated_ufc_fights.drop(random_indices)

        print('cleaning data and adding new cleaned columns to ufc_fights.csv')
        self.set('ufc_fights', updated_ufc_fights)
        self.save_csv('ufc_fights')
        
    def update_ufc_fight_data_for_website(self):
        updated_ufc_fights_crap = self.get('ufc_fights_crap')
        updated_ufc_fights_crap['index'] = list(range(updated_ufc_fights_crap.shape[0])) # add index column to dataframe

        json_columns = ['date', 'result', 'fighter', 'opponent', 'division', 'method', 'round', 'time', 'knockdowns', 'sub_attempts', 'pass', 'reversals', 'takedowns_landed', 
                        'takedowns_attempts', 'sig_strikes_landed', 'sig_strikes_attempts', 'total_strikes_landed', 'total_strikes_attempts', 'head_strikes_landed',
                        'head_strikes_attempts', 'body_strikes_landed', 'body_strikes_attempts', 'leg_strikes_landed', 'leg_strikes_attempts', 'distance_strikes_landed', 
                        'distance_strikes_attempts', 'clinch_strikes_landed', 'clinch_strikes_attempts', 'ground_strikes_landed', 'ground_strikes_attempts', 'fighter_stance', 
                        'opponent_stance', 'index',]

        ufc_fight_data_for_website = updated_ufc_fights_crap[json_columns]

        # make new csv just to send it to json
        # this is inefficient and wastes space... but its just because its the only way I know to make a json file
        # of the correct format (fix needed but not super important)
        print('exporting updated ufc_fights_crap.json for use in javascript portion of website')
        ufc_fight_data_for_website.to_csv('content/data/processed/ufc_fight_data_for_website.csv', index=False)

        # convert ufc_fights_crap.csv to json files to read via javascript in website
        csvFilePath = r'content/data/processed/ufc_fight_data_for_website.csv'
        jsonFilePath = r'content/data/external/ufc_fight_data_for_website.json'
        self.make_json(csvFilePath, jsonFilePath, 'index')
        
    def update_pictures(self):
        # updating the picture scrape
        # updated scraped fighter data (after running fight_hist_updated function from UFC_data_scraping file)
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
            for bookie in ['DraftKings', 'BetMGM', 'Caesars', 'BetRivers', 'FanDuel', 'PointsBet', 'Unibet', 'Bet365', 'BetWay', '5D', 'Ref']:
                if f'fighter {bookie}' in odds_row.columns:
                    predictions_df.at[i, f'fighter {bookie}'] = odds_row[f'{fighter_a} {bookie}'].values[0]
                    predictions_df.at[i, f'opponent {bookie}'] = odds_row[f'{fighter_b} {bookie}'].values[0]
            # add average odds for fighter and opponent
            predictions_df.at[i, f'average bookie odds'] = odds_row['average bookie odds'].values[0]
            
            # add expected values for fighter and opponent
            fighter_predicted_odds = predictions_df.at[i, 'predicted fighter odds']
            dk_fighter_vegas_odds = predictions_df.at[i, f'fighter DraftKings']
            dk_opponent_vegas_odds = predictions_df.at[i, f'opponent DraftKings']
            if not fighter_predicted_odds or not dk_fighter_vegas_odds or not dk_opponent_vegas_odds:
                continue  # skip if any of these values are missing
            fighter_predicted_odds = int(fighter_predicted_odds)
            dk_fighter_vegas_odds = int(dk_fighter_vegas_odds)
            dk_opponent_vegas_odds = int(dk_opponent_vegas_odds)
            fighter_bankroll_percentage, opponent_bankroll_percentage = get_kelly_bet_from_ev_and_dk_odds(fighter_predicted_odds, dk_fighter_vegas_odds, dk_opponent_vegas_odds)
            predictions_df.at[i, 'fighter bet bankroll percentage'] = fighter_bankroll_percentage
            predictions_df.at[i, 'opponent bet bankroll percentage'] = opponent_bankroll_percentage

            
        # save to vegas_oddsjson
        # odds_df= self.drop_irrelevant_fights(odds_df,3) #allows 3 bookies to have missing odds. can increase this to 2 or 3 as needed
        # odds_df = self.drop_non_ufc_fights(odds_df)
        #odds_df=drop_repeats(odds_df)
        print('saving odds to content/data/external/vegas_odds.json')
        result = odds_df.to_json()
        parsed = json.loads(result)
        jsonFilePath='content/data/external/vegas_odds.json'
        with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
           jsonf.write(json.dumps(parsed, indent=4))
        print('saved to '+jsonFilePath)
        return predictions_df
    
    def update_prediction_history(self):

        vegas_odds_old=self.get('vegas_odds', filetype='json') # this is the old vegas odds dataframe (from last week)
        ufc_fights_crap = self.get('ufc_fights_crap') # THIS SHOULD HAVE BEEN UPDATED AT THIS POINT! WE SHOULD ADD A CHECK TO CHECK THIS
        prediction_history=self.get('prediction_history', filetype='json')
        
        currentBankroll = prediction_history['current bankroll after'].iloc[0] if 'current bankroll after' in prediction_history.columns else 300; # default bankroll if not present in prediction history

        # getting rid of fights that didn't actually happen and adding correctness results of those that did
        vegas_odds_old = self.update_prediction_correctness(vegas_odds_old, ufc_fights_crap, currentBankroll)

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
            
    # TODO vegas_odds is really not the right name for this data as it contains predictions, not just vegas odds
    def update_vegas_odds(self, vegas_odds):
        #save to json
        result = vegas_odds.to_json()
        parsed = json.loads(result)
        jsonFilePath='content/data/external/vegas_odds.json'
        with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(parsed, indent=4))
        print('saved to '+jsonFilePath)
        
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
    def drop_non_ufc_fights(self, df):
        irr = []
        for i in df.index:
            if (not in_ufc(df['fighter name'][i])) or (not in_ufc(df['opponent name'][i])):
                irr.append(i)
        df = df.drop(irr)
        return df

    # TODO name should better indicate the context
    def drop_repeats(self, df):
        irr = []
        ufc_fights = self.get('ufc_fights')
        for i in df.index:
            fname = df['fighter name'][i]
            oname = df['opponent name'][i]
            for j in range(200):
                fname_old = ufc_fights['fighter'][j]
                oname_old = ufc_fights['opponent'][j]
                if (same_name(fname, fname_old) and same_name(oname, oname_old)) or (same_name(oname, fname_old) and same_name(fname, oname_old)):
                    irr.append(i)
        df = df.drop(irr)
        return df
    
    # TODO name should better indicate the context
    def update_prediction_correctness(self, vegas_odds_old, ufc_fights_crap, currentBankroll):
        r"""
        This function checks the vegas odds dataframe against the ufc fights dataframe to find fights that didn't happen
        and to add correctness results for those that did happen. It returns a list of indices of fights that didn't happen.
        It also updates the vegas odds dataframe with correctness results for the fights that did happen.
        """
        # getting rid of fights that didn't actually happen and adding correctness results of those that did
        bad_indices = []
        vegas_odds_old['fighter bet'] = 0
        vegas_odds_old['opponent bet'] = 0
        vegas_odds_old['current bankroll after'] = 0
        vegas_odds_old['bet result'] = 'N/A'
        for index1, row1 in vegas_odds_old.iloc[::-1].iterrows(): # iterate backwards in the order the fights actually happened
            card_date = row1['date']
            
            # if no prediction was made, throw it away
            if row1['predicted fighter odds'] == '':
                bad_indices.append(index1)
                print('no prediction made for fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'])
                continue
            
            # if a prediction was made, check if the fight actually happened and then check if the prediction / bet was correct / won
            # TODO this is slow but sort of necessary if we need to add multiple cards at the same time
            relevant_fights = ufc_fights_crap[pd.to_datetime(ufc_fights_crap['date']) == card_date]
            print(f'searching through {relevant_fights.shape[0]//2} confirmed fights on {str(card_date).split(" ")[0]} for {row1["fighter name"]} vs {row1["opponent name"]}')
            fighter_odds = int(row1['predicted fighter odds'])
            fighter_dk_odds = int(row1.get('fighter DraftKings'))
            opponent_dk_odds = int(row1.get('opponent DraftKings'))
            fighter_bankroll_percentage = float(row1.get('fighter bet bankroll percentage', 0))
            opponent_bankroll_percentage = float(row1.get('opponent bet bankroll percentage', 0))
            
            match_found = False
            for index2, row2 in relevant_fights.iterrows():
                if same_name(row1['fighter name'], row2['fighter']) and same_name(row1['opponent name'], row2['opponent']):
                    match_found = True
                    print('adding fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'])
                    if (int(fighter_odds) < 0 and row2['result'] == 'W') or (int(fighter_odds) > 0 and row2['result'] == 'L'):
                        vegas_odds_old.at[index1,'correct?'] = 1
                    else:
                        vegas_odds_old.at[index1,'correct?'] = 0
                    # update the bankroll based on the bet made
                    fighter_bet = 0
                    opponent_bet = 0
                    fighter_payout = 0
                    opponent_payout = 0
                    bet_result = 'N/A'
                    if fighter_bankroll_percentage > 0: # check if we even made a bet on the fighter
                        fighter_bet = fighter_bankroll_percentage / 100 * currentBankroll
                        vegas_odds_old.at[index1, 'fighter bet'] = fighter_bet
                        bet_result = row2['result']
                        fighter_payout = bet_payout(fighter_dk_odds, fighter_bet, bet_result)
                    if opponent_bankroll_percentage > 0: # check if we even made a bet on the opponent
                        opponent_bet = opponent_bankroll_percentage / 100 * currentBankroll
                        vegas_odds_old.at[index1, 'opponent bet'] = opponent_bet
                        # win the bet if the opponent wins (the result column is the result of the fighter, so if the fighter wins, the opponent loses)
                        bet_result = 'L' if row2['result'] == 'W' else 'W'
                        opponent_payout = bet_payout(opponent_dk_odds, opponent_bet, bet_result)
                    currentBankroll = currentBankroll + fighter_payout + opponent_payout - fighter_bet - opponent_bet
                    vegas_odds_old.at[index1, 'current bankroll after'] = currentBankroll
                    vegas_odds_old.at[index1, 'bet result'] = bet_result
                    # TODO add case for draw
                    break
            if not match_found: # if the fight didn't happen, throw it away
                bad_indices.append(index1)
                print('fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'] + ' not found in ufc_fights_crap.csv')
        vegas_odds_old = vegas_odds_old.drop(bad_indices)
        return vegas_odds_old
                
    def populate_new_fights_with_statistics(self, new_rows):
        # Note, getting this warning a lot
        # C:\Users\Alex\OneDrive\Documents\GitHub\UFC_Prediction_2022\src\data_handler\data_handler.py:821: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) 
        # instead. To get a de-fragmented frame, use `newframe = frame.copy()`
        # new_rows['opponent_abs_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])        
        print('adding physical statistics for fighter')
        new_rows_dict = {}
        new_rows_dict['fighter_wins'] = wins_before_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_losses'] = losses_before_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_age'] = fighter_age_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_height'] = new_rows['fighter'].apply(fighter_height)
        new_rows_dict['fighter_reach'] = new_rows['fighter'].apply(fighter_height)

        print('adding record statistics for fighter')
        new_rows_dict['fighter_L5Y_wins'] = L5Y_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_L5Y_losses'] = L5Y_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_L2Y_wins'] = L2Y_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_L2Y_losses'] = L2Y_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_ko_wins'] = ko_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_ko_losses'] = ko_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_L5Y_ko_wins'] = L5Y_ko_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_L5Y_ko_losses'] = L5Y_ko_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_L2Y_ko_wins'] = L2Y_ko_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_L2Y_ko_losses'] = L2Y_ko_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_sub_wins'] = sub_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_sub_losses'] = sub_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_L5Y_sub_wins'] = L5Y_sub_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_L5Y_sub_losses'] = L5Y_sub_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_L2Y_sub_wins'] = L2Y_sub_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows_dict['fighter_L2Y_sub_losses'] = L2Y_sub_losses_vect(new_rows['fighter'], new_rows['date'])

        print('adding inflicted punch, kick, grappling statistics for fighter... this will take a few minutes')

        new_rows_dict['fighter_inf_knockdowns_avg'] = avg_count_vect('knockdowns', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_pass_avg'] = avg_count_vect('pass', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_reversals_avg'] = zero_vect(new_rows['fighter'])
        new_rows_dict['fighter_inf_sub_attempts_avg'] = avg_count_vect('sub_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_takedowns_landed_avg'] = avg_count_vect('takedowns_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_takedowns_attempts_avg'] = avg_count_vect('takedowns_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        print('quarter done')
        new_rows_dict['fighter_inf_sig_strikes_landed_avg'] = avg_count_vect('sig_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_sig_strikes_attempts_avg'] = avg_count_vect('sig_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_total_strikes_landed_avg'] = avg_count_vect('total_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_total_strikes_attempts_avg'] = avg_count_vect('total_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_head_strikes_landed_avg'] = avg_count_vect('head_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_head_strikes_attempts_avg'] = avg_count_vect('head_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        print('half done')
        new_rows_dict['fighter_inf_body_strikes_landed_avg'] = avg_count_vect('body_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_body_strikes_attempts_avg'] = avg_count_vect('body_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_leg_strikes_landed_avg'] = avg_count_vect('leg_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_leg_strikes_attempts_avg'] = avg_count_vect('leg_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        print('almost done')
        new_rows_dict['fighter_inf_distance_strikes_attempts_avg'] = avg_count_vect('distance_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_clinch_strikes_landed_avg'] = avg_count_vect('clinch_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_clinch_strikes_attempts_avg'] = avg_count_vect('clinch_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_ground_strikes_landed_avg'] = avg_count_vect('ground_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows_dict['fighter_inf_ground_strikes_attempts_avg'] = avg_count_vect('ground_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])

        print('adding absorbed punch, kick, grappling statistics for fighter... this will take a few minutes')

        new_rows_dict['fighter_abs_knockdowns_avg'] = avg_count_vect('knockdowns', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_pass_avg'] = avg_count_vect('pass', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_reversals_avg'] = zero_vect(new_rows['fighter'])
        new_rows_dict['fighter_abs_sub_attempts_avg'] = avg_count_vect('sub_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_takedowns_landed_avg'] = avg_count_vect('takedowns_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_takedowns_attempts_avg'] = avg_count_vect('takedowns_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        print('quarter done')
        new_rows_dict['fighter_abs_sig_strikes_landed_avg'] = avg_count_vect('sig_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_sig_strikes_attempts_avg'] = avg_count_vect('sig_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_total_strikes_landed_avg'] = avg_count_vect('total_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_total_strikes_attempts_avg'] = avg_count_vect('total_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_head_strikes_landed_avg'] = avg_count_vect('head_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_head_strikes_attempts_avg'] = avg_count_vect('head_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        print('half done')
        new_rows_dict['fighter_abs_body_strikes_landed_avg'] = avg_count_vect('body_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_body_strikes_attempts_avg'] = avg_count_vect('body_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_leg_strikes_landed_avg'] = avg_count_vect('leg_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_leg_strikes_attempts_avg'] = avg_count_vect('leg_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        print('almost done')
        new_rows_dict['fighter_abs_distance_strikes_attempts_avg'] = avg_count_vect('distance_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_clinch_strikes_landed_avg'] = avg_count_vect('clinch_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_clinch_strikes_attempts_avg'] = avg_count_vect('clinch_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_ground_strikes_landed_avg'] = avg_count_vect('ground_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows_dict['fighter_abs_ground_strikes_attempts_avg'] = avg_count_vect('ground_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        
        # add these columns so we can call opponent_column on them
        new_rows_df1 = pd.DataFrame(new_rows_dict)
        new_rows = pd.concat([new_rows_df1, new_rows], axis=1)
        # empty new_rows_dict to add opponent statistics
        new_rows_dict = {}
        
        print('adding physical statistics for opponent')
        # TODO THESE ARE OF SHAPE (16378, 1) while new_rows is of shape (26,) leading to a ValueError when trying to create a DataFrame
        new_rows_dict['opponent_wins'] = opponent_column(new_rows['fighter_wins'])
        new_rows_dict['opponent_losses'] = opponent_column(new_rows['fighter_losses'])
        new_rows_dict['opponent_age'] = opponent_column(new_rows['fighter_age'])
        new_rows_dict['opponent_height'] = opponent_column(new_rows['fighter_height'])
        new_rows_dict['opponent_reach'] = opponent_column(new_rows['fighter_reach'])

        print('adding record statistics for opponent')

        new_rows_dict['opponent_L5Y_wins'] = opponent_column(new_rows['fighter_L5Y_wins'])
        new_rows_dict['opponent_L5Y_losses'] = opponent_column(new_rows['fighter_L5Y_losses'])
        new_rows_dict['opponent_L2Y_wins'] = opponent_column(new_rows['fighter_L2Y_wins'])
        new_rows_dict['opponent_L2Y_losses'] = opponent_column(new_rows['fighter_L2Y_losses'])
        new_rows_dict['opponent_ko_wins'] = opponent_column(new_rows['fighter_ko_wins'])
        new_rows_dict['opponent_ko_losses'] = opponent_column(new_rows['fighter_ko_losses'])
        new_rows_dict['opponent_L5Y_ko_wins'] = opponent_column(new_rows['fighter_L5Y_ko_wins'])
        new_rows_dict['opponent_L5Y_ko_losses'] = opponent_column(new_rows['fighter_L5Y_ko_losses'])
        new_rows_dict['opponent_L2Y_ko_wins'] = opponent_column(new_rows['fighter_L2Y_ko_wins'])
        new_rows_dict['opponent_L2Y_ko_losses'] = opponent_column(new_rows['fighter_L2Y_ko_losses'])
        new_rows_dict['opponent_sub_wins'] = opponent_column(new_rows['fighter_sub_wins'])
        new_rows_dict['opponent_sub_losses'] = opponent_column(new_rows['fighter_sub_losses'])
        new_rows_dict['opponent_L5Y_sub_wins'] = opponent_column(new_rows['fighter_L5Y_sub_wins'])
        new_rows_dict['opponent_L5Y_sub_losses'] = opponent_column(new_rows['fighter_L5Y_sub_losses'])
        new_rows_dict['opponent_L2Y_sub_wins'] = opponent_column(new_rows['fighter_L2Y_sub_wins'])
        new_rows_dict['opponent_L2Y_sub_losses'] = opponent_column(new_rows['fighter_L2Y_sub_losses'])

        print('adding inflicted punch, kick, grappling statistics for opponent... this will take a few minutes')

        new_rows_dict['opponent_inf_knockdowns_avg'] = avg_count_vect('knockdowns', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_pass_avg'] = avg_count_vect('pass', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_reversals_avg'] = zero_vect(new_rows['opponent'])
        new_rows_dict['opponent_inf_sub_attempts_avg'] = avg_count_vect('sub_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_takedowns_landed_avg'] = avg_count_vect('takedowns_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_takedowns_attempts_avg'] = avg_count_vect('takedowns_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        print('quarter done')
        
        
        new_rows_dict['opponent_inf_sig_strikes_landed_avg'] = avg_count_vect('sig_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_sig_strikes_attempts_avg'] = avg_count_vect('sig_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_total_strikes_landed_avg'] = avg_count_vect('total_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_total_strikes_attempts_avg'] = avg_count_vect('total_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_head_strikes_landed_avg'] = avg_count_vect('head_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        print('half done')
        new_rows_dict['opponent_inf_head_strikes_attempts_avg'] = avg_count_vect('head_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_body_strikes_landed_avg'] = avg_count_vect('body_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_body_strikes_attempts_avg'] = avg_count_vect('body_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_leg_strikes_landed_avg'] = avg_count_vect('leg_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_leg_strikes_attempts_avg'] = avg_count_vect('leg_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        print('almost done')
        new_rows_dict['opponent_inf_distance_strikes_attempts_avg'] = avg_count_vect('distance_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_clinch_strikes_landed_avg'] = avg_count_vect('clinch_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_clinch_strikes_attempts_avg'] = avg_count_vect('clinch_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_ground_strikes_landed_avg'] = avg_count_vect('ground_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows_dict['opponent_inf_ground_strikes_attempts_avg'] = avg_count_vect('ground_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])

        print('adding absorbed punch, kick, grappling statistics for opponent... this will take a few minutes')

        new_rows_dict['opponent_abs_knockdowns_avg'] = avg_count_vect('knockdowns', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_pass_avg'] = avg_count_vect('pass', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_reversals_avg'] = zero_vect(new_rows['opponent'])
        new_rows_dict['opponent_abs_sub_attempts_avg'] = avg_count_vect('sub_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_takedowns_landed_avg'] = avg_count_vect('takedowns_landed', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_takedowns_attempts_avg'] = avg_count_vect('takedowns_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_sig_strikes_landed_avg'] = avg_count_vect('sig_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        print('quarter done')
        new_rows_dict['opponent_abs_sig_strikes_attempts_avg'] = avg_count_vect('sig_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_total_strikes_landed_avg'] = avg_count_vect('total_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_total_strikes_attempts_avg'] = avg_count_vect('total_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_head_strikes_landed_avg'] = avg_count_vect('head_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_head_strikes_attempts_avg'] = avg_count_vect('head_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_body_strikes_landed_avg'] = avg_count_vect('body_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        print('half done')
        new_rows_dict['opponent_abs_body_strikes_attempts_avg'] = avg_count_vect('body_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_leg_strikes_landed_avg'] = avg_count_vect('leg_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_leg_strikes_attempts_avg'] = avg_count_vect('leg_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        print('almost done')
        new_rows_dict['opponent_abs_distance_strikes_attempts_avg'] = avg_count_vect('distance_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_clinch_strikes_landed_avg'] = avg_count_vect('clinch_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_clinch_strikes_attempts_avg'] = avg_count_vect('clinch_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_ground_strikes_landed_avg'] = avg_count_vect('ground_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows_dict['opponent_abs_ground_strikes_attempts_avg'] = avg_count_vect('ground_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])

        new_rows_dict['fighter_stance'] = stance_vect(new_rows['fighter'])
        new_rows_dict['opponent_stance'] = stance_vect(new_rows['opponent'])

        print('adding fight_math and fighter_score statistics')
        new_rows_dict['1-fight_math'] = fight_math_diff_vect(new_rows['fighter'], new_rows['opponent'], new_rows['date'], 1)
        new_rows_dict['6-fight_math'] = fight_math_diff_vect(new_rows['fighter'], new_rows['opponent'], new_rows['date'], 6)
        new_rows_dict['4-fighter_score_diff'] = fighter_score_diff_vect(new_rows['fighter'], new_rows['opponent'], new_rows['date'], 4)
        new_rows_dict['9-fighter_score_diff'] = fighter_score_diff_vect(new_rows['fighter'], new_rows['opponent'], new_rows['date'], 9)
        new_rows_dict['15-fighter_score_diff'] = fighter_score_diff_vect(new_rows['fighter'], new_rows['opponent'], new_rows['date'], 15)
                
        new_rows_df2 = pd.DataFrame(new_rows_dict)
        new_rows = pd.concat([new_rows_df2, new_rows], axis=1)
        return new_rows
    
    ########### FUNCTIONS USED IN update_data_csvs_and_jsons.py ###########

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