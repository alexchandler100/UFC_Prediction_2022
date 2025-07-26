import git, os
git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
os.chdir(f'{git_root}/src')
# from data_handler import DataHandler

import pandas as pd
from datetime import date, datetime
import random
import requests
from bs4 import BeautifulSoup
import re

#have to change directory to import functions after April 2022 restructure of folders
from fight_stat_helpers import *

def get_fight_card(url):
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
        strike_data_dict = get_fight_stats(fight_url, fighter, opponent)
                
        if strike_data_dict is None:
            continue  # skip this fight if no fight details available
        
        # join to fight details
        fight_data_dict = pd.merge(fight_data_dict, strike_data_dict, on='fighter', how='left', copy=False)
        
        # add fight details to fight card
        fight_card = pd.concat([fight_data_dict, fight_card], axis=0)
        
    fight_card = fight_card.reset_index(drop=True)
    return fight_card

    
def get_fight_stats(url, fighter, opponent):
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

# TODO for some reason this did not grab UFC 1
#function that gets stats on all fights on all cards
def get_all_fight_stats():
    url = 'http://ufcstats.com/statistics/events/completed?page=all'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser") 
    
    events_table = soup.select_one('tbody')
    events = [event['href'] for event in events_table.select('a')[1:]] #omit first event, future event

    fight_stats = pd.DataFrame()
    event_count = 0
    for event in events:
        # if event_count > 2:  # limit to the last 3 if we want to debug quickly
        #     break
        stats = get_fight_card(event)
        fight_stats = pd.concat([stats, fight_stats], axis = 0)
        event_count += 1
        
    fight_stats = fight_stats.reset_index(drop = True)
    return fight_stats      

df = get_all_fight_stats()

# save to csv
df.to_csv(f'{git_root}/src/content/data/processed/ufc_fights_reported_doubled.csv', index = False)
print(f'Saved fight history with stats to {git_root}/src/content/data/processed/ufc_fights_reported_doubled.csv')