import git, os
git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
os.chdir(f'{git_root}/src')
# from data_handler import DataHandler

import pandas as pd
import requests
from bs4 import BeautifulSoup

#have to change directory to import functions after April 2022 restructure of folders
# TODO USE datahandler to save the data
# TODO INCLUDE DWCS fights from ufcstats.com

from fight_stat_helpers import get_fight_card

# TODO for some reason this did not grab UFC 1
#function that gets stats on all fights on all cards
def get_all_fight_stats():
    url = 'http://ufcstats.com/statistics/events/completed?page=all'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser") 
    
    events_table = soup.select_one('tbody')
    events = list(events_table.select('a')[1:]) #omit first event, future event
    event_dict = {}
    for event in events:
        event_name = event.text.strip()
        event_dict[event_name] = event['href']

    fight_stats = pd.DataFrame()
    event_count = 0
    for name, event in event_dict.items():
        # if event_count > 2:  # limit to the last 3 if we want to debug quickly
        #     break
        if "road to ufc" in name.lower():
            continue  # skip Road to UFC events
        stats = get_fight_card(event)
        fight_stats = pd.concat([fight_stats, stats], axis = 0)
        event_count += 1
        
    fight_stats = fight_stats.reset_index(drop = True)
    return fight_stats      

df = get_all_fight_stats()

# save to csv
df.to_csv(f'{git_root}/src/content/data/processed/ufc_fights_reported_doubled.csv', index = False)
print(f'Saved fight history with stats to {git_root}/src/content/data/processed/ufc_fights_reported_doubled.csv')