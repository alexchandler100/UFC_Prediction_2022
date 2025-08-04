import git, os
git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
os.chdir(f'{git_root}/src')

import pandas as pd

#have to change directory to import functions after April 2022 restructure of folders
from fight_stat_helpers import *
from data_handler import DataHandler

dh = DataHandler()
fighter_stats = dh.get('fighter_stats')
ufc_fights_reported_doubled = dh.get('ufc_fights_reported_doubled')

ufc_fights_reported_doubled['date'] = pd.to_datetime(ufc_fights_reported_doubled['date'], format='%Y-%m-%d')
ufc_fights_reported_doubled = ufc_fights_reported_doubled.loc[::-1]
ufc_fights_reported_doubled.set_index('date', inplace=True)
ufc_fights_reported_derived_doubled = ufc_fights_reported_doubled[['fighter', 'opponent', 'result', 'method', 'division']].copy()
names_to_update = None # default to all names
# names_to_update = ['Anthony Hernandez'] # for debugging purposes, only update this fighter's stats
ufc_fights_reported_derived_doubled = dh.fill_in_statistics_for_fights(ufc_fights_reported_derived_doubled, ufc_fights_reported_doubled, names_to_update=names_to_update)

# save the results to a csv file 
ufc_fights_reported_derived_doubled_path = f'{git_root}/src/content/data/processed/ufc_fights_reported_derived_doubled.csv'
ufc_fights_reported_derived_doubled.to_csv(ufc_fights_reported_derived_doubled_path, index=True)
print(f'Saved ufc_fights_reported_derived_doubled to {ufc_fights_reported_derived_doubled_path}, shape {ufc_fights_reported_derived_doubled.shape}')
