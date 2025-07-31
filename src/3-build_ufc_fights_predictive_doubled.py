# NOTE: This makes a non symmetric predictive dataset where we repeat each fight twice, once with each fighter as "fighter" and once as "opponent"
# This is to allow the model to learn that it doesn't matter which fighter is which, the model should predict the same odds
# WE ARE CURRENTLY NOT USING THIS. We are using the symmetric version via ufc_fights_predictive_flattened_diffs in update_and_rebuild_model.py
import git, os
git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
os.chdir(f'{git_root}/src')

import pandas as pd
#have to change directory to import functions after April 2022 restructure of folders
# from fight_stat_helpers import *

ufc_fights_predictive_path = f'{git_root}/src/content/data/processed/ufc_fights_reported_derived_doubled.csv'
ufc_fights_crap = pd.read_csv(ufc_fights_predictive_path)

non_predictive_columns = [
    'date',
    'fighter',
    'opponent',
    'method', # TODO predict method too
    # 'division', # TODO filter by division
    'stance', # TODO incorporate stance
    # 'result', # TODO predict result too
]

predictive_columns = [col for col in ufc_fights_crap.columns if col not in non_predictive_columns]


# shuffle pairs to avoid bias
assert len(ufc_fights_crap) % 2 == 0, "DataFrame length must be even to create pairs"
shuffled_rows = []
for i in range(0, len(ufc_fights_crap), 2):
    pair = ufc_fights_crap.iloc[i:i+2]
    pair = pair.sample(frac=1).reset_index(drop=True)  # shuffle within the pair
    shuffled_rows.append(pair)
# Concatenate back into a single DataFrame
ufc_fights_crap = pd.concat(shuffled_rows).reset_index(drop=True)

# drop non-predictive columns
ufc_fights_predictive = ufc_fights_crap[predictive_columns]

fighter_col = ufc_fights_crap['fighter'].loc[::2]
opponent_col = ufc_fights_crap['opponent'].loc[::2]

# flatten into a dataframe with fighter and opponent columns
ufc_fights_predictive_even = ufc_fights_predictive.loc[::2].copy()
ufc_fights_predictive_odd = ufc_fights_predictive.loc[1::2].copy()
ufc_fights_predictive_even = ufc_fights_predictive_even[predictive_columns].reset_index(drop=True)
ufc_fights_predictive_odd = ufc_fights_predictive_odd[predictive_columns].reset_index(drop=True)
# add prefixes 
ufc_fights_predictive_even_order_1 = ufc_fights_predictive_even.copy()
ufc_fights_predictive_odd_order_1 = ufc_fights_predictive_odd.copy()
ufc_fights_predictive_even_order_2 = ufc_fights_predictive_even.copy()
ufc_fights_predictive_odd_order_2 = ufc_fights_predictive_odd.copy()
ufc_fights_predictive_even_order_1 = ufc_fights_predictive_even_order_1.add_prefix('fighter_')
ufc_fights_predictive_odd_order_1 = ufc_fights_predictive_odd_order_1.add_prefix('opponent_')
ufc_fights_predictive_odd_order_2 = ufc_fights_predictive_odd_order_2.add_prefix('fighter_')
ufc_fights_predictive_even_order_2 = ufc_fights_predictive_even_order_2.add_prefix('opponent_')

ufc_fights_predictive_flattened_order_1 = pd.concat([ufc_fights_predictive_even_order_1, ufc_fights_predictive_odd_order_1], axis=1)
# in order 2 opponent is labeled fighter and fighter is labeled opponent. This is to force the model to be symmetric
# so that it doesn't matter which fighter is which, the model will still predict the same odds
ufc_fights_predictive_flattened_order_2 = pd.concat([ufc_fights_predictive_odd_order_2, ufc_fights_predictive_even_order_2], axis=1)

# join the two orders together
ufc_fights_predictive_flattened = pd.concat([ufc_fights_predictive_flattened_order_1, ufc_fights_predictive_flattened_order_2], axis=0).reset_index(drop=True)

# save to csv
ufc_fights_predictive_doubled_path = f'{git_root}/src/content/data/processed/ufc_fights_predictive_doubled.csv'
ufc_fights_predictive_flattened.to_csv(ufc_fights_predictive_doubled_path, index=False)
print(f"Saved ufc_fights_predictive_doubled to {ufc_fights_predictive_doubled_path}")


