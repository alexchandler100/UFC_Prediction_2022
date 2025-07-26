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
    # 'method', # TODO predict method too
    # 'division', # TODO filter by division
    'stance', # TODO incorporate stance
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

# grab fighter and opponent columns for diffing in flattened dataframe
fighter_col = ufc_fights_crap['fighter'].loc[::2]
opponent_col = ufc_fights_crap['opponent'].loc[::2]
result_col = ufc_fights_crap['result'].loc[::2]
method_col = ufc_fights_crap['method'].loc[::2]
division_col = ufc_fights_crap['division'].loc[::2]

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
        ufc_fights_predictive_diffs_dict[f'{col}_sum'] = ufc_fights_predictive_even[col].values + ufc_fights_predictive_odd[col].values
        # ufc_fights_predictive_diffs_dict[f'{col}_sq_diff'] = ufc_fights_predictive_even[col].values ** 2 - ufc_fights_predictive_odd[col].values ** 2
        # ufc_fights_predictive_diffs_dict[f'{col}_sq_sum'] = ufc_fights_predictive_even[col].values ** 2 + ufc_fights_predictive_odd[col].values ** 2
        
ufc_fights_predictive_diffs = pd.DataFrame(ufc_fights_predictive_diffs_dict)

# save to csv
ufc_fights_predictive_diffs_path = f'{git_root}/src/content/data/processed/ufc_fights_predictive_flattened_diffs2.csv'
ufc_fights_predictive_diffs.to_csv(ufc_fights_predictive_diffs_path, index=False)
print(f"Saved ufc_fights_predictive_diffs to {ufc_fights_predictive_diffs_path}")

