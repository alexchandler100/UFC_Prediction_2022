import git, os, sys
git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
os.chdir(f'{git_root}/src')
sys.path.append(os.path.abspath(os.path.join(f'{git_root}/src')))
print(f'Changed working directory to {os.getcwd()}')
from data_handler import DataHandler

dh = DataHandler()

prediction_history = dh.get('prediction_history', filetype='json')

fighter_bookie_columns = [f'fighter {bookie}' for bookie in dh.bookies]
opponent_bookie_columns = [f'opponent {bookie}' for bookie in dh.bookies]

import pandas as pd
# make mask on prediction history so that fighter bookie columns are not empty strings 
recent_fights_mask = prediction_history['date'] > pd.Timestamp('2025-06-22') # get all fights after june 22 (when I started adding bookie odds)

from fight_stat_helpers import bet_payout
    
    
currentBankroll = 300

for index1, row1 in prediction_history.iloc[9::-1].iterrows(): # iterate backwards in the order the fights actually happened    
    if not row1['date']:
        continue 
        
    if row1['date'] < pd.Timestamp('2025-06-22'):
        continue
    
    # if no prediction was made, throw it away
    if row1['predicted fighter odds'] == '':
        print(f"no prediction made for fight from  between {row1['fighter name']} and {row1['opponent name']}")
        continue
    
    # import ipdb; ipdb.set_trace(context=10) # uncomment to debug
    
    fighter_odds = int(row1['predicted fighter odds'])
    opponent_odds = int(row1['predicted opponent odds'])
    best_fighter_bookie = row1['best fighter bookie']
    best_opponent_bookie = row1['best opponent bookie']
    
    # check if we ever found odds for the fighter and opponent
    best_fighter_bookie_odds = row1.get(f'fighter {best_fighter_bookie}')
    if not best_fighter_bookie_odds:
        print(f'No odds found for fighter {row1["fighter name"]} skipping...')
    best_opponent_bookie_odds = row1.get(f'opponent {best_opponent_bookie}')
    if not best_opponent_bookie_odds:
        print(f'No odds found for opponent {row1["opponent name"]} skipping...')
    
    if best_fighter_bookie_odds:
        best_fighter_bookie_odds = int(row1.get(f'fighter {best_fighter_bookie}'))
    if best_opponent_bookie_odds:
        best_opponent_bookie_odds = int(row1.get(f'opponent {best_opponent_bookie}'))
        
    fighter_bankroll_percentage = float(row1.get('fighter bet bankroll percentage', 0))
    opponent_bankroll_percentage = float(row1.get('opponent bet bankroll percentage', 0))
    
    # if a prediction was made, check if the fight actually happened and then check if the prediction / bet was correct / won
    # TODO this is slow but sort of necessary if we need to add multiple cards at the same time

    fighter_won = False
    if (int(fighter_odds) < int(opponent_odds) and prediction_history.at[index1,'correct?'] == 1):
        fighter_won = True
    elif (int(fighter_odds) > int(opponent_odds) and prediction_history.at[index1,'correct?'] == 0):
        fighter_won = True
    elif prediction_history.at[index1,'correct?'] == 'N/A' or (int(fighter_odds) == 100):
        # if the odds are 100, it means the model predicted a draw, so we don't know who won
        continue # skip this fight, we don't know who won in this context
    # update the bankroll based on the bet made
    fighter_bet = 0
    opponent_bet = 0
    fighter_payout = 0
    opponent_payout = 0
    bet_result = 'N/A'
    if fighter_bankroll_percentage > 0: # check if we even made a bet on the fighter
        bet_result = 'W' if fighter_won else 'L'
        fighter_bet = fighter_bankroll_percentage / 100 * currentBankroll
        prediction_history.at[index1, 'fighter bet'] = fighter_bet
        fighter_payout = bet_payout(best_fighter_bookie_odds, fighter_bet, bet_result)
    if opponent_bankroll_percentage > 0: # check if we even made a bet on the opponent
        bet_result = 'W' if not fighter_won else 'L'
        opponent_bet = opponent_bankroll_percentage / 100 * currentBankroll
        prediction_history.at[index1, 'opponent bet'] = opponent_bet
        # win the bet if the opponent wins (the result column is the result of the fighter, so if the fighter wins, the opponent loses)
        opponent_payout = bet_payout(best_opponent_bookie_odds, opponent_bet, bet_result)
    currentBankroll = currentBankroll + fighter_payout + opponent_payout - fighter_bet - opponent_bet
    prediction_history.at[index1, 'current bankroll after'] = currentBankroll
    prediction_history.at[index1, 'bet result'] = bet_result
    
import json
# import ipdb;ipdb.set_trace()
#saving the new prediction_history dataframe to json
result = prediction_history.to_json()
parsed = json.loads(result)
prediction_history_filtpath = dh.json_filepaths['prediction_history']

# TODO USE THE SAVE FUNCTION OF THE DATA HANDLER
with open(prediction_history_filtpath, 'w', encoding='utf-8') as jsonf:
    jsonf.write(json.dumps(parsed, indent=4))
    
print(f'saved to {prediction_history_filtpath}')