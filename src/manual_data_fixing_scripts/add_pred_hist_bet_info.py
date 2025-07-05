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

# make mask on prediction history so that fighter bookie columns are not empty strings 
has_fighter_bookie_odds_mask = prediction_history[fighter_bookie_columns].apply(lambda x: x != '').any(axis=1)

from fight_stat_helpers import get_kelly_bet_from_ev_and_dk_odds, bet_payout

# for each row in prediction history, find the bookie with the best odds for the fighter
for i, row in prediction_history[has_fighter_bookie_odds_mask].iterrows():
    predicted_fighter_odds = row['predicted fighter odds']
    predicted_opponent_odds = row['predicted opponent odds']
    
    bookies_with_fighter_odds = {}    
    bookies_with_opponent_odds = {}
    for bookie in dh.bookies:
        fighter_col = f'fighter {bookie}'
        # if the fighter bookie odds are empty, set them to the predicted fighter odds
        if row[fighter_col] == '':
            best_fighter_bookie = ''
            best_fighter_bookie_odds = ''
            continue 
        bookie_fighter_odds = row[fighter_col]
        bookies_with_fighter_odds[bookie] = bookie_fighter_odds
    
        opponent_col = f'opponent {bookie}'
        # if the opponent bookie odds are empty, set them to the predicted opponent odds
        if row[opponent_col] == '':
            best_opponent_bookie = ''
            best_opponent_bookie_odds = ''
            continue 
        bookie_opponent_odds = row[opponent_col]
        bookies_with_opponent_odds[bookie] = bookie_opponent_odds
        
    # compute the kelly bankroll for the fighter
    fighter_bookie_kelly_dict = {}
    opponent_bookie_kelly_dict = {}
    for bookie in bookies_with_fighter_odds.keys():
        bookie_fighter_odds = bookies_with_fighter_odds[bookie]
        bookie_opponent_odds = bookies_with_opponent_odds[bookie]
        if bookie_opponent_odds == '':
            print(f'weird, no opponent odds for {bookie} in row {i}')
            continue 
        
        # get the kelly bet for the fighter and opponent
        fighter_kelly, opponent_kelly = get_kelly_bet_from_ev_and_dk_odds(int(predicted_fighter_odds), int(bookie_fighter_odds), int(bookie_opponent_odds))
        fighter_bookie_kelly_dict[bookie] = fighter_kelly
        opponent_bookie_kelly_dict[bookie] = opponent_kelly
        
    best_fighter_bookie = max(fighter_bookie_kelly_dict, key=fighter_bookie_kelly_dict.get)
    best_opponent_bookie = max(opponent_bookie_kelly_dict, key=opponent_bookie_kelly_dict.get)
    prediction_history.at[i, 'best fighter bookie'] = best_fighter_bookie
    prediction_history.at[i, 'best opponent bookie'] = best_opponent_bookie
    prediction_history.at[i, 'fighter bet bankroll percentage'] = fighter_bookie_kelly_dict[best_fighter_bookie]
    prediction_history.at[i, 'opponent bet bankroll percentage'] = opponent_bookie_kelly_dict[best_opponent_bookie]
    
    
import pandas as pd
    
# needed to make custom version of the function from data handler for these specific needs
def update_prediction_correctness(vegas_odds_old, currentBankroll):
    r"""
    This function checks the vegas odds dataframe against the ufc fights dataframe to find fights that didn't happen
    and to add correctness results for those that did happen. It returns a list of indices of fights that didn't happen.
    It also updates the vegas odds dataframe with correctness results for the fights that did happen.
    """
    vegas_odds_old['fighter bet'] = 0
    vegas_odds_old['opponent bet'] = 0
    vegas_odds_old['current bankroll after'] = 0
    vegas_odds_old['bet result'] = 'N/A'
    for index1, row1 in vegas_odds_old.iloc[::-1].iterrows(): # iterate backwards in the order the fights actually happened        
        # if no prediction was made, throw it away
        if row1['predicted fighter odds'] == '':
            print(f"no prediction made for fight from  between {row1['fighter name']} and {row1['opponent name']}")
            continue
        
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
        if (int(fighter_odds) < int(opponent_odds) and vegas_odds_old.at[index1,'correct?'] == 1):
            fighter_won = True
        elif (int(fighter_odds) > int(opponent_odds) and vegas_odds_old.at[index1,'correct?'] == 0):
            fighter_won = True
        elif vegas_odds_old.at[index1,'correct?'] == 'N/A' or (int(fighter_odds) == 100):
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
            vegas_odds_old.at[index1, 'fighter bet'] = fighter_bet
            fighter_payout = bet_payout(best_fighter_bookie_odds, fighter_bet, bet_result)
        if opponent_bankroll_percentage > 0: # check if we even made a bet on the opponent
            bet_result = 'W' if not fighter_won else 'L'
            opponent_bet = opponent_bankroll_percentage / 100 * currentBankroll
            vegas_odds_old.at[index1, 'opponent bet'] = opponent_bet
            # win the bet if the opponent wins (the result column is the result of the fighter, so if the fighter wins, the opponent loses)
            opponent_payout = bet_payout(best_opponent_bookie_odds, opponent_bet, bet_result)
        currentBankroll = currentBankroll + fighter_payout + opponent_payout - fighter_bet - opponent_bet
        vegas_odds_old.at[index1, 'current bankroll after'] = currentBankroll
        vegas_odds_old.at[index1, 'bet result'] = bet_result

    return vegas_odds_old
    
import json
# import ipdb;ipdb.set_trace()
updated_prediction_history = update_prediction_correctness(prediction_history, currentBankroll=300)
#saving the new prediction_history dataframe to json
result = updated_prediction_history.to_json()
parsed = json.loads(result)
prediction_history_filtpath = dh.json_filepaths['prediction_history']

# TODO USE THE SAVE FUNCTION OF THE DATA HANDLER
with open(prediction_history_filtpath, 'w', encoding='utf-8') as jsonf:
    jsonf.write(json.dumps(parsed, indent=4))
    
print(f'saved to {prediction_history_filtpath}')