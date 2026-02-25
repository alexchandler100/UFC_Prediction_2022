from datetime import datetime
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import itertools
import networkx as nx
from unidecode import unidecode
from bs4 import BeautifulSoup
import urllib.request
import requests

from bokeh.plotting import figure, output_file, save
from bokeh.io import show
from bokeh.models import ColumnDataSource
from bokeh.models import FixedTicker
from bokeh.layouts import row, column, Spacer

from sklearn.preprocessing import StandardScaler

alias_array = [
    ['Ian Machado Garry', 'Ian Garry'],
    ['Christian Leroy Duncan', 'Christian Duncan'],
    ['Matheus Nicolau Pereira', 'Matheus Nicolau'],
    ['Katlyn Cerminara', 'Katlyn Chookagian'],
    ['Alatengheili', 'Heili Alateng'],
    ['Montserrat Conejo Ruiz', 'Montserrat Conejo', 'Montserrat Ruiz'],
    ['Tecia Pennington', 'Tecia Torres'],
    ['Mizuki', 'Mizuki Inoue'],
    ['Veronica Hardy', 'Veronica Macedo'],
    ['Daniel Lacerda', 'Daniel Da Silva'],
    ['Yana Santos', 'Yana Kunitskaya'],
    ['Da-Un Jung', 'Da Woon Jung'],
    ['Rayanne dos Santos', 'Rayanne Amanda'],
    ['Billy Ray Goff', 'Billy Goff'],
    ['Joanne Wood', 'Joanne Calderwood'],
    ['King Green', 'Bobby Green'],
    ['Zach Reese', 'Zachary Reese'],
    ['Michelle Waterson-Gomez', 'Michelle Waterson'],
    ['Bruno Korea', 'Bruno Rodrigues'],
    ['Nina Nunes', 'Nina Ansaroff'],
    ['Ariane da Silva', 'Ariane Lipski'],
    ['Brianna Fortino', 'Brianna Van Buren'],
    ['Melissa Mullins', 'Melissa Dixon'],
    ['Rongzhu', 'Rong Zhu', 'Zhu Rong'],
    ['Ricky Glenn', 'Rick Glenn'],
    ['Wulijiburen', 'Wuliji Buren'],
    ['Sumudaerji', 'Su Mudaerji'],
    ['Yizha', 'Yi Zha', 'Zha Yi'],
    ['Asu Almabaev', 'Asu Almabayev', 'Assu Almabayev'],
    ['Hyun-Sung Park', 'Hyun Sung Park', 'HyunSung Park'],
    ['Lupita Godinez', 'Loopy Godinez'],
    ['Baisangur Susurkaev', 'Baysangur Susurkaev'],
    ['Sergey Pavlovich', 'Sergei Pavlovich'],
    ['Zhao Long', 'Long Zhao'],
    ['Soo Young Yoo', 'SuYoung You'],
    ['Andreas Gustafsson Berg', 'Andreas Gustafsson'],
    ['Jafel Cavalcante Filho', 'Jafel Filho'],
    ['Thomas Peterson', 'Thomas Petersen'],
    ['Djorden Santos', 'Diorden Ribeiro'],
    ['Aoriqileng', 'Qileng Aori', 'Aori Qileng'],
    ['Jailton Junior', 'Jailton Almeida'],
    ['Jun Yong Park', 'Junyong Park'],
    ['Cyril Gane', 'Ciryl Gane'],
    ['Themba Takura Gorimbo', 'Themba Gorimbo'],
    ['Yadier DelValle', 'Yadier del Valle'],
    ['Timothy Cuamba', 'Timmy Cuamba'],
    ['ChangHo Lee', 'Chang Ho Lee'],
    ['Seok Hyun Ko', 'Seokhyun Ko', 'Seokhyeon Ko'],
    ['Marco Tulio Silva', 'Marco Tulio'],
    ['Christian Duncan', 'Christian Leroy Duncan'],
    ['Christian Quinonez', 'Cristian Quinonez'],
    ['Jose Mariscal', 'Chepe Mariscal'],
    ['Sergey Spivak', 'Serghei Spivac'],
    ['Dan Hooker', 'Daniel Hooker'],
    ['Ismail Naurdiev', 'Ismael Naurdiev'],
    ['Abdul Rakhman Yakhyaev', 'Abdulrakhman Yakhyaev'],
    ['Rafael Cerquiera', 'Raffael Cerqueira'],
    ['Joshua Van', 'Josh Van'],
    ['Muhammadjon Naimov', 'Muhammad Naimov'],
    ['Melsik Baghdasaryan', 'Melsik Bagdasaryan'],
    ['Natalia Cristina da Silva', 'Natalia Silva'],
    ['Ateba Gautier', 'Ateba Abega Gautier'],
    ['Alex Volkanovski', 'Alexander Volkanovski'],
    ['Talisson Teixeira', 'Tallison Teixeira'],
    ['Sangwook Kim', 'Kim Sang Uk'],
    ['Sulangrangbo', 'Rangbo Sulang'],
    ['Vinicius Oliveira', 'Vinicius de Oliveira'],
    ['Daniil Donchenko', 'Daniel Donchenko'],
    ['Bruna Emanuele Brasil', 'Bruna Brasil'],
    ['Gianni Di Chiara Vazquez', 'Gianni Vazquez'],
    ['Javier Reyes Rugeles', 'Javier Reyes'],
    ['Wesley Schultz', 'Wes Schultz']
    # NOTE LANCE GIBSON has the same name as his father who is also a fighter, so i may have to just go in and manually delete one of them later
]

def regularize_name(name):
    name = name.lower().replace("st.", 'saint').replace(
    " st ", ' saint ').replace("jr.", '').replace('.', '').replace("-", ' ').replace('junior', '').strip()
    name = unidecode(name)  # remove accents
    return name


alias_array_flat_reg = [regularize_name(name) for sublist in alias_array for name in sublist]


def maybe_replace_alias_by_default_name(name, verbose=False):
    if not regularize_name(name) in alias_array_flat_reg:
        return name
    for alias_list in alias_array:
        if any(check_equality_up_to_ordering_and_levenshtein(name, alias_name, verbose=False) for alias_name in alias_list):
            if verbose:
                print(f'Replacing alias name {name} by default name {alias_list[0]}')
            return alias_list[0] 
    return name

def check_equality_up_to_ordering_and_levenshtein(str1, str2, verbose = False):
    if str1 == str2:
        return True
    
    str1 = regularize_name(str1)
    str2 = regularize_name(str2)
    
    if str1 == str2:
        return True
    
    str1_list = str1.split()
    str2_list = str2.split()
    str1_set = set(str1_list)
    str2_set = set(str2_list)
                
    # TODO could be out of order and off by a character
    if str1_set == str2_set:
        if verbose:
            print(str1+' ... (same name as) ... '+str2+' ... (different ordering)')
        return True
    else:
        return False
    

# first call pip install python-Levenshtein
def same_name(str1, str2, verbose=False):
    
    if str1 == str2:
        return True
    
    str1 = regularize_name(str1)
    str2 = regularize_name(str2)
    
    if str1 == str2:
        return True
    
    str1 = maybe_replace_alias_by_default_name(str1, verbose=verbose)
    str2 = maybe_replace_alias_by_default_name(str2, verbose=verbose)

    result = check_equality_up_to_ordering_and_levenshtein(str1, str2, verbose=verbose)
        
    return result
    
same_name_vect = np.vectorize(same_name, otypes=[bool])

# checks if a fighter is in the ufc
# TODO I don't like passing in fighter_stats... find a better pattern
def in_ufc(fighter, fighter_stats):
    for name in fighter_stats['name']:
        if same_name(fighter, name):
            return True
    return False

def time_diff(day1, day2=date.today()):
    if day2 == date.today():
        answer = (day2-datetime.strptime(day1, '%B %d, %Y')).days
    else:
        answer = (datetime.strptime(day2, '%B %d, %Y') -
                  datetime.strptime(day1, '%B %d, %Y')).days
    return answer

# we now vectorize this to use in pandas/numpy
time_diff_vect = np.vectorize(time_diff)

# Methods for cleaning columns in the dataframe
@np.vectorize
def clean_method_for_method_predictions(a):
    if (a == 'KO/TKO'):
        return 'KO/TKO'
    elif (a == 'SUB'):
        return 'SUB'
    elif ((a == 'U-DEC') or (a == 'M-DEC') or (a == 'S-DEC')):
        return 'DEC'
    else:
        return 'bullshit'

@np.vectorize
def clean_method_for_winner_predictions(a):
    if (a == 'KO/TKO'):
        return 'KO/TKO'
    elif (a == 'SUB'):
        return 'SUB'
    elif ((a == 'U-DEC') or (a == 'M-DEC')):
        return 'DEC'
    # counting S-DEC as bullshit!
    else:
        return 'bullshit'

# for computing expectation values
# maybe has overlap with functions in the fight predictor
def odds_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def odds_to_return(odds):
    """Convert American odds to return per $1 bet."""
    if odds > 0:
        return odds / 100
    else:
        return 100 / abs(odds)

def expected_value(fair_odds_a, vegas_odds_a, vegas_odds_b):
    # dont need fair_odds_b since fair odds are equal
    # Convert fair odds to probabilities
    p_a = odds_to_prob(fair_odds_a)
    p_b = 1 - p_a  # assume binary outcome

    # Convert Vegas odds to returns
    r_a = odds_to_return(vegas_odds_a)
    r_b = odds_to_return(vegas_odds_b)

    # Expected value per $1 bet
    ev_a = p_a * r_a - p_b  # betting on Team A
    ev_b = p_b * r_b - p_a  # betting on Team B

    return {
        "fighter_ev": round(ev_a, 4),
        "opponent_ev": round(ev_b, 4),
        "best_bet": "fighter" if ev_a > ev_b and ev_a > 0 else ("opponent" if ev_b > ev_a and ev_b > 0 else "No +EV bet")
    }
    
def odds_to_profit(odds):
    """Convert American odds to profit per $1 bet."""
    if odds > 0:
        return odds / 100
    else:
        return 100 / abs(odds)  # profit + initial bet

def get_kelly_bet_from_ev_and_dk_odds(predicted_fighter_odds, fighter_dk_odds, opponent_dk_odds):
    """
    Calculate the profit from expected value and DraftKings odds.
    
    Parameters:
    - predicted_fighter_odds: Predicted odds for the fighter.
    - dk_odds_fighter: DraftKings odds for the fighter.
    - dk_odds_opponent: DraftKings odds for the opponent.
    
    Returns:
    - fighter_bankroll_percentage: Percentage of bankroll to bet based on Kelly Criterion.
    - opponent_bankroll_percentage: Percentage of bankroll to bet based on Kelly Criterion.
    """
    # TODO ensure ev and odds values exist and are valid
    # first convert predicted odds from American format to a probability 
    predicted_fighter_probability = odds_to_prob(predicted_fighter_odds)

    fighter_b = odds_to_profit(fighter_dk_odds)
    fighter_bankroll_percentage = (predicted_fighter_probability - (1 - predicted_fighter_probability) / fighter_b) * 100

    opponent_b = odds_to_profit(opponent_dk_odds)
    predicted_opponent_probability = 1 - predicted_fighter_probability
    opponent_bankroll_percentage = (predicted_opponent_probability - (1 - predicted_opponent_probability) / opponent_b) * 100

    return fighter_bankroll_percentage, opponent_bankroll_percentage

def bet_payout(american_odds, bet_amount, result):
    """
    Calculate the payout from a bet based on American odds.
    
    Parameters:
    - american_odds: The American odds for the bet.
    - bet_amount: The amount of money bet.
    - result: The result of the bet ('W' or 'L').
    
    Returns:
    - payout: The total payout from the bet.
    """
    american_odds = int(american_odds)
    bet_amount = float(bet_amount)
    assert result in ['W', 'L'], "Result must be 'W' or 'L'."
    if result == 'W':
        if american_odds > 0:
            payout = bet_amount + (bet_amount * american_odds / 100)
        else:
            payout = bet_amount + (bet_amount * 100 / abs(american_odds))
    elif result == 'L':
        payout = 0
    else:
        raise ValueError("Result must be 'W' or 'L'.")
    
    return payout

import re

def feet_inches_to_inches(s):
    # Match patterns like 6' 11", 6'11", or 6ft 11in
    match = re.match(r"\s*(\d+)\s*['ft]+\s*(\d+)\s*[\"]?", s)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return feet * 12 + inches
    else:
        raise ValueError(f"Invalid height format: {s}")
    
def get_fighter_stats(fighter, fighter_stats):
    height = np.nan
    reach = np.nan
    dob = np.nan
    stance_ = 'unknown'
        
    # TODO get rid of for loop and just do the computations directly
    for stat_name in ['height', 'reach', 'dob', 'stance']:
        fighter_loc = same_name_vect(fighter_stats['name'], fighter)
        stat_value = fighter_stats.loc[fighter_loc, stat_name]
        # if there is more than one name match, alert the user
        if len(stat_value) > 1:
            print(f"Warning: Multiple entries found for fighter '{fighter}' in fighter_stats. Cannot resolve. Returning None.")
            return None
        if len(stat_value) == 0:
            import ipdb;ipdb.set_trace(context=10)
            print(f"Warning: No entry found for fighter '{fighter}' in fighter_stats. Returning None.")
            return None
        stat_value = stat_value.iloc[0]  # get the first match
        if stat_name == 'height':
            if isinstance(stat_value, str):
                try:
                    height = feet_inches_to_inches(stat_value)
                except ValueError as e:
                    print(e)
                    height = np.nan
            else:
                print(f"Warning: Height for fighter '{fighter}' is not a string. Returning NaN.")
                height = np.nan
        elif stat_name == 'reach':
            if isinstance(stat_value, str):
                if stat_value == '--' or stat_value == '':
                    reach = np.nan
                try:
                    # remove " if present
                    stat_value = stat_value.replace('"', '')
                    reach = int(stat_value)
                except ValueError as e:
                    print(f"Warning: Reach for fighter '{fighter}' is not a valid integer. Returning NaN.")
                    reach = np.nan
            elif isinstance(stat_value, (int, float)):
                print(f"Warning: Reach for fighter '{fighter}' is a number, not a string. Returning NaN.")
                reach = np.nan
        elif stat_name == 'dob':
            # get date of birth from fighter_stats
            dob_str = fighter_stats.loc[same_name_vect(fighter_stats['name'], fighter), 'dob']
            if len(dob_str) == 0:
                print(f"Warning: No date of birth found for fighter '{fighter}' in fighter_stats. Returning NaN.")
                dob = np.nan
            dob_str = dob_str.iloc[0]
            dob = pd.to_datetime(dob_str, errors='coerce')
            if pd.isna(dob):
                print(f"Warning: Invalid date of birth format for fighter '{fighter}': {dob_str}. Returning NaN.")
        elif stat_name == 'stance':
            if isinstance(stat_value, str):
                if stat_value == '' or stat_value == 'Unknown':
                    stance_ = np.nan
                else:
                    stance_ = stat_value
            else:
                print(f"Warning: Stance for fighter '{fighter}' is not a string. Returning NaN.")
                stance_ = np.nan
    return height, reach, dob, stance_

def count_wins_wins_before_fight(df, fighter, timeframe):
    lxy_pattern = re.compile(r'l\dy')
    assert timeframe == 'all' or re.fullmatch(lxy_pattern, timeframe) , f'timeframe must be all or lky where k is a digit, got {timeframe}' # l2y=(last 2 years), l5y=(last 5 years)
    # make a cumsum column for the given column name, but shifted down by 1 so it doesn't include the current fight
    timeframe_mask = [True]*len(df)  # default to all True
    if not timeframe == 'all':
        # restrict to fights in the given timeframe (the index is a datetime index)
        num_years = int(timeframe[1])  # extract the number of years from the timeframe string
        num_days = num_years * 365  # approximate number of days in the given number of years
        timeframe_mask = (df.index >= (df.index.max() - pd.Timedelta(days=num_days)))
    df = df.reset_index(drop=True)  # ensure df is indexed from 0 to n-1
    opponent_wins = {}
    opponents_whose_wins_count = []
    wins_wins_before = np.zeros(len(df))
    for i in range(len(df)):
        if not timeframe_mask[i]:
            continue
        fighter1 = df['fighter'].iloc[i]
        if not same_name(fighter1, fighter):
            if not fighter1 in opponent_wins:
                opponent_wins[fighter1] = 0
            opponent_wins[fighter1] += 1
            continue
        wins_wins_before[i] = sum([opponent_wins.get(person, 0) for person in opponents_whose_wins_count])
        if df['result'].iloc[i] == 'W':
            fighter2 = df['opponent'].iloc[i]
            opponents_whose_wins_count.append(fighter2)
    return wins_wins_before

def count_losses_losses_before_fight(df, fighter, timeframe):
    lxy_pattern = re.compile(r'l\dy')
    assert timeframe == 'all' or re.fullmatch(lxy_pattern, timeframe) , f'timeframe must be all or lky where k is a digit, got {timeframe}' # l2y=(last 2 years), l5y=(last 5 years)
    # make a cumsum column for the given column name, but shifted down by 1 so it doesn't include the current fight
    timeframe_mask = [True]*len(df)  # default to all True
    if not timeframe == 'all':
        # restrict to fights in the given timeframe (the index is a datetime index)
        num_years = int(timeframe[1])  # extract the number of years from the timeframe string
        num_days = num_years * 365  # approximate number of days in the given number of years
        timeframe_mask = (df.index >= (df.index.max() - pd.Timedelta(days=num_days)))
    df = df.reset_index(drop=True)  # ensure df is indexed from 0 to n-1
    opponent_losses = {}
    opponents_whose_losses_count = []
    losses_losses_before = np.zeros(len(df))
    for i in range(len(df)):
        if not timeframe_mask[i]:
            continue
        fighter1 = df['fighter'].iloc[i]
        if not same_name(fighter1, fighter):
            if not fighter1 in opponent_losses:
                opponent_losses[fighter1] = 0
            opponent_losses[fighter1] += 1
            continue
        losses_losses_before[i] = sum([opponent_losses.get(person, 0) for person in opponents_whose_losses_count])
        if df['result'].iloc[i] == 'L':
            fighter2 = df['opponent'].iloc[i]
            opponents_whose_losses_count.append(fighter2)
    return losses_losses_before


def make_cumsum_before_current_fight(df, col_name, timeframe):
    lxy_pattern = re.compile(r'l\dy')
    assert timeframe == 'all' or re.fullmatch(lxy_pattern, timeframe) , f'timeframe must be all or lky where k is a digit, got {timeframe}' # l2y=(last 2 years), l5y=(last 5 years)
    # make a cumsum column for the given column name, but shifted down by 1 so it doesn't include the current fight
    if timeframe == 'all':
        cumsum_before = df[col_name].cumsum() - df[col_name]
    else:
        num_years = int(timeframe[1])  # extract the number of years from the timeframe string
        num_days = num_years * 365  # approximate number of days in the given number of years
        window = f'{num_days}D'  # create a rolling window string for the given number of days
        cumsum_before = df[col_name].rolling(window=window).sum() - df[col_name]  # rolling sum for the given number of days, shifted down by 1 so it doesn't include the current fight
    return cumsum_before

def make_avg_before_current_fight(df, col_name, timeframe, landed_attempts):
    assert 'total_fight_time' in df.columns, 'df must have a total_fight_time column in minutes to compute averages'
    lxy_pattern = re.compile(r'l\dy')
    assert timeframe == 'all' or re.fullmatch(lxy_pattern, timeframe) , f'timeframe must be all or lky where k is a digit, got {timeframe}' # l2y=(last 2 years), l5y=(last 5 years)
    assert landed_attempts in ['landed', 'attempts', None], f'landed_attempts must be one of landed or attempts, got {landed_attempts}' # landed=landed, attempts=attempts
    total_fight_time = df['total_fight_time'] / 60 # in minutes
    if landed_attempts is not None:
        col_name = f'{col_name}_{landed_attempts}' # e.g. sig_strikes_landed
    # make a cumsum column for the given column name, but shifted down by 1 so it doesn't include the current fight
    if timeframe == 'all':
        cumsum_before = df[col_name].cumsum() - df[col_name]
        time_before = total_fight_time.cumsum() - total_fight_time  # total fight time in minutes before current fight
    else:
        num_years = int(timeframe[1])  # extract the number of years from the timeframe string
        num_days = num_years * 365  # approximate number of days in the given number of years
        window = f'{num_days}D'  # create a rolling window string for the given number of days
        cumsum_before = df[col_name].rolling(window=window).sum() - df[col_name]  # rolling sum for the given number of days, shifted down by 1 so it doesn't include the current fight
        time_before = total_fight_time.rolling(window=window).sum() - total_fight_time  # total fight time in minutes before current fight
    avg_before = cumsum_before / time_before.replace(0, np.nan)  # avoid division by zero
    avg_before = avg_before.fillna(0)  # fill NaN values with 0
    return avg_before


def fight_math(fighter, localized_expanded_df, timeframe):
    lxy_pattern = re.compile(r'l\dy')
    assert timeframe == 'all' or re.fullmatch(lxy_pattern, timeframe) , f'timeframe must be all or lky where k is a digit, got {timeframe}' # l2y=(last 2 years), l5y=(last 5 years)
    # make a cumsum column for the given column name, but shifted down by 1 so it doesn't include the current fight
    num_days = np.inf
    if not timeframe == 'all':
        # restrict to fights in the given timeframe (the index is a datetime index)
        num_years = int(timeframe[1])  # extract the number of years from the timeframe string
        num_days = num_years * 365  # approximate number of days in the given number of years
    edges_that_count = [] # entries look like [date, (fighter, opponent)]
    localized_expanded_df = localized_expanded_df.copy()
    localized_expanded_df.reset_index(drop=False, inplace=True)  # ensure df is indexed from 0 to n-1
    localized_expanded_df[f'{timeframe}-fight_math'] = np.nan
    for i in range(len(localized_expanded_df)):
        fight_date = localized_expanded_df['date'].iloc[i]
        row = localized_expanded_df.iloc[i]
        if same_name(row['fighter'], fighter):
            # TODO This line is including future fights... why?
            fight_math_edges = [entry[1] for entry in edges_that_count if (0 < (fight_date - entry[0]).days < num_days) and not same_name(entry[1][1], fighter)]  # only count edges where the date is before the current fight and within the timeframe, and the opponent is not the fighter
            G = nx.DiGraph()
            G.add_nodes_from([row['fighter'], row['opponent']])
            G.add_edges_from(fight_math_edges)
            num_paths = count_limited_paths(G, row['fighter'], row['opponent'], max_depth=3)
            localized_expanded_df.at[i, f'{timeframe}-fight_math'] = num_paths
            if row['result'] == 'W':
                edges_that_count.append([fight_date, (row['fighter'], row['opponent'])])
        else:
            edges_that_count.append([fight_date, (row['fighter'], row['opponent'])])
    return (localized_expanded_df[f'{timeframe}-fight_math'].fillna(0)).to_numpy()

def count_limited_paths(G, source, target, max_depth=3):
    def dfs(node, depth):
        if depth > max_depth:
            return 0
        if same_name(node, target) and depth > 0:
            return 1
        count = 0
        for neighbor in G.neighbors(node):
            count += dfs(neighbor, depth + 1)
        return count
    return dfs(source, 0)

def model_test_score(X_train, X_test, features, _max_iter = 20000, scaled=True):
    # see how the new features do on the test set we already made
    scaler = StandardScaler()
    y_train = X_train['result']
    y_test = X_test['result']
    
    X_train_scaled = scaler.fit_transform(X_train[features])
    # train the model with the best features

    best_model = LogisticRegression(solver='lbfgs', max_iter=_max_iter, C=0.1, penalty='l2')#, fit_intercept=False)
    best_model.fit(X_train_scaled, y_train)

    # evaluate the model on the training set
    train_score = best_model.score(X_train_scaled, y_train)
    print(f'Training set size: {X_train.shape} accuracy: {train_score}')

    # evaluate the model on the test set
    X_test_scaled = scaler.transform(X_test[features])
    test_score = best_model.score(X_test_scaled, y_test)
    print(f'Test set size: {X_test.shape} accuracy: {test_score}')
    
    # get the neg log loss score of the test set and convert it to a probability
    y_proba_test = best_model.predict_proba(X_test_scaled)
    log_loss = sklearn.metrics.log_loss(y_test, y_proba_test)
    print(f'Test set neg log loss: {-log_loss}. Average probability to observe data given model: {np.exp(-log_loss)}')
    
    return train_score, test_score

#scores a model
def model_score(dataframe, features, iloc_val = 3200, _max_iter = 2000, scoring='neg_log_loss', scaled=True):
    yyy=dataframe['result'].iloc[0:iloc_val]
    XXX=dataframe[features].iloc[0:iloc_val]
    XXXscaler = preprocessing.StandardScaler().fit(XXX)
    XXX_scaled = XXXscaler.transform(XXX) 
    X = XXX_scaled if scaled else XXX
    winPredictionModel=LogisticRegression(solver='lbfgs', max_iter=_max_iter, C=0.1, penalty='l2')#, fit_intercept=False)
    # find the cross val score with log loss
    return cross_val_score(winPredictionModel,X,yyy,cv=4,scoring=scoring).mean()
    
#CODE FOR THE GREEDY ALGORITHM FOR FEATURE SELECTION
def greedy(dataframe, features, subsetsize, iloc_val=3200, _max_iter = 2000, scaled=True, scoring='neg_log_loss', set_of_sets=False):
    if set_of_sets:
        s=set([tuple(feature) for feature in features])
    else:
        s=set(features)
    subsets=list(map(set, itertools.combinations(s, subsetsize))) #subsets of size (subsetsize)
    scores_dict = {}
    for subset in subsets:
        if set_of_sets:
            list_of_features = []
            for feature_set in subset:
                list_of_features.extend(list(feature_set))
        else:
            list_of_features = list(subset)
        scores_dict[tuple(subset)]=model_score(dataframe, list_of_features, iloc_val, _max_iter, scaled=scaled, scoring=scoring)
    max_key = max(scores_dict, key=scores_dict.get)
    max_score = scores_dict[max_key]
    print(f'best subset: {max_key}')
    print(f'with score {max_score}')
    best_feature_list = []
    if set_of_sets:
        for feature_set in max_key:
            best_feature_list.extend(list(feature_set))
    else:
        best_feature_list = list(max_key)
    return best_feature_list, max_score

def reductive_greedy(dataframe, starting_features=None, _max_iter=20000):
    # remove one at a time until it stops going up
    if starting_features is None:
        last_best_score = - 1000
        starting_features = dataframe.columns.tolist()
    else:
        last_best_score = model_score(dataframe, starting_features, _max_iter=_max_iter)
        print(f'Starting with features: {starting_features}')
        print(f'Current best score: {last_best_score}')
    # remove result from list 
    if 'result' in starting_features:
        starting_features.remove('result')
    current_best_score = model_score(dataframe, starting_features, _max_iter=_max_iter)
    features = starting_features.copy()
    while current_best_score > last_best_score:
        last_best_score = current_best_score
        new_features, current_best_score = greedy(dataframe, features, subsetsize=len(features)-1)
        features = new_features
        print(f'Current best score: {current_best_score}')
        print(f'Current best subset: {features}')
        
    print(f'No improvement found, stopping greedy search.')
    best_score = last_best_score
    print(f'Final best score: {best_score}')
    return features

# TODO show test set accuracy at each step as well. need to pass in test set and be careful to not use the test set in the model training.
# NOTE X_train and X_test should be dataframes with the 'result' column in them. (weird but fix it if you want)
def additive_greedy(X_train, X_test, current_best_feature_set=[], search_doubles=False, _max_iter=5000, good_enough_acc=None):
    # now start adding remaining features back in one by one or in groups of two or three and see if we can improve the score
    if current_best_feature_set:
        current_model_score = model_score(X_train, current_best_feature_set, _max_iter=_max_iter)
        current_model_accuracy = model_score(X_train, current_best_feature_set, _max_iter=_max_iter, scoring='accuracy')
        print(f'Starting with current best feature set: {current_best_feature_set}')
        print(f'Current model score: {current_model_score}')
        model_test_score(X_train, X_test, current_best_feature_set, _max_iter = _max_iter, scaled=True)
    else:
        print('Starting with an empty feature set.')
        current_model_score = -1000 
        current_model_accuracy = 0.0
        
    while True:
        if good_enough_acc and current_model_accuracy >= good_enough_acc:
            print(f'Current model accuracy {current_model_accuracy} is above the threshold of {good_enough_acc}. Stopping search.')
            break
        unused_features = set(X_train.columns) - set(current_best_feature_set) - set(['result'])
        # first try adding one feature 
        single_feature_scores = {}
        for feature in unused_features:
            new_feature_set = current_best_feature_set + [feature]
            new_model_score = model_score(X_train, new_feature_set, _max_iter=_max_iter)
            single_feature_scores[tuple(new_feature_set)] = new_model_score
        best_single_feature_set = max(single_feature_scores, key=single_feature_scores.get)
        best_single_feature_score = single_feature_scores[best_single_feature_set]
        if best_single_feature_score > current_model_score:
            current_best_feature_set = list(best_single_feature_set)
            current_model_score = best_single_feature_score
            current_model_accuracy = model_score(X_train, current_best_feature_set, _max_iter=_max_iter, scoring='accuracy')
            print(f'Added single feature: {best_single_feature_set[-1]}')
            print(f'Negative log loss on training set: {current_model_score}')
            model_test_score(X_train, X_test, current_best_feature_set, _max_iter = _max_iter, scaled=True)
            continue
        else:
            print('No improvement found with single addition, trying pairs.')
        if not search_doubles:
            print('Stopping search for pairs (doubles flag set to false in inputs).')
            break
        double_feature_scores = {}
        for feature1, feature2 in itertools.combinations(unused_features, 2):
            new_feature_set = current_best_feature_set + [feature1, feature2]
            new_model_score = model_score(X_train, new_feature_set, _max_iter=_max_iter)
            double_feature_scores[tuple(new_feature_set)] = new_model_score
        best_double_feature_set = max(double_feature_scores, key=double_feature_scores.get)
        best_double_feature_score = double_feature_scores[best_double_feature_set]
        if best_double_feature_score > current_model_score:
            current_best_feature_set = list(best_double_feature_set)
            current_model_score = best_double_feature_score
            current_model_accuracy = model_score(X_train, current_best_feature_set, _max_iter=_max_iter, scoring='accuracy')

            print(f'Added double feature: {best_double_feature_set[-2]}, {best_double_feature_set[-1]}')
            print(f'Negative log loss on training set: {current_model_score}')
            model_test_score(X_train, X_test, current_best_feature_set, _max_iter = _max_iter, scaled=True)

            continue
        else:
            print('No improvement found with double addition, stopping.')
            break

    print(f'Final best feature set: {current_best_feature_set}')
    return current_best_feature_set

# FUNCTIONS THAT INTERACT WITH WEBSITES

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
        fight_card = pd.concat([fight_card, fight_data_dict], axis=0)
        
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


def visualize_prediction_bokeh(fighter, opponent, theta, card_date, derived_doubled_tuple, diff_tup_scaled):
    df = diff_tup_scaled.T
    df.columns = ['diff_value_scaled']
    # import ipdb;ipdb.set_trace(context=10)
    fighter_values = derived_doubled_tuple.iloc[0,:].T
    opponent_values = derived_doubled_tuple.iloc[1,:].T
    df['fighter_value'] = fighter_values.values
    df['opponent_value'] = opponent_values.values
    df['diff'] = df['fighter_value'] - df['opponent_value']
    df['theta'] = theta
    # df['abs_theta'] = df['theta'].abs()
    df['contribution'] = df['diff_value_scaled'] * df['theta']
    df['abs_contribution'] = df['contribution'].abs()
    # sort by contribution
    df = df.sort_values(by='abs_contribution', ascending=False)
    source = ColumnDataSource(data=dict(index=df.index, contribution=df['contribution']))

    # Prepare output file
    # convert card date from August 02, 2025 to 2025-08-02
    card_date_formatted = pd.to_datetime(card_date).strftime('%Y-%m-%d')
    # convert fighter and opponent to lowercase and replace spaces with underscores
    html_str = f'content/bokehPlots/{card_date_formatted}_{fighter.lower().replace(" ", "_")}_vs_{opponent.lower().replace(" ", "_")}_bokeh_barplot.html'
    output_file(html_str)

    # Create the figure
    p = figure(y_range=list(df.index), width=1000, height=600, title=f"{opponent} --vs-- {fighter}",
            toolbar_location=None, tools="")
    
    # color bars red when theta is negative, green when positive
    colors = ['#d62728' if val < 0 else '#2ca02c' for val in df['theta']]
    source.data['color'] = colors

    # Draw horizontal bars
    p.hbar(y='index', right='contribution', height=0.8, source=source, color='color')

    # Style the plot
    # make y grid lines for each category
    # p.ygrid.grid_line_color = None
    p.ygrid.grid_line_alpha = 0.3
    p.xgrid.grid_line_alpha = 0.3
    # make more vertical grid lines
    # p.xgrid.ticker = FixedTicker(ticks=np.arange(-10, 11, 1).tolist())
    p.xaxis.axis_label = "Contribution to log-odds"
    # p.x_range.start = min(df['contribution']) - 1
    # p.x_range.end = max(df['contribution']) + 1
    
    # make another bar plot below this one that shows the theta values
    # now sort by theta
    df = df.sort_values(by='theta', ascending=False)
    source2 = ColumnDataSource(data=dict(index=df.index, theta=df['theta']))
    p2 = figure(y_range=list(df.index), width=1000, height=600, title=f"Theta values for {opponent} and {fighter}",
            toolbar_location=None, tools="")
    p2.hbar(y='index', right='theta', height=0.8, source=source2)
    p2.xaxis.axis_label = "Theta values"
    p2.ygrid.grid_line_alpha = 0.3
    p2.xgrid.grid_line_alpha = 0.3
    
    # make another figure that contains the df as a table
    from bokeh.models import DataTable, TableColumn
    columns = [
        TableColumn(field="index", title="Feature"),
        TableColumn(field="fighter_value", title=f"{fighter} Value"),
        TableColumn(field="opponent_value", title=f"{opponent} Value"),
        TableColumn(field="diff_value_scaled", title="Diff (scaled)"),
        TableColumn(field="theta", title="Theta"),
        TableColumn(field="contribution", title="Contribution"),
        TableColumn(field="abs_contribution", title="Abs Contribution"),
    ]
    
            
    # for each feature, if the name ends in _diff, remove the _diff for display purposes
    df.index = [name[:-5] if name.endswith('_diff') else name for name in df.index]
    
    data_table_source = ColumnDataSource(data=dict(
        index=df.index,
        fighter_value=df['fighter_value'],
        opponent_value=df['opponent_value'],
        diff_value_scaled=df['diff_value_scaled'],
        theta=df['theta'],
        contribution=df['contribution'],
        abs_contribution=df['abs_contribution'],
    ))
    data_table = DataTable(source=data_table_source, columns=columns, width=1000, height=400, index_position=None)#, autosize_mode='fit_columns')
    # add columns sums to data table 
    sums = {
        'index': 'Total',
        'fighter_value': df['fighter_value'].sum(),
        'opponent_value': df['opponent_value'].sum(),
        'diff_value_scaled': df['diff_value_scaled'].sum(),
        'theta': df['theta'].sum(),
        'contribution': df['contribution'].sum(),
        'abs_contribution': df['abs_contribution'].sum(),
    }
    sums_source = ColumnDataSource(data=dict(
        index=[sums['index']],
        fighter_value=[sums['fighter_value']],
        opponent_value=[sums['opponent_value']],
        diff_value_scaled=[sums['diff_value_scaled']],
        theta=[sums['theta']],
        contribution=[sums['contribution']],
        abs_contribution=[sums['abs_contribution']],
    ))
    # add a row to the data table for the sums
    from bokeh.models import Div
    div = Div(text=f"<b>Total Contribution: {sums['contribution']:.4f}</b>", width=1000, height=30)
    data_table_source.stream(sums_source.data)
    
    # format table so all floats have 3 decimal places
    from bokeh.models import NumberFormatter
    for col in columns:
        if col.field in ['fighter_value', 'opponent_value', 'diff_value_scaled', 'theta', 'contribution', 'abs_contribution']:
            col.formatter = NumberFormatter(format="0.0000")
            
    # fit columns to width of widest content 
    # data_table.width = 1500
    # data_table.height = 400
    

    
    
    
    
    # p3 = figure(width=1000, height=400, title="Feature Contributions Table")
    # p3.add_layout(data_table)
    # ValueError: failed to validate figure(id='p1103', ...).center: expected an element of List(Instance(Renderer)), got seq with invalid items [DataTable(id='p1097', ...)] 
    # p3.toolbar.active_drag = None
    # p3.toolbar.active_scroll = None
    # p3.toolbar.active_tap = None
    # p3.xgrid.grid_line_color = None
    # p3.ygrid.grid_line_color = None
    # p3.axis.visible = False
    # p3.outline_line_color = None
    

    # Save to HTML at f'{git_root}/src/content/bokehPlots/
    save(column(p, Spacer(height=20), p2, Spacer(height=20), data_table))
    print(f'Bokeh plot saved to {html_str}')
    
def get_canonical_name(name, fighter_stats):
    name_mask = same_name_vect(name, fighter_stats['name'].to_numpy())
    if not np.any(name_mask):
        print(f'Fighter name {name} not found in fighter stats database.')
        return name
    canonical_name = fighter_stats['name'].to_numpy()[name_mask][0]
    return canonical_name