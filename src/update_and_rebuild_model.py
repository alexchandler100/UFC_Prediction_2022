#getting dependencies
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import random, sklearn, scipy, json, itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
#this imports all of the functions from the file functions.py
from functions import *

pd.options.mode.chained_assignment = None  # default='warn' (disables SettingWithCopyWarning)

print('importing dataframe from ufc_fights.csv')
#importing csv fight data and saving as dataframes
ufc_fights_winner = pd.read_csv('models/buildingMLModel/data/processed/ufc_fights.csv',low_memory=False)
ufc_fights_method = pd.read_csv('models/buildingMLModel/data/processed/ufc_fights.csv',low_memory=False)
ufcfighterscrap =pd.read_csv('models/buildingMLModel/data/processed/fighter_stats.csv',sep=',',low_memory=False)

#there are some issues with how names are saved
#it gets saved as Joanne Wood for some reason
for i in range(len(ufc_fights_winner['fighter'])):
    if ufc_fights_winner['fighter'][i]=='Joanne Wood':
        ufc_fights_winner['fighter'][i]='Joanne Calderwood'
    if ufc_fights_winner['opponent'][i]=='Joanne Wood':
        ufc_fights_winner['opponent'][i]='Joanne Calderwood'

#there are two lists of names which sometimes get scraped differently... this causes some problems which are being
#addressed in this cell.

#cleaning the methods column for winner prediction
#changing anything other than 'U-DEC','M-DEC', 'KO/TKO', 'SUB', to 'bullshit'
#changing 'U-DEC','M-DEC', to 'DEC'
#(counting split decisions as bullshit)
ufc_fights_winner['method'] = clean_method_for_winner_vect(ufc_fights_winner['method'])
ufc_fights_winner['method'].unique()

#cleaning the methods column for method prediction
#changing anything other than 'U-DEC','M-DEC', 'S-DEC', 'KO/TKO', 'SUB', to 'bullshit'
#changing 'U-DEC','M-DEC', 'S-DEC', to 'DEC'
#(counting split decisions as decisions)
ufc_fights_method['method'] = clean_method_vect(ufc_fights_method['method'])
ufc_fights_method['method'].unique()

#getting rid of rows with incomplete or useless data
#fights with outcome "Win" or "Loss" (no "Draw")
draw_mask=ufc_fights_winner['result'] != 'D'
#fights where the method of victory is TKO/SUB/DEC (no split decision or DQ or Overturned or anything else like that)
method_mask_winner=(ufc_fights_winner['method']!='bullshit')
method_mask_method=(ufc_fights_method['method']!='bullshit')
#fights where age is known
age_mask=(ufc_fights_winner['fighter_age']!='unknown')&(ufc_fights_winner['opponent_age']!='unknown')&(ufc_fights_winner['fighter_age']!=0)&(ufc_fights_winner['opponent_age']!=0)
#fights where height reach is known
height_mask=(ufc_fights_winner['fighter_height']!='unknown')&(ufc_fights_winner['opponent_height']!='unknown')
reach_mask=(ufc_fights_winner['fighter_reach']!='unknown')&(ufc_fights_winner['opponent_reach']!='unknown')
#fights where number of wins is known
wins_mask=(ufc_fights_winner['fighter_wins'] != 'unknown' )& (ufc_fights_winner['opponent_wins'] != 'unknown')
#fights where both fighters have strike statistics (gets rid of UFC debuts)
strikes_mask=(ufc_fights_winner['fighter_inf_sig_strikes_attempts_avg'] != 0)&(ufc_fights_winner['opponent_inf_sig_strikes_attempts_avg'] != 0)
#includes only the fights satisfying these conditions
ufc_fights_winner=ufc_fights_winner[draw_mask&method_mask_winner&age_mask&height_mask&reach_mask&wins_mask&strikes_mask]
ufc_fights_method=ufc_fights_method[draw_mask&method_mask_method&age_mask&height_mask&reach_mask&wins_mask&strikes_mask]
ufc_fights=ufc_fights[draw_mask&method_mask_winner&age_mask&height_mask&reach_mask&wins_mask&strikes_mask]

#listing all stats and making some new stats from them (differences often score higher in the learning models)
record_statistics=[u'fighter_wins',u'fighter_losses',u'fighter_L5Y_wins',u'fighter_L5Y_losses',u'fighter_L2Y_wins',u'fighter_L2Y_losses',u'fighter_ko_wins',u'fighter_ko_losses',u'fighter_L5Y_ko_wins',u'fighter_L5Y_ko_losses',u'fighter_L2Y_ko_wins',u'fighter_L2Y_ko_losses',u'fighter_sub_wins',u'fighter_sub_losses',u'fighter_L5Y_sub_wins',u'fighter_L5Y_sub_losses',u'fighter_L2Y_sub_wins',u'fighter_L2Y_sub_losses',u'opponent_wins',u'opponent_losses',u'opponent_L5Y_wins',u'opponent_L5Y_losses',u'opponent_L2Y_wins',u'opponent_L2Y_losses',u'opponent_ko_wins',u'opponent_ko_losses',u'opponent_L5Y_ko_wins',u'opponent_L5Y_ko_losses',u'opponent_L2Y_ko_wins',u'opponent_L2Y_ko_losses',u'opponent_sub_wins',u'opponent_sub_losses',u'opponent_L5Y_sub_wins',u'opponent_L5Y_sub_losses',u'opponent_L2Y_sub_wins',u'opponent_L2Y_sub_losses']

physical_stats=[ u'fighter_age',u'fighter_height',u'fighter_reach',u'opponent_age',u'opponent_height',u'opponent_reach']

#THERE MAY BE A PROBLEM IN AGE HEIGHT REACH TO DO WITH STRING VS FLOAT. MAKE SURE THESE ARE ALL THE CORRECT TYPE
#MAYBE WE ARE LOSING PREDICTABILITY HERE

#here is the list of all stats available (besides stance), does not include names or result
punch_statistics=[  u'fighter_inf_knockdowns_avg',u'fighter_inf_pass_avg',u'fighter_inf_reversals_avg',u'fighter_inf_sub_attempts_avg',u'fighter_inf_takedowns_landed_avg',u'fighter_inf_takedowns_attempts_avg',u'fighter_inf_sig_strikes_landed_avg',u'fighter_inf_sig_strikes_attempts_avg',u'fighter_inf_total_strikes_landed_avg',u'fighter_inf_total_strikes_attempts_avg',u'fighter_inf_head_strikes_landed_avg',u'fighter_inf_head_strikes_attempts_avg',u'fighter_inf_body_strikes_landed_avg',u'fighter_inf_body_strikes_attempts_avg',u'fighter_inf_leg_strikes_landed_avg',u'fighter_inf_leg_strikes_attempts_avg',u'fighter_inf_distance_strikes_landed_avg',u'fighter_inf_distance_strikes_attempts_avg',u'fighter_inf_clinch_strikes_landed_avg',u'fighter_inf_clinch_strikes_attempts_avg',u'fighter_inf_ground_strikes_landed_avg',u'fighter_inf_ground_strikes_attempts_avg',u'fighter_abs_knockdowns_avg',u'fighter_abs_pass_avg',u'fighter_abs_reversals_avg',u'fighter_abs_sub_attempts_avg',u'fighter_abs_takedowns_landed_avg',u'fighter_abs_takedowns_attempts_avg',u'fighter_abs_sig_strikes_landed_avg',u'fighter_abs_sig_strikes_attempts_avg',u'fighter_abs_total_strikes_landed_avg',u'fighter_abs_total_strikes_attempts_avg',u'fighter_abs_head_strikes_landed_avg',u'fighter_abs_head_strikes_attempts_avg',u'fighter_abs_body_strikes_landed_avg',u'fighter_abs_body_strikes_attempts_avg',u'fighter_abs_leg_strikes_landed_avg',u'fighter_abs_leg_strikes_attempts_avg',u'fighter_abs_distance_strikes_landed_avg',u'fighter_abs_distance_strikes_attempts_avg',u'fighter_abs_clinch_strikes_landed_avg',u'fighter_abs_clinch_strikes_attempts_avg',u'fighter_abs_ground_strikes_landed_avg',u'fighter_abs_ground_strikes_attempts_avg',u'opponent_inf_knockdowns_avg',u'opponent_inf_pass_avg',u'opponent_inf_reversals_avg',u'opponent_inf_sub_attempts_avg',u'opponent_inf_takedowns_landed_avg',u'opponent_inf_takedowns_attempts_avg',u'opponent_inf_sig_strikes_landed_avg',u'opponent_inf_sig_strikes_attempts_avg',u'opponent_inf_total_strikes_landed_avg',u'opponent_inf_total_strikes_attempts_avg',u'opponent_inf_head_strikes_landed_avg',u'opponent_inf_head_strikes_attempts_avg',u'opponent_inf_body_strikes_landed_avg',u'opponent_inf_body_strikes_attempts_avg',u'opponent_inf_leg_strikes_landed_avg',u'opponent_inf_leg_strikes_attempts_avg',u'opponent_inf_distance_strikes_landed_avg',u'opponent_inf_distance_strikes_attempts_avg',u'opponent_inf_clinch_strikes_landed_avg',u'opponent_inf_clinch_strikes_attempts_avg',u'opponent_inf_ground_strikes_landed_avg',u'opponent_inf_ground_strikes_attempts_avg',u'opponent_abs_knockdowns_avg',u'opponent_abs_pass_avg',u'opponent_abs_reversals_avg',u'opponent_abs_sub_attempts_avg',u'opponent_abs_takedowns_landed_avg',u'opponent_abs_takedowns_attempts_avg',u'opponent_abs_sig_strikes_landed_avg',u'opponent_abs_sig_strikes_attempts_avg',u'opponent_abs_total_strikes_landed_avg',u'opponent_abs_total_strikes_attempts_avg',u'opponent_abs_head_strikes_landed_avg',u'opponent_abs_head_strikes_attempts_avg',u'opponent_abs_body_strikes_landed_avg',u'opponent_abs_body_strikes_attempts_avg',u'opponent_abs_leg_strikes_landed_avg',u'opponent_abs_leg_strikes_attempts_avg',u'opponent_abs_distance_strikes_landed_avg',u'opponent_abs_distance_strikes_attempts_avg',u'opponent_abs_clinch_strikes_landed_avg',u'opponent_abs_clinch_strikes_attempts_avg',u'opponent_abs_ground_strikes_landed_avg',u'opponent_abs_ground_strikes_attempts_avg']

#here is the version of punch stats geared for comparing fighter_inf to opponent_abs
punch_statistics_alt=[    u'fighter_inf_knockdowns_avg',u'fighter_inf_pass_avg',u'fighter_inf_reversals_avg',u'fighter_inf_sub_attempts_avg',u'fighter_inf_takedowns_landed_avg',u'fighter_inf_takedowns_attempts_avg',u'fighter_inf_sig_strikes_landed_avg',u'fighter_inf_sig_strikes_attempts_avg',u'fighter_inf_total_strikes_landed_avg',u'fighter_inf_total_strikes_attempts_avg',u'fighter_inf_head_strikes_landed_avg',u'fighter_inf_head_strikes_attempts_avg',u'fighter_inf_body_strikes_landed_avg',u'fighter_inf_body_strikes_attempts_avg',u'fighter_inf_leg_strikes_landed_avg',u'fighter_inf_leg_strikes_attempts_avg',u'fighter_inf_distance_strikes_landed_avg',u'fighter_inf_distance_strikes_attempts_avg',u'fighter_inf_clinch_strikes_landed_avg',u'fighter_inf_clinch_strikes_attempts_avg',u'fighter_inf_ground_strikes_landed_avg',u'fighter_inf_ground_strikes_attempts_avg',
u'fighter_abs_knockdowns_avg',u'fighter_abs_pass_avg',u'fighter_abs_reversals_avg',u'fighter_abs_sub_attempts_avg',u'fighter_abs_takedowns_landed_avg',u'fighter_abs_takedowns_attempts_avg',u'fighter_abs_sig_strikes_landed_avg',u'fighter_abs_sig_strikes_attempts_avg',u'fighter_abs_total_strikes_landed_avg',u'fighter_abs_total_strikes_attempts_avg',u'fighter_abs_head_strikes_landed_avg',u'fighter_abs_head_strikes_attempts_avg',u'fighter_abs_body_strikes_landed_avg',u'fighter_abs_body_strikes_attempts_avg',u'fighter_abs_leg_strikes_landed_avg',u'fighter_abs_leg_strikes_attempts_avg',u'fighter_abs_distance_strikes_landed_avg',u'fighter_abs_distance_strikes_attempts_avg',u'fighter_abs_clinch_strikes_landed_avg',u'fighter_abs_clinch_strikes_attempts_avg',u'fighter_abs_ground_strikes_landed_avg',u'fighter_abs_ground_strikes_attempts_avg',
u'opponent_abs_knockdowns_avg',u'opponent_abs_pass_avg',u'opponent_abs_reversals_avg',u'opponent_abs_sub_attempts_avg',u'opponent_abs_takedowns_landed_avg',u'opponent_abs_takedowns_attempts_avg',u'opponent_abs_sig_strikes_landed_avg',u'opponent_abs_sig_strikes_attempts_avg',u'opponent_abs_total_strikes_landed_avg',u'opponent_abs_total_strikes_attempts_avg',u'opponent_abs_head_strikes_landed_avg',u'opponent_abs_head_strikes_attempts_avg',u'opponent_abs_body_strikes_landed_avg',u'opponent_abs_body_strikes_attempts_avg',u'opponent_abs_leg_strikes_landed_avg',u'opponent_abs_leg_strikes_attempts_avg',u'opponent_abs_distance_strikes_landed_avg',u'opponent_abs_distance_strikes_attempts_avg',u'opponent_abs_clinch_strikes_landed_avg',u'opponent_abs_clinch_strikes_attempts_avg',u'opponent_abs_ground_strikes_landed_avg',u'opponent_abs_ground_strikes_attempts_avg',
u'opponent_inf_knockdowns_avg',u'opponent_inf_pass_avg',u'opponent_inf_reversals_avg',u'opponent_inf_sub_attempts_avg',u'opponent_inf_takedowns_landed_avg',u'opponent_inf_takedowns_attempts_avg',u'opponent_inf_sig_strikes_landed_avg',u'opponent_inf_sig_strikes_attempts_avg',u'opponent_inf_total_strikes_landed_avg',u'opponent_inf_total_strikes_attempts_avg',u'opponent_inf_head_strikes_landed_avg',u'opponent_inf_head_strikes_attempts_avg',u'opponent_inf_body_strikes_landed_avg',u'opponent_inf_body_strikes_attempts_avg',u'opponent_inf_leg_strikes_landed_avg',u'opponent_inf_leg_strikes_attempts_avg',u'opponent_inf_distance_strikes_landed_avg',u'opponent_inf_distance_strikes_attempts_avg',u'opponent_inf_clinch_strikes_landed_avg',u'opponent_inf_clinch_strikes_attempts_avg',u'opponent_inf_ground_strikes_landed_avg',u'opponent_inf_ground_strikes_attempts_avg']

#adding record differences to ufc_fights
record_statistics_diff = []
half_length=int(len(record_statistics)/2)
for i in range(half_length):
    ufc_fights_winner[record_statistics[i]+'_diff_2']=ufc_fights_winner[record_statistics[i]]-ufc_fights_winner[record_statistics[i+half_length]]
    ufc_fights_method[record_statistics[i]+'_diff_2']=ufc_fights_method[record_statistics[i]]-ufc_fights_method[record_statistics[i+half_length]]
    record_statistics_diff.append(record_statistics[i]+'_diff_2')

#lets try and improve the greedy algorithm by considering differences. Lets start by replacing height and reach by their differences
ufc_fights_winner['height_diff']=ufc_fights_winner['fighter_height'].apply(float)-ufc_fights_winner['opponent_height'].apply(float)
ufc_fights_winner['reach_diff']=ufc_fights_winner['fighter_reach'].apply(float)-ufc_fights_winner['opponent_reach'].apply(float)
ufc_fights_method['height_diff']=ufc_fights_method['fighter_height'].apply(float)-ufc_fights_method['opponent_height'].apply(float)
ufc_fights_method['reach_diff']=ufc_fights_method['fighter_reach'].apply(float)-ufc_fights_method['opponent_reach'].apply(float)

physical_stats_diff = ['fighter_age_diff', 'height_diff', 'reach_diff']

#adding punch differences to ufc_fights
punch_statistics_diff = []
half_length=int(len(punch_statistics)/2)
for i in range(half_length):
    ufc_fights_method[punch_statistics[i]+'_diff_2']=ufc_fights_method[punch_statistics[i]]-ufc_fights_method[punch_statistics[i+half_length]]
    ufc_fights_winner[punch_statistics[i]+'_diff_2']=ufc_fights_winner[punch_statistics[i]]-ufc_fights_winner[punch_statistics[i+half_length]]
    punch_statistics_diff.append(punch_statistics[i]+'_diff_2')

possible_stats =record_statistics_diff+physical_stats_diff+punch_statistics_diff

#setting
ufc_fights_winner['fighter_age'] = ufc_fights_winner['fighter_age'].apply(float)
ufc_fights_winner['opponent_age'] = ufc_fights_winner['opponent_age'].apply(float)
ufc_fights_winner['fighter_age_diff'] = ufc_fights_winner['fighter_age']-ufc_fights_winner['opponent_age']

best_smallest_set = ['4-fighter_score_diff',
 '9-fighter_score_diff',
 '15-fighter_score_diff',
 '1-fight_math',
 '6-fight_math',
 'fighter_L5Y_sub_wins_diff_2',
 'fighter_L5Y_losses_diff_2',
 'fighter_L5Y_ko_losses_diff_2',
 'fighter_age_diff',
 'fighter_abs_total_strikes_landed_avg_diff_2',
 'fighter_abs_head_strikes_landed_avg_diff_2',
 'fighter_inf_ground_strikes_landed_avg_diff_2',
 'fighter_inf_takedowns_attempts_avg_diff_2',
 'fighter_inf_head_strikes_landed_avg_diff_2',
 ]

ufc_fights_df = ufc_fights_winner[best_smallest_set]

#decided to force intercept to be 0 due to symmetry of dataset (all stats are differences so if we switch fighters, we must get the negative of the result)
winPredictionModel=LogisticRegression(solver='lbfgs', max_iter=2000, fit_intercept=False)
X=ufc_fights_df.iloc[0:40*55].to_numpy()
y=ufc_fights_winner['result'].iloc[0:40*55]
print('Fitting Logistic Regression Model')
winPredictionModel.fit(X,y)
print('Accuracy of Logistic Regression win prediction: '+str(cross_val_score(winPredictionModel,X,y,cv=3).mean()))
print('coefficients'+str(winPredictionModel.coef_))
print('intercept'+str(winPredictionModel.intercept_))
theta = winPredictionModel.coef_
b = winPredictionModel.intercept_[0]


# We want to predict how many times out of 10 the winning fighter would win, so we look at the values
# x*theta+b. If the value is >=0 its a win and <=0 its a loss. But how far from zero gives indication of
# how likely the outcome is.

def presigmoid_value(fighter1,fighter2,date1,date2):
    value = 0
    tup = ufc_prediction_tuple(fighter1,fighter2,date1,date2)
    for i in range(len(tup)):
        value += tup[i]*theta[i]
    return value + b

def manual_prediction(fighter1,fighter2,date1,date2):
    value = presigmoid_value(fighter1,fighter2,date1,date2)
    value2 = presigmoid_value(fighter2,fighter1,date2,date1)
    return value-value2>=0

import math

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

#returns the probability that fighter1 defeats fighter2 on date1,date2
def probability(fighter1,fighter2,date1,date2):
    presig=presigmoid_value(fighter1,fighter2,date1,date2)
    return sigmoid(presig)

def odds(fighter1,fighter2):
    date1=(date.today()- relativedelta(months=1)).strftime("%B %d, %Y")
    date2=(date.today()- relativedelta(months=1)).strftime("%B %d, %Y")
    p=probability(fighter1,fighter2,date1,date2)
    if p<.5:
        fighterOdds=round(100/p - 100)
        opponentOdds = round(1 / (1 / (1 - p) - 1) * 100)
        return ['+'+str(fighterOdds),'-'+str(opponentOdds)]
    elif p>=.5:
        fighterOdds = round(1 / (1 / p - 1) * 100)
        opponentOdds = round(100 / (1 - p) - 100)
        return ['-'+str(fighterOdds),'+'+str(opponentOdds)]

def give_odds(fighter1,fighter2,date1,date2):
    value = presigmoid_value(fighter1,fighter2,date1,date2)
    value2 = presigmoid_value(fighter2,fighter1,date2,date1)
    if value-value2>=0:
        winner=fighter1
    else:
        winner=fighter2
    value2 = presigmoid_value(fighter2,fighter1,date2,date1)
    abs_value = (abs(value)+abs(value2))/2
    if abs_value >=0 and abs_value <=.2:
        print(winner+" wins a little over 5 times out of 10 times.")
    elif abs_value >=.2 and abs_value <=.4:
        print(winner+" wins 6 out of 10 times.")
    elif abs_value >=.4 and abs_value <=.6:
        print(winner+" wins 7 out of 10 times.")
    elif abs_value >=.6 and abs_value <=.8:
        print(winner+" wins 9 out of 10 times.")
    elif abs_value >=.8:
        print(winner+" wins 10 out of 10 times.")

theta = list(theta[0])

theta_dict = {}
for i in range(len(theta)):
    theta_dict[i]=theta[i]

with open('models/buildingMLModel/data/external/theta.json', 'w') as outfile:
    json.dump(theta_dict, outfile)

intercept_dict = {0:b}

with open('models/buildingMLModel/data/external/intercept.json', 'w') as outfile:
    json.dump(intercept_dict, outfile)
    
print("Saved new theta and intercept to json files to run model in website")

#I've defined this in such a way to predict what happens when fighter1 in their day1 version fights fighter2
#in their day2 version. Meaning we could compare for example 2014 Tyron Woodley to 2019 Colby Covington
def ufc_prediction_tuple(fighter1,fighter2,day1=date.today(),day2=date.today()):
    return [fighter_score_diff(fighter1,fighter2,day1, 4),
            fighter_score_diff(fighter1,fighter2,day1, 9),
            fighter_score_diff(fighter1,fighter2,day1, 15),
            fight_math_diff(fighter1,fighter2,day1, 1),
            fight_math_diff(fighter1,fighter2,day1, 6),
            L5Y_sub_wins(fighter1,day1)-L5Y_sub_wins(fighter2,day2),
            L5Y_losses(fighter1,day1)-L5Y_losses(fighter2,day2),
            L5Y_ko_losses(fighter1,day1)-L5Y_ko_losses(fighter2,day2),
            fighter_age(fighter1,day1)-fighter_age(fighter2,day2),
            avg_count('total_strikes_landed',fighter1,'abs',day1)-avg_count('total_strikes_landed',fighter2,'abs',day2),
            avg_count('head_strikes_landed',fighter1,'abs',day1)-avg_count('head_strikes_landed',fighter2,'abs',day2),
            avg_count('ground_strikes_landed',fighter1,'inf',day1)-avg_count('ground_strikes_landed',fighter2,'inf',day2),
            avg_count('takedowns_attempts',fighter1,'inf',day1)-avg_count('takedowns_attempts',fighter2,'inf',day2),
            avg_count('head_strikes_landed',fighter1,'inf',day1)-avg_count('head_strikes_landed',fighter2,'inf',day2),
           ]

#print('scraping bookie odds from bestfightodds.com')
#odds_df = get_odds()
#odds_df=drop_irrelevant_fights(odds_df,3) #allows 3 bookies to have missing odds. can increase this to 2 or 3 as needed
#odds_df=drop_non_ufc_fights(odds_df)
#odds_df=drop_repeats(odds_df)
#print('saving odds to models/buildingMLModel/data/external/vegas_odds.json')
#save to json
#result = odds_df.to_json()
#parsed = json.loads(result)
#jsonFilePath='models/buildingMLModel/data/external/vegas_odds.json'
#with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
#    jsonf.write(json.dumps(parsed, indent=4))
#print('saved to '+jsonFilePath)

print('Saving results of previous card to prediction_history.json')

vegas_odds_old=pd.read_json('models/buildingMLModel/data/external/vegas_odds.json')
ufc_fights_crap = pd.read_csv('models/buildingMLModel/data/processed/ufc_fights_crap.csv',low_memory=False)

# getting rid of fights that didn't actually happen and adding correctness results of those that did
bad_indices = []
for index1, row1 in vegas_odds_old.iterrows():
    card_date = row1['date']
    relevant_fights = ufc_fights_crap[pd.to_datetime(ufc_fights_crap['date']) == card_date]
    print(f'searching through {relevant_fights.shape[0]//2} confirmed fights on {str(card_date).split(" ")[0]} for {row1["fighter name"]} vs {row1["opponent name"]}')
    fighter_odds = row1['predicted fighter odds']
    match_found = False
    # if no prediction was made, throw it away
    if fighter_odds == '':
        bad_indices.append(index1)
        print('no prediction made for fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'])
    else: # if a prediction was made, check if the fight actually happened and then check if the prediction was correct
        for index2, row2 in relevant_fights.iterrows():
            if same_name(row1['fighter name'], row2['fighter']) and same_name(row1['opponent name'], row2['opponent']):
                match_found = True
                print('adding fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'])
                if (int(fighter_odds) < 0 and row2['result'] == 'W') or (int(fighter_odds) > 0 and row2['result'] == 'L'):
                    vegas_odds_old.at[index1,'correct?'] = 1
                else:
                    vegas_odds_old.at[index1,'correct?'] = 0
                # TODO add case for draw
                break
        if not match_found: # if the fight didn't happen, throw it away
            bad_indices.append(index1)
            print('fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'] + ' not found in ufc_fights_crap.csv')
# drop bad indices
vegas_odds_old = vegas_odds_old.drop(bad_indices)

#making a copy of vegas_odds
vegas_odds_copy=vegas_odds_old.copy()
prediction_history=pd.read_json('models/buildingMLModel/data/external/prediction_history.json')
#for col in ['predicted fighter odds','predicted opponent odds','average bookie odds','correct?']:
#    vegas_odds_copy[col]=""

#add the newly scraped fights and predicted fights to the history of prediction list (idea: might be better to wait to join until after the fights happen)
prediction_history = pd.concat([vegas_odds_copy, prediction_history], axis = 0).reset_index(drop=True)

#saving the new prediction_history dataframe to json
result = prediction_history.to_json()
parsed = json.loads(result)
jsonFilePath='models/buildingMLModel/data/external/prediction_history.json'
with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
    jsonf.write(json.dumps(parsed, indent=4))
print('saved to '+jsonFilePath)

print('Scraping next ufc fight card from bestfightodds.com')
print("###############################################################################################################")
card_date, card_title, fights_list = get_next_fight_card()
card_date = convert_scraped_date_to_standard_date(card_date)

card_info_dict = {"date":card_date, "title":card_title}

print('Writing upcoming card info to models/buildingMLModel/data/external/card_info.json')
with open('models/buildingMLModel/data/external/card_info.json', 'w') as outfile:
    json.dump(card_info_dict, outfile)
print("###############################################################################################################")


#now we build prediction_history
#next open existing prediction_history.json as a dataframe (done)... then add all fights on the books from vegas_odds
#for each fight in prediction_history, if a prediction has not yet been made, make a prediction... then export to json



vegas_odds_col_names = list(prediction_history.columns)
vegas_odds_col_values = [['' for _ in range(len(fights_list))] for _ in range(len(vegas_odds_col_names))]
vegas_odds_d = dict(zip(vegas_odds_col_names, vegas_odds_col_values))
vegas_odds = pd.DataFrame(data=vegas_odds_d)

vegas_odds['fighter name'].loc[:] = [fight[0] for fight in fights_list]
vegas_odds['opponent name'].loc[:] = [fight[1] for fight in fights_list]
# TODO add weight_class into vegas_odds and prediction history
vegas_odds['date'] = card_date

print('Making predictions for all fights on the books')
#filling in predictions
for i in vegas_odds.index:
    fighter=vegas_odds['fighter name'][i]
    opponent=vegas_odds['opponent name'][i]
    if in_ufc(fighter) and in_ufc(opponent):
        odds_calc = odds(fighter,opponent)
        print('predicting: '+fighter,'versus '+opponent,'.... '+str(odds_calc))
        vegas_odds['predicted fighter odds'][i]=odds_calc[0]
        vegas_odds['predicted opponent odds'][i]=odds_calc[1]
        sum_av_f=0
        tot_av_f=0
        sum_av_o=0
        tot_av_o=0
        for bookie in ['DraftKings', 'BetMGM', 'Caesars', 'BetRivers', 'FanDuel', 'PointsBet', 'Unibet', 'Bet365', 'BetWay', '5D', 'Ref']:
            if vegas_odds['fighter '+bookie][i]!='':
                sum_av_f+=int(vegas_odds['fighter '+bookie][i])
                tot_av_f+=1
            if vegas_odds['opponent '+bookie][i]!='':
                sum_av_o+=int(vegas_odds['opponent '+bookie][i])
                tot_av_o+=1
        if tot_av_f>0 and tot_av_o>0:
            vegas_odds['average bookie odds'][i]=[str(round(sum_av_f/tot_av_f)),str(round(sum_av_o/tot_av_o))]
            
print('saving scraped fights and predictions to models/buildingMLModel/data/external/vegas_odds.json')
print('TODO: scrape odds too. Currently only scraping names, date, and card title')
#save to json
result = vegas_odds.to_json()
parsed = json.loads(result)
jsonFilePath='models/buildingMLModel/data/external/vegas_odds.json'
with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
    jsonf.write(json.dumps(parsed, indent=4))
print('saved to '+jsonFilePath)
