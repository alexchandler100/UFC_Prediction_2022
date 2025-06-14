#getting dependencies
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from functions import *

pd.options.mode.chained_assignment = None  # default='warn' (disables SettingWithCopyWarning)

print('importing dataframe from ufc_fights.csv')
#importing csv fight data and saving as dataframes
ufc_fights_winner = pd.read_csv('models/buildingMLModel/data/processed/ufc_fights.csv',low_memory=False)
ufc_fights_method = pd.read_csv('models/buildingMLModel/data/processed/ufc_fights.csv',low_memory=False) # NOTE NOT YET USED BUT KEEPING IT FOR FUTURE USE
ufcfighterscrap =pd.read_csv('models/buildingMLModel/data/processed/fighter_stats.csv',sep=',',low_memory=False)

#there are some issues with how names are saved
#it gets saved as Joanne Wood for some reason
clean_names(ufc_fights_winner)

#cleaning the methods column for winner prediction
#changing anything other than 'U-DEC','M-DEC', 'KO/TKO', 'SUB', to 'bullshit'
#changing 'U-DEC','M-DEC', to 'DEC'
ufc_fights_winner['method'] = clean_method_for_winner_vect(ufc_fights_winner['method'])

#cleaning the methods column for method prediction
#changing anything other than 'U-DEC','M-DEC', 'S-DEC', 'KO/TKO', 'SUB', to 'bullshit'
#changing 'U-DEC','M-DEC', 'S-DEC', to 'DEC'
ufc_fights_method['method'] = clean_method_vect(ufc_fights_method['method'])

#getting rid of rows with incomplete or useless data
#fights with outcome "Win" or "Loss" (no "Draw")
draw_mask=ufc_fights_winner['result'] != 'D'

#fights where the method of victory is TKO/SUB/DEC (no split decision or DQ or Overturned or anything else like that)
method_mask_winner=(ufc_fights_winner['method']!='bullshit')
method_mask_method=(ufc_fights_method['method']!='bullshit')

#fights where age is known
age_mask=(ufc_fights_winner['fighter_age']!='unknown')&(ufc_fights_winner['opponent_age']!='unknown')&(ufc_fights_winner['fighter_age']!=0)&(ufc_fights_winner['opponent_age']!=0)

#fights where height/reach is known
height_mask=(ufc_fights_winner['fighter_height']!='unknown')&(ufc_fights_winner['opponent_height']!='unknown')
reach_mask=(ufc_fights_winner['fighter_reach']!='unknown')&(ufc_fights_winner['opponent_reach']!='unknown')

#fights where number of wins is known
wins_mask=(ufc_fights_winner['fighter_wins'] != 'unknown' )&(ufc_fights_winner['opponent_wins'] != 'unknown')

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
#MAYBE WE ARE LOSING PREDICTABILITY HERE (but we apply float later so may it is ok)

#here is the list of all stats available (besides stance), does not include names or result
punch_statistics=[  u'fighter_inf_knockdowns_avg',u'fighter_inf_pass_avg',u'fighter_inf_reversals_avg',u'fighter_inf_sub_attempts_avg',u'fighter_inf_takedowns_landed_avg',u'fighter_inf_takedowns_attempts_avg',u'fighter_inf_sig_strikes_landed_avg',u'fighter_inf_sig_strikes_attempts_avg',u'fighter_inf_total_strikes_landed_avg',u'fighter_inf_total_strikes_attempts_avg',u'fighter_inf_head_strikes_landed_avg',u'fighter_inf_head_strikes_attempts_avg',u'fighter_inf_body_strikes_landed_avg',u'fighter_inf_body_strikes_attempts_avg',u'fighter_inf_leg_strikes_landed_avg',u'fighter_inf_leg_strikes_attempts_avg',u'fighter_inf_distance_strikes_landed_avg',u'fighter_inf_distance_strikes_attempts_avg',u'fighter_inf_clinch_strikes_landed_avg',u'fighter_inf_clinch_strikes_attempts_avg',u'fighter_inf_ground_strikes_landed_avg',u'fighter_inf_ground_strikes_attempts_avg',u'fighter_abs_knockdowns_avg',u'fighter_abs_pass_avg',u'fighter_abs_reversals_avg',u'fighter_abs_sub_attempts_avg',u'fighter_abs_takedowns_landed_avg',u'fighter_abs_takedowns_attempts_avg',u'fighter_abs_sig_strikes_landed_avg',u'fighter_abs_sig_strikes_attempts_avg',u'fighter_abs_total_strikes_landed_avg',u'fighter_abs_total_strikes_attempts_avg',u'fighter_abs_head_strikes_landed_avg',u'fighter_abs_head_strikes_attempts_avg',u'fighter_abs_body_strikes_landed_avg',u'fighter_abs_body_strikes_attempts_avg',u'fighter_abs_leg_strikes_landed_avg',u'fighter_abs_leg_strikes_attempts_avg',u'fighter_abs_distance_strikes_landed_avg',u'fighter_abs_distance_strikes_attempts_avg',u'fighter_abs_clinch_strikes_landed_avg',u'fighter_abs_clinch_strikes_attempts_avg',u'fighter_abs_ground_strikes_landed_avg',u'fighter_abs_ground_strikes_attempts_avg',u'opponent_inf_knockdowns_avg',u'opponent_inf_pass_avg',u'opponent_inf_reversals_avg',u'opponent_inf_sub_attempts_avg',u'opponent_inf_takedowns_landed_avg',u'opponent_inf_takedowns_attempts_avg',u'opponent_inf_sig_strikes_landed_avg',u'opponent_inf_sig_strikes_attempts_avg',u'opponent_inf_total_strikes_landed_avg',u'opponent_inf_total_strikes_attempts_avg',u'opponent_inf_head_strikes_landed_avg',u'opponent_inf_head_strikes_attempts_avg',u'opponent_inf_body_strikes_landed_avg',u'opponent_inf_body_strikes_attempts_avg',u'opponent_inf_leg_strikes_landed_avg',u'opponent_inf_leg_strikes_attempts_avg',u'opponent_inf_distance_strikes_landed_avg',u'opponent_inf_distance_strikes_attempts_avg',u'opponent_inf_clinch_strikes_landed_avg',u'opponent_inf_clinch_strikes_attempts_avg',u'opponent_inf_ground_strikes_landed_avg',u'opponent_inf_ground_strikes_attempts_avg',u'opponent_abs_knockdowns_avg',u'opponent_abs_pass_avg',u'opponent_abs_reversals_avg',u'opponent_abs_sub_attempts_avg',u'opponent_abs_takedowns_landed_avg',u'opponent_abs_takedowns_attempts_avg',u'opponent_abs_sig_strikes_landed_avg',u'opponent_abs_sig_strikes_attempts_avg',u'opponent_abs_total_strikes_landed_avg',u'opponent_abs_total_strikes_attempts_avg',u'opponent_abs_head_strikes_landed_avg',u'opponent_abs_head_strikes_attempts_avg',u'opponent_abs_body_strikes_landed_avg',u'opponent_abs_body_strikes_attempts_avg',u'opponent_abs_leg_strikes_landed_avg',u'opponent_abs_leg_strikes_attempts_avg',u'opponent_abs_distance_strikes_landed_avg',u'opponent_abs_distance_strikes_attempts_avg',u'opponent_abs_clinch_strikes_landed_avg',u'opponent_abs_clinch_strikes_attempts_avg',u'opponent_abs_ground_strikes_landed_avg',u'opponent_abs_ground_strikes_attempts_avg']

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

possible_stats = record_statistics_diff + physical_stats_diff + punch_statistics_diff

#setting
ufc_fights_winner['fighter_age'] = ufc_fights_winner['fighter_age'].apply(float)
ufc_fights_winner['opponent_age'] = ufc_fights_winner['opponent_age'].apply(float)
ufc_fights_winner['fighter_age_diff'] = ufc_fights_winner['fighter_age']-ufc_fights_winner['opponent_age']

# WE SET UP A LOGISTIC REGRESSION MODEL BELOW WITH A MINIMAL SET OF FEATURES
# TODO TRY XGBOOST AND RANDOM FOREST # TODO ALSO PREDICT METHOD OF VICTORY

# found this to be the best set of features in our offline testing
# k-fighter_score_diff : the difference in the fighter's score in the for their last k fights 
# k-fight_math : the difference in the fighter's fight math score for the last k years
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
# say there is an average of 10 fights per week, then 2200 fights is about 55 months of data
X=ufc_fights_df.iloc[0:2200].to_numpy() # only taking most recent 2200 fights to avoid fitting to old fight trends (maybe play with this number later)
y=ufc_fights_winner['result'].iloc[0:2200]
print('Fitting Logistic Regression Model')
winPredictionModel.fit(X,y)
print('Accuracy of Logistic Regression win prediction: '+str(cross_val_score(winPredictionModel,X,y,cv=3).mean()))
print('coefficients'+str(winPredictionModel.coef_))
print('intercept'+str(winPredictionModel.intercept_))
theta = winPredictionModel.coef_
b = winPredictionModel.intercept_[0]

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
bad_indices = get_bad_indices(vegas_odds_old, ufc_fights_crap)
vegas_odds_old = vegas_odds_old.drop(bad_indices)

#making a copy of vegas_odds
vegas_odds_copy=vegas_odds_old.copy()
prediction_history=pd.read_json('models/buildingMLModel/data/external/prediction_history.json')

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
            
print('saving scraped fights and predictions to models/buildingMLModel/data/external/vegas_odds.json')
print('TODO: scrape odds too. Currently only scraping names, date, and card title')
vegas_odds = predict_upcoming_fights(prediction_history, fights_list, card_date, theta, b)
#save to json
result = vegas_odds.to_json()
parsed = json.loads(result)
jsonFilePath='models/buildingMLModel/data/external/vegas_odds.json'
with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
    jsonf.write(json.dumps(parsed, indent=4))
print('saved to '+jsonFilePath)
