import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from datetime import date
import math
from dateutil.relativedelta import relativedelta

from fight_stat_helpers import (
    fighter_score_diff, 
    fight_math_diff, 
    L5Y_sub_wins, 
    L5Y_losses, 
    L5Y_ko_losses, 
    fighter_age, 
    avg_count, 
    in_ufc, 
    clean_method_for_winner_predictions, 
    clean_method_for_method_predictions
    )

class FightPredictor:
    def __init__(self, ufc_fights):
        self.ufc_fights = ufc_fights
        fighter_names = [name for name in ufc_fights['fighter'].unique()]
        opponent_names = [name for name in ufc_fights['opponent'].unique()]
        self.ufc_fighter_names = set(fighter_names + opponent_names)
        
    def train_logistic_regression_model(self):
        r"""
        Train a logistic regression model to predict the winner of a UFC fight based on various statistics.
        This method processes the fight data, cleans the necessary columns, and prepares the data for training.
        The result is a set theta of regression coefficients and an intercept b that can be used for predictions.
        The model is trained using a subset of the most recent fights to avoid fitting to outdated trends.
        The model uses a minimal set of features that have been determined to be the most predictive based on offline testing.
        The features include differences in fighter statistics such as wins, losses, age, height, reach, and various striking statistics.
        This docstring was written by ChatGPT, an AI language model, and is intended to provide a clear understanding of the method's purpose and functionality.
        """
        #importing csv fight data and saving as dataframes
        ufc_fights_method = self.ufc_fights
        ufc_fights_winner = self.ufc_fights

        #there are some issues with how names are saved in the case when a fighter changes their name or uses a nickname
        # TODO this should be done on incoming data, not here. But to do this we must also do a global change to existing data by running this function offline and re-saving csvs and jsons
        self.clean_names(ufc_fights_winner, ['fighter', 'opponent'])
        self.clean_names(ufc_fights_method, ['fighter', 'opponent'])
        
        #cleaning the methods column for winner prediction
        #changing anything other than 'U-DEC','M-DEC', 'KO/TKO', 'SUB', to 'bullshit'
        #changing 'U-DEC','M-DEC', to 'DEC'
        ufc_fights_winner['method'] = clean_method_for_winner_predictions(ufc_fights_winner['method'])

        #cleaning the methods column for method prediction
        #changing anything other than 'U-DEC','M-DEC', 'S-DEC', 'KO/TKO', 'SUB', to 'bullshit'
        #changing 'U-DEC','M-DEC', 'S-DEC', to 'DEC'
        ufc_fights_method['method'] = clean_method_for_method_predictions(ufc_fights_method['method'])

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
        # ufc_fights=ufc_fights[draw_mask&method_mask_winner&age_mask&height_mask&reach_mask&wins_mask&strikes_mask]

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
        self.theta = list(winPredictionModel.coef_[0])
        self.b = winPredictionModel.intercept_[0]

    def get_regression_coeffs_and_intercept(self):
        return self.theta, self.b
        
    #there are some issues with how names are saved in the case when a fighter changes their name or uses a nickname
    # TODO this should be done on incoming data in DataHandler, not here. But to do this we must also do a global change to existing data by running this function offline and re-saving csvs and jsons
    def clean_names(self, df:pd.DataFrame, column_names:list):
        alias_dict = {
            'Joanne Calderwood': ['Joanne Wood'],
            'Bobby Green': ['King Green', 'Bobby King Green'],
        }
        
        for column_name in column_names:
            for i, name in enumerate(df[column_name]):
                for default_name, alias_list in alias_dict.items():
                    if name in alias_list:
                        df.at[i, 'fighter'] = default_name
                        

    # now make predictions for the new fights added to the new scraped fights
    def predict_upcoming_fights(self, prediction_history:pd.DataFrame, fights_list:list, card_date:str, theta, b):
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
                odds_calc = self.odds(fighter,opponent,theta,b)
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
        return vegas_odds
            
    #I've defined this in such a way to predict what happens when fighter1 in their day1 version fights fighter2
    #in their day2 version. Meaning we could compare for example 2014 Tyron Woodley to 2019 Colby Covington
    def ufc_prediction_tuple(self, fighter1,fighter2,day1=date.today(),day2=date.today()):
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
        

    # We want to predict how many times out of 10 the winning fighter would win, so we look at the values
    # x*theta+b. If the value is >=0 its a win and <=0 its a loss. Distance from zero gives indication of
    # how likely the outcome is.

    def presigmoid_value(self, fighter1,fighter2,date1,date2,theta,b):
        value = 0
        tup = self.ufc_prediction_tuple(fighter1,fighter2,date1,date2)
        for i in range(len(tup)):
            value += tup[i]*theta[i]
        return value + b

    def manual_prediction(self, fighter1,fighter2,date1,date2,theta,b):
        value = self.presigmoid_value(fighter1,fighter2,date1,date2,theta,b)
        value2 = self.presigmoid_value(fighter2,fighter1,date2,date1,theta,b)
        return value-value2>=0


    def sigmoid(self, x):
        sig = 1 / (1 + math.exp(-x))
        return sig

    #returns the probability that fighter1 defeats fighter2 on date1,date2
    def probability(self, fighter1,fighter2,date1,date2,theta,b):
        presig=self.presigmoid_value(fighter1,fighter2,date1,date2,theta,b)
        return self.sigmoid(presig)

    def odds(self, fighter1, fighter2, theta, b):
        date1=(date.today() - relativedelta(months=1)).strftime("%B %d, %Y")
        date2=(date.today() - relativedelta(months=1)).strftime("%B %d, %Y")
        p=self.probability(fighter1,fighter2,date1,date2,theta,b)
        
        if p < 0.5:
            fighterOdds = round(100 / p - 100)
            opponentOdds = round(1 / (1 / (1 - p) - 1) * 100)
            return ['+' + str(fighterOdds), '-' + str(opponentOdds)]
        elif p>=.5:
            fighterOdds = round(1 / (1 / p - 1) * 100)
            opponentOdds = round(100 / (1 - p) - 100)
            return ['-' + str(fighterOdds), '+' + str(opponentOdds)]

    def give_odds(self, fighter1, fighter2, date1, date2, theta, b):
        value = self.presigmoid_value(fighter1, fighter2, date1, date2, theta, b)
        value2 = self.presigmoid_value(fighter2, fighter1, date2, date1, theta, b)
        
        if value - value2 >= 0:
            winner = fighter1
        else:
            winner = fighter2
            
        value2 = self.presigmoid_value(fighter2, fighter1, date2, date1, theta, b)
        abs_value = (abs(value) + abs(value2)) / 2
        
        if abs_value >=0 and abs_value <= 0.2:
            print(winner+" wins a little over 5 times out of 10 times.")
        elif abs_value >= 0.2 and abs_value <= 0.4:
            print(winner+" wins 6 out of 10 times.")
        elif abs_value >= 0.4 and abs_value <= 0.6:
            print(winner+" wins 7 out of 10 times.")
        elif abs_value >= 0.6 and abs_value <= 0.8:
            print(winner+" wins 9 out of 10 times.")
        elif abs_value >= 0.8:
            print(winner+" wins 10 out of 10 times.")
            