import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from datetime import date
import math
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from fight_stat_helpers import (
    in_ufc, 
    visualize_prediction_bokeh
    )

class FightPredictor:
    def __init__(self, ufc_fights_winner, dh):
        self.dh = dh
        self.ufc_fights_winner = ufc_fights_winner.copy()
        fighter_names = [name for name in ufc_fights_winner['fighter'].unique()]
        opponent_names = [name for name in ufc_fights_winner['opponent'].unique()]
        self.ufc_fighter_names = set(fighter_names + opponent_names)
        self.scaler = None
        
    def train_logistic_regression_model(self, random_state=14, scaled=True, C=100):
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
        ufc_fights_winner = self.ufc_fights_winner.copy()
            
        # WE SET UP A LOGISTIC REGRESSION MODEL BELOW WITH A MINIMAL SET OF FEATURES
        # TODO TRY XGBOOST AND RANDOM FOREST # TODO ALSO PREDICT METHOD OF VICTORY
                
        # got 64 % on a test set
        self.amazing_feature_set = [
        # 'l5y_overall_fighter_score_diff', # VIF = 10 (no change)
        # physical
        'age_diff',
        'reach_diff',
        'height_diff', # added by instinct and it raised test model score
        'l5y_wins_diff',
        'l1y_wins_sub_diff',
        'l1y_wins_ko_diff',
        'l3y_wins_ko_diff',
        'all_wins_ko_diff',
        'l5y_wins_wins_diff',
        # 'all_wins_wins_diff',
        'l3y_losses_dec_diff',
        'all_losses_losses_diff',
        'l1y_losses_losses_diff',
        # punch stats
        'l1y_inf_clinch_strikes_accuracy_diff',
        'all_inf_distance_strikes_accuracy_diff', # VIF = 4 (removal decreased test model score)
        'all_inf_total_strikes_attempts_per_min_diff',
        'l1y_inf_clinch_strikes_landed_per_min_diff',
        'all_inf_clinch_strikes_accuracy_diff',
        'l5y_inf_knockdowns_per_min_diff',
        'l3y_inf_head_strikes_accuracy_diff',
        'l5y_inf_body_strikes_accuracy_diff',
        'l3y_inf_distance_strikes_landed_per_min_diff',
        'l1y_inf_total_strikes_accuracy_diff',
        'l1y_inf_body_strikes_attempts_per_min_diff',
        'all_abs_head_strikes_accuracy_diff',
        'l3y_abs_body_strikes_accuracy_diff',
        'l3y_abs_total_strikes_attempts_per_min_diff',
        'l1y_abs_clinch_strikes_attempts_per_min_diff',
        'all_abs_clinch_strikes_accuracy_diff',
        'l3y_abs_leg_strikes_attempts_per_min_diff',
        'l1y_abs_body_strikes_attempts_per_min_diff',
        'l5y_abs_body_strikes_landed_per_min_diff',
        # grappling stats
        'all_inf_sub_attempts_per_min_diff',
        'l1y_inf_ground_strikes_attempts_per_min_diff',
        'l5y_inf_takedowns_attempts_per_min_diff',
        'l3y_inf_takedowns_accuracy_diff',
        'l3y_inf_takedowns_landed_per_min_diff', # VIF = 4
        'l1y_abs_ground_strikes_landed_per_min_diff',
        'all_abs_control_per_min_diff',
        'all_abs_takedowns_attempts_per_min_diff',
        'l3y_abs_takedowns_landed_per_min_diff',
        # higher order stats
        'l5y_fight_math_diff',
        'all_offensive_grappling_score_diff',
        'all_defensive_grappling_loss_diff',
        'l3y_defensive_grappling_loss_diff',
        # 'all_abs_leg_strikes_attempts_per_min_diff', # VIF = 8 (removal increased test model score)
        # 'l3y_inf_sub_attempts_per_min_diff', # VIF = 16
        # 'l1y_inf_body_strikes_landed_per_min_diff', # VIF = 12 (removal increased test model score)
        # 'l1y_abs_total_strikes_attempts_per_min_diff', # VIF = 4
        # 'l5y_defensive_grappling_loss_diff', # VIF = 9 (removal decreased test model score)
        # 'l3y_overall_fighter_score_diff', # VIF = 24 (removal decreased test model score)
        # 'all_inf_head_strikes_accuracy_diff', # VIF = 15 (removal increased test model score)
        # 'l3y_inf_sig_strikes_attempts_per_min_diff', # VIF = 16 (removal increased test model score)
        # 'l3y_inf_body_strikes_accuracy_diff', # VIF = 19 (removal decreased test model score)
        # 'l3y_abs_control_per_min_diff', # VIF = 8 (removal increased test model score)
        # 'all_abs_sig_strikes_accuracy_diff', # VIF = 16 (removal decreased test model score)
        # 'l1y_abs_clinch_strikes_landed_per_min_diff', # VIF = 19  (removal decreased test model score)
        # 'l3y_abs_body_strikes_attempts_per_min_diff', # VIF = 8 (removal decreased test model score)
        # 'l3y_abs_body_strikes_landed_per_min_diff', # VIF = 31 (removal increased test model score)
        # 'l1y_inf_distance_strikes_accuracy_diff', # VIF = 6 (no change)
        # 'l3y_offensive_grappling_score_diff' # VIF = inf (removal increased test model score)
        ]


        ufc_fights_df = ufc_fights_winner[self.amazing_feature_set]
        results = ufc_fights_winner['result']
        # self.theta, self.b = self.find_best_regression_coeffs(ufc_fights_df, results)
        self.theta, self.b, self.scaler = self.find_regression_coeffs(
            ufc_fights_df, 
            results, 
            random_state=random_state, 
            scaled=scaled, 
            C=C
            )
    
    
    def find_regression_coeffs(self, X, y, _max_iter=20000, random_state=14, scaled=True, C=0.1):
        # do another split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        # see how the new features do on the test set we already made
        # train the model with the best features
        best_model = LogisticRegression(solver='lbfgs', max_iter=_max_iter, C=C, penalty='l2')#, fit_intercept=False)
        if scaled:
            print('Scaling features')
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

        else:
            scaler = None
            X_train_scaled = X_train
            X_test_scaled = X_test
        best_model.fit(X_train_scaled, y_train)

        # evaluate the model on the training set
        train_score = best_model.score(X_train_scaled, y_train)

        print(f'Train set accuracy: {train_score}')

        # evaluate the model on the test set
        test_score = best_model.score(X_test_scaled, y_test)
        print(f'Test set accuracy: {test_score} \n')
        
        theta = list(best_model.coef_[0])
        b = best_model.intercept_[0]
        
        return theta, b, scaler
        

    def get_regression_coeffs_intercept_and_scaler(self):
        return self.theta, self.b, self.scaler

    # now make predictions for the new fights added to the new scraped fights
    def predict_upcoming_fights(
        self, 
        prediction_history:pd.DataFrame, 
        fighter_stats:pd.DataFrame, 
        fights_list:list, 
        card_date:str, 
        theta, 
        b, 
        scaler
        ):
        vegas_odds_col_names = list(prediction_history.columns)
        vegas_odds_col_values = [['' for _ in range(len(fights_list))] for _ in range(len(vegas_odds_col_names))]
        vegas_odds_d = dict(zip(vegas_odds_col_names, vegas_odds_col_values))
        vegas_odds = pd.DataFrame(data=vegas_odds_d)

        vegas_odds['fighter name'] = [fight[0] for fight in fights_list]
        vegas_odds['opponent name'] = [fight[1] for fight in fights_list]
        # TODO add weight_class into vegas_odds and prediction history
        vegas_odds['date'] = card_date

        print('Making predictions for all fights on the books')
        # import ipdb;ipdb.set_trace(context=10)
        #filling in predictions
        for i in vegas_odds.index:
            fighter=vegas_odds['fighter name'][i]
            opponent=vegas_odds['opponent name'][i]
            if in_ufc(fighter, fighter_stats) and in_ufc(opponent, fighter_stats):
                odds_calc = self.odds(fighter,opponent,theta,b,scaler)
                print('predicting: '+fighter,'versus '+opponent,'.... '+str(odds_calc))
                vegas_odds.loc[i, 'predicted fighter odds']=odds_calc[0]
                vegas_odds.loc[i, 'predicted opponent odds']=odds_calc[1]
                
        return vegas_odds
            
    #I've defined this in such a way to predict what happens when fighter1 in their day1 version fights fighter2
    #in their day2 version. Meaning we could compare for example 2014 Tyron Woodley to 2019 Colby Covington
    def ufc_prediction_tuple(self, fighter1,fighter2,day1=date.today(),day2=date.today()):
        new_rows_dict = {
            'fighter':[fighter1, fighter2], 
            'opponent':[fighter2, fighter1], 
            'date':[day1,day1],
            'result':["N/A","N/A"],
            'method':["N/A","N/A"], 
            'division':["TODO","TODO"]
            }
        new_rows = pd.DataFrame(new_rows_dict)
        dh = self.dh
        derived_doubled_tuple = dh.make_derived_doubled_vector_for_fight(new_rows)
        # now make the predictive flattened diffs
        ufc_fights_predictive_flattened_diffs = dh.make_ufc_fights_predictive_flattened_diffs(derived_doubled_tuple, shuffle=False)
        prediction_tuple = ufc_fights_predictive_flattened_diffs[self.amazing_feature_set]
        return prediction_tuple
        
        

    # We want to predict how many times out of 10 the winning fighter would win, so we look at the values
    # x*theta+b. If the value is >=0 its a win and <=0 its a loss. Distance from zero gives indication of
    # how likely the outcome is.

    def presigmoid_value(self, fighter1,fighter2,date1,date2,theta,b,scaler):
        tup = self.ufc_prediction_tuple(fighter1,fighter2,date1,date2)
        if scaler:
            tup = scaler.transform(tup)
            
        value = np.dot(tup, theta)
        presig_value = value + b
        p = self.sigmoid(presig_value[0])
        
        visualize_prediction_bokeh(fighter1, fighter2, theta, date1, tup, p)
        return presig_value


    def sigmoid(self, x):
        sig = 1 / (1 + math.exp(-x))
        return sig

    #returns the probability that fighter1 defeats fighter2 on date1,date2
    def probability(self, fighter1,fighter2,date1,date2,theta,b,scaler):
        presig=self.presigmoid_value(fighter1,fighter2,date1,date2,theta,b,scaler)
        prob = self.sigmoid(presig)
        return prob

    def odds(self, fighter1, fighter2, theta, b, scaler):
        date_today=(date.today()).strftime("%B %d, %Y")
        p=self.probability(fighter1,fighter2,date_today,date_today,theta,b,scaler)
        
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
            

            