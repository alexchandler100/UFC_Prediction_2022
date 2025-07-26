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
    fighter_score_diff, 
    fight_math_diff, 
    L5Y_sub_wins, 
    sub_wins,
    L5Y_losses, 
    L5Y_ko_wins,
    L2Y_ko_losses,
    L5Y_ko_losses, 
    fighter_age, 
    avg_count, 
    in_ufc, 
    fighter_height,
    fighter_reach,
    L5Y_wins,
    ko_losses,
    L2Y_sub_losses
    )

class FightPredictor:
    def __init__(self, ufc_fights_winner, dh):
        self.dh = dh
        self.ufc_fights_winner = ufc_fights_winner.copy()
        fighter_names = [name for name in ufc_fights_winner['fighter'].unique()]
        opponent_names = [name for name in ufc_fights_winner['opponent'].unique()]
        self.ufc_fighter_names = set(fighter_names + opponent_names)
        self.scaler = None
        
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
        ufc_fights_winner = self.ufc_fights_winner.copy()
            
        # WE SET UP A LOGISTIC REGRESSION MODEL BELOW WITH A MINIMAL SET OF FEATURES
        # TODO TRY XGBOOST AND RANDOM FOREST # TODO ALSO PREDICT METHOD OF VICTORY
                
        # result without squares
        self.amazing_feature_set = [
            'l3y_wins_diff',
            'age_diff',
            'age_sum',
            'l5y_offensive_grappling_score_diff',
            'l5y_inf_ground_strikes_attempts_per_min_diff',
            'l5y_losses_ko_diff',
            'l5y_defensive_grappling_loss_diff',
            'all_inf_takedowns_attempts_per_min_diff',
            'l5y_abs_ground_strikes_landed_per_min_diff',
            'l1y_abs_ground_strikes_attempts_per_min_diff',
            'all_abs_ground_strikes_attempts_per_min_diff',
            'l1y_abs_control_per_min_diff',
            'l1y_abs_clinch_strikes_accuracy_diff',
            'l1y_inf_takedowns_attempts_per_min_diff',
            'l1y_losses_ko_diff',
            'reach_diff',
            'all_inf_reversals_per_min_diff',
            'l3y_inf_reversals_per_min_diff',
            'l3y_inf_ground_strikes_attempts_per_min_sum',
            'all_inf_ground_strikes_landed_per_min_sum',
            'l1y_inf_takedowns_accuracy_diff',
            'l3y_inf_leg_strikes_accuracy_diff',
            'l5y_inf_total_strikes_attempts_per_min_diff',
            'all_inf_total_strikes_landed_per_min_diff',
            'l3y_inf_head_strikes_landed_per_min_diff',
            'all_inf_clinch_strikes_landed_per_min_diff',
            'l3y_inf_clinch_strikes_attempts_per_min_diff',
            'l3y_inf_ground_strikes_accuracy_diff',
            'all_abs_body_strikes_attempts_per_min_diff',
            'l5y_abs_body_strikes_landed_per_min_diff',
            'l1y_inf_sig_strikes_accuracy_diff',
            'l1y_inf_clinch_strikes_attempts_per_min_diff',
            'l5y_inf_head_strikes_attempts_per_min_diff',
            'l3y_inf_clinch_strikes_landed_per_min_diff',
            'l5y_num_fights_diff',
            'l1y_inf_ground_strikes_landed_per_min_diff',
            'l5y_inf_knockdowns_per_min_diff',
            'l1y_inf_knockdowns_per_min_diff',
            'l5y_abs_body_strikes_attempts_per_min_diff',
            'l3y_inf_total_strikes_attempts_per_min_diff',
            'all_inf_body_strikes_landed_per_min_diff',
            'l3y_abs_sub_attempts_per_min_diff',
            'l5y_abs_sub_attempts_per_min_diff',
            'l3y_abs_ground_strikes_landed_per_min_diff',
            'l3y_abs_ground_strikes_attempts_per_min_diff',
            'l1y_abs_body_strikes_accuracy_diff',
            'l1y_inf_knockdowns_per_min_sum',
            'all_inf_knockdowns_per_min_diff',
            'all_inf_reversals_per_min_sum',
            'l5y_inf_ground_strikes_landed_per_min_diff',
            'all_abs_distance_strikes_attempts_per_min_diff',
            'all_inf_head_strikes_accuracy_diff',
            'l3y_abs_clinch_strikes_accuracy_diff',
            'l1y_abs_body_strikes_landed_per_min_diff',
            'l3y_inf_body_strikes_accuracy_diff',
            'all_abs_clinch_strikes_accuracy_diff',
            'all_inf_body_strikes_accuracy_diff',
            'l5y_inf_clinch_strikes_landed_per_min_diff',
            'l3y_inf_head_strikes_attempts_per_min_diff',
            'all_wins_dec_diff',
            'l1y_inf_total_strikes_accuracy_diff',
            'l1y_abs_head_strikes_landed_per_min_diff',
            'l1y_abs_head_strikes_accuracy_diff',
            'all_abs_head_strikes_landed_per_min_diff',
            'l1y_inf_distance_strikes_attempts_per_min_diff',
            'all_abs_head_strikes_attempts_per_min_diff',
            'all_inf_distance_strikes_landed_per_min_diff',
            'l1y_inf_clinch_strikes_accuracy_diff',
            'l1y_inf_clinch_strikes_landed_per_min_diff',
            'l1y_losses_dec_diff',
            'l5y_abs_clinch_strikes_landed_per_min_diff',
            'all_inf_head_strikes_attempts_per_min_diff',
            'l1y_inf_sub_attempts_per_min_diff',
            'l3y_inf_sub_attempts_per_min_diff',
            'l3y_abs_head_strikes_landed_per_min_diff',
            'l1y_inf_sig_strikes_attempts_per_min_diff',
            'l1y_abs_total_strikes_attempts_per_min_diff', 
            'l1y_abs_total_strikes_landed_per_min_diff',
            'all_inf_leg_strikes_accuracy_diff',
            'l3y_abs_body_strikes_accuracy_diff',
            'l3y_inf_sig_strikes_accuracy_diff',
            'all_abs_body_strikes_landed_per_min_diff',
            'all_inf_ground_strikes_accuracy_diff',
            'all_abs_leg_strikes_accuracy_diff',
            'all_abs_total_strikes_landed_per_min_diff',
            'all_abs_clinch_strikes_landed_per_min_diff',
        ]

        # y=ufc_fights_winner['result'].iloc[0:40*75]
        # X=ufc_fights_winner[amazing_feature_set].iloc[0:40*75]
        # winPredictionModel=LogisticRegression(solver='lbfgs', max_iter=5000, fit_intercept=False)
        # self.scaler = preprocessing.StandardScaler().fit(X)
        # X_scaled = self.scaler.transform(X) 
        # winPredictionModel.fit(X_scaled,y)

        # print('model score: '+str(winPredictionModel.score(X,y)))
        # print('neg log loss of model: '+str(self.model_score(ufc_fights_winner,current_best_feature_set,scoring='neg_log_loss')))
        # print('cross val avg probability of result: '+str(np.exp(self.model_score(ufc_fights_winner,current_best_feature_set,scoring='neg_log_loss'))))
        # print('cross val accuracy: '+str(self.model_score(ufc_fights_winner,current_best_feature_set,scoring='accuracy')))
        
        # self.theta = list(winPredictionModel.coef_[0])
        # self.b = winPredictionModel.intercept_[0]

        ufc_fights_df = ufc_fights_winner[self.amazing_feature_set]
        results = ufc_fights_winner['result']
        # self.theta, self.b = self.find_best_regression_coeffs(ufc_fights_df, results)
        self.theta, self.b = self.find_regression_coeffs(ufc_fights_df, results)
    
    
    def find_regression_coeffs(self, X, y, _max_iter=20000):
        # do another split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)
        # see how the new features do on the test set we already made
        # train the model with the best features

        best_model = LogisticRegression(solver='lbfgs', max_iter=_max_iter)#, C=0.1, penalty='l2')#, fit_intercept=False)
        best_model.fit(X_train, y_train)

        # evaluate the model on the training set
        train_score = best_model.score(X_train, y_train)

        print(f'Train set accuracy: {train_score}')

        # evaluate the model on the test set
        test_score = best_model.score(X_test, y_test)
        print(f'Test set accuracy: {test_score} \n')
        
        theta = list(best_model.coef_[0])
        b = best_model.intercept_[0]
        
        return theta, b
    
    #scores a model # TODO BEFORE WE USE THIS< UPDATE TO DO REGULARIZATION
    def model_score(self, dataframe, features, iloc_val = 3200, _max_iter = 2000, scoring='neg_log_loss', scaled=True):
        yyy=dataframe['result'].iloc[0:iloc_val]
        XXX=dataframe[features].iloc[0:iloc_val]
        XXXscaler = preprocessing.StandardScaler().fit(XXX)
        XXX_scaled = XXXscaler.transform(XXX) 
        X = XXX_scaled if scaled else XXX
        winPredictionModel=LogisticRegression(solver='lbfgs', max_iter=_max_iter, fit_intercept=False)
        # find the cross val score with log loss
        return cross_val_score(winPredictionModel,X,yyy,cv=4,scoring=scoring).mean()
        
    def find_best_regression_coeffs(self, ufc_fights_df, results):
        # say there is an average of 10 fights per week, then 2200 fights is about 55 months of data
        #decided to force intercept to be 0 due to symmetry of dataset (all stats are differences so if we switch fighters, we must get the negative of the result)
        # make some hyperparams
        max_iters = [2000, 2200, 2500, 5000]
        solvers = ['lbfgs']
        num_fights_in_history = [1600, 2000, 3000]
        theta_list = []
        b_list = []
        cross_val_scores = []
        neg_log_loss_scores = []
        max_iter_history = []
        solver_history = []
        num_fights_history = []
        scaler_history = []
        num_reps = 50
        for solver in solvers:
            for max_iter in max_iters:
                for num_fights in num_fights_in_history:
                    for rep in range(num_reps): # try with different random seeds
                        # print(f'Training Logistic Regression with solver={solver} and max_iter={max_iter}')
                        # create a logistic regression model
                        winPredictionModel = LogisticRegression(solver=solver, max_iter=max_iter, fit_intercept=False)
                        scaler = StandardScaler()
                        X = ufc_fights_df.iloc[0:num_fights].to_numpy()
                        X_scaled = scaler.fit_transform(X)
                        y = results.iloc[0:num_fights]
                        winPredictionModel.fit(X_scaled,y)
                        # TODO scale to zero mean and unit variance
                        cross_val_score_mean = cross_val_score(winPredictionModel, X, y, cv=3).mean()
                        neg_log_loss_score = cross_val_score(winPredictionModel, X, y, cv=3, scoring='neg_log_loss').mean()
                        theta = list(winPredictionModel.coef_[0])
                        b = winPredictionModel.intercept_[0]
                        theta_list.append(theta)
                        b_list.append(b)
                        scaler_history.append(scaler)
                        cross_val_scores.append(cross_val_score_mean)
                        neg_log_loss_scores.append(neg_log_loss_score)
                        max_iter_history.append(max_iter)
                        solver_history.append(solver)
                        num_fights_history.append(num_fights)
        # TODO change this to find the best neg_log_loss instead of accuracy
        
        # Using best hyperparameters: solver=lbfgs, max_iter=1800, num_fights=2200
        # Best cross-validation score: 0.630
        # neg_log_loss score: -0.648
        # probability of observing data given model: 0.523
        
        # WHAT THE HELL... IN THE WORKSHEET I AM GETTING A HUGELY BETTER ANSWER...
        # TODO EXPLAIN THIS... I THOUGHT IT WAS FROM THE LACK OF SCALING...
        
        # model score: 0.656
        # neg log loss of model: -0.636851776860466
        # cross val avg probability of result: 0.528955074077124
        # cross val accuracy: 0.645625

        # best_cross_val_score = max(cross_val_scores)
        # best_index = cross_val_scores.index(best_cross_val_score)
        best_neg_log_loss = max(neg_log_loss_scores)
        best_index = neg_log_loss_scores.index(best_neg_log_loss)
        cross_val_score_of_best_nll = cross_val_scores[best_index]
        best_scaler = scaler_history[best_index]
        self.scaler = best_scaler
        best_theta = theta_list[best_index]
        best_b = b_list[best_index]
        print(f'Using best hyperparameters: solver={solver_history[best_index]}, max_iter={max_iter_history[best_index]}, num_fights={num_fights_history[best_index]}')
        print(f'Best cross-validation score: {cross_val_score_of_best_nll:.3f}')
        print(f'neg_log_loss score: {neg_log_loss_score:.3f}')
        print(f'probability of observing data given model: {np.exp(neg_log_loss_score):.3f}')
        return best_theta, best_b

    def get_regression_coeffs_intercept_and_scaler(self):
        return self.theta, self.b

    # now make predictions for the new fights added to the new scraped fights
    def predict_upcoming_fights(self, prediction_history:pd.DataFrame, fights_list:list, card_date:str, theta, b):
        vegas_odds_col_names = list(prediction_history.columns)
        vegas_odds_col_values = [['' for _ in range(len(fights_list))] for _ in range(len(vegas_odds_col_names))]
        vegas_odds_d = dict(zip(vegas_odds_col_names, vegas_odds_col_values))
        vegas_odds = pd.DataFrame(data=vegas_odds_d)

        vegas_odds['fighter name'] = [fight[0] for fight in fights_list]
        vegas_odds['opponent name'] = [fight[1] for fight in fights_list]
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
        ufc_fights_predictive_flattened_diffs = dh.make_ufc_fights_predictive_flattened_diffs(derived_doubled_tuple)
        prediction_tuple = ufc_fights_predictive_flattened_diffs[self.amazing_feature_set].to_numpy()
        return prediction_tuple
        
        

    # We want to predict how many times out of 10 the winning fighter would win, so we look at the values
    # x*theta+b. If the value is >=0 its a win and <=0 its a loss. Distance from zero gives indication of
    # how likely the outcome is.

    def presigmoid_value(self, fighter1,fighter2,date1,date2,theta,b):
        tup = self.ufc_prediction_tuple(fighter1,fighter2,date1,date2)
        value = np.dot(tup, theta)
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
        date_today=(date.today()).strftime("%B %d, %Y")
        p=self.probability(fighter1,fighter2,date_today,date_today,theta,b)
        
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
            