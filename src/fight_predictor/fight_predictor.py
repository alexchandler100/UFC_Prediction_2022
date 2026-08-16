import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from datetime import date
import math
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sklearn.metrics

from fight_stat_helpers import (
    in_ufc, 
    visualize_prediction_bokeh,
    get_canonical_name,
    )

class FightPredictor:
    def __init__(self, ufc_fights_winner, dh):
        self.dh = dh
        self.ufc_fights_winner = ufc_fights_winner.copy()
        fighter_names = [name for name in ufc_fights_winner['fighter'].unique()]
        opponent_names = [name for name in ufc_fights_winner['opponent'].unique()]
        self.ufc_fighter_names = set(fighter_names + opponent_names)
        self.scaler = None
        self.imputer = None
        
    def train_logistic_regression_model(self, random_state=56, scaled=True, C=100):
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
        self.amazing_feature_set =  [
        'age_diff',
        'reach_diff',
        'l5y_wins_diff',
        'l5y_losses_ko_diff',
        'all_wins_wins_diff',
        'l5y_wins_wins_diff',
        'l5y_losses_losses_diff',
        'all_losses_losses_diff',
        'l3y_losses_sub_diff',
        'l1y_wins_sub_diff',
        'l1y_wins_diff',
        # 'all_wins_diff',
        'l3y_fight_math_diff',
        'all_inf_control_per_min_diff',
        'all_inf_distance_strikes_accuracy_diff',
        'l1y_inf_takedowns_landed_per_min_diff',
        # 'l1y_inf_takedowns_attempts_per_min_diff',
        'l3y_inf_takedowns_attempts_per_min_diff',
        'l3y_inf_ground_strikes_attempts_per_min_diff',
        'all_inf_body_strikes_accuracy_diff',
        'l1y_inf_body_strikes_attempts_per_min_diff',
        'l5y_inf_body_strikes_attempts_per_min_diff',
        'all_inf_clinch_strikes_landed_per_min_diff',
        # 'l5y_inf_clinch_strikes_attempts_per_min_diff',
        'l1y_inf_total_strikes_landed_per_min_diff',
        'l1y_abs_knockdowns_per_min_diff',
        'l1y_abs_takedowns_attempts_per_min_diff',
        'all_abs_takedowns_attempts_per_min_diff',
        'l3y_abs_head_strikes_accuracy_diff',
        'l1y_abs_body_strikes_accuracy_diff',
        'l3y_abs_body_strikes_accuracy_diff',
        # 'l1y_abs_clinch_strikes_accuracy_diff',
        'l3y_abs_clinch_strikes_landed_per_min_diff',
        'l3y_abs_clinch_strikes_accuracy_diff',
        ]
        
        # TODO FIGURE OUT WHY PERF IS SO MUCH WORSE WHEN WE do it here versus in the notebook 
        # C:\Users\Alex\OneDrive\Documents\GitHub\UFC_Prediction_2022\src\models\notebooks\Exploratory\FeatureSelection_REMOVE_DERIVED_SCORES.ipynb


        if 'date' in ufc_fights_winner.columns:
            ufc_fights_winner = ufc_fights_winner.sort_values('date', kind='stable')
        ufc_fights_df = ufc_fights_winner[self.amazing_feature_set]
        results = ufc_fights_winner['result']
        dates = ufc_fights_winner['date'] if 'date' in ufc_fights_winner.columns else None
        # self.theta, self.b = self.find_best_regression_coeffs(ufc_fights_df, results)
        self.theta, self.b, self.scaler = self.find_regression_coeffs(
            ufc_fights_df, 
            results, 
            dates=dates,
            random_state=random_state, 
            scaled=scaled, 
            C=C
            )
    
    
    def find_regression_coeffs(self, X, y, dates=None, _max_iter=30000, random_state=14, scaled=True, C=0.1):
        if len(X) < 10:
            raise ValueError(f'At least 10 fights are required to train, got {len(X)}')

        # Evaluate the way this system is deployed: older cards train the
        # model and the newest 20% of cards are held out.  Random row splits
        # substantially overstated recent performance.
        if dates is not None:
            parsed_dates = pd.to_datetime(dates).to_numpy()
            order = np.argsort(parsed_dates, kind='stable')
        else:
            parsed_dates = None
            order = np.arange(len(X))
            print('Warning: no dates supplied; temporal validation uses the existing row order')
        split_index = max(1, int(len(order) * 0.8))
        if parsed_dates is not None and split_index < len(order):
            cutoff = parsed_dates[order[split_index]]
            train_indices = order[parsed_dates[order] < cutoff]
            test_indices = order[parsed_dates[order] >= cutoff]
        else:
            train_indices = order[:split_index]
            test_indices = order[split_index:]
        if not len(train_indices) or not len(test_indices):
            raise ValueError('Temporal validation needs at least two distinct fight dates')
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        # Zero means "no known difference" for the antisymmetric matchup
        # features.  This preserves fights with sparse physical/history data
        # instead of selecting only well-documented veterans.
        evaluation_imputer = SimpleImputer(
            strategy='constant', fill_value=0.0, keep_empty_features=True
        )
        X_train_imputed = evaluation_imputer.fit_transform(X_train)
        X_test_imputed = evaluation_imputer.transform(X_test)

        evaluation_model = LogisticRegression(
            solver='lbfgs', max_iter=_max_iter, C=C, penalty='l2',
            fit_intercept=False, random_state=random_state,
        )
        if scaled:
            print('Scaling features')
            evaluation_scaler = StandardScaler(with_mean=False)
            X_train_scaled = evaluation_scaler.fit_transform(X_train_imputed)
            X_test_scaled = evaluation_scaler.transform(X_test_imputed)

        else:
            evaluation_scaler = None
            X_train_scaled = X_train_imputed
            X_test_scaled = X_test_imputed
        evaluation_model.fit(X_train_scaled, y_train)
        
        # evaluate the model on the training set
        train_score = evaluation_model.score(X_train_scaled, y_train)
        print(f'Training set size: {X_train.shape} accuracy: {train_score}')

        # evaluate the model on the test set
        test_score = evaluation_model.score(X_test_scaled, y_test)
        print(f'Test set size: {X_test.shape} accuracy: {test_score}')
        
        # get the neg log loss score of the test set and convert it to a probability
        y_proba_test = evaluation_model.predict_proba(X_test_scaled)
        log_loss = sklearn.metrics.log_loss(y_test, y_proba_test)
        brier = sklearn.metrics.brier_score_loss(y_test, y_proba_test[:, 1])
        print(f'Temporal test log loss: {log_loss}. Brier score: {brier}. Average probability to observe data given model: {np.exp(-log_loss)}')

        # The holdout model is for evaluation only.  Production must use all
        # eligible history after the feature contract/hyperparameters are fixed.
        final_model = LogisticRegression(
            solver='lbfgs', max_iter=_max_iter, C=C, penalty='l2',
            fit_intercept=False, random_state=random_state,
        )
        self.imputer = SimpleImputer(
            strategy='constant', fill_value=0.0, keep_empty_features=True
        )
        X_imputed = self.imputer.fit_transform(X)
        if scaled:
            scaler = StandardScaler(with_mean=False)
            X_final = scaler.fit_transform(X_imputed)
        else:
            scaler = None
            X_final = X_imputed
        final_model.fit(X_final, y)

        theta = list(final_model.coef_[0])
        b = float(final_model.intercept_[0])
        
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
            # replace with canonical names from fighter_stats
            fighter_canonical_name = get_canonical_name(fighter, fighter_stats)
            opponent_canonical_name = get_canonical_name(opponent, fighter_stats)
            fighter = fighter_canonical_name
            opponent = opponent_canonical_name
            vegas_odds.at[i, 'fighter name'] = fighter
            vegas_odds.at[i, 'opponent name'] = opponent
            
            if in_ufc(fighter, fighter_stats) and in_ufc(opponent, fighter_stats):
                derived_doubled_tuple = self.get_ufc_fights_reported_derived_doubled_for_upcoming_fight(fighter, opponent, day1=pd.to_datetime(card_date), day2=pd.to_datetime(card_date))
                diff_tup = self.ufc_prediction_tuple(derived_doubled_tuple)
                diffless_feature_set = [feature.replace('_diff','') for feature in self.amazing_feature_set]
                derived_doubled_tuple_localized = derived_doubled_tuple[diffless_feature_set]
                # make a bokeh plot to visualize the prediction in a html file viewable on the website
                # make scaled diff tup 
                diff_tup_scaled = self.imputer.transform(diff_tup)
                if self.scaler is not None:
                    diff_tup_scaled = self.scaler.transform(diff_tup_scaled)
                    diff_tup_scaled = pd.DataFrame(diff_tup_scaled, columns=diff_tup.columns, index=diff_tup.index)
                else:
                    diff_tup_scaled = pd.DataFrame(
                        diff_tup_scaled, columns=diff_tup.columns, index=diff_tup.index
                    )
                card_date_formatted = pd.to_datetime(card_date).strftime('%Y-%m-%d')
                plot_path = Path(
                    'content/bokehPlots/'
                    f'{card_date_formatted}_{fighter.lower().replace(" ", "_")}'
                    f'_vs_{opponent.lower().replace(" ", "_")}_bokeh_barplot.html'
                )
                data_changed = getattr(self.dh, 'update_time', 0) > 0
                if data_changed or not plot_path.exists():
                    try:
                        visualize_prediction_bokeh(
                            fighter, opponent, self.theta, card_date,
                            derived_doubled_tuple_localized, diff_tup_scaled,
                        )
                    except Exception as error:
                        # Plot generation is presentation-only.  A filesystem
                        # or Bokeh regression must not discard an otherwise
                        # valid weekly prediction card.
                        print(
                            f'WARNING: prediction plot failed for {fighter} vs '
                            f'{opponent} ({type(error).__name__}: {error})'
                        )
                odds_calc = self.odds(diff_tup)
                print('predicting: '+fighter,'versus '+opponent,'.... '+str(odds_calc))
                if not odds_calc:
                    print('odds calculation failed for some reason')
                    vegas_odds.loc[i, 'predicted fighter odds'] = None
                    vegas_odds.loc[i, 'predicted opponent odds'] = None
                    continue
                vegas_odds.loc[i, 'predicted fighter odds']=odds_calc[0]
                vegas_odds.loc[i, 'predicted opponent odds']=odds_calc[1]
                
        return vegas_odds
    
    def get_ufc_fights_reported_derived_doubled_for_upcoming_fight(self, fighter1, fighter2, day1=date.today(), day2=date.today()):
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
        # check if anthony hernandez is one of the fighters (TO DEBUG SPECIFIC FIGHTERS)
        # if fighter1 == 'Anthony Hernandez' or fighter2 == 'Anthony Hernandez':
        #     import ipdb;ipdb.set_trace(context=10)
        derived_doubled_tuple = dh.make_derived_doubled_vector_for_fight(new_rows)
        return derived_doubled_tuple
            
    #I've defined this in such a way to predict what happens when fighter1 in their day1 version fights fighter2
    #in their day2 version. Meaning we could compare for example 2014 Tyron Woodley to 2019 Colby Covington
    def ufc_prediction_tuple(self, derived_doubled_tuple):
        dh = self.dh
        ufc_fights_predictive_flattened_diffs = dh.make_ufc_fights_predictive_flattened_diffs(derived_doubled_tuple, shuffle=False)
        prediction_tuple = ufc_fights_predictive_flattened_diffs[self.amazing_feature_set]
        return prediction_tuple
        
        

    # We want to predict how many times out of 10 the winning fighter would win, so we look at the values
    # x*theta+b. If the value is >=0 its a win and <=0 its a loss. Distance from zero gives indication of
    # how likely the outcome is.

    def presigmoid_value(self, diff_tup):
        if self.imputer is not None:
            diff_tup = self.imputer.transform(diff_tup)
        if self.scaler is not None:
            diff_tup = self.scaler.transform(diff_tup)

        value = float(np.dot(np.asarray(diff_tup).reshape(-1), self.theta))
        return value + float(self.b)


    def sigmoid(self, x):
        x = float(np.clip(x, -709, 709))
        return 1 / (1 + math.exp(-x))

    #returns the probability that fighter1 defeats fighter2 on date1,date2
    def probability(self, diff_tup):
        presig=self.presigmoid_value(diff_tup)
        prob = self.sigmoid(presig)
        return prob

    def odds(self, diff_tup):
        p=self.probability(diff_tup)
        
        if p < 0.5:
            fighterOdds = round(100 / p - 100)
            opponentOdds = round(1 / (1 / (1 - p) - 1) * 100)
            return ['+' + str(fighterOdds), '-' + str(opponentOdds)]
        elif p>=.5:
            fighterOdds = round(1 / (1 / p - 1) * 100)
            opponentOdds = round(100 / (1 - p) - 100)
            return ['-' + str(fighterOdds), '+' + str(opponentOdds)]

