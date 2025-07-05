import git, os, sys
git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
os.chdir(f'{git_root}/src')
sys.path.append(os.path.abspath(os.path.join(f'{git_root}/src')))
print(f'Changed working directory to {os.getcwd()}')

from data_handler import DataHandler
from fight_predictor import FightPredictor
dh = DataHandler()
dh.update_card_info()
card_date, card_title, fights_list = dh.get_next_fight_card()
prediction_history = dh.get('prediction_history', filetype='json')
fight_predictor = FightPredictor(dh.get('ufc_fights'))
fight_predictor.train_logistic_regression_model()
theta, b = fight_predictor.get_regression_coeffs_and_intercept()
predicted_odds_df = fight_predictor.predict_upcoming_fights(prediction_history, fights_list, card_date, theta, b)
predicted_odds_df_with_vegas_odds = dh.save_fightoddsio_to_vegas_odds_json_and_merge_with_predictions_df(predicted_odds_df)
dh.update_vegas_odds(predicted_odds_df_with_vegas_odds)