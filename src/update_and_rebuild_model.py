#getting dependencies
import pandas as pd
from fight_stat_helpers import *

pd.options.mode.chained_assignment = None  # default='warn' (disables SettingWithCopyWarning)

from data_handler import DataHandler
from fight_predictor import FightPredictor

# create a data handler object to access the data stored in csvs and jsons
# has built-in dataframes mirroring the csvs and jsons
# has built-in methods to update the csvs and jsons from ufcstats.com
dh = DataHandler()

# bring csv files up to date and overwrite the old ones
print('scraping new statistics from ufcstats.com')
dh.update_data_csvs_and_jsons()

fight_predictor = FightPredictor(dh.get('ufc_fights'))
fight_predictor.train_logistic_regression_model()

theta, b = fight_predictor.get_regression_coeffs_and_intercept()

# use the data handler to update the model coefficients in the json files
dh.set_regression_coeffs_and_intercept(theta, b)
print("Saved new regression coefficients and intercept to json files to run model in website")

# TODO HAVE THE DATA HANDLER DO THIS
#print('scraping bookie odds from bestfightodds.com')
#odds_df = get_odds()
#odds_df=drop_irrelevant_fights(odds_df,3) #allows 3 bookies to have missing odds. can increase this to 2 or 3 as needed
#odds_df=drop_non_ufc_fights(odds_df)
#odds_df=drop_repeats(odds_df)
#print('saving odds to content/data/external/vegas_odds.json')
#save to json
#result = odds_df.to_json()
#parsed = json.loads(result)
#jsonFilePath='content/data/external/vegas_odds.json'
#with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
#    jsonf.write(json.dumps(parsed, indent=4))
#print('saved to '+jsonFilePath)

print('Saving results of previous card to prediction_history.json')
# now that the previous card which we made predictions for has happened, we can add the results to the prediction history
# vegas odds is always a week ahead of the prediction history, so we can use it to update the prediction history by comparing vegas_odds and ufc_fights_crap which contains the results from last week
dh.update_prediction_history()

print('Scraping next ufc fight card from bestfightodds.com')
print("###############################################################################################################")    
dh.update_card_info()
card_date, card_title, fights_list = dh.get_next_fight_card()
prediction_history = dh.get('prediction_history', filetype='json')
vegas_odds = fight_predictor.predict_upcoming_fights(prediction_history, fights_list, card_date, theta, b)
dh.update_vegas_odds(vegas_odds)
print('saving scraped fights and predictions to content/data/external/vegas_odds.json')
print('TODO: scrape odds too. Currently only scraping names, date, and card title')
print("###############################################################################################################")
            

