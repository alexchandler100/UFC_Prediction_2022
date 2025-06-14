# getting dependencies
import pandas as pd
import random
import requests
import urllib.request
# this imports all of the functions from the file functions.py
from functions import *
from bs4 import BeautifulSoup
from PIL import Image

pd.options.mode.chained_assignment = None # default='warn' (disables SettingWithCopyWarning)

import ipdb;ipdb.set_trace(context=10) # uncomment to debug

# grab current data stored in csv files
fight_hist_old = pd.read_csv('models/buildingMLModel/data/processed/fight_hist.csv')
fighter_stats_old = pd.read_csv('models/buildingMLModel/data/processed/fighter_stats.csv')

# bring csv files up to date and overwrite the old ones
print('scraping new statistics from ufcstats.com')
fight_hist_updated = update_fight_stats(fight_hist_old)
fighter_stats_updated = update_fighter_details(fight_hist_updated.fighter_url.unique(), fighter_stats_old)
fight_hist_updated.to_csv('models/buildingMLModel/data/processed/fight_hist.csv', index=False)
fighter_stats_updated.to_csv('models/buildingMLModel/data/processed/fighter_stats.csv', index=False)

# all stats fight history file which is one update behind fight_hist_updated
ufc_fights_crap = pd.read_csv('models/buildingMLModel/data/processed/ufc_fights_crap.csv', sep=',', low_memory=False)

# most recent fight in fight_hist_updated versus most recent fight in ufc_fights_crap
update_time = time_diff(ufc_fights_crap['date'][0], fight_hist_updated['date'][0])
print('days since last update: '+str(update_time))

# this gives the new rows in fight_hist_updated which do not appear in ufc_fights_crapd
new_rows = fight_hist_updated.loc[time_diff_vect(fight_hist_updated['date'], fight_hist_updated['date'][0]) < update_time].copy()

if update_time > 0: # should just stop the script here. Can do this later once we do everything inside a function call
    populate_new_fights_with_statistics(new_rows)
    # making sure new columns coincide with old columns
    crapcolumns = list(ufc_fights_crap.columns)
    new_rows = new_rows[crapcolumns]
    print('New columns coincide with old columns: ' + str(all(ufc_fights_crap.columns == new_rows.columns)))
    print('joining new data to ufc_fights_crap.csv')

else:
    print('nothing to update')
    
frames = [new_rows, ufc_fights_crap]
updated_ufc_fights_crap = pd.concat(frames, ignore_index=True)

# saving the updated ufc_fights_crap file
updated_ufc_fights_crap.to_csv('models/buildingMLModel/data/processed/ufc_fights_crap.csv', index=False)

# here is the list of all stats available (besides stance), does not include names or result
computed_statistics = [u'fighter_wins', u'fighter_losses', u'fighter_age', u'fighter_height', u'fighter_reach', u'fighter_L5Y_wins', u'fighter_L5Y_losses', 
                       u'fighter_L2Y_wins', u'fighter_L2Y_losses', u'fighter_ko_wins', u'fighter_ko_losses', u'fighter_L5Y_ko_wins', u'fighter_L5Y_ko_losses', 
                       u'fighter_L2Y_ko_wins', u'fighter_L2Y_ko_losses', u'fighter_sub_wins', u'fighter_sub_losses', u'fighter_L5Y_sub_wins', u'fighter_L5Y_sub_losses', 
                       u'fighter_L2Y_sub_wins', u'fighter_L2Y_sub_losses', u'fighter_inf_knockdowns_avg', u'fighter_inf_pass_avg', u'fighter_inf_reversals_avg', 
                       u'fighter_inf_sub_attempts_avg', u'fighter_inf_takedowns_landed_avg', u'fighter_inf_takedowns_attempts_avg', u'fighter_inf_sig_strikes_landed_avg', 
                       u'fighter_inf_sig_strikes_attempts_avg', u'fighter_inf_total_strikes_landed_avg', u'fighter_inf_total_strikes_attempts_avg', 
                       u'fighter_inf_head_strikes_landed_avg', u'fighter_inf_head_strikes_attempts_avg', u'fighter_inf_body_strikes_landed_avg', 
                       u'fighter_inf_body_strikes_attempts_avg', u'fighter_inf_leg_strikes_landed_avg', u'fighter_inf_leg_strikes_attempts_avg', 
                       u'fighter_inf_distance_strikes_landed_avg', u'fighter_inf_distance_strikes_attempts_avg', u'fighter_inf_clinch_strikes_landed_avg', 
                       u'fighter_inf_clinch_strikes_attempts_avg', u'fighter_inf_ground_strikes_landed_avg', u'fighter_inf_ground_strikes_attempts_avg', 
                       u'fighter_abs_knockdowns_avg', u'fighter_abs_pass_avg', u'fighter_abs_reversals_avg', u'fighter_abs_sub_attempts_avg', 
                       u'fighter_abs_takedowns_landed_avg', u'fighter_abs_takedowns_attempts_avg', u'fighter_abs_sig_strikes_landed_avg', 
                       u'fighter_abs_sig_strikes_attempts_avg', u'fighter_abs_total_strikes_landed_avg', u'fighter_abs_total_strikes_attempts_avg', 
                       u'fighter_abs_head_strikes_landed_avg', u'fighter_abs_head_strikes_attempts_avg', u'fighter_abs_body_strikes_landed_avg', 
                       u'fighter_abs_body_strikes_attempts_avg', u'fighter_abs_leg_strikes_landed_avg', u'fighter_abs_leg_strikes_attempts_avg', 
                       u'fighter_abs_distance_strikes_landed_avg', u'fighter_abs_distance_strikes_attempts_avg', u'fighter_abs_clinch_strikes_landed_avg', 
                       u'fighter_abs_clinch_strikes_attempts_avg', u'fighter_abs_ground_strikes_landed_avg', u'fighter_abs_ground_strikes_attempts_avg', u'opponent_wins', 
                       u'opponent_losses', u'opponent_age',  u'opponent_height', u'opponent_reach', u'opponent_L5Y_wins', u'opponent_L5Y_losses', u'opponent_L2Y_wins', 
                       u'opponent_L2Y_losses', u'opponent_ko_wins', u'opponent_ko_losses', u'opponent_L5Y_ko_wins', u'opponent_L5Y_ko_losses', u'opponent_L2Y_ko_wins',
                       u'opponent_L2Y_ko_losses', u'opponent_sub_wins', u'opponent_sub_losses', u'opponent_L5Y_sub_wins', u'opponent_L5Y_sub_losses', 
                       u'opponent_L2Y_sub_wins', u'opponent_L2Y_sub_losses', u'opponent_inf_knockdowns_avg', u'opponent_inf_pass_avg', u'opponent_inf_reversals_avg', 
                       u'opponent_inf_sub_attempts_avg', u'opponent_inf_takedowns_landed_avg', u'opponent_inf_takedowns_attempts_avg', u'opponent_inf_sig_strikes_landed_avg',
                       u'opponent_inf_sig_strikes_attempts_avg', u'opponent_inf_total_strikes_landed_avg', u'opponent_inf_total_strikes_attempts_avg', 
                       u'opponent_inf_head_strikes_landed_avg', u'opponent_inf_head_strikes_attempts_avg', u'opponent_inf_body_strikes_landed_avg',
                       u'opponent_inf_body_strikes_attempts_avg', u'opponent_inf_leg_strikes_landed_avg', u'opponent_inf_leg_strikes_attempts_avg', 
                       u'opponent_inf_distance_strikes_landed_avg', u'opponent_inf_distance_strikes_attempts_avg', u'opponent_inf_clinch_strikes_landed_avg', 
                       u'opponent_inf_clinch_strikes_attempts_avg', u'opponent_inf_ground_strikes_landed_avg', u'opponent_inf_ground_strikes_attempts_avg', 
                       u'opponent_abs_knockdowns_avg', u'opponent_abs_pass_avg', u'opponent_abs_reversals_avg', u'opponent_abs_sub_attempts_avg', 
                       u'opponent_abs_takedowns_landed_avg', u'opponent_abs_takedowns_attempts_avg', u'opponent_abs_sig_strikes_landed_avg', 
                       u'opponent_abs_sig_strikes_attempts_avg', u'opponent_abs_total_strikes_landed_avg', u'opponent_abs_total_strikes_attempts_avg', 
                       u'opponent_abs_head_strikes_landed_avg', u'opponent_abs_head_strikes_attempts_avg', u'opponent_abs_body_strikes_landed_avg', 
                       u'opponent_abs_body_strikes_attempts_avg', u'opponent_abs_leg_strikes_landed_avg', u'opponent_abs_leg_strikes_attempts_avg', 
                       u'opponent_abs_distance_strikes_landed_avg', u'opponent_abs_distance_strikes_attempts_avg', u'opponent_abs_clinch_strikes_landed_avg', 
                       u'opponent_abs_clinch_strikes_attempts_avg', u'opponent_abs_ground_strikes_landed_avg', u'opponent_abs_ground_strikes_attempts_avg', 
                       u'fighter_stance', u'opponent_stance', '1-fight_math', '6-fight_math', '4-fighter_score_diff', '9-fighter_score_diff', '15-fighter_score_diff',]

# list containing all columns of any interest
relevant_list = ['date', 'division', 'fighter', 'opponent', 'result', 'method']
relevant_list.extend(computed_statistics)

# creates a clean file with only columns which are relevant to predicting
updated_ufc_fights = updated_ufc_fights_crap[relevant_list]

# lets randomly remove one of every two copied fights
random_indices = []
for i in range(0, len(updated_ufc_fights['fighter_wins']), 2):
    random_indices.append(random.choice([i, i+1]))

updated_ufc_fights = updated_ufc_fights.drop(random_indices)

print('cleaning data and adding new cleaned columns to ufc_fights.csv')

# we worked hard to build this, lets save it (only run this once we're sure that the new file is correct)
updated_ufc_fights.to_csv(
    'models/buildingMLModel/data/processed/ufc_fights.csv', index=False)

print('sending updated fighter_stats.csv to fighter_stats.json')
# convert fighter_stats.csv to json files to read via javascript in website
csvFilePath = r'models/buildingMLModel/data/processed/fighter_stats.csv'
jsonFilePath = r'models/buildingMLModel/data/external/fighter_stats.json'
make_json(csvFilePath, jsonFilePath, 'name')

updated_ufc_fights_crap['index'] = updated_ufc_fights_crap['fighter']
for i in range(len(updated_ufc_fights_crap['date'])):
    updated_ufc_fights_crap['index'][i] = i

json_columns = ['date', 'result', 'fighter', 'opponent', 'division', 'method', 'round', 'time', 'knockdowns', 'sub_attempts', 'pass', 'reversals', 'takedowns_landed', 
                'takedowns_attempts', 'sig_strikes_landed', 'sig_strikes_attempts', 'total_strikes_landed', 'total_strikes_attempts', 'head_strikes_landed',
                'head_strikes_attempts', 'body_strikes_landed', 'body_strikes_attempts', 'leg_strikes_landed', 'leg_strikes_attempts', 'distance_strikes_landed', 
                'distance_strikes_attempts', 'clinch_strikes_landed', 'clinch_strikes_attempts', 'ground_strikes_landed', 'ground_strikes_attempts', 'fighter_stance', 
                'opponent_stance', 'index',]

ufc_fight_data_for_website = updated_ufc_fights_crap[json_columns]

# make new csv just to send it to json
# this is inefficient and wastes space... but its just because its the only way I know to make a json file
# of the correct format (fix needed but not super important)
print('exporting updated ufc_fights_crap.json for use in javascript portion of website')
ufc_fight_data_for_website.to_csv('models/buildingMLModel/data/processed/ufc_fight_data_for_website.csv', index=False)

# convert ufc_fights_crap.csv to json files to read via javascript in website
csvFilePath = r'models/buildingMLModel/data/processed/ufc_fight_data_for_website.csv'
jsonFilePath = r'models/buildingMLModel/data/external/ufc_fight_data_for_website.json'
make_json(csvFilePath, jsonFilePath, 'index')

# updating the picture scrape
# updated scraped fighter data (after running fight_hist_updated function from UFC_data_scraping file)
fighter_stats = pd.read_csv('models/buildingMLModel/data/processed/fighter_stats.csv', sep=',')
names = list(fighter_stats['name'])

def scrape_pictures(name):
    try:
        URL = "https://www.google.com/search?q="+name+" ufc fighting" + \
            "&sxsrf=ALeKk03xBalIZi7BAzyIRw8R4_KrIEYONg:1620885765119&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjv44CC_sXwAhUZyjgGHSgdAQ8Q_AUoAXoECAEQAw&cshid=1620885828054361"
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        # ... or ... image_tags = soup.find_all('img', class_='t0fcAb')
        image_tags = soup.find_all('img')
        links = []
        for image_tag in image_tags:
            links.append(image_tag['src'])
            name_reduced = name.replace(" ", "")
        for i in range(1, 5):
            urllib.request.urlretrieve(
                links[i], "models/buildingMLModel/images/"+str(i)+name_reduced+".jpg")
        print('scraped 5 random pictures of '+name+' from Google search')

    except:
        print('The scrape did not work for '+name)


print('Scraping pictures of newly added fighters from Google image search')
# run this to update the image scrape
for name in names:
    try:
        name_reduced = name.replace(" ", "")
        k = Image.open("models/buildingMLModel/images/" +
                       str(1)+name_reduced+".jpg")
    except:
        scrape_pictures(name)
