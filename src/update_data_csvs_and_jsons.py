# getting dependencies
import pandas as pd
import numpy as npy
from datetime import datetime, date, timedelta
import random
import requests
import urllib.request
# this imports all of the functions from the file functions.py
from functions import *
from bs4 import BeautifulSoup
from PIL import Image

# default='warn' (disables SettingWithCopyWarning)
pd.options.mode.chained_assignment = None

print('scraping new statistics from ufcstats.com')
# Run this cell to bring csv files up to date. Last ran March 16 2022.
fight_hist_old = pd.read_csv('models/buildingMLModel/data/processed/fight_hist.csv')
fighter_stats_old = pd.read_csv('models/buildingMLModel/data/processed/fighter_stats.csv')
fight_hist_updated = update_fight_stats(fight_hist_old)
fighter_stats_updated = update_fighter_details(fight_hist_updated.fighter_url.unique(), fighter_stats_old)
fight_hist_updated.to_csv('models/buildingMLModel/data/processed/fight_hist.csv', index=False)
fighter_stats_updated.to_csv('models/buildingMLModel/data/processed/fighter_stats.csv', index=False)

# At the end, we have files fight_hist.csv and fighter_stats.csv which we can use to construct a dataframe in building_ufc_fights and building_ufc_fighters
# updated scraped fight data (after running fight_hist_updated function from UFC_data_scraping file)
fight_hist = pd.read_csv('models/buildingMLModel/data/processed/fight_hist.csv', sep=',')
# all stats fight history file which is one update behind fight_hist
ufcfightscrap = pd.read_csv('models/buildingMLModel/data/processed/ufc_fights_crap.csv', sep=',', low_memory=False)
# updated scraped fighter data (after running fight_hist_updated function from UFC_data_scraping file)
ufcfighterscrap = pd.read_csv('models/buildingMLModel/data/processed/fighter_stats.csv', sep=',')
# most recent fight in fight_hist versus most recent fight in ufcfightscrap

update_time = time_diff(ufcfightscrap['date'][0], fight_hist['date'][0])
print('days since last update: '+str(update_time))
# this gives the new rows in fight_hist which do not appear in ufcfightscrapd
new_rows = fight_hist.loc[time_diff_vect(fight_hist['date'], fight_hist['date'][0]) < update_time]
# convert to numpy array
numpy_new_rows = new_rows.values
# gives new_rows as a dataframe instead of a clone
test_new_rows = pd.DataFrame(data = numpy_new_rows, columns = new_rows.columns)

if update_time > 0:
    print('adding physical statistics for fighter')
    test_new_rows['fighter_wins'] = wins_before_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_losses'] = losses_before_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_age'] = fighter_age_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_height'] = test_new_rows['fighter'].apply(fighter_height)
    test_new_rows['fighter_reach'] = test_new_rows['fighter'].apply(fighter_height)

    print('adding record statistics for fighter')
    test_new_rows['fighter_L5Y_wins'] = L5Y_wins_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_L5Y_losses'] = L5Y_losses_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_L2Y_wins'] = L2Y_wins_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_L2Y_losses'] = L2Y_losses_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_ko_wins'] = ko_wins_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_ko_losses'] = ko_losses_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_L5Y_ko_wins'] = L5Y_ko_wins_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_L5Y_ko_losses'] = L5Y_ko_losses_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_L2Y_ko_wins'] = L2Y_ko_wins_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_L2Y_ko_losses'] = L2Y_ko_losses_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_sub_wins'] = sub_wins_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_sub_losses'] = sub_losses_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_L5Y_sub_wins'] = L5Y_sub_wins_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_L5Y_sub_losses'] = L5Y_sub_losses_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_L2Y_sub_wins'] = L2Y_sub_wins_vect(test_new_rows['fighter'], test_new_rows['date'])
    test_new_rows['fighter_L2Y_sub_losses'] = L2Y_sub_losses_vect(test_new_rows['fighter'], test_new_rows['date'])

    print('adding inflicted punch, kick, grappling statistics for fighter... this will take a few minutes')

    test_new_rows['fighter_inf_knockdowns_avg'] = avg_count_vect('knockdowns', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_pass_avg'] = avg_count_vect('pass', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_reversals_avg'] = zero_vect(test_new_rows['fighter'])
    test_new_rows['fighter_inf_sub_attempts_avg'] = avg_count_vect('sub_attempts', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_takedowns_landed_avg'] = avg_count_vect('takedowns_landed', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_takedowns_attempts_avg'] = avg_count_vect('takedowns_attempts', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    print('quarter done')
    test_new_rows['fighter_inf_sig_strikes_landed_avg'] = avg_count_vect('sig_strikes_landed', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_sig_strikes_attempts_avg'] = avg_count_vect('sig_strikes_attempts', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_total_strikes_landed_avg'] = avg_count_vect('total_strikes_landed', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_total_strikes_attempts_avg'] = avg_count_vect('total_strikes_attempts', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_head_strikes_landed_avg'] = avg_count_vect('head_strikes_landed', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_head_strikes_attempts_avg'] = avg_count_vect('head_strikes_attempts', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    print('half done')
    test_new_rows['fighter_inf_body_strikes_landed_avg'] = avg_count_vect('body_strikes_landed', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_body_strikes_attempts_avg'] = avg_count_vect('body_strikes_attempts', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_leg_strikes_landed_avg'] = avg_count_vect('leg_strikes_landed', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_leg_strikes_attempts_avg'] = avg_count_vect('leg_strikes_attempts', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    print('almost done')
    test_new_rows['fighter_inf_distance_strikes_attempts_avg'] = avg_count_vect('distance_strikes_attempts', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_clinch_strikes_landed_avg'] = avg_count_vect('clinch_strikes_landed', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_clinch_strikes_attempts_avg'] = avg_count_vect('clinch_strikes_attempts', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_ground_strikes_landed_avg'] = avg_count_vect('ground_strikes_landed', test_new_rows['fighter'], 'inf', test_new_rows['date'])
    test_new_rows['fighter_inf_ground_strikes_attempts_avg'] = avg_count_vect('ground_strikes_attempts', test_new_rows['fighter'], 'inf', test_new_rows['date'])

    print('adding absorbed punch, kick, grappling statistics for fighter... this will take a few minutes')

    test_new_rows['fighter_abs_knockdowns_avg'] = avg_count_vect('knockdowns', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_pass_avg'] = avg_count_vect('pass', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_reversals_avg'] = zero_vect(test_new_rows['fighter'])
    test_new_rows['fighter_abs_sub_attempts_avg'] = avg_count_vect('sub_attempts', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_takedowns_landed_avg'] = avg_count_vect('takedowns_landed', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_takedowns_attempts_avg'] = avg_count_vect('takedowns_attempts', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    print('quarter done')
    test_new_rows['fighter_abs_sig_strikes_landed_avg'] = avg_count_vect('sig_strikes_landed', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_sig_strikes_attempts_avg'] = avg_count_vect('sig_strikes_attempts', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_total_strikes_landed_avg'] = avg_count_vect('total_strikes_landed', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_total_strikes_attempts_avg'] = avg_count_vect('total_strikes_attempts', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_head_strikes_landed_avg'] = avg_count_vect('head_strikes_landed', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_head_strikes_attempts_avg'] = avg_count_vect('head_strikes_attempts', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    print('half done')
    test_new_rows['fighter_abs_body_strikes_landed_avg'] = avg_count_vect('body_strikes_landed', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_body_strikes_attempts_avg'] = avg_count_vect('body_strikes_attempts', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_leg_strikes_landed_avg'] = avg_count_vect('leg_strikes_landed', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_leg_strikes_attempts_avg'] = avg_count_vect('leg_strikes_attempts', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    print('almost done')
    test_new_rows['fighter_abs_distance_strikes_attempts_avg'] = avg_count_vect('distance_strikes_attempts', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_clinch_strikes_landed_avg'] = avg_count_vect('clinch_strikes_landed', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_clinch_strikes_attempts_avg'] = avg_count_vect('clinch_strikes_attempts', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_ground_strikes_landed_avg'] = avg_count_vect('ground_strikes_landed', test_new_rows['fighter'], 'abs', test_new_rows['date'])
    test_new_rows['fighter_abs_ground_strikes_attempts_avg'] = avg_count_vect('ground_strikes_attempts', test_new_rows['fighter'], 'abs', test_new_rows['date'])

    print('adding physical statistics for opponent')

    test_new_rows['opponent_wins'] = opponent_column('fighter_wins')
    test_new_rows['opponent_losses'] = opponent_column('fighter_losses')
    test_new_rows['opponent_age'] = opponent_column('fighter_age')
    test_new_rows['opponent_height'] = opponent_column('fighter_height')
    test_new_rows['opponent_reach'] = opponent_column('fighter_reach')

    print('adding record statistics for opponent')

    test_new_rows['opponent_L5Y_wins'] = opponent_column('fighter_L5Y_wins')
    test_new_rows['opponent_L5Y_losses'] = opponent_column('fighter_L5Y_losses')
    test_new_rows['opponent_L2Y_wins'] = opponent_column('fighter_L2Y_wins')
    test_new_rows['opponent_L2Y_losses'] = opponent_column('fighter_L2Y_losses')
    test_new_rows['opponent_ko_wins'] = opponent_column('fighter_ko_wins')
    test_new_rows['opponent_ko_losses'] = opponent_column('fighter_ko_losses')
    test_new_rows['opponent_L5Y_ko_wins'] = opponent_column('fighter_L5Y_ko_wins')
    test_new_rows['opponent_L5Y_ko_losses'] = opponent_column('fighter_L5Y_ko_losses')
    test_new_rows['opponent_L2Y_ko_wins'] = opponent_column('fighter_L2Y_ko_wins')
    test_new_rows['opponent_L2Y_ko_losses'] = opponent_column('fighter_L2Y_ko_losses')
    test_new_rows['opponent_sub_wins'] = opponent_column('fighter_sub_wins')
    test_new_rows['opponent_sub_losses'] = opponent_column('fighter_sub_losses')
    test_new_rows['opponent_L5Y_sub_wins'] = opponent_column('fighter_L5Y_sub_wins')
    test_new_rows['opponent_L5Y_sub_losses'] = opponent_column('fighter_L5Y_sub_losses')
    test_new_rows['opponent_L2Y_sub_wins'] = opponent_column('fighter_L2Y_sub_wins')
    test_new_rows['opponent_L2Y_sub_losses'] = opponent_column('fighter_L2Y_sub_losses')

    print('adding inflicted punch, kick, grappling statistics for opponent... this will take a few minutes')

    test_new_rows['opponent_inf_knockdowns_avg'] = avg_count_vect('knockdowns', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_pass_avg'] = avg_count_vect('pass', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_reversals_avg'] = zero_vect(test_new_rows['opponent'])
    test_new_rows['opponent_inf_sub_attempts_avg'] = avg_count_vect('sub_attempts', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_takedowns_landed_avg'] = avg_count_vect('takedowns_landed', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_takedowns_attempts_avg'] = avg_count_vect('takedowns_attempts', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    print('quarter done')
    test_new_rows['opponent_inf_sig_strikes_landed_avg'] = avg_count_vect('sig_strikes_landed', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_sig_strikes_attempts_avg'] = avg_count_vect('sig_strikes_attempts', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_total_strikes_landed_avg'] = avg_count_vect('total_strikes_landed', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_total_strikes_attempts_avg'] = avg_count_vect('total_strikes_attempts', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_head_strikes_landed_avg'] = avg_count_vect('head_strikes_landed', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    print('half done')
    test_new_rows['opponent_inf_head_strikes_attempts_avg'] = avg_count_vect('head_strikes_attempts', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_body_strikes_landed_avg'] = avg_count_vect('body_strikes_landed', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_body_strikes_attempts_avg'] = avg_count_vect('body_strikes_attempts', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_leg_strikes_landed_avg'] = avg_count_vect('leg_strikes_landed', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_leg_strikes_attempts_avg'] = avg_count_vect('leg_strikes_attempts', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    print('almost done')
    test_new_rows['opponent_inf_distance_strikes_attempts_avg'] = avg_count_vect('distance_strikes_attempts', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_clinch_strikes_landed_avg'] = avg_count_vect('clinch_strikes_landed', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_clinch_strikes_attempts_avg'] = avg_count_vect('clinch_strikes_attempts', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_ground_strikes_landed_avg'] = avg_count_vect('ground_strikes_landed', test_new_rows['opponent'], 'inf', test_new_rows['date'])
    test_new_rows['opponent_inf_ground_strikes_attempts_avg'] = avg_count_vect('ground_strikes_attempts', test_new_rows['opponent'], 'inf', test_new_rows['date'])

    print('adding absorbed punch, kick, grappling statistics for opponent... this will take a few minutes')

    test_new_rows['opponent_abs_knockdowns_avg'] = avg_count_vect('knockdowns', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_pass_avg'] = avg_count_vect('pass', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_reversals_avg'] = zero_vect(test_new_rows['opponent'])
    test_new_rows['opponent_abs_sub_attempts_avg'] = avg_count_vect('sub_attempts', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_takedowns_landed_avg'] = avg_count_vect('takedowns_landed', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_takedowns_attempts_avg'] = avg_count_vect('takedowns_attempts', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_sig_strikes_landed_avg'] = avg_count_vect('sig_strikes_landed', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    print('quarter done')
    test_new_rows['opponent_abs_sig_strikes_attempts_avg'] = avg_count_vect('sig_strikes_attempts', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_total_strikes_landed_avg'] = avg_count_vect('total_strikes_landed', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_total_strikes_attempts_avg'] = avg_count_vect('total_strikes_attempts', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_head_strikes_landed_avg'] = avg_count_vect('head_strikes_landed', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_head_strikes_attempts_avg'] = avg_count_vect('head_strikes_attempts', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_body_strikes_landed_avg'] = avg_count_vect('body_strikes_landed', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    print('half done')
    test_new_rows['opponent_abs_body_strikes_attempts_avg'] = avg_count_vect('body_strikes_attempts', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_leg_strikes_landed_avg'] = avg_count_vect('leg_strikes_landed', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_leg_strikes_attempts_avg'] = avg_count_vect('leg_strikes_attempts', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    print('almost done')
    test_new_rows['opponent_abs_distance_strikes_attempts_avg'] = avg_count_vect('distance_strikes_attempts', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_clinch_strikes_landed_avg'] = avg_count_vect('clinch_strikes_landed', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_clinch_strikes_attempts_avg'] = avg_count_vect('clinch_strikes_attempts', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_ground_strikes_landed_avg'] = avg_count_vect('ground_strikes_landed', test_new_rows['opponent'], 'abs', test_new_rows['date'])
    test_new_rows['opponent_abs_ground_strikes_attempts_avg'] = avg_count_vect('ground_strikes_attempts', test_new_rows['opponent'], 'abs', test_new_rows['date'])

    test_new_rows['fighter_stance'] = stance_vect(test_new_rows['fighter'])
    test_new_rows['opponent_stance'] = stance_vect(test_new_rows['opponent'])

    print('adding fight_math and fighter_score statistics')
    test_new_rows['1-fight_math'] = fight_math_diff_vect(test_new_rows['fighter'], test_new_rows['opponent'], test_new_rows['date'], 1)
    test_new_rows['6-fight_math'] = fight_math_diff_vect(test_new_rows['fighter'], test_new_rows['opponent'], test_new_rows['date'], 6)
    test_new_rows['4-fighter_score_diff'] = fighter_score_diff_vect(test_new_rows['fighter'], test_new_rows['opponent'], test_new_rows['date'], 4)
    test_new_rows['9-fighter_score_diff'] = fighter_score_diff_vect(test_new_rows['fighter'], test_new_rows['opponent'], test_new_rows['date'], 9)
    test_new_rows['15-fighter_score_diff'] = fighter_score_diff_vect(test_new_rows['fighter'], test_new_rows['opponent'], test_new_rows['date'], 15)

    # making sure new columns coincide with old columns
    crapcolumns = list(ufcfightscrap.columns)
    test_new_rows = test_new_rows[crapcolumns]
    print('New columns coincide with old columns: ' + str(all(ufcfightscrap.columns == test_new_rows.columns)))
    print('joining new data to ufc_fights_crap.csv')

else:
    print('nothing to update')
frames = [test_new_rows, ufcfightscrap]
updated_ufcfightscrap = pd.concat(frames, ignore_index=True)

# saving the updated ufcfightscrap file
updated_ufcfightscrap.to_csv('models/buildingMLModel/data/processed/ufc_fights_crap.csv', index=False)

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
updated_ufc_fights = updated_ufcfightscrap[relevant_list]

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

updated_ufcfightscrap['index'] = updated_ufcfightscrap['fighter']
for i in range(len(updated_ufcfightscrap['date'])):
    updated_ufcfightscrap['index'][i] = i

json_columns = ['date', 'result', 'fighter', 'opponent', 'division', 'method', 'round', 'time', 'knockdowns', 'sub_attempts', 'pass', 'reversals', 'takedowns_landed', 
                'takedowns_attempts', 'sig_strikes_landed', 'sig_strikes_attempts', 'total_strikes_landed', 'total_strikes_attempts', 'head_strikes_landed',
                'head_strikes_attempts', 'body_strikes_landed', 'body_strikes_attempts', 'leg_strikes_landed', 'leg_strikes_attempts', 'distance_strikes_landed', 
                'distance_strikes_attempts', 'clinch_strikes_landed', 'clinch_strikes_attempts', 'ground_strikes_landed', 'ground_strikes_attempts', 'fighter_stance', 
                'opponent_stance', 'index',]

ufcfightscrap_for_json = updated_ufcfightscrap[json_columns]

# make new csv just to send it to json
# this is inefficient and wastes space... but its just because its the only way I know to make a json file
# of the correct format (fix needed but not super important)
print('exporting updated ufcfightscrap.json for use in javascript portion of website')
ufcfightscrap_for_json.to_csv('models/buildingMLModel/data/processed/ufcfightscrap.csv', index=False)

# convert fighter_stats.csv to json files to read via javascript in website
csvFilePath = r'models/buildingMLModel/data/processed/ufcfightscrap.csv'
jsonFilePath = r'models/buildingMLModel/data/external/ufcfightscrap.json'
make_json(csvFilePath, jsonFilePath, 'index')

# updating the picture scrape
names = list(ufcfighterscrap['name'])


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
