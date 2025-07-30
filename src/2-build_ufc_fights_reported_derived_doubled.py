import git, os
git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
os.chdir(f'{git_root}/src')
from data_handler import DataHandler

import pandas as pd
import numpy as np
from datetime import date, datetime
import random
import requests
from bs4 import BeautifulSoup
import re

#have to change directory to import functions after April 2022 restructure of folders
from fight_stat_helpers import *

fighter_stats_path = f'{git_root}/src/content/data/processed/fighter_stats.csv'
fighter_stats = pd.read_csv(fighter_stats_path)

ufc_fights_reported_doubled_path = f'{git_root}/src/content/data/processed/ufc_fights_reported_doubled.csv'
ufc_fights_reported_doubled = pd.read_csv(ufc_fights_reported_doubled_path)
ufc_fights_reported_doubled['date'] = pd.to_datetime(ufc_fights_reported_doubled['date'], errors='coerce')
ufc_fights_reported_doubled = ufc_fights_reported_doubled.loc[::-1]
ufc_fights_reported_doubled.set_index('date', inplace=True)
print(f'Loaded existing fight history with stats from {ufc_fights_reported_doubled_path}, shape {ufc_fights_reported_doubled.shape}')

ufc_fights_reported_derived_doubled = ufc_fights_reported_doubled[['fighter', 'opponent', 'result', 'method', 'division']].copy()

# GOAL reproduce these statistics 

# TODO MAKE THIS GRABBABLE WITH A FUNCTION CALL
physical_stats = [u'age', u'height', u'reach', u'stance']
# the rest will have total, l2y and l5y versions
record_stats = [u'wins', u'losses', u'wins_ko', u'wins_sub', u'wins_dec', u'losses_ko', u'losses_sub' u'losses_dec']
# the following will also have inflicted (inf) and absorbed (abs) versions
grappling_event_stats = [u'reversals', u'control', u'sub_attempts']
striking_event_stats = [u'knockdowns']
# the following will also have attempts and landed versions
grappling_stats = [u'takedowns']
striking_stats = [u'sig_strikes', u'total_strikes', u'head_strikes', u'body_strikes', u'leg_strikes', u'distance_strikes', u'clinch_strikes', u'ground_strikes']

# NOTE we don't actually use these all_..._stats lists, but they are useful to see what combinations we are making
all_record_stats = []
for stat in record_stats:
    for timeframe in ['all', 'l2y', 'l5y']:
        all_record_stats.append(f'{timeframe}_{stat}')
        
all_grappling_event_stats = []
for stat in grappling_event_stats:
    for timeframe in ['all', 'l2y', 'l5y']:
        for inf_abs in ['inf', 'abs']:
            all_grappling_event_stats.append(f'{timeframe}_{stat}_{inf_abs}')
            
all_striking_event_stats = []
for stat in striking_event_stats:
    for timeframe in ['all', 'l2y', 'l5y']:
        for inf_abs in ['inf', 'abs']:
            all_striking_event_stats.append(f'{timeframe}_{stat}_{inf_abs}')
            
all_grappling_stats = []
for stat in grappling_stats:
    for timeframe in ['all', 'l2y', 'l5y']:
        for inf_abs in ['inf', 'abs']:
            for land_att in ['landed', 'attempts']:
                all_grappling_stats.append(f'{timeframe}_{stat}_{inf_abs}_{land_att}')
                
all_striking_stats = []
for stat in striking_stats:
    for timeframe in ['all', 'l2y', 'l5y']:
        for inf_abs in ['inf', 'abs']:
            for land_att in ['landed', 'attempts']:
                all_striking_stats.append(f'{timeframe}_{stat}_{inf_abs}_{land_att}')

for idx, name in enumerate(fighter_stats.name):
    if idx % 100 == 0:
        print(f'Processing fighter {idx+1}/{len(fighter_stats)}: {name}')
    fighter_inf_mask = same_name_vect(ufc_fights_reported_doubled['fighter'], name)
    fighter_mask = fighter_inf_mask # mask where the fighter is the given name (choosing to make this name without the word inf to avoid confusion later)
    fighter_abs_mask = same_name_vect(ufc_fights_reported_doubled['opponent'], name)
    localized_df     = ufc_fights_reported_doubled[fighter_inf_mask].copy() # to store results of computations for this fighter
    localized_df_inf = ufc_fights_reported_doubled[fighter_inf_mask].copy() # to compute inflicted stats for this fighter
    localized_df_abs = ufc_fights_reported_doubled[fighter_abs_mask].copy() # to compute absorbed stats for this fighter
    
    fighter_1_wins_mask = ufc_fights_reported_doubled['result'] == 'W'
    # Use these to do stuff like fight math and fighter score
    global_inf_wins_mask = fighter_inf_mask & fighter_1_wins_mask
    global_inf_losses_mask = fighter_inf_mask & ~fighter_1_wins_mask
    # global_abs_wins_mask = fighter_abs_mask & ~fighter_1_wins_mask  # if the fighter is the opponent and the result is L, then the fighter won
    # global_abs_losses_mask = fighter_abs_mask & fighter_1_wins_mask  # if the fighter is the opponent and the result is W, then the fighter lost
    
    # make localized versions of the masks above     
    localized_inf_wins_mask = localized_df_inf['result'] == 'W'
    localized_inf_losses_mask = localized_df_inf['result'] == 'L'
    # localized_abs_wins_mask = localized_df_abs['result'] == 'L'
    # localized_abs_losses_mask = localized_df_abs['result'] == 'W'
    
    # make corresponding dataframes
    localized_inf_wins_df = localized_df_inf[localized_inf_wins_mask].copy()
    localized_inf_losses_df = localized_df_inf[localized_inf_losses_mask].copy()
    # localized_abs_wins_df = localized_df_abs[localized_abs_wins_mask].copy()
    # localized_abs_losses_df = localized_df_abs[localized_abs_losses_mask].copy()
    
    # find all people who this fighter has beaten
    fighter_has_beaten = localized_inf_wins_df['opponent'].unique()
    # find all people who this fighter has lost to
    fighter_has_lost_to = localized_inf_losses_df['opponent'].unique()
    # make dataframe of all fights where fighter won or someone they beat won
    fighter_2deg_of_sep_wins_df = ufc_fights_reported_doubled.loc[fighter_inf_mask | (ufc_fights_reported_doubled['fighter'].isin(fighter_has_beaten) & (ufc_fights_reported_doubled['result'] == 'W'))].copy()
    # make dataframe of all fights where fighter lost or someone they lost to lost
    fighter_2deg_of_sep_loss_df = ufc_fights_reported_doubled.loc[fighter_inf_mask | (ufc_fights_reported_doubled['fighter'].isin(fighter_has_lost_to) & (ufc_fights_reported_doubled['result'] == 'L'))].copy()
    # maybe include cross terms too, e.g. losses of people you beat or wins of people you lost to
    
    # compute record stats first
    record_indicator_df = pd.DataFrame(index=localized_df.index)  # will hold indicators for wins, losses, etc for cumsum calculations
    # do some cumsum computations to make rolling averages for the fighter
    record_indicator_df['won'] = (localized_df['result'] == 'W').astype(int)
    record_indicator_df['won_ko'] = ((localized_df['result'] == 'W') & (localized_df['method'].str.contains('KO|TKO', na=False))).astype(int)
    record_indicator_df['won_sub'] = ((localized_df['result'] == 'W') & (localized_df['method'].str.contains('SUB', na=False))).astype(int)
    record_indicator_df['won_dec'] = ((localized_df['result'] == 'W') & (localized_df['method'].str.contains('DEC', na=False))).astype(int)
    record_indicator_df['lost'] = (localized_df['result'] == 'L').astype(int)
    record_indicator_df['lost_ko'] = ((localized_df['result'] == 'L') & (localized_df['method'].str.contains('KO|TKO', na=False))).astype(int)
    record_indicator_df['lost_sub'] = ((localized_df['result'] == 'L') & (localized_df['method'].str.contains('SUB', na=False))).astype(int)
    record_indicator_df['lost_dec'] = ((localized_df['result'] == 'L') & (localized_df['method'].str.contains('DEC', na=False))).astype(int)
    record_indicator_df['num_fights'] = np.arange(0, len(localized_df))  # cumulative fights
    # column of all ones to use for cumsum calculations
    record_indicator_df['ones'] = 1
    
    stats_to_add_to_main_df = [] # keep track of new columns we are adding to the main df (to avoid highly fragmented df warning)
    new_columns_dict = {}
    
    # add physical stats (age, height, reach, stance) which don't need rolling averages
    physical_stats = [u'age', u'height', u'reach', u'stance']

    fighter_stats_results = get_fighter_stats(name, fighter_stats)
    if fighter_stats_results is None:
        print(f'Warning: Either no stats or too many stats found for fighter {name}, not populating fighters statistics')
        continue
    height, reach, dob, stance_ = fighter_stats_results
    # use dob to compute age at time of fight
    if pd.isna(dob):
        age_series = pd.Series([np.nan] * len(localized_df), index=localized_df.index)
    else:
        dob_date = pd.to_datetime(dob, errors='coerce')
        age_series = (localized_df.index - dob_date).days / 365.25 # ends up in weird format so we need to convert to a numpy array first
        age_series = pd.Series(np.array(age_series), index=localized_df.index)
    # add stats to new_columns_dict
    new_columns_dict['age'] = age_series
    new_columns_dict['height'] = pd.Series([height] * len(localized_df), index=localized_df.index)
    new_columns_dict['reach'] = pd.Series([reach] * len(localized_df), index=localized_df.index)
    new_columns_dict['stance'] = pd.Series([stance_] * len(localized_df), index=localized_df.index)
    stats_to_add_to_main_df.extend(['age', 'height', 'reach', 'stance'])
    
    
    # record statistic columns 
    for stat_name, stat_indicator in zip(['wins', 'wins_ko', 'wins_sub', 'wins_dec', 'losses', 'losses_ko', 'losses_sub', 'losses_dec', 'num_fights'],
                                         ['won', 'won_ko', 'won_sub', 'won_dec', 'lost', 'lost_ko', 'lost_sub', 'lost_dec', 'ones']):
        for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
            new_col_name = f'{timeframe}_{stat_name}'
            stats_to_add_to_main_df.append(new_col_name)
            new_columns_dict[new_col_name] = make_cumsum_before_current_fight(record_indicator_df, stat_indicator, timeframe=timeframe)
            
    # captures the quality of wins and losses
    # compute 2nd degree of separation stats (e.g. wins of people you beat, losses of people you lost to)
        
    fighter_2deg_wins_mask = same_name_vect(fighter_2deg_of_sep_wins_df['fighter'], name)
    fighter_2deg_losses_mask = same_name_vect(fighter_2deg_of_sep_loss_df['fighter'], name)
    for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
        new_col_name = f'{timeframe}_wins_wins'
        stats_to_add_to_main_df.append(new_col_name)
        wins_wins_extended = count_wins_wins_before_fight(fighter_2deg_of_sep_wins_df, name, timeframe=timeframe)
        # get the sub series that has the fighter as the fighter (not opponent)
        wins_wins = wins_wins_extended[fighter_2deg_wins_mask]
        new_columns_dict[new_col_name] = wins_wins
        
        new_col_name = f'{timeframe}_losses_losses'
        stats_to_add_to_main_df.append(new_col_name)
        losses_losses_extended = count_losses_losses_before_fight(fighter_2deg_of_sep_loss_df, name, timeframe=timeframe)
        # get the sub series that has the fighter as the fighter (not opponent)
        losses_losses = losses_losses_extended[fighter_2deg_losses_mask]
        new_columns_dict[new_col_name] = losses_losses
        # TODO MAYBE TRY THIS?
        # new_col_name = f'{timeframe}_wins_losses'
        # new_col_name = f'{timeframe}_losses_wins'
        
        new_col_name = f'{timeframe}_fight_math'
        stats_to_add_to_main_df.append(new_col_name)
        fight_math_extended = fight_math(name, fighter_2deg_of_sep_wins_df, timeframe)
        fight_math_col = fight_math_extended[fighter_2deg_wins_mask]
        new_columns_dict[new_col_name] = fight_math_col
        
            
    # compute grappling stats
    for stat in grappling_event_stats:
        col_name = f'{stat}'
        for inf_abs in ['inf', 'abs']:
            for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                new_col_name = f'{timeframe}_{inf_abs}_{col_name}_per_min'
                stats_to_add_to_main_df.append(new_col_name)
                if inf_abs == 'inf':
                    new_columns_dict[new_col_name] = make_avg_before_current_fight(
                        localized_df_inf, # use inflicted stats df to find inflicted stats
                        col_name, 
                        timeframe=timeframe, 
                        landed_attempts=None
                        )
                elif inf_abs == 'abs':
                    new_columns_dict[new_col_name] = make_avg_before_current_fight(
                        localized_df_abs, # use absorbed stats df to find absorbed stats
                        col_name, 
                        timeframe=timeframe, 
                        landed_attempts=None
                        )
                    
    for stat in striking_event_stats:
        col_name = f'{stat}'
        for inf_abs in ['inf', 'abs']:
            for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                new_col_name = f'{timeframe}_{inf_abs}_{col_name}_per_min'
                stats_to_add_to_main_df.append(new_col_name)
                if inf_abs == 'inf':
                    new_columns_dict[new_col_name] = make_avg_before_current_fight(
                        localized_df_inf, # use inflicted stats df to find inflicted stats
                        col_name, 
                        timeframe=timeframe, 
                        landed_attempts=None
                        )
                elif inf_abs == 'abs':
                    new_columns_dict[new_col_name] = make_avg_before_current_fight(
                        localized_df_abs, # use absorbed stats df to find absorbed stats
                        col_name, 
                        timeframe=timeframe, 
                        landed_attempts=None
                        )
                    
    # adding grappling stats
    for stat in grappling_stats:
        col_name = f'{stat}'
        for inf_abs in ['inf', 'abs']:
            for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                for landed_attempts in ['landed', 'attempts']:
                    new_col_name_per_min = f'{timeframe}_{inf_abs}_{col_name}_{landed_attempts}_per_min'
                    stats_to_add_to_main_df.append(new_col_name_per_min)
                    if inf_abs == 'inf':
                        new_columns_dict[new_col_name_per_min] = make_avg_before_current_fight(
                            localized_df_inf, # use inflicted stats df to find inflicted stats
                            col_name, 
                            timeframe=timeframe, 
                            landed_attempts=landed_attempts
                            )
                    elif inf_abs == 'abs':
                        new_columns_dict[new_col_name_per_min] = make_avg_before_current_fight(
                            localized_df_abs, # use absorbed stats df to find absorbed stats
                            col_name, 
                            timeframe=timeframe, 
                            landed_attempts=landed_attempts
                            )
                # division by number of minutes cancels out, so accuracy is just landed / attempts
                new_col_name_accuracy = f'{timeframe}_{inf_abs}_{col_name}_accuracy'
                stats_to_add_to_main_df.append(new_col_name_accuracy)
                accuracy = new_columns_dict[f'{timeframe}_{inf_abs}_{col_name}_landed_per_min'] / new_columns_dict[f'{timeframe}_{inf_abs}_{col_name}_attempts_per_min'].replace(0, np.nan) # avoid division by zero
                accuracy.replace(np.nan, 0, inplace=True)  # replace NaN with 0
                new_columns_dict[new_col_name_accuracy] = accuracy
                        
    # adding striking stats
    for stat in striking_stats:
        col_name = f'{stat}'
        for inf_abs in ['inf', 'abs']:
            for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
                for landed_attempts in ['landed', 'attempts']:
                    new_col_name = f'{timeframe}_{inf_abs}_{col_name}_{landed_attempts}_per_min'
                    stats_to_add_to_main_df.append(new_col_name)
                    if inf_abs == 'inf':
                        new_columns_dict[new_col_name] = make_avg_before_current_fight(
                            localized_df_inf, # use inflicted stats df to find inflicted stats
                            col_name, 
                            timeframe=timeframe, 
                            landed_attempts=landed_attempts
                            )
                    elif inf_abs == 'abs':
                        new_columns_dict[new_col_name] = make_avg_before_current_fight(
                            localized_df_abs, # use absorbed stats df to find absorbed stats
                            col_name, 
                            timeframe=timeframe, 
                            landed_attempts=landed_attempts
                            )
                # division by number of minutes cancels out, so accuracy is just landed / attempts
                new_col_name_accuracy = f'{timeframe}_{inf_abs}_{col_name}_accuracy'
                stats_to_add_to_main_df.append(new_col_name_accuracy)
                accuracy = new_columns_dict[f'{timeframe}_{inf_abs}_{col_name}_landed_per_min'] / new_columns_dict[f'{timeframe}_{inf_abs}_{col_name}_attempts_per_min'].replace(0, np.nan) # avoid division by zero
                accuracy.replace(np.nan, 0, inplace=True)  # replace NaN with 0
                new_columns_dict[new_col_name_accuracy] = accuracy
                
    # STATS COMPUTED FROM THE new_columns_dict
    # add an offensive striking score
    standup_striking_score_stats = [u'sig_strikes', u'total_strikes', u'head_strikes', u'body_strikes', u'leg_strikes', u'distance_strikes', u'clinch_strikes']
    # say a strike landed is 3 times more valuable than a strike attempted
    # say a knockdown is worth 10 times a landed strike
    # add up all inf attempts
    inf_abs = 'inf'  # we only care about inflicted stats for the striking score
    for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
        new_col_name = f'{timeframe}_offensive_standing_striking_score'
        stats_to_add_to_main_df.append(new_col_name)
        offensive_standup_striking_score = new_columns_dict[f'{timeframe}_{inf_abs}_knockdowns_per_min'] * 10 # 10 times more valuable than a attempted strike
        for stat in standup_striking_score_stats:
            offensive_standup_striking_score += new_columns_dict[f'{timeframe}_{inf_abs}_{stat}_landed_per_min'] * 3  # 3 times more valuable than a strike attempted
            offensive_standup_striking_score += new_columns_dict[f'{timeframe}_{inf_abs}_{stat}_attempts_per_min']  # add attempts
        # knockout win is worth 3 times more than a knockdown per minute
        offensive_standup_striking_score += new_columns_dict[f'{timeframe}_wins_ko'] * 3  # 3 times more valuable than a knockdown per minute
        new_columns_dict[new_col_name] = offensive_standup_striking_score
        
    # add an defensive striking loss (smaller is better)
    inf_abs = 'abs'  # we only care about absorbed stats for the striking loss
    for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
        new_col_name = f'{timeframe}_defensive_standing_striking_loss'
        stats_to_add_to_main_df.append(new_col_name)
        defensive_standup_striking_loss = new_columns_dict[f'{timeframe}_{inf_abs}_knockdowns_per_min'] * 10
        for stat in standup_striking_score_stats:
            defensive_standup_striking_loss += new_columns_dict[f'{timeframe}_{inf_abs}_{stat}_landed_per_min'] * 3 # 3 times more costly than a strike attempted
            defensive_standup_striking_loss += new_columns_dict[f'{timeframe}_{inf_abs}_{stat}_attempts_per_min']  # add attempts
        # knockout loss is worth 3 times more than a knockdown per minute
        defensive_standup_striking_loss += new_columns_dict[f'{timeframe}_losses_ko'] * 3  # 3 times more costly than a knockdown per minute
        # add the score to the new columns dict
        new_columns_dict[new_col_name] = defensive_standup_striking_loss
        
    # add an offensive grappling score
    offensive_grappling_score_stats = [u'takedowns', u'sub_attempts', u'reversals', u'control']
    # takedowns and sub attempts and reversals are equally weighted. 30 seconds of control is worth 1 takedown or sub attempt
    inf_abs = 'inf'  # we only care about inflicted stats for the grappling score
    for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
        new_col_name = f'{timeframe}_offensive_grappling_score'
        stats_to_add_to_main_df.append(new_col_name)
        offensive_grappling_score = new_columns_dict[f'{timeframe}_{inf_abs}_takedowns_landed_per_min'] 
        offensive_grappling_score += new_columns_dict[f'{timeframe}_{inf_abs}_takedowns_attempts_per_min'] / 5 # 5 takedown attempts are worth 1 takedown landed
        offensive_grappling_score += new_columns_dict[f'{timeframe}_{inf_abs}_sub_attempts_per_min'] 
        offensive_grappling_score += new_columns_dict[f'{timeframe}_{inf_abs}_reversals_per_min']
        offensive_grappling_score += new_columns_dict[f'{timeframe}_{inf_abs}_control_per_min'] / 30  # 30 seconds of control is worth 1 takedown or sub attempt
        # ground strikes are included. Say 5 ground strikes are worth 1 takedown or sub attempt
        offensive_grappling_score += new_columns_dict[f'{timeframe}_{inf_abs}_ground_strikes_landed_per_min'] / 5  # 5 ground strikes are worth 1 takedown or sub attempt
        offensive_grappling_score += new_columns_dict[f'{timeframe}_{inf_abs}_ground_strikes_attempts_per_min'] / 15  # 5 ground strikes are worth 1 takedown or sub attempt

        # add the score to the new columns dict
        # submission win is worth 3 times more than a takedown per minute
        offensive_grappling_score += new_columns_dict[f'{timeframe}_wins_sub'] * 3  # 3 times more valuable than a takedown per minute
        # add the score to the new columns dict
        new_columns_dict[new_col_name] = offensive_grappling_score
        
    # add an defensive grappling loss (smaller is better)
    defensive_grappling_loss_stats = [u'takedowns', u'sub_attempts', u'reversals', u'control']
    # takedowns and sub attempts and reversals are equally weighted. 30 seconds of control is worth 1 takedown or sub attempt
    inf_abs = 'abs'  # we only care about absorbed stats for the grappling
    for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
        new_col_name = f'{timeframe}_defensive_grappling_loss'
        stats_to_add_to_main_df.append(new_col_name)
        defensive_grappling_loss = new_columns_dict[f'{timeframe}_{inf_abs}_takedowns_landed_per_min'] 
        defensive_grappling_loss = new_columns_dict[f'{timeframe}_{inf_abs}_takedowns_landed_per_min'] / 5 # 5 takedown attempts are worth 1 takedown landed
        defensive_grappling_loss += new_columns_dict[f'{timeframe}_{inf_abs}_sub_attempts_per_min'] 
        defensive_grappling_loss += new_columns_dict[f'{timeframe}_{inf_abs}_reversals_per_min']
        defensive_grappling_loss += new_columns_dict[f'{timeframe}_{inf_abs}_control_per_min'] / 30  # 30 seconds of control is worth 1 takedown or sub attempt
        # ground strikes are included. Say 5 ground strikes are worth 1 takedown or sub attempt
        defensive_grappling_loss += new_columns_dict[f'{timeframe}_{inf_abs}_ground_strikes_landed_per_min'] / 5  # 5 ground strikes are worth 1 takedown or sub attempt
        defensive_grappling_loss += new_columns_dict[f'{timeframe}_{inf_abs}_ground_strikes_attempts_per_min'] / 15  # 5 ground strikes are worth 1 takedown or sub attempt
        # submission loss is worth 3 times more than a takedown per minute
        defensive_grappling_loss += new_columns_dict[f'{timeframe}_losses_sub'] * 3  # 3 times more costly than a takedown per minute
        # add the score to the new columns dict
        new_columns_dict[new_col_name] = defensive_grappling_loss
    
    # make overall fighter scores
    for timeframe in ['all', 'l1y', 'l3y', 'l5y']:
        new_col_name = f'{timeframe}_overall_fighter_score'
        stats_to_add_to_main_df.append(new_col_name)
        overall_fighter_score = new_columns_dict[f'{timeframe}_offensive_standing_striking_score'] - new_columns_dict[f'{timeframe}_defensive_standing_striking_loss']
        overall_fighter_score += new_columns_dict[f'{timeframe}_offensive_grappling_score'] - new_columns_dict[f'{timeframe}_defensive_grappling_loss']
        # add a bonus for winning fights
        overall_fighter_score += new_columns_dict[f'{timeframe}_wins'] * 2  # each win is worth 2 points
        overall_fighter_score -= new_columns_dict[f'{timeframe}_losses'] * 2  # each loss is worth -2 points
        new_columns_dict[new_col_name] = overall_fighter_score
    # TODO make fight_math stats for all timeframes
                        
    # add all new columns to localized df at once to avoid highly fragmented df warning
    new_columns_df = pd.DataFrame(new_columns_dict, index=localized_df.index)
    localized_df = pd.concat([localized_df, new_columns_df], axis=1)
    # add all new stats to main df at once to avoid highly fragmented df warning
    ufc_fights_reported_derived_doubled.loc[fighter_mask, stats_to_add_to_main_df] = localized_df
        
# save the results to a csv file 
ufc_fights_reported_derived_doubled_path = f'{git_root}/src/content/data/processed/ufc_fights_reported_derived_doubled.csv'
ufc_fights_reported_derived_doubled.to_csv(ufc_fights_reported_derived_doubled_path, index=True)
print(f'Saved predictive fight history with stats to {ufc_fights_reported_derived_doubled_path}, shape {ufc_fights_reported_derived_doubled.shape}')