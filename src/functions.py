from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
import urllib3
import requests
from Levenshtein import distance as lev
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import csv
import json
import git
import os

#from https://stackoverflow.com/questions/22081209/find-the-root-of-the-git-repository-where-the-file-lives 
def get_root():
    git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


# updated scraped fight data (after running fight_hist_updated function from UFC_data_scraping file)
fight_hist = pd.read_csv(
    'models/buildingMLModel/data/processed/fight_hist.csv', sep=',', low_memory=False)
# all stats fight history file which is one update behind fight_hist
ufcfightscrap = pd.read_csv(
    'models/buildingMLModel/data/processed/ufc_fights_crap.csv', sep=',', low_memory=False)
# updated scraped fighter data (after running fight_hist_updated function from UFC_data_scraping file)
ufcfighters = pd.read_csv(
    'models/buildingMLModel/data/processed/fighter_stats.csv', sep=',', low_memory=False)

# first call pip install python-Levenshtein


def same_name(str1, str2, verbose = False):
    try:
        str1 = str1.lower().replace("st.", 'saint').replace(
            " st ", ' saint ').replace('.', '').replace("-", ' ')
        str2 = str2.lower().replace("st.", 'saint').replace(
            " st ", ' saint ').replace('.', '').replace("-", ' ')
        str1_list = str1.split()
        str2_list = str2.split()
        str1_set = set(str1_list)
        str2_set = set(str2_list)
        if str1 == str2:
            return True
        elif str1_set == str2_set:
            if verbose:
                print(str1+' ... (same name as) ... '+str2+' ... (different ordering)')
            return True
        elif lev(str1, str2) < 3:
            if verbose:
                print(str1+' ... (same name as) ... '+str2 + ' ... (small Levenshtein distance apart)')
            return True
        else:
            return False
    except:
        return False

# checks if a fighter is in the ufc


def in_ufc(fighter):
    for name in ufcfighters['name']:
        if same_name(fighter, name):
            return True
    return False

# this cell contains all functions defined for building columns in ufcfightscrap
# converts from '%B %d, %Y' (i.e. August 22, 2020) to date (i.e. 2020-08-22)


def convert_to_datetime(day1):
    return datetime.strptime(day1, '%B %d, %Y').date()


convert_to_datetime_vect = np.vectorize(convert_to_datetime)

# converts 'August 15,2019' to 'Aug 15,2019'


def convert_date_to_abbrev(day):
    return datetime.strptime(day, '%B %d, %Y').strftime('%b %d, %Y')

# converts 'Aug 15,2019' to 'August 15,2019'


def convert_date_to_unabbrev(day):
    return datetime.strptime(day, '%b %d, %Y').strftime('%B %d, %Y')

# this age function needs to be rewritten.


def age(birthDate, day=date.today(), form1='%B %d, %Y', form2='%B %d, %Y'):
    if birthDate == '--':
        aa = 0
    elif type(birthDate) == str and not type(day) == str:
        birthDate = convert_date_to_unabbrev(birthDate)
        bd = datetime.strptime(birthDate, form1)
        today = day
        aa = today.year - bd.year - \
            ((today.month, today.day) < (bd.month, bd.day))

    elif type(birthDate) == str and type(day) == str:
        bd = datetime.strptime(birthDate, form1)
        today = datetime.strptime(day, form2)
        aa = today.year - bd.year - \
            ((today.month, today.day) < (bd.month, bd.day))

    elif not type(birthDate) == str and type(day) == str:
        bd = birthDate
        today = datetime.strptime(day, form2)
        aa = today.year - bd.year - \
            ((today.month, today.day) < (bd.month, bd.day))

    else:
        bd = birthDate
        today = day
        aa = today.year - bd.year - \
            ((today.month, today.day) < (bd.month, bd.day))
    return aa


age_vect = np.vectorize(age)


def fighter_age(fighter, day=date.today(), form1='%B %d, %Y', form2='%B %d, %Y'):
    a = 0
    for i in range(len(ufcfighterscrap['name'])):
        # if ufcfighterscrap['name'][i]==fighter: #replaced for better accuracy
        if same_name(ufcfighterscrap['name'][i], fighter):
            try:
                dob = datetime.strptime(
                    ufcfighterscrap['dob'][i], '%b %d, %Y').strftime('%B %d, %Y')
                a = age(dob, day, form1, form2)
                break
            except:
                a = 0 
    return a


fighter_age_vect = np.vectorize(fighter_age)

# zero function (needed for reversals)


def zero(x):
    return 0


zero_vect = np.vectorize(zero)


def wins_before(guy, day1=date.today(), something='%B %d, %Y'):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if time_diff(ufcfightscrap['date'][i], day1) > 0 and ufcfightscrap['result'][i] == 'W':
            summ += 1
        else:
            summ += 0
    return summ


def losses_before(guy, day1=date.today(), something='%B %d, %Y'):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if time_diff(ufcfightscrap['date'][i], day1) > 0 and ufcfightscrap['result'][i] == 'L':
            summ += 1
        else:
            summ += 0
    return summ


def record_before(guy, something, day1=date.today()):
    return 0


def ties_before(guy, something, day1=date.today()):
    return 0

# functions for height and reach


def fighter_height(fighter):
    a = 0
    for i in range(len(ufcfighterscrap['name'])):
        # if ufcfighterscrap['name'][i]==fighter:
        if same_name(ufcfighterscrap['name'][i], fighter):
            a = ufcfighterscrap['height'][i]
            break
    if a == '--' or a == 0:
        b = 'unknown'
    elif a[4] == '"':
        b = int(a[0])*30.48+int(a[3])*2.54
    else:
        b = int(a[0])*30.48+int(a[3]+a[4])*2.54
    return b


def fighter_reach(fighter):
    a = 0
    for i in range(len(ufcfighterscrap['name'])):
        # if ufcfighterscrap['name'][i]==fighter:
        if same_name(ufcfighterscrap['name'][i], fighter):
            a = ufcfighterscrap['reach'][i]
            break
    if a == '--' or a == 0:
        b = 'unknown'
    else:
        b = int(a[0]+a[1])*2.54
    return b


wins_before_vect = np.vectorize(wins_before)
losses_before_vect = np.vectorize(losses_before)
fighter_height_vect = np.vectorize(fighter_height)
fighter_reach_vect = np.vectorize(fighter_reach)

# day1 should be input in the form '%B %d, %Y' i.e. 'August 20, 1962'
# conversions can be made via day=datetime.strptime(ufcfightsML_known_df['date'][i], '%B %d, %Y').strftime('%b %d, %Y')


def time_diff(day1, day2=date.today()):
    if day2 == date.today():
        answer = (day2-datetime.strptime(day1, '%B %d, %Y')).days
    else:
        answer = (datetime.strptime(day2, '%B %d, %Y') -
                  datetime.strptime(day1, '%B %d, %Y')).days
    return answer


# we now vectorize this to use in pandas/numpy
time_diff_vect = np.vectorize(time_diff)

# can make a single function to do all of these... actually maybe the count function would even work as is


def L5Y_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufcfightscrap['date'][i], day1) < 1825 and ufcfightscrap['result'][i] == 'W':
            summ += 1
        else:
            summ += 0
        if time_diff(ufcfightscrap['date'][i], day1) > 1825:
            break
    return summ


def L5Y_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufcfightscrap['date'][i], day1) < 1825 and ufcfightscrap['result'][i] == 'L':
            summ += 1
        else:
            summ += 0
        if time_diff(ufcfightscrap['date'][i], day1) > 1825:
            break
    return summ


def L2Y_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufcfightscrap['date'][i], day1) < 730 and ufcfightscrap['result'][i] == 'W':
            summ += 1
        else:
            summ += 0
        if time_diff(ufcfightscrap['date'][i], day1) > 730:
            break
    return summ


def L2Y_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufcfightscrap['date'][i], day1) < 730 and ufcfightscrap['result'][i] == 'L':
            summ += 1
        else:
            summ += 0
        if time_diff(ufcfightscrap['date'][i], day1) > 730:
            break
    return summ


def ko_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if ufcfightscrap['result'][i] == 'W' and ufcfightscrap['method'][i] == 'KO/TKO':
            summ += 1
        else:
            summ += 0
    return summ


def ko_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if ufcfightscrap['result'][i] == 'L' and ufcfightscrap['method'][i] == 'KO/TKO':
            summ += 1
        else:
            summ += 0
    return summ


def L5Y_ko_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufcfightscrap['date'][i], day1) < 1825 and ufcfightscrap['result'][i] == 'W' and ufcfightscrap['method'][i] == 'KO/TKO':
            summ += 1
        else:
            summ += 0
        if time_diff(ufcfightscrap['date'][i], day1) > 1825:
            break
    return summ


def L5Y_ko_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufcfightscrap['date'][i], day1) < 1825 and ufcfightscrap['result'][i] == 'L' and ufcfightscrap['method'][i] == 'KO/TKO':
            summ += 1
        else:
            summ += 0
        if time_diff(ufcfightscrap['date'][i], day1) > 1825:
            break
    return summ


def L2Y_ko_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufcfightscrap['date'][i], day1) < 730 and ufcfightscrap['result'][i] == 'W' and ufcfightscrap['method'][i] == 'KO/TKO':
            summ += 1
        else:
            summ += 0
        if time_diff(ufcfightscrap['date'][i], day1) > 730:
            break
    return summ


def L2Y_ko_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufcfightscrap['date'][i], day1) < 730 and ufcfightscrap['result'][i] == 'L' and ufcfightscrap['method'][i] == 'KO/TKO':
            summ += 1
        else:
            summ += 0
        if time_diff(ufcfightscrap['date'][i], day1) > 730:
            break
    return summ


def sub_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if ufcfightscrap['result'][i] == 'W' and ufcfightscrap['method'][i] == 'SUB':
            summ += 1
        else:
            summ += 0
    return summ


def sub_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if ufcfightscrap['result'][i] == 'L' and ufcfightscrap['method'][i] == 'SUB':
            summ += 1
        else:
            summ += 0
    return summ


def L5Y_sub_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufcfightscrap['date'][i], day1) < 1825 and ufcfightscrap['result'][i] == 'W' and ufcfightscrap['method'][i] == 'SUB':
            summ += 1
        else:
            summ += 0
        if time_diff(ufcfightscrap['date'][i], day1) > 1825:
            break
    return summ


def L5Y_sub_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufcfightscrap['date'][i], day1) < 1825 and ufcfightscrap['result'][i] == 'L' and ufcfightscrap['method'][i] == 'SUB':
            summ += 1
        else:
            summ += 0
        if time_diff(ufcfightscrap['date'][i], day1) > 1825:
            break
    return summ


def L2Y_sub_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufcfightscrap['date'][i], day1) < 730 and ufcfightscrap['result'][i] == 'W' and ufcfightscrap['method'][i] == 'SUB':
            summ += 1
        else:
            summ += 0
        if time_diff(ufcfightscrap['date'][i], day1) > 730:
            break
    return summ


def L2Y_sub_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if same_name(
        ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufcfightscrap['date'][i], day1) < 730 and ufcfightscrap['result'][i] == 'L' and ufcfightscrap['method'][i] == 'SUB':
            summ += 1
        else:
            summ += 0
        if time_diff(ufcfightscrap['date'][i], day1) > 730:
            break
    return summ


# vectorize all of these functions
L5Y_wins_vect = np.vectorize(L5Y_wins)
L5Y_losses_vect = np.vectorize(L5Y_losses)
L2Y_wins_vect = np.vectorize(L2Y_wins)
L2Y_losses_vect = np.vectorize(L2Y_losses)
ko_wins_vect = np.vectorize(ko_wins)
ko_losses_vect = np.vectorize(ko_losses)
L5Y_ko_wins_vect = np.vectorize(L5Y_ko_wins)
L5Y_ko_losses_vect = np.vectorize(L5Y_ko_losses)
L2Y_ko_wins_vect = np.vectorize(L2Y_ko_wins)
L2Y_ko_losses_vect = np.vectorize(L2Y_ko_losses)
sub_wins_vect = np.vectorize(sub_wins)
sub_losses_vect = np.vectorize(sub_losses)
L5Y_sub_wins_vect = np.vectorize(L5Y_sub_wins)
L5Y_sub_losses_vect = np.vectorize(L5Y_sub_losses)
L2Y_sub_wins_vect = np.vectorize(L2Y_sub_wins)
L2Y_sub_losses_vect = np.vectorize(L2Y_sub_losses)

# for columns like fighter_rec which contains the information for the opponent as well, we use the following
def opponent_column(stat):
    col = dict()
    for i in range(len(ufcfightscrap['fighter'])):
        if i % 2 == 0:
            col[i] = ufcfightscrap[stat][i+1]
        else:
            col[i] = ufcfightscrap[stat][i-1]
    statdict = {'stat': col}
    return pd.DataFrame(statdict, columns=['stat'])

# enter date unabbreviated 'July 4, 2019'
# here the average gives avg per fight. Later in avg_count we change to average per time spent in octagon


def count(stat, guy, inf_abs, total_L5Y_L2Y_avg, day1=date.today().strftime('%B %d, %Y')):
    summ = 0
    if total_L5Y_L2Y_avg == 'total' or total_L5Y_L2Y_avg == 'avg':
        good_indices_1 = [i for i in ufcfightscrap.index.values if time_diff(
            ufcfightscrap['date'][i], day1) > 0]
    elif total_L5Y_L2Y_avg == 'L2Y':
        good_indices_1 = [i for i in ufcfightscrap.index.values if 0 < time_diff(
            ufcfightscrap['date'][i], day1) < 730]
    else:
        good_indices_1 = [i for i in ufcfightscrap.index.values if 0 < time_diff(
            ufcfightscrap['date'][i], day1) < 1825]
    if inf_abs == 'inf':
        good_indices_2 = [i for i in ufcfightscrap.index.values if same_name(
            ufcfightscrap['fighter'][i], guy)]
    else:
        good_indices_2 = [i for i in ufcfightscrap.index.values if same_name(
            ufcfightscrap['opponent'][i], guy)]
    good_indices = [i for i in good_indices_1 if i in good_indices_2]
    if total_L5Y_L2Y_avg != 'avg':
        for i in good_indices:
            summ += ufcfightscrap[stat][i]
    else:
        for i in good_indices:
            summ += ufcfightscrap[stat][i]
        day1 = convert_date_to_abbrev(day1)
        number_fights = wins_before(guy, day1)+losses_before(guy, day1)
        summ = summ/float(number_fights)
    return summ

# note a better average is per time not per fight. We will do each stat as an average per one minutes

# enter time in the form 'August 24, 2018'


def time_in_octagon(guy, day1=date.today().strftime('%B %d, %Y')):
    summ = 0
    good_indices = [i for i in ufcfightscrap.index.values if time_diff(
        ufcfightscrap['date'][i], day1) > 0 and same_name(ufcfightscrap['fighter'][i], guy)]
    for i in good_indices:
        if ufcfightscrap['time'][i][2] == ':':
            summ = int(ufcfightscrap['time'][i][0:2]) + \
                int(ufcfightscrap['time'][i][3:])/60.0
        else:
            summ += 5*(ufcfightscrap['round'][i]-1)+int(ufcfightscrap['time']
                                                        [i][0])+int(ufcfightscrap['time'][i][2:])/60.0
    return summ

# enter date unabbreviated 'July 4, 2019'
# gives takedowns per minute


def avg_count(stat, guy, inf_abs, day1=date.today().strftime('%B %d, %Y')):
    summ = 0
    if inf_abs == 'inf':
        good_indices = [i for i in ufcfightscrap.index.values if same_name(
            ufcfightscrap['fighter'][i], guy) and time_diff(ufcfightscrap['date'][i], day1) > 0]
    else:
        good_indices = [i for i in ufcfightscrap.index.values if same_name(
            ufcfightscrap['opponent'][i], guy) and time_diff(ufcfightscrap['date'][i], day1) > 0]
    for i in good_indices:
        summ += ufcfightscrap[stat][i]
    t = time_in_octagon(guy, day1)
    if t == 0:
        summ = 0
    else:
        summ = summ/t
    return summ


# vectorize these functions
count_vect = np.vectorize(count)
avg_count_vect = np.vectorize(avg_count)

ufc_fights = pd.read_csv(
    'models/buildingMLModel/data/processed/ufc_fights.csv', low_memory=False)
ufcfighterscrap = pd.read_csv(
    'models/buildingMLModel/data/processed/fighter_stats.csv', sep=',', low_memory=False)


def stance(fighter):
    find_fighter = [i for i in range(len(ufcfighterscrap['name'])) if same_name(
        ufcfighterscrap['name'][i], fighter)]
    b = next(iter(find_fighter), 'unknown')
    if b == 'unknown':
        a = 'unknown'
    else:
        a = ufcfighterscrap['stance'][b]
    if a == 'Orthodox':
        return 0
    elif a == 'Switch' or a == 'Southpaw' or a == 'Open Stance' or a == 'Sideways':
        return 1
    else:
        return 5


stance_vect = np.vectorize(stance)


def clean_method(a):
    if (a == 'KO/TKO'):
        return 'KO/TKO'
    elif (a == 'SUB'):
        return 'SUB'
    elif ((a == 'U-DEC') or (a == 'M-DEC') or (a == 'S-DEC')):
        return 'DEC'
    else:
        return 'bullshit'


clean_method_vect = np.vectorize(clean_method)


def clean_method_for_winner_prediction(a):
    if (a == 'KO/TKO'):
        return 'KO/TKO'
    elif (a == 'SUB'):
        return 'SUB'
    elif ((a == 'U-DEC') or (a == 'M-DEC')):
        return 'DEC'
    # counting S-DEC as bullshit!
    else:
        return 'bullshit'


clean_method_for_winner_vect = np.vectorize(clean_method_for_winner_prediction)

def get_odds_two_rows_per_fight():
    url = 'https://www.bestfightodds.com'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    mydivs = soup.find_all("tr", {"class": ""})
    rows = [tr for tr in mydivs if 'bestbet' in str(tr)]
    names = []
    oddsDicts = []
    books = ['DraftKings', 'BetMGM', 'Caesars', 'BetRivers', 'FanDuel',
             'PointsBet', 'Unibet', 'Bet365', 'BetWay', '5D', 'Ref']
    for row in rows:
        # gets name of fighter in row
        name = row.find_all("span", {"class": "t-b-fcc"})[0].text
        oddsList = [name]
        i = 0
        for stat in row.select('td'):
            i += 1
            if i > 11:
                break
            try:
                odds = stat.select('span')[0].text
                oddsList.append(odds)
            except:
                oddsList.append('')
        names.append(name)
        oddsDicts.append(dict(zip(['name']+books, oddsList)))
    oddsDict = dict(zip(names, oddsDicts))
    names = list(oddsDict.keys())
    row0 = oddsDict[list(oddsDict.keys())[0]]
    odds_df = pd.DataFrame(row0, index=[0])
    for i in range(1, len(names)):
        row = oddsDict[names[i]]
        odds_df = pd.concat([odds_df, pd.DataFrame(row, index=[i])], axis=0)
    return odds_df


# problem: the fights in the "future events" category do not get lined up properly
def get_odds():
    url = 'https://www.bestfightodds.com'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    mydivs = soup.find_all("tr", {"class": ""})
    rows = [tr for tr in mydivs if 'bestbet' in str(tr)]
    names = []
    oddsDicts = []
    books = ['DraftKings', 'BetMGM', 'Caesars', 'BetRivers', 'FanDuel',
             'PointsBet', 'Unibet', 'Bet365', 'BetWay', '5D', 'Ref']
    for row in rows:
        # gets name of fighter in row
        name = row.find_all("span", {"class": "t-b-fcc"})[0].text
        oddsList = [name]
        i = 0
        for stat in row.select('td'):
            i += 1
            if i > 11:
                break
            try:
                odds = stat.select('span')[0].text
                oddsList.append(odds)
            except:
                oddsList.append('')
        while name in names:
            name += '.'
        names.append(name)
        oddsDicts.append(dict(zip(['name']+books, oddsList)))
    oddsDict = dict(zip(names, oddsDicts))
    names = list(oddsDict.keys())
    row0 = oddsDict[list(oddsDict.keys())[0]]
    odds_df = pd.DataFrame(row0, index=[0])
    for i in range(1, len(names)):
        row = oddsDict[names[i]]
        odds_df = pd.concat([odds_df, pd.DataFrame(row, index=[i])], axis=0)
    # making it so each fight has just a single row instead of two rows
    # making dataframe just for even indexed columns
    odds_df_evens = odds_df[odds_df.index % 2 == 0]
    newcolumns1 = {}
    for col in list(odds_df_evens.columns):
        newcolumns1[col] = 'fighter '+col
    odds_df_evens = odds_df_evens.rename(columns=newcolumns1)
    odds_df_evens.reset_index(drop=True, inplace=True)
    # making dataframe just for odd indexed columns
    odds_df_odds = odds_df[odds_df.index % 2 == 1]
    newcolumns2 = {}
    for col in list(odds_df_odds.columns):
        newcolumns2[col] = 'opponent '+col
    odds_df_odds = odds_df_odds.rename(columns=newcolumns2)
    odds_df_odds.reset_index(drop=True, inplace=True)
    new_odds_df = pd.concat([odds_df_evens, odds_df_odds], axis=1)
    return new_odds_df

# input looks like 'July 15th'. Need to add year to it
def convert_scraped_date_to_standard_date(input_date):
    year = date.today().strftime('%B %d, %Y')[-4:]
    month = input_date.split(' ')[0]
    day = input_date.split(' ')[1]
    i = 0
    while day[i].isdigit():
        i+=1
    day = day[:i]
    return f'{month} {day}, {year}'

def get_next_fight_card():
    url = 'http://ufcstats.com/statistics/events/upcoming'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser") 
    mycards = soup.find_all("a", {"class": "b-link b-link_style_black"})
    mydates = soup.find_all("span", {"class":"b-statistics__date"})
    date = mydates[0]
    card = mycards[0] 
    card_date = date.get_text().strip()
    card_title = card.get_text().strip()
    fights_list = []
    card_link = card.attrs['href']
    page = requests.get(card_link)
    soup = BeautifulSoup(page.content, "html.parser")
    fights = soup.find_all("tr",{"class": "b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click"})
    for fight in fights:
        fighter, opponent, _, weight_class = [entry.get_text().strip() for entry in fight.find_all('p') if entry.get_text().strip()!= '']
        fights_list.append([fighter,opponent,weight_class])
    return card_date, card_title, fights_list


# thresh is the number of bookies we allow to not have odds on the books


def drop_irrelevant_fights(df, thresh):
    irr = []
    for i in df.index:
        count = 0
        row = list(df.loc[i])
        for j in row:
            if j == '':
                count += 1
        if count > 2*thresh:
            irr.append(i)
    df = df.drop(irr)
    return df


# there is a problem for collecting reversals (fix needed) seems like it now collect riding time since sept 2020
# function for getting individual fight stats
def get_fight_stats(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    fd_columns = {'fighter': [], 'knockdowns': [], 'sig_strikes': [], 'total_strikes': [], 'takedowns': [], 'sub_attempts': [], 'pass': [],
                  'reversals': []}

    # gets overall fight details
    fight_details = soup.select_one('tbody.b-fight-details__table-body')
    if fight_details == None:
        print('missing fight details for:', url)
        return None
    else:
        fd_cols = fight_details.select('td.b-fight-details__table-col')
        for i in range(len(fd_cols)):
            # skip 3 and 6: strike % and takedown %, will calculate these later
            if i == 3 or i == 6:
                pass
            else:
                col = fd_cols[i].select('p')
                for row in col:
                    data = row.text.strip()
                    if i == 0:  # add to fighter
                        fd_columns['fighter'].append(data)
                    elif i == 1:  # add to sig strikes
                        fd_columns['knockdowns'].append(data)
                    elif i == 2:  # add to total strikes
                        fd_columns['sig_strikes'].append(data)
                    elif i == 4:  # add to total strikes
                        fd_columns['total_strikes'].append(data)
                    elif i == 5:  # add to takedowns
                        fd_columns['takedowns'].append(data)
                    elif i == 7:  # add to sub attempts
                        fd_columns['sub_attempts'].append(data)
                    elif i == 8:  # add to passes
                        fd_columns['pass'].append(data)
                    elif i == 9:  # add to reversals
                        fd_columns['reversals'].append(data)
        ov_details = pd.DataFrame(fd_columns)

        # get sig strike details
        sig_strike_details = soup.find('p', class_='b-fight-details__collapse-link_tot', text=re.compile(
            'Significant Strikes')).find_next('tbody', class_='b-fight-details__table-body')
        sig_columns = {'fighter': [], 'head_strikes': [], 'body_strikes': [], 'leg_strikes': [], 'distance_strikes': [],
                       'clinch_strikes': [], 'ground_strikes': []}
        fd_cols = sig_strike_details.select('td.b-fight-details__table-col')
        for i in range(len(fd_cols)):
            # skip 1, 2 (sig strikes, sig %)
            if i == 1 or i == 2:
                pass
            else:
                col = fd_cols[i].select('p')
                for row in col:
                    data = row.text.strip()
                    if i == 0:  # add to fighter
                        sig_columns['fighter'].append(data)
                    elif i == 3:  # add to head strikes
                        sig_columns['head_strikes'].append(data)
                    elif i == 4:  # add to body strikes
                        sig_columns['body_strikes'].append(data)
                    elif i == 5:  # add to leg strikes
                        sig_columns['leg_strikes'].append(data)
                    elif i == 6:  # add to distance strikes
                        sig_columns['distance_strikes'].append(data)
                    elif i == 7:  # add to clinch strikes
                        sig_columns['clinch_strikes'].append(data)
                    elif i == 8:  # add to ground strikes
                        sig_columns['ground_strikes'].append(data)
        sig_details = pd.DataFrame(sig_columns)

        cfd = pd.merge(ov_details, sig_details,
                       on='fighter', how='left', copy=False)

        cfd['takedowns_landed'] = cfd.takedowns.str.split(
            ' of ').str[0].astype(int)
        cfd['takedowns_attempts'] = cfd.takedowns.str.split(
            ' of ').str[-1].astype(int)
        cfd['sig_strikes_landed'] = cfd.sig_strikes.str.split(
            ' of ').str[0].astype(int)
        cfd['sig_strikes_attempts'] = cfd.sig_strikes.str.split(
            ' of ').str[-1].astype(int)
        cfd['total_strikes_landed'] = cfd.total_strikes.str.split(
            ' of ').str[0].astype(int)
        cfd['total_strikes_attempts'] = cfd.total_strikes.str.split(
            ' of ').str[-1].astype(int)
        cfd['head_strikes_landed'] = cfd.head_strikes.str.split(
            ' of ').str[0].astype(int)
        cfd['head_strikes_attempts'] = cfd.head_strikes.str.split(
            ' of ').str[-1].astype(int)
        cfd['body_strikes_landed'] = cfd.body_strikes.str.split(
            ' of ').str[0].astype(int)
        cfd['body_strikes_attempts'] = cfd.body_strikes.str.split(
            ' of ').str[-1].astype(int)
        cfd['leg_strikes_landed'] = cfd.leg_strikes.str.split(
            ' of ').str[0].astype(int)
        cfd['leg_strikes_attempts'] = cfd.leg_strikes.str.split(
            ' of ').str[-1].astype(int)
        cfd['distance_strikes_landed'] = cfd.distance_strikes.str.split(
            ' of ').str[0].astype(int)
        cfd['distance_strikes_attempts'] = cfd.distance_strikes.str.split(
            ' of ').str[-1].astype(int)
        cfd['clinch_strikes_landed'] = cfd.clinch_strikes.str.split(
            ' of ').str[0].astype(int)
        cfd['clinch_strikes_attempts'] = cfd.clinch_strikes.str.split(
            ' of ').str[-1].astype(int)
        cfd['ground_strikes_landed'] = cfd.ground_strikes.str.split(
            ' of ').str[0].astype(int)
        cfd['ground_strikes_attempts'] = cfd.ground_strikes.str.split(
            ' of ').str[-1].astype(int)

        cfd = cfd.drop(['takedowns', 'sig_strikes', 'total_strikes', 'head_strikes', 'body_strikes', 'leg_strikes', 'distance_strikes',
                        'clinch_strikes', 'ground_strikes'], axis=1)
        return (cfd)

# function for getting fight stats for all fights on a card


def get_fight_card(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    fight_card = pd.DataFrame()
    date = soup.select_one(
        'li.b-list__box-list-item').text.strip().split('\n')[-1].strip()
    rows = soup.select('tr.b-fight-details__table-row')[1:]
    for row in rows:
        fight_det = {'date': [], 'fight_url': [], 'event_url': [], 'result': [], 'fighter': [], 'opponent': [], 'division': [], 'method': [],
                     'round': [], 'time': [], 'fighter_url': [], 'opponent_url': []}
        fight_det['date'] += [date, date]  # add date of fight
        fight_det['event_url'] += [url, url]  # add event url
        cols = row.select('td')
        for i in range(len(cols)):
            if i in set([2, 3, 4, 5]):  # skip sub, td, pass, strikes
                pass
            elif i == 0:  # get fight url and results
                fight_url = cols[i].select_one('a')['href']  # get fight url
                fight_det['fight_url'] += [fight_url, fight_url]

                results = cols[i].select('p')
                if len(results) == 2:  # was a draw, table shows two draws
                    fight_det['result'] += ['D', 'D']
                else:  # first fighter won, second lost
                    fight_det['result'] += ['W', 'L']

            elif i == 1:  # get fighter names and fighter urls
                fighter_1 = cols[i].select('p')[0].text.strip()
                fighter_2 = cols[i].select('p')[1].text.strip()

                fighter_1_url = cols[i].select('a')[0]['href']
                fighter_2_url = cols[i].select('a')[1]['href']

                fight_det['fighter'] += [fighter_1, fighter_2]
                fight_det['opponent'] += [fighter_2, fighter_1]

                fight_det['fighter_url'] += [fighter_1_url, fighter_2_url]
                fight_det['opponent_url'] += [fighter_2_url, fighter_1_url]
            elif i == 6:  # get division
                division = cols[i].select_one('p').text.strip()
                fight_det['division'] += [division, division]
            elif i == 7:  # get method
                method = cols[i].select_one('p').text.strip()
                fight_det['method'] += [method, method]
            elif i == 8:  # get round
                rd = cols[i].select_one('p').text.strip()
                fight_det['round'] += [rd, rd]
            elif i == 9:  # get time
                time = cols[i].select_one('p').text.strip()
                fight_det['time'] += [time, time]

        fight_det = pd.DataFrame(fight_det)
        # get striking details
        str_det = get_fight_stats(fight_url)
        if str_det is None:
            pass
        else:
            # join to fight details
            fight_det = pd.merge(fight_det, str_det,
                                 on='fighter', how='left', copy=False)
            # add fight details to fight card
            fight_card = pd.concat([fight_card, fight_det], axis=0)
    fight_card = fight_card.reset_index(drop=True)
    return fight_card

# function that gets stats on all fights on all cards


def get_all_fight_stats():
    url = 'http://ufcstats.com/statistics/events/completed?page=all'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    events_table = soup.select_one('tbody')
    events = [event['href'] for event in events_table.select(
        'a')[1:]]  # omit first event, future event

    fight_stats = pd.DataFrame()
    for event in events:
        print(event)
        stats = get_fight_card(event)
        fight_stats = pd.concat([fight_stats, stats], axis=0)

    fight_stats = fight_stats.reset_index(drop=True)
    return fight_stats

# gets individual fighter attributes


def get_fighter_details(fighter_urls):
    fighter_details = {'name': [], 'height': [],
                       'reach': [], 'stance': [], 'dob': [], 'url': []}

    for f_url in fighter_urls:
        print(f_url)
        page = requests.get(f_url)
        soup = BeautifulSoup(page.content, "html.parser")

        fighter_name = soup.find(
            'span', class_='b-content__title-highlight').text.strip()
        fighter_details['name'].append(fighter_name)

        fighter_details['url'].append(f_url)

        fighter_attr = soup.find(
            'div', class_='b-list__info-box b-list__info-box_style_small-width js-guide').select('li')
        for i in range(len(fighter_attr)):
            attr = fighter_attr[i].text.split(':')[-1].strip()
            if i == 0:
                fighter_details['height'].append(attr)
            elif i == 1:
                pass  # weight is always just whatever weightclass they were fighting at
            elif i == 2:
                fighter_details['reach'].append(attr)
            elif i == 3:
                fighter_details['stance'].append(attr)
            else:
                fighter_details['dob'].append(attr)
    return pd.DataFrame(fighter_details)

# updates fight stats with newer fights


def update_fight_stats(old_stats):  # takes dataframe of fight stats as input
    url = 'http://ufcstats.com/statistics/events/completed?page=all'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    events_table = soup.select_one('tbody')
    new_stats = pd.DataFrame()
    try:
        events = [event['href'] for event in events_table.select(
            'a')[1:]]  # omit first event, future event
        saved_events = set(old_stats.event_url.unique())
        for event in events:
            if event in saved_events:
                break
            else:
                print(event)
                stats = get_fight_card(event)
                new_stats = pd.concat([new_stats, stats], axis=0)
    except:
        print('that didnt work... if there is an event going on right now, this will not run correctly')
    updated_stats = pd.concat([new_stats, old_stats], axis=0)
    updated_stats = updated_stats.reset_index(drop=True)
    return (updated_stats)

# updates fighter attributes with new fighters not yet saved yet


def update_fighter_details(fighter_urls, saved_fighters):
    fighter_details = {'name': [], 'height': [],
                       'reach': [], 'stance': [], 'dob': [], 'url': []}
    fighter_urls = set(fighter_urls)
    saved_fighter_urls = set(saved_fighters.url.unique())

    for f_url in fighter_urls:
        if f_url in saved_fighter_urls:
            pass
        else:
            print('adding new fighter:', f_url)
            page = requests.get(f_url)
            soup = BeautifulSoup(page.content, "html.parser")

            fighter_name = soup.find(
                'span', class_='b-content__title-highlight').text.strip()
            fighter_details['name'].append(fighter_name)

            fighter_details['url'].append(f_url)

            fighter_attr = soup.find(
                'div', class_='b-list__info-box b-list__info-box_style_small-width js-guide').select('li')
            for i in range(len(fighter_attr)):
                attr = fighter_attr[i].text.split(':')[-1].strip()
                if i == 0:
                    fighter_details['height'].append(attr)
                elif i == 1:
                    pass  # weight is always just whatever weightclass they were fighting at
                elif i == 2:
                    fighter_details['reach'].append(attr)
                elif i == 3:
                    fighter_details['stance'].append(attr)
                else:
                    fighter_details['dob'].append(attr)
    new_fighters = pd.DataFrame(fighter_details)
    updated_fighters = pd.concat([new_fighters, saved_fighters])
    updated_fighters = updated_fighters.reset_index(drop=True)
    return updated_fighters


ufc_fights = pd.read_csv(
    'models/buildingMLModel/data/processed/ufc_fights.csv', low_memory=False)
ufc_fights_graph = pd.read_csv(
    'models/buildingMLModel/data/processed/ufc_fights_crap.csv', low_memory=False)
odd_indices = range(1, len(ufc_fights_graph.index), 2)
ufc_fights_graph = ufc_fights_graph.drop(odd_indices)
ufc_fights_graph = ufc_fights_graph[[
    'fighter', 'opponent', 'method', 'date', 'division']]
ufc_fights_graph = ufc_fights_graph.reset_index(drop=True)
ufc_wins_list = []

for i in ufc_fights_graph.index:
    temp_list = []
    temp_list.append(ufc_fights_graph['fighter'][i])
    temp_list.append(ufc_fights_graph['opponent'][i])
    temp_list.append(ufc_fights_graph['date'][i])
    temp_list.append(ufc_fights_graph['division'][i])
    ufc_wins_list.append(temp_list)


def fight_math(fighter, opponent, date, years):
    fighter_advantage = 0
    ufc_wins_list_l5y = [fight for fight in ufc_wins_list if 0 < time_diff(
        fight[2], date) < years*365]
    fighter_wins = [fight[1]
                    for fight in ufc_wins_list_l5y if fight[0] == fighter]
    fighter_wins.append(fighter)
    fighter_wins_wins = [fight[1]
                         for fight in ufc_wins_list_l5y if fight[0] in fighter_wins]
    relevant_wins = list(set(fighter_wins+fighter_wins_wins))
    fight_math_wins = [fight for fight in ufc_wins_list_l5y if (
        fight[0] in relevant_wins and fight[1] == opponent)]
    fighter_advantage += len(fight_math_wins)
    return fighter_advantage


def fight_math_diff(fighter, opponent, date, years):
    return fight_math(fighter, opponent, date, years)-fight_math(opponent, fighter, date, years)


fight_math_diff_vect = np.vectorize(fight_math_diff)

# need to define variable "years" before calling this function
# perhaps a better score would weight more recent fights more strongly (weight drops by 1/3 every year?)


def fighter_score(fighter, date, years):
    fighter_score = 0
    ufc_wins_list_l5y = [fight for fight in ufc_wins_list if 0 < time_diff(
        fight[2], date) < years*365]
    # calculating contribution from wins
    fighter_wins = [fight[1]
                    for fight in ufc_wins_list_l5y if fight[0] == fighter]
    fighter_wins_wins = [fight[1]
                         for fight in ufc_wins_list_l5y if fight[0] in fighter_wins]
    relevant_wins = list(set(fighter_wins+fighter_wins_wins))
    # calculating contribution from losses
    fighter_losses = [fight[0]
                      for fight in ufc_wins_list_l5y if fight[1] == fighter]
    fighter_losses_losses = [fight[0]
                             for fight in ufc_wins_list_l5y if fight[1] in fighter_losses]
    relevant_losses = list(set(fighter_losses+fighter_losses_losses))
    # print(fighter+' wins '+str(relevant_wins))
    # print(fighter+ ' losses '+str(relevant_losses))
    return len(relevant_wins)-len(relevant_losses)


def fighter_score_diff(fighter, opponent, date, years):
    return fighter_score(fighter, date, years)-fighter_score(opponent, date, years)


fighter_score_diff_vect = np.vectorize(fighter_score_diff)


def fighter_age_diff(fighter, opponent):
    age1 = fighter_age(fighter)
    age2 = fighter_age(opponent)
    try:
        return age1-age2
    except:
        return 'unknown'


fighter_age_diff_vect = np.vectorize(fighter_age_diff)

# Function to convert a CSV to JSON
# Takes the file paths as arguments


def make_json(csvFilePath, jsonFilePath, column):

    # create a dictionary
    data = {}

    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        # Convert each row into a dictionary
        # and add it to data
        for rows in csvReader:

            # primary key given by column variable
            key = rows[column]
            data[key] = rows

    # Open a json writer, and use the json.dumps()
    # function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

# thresh is the number of bookies we allow to not have odds on the books


def drop_irrelevant_fights(df, thresh):
    irr = []
    for i in df.index:
        count = 0
        row = list(df.loc[i])
        for j in row:
            if j == '':
                count += 1
        if count > 2*thresh:
            irr.append(i)
    df = df.drop(irr)
    return df

# thresh is the number of bookies we allow to not have odds on the books


def drop_non_ufc_fights(df):
    irr = []
    for i in df.index:
        if (not in_ufc(df['fighter name'][i])) or (not in_ufc(df['opponent name'][i])):
            irr.append(i)
    df = df.drop(irr)
    return df

# thresh is the number of bookies we allow to not have odds on the books


def drop_repeats(df):
    irr = []
    for i in df.index:
        fname = df['fighter name'][i]
        oname = df['opponent name'][i]
        for j in range(200):
            fname_old = ufc_fights['fighter'][j]
            oname_old = ufc_fights['opponent'][j]
            if (same_name(fname, fname_old) and same_name(oname, oname_old)) or (same_name(oname, fname_old) and same_name(fname, oname_old)):
                irr.append(i)
    df = df.drop(irr)
    return df
