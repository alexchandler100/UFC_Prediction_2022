from datetime import datetime
from Levenshtein import distance as lev
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import git 
import os

# TODO this should all be done with the DataHandler class but this leads to circular imports. Maybe make a DataContainer class
# that both can import
git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
# updated scraped fight data (after running fight_hist_updated function from UFC_data_scraping file)
fight_hist = pd.read_csv('content/data/processed/ufc_fights.csv')
# all stats fight history file which is one update behind fight_hist
ufc_fights_crap = pd.read_csv('content/data/processed/ufc_fights_crap.csv')
# updated scraped fighter data (after running fight_hist_updated function from UFC_data_scraping file)
fighter_stats = pd.read_csv('content/data/processed/fighter_stats.csv')

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
    
same_name_vect = np.vectorize(same_name)

# checks if a fighter is in the ufc


def in_ufc(fighter):
    for name in fighter_stats['name']:
        if same_name(fighter, name):
            return True
    return False

# this cell contains all functions defined for building columns in ufc_fights_crap
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
    for i in range(len(fighter_stats['name'])):
        # if fighter_stats['name'][i]==fighter: #replaced for better accuracy
        if same_name(fighter_stats['name'][i], fighter):
            try:
                dob = datetime.strptime(
                    fighter_stats['dob'][i], '%b %d, %Y').strftime('%B %d, %Y')
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
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if time_diff(ufc_fights_crap['date'][i], day1) > 0 and ufc_fights_crap['result'][i] == 'W':
            summ += 1
        else:
            summ += 0
    return summ


def losses_before(guy, day1=date.today(), something='%B %d, %Y'):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if time_diff(ufc_fights_crap['date'][i], day1) > 0 and ufc_fights_crap['result'][i] == 'L':
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
    for i in range(len(fighter_stats['name'])):
        # if fighter_stats['name'][i]==fighter:
        if same_name(fighter_stats['name'][i], fighter):
            a = fighter_stats['height'][i]
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
    for i in range(len(fighter_stats['name'])):
        # if fighter_stats['name'][i]==fighter:
        if same_name(fighter_stats['name'][i], fighter):
            a = fighter_stats['reach'][i]
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
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufc_fights_crap['date'][i], day1) < 1825 and ufc_fights_crap['result'][i] == 'W':
            summ += 1
        else:
            summ += 0
        if time_diff(ufc_fights_crap['date'][i], day1) > 1825:
            break
    return summ


def L5Y_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufc_fights_crap['date'][i], day1) < 1825 and ufc_fights_crap['result'][i] == 'L':
            summ += 1
        else:
            summ += 0
        if time_diff(ufc_fights_crap['date'][i], day1) > 1825:
            break
    return summ


def L2Y_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufc_fights_crap['date'][i], day1) < 730 and ufc_fights_crap['result'][i] == 'W':
            summ += 1
        else:
            summ += 0
        if time_diff(ufc_fights_crap['date'][i], day1) > 730:
            break
    return summ


def L2Y_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufc_fights_crap['date'][i], day1) < 730 and ufc_fights_crap['result'][i] == 'L':
            summ += 1
        else:
            summ += 0
        if time_diff(ufc_fights_crap['date'][i], day1) > 730:
            break
    return summ


def ko_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if ufc_fights_crap['result'][i] == 'W' and ufc_fights_crap['method'][i] == 'KO/TKO':
            summ += 1
        else:
            summ += 0
    return summ


def ko_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if ufc_fights_crap['result'][i] == 'L' and ufc_fights_crap['method'][i] == 'KO/TKO':
            summ += 1
        else:
            summ += 0
    return summ


def L5Y_ko_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufc_fights_crap['date'][i], day1) < 1825 and ufc_fights_crap['result'][i] == 'W' and ufc_fights_crap['method'][i] == 'KO/TKO':
            summ += 1
        else:
            summ += 0
        if time_diff(ufc_fights_crap['date'][i], day1) > 1825:
            break
    return summ


def L5Y_ko_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufc_fights_crap['date'][i], day1) < 1825 and ufc_fights_crap['result'][i] == 'L' and ufc_fights_crap['method'][i] == 'KO/TKO':
            summ += 1
        else:
            summ += 0
        if time_diff(ufc_fights_crap['date'][i], day1) > 1825:
            break
    return summ


def L2Y_ko_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufc_fights_crap['date'][i], day1) < 730 and ufc_fights_crap['result'][i] == 'W' and ufc_fights_crap['method'][i] == 'KO/TKO':
            summ += 1
        else:
            summ += 0
        if time_diff(ufc_fights_crap['date'][i], day1) > 730:
            break
    return summ


def L2Y_ko_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufc_fights_crap['date'][i], day1) < 730 and ufc_fights_crap['result'][i] == 'L' and ufc_fights_crap['method'][i] == 'KO/TKO':
            summ += 1
        else:
            summ += 0
        if time_diff(ufc_fights_crap['date'][i], day1) > 730:
            break
    return summ


def sub_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if ufc_fights_crap['result'][i] == 'W' and ufc_fights_crap['method'][i] == 'SUB':
            summ += 1
        else:
            summ += 0
    return summ


def sub_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if ufc_fights_crap['result'][i] == 'L' and ufc_fights_crap['method'][i] == 'SUB':
            summ += 1
        else:
            summ += 0
    return summ


def L5Y_sub_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufc_fights_crap['date'][i], day1) < 1825 and ufc_fights_crap['result'][i] == 'W' and ufc_fights_crap['method'][i] == 'SUB':
            summ += 1
        else:
            summ += 0
        if time_diff(ufc_fights_crap['date'][i], day1) > 1825:
            break
    return summ


def L5Y_sub_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufc_fights_crap['date'][i], day1) < 1825 and ufc_fights_crap['result'][i] == 'L' and ufc_fights_crap['method'][i] == 'SUB':
            summ += 1
        else:
            summ += 0
        if time_diff(ufc_fights_crap['date'][i], day1) > 1825:
            break
    return summ


def L2Y_sub_wins(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufc_fights_crap['date'][i], day1) < 730 and ufc_fights_crap['result'][i] == 'W' and ufc_fights_crap['method'][i] == 'SUB':
            summ += 1
        else:
            summ += 0
        if time_diff(ufc_fights_crap['date'][i], day1) > 730:
            break
    return summ


def L2Y_sub_losses(guy, day1=date.today()):
    if day1 == date.today():
        day1 = date.today().strftime('%B %d, %Y')
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if same_name(
        ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if 0 < time_diff(ufc_fights_crap['date'][i], day1) < 730 and ufc_fights_crap['result'][i] == 'L' and ufc_fights_crap['method'][i] == 'SUB':
            summ += 1
        else:
            summ += 0
        if time_diff(ufc_fights_crap['date'][i], day1) > 730:
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
def opponent_column(fighter_col):
    col = np.array(fighter_col)
    for i in range(len(fighter_col)):
        if i % 2 == 0:
            col[i] = fighter_col[i+1]
        else:
            col[i] = fighter_col[i-1]
    return pd.Series(col)

# enter date unabbreviated 'July 4, 2019'
# here the average gives avg per fight. Later in avg_count we change to average per time spent in octagon


def count(stat, guy, inf_abs, total_L5Y_L2Y_avg, day1=date.today().strftime('%B %d, %Y')):
    summ = 0
    if total_L5Y_L2Y_avg == 'total' or total_L5Y_L2Y_avg == 'avg':
        good_indices_1 = [i for i in ufc_fights_crap.index.values if time_diff(
            ufc_fights_crap['date'][i], day1) > 0]
    elif total_L5Y_L2Y_avg == 'L2Y':
        good_indices_1 = [i for i in ufc_fights_crap.index.values if 0 < time_diff(
            ufc_fights_crap['date'][i], day1) < 730]
    else:
        good_indices_1 = [i for i in ufc_fights_crap.index.values if 0 < time_diff(
            ufc_fights_crap['date'][i], day1) < 1825]
    if inf_abs == 'inf':
        good_indices_2 = [i for i in ufc_fights_crap.index.values if same_name(
            ufc_fights_crap['fighter'][i], guy)]
    else:
        good_indices_2 = [i for i in ufc_fights_crap.index.values if same_name(
            ufc_fights_crap['opponent'][i], guy)]
    good_indices = [i for i in good_indices_1 if i in good_indices_2]
    if total_L5Y_L2Y_avg != 'avg':
        for i in good_indices:
            summ += ufc_fights_crap[stat][i]
    else:
        for i in good_indices:
            summ += ufc_fights_crap[stat][i]
        day1 = convert_date_to_abbrev(day1)
        number_fights = wins_before(guy, day1)+losses_before(guy, day1)
        summ = summ/float(number_fights)
    return summ

# note a better average is per time not per fight. We will do each stat as an average per one minutes

# enter time in the form 'August 24, 2018'


def time_in_octagon(guy, day1=date.today().strftime('%B %d, %Y')):
    summ = 0
    good_indices = [i for i in ufc_fights_crap.index.values if time_diff(
        ufc_fights_crap['date'][i], day1) > 0 and same_name(ufc_fights_crap['fighter'][i], guy)]
    for i in good_indices:
        if ufc_fights_crap['time'][i][2] == ':':
            summ = int(ufc_fights_crap['time'][i][0:2]) + \
                int(ufc_fights_crap['time'][i][3:])/60.0
        else:
            summ += 5*(ufc_fights_crap['round'][i]-1)+int(ufc_fights_crap['time']
                                                        [i][0])+int(ufc_fights_crap['time'][i][2:])/60.0
    return summ

# enter date unabbreviated 'July 4, 2019'
# gives takedowns per minute


def avg_count(stat, guy, inf_abs, day1=date.today().strftime('%B %d, %Y')):
    # TODO make this faster by using vectorized operations like masking, same_name_vect, time_diff_vect, etc.
    summ = 0
    if inf_abs == 'inf':
        good_indices = [i for i in ufc_fights_crap.index.values if same_name(
            ufc_fights_crap['fighter'][i], guy) and time_diff(ufc_fights_crap['date'][i], day1) > 0]
    else:
        good_indices = [i for i in ufc_fights_crap.index.values if same_name(
            ufc_fights_crap['opponent'][i], guy) and time_diff(ufc_fights_crap['date'][i], day1) > 0]
    for i in good_indices:
        summ += ufc_fights_crap[stat][i]
    t = time_in_octagon(guy, day1)
    if t == 0:
        summ = 0
    else:
        summ = summ/t
    return summ

# vectorize these functions
count_vect = np.vectorize(count)
avg_count_vect = np.vectorize(avg_count)

def stance(fighter):
    find_fighter = [i for i in range(len(fighter_stats['name'])) if same_name(
        fighter_stats['name'][i], fighter)]
    b = next(iter(find_fighter), 'unknown')
    if b == 'unknown':
        a = 'unknown'
    else:
        a = fighter_stats['stance'][b]
    if a == 'Orthodox':
        return 0
    elif a == 'Switch' or a == 'Southpaw' or a == 'Open Stance' or a == 'Sideways':
        return 1
    else:
        return 5


stance_vect = np.vectorize(stance)

# Some stuff needed to compute fight math and fighter scores
ufc_fights_graph = ufc_fights_crap.copy()
odd_indices = range(1, len(ufc_fights_graph.index), 2)
ufc_fights_graph = ufc_fights_graph.drop(odd_indices)
ufc_fights_graph = ufc_fights_graph[['fighter', 'opponent', 'method', 'date', 'division']]
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

# Methods for cleaning columns in the dataframe
@np.vectorize
def clean_method_for_method_predictions(a):
    if (a == 'KO/TKO'):
        return 'KO/TKO'
    elif (a == 'SUB'):
        return 'SUB'
    elif ((a == 'U-DEC') or (a == 'M-DEC') or (a == 'S-DEC')):
        return 'DEC'
    else:
        return 'bullshit'

@np.vectorize
def clean_method_for_winner_predictions(a):
    if (a == 'KO/TKO'):
        return 'KO/TKO'
    elif (a == 'SUB'):
        return 'SUB'
    elif ((a == 'U-DEC') or (a == 'M-DEC')):
        return 'DEC'
    # counting S-DEC as bullshit!
    else:
        return 'bullshit'

# for computing expectation values
# maybe has overlap with functions in the fight predictor
def odds_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def odds_to_return(odds):
    """Convert American odds to return per $1 bet."""
    if odds > 0:
        return odds / 100
    else:
        return 100 / abs(odds)

def expected_value(fair_odds_a, vegas_odds_a, vegas_odds_b):
    # dont need fair_odds_b since fair odds are equal
    # Convert fair odds to probabilities
    p_a = odds_to_prob(fair_odds_a)
    p_b = 1 - p_a  # assume binary outcome

    # Convert Vegas odds to returns
    r_a = odds_to_return(vegas_odds_a)
    r_b = odds_to_return(vegas_odds_b)

    # Expected value per $1 bet
    ev_a = p_a * r_a - p_b  # betting on Team A
    ev_b = p_b * r_b - p_a  # betting on Team B

    return {
        "fighter_ev": round(ev_a, 4),
        "opponent_ev": round(ev_b, 4),
        "best_bet": "fighter" if ev_a > ev_b and ev_a > 0 else ("opponent" if ev_b > ev_a and ev_b > 0 else "No +EV bet")
    }
    
def odds_to_profit(odds):
    """Convert American odds to profit per $1 bet."""
    if odds > 0:
        return odds / 100
    else:
        return 100 / abs(odds)  # profit + initial bet

def get_kelly_bet_from_ev_and_dk_odds(predicted_fighter_odds, fighter_dk_odds, opponent_dk_odds):
    """
    Calculate the profit from expected value and DraftKings odds.
    
    Parameters:
    - predicted_fighter_odds: Predicted odds for the fighter.
    - dk_odds_fighter: DraftKings odds for the fighter.
    - dk_odds_opponent: DraftKings odds for the opponent.
    
    Returns:
    - fighter_bankroll_percentage: Percentage of bankroll to bet based on Kelly Criterion.
    - opponent_bankroll_percentage: Percentage of bankroll to bet based on Kelly Criterion.
    """
    # TODO ensure ev and odds values exist and are valid
    # first convert predicted odds from American format to a probability 
    predicted_fighter_probability = odds_to_prob(predicted_fighter_odds)

    fighter_b = odds_to_profit(fighter_dk_odds)
    fighter_bankroll_percentage = (predicted_fighter_probability - (1 - predicted_fighter_probability) / fighter_b) * 100

    opponent_b = odds_to_profit(opponent_dk_odds)
    predicted_opponent_probability = 1 - predicted_fighter_probability
    opponent_bankroll_percentage = (predicted_opponent_probability - (1 - predicted_opponent_probability) / opponent_b) * 100

    return fighter_bankroll_percentage, opponent_bankroll_percentage

def bet_payout(american_odds, bet_amount, result):
    """
    Calculate the payout from a bet based on American odds.
    
    Parameters:
    - american_odds: The American odds for the bet.
    - bet_amount: The amount of money bet.
    - result: The result of the bet ('W' or 'L').
    
    Returns:
    - payout: The total payout from the bet.
    """
    american_odds = int(american_odds)
    bet_amount = float(bet_amount)
    assert result in ['W', 'L'], "Result must be 'W' or 'L'."
    if result == 'W':
        if american_odds > 0:
            payout = bet_amount + (bet_amount * american_odds / 100)
        else:
            payout = bet_amount + (bet_amount * 100 / abs(american_odds))
    elif result == 'L':
        payout = 0
    else:
        raise ValueError("Result must be 'W' or 'L'.")
    
    return payout