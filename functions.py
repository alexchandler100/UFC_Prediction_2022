import pandas as pd
import numpy as npy
from datetime import datetime
from datetime import date

#updated scraped fight data (after running fight_hist_updated function from UFC_data_scraping file)
fight_hist=pd.read_csv('fight_hist.csv',sep=',',low_memory=False)
#all stats fight history file which is one update behind fight_hist
ufcfightscrap=pd.read_csv('ufc_fights_crap.csv',sep=',',low_memory=False)
#updated scraped fighter data (after running fight_hist_updated function from UFC_data_scraping file)
ufcfighters=pd.read_csv('fighter_stats.csv',sep=',',low_memory=False)

#this cell contains all functions defined for building columns in ufcfightscrap
#converts from '%B %d, %Y' (i.e. August 22, 2020) to date (i.e. 2020-08-22)
def convert_to_datetime(day1):
    return datetime.strptime(day1, '%B %d, %Y').date()
convert_to_datetime_vect = npy.vectorize(convert_to_datetime)

#converts 'August 15,2019' to 'Aug 15,2019'
def convert_date_to_abbrev(day):
    return datetime.strptime(day, '%B %d, %Y').strftime('%b %d, %Y')

#converts 'Aug 15,2019' to 'August 15,2019'
def convert_date_to_unabbrev(day):
    return datetime.strptime(day, '%b %d, %Y').strftime('%B %d, %Y')

#this age function is written in such a stupid way
def age(birthDate,day=date.today(),form1='%B %d, %Y',form2='%B %d, %Y'):
    if birthDate=='--':
        aa='unknown'
    elif type(birthDate)==str and not type(day)==str:
        bd=datetime.strptime(birthDate, form1)
        today = day
        aa = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day)) 
    elif type(birthDate)==str and type(day)==str:
        bd=datetime.strptime(birthDate, form1)
        today = datetime.strptime(day, form2)
        aa = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day)) 
    elif not type(birthDate)==str and type(day)==str:
        bd=birthDate
        today = datetime.strptime(day, form2)
        aa = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day)) 
    else:
        bd=birthDate
        today = day
        aa = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day)) 
    return aa

age_vect= npy.vectorize(age)

def fighter_age(fighter,day=date.today(),form1='%B %d, %Y',form2='%B %d, %Y'):
    a=0
    for i in range(len(ufcfighterscrap['name'])):
        if ufcfighterscrap['name'][i]==fighter:
            dob=datetime.strptime(ufcfighterscrap['dob'][i], '%b %d, %Y').strftime('%B %d, %Y')
            a=age(dob,day,form1,form2)
            break
    return a

def wins_before(guy,day1=date.today(),something='%B %d, %Y'):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if time_diff(ufcfightscrap['date'][i],day1)>0 and ufcfightscrap['result'][i]=='W':
            summ+=1
        else:
            summ+=0
    return summ

def losses_before(guy,day1=date.today(),something='%B %d, %Y'):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if time_diff(ufcfightscrap['date'][i],day1)>0 and ufcfightscrap['result'][i]=='L':
            summ+=1
        else:
            summ+=0
    return summ

def record_before(guy,something,day1=date.today()):
    return 0

def ties_before(guy,something,day1=date.today()):
    return 0

#functions for height and reach
def fighter_height(fighter):
    a=0
    for i in range(len(ufcfighterscrap['name'])):
        if ufcfighterscrap['name'][i]==fighter:
            a=ufcfighterscrap['height'][i]
            break
    if a=='--' or a==0:
        b='unknown'
    elif a[4]=='"':
        b=int(a[0])*30.48+int(a[3])*2.54
    else:
        b=int(a[0])*30.48+int(a[3]+a[4])*2.54
    return b

def fighter_reach(fighter):
    a=0
    for i in range(len(ufcfighterscrap['name'])):
        if ufcfighterscrap['name'][i]==fighter:
            a=ufcfighterscrap['reach'][i]
            break
    if a=='--' or a==0:
        b='unknown'
    else:
        b=int(a[0]+a[1])*2.54
    return b

wins_before_vect= npy.vectorize(wins_before)
losses_before_vect= npy.vectorize(losses_before)
fighter_height_vect= npy.vectorize(fighter_height)
fighter_reach_vect= npy.vectorize(fighter_reach)

#day1 should be input in the form '%B %d, %Y' i.e. 'August 20, 1962'
#conversions can be made via day=datetime.strptime(ufcfightsML_known_df['date'][i], '%B %d, %Y').strftime('%b %d, %Y')
def time_diff(day1,day2=date.today()):
    if day2==date.today():
        answer=(day2-datetime.strptime(day1, '%B %d, %Y')).days
    else:
        answer=(datetime.strptime(day2, '%B %d, %Y')-datetime.strptime(day1, '%B %d, %Y')).days
    return answer

#we now vectorize this to use in pandas/numpy
time_diff_vect= npy.vectorize(time_diff)

#can make a single function to do all of these... actually maybe the count function would even work as is

def L5Y_wins(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if 0<time_diff(ufcfightscrap['date'][i],day1)<1825 and ufcfightscrap['result'][i]=='W':
            summ+=1
        else:
            summ+=0
        if time_diff(ufcfightscrap['date'][i],day1)>1825:
            break
    return summ

def L5Y_losses(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if 0<time_diff(ufcfightscrap['date'][i],day1)<1825 and ufcfightscrap['result'][i]=='L':
            summ+=1
        else:
            summ+=0
        if time_diff(ufcfightscrap['date'][i],day1)>1825:
            break
    return summ

def L2Y_wins(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if 0<time_diff(ufcfightscrap['date'][i],day1)<730 and ufcfightscrap['result'][i]=='W':
            summ+=1
        else:
            summ+=0
        if time_diff(ufcfightscrap['date'][i],day1)>730:
            break
    return summ

def L2Y_losses(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if 0<time_diff(ufcfightscrap['date'][i],day1)<730 and ufcfightscrap['result'][i]=='L':
            summ+=1
        else:
            summ+=0
        if time_diff(ufcfightscrap['date'][i],day1)>730:
            break
    return summ

def ko_wins(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if ufcfightscrap['result'][i]=='W' and ufcfightscrap['method'][i]=='KO/TKO':
            summ+=1
        else:
            summ+=0
    return summ

def ko_losses(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if ufcfightscrap['result'][i]=='L' and ufcfightscrap['method'][i]=='KO/TKO':
            summ+=1
        else:
            summ+=0
    return summ

def L5Y_ko_wins(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if 0<time_diff(ufcfightscrap['date'][i],day1)<1825 and ufcfightscrap['result'][i]=='W' and ufcfightscrap['method'][i]=='KO/TKO':
            summ+=1
        else:
            summ+=0
        if time_diff(ufcfightscrap['date'][i],day1)>1825:
            break
    return summ

def L5Y_ko_losses(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if 0<time_diff(ufcfightscrap['date'][i],day1)<1825 and ufcfightscrap['result'][i]=='L' and ufcfightscrap['method'][i]=='KO/TKO':
            summ+=1
        else:
            summ+=0
        if time_diff(ufcfightscrap['date'][i],day1)>1825:
            break
    return summ

def L2Y_ko_wins(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if 0<time_diff(ufcfightscrap['date'][i],day1)<730 and ufcfightscrap['result'][i]=='W' and ufcfightscrap['method'][i]=='KO/TKO':
            summ+=1
        else:
            summ+=0
        if time_diff(ufcfightscrap['date'][i],day1)>730:
            break
    return summ

def L2Y_ko_losses(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if 0<time_diff(ufcfightscrap['date'][i],day1)<730 and ufcfightscrap['result'][i]=='L' and ufcfightscrap['method'][i]=='KO/TKO':
            summ+=1
        else:
            summ+=0
        if time_diff(ufcfightscrap['date'][i],day1)>730:
            break
    return summ

def sub_wins(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if ufcfightscrap['result'][i]=='W' and ufcfightscrap['method'][i]=='SUB':
            summ+=1
        else:
            summ+=0
    return summ

def sub_losses(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if ufcfightscrap['result'][i]=='L' and ufcfightscrap['method'][i]=='SUB':
            summ+=1
        else:
            summ+=0
    return summ

def L5Y_sub_wins(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if 0<time_diff(ufcfightscrap['date'][i],day1)<1825 and ufcfightscrap['result'][i]=='W' and ufcfightscrap['method'][i]=='SUB':
            summ+=1
        else:
            summ+=0
        if time_diff(ufcfightscrap['date'][i],day1)>1825:
            break
    return summ

def L5Y_sub_losses(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if 0<time_diff(ufcfightscrap['date'][i],day1)<1825 and ufcfightscrap['result'][i]=='L' and ufcfightscrap['method'][i]=='SUB':
            summ+=1
        else:
            summ+=0
        if time_diff(ufcfightscrap['date'][i],day1)>1825:
            break
    return summ

def L2Y_sub_wins(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if 0<time_diff(ufcfightscrap['date'][i],day1)<730 and ufcfightscrap['result'][i]=='W' and ufcfightscrap['method'][i]=='SUB':
            summ+=1
        else:
            summ+=0
        if time_diff(ufcfightscrap['date'][i],day1)>730:
            break
    return summ

def L2Y_sub_losses(guy,day1=date.today()):
    if day1==date.today():
        day1=date.today().strftime('%B %d, %Y')
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    for i in good_indices:
        if 0<time_diff(ufcfightscrap['date'][i],day1)<730 and ufcfightscrap['result'][i]=='L' and ufcfightscrap['method'][i]=='SUB':
            summ+=1
        else:
            summ+=0
        if time_diff(ufcfightscrap['date'][i],day1)>730:
            break
    return summ

#vectorize all of these functions
L5Y_wins_vect= npy.vectorize(L5Y_wins)
L5Y_losses_vect= npy.vectorize(L5Y_losses)
L2Y_wins_vect= npy.vectorize(L2Y_wins)
L2Y_losses_vect= npy.vectorize(L2Y_losses)
ko_wins_vect= npy.vectorize(ko_wins)
ko_losses_vect= npy.vectorize(ko_losses)
L5Y_ko_wins_vect= npy.vectorize(L5Y_ko_wins)
L5Y_ko_losses_vect= npy.vectorize(L5Y_ko_losses)
L2Y_ko_wins_vect= npy.vectorize(L2Y_ko_wins)
L2Y_ko_losses_vect= npy.vectorize(L2Y_ko_losses)
sub_wins_vect= npy.vectorize(sub_wins)
sub_losses_vect= npy.vectorize(sub_losses)
L5Y_sub_wins_vect= npy.vectorize(L5Y_sub_wins)
L5Y_sub_losses_vect= npy.vectorize(L5Y_sub_losses)
L2Y_sub_wins_vect= npy.vectorize(L2Y_sub_wins)
L2Y_sub_losses_vect= npy.vectorize(L2Y_sub_losses)

#for columns like fighter_rec which contains the information for the opponent as well, we use the following
def opponent_column(stat):
    col=dict()
    for i in range(len(ufcfightscrap['fighter'])):
        if i%2==0:
            col[i]=ufcfightscrap[stat][i+1]
        else:
            col[i]=ufcfightscrap[stat][i-1]
    statdict={'stat':col}
    return pd.DataFrame (statdict, columns = ['stat'])

#enter date unabbreviated 'July 4, 2019'
#here the average gives avg per fight. Later in avg_count we change to average per time spent in octagon
def count(stat, guy,inf_abs, total_L5Y_L2Y_avg, day1=date.today().strftime('%B %d, %Y')):
    summ=0
    if total_L5Y_L2Y_avg=='total' or total_L5Y_L2Y_avg=='avg':
        good_indices_1=[i for i in ufcfightscrap.index.values if time_diff(ufcfightscrap['date'][i],day1)>0]
    elif total_L5Y_L2Y_avg=='L2Y':
        good_indices_1=[i for i in ufcfightscrap.index.values if 0<time_diff(ufcfightscrap['date'][i],day1)<730]
    else:
        good_indices_1=[i for i in ufcfightscrap.index.values if 0<time_diff(ufcfightscrap['date'][i],day1)<1825]
    if inf_abs=='inf':
        good_indices_2=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy]
    else:
        good_indices_2=[i for i in ufcfightscrap.index.values if ufcfightscrap['opponent'][i]==guy]
    good_indices=[i for i in good_indices_1 if i in good_indices_2]
    if total_L5Y_L2Y_avg!='avg':
        for i in good_indices:
            summ+=ufcfightscrap[stat][i]
    else:
        for i in good_indices:
            summ+=ufcfightscrap[stat][i]
        day1=convert_date_to_abbrev(day1)
        number_fights=wins_before(guy,day1)+losses_before(guy,day1)
        summ=summ/float(number_fights)
    return summ
        
#note a better average is per time not per fight. We will do each stat as an average per one minutes

#enter time in the form 'August 24, 2018'
def time_in_octagon(guy,day1=date.today().strftime('%B %d, %Y')):
    summ=0
    good_indices=[i for i in ufcfightscrap.index.values if time_diff(ufcfightscrap['date'][i],day1)>0 and ufcfightscrap['fighter'][i]==guy] 
    for i in good_indices:
        if ufcfightscrap['time'][i][2]==':':
            summ=int(ufcfightscrap['time'][i][0:2])+int(ufcfightscrap['time'][i][3:])/60.0
        else:
            summ+=5*(ufcfightscrap['round'][i]-1)+int(ufcfightscrap['time'][i][0])+int(ufcfightscrap['time'][i][2:])/60.0
    return summ

#enter date unabbreviated 'July 4, 2019'
#gives takedowns per minute
def avg_count(stat, guy,inf_abs, day1=date.today().strftime('%B %d, %Y')):
    summ=0
    if inf_abs=='inf':
        good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['fighter'][i]==guy and time_diff(ufcfightscrap['date'][i],day1)>0]
    else:
        good_indices=[i for i in ufcfightscrap.index.values if ufcfightscrap['opponent'][i]==guy and time_diff(ufcfightscrap['date'][i],day1)>0]
    for i in good_indices:
        summ+=ufcfightscrap[stat][i]
    t= time_in_octagon(guy,day1)
    if t==0:
        summ=0
    else:
        summ=summ/t
    return summ

#vectorize these functions
count_vect= npy.vectorize(count)
avg_count_vect= npy.vectorize(avg_count)

ufc_fights = pd.read_csv('ufc_fights.csv',low_memory=False)
ufcfighterscrap =pd.read_csv('fighter_stats.csv',sep=',',low_memory=False)

def stance(fighter):
    find_fighter=[i for i in range(len(ufcfighterscrap['name'])) if ufcfighterscrap['name'][i]==fighter]
    b=next(iter(find_fighter),'unknown')
    if b=='unknown':
        a='unknown'
    else:
        a = ufcfighterscrap['stance'][b]
    #for i in range(len(ufcfighterscrap['name'])):
        #if ufcfighterscrap['name'][i]==fighter:
            #a=ufcfighterscrap['stance'][i]
            #break
    if a=='Orthodox':
        return 0
    elif a=='Switch' or a=='Southpaw' or a=='Open Stance' or a=='Sideways':
        return 1
    else:
        return 5
    
stance_vect= npy.vectorize(stance)

def clean_method(a):
    if (a=='KO/TKO'):
        return 'KO/TKO'
    elif (a == 'SUB'):
        return 'SUB'
    elif ((a=='U-DEC') or (a=='M-DEC') or (a=='S-DEC')):
        return 'DEC'
    else:
        return 'bullshit'
    
clean_method_vect= npy.vectorize(clean_method)

def clean_method_for_winner_prediction(a):
    if (a=='KO/TKO'):
        return 'KO/TKO'
    elif (a == 'SUB'):
        return 'SUB'
    elif ((a=='U-DEC') or (a=='M-DEC')):
        return 'DEC'
    #counting S-DEC as bullshit!
    else:
        return 'bullshit'
    
clean_method_for_winner_vect= npy.vectorize(clean_method_for_winner_prediction)