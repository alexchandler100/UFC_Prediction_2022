import git
import os 
import pandas as pd
from bs4 import BeautifulSoup
import urllib.request
import requests
import csv
import json
import date

# local imports
from functions import (in_ufc, 
                       same_name, 
                       wins_before_vect, 
                       losses_before_vect, 
                       fighter_age_vect, 
                       fighter_height, 
                       L5Y_wins_vect, 
                       L5Y_losses_vect, 
                       L2Y_wins_vect, 
                       L2Y_losses_vect, 
                       ko_wins_vect, 
                       ko_losses_vect, 
                       L5Y_ko_wins_vect, 
                       L5Y_ko_losses_vect, 
                       L2Y_ko_wins_vect, 
                       L2Y_ko_losses_vect, 
                       sub_wins_vect, 
                       sub_losses_vect, 
                       L5Y_sub_wins_vect, 
                       L5Y_sub_losses_vect, 
                       L2Y_sub_wins_vect, 
                       L2Y_sub_losses_vect, 
                       avg_count_vect, 
                       zero_vect, 
                       opponent_column,
                       stance_vect
            )

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")

class DataHandler:
    def __init__(self):
        # updated scraped fight data (after running fight_hist_updated function from UFC_data_scraping file)
        self.filepaths = {
            'fight_hist': f'{git_root}/src/content/data/processed/fight_hist.csv',
            'ufc_fights_crap': f'{git_root}/src/content/data/processed/ufc_fights_crap.csv',
            'fighter_stats': f'{git_root}/src/content/data/processed/fighter_stats.csv',
            'ufc_fights': f'{git_root}/src/content/data/processed/ufc_fights.csv'
        }
        self.data = {key : pd.read_csv(self.filepaths[key], sep=',', low_memory=False) for key in self.filepaths.keys()}

    def get(self, key):
        assert key in list(self.data.keys()), "Invalid key provided"
        return self.data[key]
    
    def set(self, key, value):
        assert key in list(self.data.keys()), "Invalid key provided"
        self.data[key] = value
    
    def save(self, key):
        assert key in list(self.data.keys()), "Invalid key provided"
        self.data[key].to_csv(self.filepaths[key], index=False)
        
    def update(self, key):
        assert key in list(self.data.keys()), "Invalid key provided"
        if key == 'fight_hist':
            self.update_fight_hist()
        elif key == 'fighter_stats':
            # TODO implement fighter stats update
            pass
        elif key == 'ufc_fights_crap':
            # TODO implement ufc fights crap update
            pass
        elif key == 'ufc_fights':
            # TODO implement ufc fights update
            pass
        else:
            raise ValueError("No update function implemented for this key")
                
    def update_fight_hist(self):  # takes dataframe of fight stats as input
        old_stats = self.get('fight_hist')
        url = 'http://ufcstats.com/statistics/events/completed?page=all'
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        events_table = soup.select_one('tbody')
        new_stats = pd.DataFrame()
        try:
            events = [event['href'] for event in events_table.select( 'a')[1:]] # omit first event (future event) # TODO WE MAY AS WELL USE THIS TO POPULATE THE FUTURE EVENT INSTEAD OF GETTING IT FROM ANOTHER WEBSITE LATER...
            saved_events = set(old_stats.event_url.unique())
            new_events = [event for event in events if event not in saved_events]  # get only new events
            for event in new_events: # skip events that are already in the old_stats
                print(event)
                stats = self.get_fight_card(event)
                new_stats = pd.concat([new_stats, stats], axis=0)
        except:
            print('update_fight_hist failed... if there is an event going on right now, this will not run correctly')
        updated_stats = pd.concat([new_stats, old_stats], axis=0)
        updated_stats = updated_stats.reset_index(drop=True)
        # set fight_hist and save it to csv
        self.set('fight_hist', updated_stats)
        self.save('fight_hist')
    

    def get_fight_card(self, url):
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
            str_det = self.get_fight_stats(fight_url)
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
    
        
    # there is a problem for collecting reversals (fix needed) seems like it now collect riding time since sept 2020
    # function for getting individual fight stats
    def get_fight_stats(self, url):
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

            cfd = pd.merge(ov_details, sig_details, on='fighter', how='left', copy=False)

            cfd['takedowns_landed'] = cfd.takedowns.str.split( ' of ').str[0].astype(int)
            cfd['takedowns_attempts'] = cfd.takedowns.str.split( ' of ').str[-1].astype(int)
            cfd['sig_strikes_landed'] = cfd.sig_strikes.str.split( ' of ').str[0].astype(int)
            cfd['sig_strikes_attempts'] = cfd.sig_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['total_strikes_landed'] = cfd.total_strikes.str.split( ' of ').str[0].astype(int)
            cfd['total_strikes_attempts'] = cfd.total_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['head_strikes_landed'] = cfd.head_strikes.str.split( ' of ').str[0].astype(int)
            cfd['head_strikes_attempts'] = cfd.head_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['body_strikes_landed'] = cfd.body_strikes.str.split( ' of ').str[0].astype(int)
            cfd['body_strikes_attempts'] = cfd.body_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['leg_strikes_landed'] = cfd.leg_strikes.str.split( ' of ').str[0].astype(int)
            cfd['leg_strikes_attempts'] = cfd.leg_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['distance_strikes_landed'] = cfd.distance_strikes.str.split( ' of ').str[0].astype(int)
            cfd['distance_strikes_attempts'] = cfd.distance_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['clinch_strikes_landed'] = cfd.clinch_strikes.str.split( ' of ').str[0].astype(int)
            cfd['clinch_strikes_attempts'] = cfd.clinch_strikes.str.split( ' of ').str[-1].astype(int)
            cfd['ground_strikes_landed'] = cfd.ground_strikes.str.split( ' of ').str[0].astype(int)
            cfd['ground_strikes_attempts'] = cfd.ground_strikes.str.split( ' of ').str[-1].astype(int)

            cfd = cfd.drop(['takedowns', 'sig_strikes', 'total_strikes', 'head_strikes', 'body_strikes', 'leg_strikes', 'distance_strikes',
                            'clinch_strikes', 'ground_strikes'], axis=1)
            return (cfd)
        
    # updates fighter attributes with new fighters not yet saved yet
    def update_fighter_stats(self, saved_fighters):
        # TODO find a way to avoid using the old version of fighter_stats.csv ... this is clunky
        fight_hist = self.get('fight_hist')
        fighter_urls = fight_hist.fighter_url.unique()
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
                urllib.request.urlretrieve(links[i], f"{git_root}/src/content/images/"+str(i)+name_reduced+".jpg")
            print('scraped 5 random pictures of '+name+' from Google search')

        except:
            print('The scrape did not work for '+name)
        
    # Function to convert a CSV to JSON
    # Takes the file paths as arguments
    def make_json(self, csvFilePath, jsonFilePath, column):

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
    # TODO name should better indicate the context
    def drop_irrelevant_fights(self, df, thresh):
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


    # TODO name should better indicate the context
    def drop_non_ufc_fights(self, df):
        irr = []
        for i in df.index:
            if (not in_ufc(df['fighter name'][i])) or (not in_ufc(df['opponent name'][i])):
                irr.append(i)
        df = df.drop(irr)
        return df

    # TODO name should better indicate the context
    def drop_repeats(self, df):
        irr = []
        ufc_fights = self.get('ufc_fights')
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
    
    # TODO name should better indicate the context
    def get_bad_indices(self, vegas_odds_old, ufc_fights_crap):
        r"""
        This function checks the vegas odds dataframe against the ufc fights dataframe to find fights that didn't happen
        and to add correctness results for those that did happen. It returns a list of indices of fights that didn't happen.
        It also updates the vegas odds dataframe with correctness results for the fights that did happen.
        """
        # getting rid of fights that didn't actually happen and adding correctness results of those that did
        bad_indices = []
        for index1, row1 in vegas_odds_old.iterrows():
            card_date = row1['date']
            relevant_fights = ufc_fights_crap[pd.to_datetime(ufc_fights_crap['date']) == card_date]
            print(f'searching through {relevant_fights.shape[0]//2} confirmed fights on {str(card_date).split(" ")[0]} for {row1["fighter name"]} vs {row1["opponent name"]}')
            fighter_odds = row1['predicted fighter odds']
            match_found = False
            # if no prediction was made, throw it away
            if fighter_odds == '':
                bad_indices.append(index1)
                print('no prediction made for fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'])
            else: # if a prediction was made, check if the fight actually happened and then check if the prediction was correct
                for index2, row2 in relevant_fights.iterrows():
                    if same_name(row1['fighter name'], row2['fighter']) and same_name(row1['opponent name'], row2['opponent']):
                        match_found = True
                        print('adding fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'])
                        if (int(fighter_odds) < 0 and row2['result'] == 'W') or (int(fighter_odds) > 0 and row2['result'] == 'L'):
                            vegas_odds_old.at[index1,'correct?'] = 1
                        else:
                            vegas_odds_old.at[index1,'correct?'] = 0
                        # TODO add case for draw
                        break
                if not match_found: # if the fight didn't happen, throw it away
                    bad_indices.append(index1)
                    print('fight from '+str(card_date)+' between '+row1['fighter name']+' and '+row1['opponent name'] + ' not found in ufc_fights_crap.csv')
                
                
    def populate_new_fights_with_statistics(self, new_rows):
        print('adding physical statistics for fighter')
        new_rows['fighter_wins'] = wins_before_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_losses'] = losses_before_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_age'] = fighter_age_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_height'] = new_rows['fighter'].apply(fighter_height)
        new_rows['fighter_reach'] = new_rows['fighter'].apply(fighter_height)

        print('adding record statistics for fighter')
        new_rows['fighter_L5Y_wins'] = L5Y_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_L5Y_losses'] = L5Y_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_L2Y_wins'] = L2Y_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_L2Y_losses'] = L2Y_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_ko_wins'] = ko_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_ko_losses'] = ko_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_L5Y_ko_wins'] = L5Y_ko_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_L5Y_ko_losses'] = L5Y_ko_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_L2Y_ko_wins'] = L2Y_ko_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_L2Y_ko_losses'] = L2Y_ko_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_sub_wins'] = sub_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_sub_losses'] = sub_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_L5Y_sub_wins'] = L5Y_sub_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_L5Y_sub_losses'] = L5Y_sub_losses_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_L2Y_sub_wins'] = L2Y_sub_wins_vect(new_rows['fighter'], new_rows['date'])
        new_rows['fighter_L2Y_sub_losses'] = L2Y_sub_losses_vect(new_rows['fighter'], new_rows['date'])

        print('adding inflicted punch, kick, grappling statistics for fighter... this will take a few minutes')

        new_rows['fighter_inf_knockdowns_avg'] = avg_count_vect('knockdowns', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_pass_avg'] = avg_count_vect('pass', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_reversals_avg'] = zero_vect(new_rows['fighter'])
        new_rows['fighter_inf_sub_attempts_avg'] = avg_count_vect('sub_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_takedowns_landed_avg'] = avg_count_vect('takedowns_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_takedowns_attempts_avg'] = avg_count_vect('takedowns_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        print('quarter done')
        new_rows['fighter_inf_sig_strikes_landed_avg'] = avg_count_vect('sig_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_sig_strikes_attempts_avg'] = avg_count_vect('sig_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_total_strikes_landed_avg'] = avg_count_vect('total_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_total_strikes_attempts_avg'] = avg_count_vect('total_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_head_strikes_landed_avg'] = avg_count_vect('head_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_head_strikes_attempts_avg'] = avg_count_vect('head_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        print('half done')
        new_rows['fighter_inf_body_strikes_landed_avg'] = avg_count_vect('body_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_body_strikes_attempts_avg'] = avg_count_vect('body_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_leg_strikes_landed_avg'] = avg_count_vect('leg_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_leg_strikes_attempts_avg'] = avg_count_vect('leg_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        print('almost done')
        new_rows['fighter_inf_distance_strikes_attempts_avg'] = avg_count_vect('distance_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_clinch_strikes_landed_avg'] = avg_count_vect('clinch_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_clinch_strikes_attempts_avg'] = avg_count_vect('clinch_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_ground_strikes_landed_avg'] = avg_count_vect('ground_strikes_landed', new_rows['fighter'], 'inf', new_rows['date'])
        new_rows['fighter_inf_ground_strikes_attempts_avg'] = avg_count_vect('ground_strikes_attempts', new_rows['fighter'], 'inf', new_rows['date'])

        print('adding absorbed punch, kick, grappling statistics for fighter... this will take a few minutes')

        new_rows['fighter_abs_knockdowns_avg'] = avg_count_vect('knockdowns', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_pass_avg'] = avg_count_vect('pass', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_reversals_avg'] = zero_vect(new_rows['fighter'])
        new_rows['fighter_abs_sub_attempts_avg'] = avg_count_vect('sub_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_takedowns_landed_avg'] = avg_count_vect('takedowns_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_takedowns_attempts_avg'] = avg_count_vect('takedowns_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        print('quarter done')
        new_rows['fighter_abs_sig_strikes_landed_avg'] = avg_count_vect('sig_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_sig_strikes_attempts_avg'] = avg_count_vect('sig_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_total_strikes_landed_avg'] = avg_count_vect('total_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_total_strikes_attempts_avg'] = avg_count_vect('total_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_head_strikes_landed_avg'] = avg_count_vect('head_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_head_strikes_attempts_avg'] = avg_count_vect('head_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        print('half done')
        new_rows['fighter_abs_body_strikes_landed_avg'] = avg_count_vect('body_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_body_strikes_attempts_avg'] = avg_count_vect('body_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_leg_strikes_landed_avg'] = avg_count_vect('leg_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_leg_strikes_attempts_avg'] = avg_count_vect('leg_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        print('almost done')
        new_rows['fighter_abs_distance_strikes_attempts_avg'] = avg_count_vect('distance_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_clinch_strikes_landed_avg'] = avg_count_vect('clinch_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_clinch_strikes_attempts_avg'] = avg_count_vect('clinch_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_ground_strikes_landed_avg'] = avg_count_vect('ground_strikes_landed', new_rows['fighter'], 'abs', new_rows['date'])
        new_rows['fighter_abs_ground_strikes_attempts_avg'] = avg_count_vect('ground_strikes_attempts', new_rows['fighter'], 'abs', new_rows['date'])

        print('adding physical statistics for opponent')

        new_rows['opponent_wins'] = opponent_column('fighter_wins')
        new_rows['opponent_losses'] = opponent_column('fighter_losses')
        new_rows['opponent_age'] = opponent_column('fighter_age')
        new_rows['opponent_height'] = opponent_column('fighter_height')
        new_rows['opponent_reach'] = opponent_column('fighter_reach')

        print('adding record statistics for opponent')

        new_rows['opponent_L5Y_wins'] = opponent_column('fighter_L5Y_wins')
        new_rows['opponent_L5Y_losses'] = opponent_column('fighter_L5Y_losses')
        new_rows['opponent_L2Y_wins'] = opponent_column('fighter_L2Y_wins')
        new_rows['opponent_L2Y_losses'] = opponent_column('fighter_L2Y_losses')
        new_rows['opponent_ko_wins'] = opponent_column('fighter_ko_wins')
        new_rows['opponent_ko_losses'] = opponent_column('fighter_ko_losses')
        new_rows['opponent_L5Y_ko_wins'] = opponent_column('fighter_L5Y_ko_wins')
        new_rows['opponent_L5Y_ko_losses'] = opponent_column('fighter_L5Y_ko_losses')
        new_rows['opponent_L2Y_ko_wins'] = opponent_column('fighter_L2Y_ko_wins')
        new_rows['opponent_L2Y_ko_losses'] = opponent_column('fighter_L2Y_ko_losses')
        new_rows['opponent_sub_wins'] = opponent_column('fighter_sub_wins')
        new_rows['opponent_sub_losses'] = opponent_column('fighter_sub_losses')
        new_rows['opponent_L5Y_sub_wins'] = opponent_column('fighter_L5Y_sub_wins')
        new_rows['opponent_L5Y_sub_losses'] = opponent_column('fighter_L5Y_sub_losses')
        new_rows['opponent_L2Y_sub_wins'] = opponent_column('fighter_L2Y_sub_wins')
        new_rows['opponent_L2Y_sub_losses'] = opponent_column('fighter_L2Y_sub_losses')

        print('adding inflicted punch, kick, grappling statistics for opponent... this will take a few minutes')

        new_rows['opponent_inf_knockdowns_avg'] = avg_count_vect('knockdowns', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_pass_avg'] = avg_count_vect('pass', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_reversals_avg'] = zero_vect(new_rows['opponent'])
        new_rows['opponent_inf_sub_attempts_avg'] = avg_count_vect('sub_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_takedowns_landed_avg'] = avg_count_vect('takedowns_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_takedowns_attempts_avg'] = avg_count_vect('takedowns_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        print('quarter done')
        new_rows['opponent_inf_sig_strikes_landed_avg'] = avg_count_vect('sig_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_sig_strikes_attempts_avg'] = avg_count_vect('sig_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_total_strikes_landed_avg'] = avg_count_vect('total_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_total_strikes_attempts_avg'] = avg_count_vect('total_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_head_strikes_landed_avg'] = avg_count_vect('head_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        print('half done')
        new_rows['opponent_inf_head_strikes_attempts_avg'] = avg_count_vect('head_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_body_strikes_landed_avg'] = avg_count_vect('body_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_body_strikes_attempts_avg'] = avg_count_vect('body_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_leg_strikes_landed_avg'] = avg_count_vect('leg_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_leg_strikes_attempts_avg'] = avg_count_vect('leg_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        print('almost done')
        new_rows['opponent_inf_distance_strikes_attempts_avg'] = avg_count_vect('distance_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_clinch_strikes_landed_avg'] = avg_count_vect('clinch_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_clinch_strikes_attempts_avg'] = avg_count_vect('clinch_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_ground_strikes_landed_avg'] = avg_count_vect('ground_strikes_landed', new_rows['opponent'], 'inf', new_rows['date'])
        new_rows['opponent_inf_ground_strikes_attempts_avg'] = avg_count_vect('ground_strikes_attempts', new_rows['opponent'], 'inf', new_rows['date'])

        print('adding absorbed punch, kick, grappling statistics for opponent... this will take a few minutes')

        new_rows['opponent_abs_knockdowns_avg'] = avg_count_vect('knockdowns', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_pass_avg'] = avg_count_vect('pass', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_reversals_avg'] = zero_vect(new_rows['opponent'])
        new_rows['opponent_abs_sub_attempts_avg'] = avg_count_vect('sub_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_takedowns_landed_avg'] = avg_count_vect('takedowns_landed', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_takedowns_attempts_avg'] = avg_count_vect('takedowns_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_sig_strikes_landed_avg'] = avg_count_vect('sig_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        print('quarter done')
        new_rows['opponent_abs_sig_strikes_attempts_avg'] = avg_count_vect('sig_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_total_strikes_landed_avg'] = avg_count_vect('total_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_total_strikes_attempts_avg'] = avg_count_vect('total_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_head_strikes_landed_avg'] = avg_count_vect('head_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_head_strikes_attempts_avg'] = avg_count_vect('head_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_body_strikes_landed_avg'] = avg_count_vect('body_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        print('half done')
        new_rows['opponent_abs_body_strikes_attempts_avg'] = avg_count_vect('body_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_leg_strikes_landed_avg'] = avg_count_vect('leg_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_leg_strikes_attempts_avg'] = avg_count_vect('leg_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_distance_strikes_landed_avg'] = avg_count_vect('distance_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        print('almost done')
        new_rows['opponent_abs_distance_strikes_attempts_avg'] = avg_count_vect('distance_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_clinch_strikes_landed_avg'] = avg_count_vect('clinch_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_clinch_strikes_attempts_avg'] = avg_count_vect('clinch_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_ground_strikes_landed_avg'] = avg_count_vect('ground_strikes_landed', new_rows['opponent'], 'abs', new_rows['date'])
        new_rows['opponent_abs_ground_strikes_attempts_avg'] = avg_count_vect('ground_strikes_attempts', new_rows['opponent'], 'abs', new_rows['date'])

        new_rows['fighter_stance'] = stance_vect(new_rows['fighter'])
        new_rows['opponent_stance'] = stance_vect(new_rows['opponent'])

        print('adding fight_math and fighter_score statistics')
        new_rows['1-fight_math'] = fight_math_diff_vect(new_rows['fighter'], new_rows['opponent'], new_rows['date'], 1)
        new_rows['6-fight_math'] = fight_math_diff_vect(new_rows['fighter'], new_rows['opponent'], new_rows['date'], 6)
        new_rows['4-fighter_score_diff'] = fighter_score_diff_vect(new_rows['fighter'], new_rows['opponent'], new_rows['date'], 4)
        new_rows['9-fighter_score_diff'] = fighter_score_diff_vect(new_rows['fighter'], new_rows['opponent'], new_rows['date'], 9)
        new_rows['15-fighter_score_diff'] = fighter_score_diff_vect(new_rows['fighter'], new_rows['opponent'], new_rows['date'], 15)
    
    ########### FUNCTIONS USED IN update_data_csvs_and_jsons.py ###########
    def get_odds_two_rows_per_fight(self):
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
    def get_odds(self):
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
    def convert_scraped_date_to_standard_date(self, input_date) -> str:
        year = date.today().strftime('%B %d, %Y')[-4:]
        month = input_date.split(' ')[0]
        day = input_date.split(' ')[1]
        i = 0
        while day[i].isdigit():
            i+=1
        day = day[:i]
        return f'{month} {day}, {year}'

    def get_next_fight_card(self):
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