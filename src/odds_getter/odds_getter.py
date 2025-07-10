# need to use selenium as the javascript renders the html after the page load
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

class OddsGetter:
    def __init__(self):
        self.fight_odds_url = "https://fightodds.io"
        
    def make_odds_df(self):
        # Setup Chrome options
        options = Options()
        options.add_argument("--headless")  # Run in headless mode (no window)
        options.add_argument("--disable-gpu")  # Optional: disables GPU (for stability)
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")


        # Create WebDriver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        # Load the page
        driver.get(self.fight_odds_url)

        # Wait for JavaScript to render (adjust time if needed or use WebDriverWait)
        time.sleep(10) # changed from 5 to 10 seconds to avoid index out of range error

        # Get the rendered HTML
        html = driver.page_source

        # Close the browser
        driver.quit()

        # Now parse with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        
        # do we need the date?
        # date = soup.find_all("span", class_="jss2286")[0].get_text()
        # # Looks like "June 27" with no year, so we need to add the current year
        # current_year = pd.Timestamp.now().year
        # date = np.datetime.datetime.strptime(f"{current_year} {date}", "%Y %B %d").strftime("%Y-%m-%d")
        
        # TODO figure out why I got an index out of range error here.. need to do time.sleep for longer?
        odds_sections = soup.find_all("thead", class_="MuiTableHead-root")[0]
        odds_data = soup.find_all("tbody", class_="MuiTableBody-root")[0]

        self.df = self.get_fighter_odds_for_card(odds_data, odds_sections)
        # self.df["date"] = date
        return self.df
    
    def get_name(self, row):
        name = row.find_all("td")[0].find("a").get_text()
        return name

    def get_odds(self, row, bookies_list):
        # td_list = row.find_all("td")
        # odds_list = []
        # for td in td_list[1:]:
        #     odds = td.find_all("span", class_="jss1669 false")[0].get_text()
        #     odds_list.append(odds)
        # return odds_list
        td_list = row.find_all("td")
        td_list
        odds_list = []
        for td in td_list[1:len(bookies_list)+1]:
            odds_results = td.find_all("span")
            if len(odds_results) > 0:
                text = odds_results[0].get_text()
                odds_list.append(text)
            else:
                # for debugging purposes
                # print("No odds found in this td")
                # print(td.prettify())
                odds_list.append("")
        return odds_list

    def get_fighter_odds_for_card(self, odds_data, odds_sections):
        print(f"Found {len(odds_data)} odds containers")
        section_rows = odds_sections.find_all("tr")
        bookies = section_rows[0].find_all("th")
        bookies_list = [bookie.get_text() for bookie in bookies[1:-1]] # empty first and last bookies
        rows = odds_data.find_all("tr")
        data = []

        print(f'bookies_list: {bookies_list}')
        for half_row_idx in range(len(rows) // 2):
            fighter_row_idx = half_row_idx * 2
            opponent_row_idx = fighter_row_idx + 1
            fighter_row = rows[fighter_row_idx]
            opponent_row = rows[opponent_row_idx]
            
            fighter_name = self.get_name(fighter_row)
            opponent_name = self.get_name(opponent_row)
            fighter_odds_list = self.get_odds(fighter_row, bookies_list)
            opponent_odds_list = self.get_odds(opponent_row, bookies_list)
            
            # add fighter name and odds to dataframe row
            df_row = {"fighter name": fighter_name}
            for bookie, fighter_odds in zip(bookies_list, fighter_odds_list):
                df_row[f"fighter {bookie}"] = fighter_odds
                
            # add opponent name and odds to dataframe row
            df_row["opponent name"] = opponent_name
            for bookie, opponent_odds in zip(bookies_list, opponent_odds_list):
                df_row[f"opponent {bookie}"] = opponent_odds
                
            data.append(df_row)
            
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)
        df["predicted fighter odds"] = np.nan
        df["predicted opponent odds"] = np.nan
        average_bookie_fighter_odds = np.mean(df[[f"fighter {bookie}" for bookie in bookies_list]].replace("", np.nan).astype(float), axis=1)
        average_bookie_opponent_odds = np.mean(df[[f"opponent {bookie}" for bookie in bookies_list]].replace("", np.nan).astype(float), axis=1)
        # make a column with values of the form [str(avg_fighter_odds), str(avg_opponent_odds)]
        df['average bookie odds'] = [[str(round(av_fighter_odds)), str(round(av_opponent_odds))] for av_fighter_odds, av_opponent_odds in zip(average_bookie_fighter_odds, average_bookie_opponent_odds)]
        return df
