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

        # Create WebDriver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        # Load the page
        driver.get(self.fight_odds_url)

        # Wait for JavaScript to render (adjust time if needed or use WebDriverWait)
        time.sleep(5)

        # Get the rendered HTML
        html = driver.page_source

        # Close the browser
        driver.quit()

        # Now parse with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        
        odds_sections = soup.find_all("thead", class_="MuiTableHead-root")[0]
        odds_data = soup.find_all("tbody", class_="MuiTableBody-root")[0]

        self.df = self.get_fighter_odds_for_card(odds_data, odds_sections)
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
                print("No odds found in this td")
                print(td.prettify())
                odds_list.append("")
        return odds_list

    def get_fighter_odds_for_card(self, odds_data, odds_sections):
        print(f"Found {len(odds_data)} odds containers")
        section_rows = odds_sections.find_all("tr")
        # td0 = row0.find_all("td")[0]
        # print(td0.prettify())
        # print(rows[0].prettify())
        bookies = section_rows[0].find_all("th")
        bookies_list = [bookie.get_text() for bookie in bookies[1:-1]] # empty first and last bookies


        rows = odds_data.find_all("tr")
        
        # odds_dict = {}
        data = []

        print(f'bookies_list: {bookies_list}')
        for row in rows:
            name = self.get_name(row)
            odds_list = self.get_odds(row, bookies_list) 
            # odds_dict[name] = dict(zip(bookies_list, odds_list))
            df_row = {"Fighter": name}
            for bookie, odds in zip(bookies_list, odds_list):
                df_row[bookie] = odds
            data.append(df_row)
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)
        # Optional: set Fighter as index
        df.set_index("Fighter", inplace=True)
        return df
