# need to use selenium as the javascript renders the html after the page load
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
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


        # Selenium Manager uses the Chrome/driver installed on the runner and
        # resolves a compatible driver when necessary.  Always quit, including
        # after a timeout or parser exception.
        driver = webdriver.Chrome(options=options)
        try:
            driver.set_page_load_timeout(60)
            driver.get(self.fight_odds_url)
            WebDriverWait(driver, 45).until(
                lambda current_driver: (
                    current_driver.find_elements(By.CSS_SELECTOR, "thead.MuiTableHead-root")
                    and current_driver.find_elements(By.CSS_SELECTOR, "tbody.MuiTableBody-root tr")
                )
            )
            html = driver.page_source
        finally:
            driver.quit()

        # Now parse with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        
        # do we need the date?
        # date = soup.find_all("span", class_="jss2286")[0].get_text()
        # # Looks like "June 27" with no year, so we need to add the current year
        # current_year = pd.Timestamp.now().year
        # date = np.datetime.datetime.strptime(f"{current_year} {date}", "%Y %B %d").strftime("%Y-%m-%d")
        
        odds_sections = soup.find("thead", class_="MuiTableHead-root")
        odds_data = soup.find("tbody", class_="MuiTableBody-root")
        if odds_sections is None or odds_data is None:
            raise RuntimeError(
                "fightodds.io loaded without the expected odds table; its layout may have changed"
            )

        self.df = self.get_fighter_odds_for_card(odds_data, odds_sections)
        # self.df["date"] = date
        return self.df
    
    def get_name(self, row):
        cells = row.find_all("td")
        link = cells[0].find("a") if cells else None
        if link is None:
            raise ValueError("fightodds.io row did not contain a fighter link")
        return link.get_text(strip=True)

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
        if not rows or len(rows) % 2:
            raise ValueError(
                f"Expected a non-empty even number of fightodds.io rows, got {len(rows)}"
            )
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
        if df.empty or df[['fighter name', 'opponent name']].isna().any().any():
            raise ValueError("fightodds.io produced an empty or unnamed matchup")
        if df.duplicated(['fighter name', 'opponent name']).any():
            raise ValueError("fightodds.io produced duplicate matchup rows")
        df["predicted fighter odds"] = np.nan
        df["predicted opponent odds"] = np.nan
        # American moneylines are nonlinear, so averaging the signed odds is
        # invalid.  Convert each complete two-sided book quote to probability,
        # remove that book's vig, then aggregate probabilities.
        consensus_probabilities = [
            self.get_consensus_probability(row, bookies_list)
            for _, row in df.iterrows()
        ]
        df['average bookie probability'] = consensus_probabilities
        df['average bookie odds'] = [
            [
                self.probability_to_odds(probability),
                self.probability_to_odds(1 - probability),
            ]
            if probability is not None else [None, None]
            for probability in consensus_probabilities
        ]
        return df

    @staticmethod
    def parse_american_odds(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        normalized = str(value).strip().replace('−', '-').replace('–', '-')
        if not normalized:
            return None
        if normalized.upper() in {'EVEN', 'EV', 'PK', 'PICK'}:
            return 100
        try:
            numeric = float(normalized)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or not numeric.is_integer():
            return None
        odds = int(numeric)
        return odds if 100 <= abs(odds) <= 100_000 else None

    @staticmethod
    def odds_to_probability(odds):
        return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

    @staticmethod
    def probability_to_odds(probability):
        if not 0 < probability < 1:
            return None
        if probability >= 0.5:
            return -round(100 * probability / (1 - probability))
        return round(100 * (1 - probability) / probability)

    def get_consensus_probability(self, row, bookies_list):
        fighter_probabilities = []
        for bookie in bookies_list:
            fighter_odds = self.parse_american_odds(row.get(f'fighter {bookie}'))
            opponent_odds = self.parse_american_odds(row.get(f'opponent {bookie}'))
            if fighter_odds is None or opponent_odds is None:
                continue
            fighter_implied = self.odds_to_probability(fighter_odds)
            opponent_implied = self.odds_to_probability(opponent_odds)
            overround = fighter_implied + opponent_implied
            if not 0.9 <= overround <= 1.3:
                continue
            fighter_probabilities.append(fighter_implied / overround)

        if not fighter_probabilities:
            return None
        return float(np.mean(fighter_probabilities))

    def get_consensus_odds(self, row, bookies_list):
        fighter_probability = self.get_consensus_probability(row, bookies_list)
        if fighter_probability is None:
            return [None, None]
        return [
            self.probability_to_odds(fighter_probability),
            self.probability_to_odds(1 - fighter_probability),
        ]
