# UFC Prediction

In this project, we scrape data from ufcstats.com and apply machine learning techniques to this data to make UFC fight predictions (winner and method).

## Installation

Make sure Python 3 is installed on your computer. Clone this repository. Open a terminal and cd to the location you saved it. Run the following command to open a jupyter notebook from the directory:

```bash
jupyter notebook
```

## Usage

If you want to see the most up to date version of the predictor in action, follow these instructions:

1. Open the python notebook titled UFC_data_scraping.ipynb and run the first and third cells with shift+enter. This will bring the csv files 'fight_hist.csv' and 'fighter_stats.csv' up to date. You can now close this notebook.
2. Open the python notebook titled Updating_ufc_fights.ipynb and run every cell top to bottom. This brings the file 'ufc_fights.csv' up to date. You can now close this notebook.
3. If you want to see in more detail how the statistics are being calculated, and how the features for learning models are selected, you can look into the files Building_ufc_fighters.ipynb, Building_ufc_fights.ipynb, UFC_LR_Winner_Prediction_Feature_Selection.ipynb, and UFC_Optimizing_Method_Prediction.ipynb but this is not necessary.
4. Open the python notebook titled UFC_Prediction_Model.ipynb and run every cell top to bottom (you don't need to run all of the  example predictions, but they should run quickly anyway. Try some of your own as well. Note: you can enter in any date for either fighter, so for example, you could compare 2012 GSP to 2020 Khabib.

## Contributing
If you find any feature sets for winner prediction that score higher that the current highest (.637 as of March 9 2022), or for method prediction which score higher than the current highest (.52 as of March 9 2022) let me know!

## Creating a Pull Request