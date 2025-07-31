[![Update Data](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/update-data.yml/badge.svg)](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/update-data.yml)

# UFC Prediction

In this project, we scrape data from ufcstats.com and apply machine learning techniques to this data to make UFC fight predictions (winner and method). The predictor is available [here](https://alexchandler100.github.io/UFC_Prediction_2022/).

## For the Developer

Clone this repository. Open a terminal and cd to repo.

Make sure Python3.9 is installed on your computer. To install all requirements you can do:

```console
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dataset is already included in the repository so there is no need to scrape the stats and build the dataframe from scratch, but you can do so from the scripts `1-build_ufc_fights_reported_doubled.py`, and `2-build_ufc_fights_reported_doubled_derived.py` (running one after the other should take about an hour).

After every UFC event, the dataset and model will need to be rebuilt. You can do this by running `update_and_rebuild_model.py`. 

This rebuilds the machine learning model to incorporate the updated data. It also scrapes vegas odds for upcoming events from [here](https://fightodds.io), makes predictions for upcoming events, and updates the json files used to populate tables for the website.

3. Now, to make sure the current build is working, locally serve the index.html file in the root directory of the repository. You can do this by running the following commands:

```console
cd ..
python3 -m http.server 8000
```

Now go to chrome and type `localhost:8000` into the address bar. This opens the updated version of the website.