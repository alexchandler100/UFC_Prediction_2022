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

The dataset is already included in the repository so there is no need to scrape the stats and build the dataframe from scratch, but you can do so from the jupyter notebooks in src/content/notebooks/ if desired. In particular, you can run all cells in UFC_data_scraping.ipynb to scrape the entire dataset from the source ufcstats.com, and run cells in Building_ufc_fights.ipynb to compile the dataframe we use to build our ML model (though this would take weeks on a laptop). To scrape and stylize fighter pictures used on the website, you can run UFC_picture_scraping.ipynb and Pytorch Convolutional Neural Network Style Transfer.ipynb. The rest of the notebooks are dedicated to building and testing machine learning models for fight and method prediction.

After every UFC event, the dataset and model will need to be rebuilt. The Cron job run via github actions will automatically do the following steps every Wednesday at 8pm MST. So to test changes on the upcoming fight card, do the following between Sunday and Wednesday:

1. Open a terminal in the UFC_Prediction_2022 directory and run

```console
cd src
```

2. Now, run the following command to update the dataset and rebuild the model:
```console
python update_and_rebuild_model.py
```

This will take about 20 minutes to run through. This rebuilds the machine learning model to incorporate the updated data. This will send the updated coefficients to the files theta.json and intercept.json which are used on the website to implement the current build. It also scrapes vegas odds for upcoming events from [here](https://fightodds.io), makes predictions for upcoming events, and updates the json files used to populate tables for the website.

3. Now, to make sure the current build is working, locally serve the index.html file in the root directory of the repository. You can do this by running the following commands:

```console
cd ..
python3 -m http.server 8000
```

Now go to chrome and type `localhost:8000` into the address bar. This opens the updated version of the website.