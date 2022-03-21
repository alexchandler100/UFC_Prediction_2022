# UFC Prediction

In this project, we scrape data from ufcstats.com and apply machine learning techniques to this data to make UFC fight predictions (winner and method).

## Installation

Make sure Python 3 is installed on your computer. Clone this repository. Open a terminal and cd to the location you saved it.

## Usage

If you want to see the most up to date version of the predictor in action, follow these instructions:

1. First we will scrape ufcstats.com and update the dataset. cd (change directory) to the buildingMLModel directory and type
```console
python3 update_data_csvs_and_jsons.py
```
and press enter (this will take awhile to run)
2. Now type
```console
python3 update_and_rebuild_model.py
```
and press enter. This will run quickly. This rebuilds the machine learning model to incorporate the updated data. This will send the updated coefficients to the files theta.json and intercept.json.
3. Now cd to the main directory UFC_Prediction_2022 and type
```console
python3 -m http.server 9089
```
and press enter to serve html via a local host. This should automatically open a browser with the website running.

## Contributing
If you find any feature sets for winner prediction that score higher that the current highest (.637 as of March 9 2022), or for method prediction which score higher than the current highest (.52 as of March 9 2022) let me know!

## Creating a Pull Request
