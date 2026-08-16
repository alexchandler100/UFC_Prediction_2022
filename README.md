[![Update Data](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/update-data.yml/badge.svg)](https://github.com/alexchandler100/UFC_Prediction_2022/actions/workflows/update-data.yml)

# UFC Prediction

In this project, we scrape data from ufcstats.com and apply machine learning techniques to this data to make UFC fight predictions (winner and method). The predictor is available [here](https://alexchandler100.github.io/UFC_Prediction_2022/).

## For the Developer

Clone this repository. Open a terminal and cd to repo.

Use Python 3.11, matching GitHub Actions. Install and verify the pinned runtime requirements:

```console
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The dataset is already included in the repository so there is no need to scrape the stats and build the dataframe from scratch, but you can do so from the scripts `1-build_ufc_fights_reported_doubled.py`, and `2-build_ufc_fights_reported_doubled_derived.py` (running one after the other should take about an hour).

The scheduled workflow rebuilds the data and predictions every Wednesday at
9:33 PM America/Chicago. To verify the reliability contracts locally:

```console
python -m unittest discover -s tests -v
python src/validate_data.py --allow-stale
```

To run the update manually from the repository root:

```console
cd src
python update_and_rebuild_model.py
cd ..
python src/validate_data.py
```

This rebuilds the machine learning model to incorporate the updated data. It also scrapes vegas odds for upcoming events from [here](https://fightodds.io), makes predictions for upcoming events, and updates the json files used to populate tables for the website.

The update refuses malformed or stale UFCStats output, writes individual files
atomically, and treats unavailable FightOdds lines as optional enrichment. The
workflow commits only validated generated data and succeeds cleanly when there
is nothing to commit.

If GitHub shows the workflow as disabled, push the workflow change and open
**Actions → Update UFC data → Enable workflow** once. `workflow_dispatch` then
provides a **Run workflow** button for a one-time verification run; normal
weekly updates require no manual command.

To make sure the current website build is working, locally serve `index.html`
from the repository root:

```console
python -m http.server 8000
```

Now go to chrome and type `localhost:8000` into the address bar. This opens the updated version of the website.
