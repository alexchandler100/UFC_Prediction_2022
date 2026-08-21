# import ipdb;ipdb.set_trace(context=10) # uncomment to debug
from pathlib import Path
import os
from datetime import datetime, timezone
import pandas as pd
from git import Repo

# local imports
from data_handler import DataHandler
from data_handler.data_handler import atomic_to_csv
from fight_predictor import PointInTimeDatasetBuilder, TemporalFightPredictor


def _forecast_source_revision() -> str:
    """Return a collector-compatible revision for forecast provenance."""

    workflow_sha = os.environ.get('GITHUB_SHA', '').strip()
    if len(workflow_sha) in {40, 64} and all(
        character in '0123456789abcdefABCDEF' for character in workflow_sha
    ):
        return workflow_sha.lower()
    repo_root = Path(__file__).resolve().parents[1]
    revision = Repo(repo_root).head.commit.hexsha.strip().lower()
    if len(revision) != 40 or any(
        character not in '0123456789abcdef' for character in revision
    ):
        raise RuntimeError('Could not resolve a valid source revision for forecasts')
    return revision

# create a data handler object to access the data stored in csvs and jsons
# has built-in dataframes mirroring the csvs and jsons
# has built-in methods to update the csvs and jsons from ufcstats.com
dh = DataHandler()

# bring csv files up to date and overwrite the old ones
print('scraping new statistics from ufcstats.com')
dh.update_data_csvs_and_jsons()

raw_fights = dh.get('ufc_fights_reported_doubled')
fighter_stats = dh.get('fighter_stats')
print('Building stable-ID point-in-time training data')
feature_builder = PointInTimeDatasetBuilder(raw_fights, fighter_stats)
point_in_time_fights = feature_builder.build()
point_in_time_path = (
    Path(__file__).resolve().parent
    / 'content/data/processed/ufc_fights_point_in_time.csv'
)
atomic_to_csv(point_in_time_fights, point_in_time_path, index=False)
# Treat the persisted table as the canonical training input. This guarantees
# that artifact fingerprints and strict validation see exactly the same
# round-tripped numeric values on every platform.
point_in_time_fights = pd.read_csv(point_in_time_path, low_memory=False)
feature_builder.training_data = point_in_time_fights.copy()

print('Training and temporally evaluating the point-in-time winner model')
candidate_predictor = TemporalFightPredictor(point_in_time_fights, feature_builder, dh)
evaluation = candidate_predictor.train()
walk_forward = evaluation['walk_forward']['aggregate']
holdout = evaluation['calibrated_model']
if (
    walk_forward['log_loss'] >= 0.683
    or walk_forward['accuracy'] <= 0.58
    or holdout['log_loss'] >= 0.70
):
    raise RuntimeError(
        'Refusing to publish a degraded winner model: '
        f'walk-forward accuracy={walk_forward["accuracy"]:.3f}, '
        f'walk-forward log loss={walk_forward["log_loss"]:.3f}, '
        f'holdout log loss={holdout["log_loss"]:.3f}'
    )
artifact_path = (
    Path(__file__).resolve().parent
    / 'content/data/external/winner_model.json'
)
candidate_predictor.save_artifact(artifact_path)
# Predict from the just-written portable artifact, not an unpersisted in-memory
# object.  This makes every published probability exactly reproducible.
fight_predictor = TemporalFightPredictor.load_artifact(
    artifact_path, feature_builder, dh
)

print('Saving results of previous card to prediction_history.json')
# now that the previous card which we made predictions for has happened, we can add the results to the prediction history
# vegas odds is always a week ahead of the prediction history, so we can use it to update the prediction history by comparing vegas_odds and ufc_fights_crap which contains the results from last week
dh.update_prediction_history()

print('Scraping the next UFC fight card from ufcstats.com')
print("###############################################################################################################")    
card_date, card_title, fights_list = dh.update_card_info()
prediction_history = dh.get('prediction_history', filetype='json')
predicted_odds_df = fight_predictor.predict_upcoming_fights(prediction_history, fighter_stats, fights_list, card_date)
# Freeze when this exact model/card forecast was issued. Market collectors run
# independently later and must never confuse their retrieval time with the
# timestamp of the stats probability they are evaluating.
predicted_odds_df['forecast issued at'] = (
    datetime.now(timezone.utc).replace(microsecond=0).isoformat()
)
predicted_odds_df['forecast source commit'] = _forecast_source_revision()
# Merge available sportsbook odds from the configured market source.
predicted_odds_df_with_vegas_odds = dh.save_fightoddsio_to_vegas_odds_json_and_merge_with_predictions_df(predicted_odds_df)
dh.update_vegas_odds(predicted_odds_df_with_vegas_odds)
print('saving scraped fights and predictions to content/data/external/vegas_odds.json')
print("###############################################################################################################")

# Put the important health signals on the Actions run summary. Missing/misaligned
# third-party odds are degraded enrichment, not a reason to discard valid model
# and UFCStats updates.
summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
if summary_path:
    try:
        matched_odds = int(
            predicted_odds_df_with_vegas_odds.get(
                'odds source status', pd.Series(dtype=object)
            ).eq('matched').sum()
        )
        resolved_models = int(
            predicted_odds_df_with_vegas_odds.get(
                'model probability', pd.Series(dtype=float)
            ).pipe(pd.to_numeric, errors='coerce').notna().sum()
        )
        artifact = fight_predictor.artifact()
        odds_source = getattr(dh.odds_getter, 'last_source', '') or 'unavailable'
        odds_request = getattr(dh.odds_getter, 'last_request_metadata', {})
        summary_lines = [
            '## UFC weekly update',
            '',
            f'- Completed raw fights: {raw_fights["fight_url"].nunique():,}',
            f'- Point-in-time W/L rows: {len(point_in_time_fights):,}',
            f'- Model: `{artifact["model_id"]}` through {artifact["data_through"]}',
            (
                f'- Regularization: C={artifact["selected_c"]:g} selected from '
                f'{artifact["regularization_c_grid"]}'
            ),
            (
                '- Walk-forward: '
                f'{walk_forward["accuracy"]:.1%} accuracy, '
                f'{walk_forward["log_loss"]:.4f} log loss'
            ),
            f'- Upcoming card: {card_title} ({card_date})',
            (
                f'- Forecast coverage: {resolved_models}/'
                f'{len(predicted_odds_df_with_vegas_odds)}'
            ),
            (
                f'- Sportsbook coverage: {matched_odds}/'
                f'{len(predicted_odds_df_with_vegas_odds)}'
            ),
            f'- Sportsbook source: `{odds_source}`',
            (
                '- API credits remaining: '
                f'{odds_request.get("requests_remaining")}'
                if odds_source == 'the-odds-api.com'
                else '- API credits remaining: not applicable'
            ),
            '- Betting: disabled pending timestamped market-relative validation',
            '',
        ]
        if matched_odds < len(predicted_odds_df_with_vegas_odds):
            summary_lines.extend(
                [
                    '> [!WARNING]',
                    '> The sportsbook feed did not match every UFCStats matchup; unmatched rows use the stats model and contain no bet recommendation.',
                    '',
                ]
            )
        with open(summary_path, 'a', encoding='utf-8') as summary_file:
            summary_file.write('\n'.join(summary_lines))
    except Exception as error:
        print(f'Could not write optional GitHub Actions summary: {error}')
