# import ipdb;ipdb.set_trace(context=10) # uncomment to debug
from pathlib import Path
import os
from datetime import datetime, timezone
from hashlib import sha256
import json
import pandas as pd
from git import Repo

# local imports
from data_handler import DataHandler
from data_handler.data_handler import atomic_to_csv
from build_fighter_explorer import (
    build_fighter_explorer,
    load_external_history_inputs,
    load_fighter_history_supplements,
    write_fighter_explorer,
)
from external_mma import load_approved_auxiliary
from fight_predictor import (
    BayesianLogisticChallenger,
    PointInTimeDatasetBuilder,
    TemporalFightPredictor,
    build_outcome_forecast_publication,
    evaluate_outcome_model,
    write_outcome_forecast_publication,
)
from fight_predictor.bayesian_logistic_shadow import (
    BayesianLogisticShadowStore,
    build_shadow_forecasts as build_bayesian_logistic_shadow_forecasts,
)
from market_tracker import EarlyMarketObservationStore
from market_tracker._common import canonical_hash
from upcoming_bet_board import (
    build_upcoming_bet_board,
    build_upcoming_forecast_publication,
    write_upcoming_bet_board,
    write_upcoming_forecast_publication,
)


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
auxiliary_path = (
    Path(__file__).resolve().parent
    / 'content/data/processed/external_mma_auxiliary_doubled.csv'
)
auxiliary_policy_path = (
    Path(__file__).resolve().parent
    / 'content/data/external_mma/model_policy.json'
)
auxiliary_fights = load_approved_auxiliary(auxiliary_path, auxiliary_policy_path)
print('Building stable-ID point-in-time training data')
if auxiliary_fights is not None and not auxiliary_fights.empty:
    print(
        'Replaying state-only external MMA history: '
        f'{auxiliary_fights["fight_url"].nunique():,} bouts'
    )
feature_builder = PointInTimeDatasetBuilder(
    raw_fights, fighter_stats, auxiliary_fights=auxiliary_fights
)
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

print('Training and evaluating the candidate outcome/finish-time model')
outcome_feature_columns = tuple(
    column for column in point_in_time_fights if column.endswith('_diff')
)
outcome_model, outcome_evaluation = evaluate_outcome_model(
    point_in_time_fights, outcome_feature_columns
)
outcome_training_sha256 = sha256(point_in_time_path.read_bytes()).hexdigest()
outcome_evaluation['training_input_sha256'] = outcome_training_sha256
outcome_evaluation['feature_count'] = len(outcome_feature_columns)
outcome_evaluation_path = (
    Path(__file__).resolve().parent
    / 'content/data/external/outcome_model_evaluation.json'
)
outcome_evaluation_path.write_text(
    json.dumps(
        outcome_evaluation,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + '\n',
    encoding='utf-8',
    newline='',
)

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

print('Building and chronologically evaluating the Bayesian winner challenger')
bayesian_challenger = BayesianLogisticChallenger.fit(candidate_predictor)
bayesian_artifact_path = (
    Path(__file__).resolve().parent
    / 'content/data/external/bayesian_winner_challenger.json'
)
bayesian_challenger.save_artifact(bayesian_artifact_path)
# As with the point model, all published posterior summaries must come from
# the just-written portable artifact rather than an in-memory-only fit.
bayesian_challenger = BayesianLogisticChallenger.load_artifact(
    bayesian_artifact_path,
    builder=feature_builder,
    base_artifact=fight_predictor.artifact(),
)

print('Saving results of previous card to prediction_history.json')
# now that the previous card which we made predictions for has happened, we can add the results to the prediction history
# vegas odds is always a week ahead of the prediction history, so we can use it to update the prediction history by comparing vegas_odds and ufc_fights_crap which contains the results from last week
dh.update_prediction_history()

print('Scraping all announced UFC fight cards from ufcstats.com')
print("###############################################################################################################")    
upcoming_cards = dh.get_upcoming_fight_cards()
card_date, card_title, fights_list = dh.update_card_info(upcoming_cards[0])
prediction_history = dh.get('prediction_history', filetype='json')
# Freeze when this exact model/card forecast was issued. Market collectors run
# independently later and must never confuse their retrieval time with the
# timestamp of the stats probability they are evaluating.
forecast_issued_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
forecast_source_revision = _forecast_source_revision()
upcoming_frames = []
for raw_date, event_title, event_fights in upcoming_cards:
    event_date = dh.convert_scraped_date_to_standard_date(raw_date)
    event_frame = fight_predictor.predict_upcoming_fights(
        prediction_history,
        fighter_stats,
        event_fights,
        event_date,
    )
    event_frame = bayesian_challenger.annotate_upcoming_fights(
        event_frame, event_date
    )
    event_frame['forecast issued at'] = forecast_issued_at
    event_frame['forecast source commit'] = forecast_source_revision
    event_frame['event title'] = event_title
    event_frame['bout order'] = list(range(len(event_frame)))
    upcoming_frames.append(event_frame)
all_upcoming_frame = pd.concat(upcoming_frames, ignore_index=True)
predicted_odds_df = upcoming_frames[0].copy(deep=True)
all_upcoming_forecasts = build_upcoming_forecast_publication(
    all_upcoming_frame,
    generated_at_utc=forecast_issued_at,
)
write_upcoming_forecast_publication(all_upcoming_forecasts)
# Keep the latest timestamped prices visible across a model refresh. The board
# is rebuilt against the new announced-fight publication, so removed or changed
# matchups fall out while unchanged matchups retain their last captured prices.
market_root = Path(__file__).resolve().parent / 'content/data/market'
current_opportunities_path = market_root / 'current_opportunities.json'
current_opportunities = None
early_market_observations = EarlyMarketObservationStore(
    market_root / 'early_market_observations.csv',
    market_root / 'early_market_observations.jsonl',
).read()
board_observed_at = forecast_issued_at
board_source = 'model-update-awaiting-market-capture'
if current_opportunities_path.exists():
    current_opportunities = json.loads(
        current_opportunities_path.read_text(encoding='utf-8')
    )
    current_body = dict(current_opportunities)
    current_fingerprint = current_body.pop('publication_sha256', None)
    if current_fingerprint != canonical_hash(current_body):
        raise RuntimeError(
            'Refusing to carry forward an altered current-opportunities file'
        )
    board_observed_at = current_opportunities['observed_at_utc']
    board_source = str(
        current_opportunities.get('source') or 'last-valid-market-capture'
    )
write_upcoming_bet_board(
    build_upcoming_bet_board(
        all_upcoming_forecasts,
        early_market_observations,
        observed_at_utc=board_observed_at,
        source=board_source,
        current_opportunities=current_opportunities,
    )
)
print(
    'Published all announced cards: '
    f'{all_upcoming_forecasts["event_count"]} events / '
    f'{all_upcoming_forecasts["matchup_count"]} matchups'
)

# Lock the newly evaluated Bayesian/logistic blend separately from production.
# A date-only card can prove a forecast is prospective only before the UTC
# event date, so a stale same-day updater is skipped instead of backdating it.
bayesian_shadow_root = (
    Path(__file__).resolve().parent / 'content/data/market'
)
bayesian_shadow_store = BayesianLogisticShadowStore(
    bayesian_shadow_root / 'bayesian_logistic_shadow_forecasts.csv',
    bayesian_shadow_root / 'bayesian_logistic_shadow_forecasts.jsonl',
)
bayesian_shadow_event_day = pd.to_datetime(card_date, errors='raise').date()
bayesian_shadow_issued = datetime.fromisoformat(
    str(predicted_odds_df['forecast issued at'].iloc[0])
)
if bayesian_shadow_issued.date() >= bayesian_shadow_event_day:
    print(
        'Skipping Bayesian/logistic paper shadow: date-only event timing '
        'cannot prove this same-day forecast preceded the event.'
    )
else:
    existing_bayesian_shadows = bayesian_shadow_store.read()
    experiment_path = (
        Path(__file__).resolve().parent
        / 'content/data/model_research/bayesian_logistic_comparison.json'
    )
    new_bayesian_shadows = build_bayesian_logistic_shadow_forecasts(
        candidate_predictor.training_data,
        feature_builder,
        predicted_odds_df,
        forecast_issued_at_utc=predicted_odds_df['forecast issued at'].iloc[0],
        source_commit_sha=predicted_odds_df['forecast source commit'].iloc[0],
        experiment_sha256=sha256(experiment_path.read_bytes()).hexdigest(),
        existing_matchup_ids={
            item.matchup_id for item in existing_bayesian_shadows
        },
    )
    if new_bayesian_shadows:
        append_result = bayesian_shadow_store.append(new_bayesian_shadows)
        print(
            'Locked Bayesian/logistic paper forecasts: '
            f'{len(append_result.added_ids)} new; '
            f'{append_result.total_records} total.'
        )
    else:
        print('Bayesian/logistic paper forecasts were already locked.')
outcome_publication = build_outcome_forecast_publication(
    outcome_model,
    feature_builder,
    predicted_odds_df,
    {
        'date': card_date,
        'title': card_title,
        'event_url': predicted_odds_df['event url'].iloc[0],
        'event_id': predicted_odds_df['event id'].iloc[0],
    },
    selected_c=float(outcome_evaluation['selected_c']),
    training_input_sha256=outcome_training_sha256,
    model_trained_through=str(point_in_time_fights['date'].max()),
    forecast_issued_at_utc=predicted_odds_df['forecast issued at'].iloc[0],
    source_commit_sha=predicted_odds_df['forecast source commit'].iloc[0],
)
write_outcome_forecast_publication(
    Path(__file__).resolve().parent
    / 'content/data/external/outcome_forecasts.json',
    outcome_publication,
)

# Merge available sportsbook odds from the configured market source.
predicted_odds_df_with_vegas_odds = dh.save_fightoddsio_to_vegas_odds_json_and_merge_with_predictions_df(predicted_odds_df)
predicted_odds_df_with_vegas_odds = (
    bayesian_challenger.annotate_best_price_expected_returns(
        predicted_odds_df_with_vegas_odds,
        dh.bookies,
    )
)
dh.update_vegas_odds(predicted_odds_df_with_vegas_odds)
print('Building compact fighter explorer publication')
external_history, external_identity_map = load_external_history_inputs()
external_history_supplements = load_fighter_history_supplements()
write_fighter_explorer(
    build_fighter_explorer(
        raw_fights,
        fighter_stats,
        all_upcoming_frame,
        external_history,
        external_identity_map,
        external_history_supplements,
    )
)
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
        bayesian_artifact = bayesian_challenger.artifact()
        bayesian_walk = (
            bayesian_artifact.get('temporal_evaluation', {})
            .get('walk_forward', {})
            .get('aggregate', {})
        )
        odds_source = getattr(dh.odds_getter, 'last_source', '') or 'unavailable'
        odds_request = getattr(dh.odds_getter, 'last_request_metadata', {})
        summary_lines = [
            '## UFC weekly update',
            '',
            f'- Completed raw fights: {raw_fights["fight_url"].nunique():,}',
            '- Simulation shadow: isolated dependent job (no production effect)',
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
            (
                '- Bayesian challenger: '
                f'`{bayesian_artifact["model_id"]}`; '
                f'{bayesian_walk.get("log_loss", float("nan")):.4f} '
                'posterior-mean walk-forward log loss; execution disabled'
            ),
            f'- Upcoming card: {card_title} ({card_date})',
            (
                '- Candidate outcome model: '
                f'`{outcome_publication["model_id"]}`; '
                f'{outcome_publication["forecast_matchup_count"]}/'
                f'{outcome_publication["matchup_count"]} matchups forecast'
            ),
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
