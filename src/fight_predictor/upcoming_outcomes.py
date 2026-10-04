"""Publish the existing outcome model for every announced future UFC card."""
from pathlib import Path

import pandas as pd

from .outcome_publication import build_outcome_forecast_publication, write_outcome_forecast_publication
from upcoming_bet_board import validate_upcoming_forecast_publication


def publish_upcoming_outcomes(model, builder, announced, directory, *, archive_directory=None, **contract):
    """Use the same fitted model, schedule rules, and issuance for every card.

    Old event files remain readable history. Collection only uses identities in
    the latest announced-card publication, so removed fights cannot be selected.
    """
    announced = validate_upcoming_forecast_publication(announced)
    publications = []
    for event in announced['events']:
        rows = sorted((row for row in announced['matchups'] if row['event_id'] == event['event_id']),
                      key=lambda row: row['bout_order'])
        frame = pd.DataFrame([{
            'fighter id': row['fighter_id'], 'opponent id': row['opponent_id'],
            'fighter name': row['fighter_name'], 'opponent name': row['opponent_name'],
            'division': row['division'], 'model status': row['model_status'],
        } for row in rows])
        publication = build_outcome_forecast_publication(model, builder, frame, {
            'event_id': event['event_id'], 'event_url': event['event_url'],
            'date': event['event_date'], 'title': event['event_title'],
        }, **contract)
        # A UFCStats event ID is a hex identifier; never accept a path from data.
        if not event['event_id'] or any(ch not in '0123456789abcdef' for ch in event['event_id']):
            raise ValueError('invalid UFC event ID for outcome publication')
        write_outcome_forecast_publication(Path(directory) / f"{event['event_id']}.json", publication,
                                          archive_directory=archive_directory)
        publications.append(publication)
    return publications
