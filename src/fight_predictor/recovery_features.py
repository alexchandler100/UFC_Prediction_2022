"""Predeclared inactivity/recovery research features; production is unchanged."""
import math

from .point_in_time import FighterState, PointInTimeDatasetBuilder


class RecoveryDatasetBuilder(PointInTimeDatasetBuilder):
    def _side_features(self, fighter_id, fight_date, division):
        values=super()._side_features(fighter_id,fight_date,division)
        records=[r for r in self.states.get(fighter_id,FighterState()).records if r.date <= fight_date]
        days=min(max((fight_date-records[-1].date).days,0),1825) if records else 730
        ko_losses=[r for r in records if r.result == 0 and r.method.upper() in ('KO/TKO','KO','TKO')]
        ko_days=min(max((fight_date-ko_losses[-1].date).days,0),1825) if ko_losses else 1825
        values.update({'layoff_under_90_days':float(bool(records) and days < 90),
            'layoff_180_to_365_days':float(bool(records) and 180 <= days < 365),
            'layoff_over_365_days':float(bool(records) and days >= 365),
            'age_times_log_layoff':(values['age']-29)*math.log1p(days)/10,
            'prior_ko_loss':float(bool(ko_losses)),
            'log_days_since_ko_loss':math.log1p(ko_days),
            'ko_loss_within_180_days':float(bool(ko_losses) and ko_days <= 180),
            'fights_last_180_days_log':math.log1p(sum((fight_date-r.date).days <= 180 for r in records))})
        return values
