"""Small, fixed nonlinear feature families for research only.

Compute each expression for each fighter, THEN subtract. Products of two
fighter differences would instead be unchanged by swapping the fighters.
"""
from .point_in_time import PointInTimeDatasetBuilder


GROUPS = {
    "age_curve": ("age_after_30", "age_after_35"),
    "age_experience": ("age_experience",),
    "reach_activity": ("reach_striking", "reach_wrestling"),
    "age_damage": ("age_absorption", "age_knockdowns_absorbed"),
}


def nonlinear_terms(v):
    age = (v["age"] - 29) * v["age_known"]
    reach = (v["reach"] - 70) * v["reach_known"]
    return {
        "age_after_30": max(v["age"] - 30, 0) * v["age_known"],
        "age_after_35": max(v["age"] - 35, 0) * v["age_known"],
        "age_experience": age * v["career_fights_log"],
        "reach_striking": reach * v["recent_3y_sig_landed_per15"],
        "reach_wrestling": reach * v["recent_3y_td_landed_per15"],
        "age_absorption": age * v["career_sig_absorbed_per15"],
        "age_knockdowns_absorbed": age * v["career_knockdowns_absorbed_per15"],
    }


class NonlinearResearchBuilder(PointInTimeDatasetBuilder):
    def _side_features(self, fighter_id, fight_date, division):
        values = super()._side_features(fighter_id, fight_date, division)
        values.update(nonlinear_terms(values))
        return values
