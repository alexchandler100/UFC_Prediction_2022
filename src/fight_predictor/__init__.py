from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .fight_predictor import FightPredictor
from .point_in_time import (
    MODEL_VERSION,
    PointInTimeDatasetBuilder,
    TemporalFightPredictor,
)
from .style_matchup import StyleMatchupDatasetBuilder
from .stance_matchup import StanceMatchupDatasetBuilder
from .outcome_model import (
    CompetingRiskPrediction,
    DiscreteTimeOutcomeModel,
    evaluate_outcome_model,
)
from .outcome_publication import (
    OUTCOME_MODEL_VERSION,
    build_outcome_forecast_publication,
    scheduled_rounds_for_upcoming,
    validate_outcome_forecast_publication,
    write_outcome_forecast_publication,
)
from .bayesian import (
    BAYESIAN_CREDIBLE_LEVEL,
    BAYESIAN_MINIMUM_MEAN_EV,
    BAYESIAN_MINIMUM_PROBABILITY_POSITIVE_EV,
    BAYESIAN_MODEL_VERSION,
    BayesianLogisticChallenger,
    LogitNormalPrediction,
    american_to_decimal,
    laplace_covariance,
)


def __getattr__(name: str) -> Any:
    """Load the legacy predictor only when callers explicitly request it.

    Importing a focused research or validation module must not pull in the
    scraper/round-stat stack through ``fight_predictor.FightPredictor``.
    ``from fight_predictor import FightPredictor`` remains compatible.
    """

    if name == "FightPredictor":
        from .fight_predictor import FightPredictor

        globals()[name] = FightPredictor
        return FightPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
