"""Licensed external-MMA collection and state-only replay support."""

from .adapters import CanonicalCsvAdapter, KaggleProMmaAdapter
from .identity import propose_ufcstats_crosswalk
from .integration import (
    build_auxiliary_doubled,
    load_approved_auxiliary,
    load_identity_map,
)
from .schema import ExternalBoutObservation, ExternalDataError
from .storage import ExternalMmaStore

__all__ = [
    "CanonicalCsvAdapter",
    "ExternalBoutObservation",
    "ExternalDataError",
    "ExternalMmaStore",
    "KaggleProMmaAdapter",
    "build_auxiliary_doubled",
    "load_approved_auxiliary",
    "load_identity_map",
    "propose_ufcstats_crosswalk",
]
