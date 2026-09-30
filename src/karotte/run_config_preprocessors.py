"""Run config transforms that installed packages register as entry points."""

from importlib.metadata import entry_points

from karotte.schemas.evaluation_run_config import EvaluationRunConfig

ENTRY_POINT_GROUP = "karotte.run_config_preprocessors"


def apply_run_config_preprocessors(
    run_config: EvaluationRunConfig,
) -> EvaluationRunConfig:
    """Passes the config through every registered preprocessor, in entry point name order."""
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda ep: ep.name):
        run_config = ep.load()(run_config)
    return run_config
