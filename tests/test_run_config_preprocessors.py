from unittest.mock import MagicMock, patch

from karotte.run_config_preprocessors import (
    ENTRY_POINT_GROUP,
    apply_run_config_preprocessors,
)
from karotte.schemas.evaluation_run_config import EvaluationRunConfig


def _config() -> EvaluationRunConfig:
    return EvaluationRunConfig(run_id="r", task_id="t", model="m", model_api_key="k")


def _entry_point(name: str, func: object) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = func
    return ep


def test_group_name():
    assert ENTRY_POINT_GROUP == "karotte.run_config_preprocessors"


def test_without_preprocessors_returns_the_config_unchanged():
    config = _config()
    with patch("karotte.run_config_preprocessors.entry_points", return_value=[]):
        assert apply_run_config_preprocessors(config) is config


def test_applies_every_preprocessor_in_name_order():
    def add(suffix: str):
        def preprocess(c: EvaluationRunConfig) -> EvaluationRunConfig:
            return c.model_copy(update={"model_api_key": f"{c.model_api_key}{suffix}"})

        return preprocess

    eps = [_entry_point("b", add("b")), _entry_point("a", add("a"))]
    with patch(
        "karotte.run_config_preprocessors.entry_points", return_value=eps
    ) as found:
        result = apply_run_config_preprocessors(_config())

    found.assert_called_once_with(group=ENTRY_POINT_GROUP)
    assert result.model_api_key == "kab"
