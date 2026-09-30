from karotte.agents import FakeSource, MessageLoopAgent
from karotte.evaluation_runner import EvaluationRunner
from karotte.schemas.evaluation_run_config import EvaluationRunConfig


def setup_fake_model(
    evaluation_runner: EvaluationRunner, config: EvaluationRunConfig
) -> None:
    try:
        from environment.fake_model import (  # pyright: ignore[reportMissingImports]
            get_messages,
        )
    except (ImportError, ModuleNotFoundError) as e:
        msg = (
            "Cannot use fake model: environment does not provide "
            f"fake_model.py with a get_messages function. Error: {e}"
        )
        raise RuntimeError(msg) from e

    evaluation_runner._agent = MessageLoopAgent(FakeSource(get_messages(config)))  # pyright: ignore[reportPrivateUsage]
