from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from karotte import EvaluationRunConfig, Step, Task, create_task
from karotte.judges.regex_judge import RegexJudge
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig
from karotte.task_factory import (
    StepConfig,
    _make_step_class,  # pyright: ignore[reportPrivateUsage]
)
from tests.conftest import register_hardware_plugins


def _make_config(**overrides: Any) -> EvaluationRunConfig:
    defaults: dict[str, Any] = {
        "run_id": "r1",
        "task_id": "t1",
        "model": "test_model",
        "model_api_key": "test_key",
        "mcp_server_config": HttpMcpServerConfig(host="0.0.0.0", port=8080),
    }
    return EvaluationRunConfig(**(defaults | overrides))


@pytest.fixture
def config() -> EvaluationRunConfig:
    return _make_config()


@pytest.fixture
def judge() -> RegexJudge:
    return RegexJudge([])


# ---------------------------------------------------------------------------
# StepConfig
# ---------------------------------------------------------------------------


class TestStepConfig:
    def test_static_values(self, judge: RegexJudge) -> None:
        sc = StepConfig(instructions="do X", judge=judge)
        assert sc.instructions == "do X"
        assert sc.judge is judge
        assert sc.post_hook is None

    def test_callable_values(self) -> None:
        def instr_fn(cfg: EvaluationRunConfig) -> str:
            return f"hint: {cfg.use_hints}"

        def judge_fn(_cfg: EvaluationRunConfig) -> RegexJudge:
            return RegexJudge([])

        sc = StepConfig(instructions=instr_fn, judge=judge_fn)
        assert callable(sc.instructions)
        assert callable(sc.judge)

    def test_post_hook_default_none(self, judge: RegexJudge) -> None:
        sc = StepConfig(instructions="x", judge=judge)
        assert sc.post_hook is None

    def test_post_hook_set(self, judge: RegexJudge) -> None:
        def hook(_cfg: EvaluationRunConfig) -> None:
            return None

        sc = StepConfig(instructions="x", judge=judge, post_hook=hook)
        assert sc.post_hook is hook

    def test_pre_scoring_hook_default_none(self, judge: RegexJudge) -> None:
        sc = StepConfig(instructions="x", judge=judge)
        assert sc.pre_scoring_hook is None

    def test_pre_scoring_hook_set(self, judge: RegexJudge) -> None:
        def hook(_cfg: EvaluationRunConfig) -> None:
            return None

        sc = StepConfig(instructions="x", judge=judge, pre_scoring_hook=hook)
        assert sc.pre_scoring_hook is hook


# ---------------------------------------------------------------------------
# _make_step_class
# ---------------------------------------------------------------------------


class TestMakeStepClass:
    def test_static_instructions_and_judge(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        sc = StepConfig(instructions="static text", judge=judge)
        cls = _make_step_class(0, sc)

        assert issubclass(cls, Step)
        assert cls.__name__ == "step_1"

        instance = cls(config=config)
        assert instance.instructions == "static text"
        assert instance.judge is judge

    def test_callable_instructions(self, config: EvaluationRunConfig) -> None:
        judge = RegexJudge([])
        sc = StepConfig(
            instructions=lambda cfg: f"hints={cfg.use_hints}",
            judge=judge,
        )
        instance = _make_step_class(0, sc)(config=config)
        assert instance.instructions == f"hints={config.use_hints}"

    def test_callable_judge(self, config: EvaluationRunConfig) -> None:
        expected = RegexJudge([])
        sc = StepConfig(
            instructions="x",
            judge=lambda cfg: expected,
        )
        instance = _make_step_class(0, sc)(config=config)
        assert instance.judge is expected

    def test_index_naming(self, judge: RegexJudge) -> None:
        """Step class names are 1-indexed: step_1, step_2, …"""
        for i, expected_name in [(0, "step_1"), (4, "step_5"), (99, "step_100")]:
            cls = _make_step_class(i, StepConfig(instructions="x", judge=judge))
            assert cls.__name__ == expected_name

    def test_post_hook_called_with_config(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        hook = MagicMock()
        sc = StepConfig(instructions="x", judge=judge, post_hook=hook)
        instance = _make_step_class(0, sc)(config=config)

        instance.post_hook()
        hook.assert_called_once_with(config)

    def test_no_post_hook_uses_base_default(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        """Without a post_hook, the base Step.post_hook (returns None) is used."""
        sc = StepConfig(instructions="x", judge=judge)
        instance = _make_step_class(0, sc)(config=config)
        assert instance.post_hook() is None

    def test_pre_scoring_hook_called_with_config(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        hook = MagicMock()
        sc = StepConfig(instructions="x", judge=judge, pre_scoring_hook=hook)
        instance = _make_step_class(0, sc)(config=config)

        instance.pre_scoring_hook()
        hook.assert_called_once_with(config)

    def test_no_pre_scoring_hook_uses_base_default(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        """Without a pre_scoring_hook, the base Step.pre_scoring_hook (returns None) is used."""
        sc = StepConfig(instructions="x", judge=judge)
        instance = _make_step_class(0, sc)(config=config)
        assert instance.pre_scoring_hook() is None

    def test_callable_receives_different_configs(self, judge: RegexJudge) -> None:
        """A callable instruction should reflect each config it receives."""
        sc = StepConfig(
            instructions=lambda cfg: f"model={cfg.model}",
            judge=judge,
        )
        cls = _make_step_class(0, sc)

        cfg_a = _make_config(model="model_a")
        cfg_b = _make_config(model="model_b")

        assert cls(config=cfg_a).instructions == "model=model_a"
        assert cls(config=cfg_b).instructions == "model=model_b"


# ---------------------------------------------------------------------------
# create_task – basic
# ---------------------------------------------------------------------------


class TestCreateTaskBasic:
    def test_returns_task_subclass(self, judge: RegexJudge) -> None:
        cls = create_task(
            id="my-task",
            tools=["bash"],
            steps=[StepConfig(instructions="do it", judge=judge)],
            system_prompt=None,
        )
        assert issubclass(cls, Task)

    def test_class_name_replaces_hyphens(self, judge: RegexJudge) -> None:
        cls = create_task(
            id="my-cool-task",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=None,
        )
        assert cls.__name__ == "my_cool_task"

    def test_id_attribute(self, config: EvaluationRunConfig, judge: RegexJudge) -> None:
        cls = create_task(
            id="task-1",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=None,
        )
        assert cls.id == "task-1"
        # Also accessible from instance
        assert cls(config=config).id == "task-1"

    def test_tools_property(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        cls = create_task(
            id="t",
            tools=["bash", "view_lines_in_file"],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=None,
        )
        assert cls(config=config).tools == ["bash", "view_lines_in_file"]

    def test_empty_tools(self, config: EvaluationRunConfig, judge: RegexJudge) -> None:
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=None,
        )
        assert cls(config=config).tools == []


# ---------------------------------------------------------------------------
# create_task – required_hardware
# ---------------------------------------------------------------------------


class TestCreateTaskScoringTimeLimit:
    def test_tasks_do_not_declare_one_by_default(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=None,
        )
        assert cls(config=config).scoring_time_limit_seconds is None

    def test_a_declared_limit_is_returned(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            scoring_time_limit_seconds=10800.0,
            system_prompt=None,
        )
        assert cls(config=config).scoring_time_limit_seconds == 10800.0


class TestCreateTaskRequiredHardware:
    def test_no_hardware_without_a_plugin(
        self,
        config: EvaluationRunConfig,
        judge: RegexJudge,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        register_hardware_plugins(monkeypatch)
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=None,
        )
        assert cls(config=config).required_hardware is None

    def test_the_plugin_default_hardware(
        self,
        config: EvaluationRunConfig,
        judge: RegexJudge,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        register_hardware_plugins(monkeypatch, default={"a": "small"})
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=None,
        )
        assert cls(config=config).required_hardware == "small"

    def test_custom_hardware(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            required_hardware="big",
            system_prompt=None,
        )
        assert cls(config=config).required_hardware == "big"


# ---------------------------------------------------------------------------
# create_task – steps
# ---------------------------------------------------------------------------


class TestCreateTaskSteps:
    def test_single_step(self, config: EvaluationRunConfig, judge: RegexJudge) -> None:
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="step one", judge=judge)],
            system_prompt=None,
        )
        task = cls(config=config)
        steps = list(task.steps)
        assert len(steps) == 1
        assert steps[0].instructions == "step one"

    def test_multiple_steps_ordering(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        step_configs = [
            StepConfig(instructions=f"step {i}", judge=judge) for i in range(5)
        ]
        cls = create_task(id="t", tools=[], steps=step_configs, system_prompt=None)
        task = cls(config=config)
        instructions = [s.instructions for s in task.steps]
        assert instructions == [f"step {i}" for i in range(5)]

    def test_steps_are_fresh_instances_each_access(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        """Each access to task.steps should create new Step instances."""
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=None,
        )
        task = cls(config=config)
        steps_a = list(task.steps)
        steps_b = list(task.steps)
        assert steps_a[0] is not steps_b[0]

    def test_steps_receive_task_config(self, judge: RegexJudge) -> None:
        cfg = _make_config(use_hints=False)
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=None,
        )
        task = cls(config=cfg)
        step = list(task.steps)[0]
        assert step.config is cfg

    def test_callable_step_instructions_receive_config(self) -> None:
        judge = RegexJudge([])
        cfg = _make_config(use_hints=True)
        cls = create_task(
            id="t",
            tools=[],
            steps=[
                StepConfig(
                    instructions=lambda c: f"hints={c.use_hints}",
                    judge=judge,
                ),
            ],
            system_prompt=None,
        )
        task = cls(config=cfg)
        assert list(task.steps)[0].instructions == "hints=True"

    def test_callable_step_judge_receives_config(self) -> None:
        judge_with_hints = RegexJudge([])
        judge_without_hints = RegexJudge([])

        def pick_judge(cfg: EvaluationRunConfig) -> RegexJudge:
            return judge_with_hints if cfg.use_hints else judge_without_hints

        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=pick_judge)],
            system_prompt=None,
        )

        task_hints = cls(config=_make_config(use_hints=True))
        assert list(task_hints.steps)[0].judge is judge_with_hints

        task_no_hints = cls(config=_make_config(use_hints=False))
        assert list(task_no_hints.steps)[0].judge is judge_without_hints

    def test_step_post_hook_from_factory(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        hook = MagicMock()
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge, post_hook=hook)],
            system_prompt=None,
        )
        step = list(cls(config=config).steps)[0]
        step.post_hook()
        hook.assert_called_once_with(config)

    def test_step_pre_scoring_hook_from_factory(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        hook = MagicMock()
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge, pre_scoring_hook=hook)],
            system_prompt=None,
        )
        step = list(cls(config=config).steps)[0]
        step.pre_scoring_hook()
        hook.assert_called_once_with(config)


# ---------------------------------------------------------------------------
# create_task – configure_tools
# ---------------------------------------------------------------------------


class TestCreateTaskConfigureTools:
    def test_configure_tools_not_set(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        """Without configure_tools, the base Task.configure_tools (no-op) is used."""
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=None,
        )
        # Should not raise
        cls(config=config).configure_tools()

    def test_configure_tools_called(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        callback = MagicMock()
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            configure_tools=callback,
            system_prompt=None,
        )
        cls(config=config).configure_tools()
        callback.assert_called_once_with()

    def test_configure_tools_receives_no_args(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        """configure_tools callback signature is () -> None, no config passed."""
        received_args: list[tuple[Any, ...]] = []

        def spy(*args: Any) -> None:
            received_args.append(args)

        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            configure_tools=spy,
            system_prompt=None,
        )
        cls(config=config).configure_tools()
        assert received_args == [()]


# ---------------------------------------------------------------------------
# create_task – pre_hook
# ---------------------------------------------------------------------------


class TestCreateTaskPreHook:
    def test_pre_hook_not_set(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        """Without pre_hook, the base Task.pre_hook (returns {}) is used."""
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=None,
        )
        assert cls(config=config).pre_hook() == {}

    def test_pre_hook_called_with_config(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        callback = MagicMock(return_value={"key": "value"})
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            pre_hook=callback,
            system_prompt=None,
        )
        result = cls(config=config).pre_hook()
        callback.assert_called_once_with(config)
        assert result == {"key": "value"}

    def test_pre_hook_receives_correct_config(self, judge: RegexJudge) -> None:
        received: list[EvaluationRunConfig] = []

        def capture(cfg: EvaluationRunConfig) -> dict[str, Any]:
            received.append(cfg)
            return {}

        cfg = _make_config(model="captured_model")
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            pre_hook=capture,
            system_prompt=None,
        )
        cls(config=cfg).pre_hook()
        assert received == [cfg]
        assert received[0].model == "captured_model"


# ---------------------------------------------------------------------------
# create_task – system_prompt
# ---------------------------------------------------------------------------


class TestCreateTaskSystemPrompt:
    def test_system_prompt_is_required(self, judge: RegexJudge) -> None:
        with pytest.raises(TypeError, match="system_prompt"):
            create_task(  # pyright: ignore[reportCallIssue]
                id="t",
                tools=[],
                steps=[StepConfig(instructions="x", judge=judge)],
            )

    def test_static_system_prompt(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt="You are a custom agent.",
        )
        assert cls(config=config).system_prompt == "You are a custom agent."

    def test_none_system_prompt(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        """Passing None explicitly means 'no system prompt'."""
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=None,
        )
        assert cls(config=config).system_prompt is None

    def test_callable_system_prompt_receives_config(self, judge: RegexJudge) -> None:
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=lambda cfg: f"model={cfg.model}",
        )
        cfg = _make_config(model="claude-opus-4-8")
        assert cls(config=cfg).system_prompt == "model=claude-opus-4-8"

    def test_callable_system_prompt_can_return_none(
        self, config: EvaluationRunConfig, judge: RegexJudge
    ) -> None:
        cls = create_task(
            id="t",
            tools=[],
            steps=[StepConfig(instructions="x", judge=judge)],
            system_prompt=lambda _cfg: None,
        )
        assert cls(config=config).system_prompt is None


# ---------------------------------------------------------------------------
# create_task – multiple instances / isolation
# ---------------------------------------------------------------------------


class TestCreateTaskInstances:
    def test_two_instances_with_different_configs(self, judge: RegexJudge) -> None:
        cls = create_task(
            id="t",
            tools=[],
            steps=[
                StepConfig(
                    instructions=lambda c: f"model={c.model}",
                    judge=judge,
                )
            ],
            system_prompt=None,
        )
        a = cls(config=_make_config(model="model_a"))
        b = cls(config=_make_config(model="model_b"))
        assert list(a.steps)[0].instructions == "model=model_a"
        assert list(b.steps)[0].instructions == "model=model_b"

    def test_two_different_tasks_are_independent(
        self, config: EvaluationRunConfig
    ) -> None:
        judge_a = RegexJudge([])
        judge_b = RegexJudge([])

        cls_a = create_task(
            id="task-a",
            tools=["bash"],
            steps=[StepConfig(instructions="A", judge=judge_a)],
            required_hardware="small",
            system_prompt=None,
        )
        cls_b = create_task(
            id="task-b",
            tools=["view_lines_in_file"],
            steps=[StepConfig(instructions="B", judge=judge_b)],
            required_hardware="medium",
            system_prompt=None,
        )

        a = cls_a(config=config)
        b = cls_b(config=config)

        assert a.id == "task-a"
        assert b.id == "task-b"
        assert a.tools == ["bash"]
        assert b.tools == ["view_lines_in_file"]
        assert a.required_hardware == "small"
        assert b.required_hardware == "medium"
        assert list(a.steps)[0].instructions == "A"
        assert list(b.steps)[0].instructions == "B"


# ---------------------------------------------------------------------------
# create_task – closure correctness in loops
# ---------------------------------------------------------------------------


class TestClosureCapture:
    def test_steps_closure_captures_correctly_in_loop(
        self, config: EvaluationRunConfig
    ) -> None:
        """Verify that building multiple StepConfigs in a loop doesn't
        cause all of them to share the last loop variable."""
        judges = [RegexJudge([]) for _ in range(3)]
        step_configs = [
            StepConfig(instructions=f"step {i}", judge=judges[i]) for i in range(3)
        ]
        cls = create_task(id="t", tools=[], steps=step_configs, system_prompt=None)
        steps = list(cls(config=config).steps)

        for i in range(3):
            assert steps[i].instructions == f"step {i}"
            assert steps[i].judge is judges[i]

    def test_callable_closure_captures_correctly_in_loop(
        self, config: EvaluationRunConfig
    ) -> None:
        """Same test but with callable instructions to exercise the lambda
        closure capture with default args."""
        step_configs = [
            StepConfig(
                instructions=lambda cfg, idx=i: f"step {idx}",
                judge=RegexJudge([]),
            )
            for i in range(3)
        ]
        cls = create_task(id="t", tools=[], steps=step_configs, system_prompt=None)
        instructions = [s.instructions for s in cls(config=config).steps]
        assert instructions == ["step 0", "step 1", "step 2"]


# ---------------------------------------------------------------------------
# create_task – mixed static & callable steps
# ---------------------------------------------------------------------------


class TestMixedSteps:
    def test_mix_of_static_and_callable(self, config: EvaluationRunConfig) -> None:
        static_judge = RegexJudge([])
        dynamic_judge = RegexJudge([])

        cls = create_task(
            id="mixed",
            tools=["bash"],
            steps=[
                StepConfig(instructions="static instr", judge=static_judge),
                StepConfig(
                    instructions=lambda cfg: f"dynamic hints={cfg.use_hints}",
                    judge=lambda cfg: dynamic_judge,
                ),
            ],
            system_prompt=None,
        )
        task = cls(config=config)
        steps = list(task.steps)

        assert steps[0].instructions == "static instr"
        assert steps[0].judge is static_judge

        assert steps[1].instructions == f"dynamic hints={config.use_hints}"
        assert steps[1].judge is dynamic_judge
