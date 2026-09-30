"""Tests for the task discovery functionality in environment/__init__.py."""

from pathlib import Path
from types import ModuleType
from typing import final, override
from unittest.mock import MagicMock, patch

import pytest
from karotte import EvaluationRunConfig, Step, Task
from karotte.judges import AlwaysPassJudge

import environment

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> EvaluationRunConfig:
    return EvaluationRunConfig(
        run_id="test_run",
        task_id="test-task",
        model="test_model",
        model_api_key="test_key",
    )


@pytest.fixture
def task_package_dir(tmp_path: Path) -> Path:
    tasks_dir = tmp_path / "environment" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "__init__.py").write_text("")
    return tasks_dir


# =============================================================================
# Helper Classes
# =============================================================================


class DummyStep(Step):
    @property
    @override
    def instructions(self) -> str:
        return "Do something."

    @property
    @override
    def judge(self):
        return AlwaysPassJudge()


@final
class ValidTask(Task):
    @property
    @override
    def system_prompt(self) -> str | None:
        return None

    id = "valid-task"

    @property
    @override
    def steps(self):
        return [DummyStep(config=self.config)]

    @property
    @override
    def tools(self):
        return ["bash"]


@final
class AnotherValidTask(Task):
    @property
    @override
    def system_prompt(self) -> str | None:
        return None

    id = "another-valid-task"

    @property
    @override
    def steps(self):
        return [DummyStep(config=self.config)]

    @property
    @override
    def tools(self):
        return ["bash", "view_lines_in_file"]


@final
class EmptyIdTask(Task):
    @property
    @override
    def system_prompt(self) -> str | None:
        return None

    id = ""

    @property
    @override
    def steps(self):
        return [DummyStep(config=self.config)]

    @property
    @override
    def tools(self):
        return ["bash"]


def _make_module(attrs: dict) -> MagicMock:
    module = MagicMock()
    module.__name__ = "test_module"
    attrs["__name__"] = "test_module"
    module.configure_mock(**attrs)
    return module


def _make_task_module(name: str, attrs: dict) -> ModuleType:
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# =============================================================================
# Tests: _get_tasks_from_module
# =============================================================================


class TestGetTasksFromModule:
    def test_extracts_single_task(self):
        module = _make_module({"ValidTask": ValidTask, "Task": Task})

        with patch.object(
            type(module),
            "__dir__",
            return_value=["ValidTask", "Task", "__name__"],
        ):
            tasks = environment._get_tasks_from_module(module)

        assert tasks == [ValidTask]

    def test_extracts_multiple_tasks(self):
        module = _make_module(
            {"ValidTask": ValidTask, "AnotherValidTask": AnotherValidTask, "Task": Task}
        )

        with patch.object(
            type(module),
            "__dir__",
            return_value=["ValidTask", "AnotherValidTask", "Task", "__name__"],
        ):
            tasks = environment._get_tasks_from_module(module)

        assert {t.id for t in tasks} == {"valid-task", "another-valid-task"}

    def test_excludes_task_base_class(self):
        module = _make_module({"Task": Task, "ValidTask": ValidTask})

        with patch.object(
            type(module), "__dir__", return_value=["Task", "ValidTask", "__name__"]
        ):
            tasks = environment._get_tasks_from_module(module)

        assert tasks == [ValidTask]

    def test_ignores_non_task_classes(self):
        class NotATask:
            pass

        module = _make_module(
            {
                "ValidTask": ValidTask,
                "NotATask": NotATask,
                "some_function": lambda: None,
                "some_string": "hello",
                "some_number": 42,
            }
        )

        with patch.object(
            type(module),
            "__dir__",
            return_value=[
                "ValidTask",
                "NotATask",
                "some_function",
                "some_string",
                "some_number",
                "__name__",
            ],
        ):
            tasks = environment._get_tasks_from_module(module)

        assert tasks == [ValidTask]

    def test_returns_empty_for_module_without_tasks(self):
        class NotATask:
            pass

        module = _make_module({"NotATask": NotATask, "some_function": lambda: None})

        with patch.object(
            type(module),
            "__dir__",
            return_value=["NotATask", "some_function", "__name__"],
        ):
            tasks = environment._get_tasks_from_module(module)

        assert tasks == []

    def test_skips_empty_task_id(self):
        module = _make_module({"EmptyIdTask": EmptyIdTask})

        with patch.object(
            type(module), "__dir__", return_value=["EmptyIdTask", "__name__"]
        ):
            tasks = environment._get_tasks_from_module(module)

        assert tasks == []

    def test_ignores_step_classes(self):
        module = _make_module(
            {"ValidTask": ValidTask, "DummyStep": DummyStep, "Step": Step}
        )

        with patch.object(
            type(module),
            "__dir__",
            return_value=["ValidTask", "DummyStep", "Step", "__name__"],
        ):
            tasks = environment._get_tasks_from_module(module)

        assert tasks == [ValidTask]


# =============================================================================
# Tests: INCLUDE_TASKS / EXCLUDE_TASKS filtering
# =============================================================================


class TestTaskFiltering:
    @pytest.fixture(autouse=True)
    def _reset_filters(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(environment, "INCLUDE_TASKS", [])
        monkeypatch.setattr(environment, "EXCLUDE_TASKS", [])

    def test_empty_lists_returns_all_tasks(self):
        module = _make_module(
            {"ValidTask": ValidTask, "AnotherValidTask": AnotherValidTask}
        )

        with patch.object(
            type(module),
            "__dir__",
            return_value=["ValidTask", "AnotherValidTask", "__name__"],
        ):
            tasks = environment._get_tasks_from_module(module)

        assert {t.id for t in tasks} == {"valid-task", "another-valid-task"}

    def test_exclude_removes_matching_task(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(environment, "EXCLUDE_TASKS", ["valid-task"])
        module = _make_module(
            {"ValidTask": ValidTask, "AnotherValidTask": AnotherValidTask}
        )

        with patch.object(
            type(module),
            "__dir__",
            return_value=["ValidTask", "AnotherValidTask", "__name__"],
        ):
            tasks = environment._get_tasks_from_module(module)

        assert tasks == [AnotherValidTask]

    def test_exclude_all_tasks(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            environment, "EXCLUDE_TASKS", ["valid-task", "another-valid-task"]
        )
        module = _make_module(
            {"ValidTask": ValidTask, "AnotherValidTask": AnotherValidTask}
        )

        with patch.object(
            type(module),
            "__dir__",
            return_value=["ValidTask", "AnotherValidTask", "__name__"],
        ):
            tasks = environment._get_tasks_from_module(module)

        assert tasks == []

    def test_exclude_nonexistent_id_has_no_effect(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(environment, "EXCLUDE_TASKS", ["nonexistent-task"])
        module = _make_module({"ValidTask": ValidTask})

        with patch.object(
            type(module), "__dir__", return_value=["ValidTask", "__name__"]
        ):
            tasks = environment._get_tasks_from_module(module)

        assert tasks == [ValidTask]

    def test_include_selects_only_matching_tasks(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(environment, "INCLUDE_TASKS", ["valid-task"])
        module = _make_module(
            {"ValidTask": ValidTask, "AnotherValidTask": AnotherValidTask}
        )

        with patch.object(
            type(module),
            "__dir__",
            return_value=["ValidTask", "AnotherValidTask", "__name__"],
        ):
            tasks = environment._get_tasks_from_module(module)

        assert tasks == [ValidTask]

    def test_include_nonexistent_id_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(environment, "INCLUDE_TASKS", ["nonexistent-task"])
        module = _make_module({"ValidTask": ValidTask})

        with patch.object(
            type(module), "__dir__", return_value=["ValidTask", "__name__"]
        ):
            tasks = environment._get_tasks_from_module(module)

        assert tasks == []

    def test_exclude_takes_priority_over_include(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            environment, "INCLUDE_TASKS", ["valid-task", "another-valid-task"]
        )
        monkeypatch.setattr(environment, "EXCLUDE_TASKS", ["valid-task"])
        module = _make_module(
            {"ValidTask": ValidTask, "AnotherValidTask": AnotherValidTask}
        )

        with patch.object(
            type(module),
            "__dir__",
            return_value=["ValidTask", "AnotherValidTask", "__name__"],
        ):
            tasks = environment._get_tasks_from_module(module)

        assert tasks == [AnotherValidTask]

    def test_include_multiple_tasks(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            environment, "INCLUDE_TASKS", ["valid-task", "another-valid-task"]
        )
        module = _make_module(
            {"ValidTask": ValidTask, "AnotherValidTask": AnotherValidTask}
        )

        with patch.object(
            type(module),
            "__dir__",
            return_value=["ValidTask", "AnotherValidTask", "__name__"],
        ):
            tasks = environment._get_tasks_from_module(module)

        assert {t.id for t in tasks} == {"valid-task", "another-valid-task"}


# =============================================================================
# Tests: Directory filtering logic
# =============================================================================


class TestDirectoryFiltering:
    def test_skips_files(self, task_package_dir: Path):
        (task_package_dir / "not_a_directory.py").write_text("# some file")

        candidates = list(task_package_dir.glob("*"))
        filtered = [c for c in candidates if c.is_dir() and not c.name.startswith("_")]

        assert len(filtered) == 0

    def test_skips_underscore_prefixed_directories(self, task_package_dir: Path):
        (task_package_dir / "_template").mkdir()
        (task_package_dir / "_private").mkdir()
        (task_package_dir / "__pycache__").mkdir()
        (task_package_dir / "valid_task").mkdir()

        candidates = list(task_package_dir.glob("*"))
        filtered = [c for c in candidates if c.is_dir() and not c.name.startswith("_")]

        assert len(filtered) == 1
        assert filtered[0].name == "valid_task"

    def test_includes_valid_task_directories(self, task_package_dir: Path):
        (task_package_dir / "example_task").mkdir()
        (task_package_dir / "another_task").mkdir()
        (task_package_dir / "my_cool_task").mkdir()

        candidates = list(task_package_dir.glob("*"))
        filtered = [c for c in candidates if c.is_dir() and not c.name.startswith("_")]

        assert len(filtered) == 3
        names = {c.name for c in filtered}
        assert names == {"example_task", "another_task", "my_cool_task"}

    def test_mixed_content_filtering(self, task_package_dir: Path):
        (task_package_dir / "valid_task_1").mkdir()
        (task_package_dir / "valid_task_2").mkdir()
        (task_package_dir / "_template").mkdir()
        (task_package_dir / "__pycache__").mkdir()
        (task_package_dir / "__init__.py").write_text("")
        (task_package_dir / "helpers.py").write_text("# helper functions")
        (task_package_dir / "README.md").write_text("# Tasks")

        candidates = list(task_package_dir.glob("*"))
        filtered = [c for c in candidates if c.is_dir() and not c.name.startswith("_")]

        assert len(filtered) == 2
        names = {c.name for c in filtered}
        assert names == {"valid_task_1", "valid_task_2"}


# =============================================================================
# Tests: get_tasks (integration)
# =============================================================================


class TestGetTasks:
    @pytest.fixture(autouse=True)
    def _reset_filters(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(environment, "INCLUDE_TASKS", [])
        monkeypatch.setattr(environment, "EXCLUDE_TASKS", [])

    def test_discovers_tasks_from_directories(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        (tmp_path / "task_one").mkdir()
        (tmp_path / "task_two").mkdir()

        import environment.tasks

        monkeypatch.setattr(environment.tasks, "__path__", [str(tmp_path)])

        mod_one = _make_task_module(
            "environment.tasks.task_one", {"ValidTask": ValidTask, "Task": Task}
        )
        mod_two = _make_task_module(
            "environment.tasks.task_two",
            {"AnotherValidTask": AnotherValidTask, "Task": Task},
        )

        def import_side_effect(name):
            return {
                "environment.tasks.task_one": mod_one,
                "environment.tasks.task_two": mod_two,
            }[name]

        with patch("importlib.import_module", side_effect=import_side_effect):
            tasks = environment.get_tasks()

        assert len(tasks) == 2
        assert {t.id for t in tasks} == {"valid-task", "another-valid-task"}

    def test_skips_underscore_prefixed_directories(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        (tmp_path / "valid_task").mkdir()
        (tmp_path / "_template").mkdir()
        (tmp_path / "__pycache__").mkdir()

        import environment.tasks

        monkeypatch.setattr(environment.tasks, "__path__", [str(tmp_path)])

        mod = _make_task_module(
            "environment.tasks.valid_task", {"ValidTask": ValidTask, "Task": Task}
        )

        with patch("importlib.import_module", return_value=mod):
            tasks = environment.get_tasks()

        assert tasks == [ValidTask]

    def test_skips_non_directory_entries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        (tmp_path / "valid_task").mkdir()
        (tmp_path / "not_a_directory.py").write_text("# some file")

        import environment.tasks

        monkeypatch.setattr(environment.tasks, "__path__", [str(tmp_path)])

        mod = _make_task_module(
            "environment.tasks.valid_task", {"ValidTask": ValidTask, "Task": Task}
        )

        with patch("importlib.import_module", return_value=mod):
            tasks = environment.get_tasks()

        assert tasks == [ValidTask]

    def test_handles_empty_tasks_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import environment.tasks

        monkeypatch.setattr(environment.tasks, "__path__", [str(tmp_path)])

        tasks = environment.get_tasks()

        assert tasks == []
