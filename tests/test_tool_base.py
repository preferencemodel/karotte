from pathlib import Path
from typing import final

import pytest
from pydantic import BaseModel

from karotte.tool_base import ToolBase, ToolConfigWriter
from karotte.tools.bash import BashConfig, bash
from karotte.tools.view_lines_in_file import ViewLinesInFileConfig, view_lines_in_file


class DummyConfig(BaseModel):
    value: int = 10
    name: str = "default"


@final
class dummy_tool(ToolBase[DummyConfig]):
    config_schema = DummyConfig


class TestToolBase:
    def test_returns_default_config_when_directory_does_not_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        nonexistent_dir = tmp_path / "nonexistent"
        monkeypatch.setattr("karotte.tool_base.CONFIGS_DIR", nonexistent_dir)

        tool = dummy_tool()

        assert tool.config.value == 10
        assert tool.config.name == "default"

    def test_returns_default_config_when_file_does_not_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("karotte.tool_base.CONFIGS_DIR", tmp_path)

        tool = dummy_tool()

        assert tool.config.value == 10
        assert tool.config.name == "default"

    def test_loads_config_from_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        config_file = tmp_path / "dummy_tool.json"
        config_file.write_text('{"value": 42, "name": "custom"}')
        monkeypatch.setattr("karotte.tool_base.CONFIGS_DIR", tmp_path)

        tool = dummy_tool()

        assert tool.config.value == 42
        assert tool.config.name == "custom"

    def test_tool_name_returns_class_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("karotte.tool_base.CONFIGS_DIR", tmp_path)

        tool = dummy_tool()

        assert tool.tool_name == "dummy_tool"


class TestToolConfigWriter:
    def test_write_creates_directory_if_missing(self, tmp_path: Path):
        config_dir = tmp_path / "nested" / "config" / "dir"
        writer = ToolConfigWriter(config_dir=config_dir)

        writer.write("my_tool", DummyConfig(value=99))

        assert config_dir.is_dir()
        assert (config_dir / "my_tool.json").is_file()

    def test_write_creates_valid_json(self, tmp_path: Path):
        writer = ToolConfigWriter(config_dir=tmp_path)

        writer.write("my_tool", DummyConfig(value=123, name="test"))

        config_file = tmp_path / "my_tool.json"
        loaded = DummyConfig.model_validate_json(config_file.read_text())
        assert loaded.value == 123
        assert loaded.name == "test"

    def test_write_returns_self_for_chaining(self, tmp_path: Path):
        writer = ToolConfigWriter(config_dir=tmp_path)

        result = writer.write("tool1", DummyConfig(value=1))

        assert result is writer

    def test_chained_writes(self, tmp_path: Path):
        writer = ToolConfigWriter(config_dir=tmp_path)

        writer.write("tool1", DummyConfig(value=1)).write(
            "tool2", DummyConfig(value=2)
        ).write("tool3", DummyConfig(value=3))

        assert (tmp_path / "tool1.json").is_file()
        assert (tmp_path / "tool2.json").is_file()
        assert (tmp_path / "tool3.json").is_file()

    def test_clear_removes_all_json_files(self, tmp_path: Path):
        (tmp_path / "tool1.json").write_text("{}")
        (tmp_path / "tool2.json").write_text("{}")
        (tmp_path / "other.txt").write_text("keep me")
        writer = ToolConfigWriter(config_dir=tmp_path)

        writer.clear()

        assert not (tmp_path / "tool1.json").exists()
        assert not (tmp_path / "tool2.json").exists()
        assert (tmp_path / "other.txt").exists()  # Non-JSON files preserved

    def test_clear_handles_empty_directory(self, tmp_path: Path):
        writer = ToolConfigWriter(config_dir=tmp_path)

        writer.clear()  # Should not raise

    def test_clear_handles_nonexistent_directory(self, tmp_path: Path):
        nonexistent = tmp_path / "nonexistent"
        writer = ToolConfigWriter(config_dir=nonexistent)

        writer.clear()  # Should not raise


class TestToolConfigIntegration:
    def test_bash_loads_custom_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        config_file = tmp_path / "bash.json"
        config_file.write_text(BashConfig(default_timeout_s=42).model_dump_json())
        monkeypatch.setattr("karotte.tool_base.CONFIGS_DIR", tmp_path)

        tool = bash()

        assert tool.config.default_timeout_s == 42
        assert tool.__call__.__doc__
        assert "42" in tool.__call__.__doc__

    @pytest.mark.asyncio
    async def test_view_lines_in_file_loads_custom_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_file = config_dir / "view_lines_in_file.json"
        config_file.write_text(ViewLinesInFileConfig(max_lines=5).model_dump_json())
        monkeypatch.setattr("karotte.tool_base.CONFIGS_DIR", config_dir)

        test_file = tmp_path / "test.txt"
        test_file.write_text("line\n" * 20)

        tool = view_lines_in_file()

        with pytest.raises(ValueError, match="limit is 5 lines"):
            await tool(test_file, from_line=1, to_line=10)
