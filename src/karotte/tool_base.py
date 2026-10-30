from abc import ABC
from pathlib import Path
from typing import ClassVar, Final, Self, cast

from pydantic import BaseModel

CONFIGS_DIR = Path("~/.config/karotte/tool_configs").expanduser()


class ToolBase[ConfigT: BaseModel](ABC):
    config_schema: ClassVar[type[BaseModel]]

    def __init__(self) -> None:
        self.config: Final = self.load_config()

    @property
    def tool_name(self) -> str:
        return self.__class__.__name__

    @classmethod
    def load_config(cls) -> ConfigT:
        """The config a task wrote for this tool with `ToolConfigWriter`, or the
        schema's defaults if it wrote none. Reads it without making a tool."""
        if CONFIGS_DIR.is_dir():
            config_path = CONFIGS_DIR / f"{cls.__name__}.json"
            if config_path.is_file():
                return cast(
                    ConfigT,
                    cls.config_schema.model_validate_json(config_path.read_text()),
                )

        return cast(ConfigT, cls.config_schema())


class ToolConfigWriter:
    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir: Final = CONFIGS_DIR if config_dir is None else config_dir

    def write(self, tool_name: str, config: BaseModel) -> Self:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        path = self.config_dir / f"{tool_name}.json"
        path.write_text(config.model_dump_json())
        return self

    def clear(self) -> None:
        """Remove all config files."""
        for f in self.config_dir.glob("*.json"):
            f.unlink()
