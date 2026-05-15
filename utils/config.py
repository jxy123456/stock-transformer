import os
from pathlib import Path
from typing import Any

import yaml


class ConfigManager:
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)
        self._config = self._load_and_merge()

    def _load_and_merge(self) -> dict:
        default_path = self.config_dir / "default.yaml"
        if default_path.exists():
            base = yaml.safe_load(open(default_path, "r", encoding="utf-8"))
        else:
            base = {}

        for yaml_file in sorted(self.config_dir.glob("*.yaml")):
            if yaml_file.name == "default.yaml":
                continue
            override = yaml.safe_load(open(yaml_file, "r", encoding="utf-8"))
            if override:
                base = self._deep_merge(base, override)

        return base

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        value = self._config.get(name)
        if isinstance(value, dict):
            return _ConfigProxy(value)
        return value

    @property
    def raw(self) -> dict:
        return self._config


class _ConfigProxy:
    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        value = self._data.get(name)
        if isinstance(value, dict):
            return _ConfigProxy(value)
        return value

    def __repr__(self) -> str:
        return repr(self._data)
