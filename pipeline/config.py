"""配置加载：实验继承 + 深合并 + 校验。"""

import os
import re
from pathlib import Path

import yaml

EXPERIMENTS_DIR = Path(__file__).parent.parent / "config" / "experiments"


def load_experiment(name: str) -> dict:
    """加载实验配置，支持 parent 继承。"""
    path = EXPERIMENTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # parent inheritance
    if "parent" in cfg:
        parent = load_experiment(cfg.pop("parent"))
        cfg = deep_merge(parent, cfg)

    return interpolate_env(cfg)


def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def interpolate_env(cfg: dict) -> dict:
    def resolve(v):
        if isinstance(v, str):
            def sub(m):
                var = m.group(1)
                val = os.environ.get(var)
                if val is None:
                    return ""
                return val
            return re.sub(r"\$\{(\w+)\}", sub, v)
        if isinstance(v, dict):
            return {k: resolve(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [resolve(vv) for vv in v]
        return v
    return resolve(cfg)
