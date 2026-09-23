"""
Shared configuration helpers.

Paths in config.yaml are treated as project-root-relative so the app behaves
the same whether it is launched from the repository root or from src/.
"""

import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    _load_env_file(PROJECT_ROOT / ".env")

PATH_KEYS = {
    "index_save_path",
    "chunks_save_path",
    "eval_qa_path",
    "results_path",
}


def _resolve_project_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        candidates = [Path.cwd() / config_path, PROJECT_ROOT / config_path]
        config_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])

    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file at {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for key in PATH_KEYS:
        if key in cfg and cfg[key]:
            cfg[key] = _resolve_project_path(cfg[key])

    return cfg
