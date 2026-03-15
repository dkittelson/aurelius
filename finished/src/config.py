from pathlib import Path
from functools import lru_cache

import yaml
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    faiss_index_path: str = Field(
        default="data/vectordb/forensic_index.faiss",
        alias="FAISS_INDEX_PATH",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    model_checkpoint_dir: str = Field(
        default="data/processed/checkpoints",
        alias="MODEL_CHECKPOINT_DIR",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def load_yaml_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_yaml_config() -> dict:
    return load_yaml_config()
