import yaml
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

# --- YAML Loader ---
@lru_cache
def load_yaml_config(path: str = "config.yaml"):
    full_path = Path(__file__).parent.parent / path
    with open(full_path) as f:
        return yaml.safe_load(f)


# --- Settings (env vars) ---
class Settings(BaseSettings):
    gemini_api_key: str | None = None
    dataset: str = "elliptic"
    log_level: str = "INFO"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()