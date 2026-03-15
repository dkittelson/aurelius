import pytest
from pathlib import Path


def test_yaml_config_loads():
    from src.config import load_yaml_config

    config = load_yaml_config()
    assert "project" in config
    assert config["project"]["name"] == "aurelius"


def test_yaml_config_has_required_sections():
    from src.config import load_yaml_config

    config = load_yaml_config()
    required = ["project", "data", "graph", "features", "model", "agent", "api"]
    for section in required:
        assert section in config, f"Missing config section: {section}"


def test_yaml_config_model_params():
    from src.config import load_yaml_config

    config = load_yaml_config()
    gnn = config["model"]["gnn"]
    assert gnn["type"] == "GATv2Conv"
    assert gnn["hidden_channels"] == 128
    assert gnn["num_heads"] == 4
    assert gnn["num_layers"] == 3


def test_yaml_config_missing_file():
    from src.config import load_yaml_config

    with pytest.raises(FileNotFoundError):
        load_yaml_config("nonexistent.yaml")


def test_settings_defaults():
    from src.config import Settings

    s = Settings()
    assert s.log_level == "INFO"
    assert s.faiss_index_path == "data/vectordb/forensic_index.faiss"
    assert s.model_checkpoint_dir == "data/processed/checkpoints"
