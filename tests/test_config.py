from pathlib import Path

from hww.config import load_config


def test_default_config_resolves_paths_from_repository_root():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/pipeline.toml")
    assert config.root == root
    assert config.samples["truth_signal"].input_path == root / "data/raw/truth/signal.root"

