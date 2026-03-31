from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return ROOT / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def features_dir() -> Path:
    return data_dir() / "features"


def models_dir() -> Path:
    return data_dir() / "models"
