"""Small shared path/config helpers; all paths are relative to the project."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path).resolve() if path else ROOT / "configs/main.yaml"
    with path.open() as handle:
        config = yaml.safe_load(handle)
    config["_root"] = str(ROOT)
    return config


def project_path(config, value):
    value = Path(value).expanduser()
    return value if value.is_absolute() else Path(config["_root"]) / value
