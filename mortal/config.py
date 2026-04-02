import os
from pathlib import Path

try:
    from .toml_utils import load_toml_file
except ImportError:
    from toml_utils import load_toml_file


PATH_KEYS = {
    'init_state_file',
    'state_file',
    'latest_state_file',
    'best_loss_state_file',
    'best_acc_state_file',
    'best_state_file',
    'tensorboard_dir',
    'log_dir',
    'file_index',
    'buffer_dir',
    'drain_dir',
    'dir',
    'tactics',
}

GLOB_KEYS = {
    'globs',
    'train_globs',
    'val_globs',
}


def _resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_config_paths(node, base_dir: Path):
    if isinstance(node, dict):
        resolved = {}
        for key, value in node.items():
            if isinstance(value, dict):
                resolved[key] = _resolve_config_paths(value, base_dir)
            elif key in PATH_KEYS and isinstance(value, str):
                resolved[key] = _resolve_path(value, base_dir)
            elif key in GLOB_KEYS and isinstance(value, list):
                resolved[key] = [
                    _resolve_path(item, base_dir) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                resolved[key] = value
        return resolved
    return node


config_file = os.environ.get('MORTAL_CFG')
if config_file is None:
    config_path = Path(__file__).with_name('config.toml').resolve()
else:
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

config = _resolve_config_paths(load_toml_file(config_path), config_path.parent)
