from pathlib import Path
from typing import Any
import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "in_channels": 1,
        "base_channels": 64,
        "num_classes": 5,
        "k_iters": 10,
    },
    "loss": {
        "lambda_topo": 0.10,
        "gamma": 0.5,
        "alpha": 1.0,
        "max_epochs": 50,
    },
    "training": {
        "batch_size": 4,
        "learning_rate": 0.0001,
        "weight_decay": 0.00005,
        "amp": True,
        "accum_steps": 1,
        "num_workers": 2,
        "log_interval": 20,
        "save_period": 1,
    },
}


def load_yaml(path_or_str: str | Path) -> dict[str, Any]:
    """Nạp file YAML một cách an toàn."""
    path = Path(path_or_str)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: dict, update: dict) -> dict:
    """Hợp nhất đệ quy 2 dictionary."""
    result = base.copy()
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def resolve_data_config(data_input: str | Path | dict) -> dict[str, Any]:
    """
    Chuẩn hóa cấu hình dữ liệu đầu vào.
    Hỗ trợ đường dẫn file YAML hoặc dict Python trực tiếp.
    """
    if isinstance(data_input, (str, Path)):
        p = Path(data_input)
        raw_cfg = load_yaml(p)
        base_dir = p.parent
    elif isinstance(data_input, dict):
        raw_cfg = data_input.copy()
        base_dir = Path(".")
    else:
        raise TypeError(f"Dữ liệu cấu hình không hợp lệ: {type(data_input)}")

    # Trích xuất data_cfg nếu nằm trong key 'dataset'
    data_cfg = raw_cfg.get("dataset", raw_cfg)

    # Chuẩn hóa root_dir / path
    data_path = Path(data_cfg.get("path") or data_cfg.get("root_dir", "."))
    if not data_path.is_absolute():
        data_path = (base_dir / data_path).resolve()

    resolved = {
        "type": data_cfg.get("format") or data_cfg.get("type", "coco").lower(),
        "root_dir": str(data_path),
        "train": data_cfg.get("train", "train"),
        "val": data_cfg.get("val") or data_cfg.get("valid", "valid"),
        "test": data_cfg.get("test", "test"),
        "image_size": int(data_cfg.get("image_size", 256)),
        "labeled_ratio": float(data_cfg.get("labeled_ratio", 0.2)),
        "names": data_cfg.get("names", {}),
    }
    return resolved
