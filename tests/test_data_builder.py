import json
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch

from tasdf.data.builder import build_dataset, build_dataloaders


@pytest.fixture
def synthetic_coco_dir(tmp_path: Path) -> Path:
    """Tạo bộ dữ liệu COCO giả lập trên thư mục tạm."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "valid"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    # Tạo 4 ảnh train và 2 ảnh val
    for i in range(4):
        img = Image.fromarray(np.uint8(np.random.rand(64, 64) * 255))
        img.save(train_dir / f"embryo_{i}.png")
    for i in range(2):
        img = Image.fromarray(np.uint8(np.random.rand(64, 64) * 255))
        img.save(val_dir / f"val_embryo_{i}.png")

    # Tạo annotations json
    coco_data = {
        "images": [{"id": i, "file_name": f"embryo_{i}.png", "width": 64, "height": 64} for i in range(4)],
        "categories": [
            {"id": 1, "name": "ZP"},
            {"id": 2, "name": "TE"},
            {"id": 3, "name": "ICM"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
            }
        ],
    }
    with open(train_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
        json.dump(coco_data, f)

    val_data = coco_data.copy()
    val_data["images"] = [{"id": i, "file_name": f"val_embryo_{i}.png", "width": 64, "height": 64} for i in range(2)]
    with open(val_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f)

    return tmp_path


def test_build_dataloaders_synthetic(synthetic_coco_dir: Path):
    cfg = {
        "format": "coco",
        "path": str(synthetic_coco_dir),
        "train": "train",
        "val": "valid",
        "image_size": 64,
        "labeled_ratio": 0.5,
    }

    labeled_loader, unlabeled_loader, val_loader = build_dataloaders(
        cfg, batch_size=2, num_workers=0
    )

    assert len(labeled_loader.dataset) == 2
    assert len(unlabeled_loader.dataset) == 2
    assert len(val_loader.dataset) == 2

    # Kiểm tra kích thước batch
    x_l, y_l = next(iter(labeled_loader))
    assert x_l.shape == (2, 1, 64, 64)
    assert y_l.shape == (2, 64, 64)

    x_u = next(iter(unlabeled_loader))
    assert x_u.shape == (2, 1, 64, 64)
