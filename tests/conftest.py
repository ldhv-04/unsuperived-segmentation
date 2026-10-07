import json
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tasdf.models import TASDFNet


@pytest.fixture
def dummy_model():
    """Mô hình TASDFNet thu nhỏ để chạy test nhanh trên CPU."""
    return TASDFNet(in_channels=1, base_channels=16, num_classes=5, k_iters=2)


@pytest.fixture
def dummy_batch():
    """Tạo batch dữ liệu giả lập (ảnh + nhãn) cho bài toán phân vùng."""
    x = torch.rand(2, 1, 32, 32)
    y = torch.randint(0, 5, (2, 32, 32))
    return x, y


@pytest.fixture
def dummy_loaders():
    """DataLoaders có nhãn và không nhãn cho test vòng lặp huấn luyện."""
    x_l = torch.rand(4, 1, 32, 32)
    y_l = torch.randint(0, 5, (4, 32, 32))
    x_u = torch.rand(4, 1, 32, 32)

    l_ds = TensorDataset(x_l, y_l)
    labeled_loader = DataLoader(l_ds, batch_size=2)
    unlabeled_loader = DataLoader(x_u, batch_size=2)
    return labeled_loader, unlabeled_loader


@pytest.fixture
def synthetic_coco_dir(tmp_path: Path) -> Path:
    """Tạo bộ dữ liệu COCO chuẩn giả lập không phụ thuộc đường dẫn máy chủ."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for i in range(4):
        img = Image.fromarray(np.uint8(np.random.rand(64, 64) * 255))
        img.save(train_dir / f"embryo_{i}.png")
    for i in range(2):
        img = Image.fromarray(np.uint8(np.random.rand(64, 64) * 255))
        img.save(val_dir / f"val_embryo_{i}.png")

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
