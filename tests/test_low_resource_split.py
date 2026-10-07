import json
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch

from tasdf.data.builder import build_dataloaders


@pytest.fixture
def synthetic_partial_coco_dir(tmp_path: Path) -> Path:
    """Tạo bộ dữ liệu COCO giả lập với đúng 2 ảnh có nhãn và 4 ảnh không nhãn."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "valid"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    # 6 ảnh train (2 có nhãn, 4 không nhãn)
    for i in range(6):
        img = Image.fromarray(np.uint8(np.random.rand(64, 64) * 255))
        img.save(train_dir / f"embryo_{i}.png")
    for i in range(2):
        img = Image.fromarray(np.uint8(np.random.rand(64, 64) * 255))
        img.save(val_dir / f"val_embryo_{i}.png")

    # Chỉ ghi nhãn cho image_id 0 và image_id 1
    coco_train = {
        "images": [{"id": i, "file_name": f"embryo_{i}.png", "width": 64, "height": 64} for i in range(6)],
        "categories": [
            {"id": 1, "name": "ZP"},
            {"id": 2, "name": "TE"},
            {"id": 3, "name": "ICM"},
        ],
        "annotations": [
            {"id": 101, "image_id": 0, "category_id": 1, "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]},
            {"id": 102, "image_id": 1, "category_id": 2, "segmentation": [[20, 20, 40, 20, 40, 40, 20, 40]]},
        ],
    }
    with open(train_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
        json.dump(coco_train, f)

    coco_val = {
        "images": [{"id": i, "file_name": f"val_embryo_{i}.png", "width": 64, "height": 64} for i in range(2)],
        "categories": coco_train["categories"],
        "annotations": [
            {"id": 201, "image_id": 0, "category_id": 1, "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]},
        ],
    }
    with open(val_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
        json.dump(coco_val, f)

    return tmp_path


def test_builder_auto_detects_physical_unlabeled(synthetic_partial_coco_dir: Path):
    """Kiểm tra build_dataloaders tự động nhận diện đúng 2 ảnh có nhãn và 4 ảnh không nhãn."""
    cfg = {
        "format": "coco",
        "path": str(synthetic_partial_coco_dir),
        "train": "train",
        "val": "valid",
        "image_size": 64,
    }

    labeled_loader, unlabeled_loader, val_loader = build_dataloaders(
        cfg, batch_size=2, num_workers=0
    )

    assert len(labeled_loader.dataset) == 2, f"Kỳ vọng 2 ảnh có nhãn, nhận {len(labeled_loader.dataset)}"
    assert len(unlabeled_loader.dataset) == 4, f"Kỳ vọng 4 ảnh không nhãn, nhận {len(unlabeled_loader.dataset)}"

    # Labeled loader phải trả về (x, y) với nhãn không rỗng
    x_l, y_l = next(iter(labeled_loader))
    assert x_l.shape == (2, 1, 64, 64)
    assert y_l.shape == (2, 64, 64)
    assert y_l.sum() > 0, "Mặt nạ nhãn không được rỗng toàn bộ!"

    # Unlabeled loader phải trả về tensor ảnh duy nhất
    x_u = next(iter(unlabeled_loader))
    assert isinstance(x_u, torch.Tensor)
    assert x_u.shape == (2, 1, 64, 64)


def test_real_10p_dataset_integrity():
    """Kiểm tra tính toàn vẹn của bộ dữ liệu 10% thực tế được sinh ra trên đĩa."""
    dataset_dir = Path("/home/terminator/workspace/Attech/Research/dataset/Blast_Segment_10p_coco")
    if not dataset_dir.exists():
        pytest.skip("Chưa khởi tạo thư mục Blast_Segment_10p_coco")

    train_ann_file = dataset_dir / "train" / "_annotations.coco.json"
    assert train_ann_file.exists()

    with open(train_ann_file, "r", encoding="utf-8") as f:
        d = json.load(f)

    images = d["images"]
    annotations = d["annotations"]

    assert len(images) == 621, f"Tổng số ảnh train phải là 621, nhận: {len(images)}"

    # Tập hợp các image_id có nhãn
    labeled_img_ids = set(a["image_id"] for a in annotations)
    assert len(labeled_img_ids) == 62, f"Số ảnh có nhãn phải đúng 62 (10%), nhận: {len(labeled_img_ids)}"

    # 559 ảnh còn lại phải hoàn toàn không có annotation
    unlabeled_count = len(images) - len(labeled_img_ids)
    assert unlabeled_count == 559

    # Kiểm tra cấu trúc của 62 ảnh: Đảm bảo có đầy đủ 3 cấu trúc ZP, TE, ICM
    img_structs = {}
    for a in annotations:
        cid = a["category_id"]
        if cid in (7, 8, 9):
            img_structs.setdefault(a["image_id"], set()).add("ZP")
        elif cid in (4, 5, 6):
            img_structs.setdefault(a["image_id"], set()).add("TE")
        elif cid in (1, 2, 3):
            img_structs.setdefault(a["image_id"], set()).add("ICM")

    for iid in labeled_img_ids:
        assert img_structs[iid] == {"ZP", "TE", "ICM"}, f"Ảnh {iid} không đủ cả 3 cấu trúc ZP, TE, ICM!"

    # Kiểm tra DataLoader nạp chuẩn
    cfg = {
        "format": "coco",
        "path": str(dataset_dir),
        "train": "train",
        "val": "valid",
        "image_size": 64,
    }
    labeled_loader, unlabeled_loader, val_loader = build_dataloaders(
        cfg, batch_size=4, num_workers=0
    )
    assert len(labeled_loader.dataset) == 62
    assert len(unlabeled_loader.dataset) == 559
    assert len(val_loader.dataset) == 76
