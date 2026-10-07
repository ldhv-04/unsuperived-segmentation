import pytest
from tasdf.data.builder import build_dataloaders


def test_build_dataloaders_from_config(synthetic_coco_dir):
    cfg = {
        "format": "coco",
        "path": str(synthetic_coco_dir),
        "train": "train",
        "val": "valid",
        "image_size": 64,
        "labeled_ratio": 0.5,
    }

    labeled_loader, unlabeled_loader, val_loader = build_dataloaders(cfg, batch_size=2, num_workers=0)

    assert len(labeled_loader.dataset) == 2
    assert len(unlabeled_loader.dataset) == 2
    assert len(val_loader.dataset) == 2

    # Lấy thử 1 batch từ unlabeled_loader để kiểm tra kích thước
    x_u = next(iter(unlabeled_loader))
    assert x_u.shape == (2, 1, 64, 64)
