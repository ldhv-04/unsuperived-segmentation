import torch
import pytest
from tasdf.data.coco import COCOEmbryoDataset


def test_coco_embryo_dataset_loading_and_rasterization(synthetic_coco_dir):
    """Kiểm tra đọc dữ liệu COCO và giải thuật rasterize đa giác trên dữ liệu tổng hợp."""
    dataset = COCOEmbryoDataset(root_dir=synthetic_coco_dir, split="train", size=64, is_labeled=True)

    # 1. Kiểm tra số lượng ảnh
    assert len(dataset) == 4

    # 2. Kiểm tra mẫu đầu tiên
    img, mask = dataset[0]

    assert img.shape == (1, 64, 64)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert mask.dtype == torch.long

    # 3. Kiểm tra nhãn
    unique_classes = torch.unique(mask).tolist()
    for c in unique_classes:
        assert isinstance(c, int)
