from typing import Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .coco import COCOEmbryoDataset
from .folder import FolderEmbryoDataset


def build_dataset(
    data_cfg: dict, split: str = "train", is_labeled: bool = True
) -> Dataset:
    """
    Factory khởi tạo Dataset tương ứng dựa trên cấu hình 'type'/'format'.
    """
    ds_type = data_cfg.get("type") or data_cfg.get("format", "coco").lower()
    root_dir = data_cfg.get("path") or data_cfg.get("root_dir", ".")
    size = data_cfg.get("image_size", 256)

    # Lấy tên split thực tế từ config nếu có (ví dụ: data_cfg["train"])
    actual_split = data_cfg.get(split, split)

    if ds_type == "coco":
        return COCOEmbryoDataset(
            root_dir=root_dir,
            split=actual_split,
            size=size,
            is_labeled=is_labeled,
        )
    elif ds_type in ("folder", "embryo", "mask"):
        return FolderEmbryoDataset(
            data_dir=root_dir,
            split=actual_split,
            size=size,
            is_labeled=is_labeled,
        )
    else:
        raise ValueError(
            f"Định dạng dữ liệu '{ds_type}' chưa được hỗ trợ. Hỗ trợ: ['coco', 'folder']"
        )


def build_dataloaders(
    data_cfg: dict,
    batch_size: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo bộ ba DataLoader: (labeled_loader, unlabeled_loader, val_loader)
    """
    labeled_ratio = float(data_cfg.get("labeled_ratio", 0.2))

    # 1. Khởi tạo tập huấn luyện
    full_train = build_dataset(data_cfg, split="train", is_labeled=True)
    unlabeled_train = build_dataset(data_cfg, split="train", is_labeled=False)

    num_train = len(full_train)

    # Tự động phát hiện phân vùng vật lý (Physical Annotation Auto-Detection):
    # Nếu tập train COCO đã chứa sẵn các ảnh bị tước nhãn (len(anns) == 0),
    # ta tự động gom ảnh có nhãn vào labeled_loader và ảnh không nhãn vào unlabeled_loader.
    physical_labeled = []
    physical_unlabeled = []
    if hasattr(full_train, "images") and hasattr(full_train, "img_id_to_anns"):
        for i, img in enumerate(full_train.images):
            anns = full_train.img_id_to_anns.get(img["id"], [])
            if len(anns) > 0:
                physical_labeled.append(i)
            else:
                physical_unlabeled.append(i)

    use_physical = data_cfg.get("split_mode") == "physical" or data_cfg.get("use_physical_split", False)
    if use_physical or (len(physical_unlabeled) > 0 and len(physical_labeled) > 0 and "labeled_ratio" not in data_cfg):
        labeled_indices = physical_labeled
        unlabeled_indices = physical_unlabeled
    else:
        num_labeled = max(1, int(num_train * labeled_ratio))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(num_train, generator=generator).tolist()
        labeled_indices = indices[:num_labeled]
        unlabeled_indices = indices[num_labeled:]

    pin = pin_memory and torch.cuda.is_available()

    labeled_loader = DataLoader(
        Subset(full_train, labeled_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(labeled_indices) >= batch_size else False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    unlabeled_loader = DataLoader(
        Subset(unlabeled_train, unlabeled_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(unlabeled_indices) >= batch_size else False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    # 2. Khởi tạo tập kiểm tra
    val_split_name = "val" if "val" in data_cfg else "valid"
    val_ds = build_dataset(data_cfg, split=val_split_name, is_labeled=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return labeled_loader, unlabeled_loader, val_loader
