from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from .base import BaseEmbryoDataset


class FolderEmbryoDataset(BaseEmbryoDataset):
    """
    Bộ nạp dữ liệu từ thư mục ảnh và mặt nạ rời (Mask PNG).
    Cấu trúc:
      data_dir/
        ├── images/ (chứa .png, .jpg)
        └── masks/  (chứa .png mặt nạ nhãn, nếu có nhãn)
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "",
        size: int = 256,
        is_labeled: bool = True,
    ):
        super().__init__(size=size, is_labeled=is_labeled)
        base = Path(data_dir)
        self.folder_path = base / split if split and (base / split).exists() else base

        # Tìm thư mục ảnh
        img_candidates = [
            self.folder_path / "images",
            self.folder_path,
        ]
        self.img_dir = self.folder_path
        for c in img_candidates:
            if c.exists() and (list(c.glob("*.jpg")) or list(c.glob("*.png"))):
                self.img_dir = c
                break

        self.img_files = sorted(
            list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        )

        # Tìm thư mục masks nếu có nhãn
        self.mask_dir = self.folder_path / "masks"
        if not self.mask_dir.exists():
            self.mask_dir = self.folder_path

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_p = self.img_files[idx]
        img_tensor = self.preprocess_image(str(img_p))

        if not self.is_labeled:
            return img_tensor

        # Tìm mask tương ứng cùng tên
        mask_candidates = [
            self.mask_dir / f"{img_p.stem}.png",
            self.mask_dir / f"{img_p.stem}_mask.png",
            self.mask_dir / img_p.name,
        ]
        mask_p = None
        for c in mask_candidates:
            if c.exists():
                mask_p = c
                break

        if mask_p:
            mask_img = Image.open(mask_p).resize((self.size, self.size), Image.NEAREST)
            mask_np = np.array(mask_img, dtype=np.int64)
        else:
            # Fallback mask rỗng nếu không tìm thấy
            mask_np = np.zeros((self.size, self.size), dtype=np.int64)

        return img_tensor, torch.tensor(mask_np, dtype=torch.long)
