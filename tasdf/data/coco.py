import json
from pathlib import Path
import cv2
import numpy as np
import torch
from .base import BaseEmbryoDataset


class COCOEmbryoDataset(BaseEmbryoDataset):
    """
    Bộ nạp dữ liệu phôi nang từ định dạng COCO Segmentation JSON.
    Hỗ trợ linh hoạt cấu trúc thư mục từ Roboflow hoặc COCO chuẩn.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        size: int = 256,
        is_labeled: bool = True,
        class_mapping: dict[str, int] | None = None,
    ):
        super().__init__(size=size, is_labeled=is_labeled)
        self.root_path = Path(root_dir)
        self.split = split

        # Tự động tìm kiếm file nhãn COCO
        candidates = [
            self.root_path / split / "_annotations.coco.json",
            self.root_path / split / "annotations.json",
            self.root_path / f"annotations/{split}.json",
            self.root_path / "_annotations.coco.json",
            self.root_path / "annotations.json",
        ]
        json_path = None
        for c in candidates:
            if c.exists():
                json_path = c
                break

        if not json_path:
            raise FileNotFoundError(
                f"Không tìm thấy file nhãn COCO trong '{self.root_path}' cho tập '{split}'."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            self.coco = json.load(f)

        self.img_dir = json_path.parent
        self.images = self.coco["images"]
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in self.coco["categories"]}

        # Lập chỉ mục: image_id -> danh sách annotations
        self.img_id_to_anns = {}
        for ann in self.coco.get("annotations", []):
            self.img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.class_mapping = class_mapping

    def _map_category_to_class_id(self, cat_name: str) -> int:
        if self.class_mapping and cat_name in self.class_mapping:
            return self.class_mapping[cat_name]
        if "ZP" in cat_name:
            return 1
        elif "TE" in cat_name:
            return 2
        elif "ICM" in cat_name:
            return 3
        elif "Blastocyst" in cat_name:
            return 4
        return 0

    def _rasterize_polygons(self, anns: list, orig_w: int, orig_h: int) -> np.ndarray:
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        def draw_priority(ann):
            name = self.cat_id_to_name.get(ann["category_id"], "")
            if "Blastocyst" in name:
                return 0
            if "ZP" in name:
                return 1
            if "TE" in name:
                return 2
            return 3

        sorted_anns = sorted(anns, key=draw_priority)
        for ann in sorted_anns:
            cat_name = self.cat_id_to_name.get(ann["category_id"], "")
            class_id = self._map_category_to_class_id(cat_name)

            for seg in ann.get("segmentation", []):
                poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [poly], color=class_id)
        return mask

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_path = self.img_dir / img_info["file_name"]
        if not img_path.exists():
            # Thử tìm trong root_path
            img_path = self.root_path / img_info["file_name"]

        img_tensor = self.preprocess_image(str(img_path))
        if not self.is_labeled:
            return img_tensor

        anns = self.img_id_to_anns.get(img_info["id"], [])
        orig_w = img_info.get("width", self.size)
        orig_h = img_info.get("height", self.size)
        mask_np = self._rasterize_polygons(anns, orig_w, orig_h)
        mask_resized = cv2.resize(
            mask_np, (self.size, self.size), interpolation=cv2.INTER_NEAREST
        )
        return img_tensor, torch.tensor(mask_resized, dtype=torch.long)
