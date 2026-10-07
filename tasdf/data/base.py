from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BaseEmbryoDataset(Dataset, ABC):
    """
    Interface trừu tượng chuẩn (SPI) cho tất cả các loại Dataset phôi học.
    Đảm bảo tính nhất quán của đầu ra tensor cho mô hình TASDF.
    """

    def __init__(self, size: int = 256, is_labeled: bool = True):
        self.size = size
        self.is_labeled = is_labeled

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    def preprocess_image(self, img_path: str) -> torch.Tensor:
        """Đọc và chuẩn hóa ảnh phôi xám về khoảng [0, 1] kích thước [1, size, size]."""
        img = Image.open(img_path).convert("L").resize((self.size, self.size))
        img_np = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
