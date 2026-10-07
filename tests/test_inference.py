import numpy as np
import pytest
from tasdf.utils.metrics import decompose_te_and_blastocoel


def test_decompose_te_and_blastocoel():
    # Tạo một mặt nạ giả lập: hình tròn bán kính 50 điểm ảnh mang Lớp 2 (TE)
    h, w = 120, 120
    mask = np.zeros((h, w), dtype=np.uint8)
    y, x = np.ogrid[:h, :w]
    circle = (x - 60) ** 2 + (y - 60) ** 2 <= 40 ** 2
    mask[circle] = 2  # Lớp 2

    # Đặt một khối ICM (Lớp 3) ở góc trong
    icm_circle = (x - 50) ** 2 + (y - 50) ** 2 <= 10 ** 2
    mask[icm_circle] = 3

    # Phân rã
    decomposed = decompose_te_and_blastocoel(mask, te_thickness_ratio=0.1)

    unique_classes = set(np.unique(decomposed))
    # Phải có cả 4 lớp: 0 (Nền), 2 (TE viền), 3 (ICM), 4 (Blastocoel lõi)
    assert 0 in unique_classes
    assert 2 in unique_classes
    assert 3 in unique_classes
    assert 4 in unique_classes
