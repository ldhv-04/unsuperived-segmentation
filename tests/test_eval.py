import numpy as np
import pytest
from tasdf.utils.metrics import compute_sample_metrics


def test_compute_sample_metrics_identical():
    # 1. Khi dự đoán trùng khớp hoàn toàn với nhãn chuẩn
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1  # ZP
    mask[40:60, 40:60] = 2  # TE

    metrics = compute_sample_metrics(mask, mask, num_classes=5)
    assert metrics[1]["dice"] == pytest.approx(1.0, rel=1e-5)
    assert metrics[1]["iou"] == pytest.approx(1.0, rel=1e-5)
    assert metrics[2]["dice"] == pytest.approx(1.0, rel=1e-5)
    assert metrics[2]["iou"] == pytest.approx(1.0, rel=1e-5)


def test_compute_sample_metrics_disjoint():
    # 2. Khi dự đoán hoàn toàn lệch pha với nhãn chuẩn
    pred = np.zeros((100, 100), dtype=np.uint8)
    target = np.zeros((100, 100), dtype=np.uint8)
    pred[10:30, 10:30] = 1
    target[50:70, 50:70] = 1

    metrics = compute_sample_metrics(pred, target, num_classes=5)
    assert metrics[1]["dice"] < 1e-4
    assert metrics[1]["iou"] < 1e-4
