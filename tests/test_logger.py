import json
from pathlib import Path
import pytest
from tasdf.engine.logger import DualLogger


def test_dual_logger_creates_files(tmp_path: Path):
    logger = DualLogger(output_dir=tmp_path, exp_name="test_run")
    assert (tmp_path / "train.log").exists()

    logger.log_header()
    logger.log_message("Khởi động thử nghiệm")

    metrics_sample = {
        "epoch": 1,
        "total_epochs": 10,
        "train_total_loss": 0.5432,
        "train_loss_sup": 1.0864,
        "train_loss_distill": 0.0050,
        "val_mDice": 0.7850,
        "per_class_dice": {
            "Zona Pellucida (ZP)": 0.65,
            "Trophectoderm (TE)": 0.75,
            "Inner Cell Mass (ICM)": 0.50,
            "Blastocyst Cavity (BC)": 1.0,
        },
    }

    logger.log_epoch(metrics_sample)

    assert (tmp_path / "metrics.json").exists()

    # Kiểm tra tính đúng đắn của file JSON
    with open(tmp_path / "metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["epoch"] == 1
    assert data[0]["val_mDice"] == 0.7850
    assert "timestamp" in data[0]

    # Kiểm tra file train.log có chứa thông tin
    with open(tmp_path / "train.log", "r", encoding="utf-8") as f:
        log_content = f.read()

    assert "01/10" in log_content
    assert "78.50%" in log_content
