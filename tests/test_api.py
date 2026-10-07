import json
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch

from tasdf import TASDF


@pytest.fixture
def api_synthetic_coco(tmp_path: Path) -> Path:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "valid"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    for i in range(4):
        img = Image.fromarray(np.uint8(np.random.rand(64, 64) * 255))
        img.save(train_dir / f"embryo_{i}.png")
    for i in range(2):
        img = Image.fromarray(np.uint8(np.random.rand(64, 64) * 255))
        img.save(val_dir / f"val_embryo_{i}.png")

    coco_data = {
        "images": [{"id": i, "file_name": f"embryo_{i}.png", "width": 64, "height": 64} for i in range(4)],
        "categories": [
            {"id": 1, "name": "ZP"},
            {"id": 2, "name": "TE"},
            {"id": 3, "name": "ICM"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
            }
        ],
    }
    with open(train_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
        json.dump(coco_data, f)

    val_data = coco_data.copy()
    val_data["images"] = [{"id": i, "file_name": f"val_embryo_{i}.png", "width": 64, "height": 64} for i in range(2)]
    with open(val_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f)

    return tmp_path


def test_api_workflow_jupyter_style(api_synthetic_coco: Path, tmp_path: Path):
    data_dict = {
        "format": "coco",
        "path": str(api_synthetic_coco),
        "train": "train",
        "val": "valid",
        "names": {1: "ZP", 2: "TE", 3: "ICM"},
    }

    # 1. Khởi tạo mô hình
    model = TASDF(model={"model": {"base_channels": 16, "k_iters": 2}})
    assert model.model is not None

    # 2. Huấn luyện 1 epoch với API gọn gàng
    runs_dir = tmp_path / "runs"
    results = model.train(
        data=data_dict,
        epochs=1,
        batch=2,
        imgsz=32,
        num_workers=0,
        project=str(runs_dir),
        name="test_exp",
    )

    exp_dir = Path(results["exp_dir"])
    assert exp_dir.exists()
    assert (exp_dir / "train.log").exists()
    assert (exp_dir / "metrics.json").exists()
    assert (exp_dir / "checkpoints" / "latest.pt").exists()

    # 3. Đánh giá (Val)
    val_res = model.val(data=data_dict, imgsz=32, batch=2)
    assert "mDice" in val_res
    assert "mIoU" in val_res

    # Kiểm tra log file đã ghi nhận bảng đo lường tham số trước epoch 1
    log_content = (exp_dir / "train.log").read_text(encoding="utf-8")
    assert "MODEL PARAMETER AUDIT" in log_content
    assert "net1" in log_content
    assert "net2" in log_content

    # 4. Suy luận trực quan (Predict)
    dummy_img = api_synthetic_coco / "train" / "embryo_0.png"
    preds = model.predict(source=dummy_img, imgsz=32, save=True, output_dir=runs_dir / "preds")
    assert len(preds) == 1
    assert "mask" in preds[0]
    assert Path(preds[0]["saved_paths"]["composite"]).exists()


def test_model_parameter_audit():
    """Kiểm tra độ chính xác của hàm đo lường và hiển thị tham số mô hình."""
    from tasdf.utils import get_model_parameter_stats, format_parameter_summary

    model = TASDF(model={"model": {"base_channels": 16, "k_iters": 2}})
    stats = model.info(detailed=False)

    assert isinstance(stats, dict)
    assert stats["total_params"] > 0
    assert stats["trainable_params"] == stats["total_params"]
    assert stats["non_trainable_params"] == 0
    assert stats["trainable_ratio"] == 100.0
    assert stats["size_mb"] > 0.0

    assert "net1" in stats["submodules"]
    assert "net2" in stats["submodules"]
    assert stats["submodules"]["net1"]["total_params"] == stats["submodules"]["net2"]["total_params"]
    assert stats["submodules"]["net1"]["total_params"] * 2 == stats["total_params"]

    table_str = format_parameter_summary(stats)
    assert "MODEL PARAMETER AUDIT" in table_str
    assert "TỔNG CỘNG (TOTAL)" in table_str
    assert "net1" in table_str

