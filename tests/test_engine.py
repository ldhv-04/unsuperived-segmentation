from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tasdf.models import TASDFNet
from tasdf.engine.validator import Validator
from tasdf.engine.predictor import Predictor


@pytest.fixture
def dummy_model():
    return TASDFNet(in_channels=1, base_channels=16, num_classes=5, k_iters=2)


def test_validator_metrics(dummy_model):
    x = torch.rand(2, 1, 32, 32)
    y = torch.randint(0, 5, (2, 32, 32))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    validator = Validator(loader, device=torch.device("cpu"), num_classes=5)
    res = validator.validate(dummy_model)

    assert "mDice" in res
    assert "mIoU" in res
    assert 0.0 <= res["mDice"] <= 1.0


def test_predictor_single_image(dummy_model, tmp_path: Path):
    # Tạo 1 ảnh phôi giả lập
    img_p = tmp_path / "test_embryo.png"
    Image.fromarray(np.uint8(np.random.rand(64, 64) * 255)).save(img_p)

    predictor = Predictor(image_size=32, device=torch.device("cpu"))
    results = predictor.predict(dummy_model, source=img_p, output_dir=tmp_path / "preds", save=True)

    assert len(results) == 1
    assert results[0]["mask"].shape == (64, 64)
    assert Path(results[0]["saved_paths"]["composite"]).exists()
