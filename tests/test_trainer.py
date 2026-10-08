from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tasdf.models import TASDFNet
from tasdf.engine.trainer import Trainer
from tasdf.engine.logger import DualLogger


@pytest.fixture
def dummy_model():
    return TASDFNet(in_channels=1, base_channels=16, num_classes=5, k_iters=2)


@pytest.fixture
def dummy_loaders():
    x_l = torch.rand(4, 1, 32, 32)
    y_l = torch.randint(0, 5, (4, 32, 32))
    x_u = torch.rand(4, 1, 32, 32)

    l_ds = TensorDataset(x_l, y_l)
    u_ds = TensorDataset(x_u)

    labeled_loader = DataLoader(l_ds, batch_size=2)
    # unlabeled loader yields x_u (tensor directly)
    unlabeled_loader = DataLoader(x_u, batch_size=2)
    return labeled_loader, unlabeled_loader


def test_trainer_one_epoch_and_checkpoint(dummy_model, dummy_loaders, tmp_path: Path):
    l_loader, u_loader = dummy_loaders
    cfg = {
        "training": {
            "device": "cpu",
            "amp": False,
            "learning_rate": 1e-3,
            "accum_steps": 1,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
        "loss": {"lambda_topo": 0.1, "gamma": 0.5, "alpha": 1.0, "max_epochs": 2},
    }

    logger = DualLogger(output_dir=tmp_path / "logs")
    trainer = Trainer(dummy_model, cfg, logger=logger, checkpoint_dir=tmp_path / "checkpoints")

    res = trainer.train_one_epoch(l_loader, u_loader, epoch=1)
    assert res["total_loss"] > 0.0

    # Test lưu checkpoint
    trainer.save_checkpoint(epoch=1, is_best=True)
    assert (tmp_path / "checkpoints" / "checkpoint_latest.pt").exists()
    assert (tmp_path / "checkpoints" / "checkpoint_best.pt").exists()
    assert (tmp_path / "checkpoints" / "best.pt").exists()
    assert (tmp_path / "checkpoints" / "last.pt").exists()

    # Test nạp lại checkpoint (Resume)
    new_trainer = Trainer(dummy_model, cfg, logger=logger, checkpoint_dir=tmp_path / "checkpoints")
    new_trainer.load_checkpoint(tmp_path / "checkpoints" / "last.pt", resume=True)
    assert new_trainer.start_epoch == 2


def test_trainer_exports_best_and_last_pt(dummy_model, dummy_loaders, tmp_path: Path):
    """Kiểm tra chu trình huấn luyện luôn tạo ra chính xác best.pt và last.pt."""
    l_loader, u_loader = dummy_loaders
    root_ckpt = tmp_path / "global_checkpoints"
    cfg = {
        "training": {
            "device": "cpu",
            "amp": False,
            "learning_rate": 1e-3,
            "accum_steps": 1,
            "epochs": 2,
            "checkpoint_dir": str(root_ckpt),
        },
        "loss": {"lambda_topo": 0.1, "gamma": 0.5, "alpha": 1.0, "max_epochs": 2},
    }

    local_ckpt = tmp_path / "runs" / "exp" / "weights"
    logger = DualLogger(output_dir=tmp_path / "runs" / "exp" / "logs")
    trainer = Trainer(dummy_model, cfg, logger=logger, checkpoint_dir=local_ckpt)

    results = trainer.train(l_loader, u_loader, epochs=2)

    # 1. Kiểm tra file tại thư mục cục bộ của phiên chạy
    assert (local_ckpt / "best.pt").exists()
    assert (local_ckpt / "last.pt").exists()

    # 2. Kiểm tra file được đồng bộ sang thư mục checkpoints cấu hình toàn cục
    assert (root_ckpt / "best.pt").exists()
    assert (root_ckpt / "last.pt").exists()

    # 3. Kiểm tra kết quả trả về trong từ điển kết quả
    assert results["best_checkpoint"] == str(local_ckpt / "best.pt")
    assert results["last_checkpoint"] == str(local_ckpt / "last.pt")

