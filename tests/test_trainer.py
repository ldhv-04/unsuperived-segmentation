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

    # Test nạp lại checkpoint (Resume)
    new_trainer = Trainer(dummy_model, cfg, logger=logger, checkpoint_dir=tmp_path / "checkpoints")
    new_trainer.load_checkpoint(tmp_path / "checkpoints" / "checkpoint_latest.pt", resume=True)
    assert new_trainer.start_epoch == 2
