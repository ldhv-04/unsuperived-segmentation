import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_rampup(current_epoch: int, max_epochs: int, alpha: float = 1.0) -> float:
    """Hàm điều phối trọng số động Sigmoid Ramp-up."""
    if max_epochs <= 0:
        return alpha
    current = np.clip(current_epoch, 0, max_epochs)
    phase = 1.0 - (current / max_epochs)
    return float(alpha * np.exp(-5.0 * phase * phase))


class MainLoss(nn.Module):
    """
    Hàm mất mát toàn thể TASDF:
    Kết hợp tổn thất giám sát (CE), chưng cất xác suất pixel và chưng cất tô-pô ITGM.
    """

    def __init__(
        self,
        lambda_topo: float = 0.10,
        gamma: float = 0.5,
        alpha: float = 1.0,
        max_epochs: int = 200,
    ):
        super(MainLoss, self).__init__()
        self.lambda_topo = lambda_topo
        self.gamma = gamma
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        p1_labeled: torch.Tensor,
        p2_labeled: torch.Tensor,
        targets_labeled: torch.Tensor,
        p1_unlabeled: torch.Tensor,
        p2_unlabeled: torch.Tensor,
        t1_unlabeled: torch.Tensor,
        t2_unlabeled: torch.Tensor,
        current_epoch: int,
    ) -> tuple[torch.Tensor, dict]:
        # 1. Tổn thất có giám sát (Supervised Loss) trên cả 2 nhánh mạng
        loss_sup1 = self.ce_loss(p1_labeled, targets_labeled)
        loss_sup2 = self.ce_loss(p2_labeled, targets_labeled)
        loss_sup = loss_sup1 + loss_sup2

        # 2. Chưng cất xác suất pixel trên ảnh không nhãn (KL / MSE)
        prob1_unlabeled = F.softmax(p1_unlabeled, dim=1)
        prob2_unlabeled = F.softmax(p2_unlabeled, dim=1)
        loss_pc = F.mse_loss(prob1_unlabeled, prob2_unlabeled)

        # 3. Chưng cất cấu trúc tô-pô ITGM trên ảnh không nhãn
        loss_tc = F.mse_loss(t1_unlabeled, t2_unlabeled)

        # 4. Trọng số động w(t) và tổn thất chưng cất tự thân
        w_t = sigmoid_rampup(current_epoch, self.max_epochs, self.alpha)
        loss_distill = w_t * (loss_pc + self.lambda_topo * loss_tc)

        total_loss = self.gamma * loss_sup + (1.0 - self.gamma) * loss_distill

        metrics = {
            "total_loss": total_loss.item(),
            "loss_sup": loss_sup.item(),
            "loss_pc": loss_pc.item(),
            "loss_tc": loss_tc.item(),
            "w_t": w_t,
        }
        return total_loss, metrics
