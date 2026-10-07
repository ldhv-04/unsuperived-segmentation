import torch
import torch.nn as nn
from .unet_backbone import UNetBackbone
from .dual_heads import DualHeads


class TASDFSubNet(nn.Module):
    """
    Một nhánh mạng con đơn lẻ N_i trong mô hình TASDF.
    Gồm: UNetBackbone trích xuất đặc trưng + DualHeads (PixelHead & TopoHead).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 5,
        k_iters: int = 10,
    ):
        super(TASDFSubNet, self).__init__()
        self.backbone = UNetBackbone(
            in_channels=in_channels, base_channels=base_channels
        )
        self.heads = DualHeads(
            in_channels=base_channels,
            num_classes=num_classes,
            k_iters=k_iters,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, 1, H, W]
        Trả về: (p_logits, t_skeleton)
          - p_logits: [B, C, H, W]
          - t_skeleton: [B, 1, H, W]
        """
        f = self.backbone(x)
        p, t = self.heads(f)
        return p, t


class TASDFNet(nn.Module):
    """
    Mô hình hoàn chỉnh TASDF (Topology-Aware Semi-Supervised Self-Distillation Framework).
    Chứa 2 mạng song sinh Non-Shared Weights (net1 và net2).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 5,
        k_iters: int = 10,
    ):
        super(TASDFNet, self).__init__()
        # Khởi tạo 2 mạng độc lập trọng số (Non-Shared Weights)
        self.net1 = TASDFSubNet(in_channels, base_channels, num_classes, k_iters)
        self.net2 = TASDFSubNet(in_channels, base_channels, num_classes, k_iters)

    def forward(
        self, x: torch.Tensor, branch: str = "both"
    ):
        """
        x: [B, 1, H, W]
        branch:
          - 'both': Dùng khi huấn luyện bán giám sát, trả về ((p1, t1), (p2, t2))
          - 'net1': Dùng khi inference / validation để tiết kiệm 50% tính toán
          - 'net2': Suy luận bằng mạng net2
          - 'ensemble': Lấy trung bình dự đoán pixel xác suất của cả 2 mạng
        """
        if branch == "net1":
            return self.net1(x)
        elif branch == "net2":
            return self.net2(x)
        elif branch == "both":
            p1, t1 = self.net1(x)
            p2, t2 = self.net2(x)
            return (p1, t1), (p2, t2)
        elif branch == "ensemble":
            p1, t1 = self.net1(x)
            p2, t2 = self.net2(x)
            p_mean = (p1 + p2) / 2.0
            t_mean = (t1 + t2) / 2.0
            return p_mean, t_mean
        else:
            raise ValueError(
                f"Giá trị branch không hợp lệ: {branch}. Chọn 'both', 'net1', 'net2' hoặc 'ensemble'."
            )