import torch
import pytest
from tasdf.models.unet_backbone import UNetBackbone


def test_unet_backbone_shape():
    """Kiểm tra kích thước tensor đầu ra của UNetBackbone."""
    model = UNetBackbone(in_channels=1, base_channels=64)
    model.eval()

    # Thử nghiệm với Batch = 2 ảnh kích thước chuẩn 256x256
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out = model(x)

    # Đầu ra phải giữ nguyên kích thước không gian 256x256 và có 64 kênh đặc trưng
    assert out.shape == (2, 64, 256, 256), f"Kích thước sai lệch: {out.shape}"


def test_unet_backbone_gradient_flow():
    """Kiểm tra dòng chảy gradient đảm bảo không tầng nào bị triệt tiêu đạo hàm."""
    model = UNetBackbone(in_channels=1, base_channels=64)
    model.train()

    x = torch.randn(1, 1, 64, 64)  # Kích thước nhỏ để test nhanh
    out = model(x)

    # Giả lập hàm loss đơn giản
    loss = out.sum()
    loss.backward()

    # Kiểm tra xem tầng đầu tiên (inc) có nhận được gradient hay không
    first_conv_weight = model.inc.double_conv[0].weight
    assert first_conv_weight.grad is not None, "Gradient không truyền về được tầng đầu tiên!"
    assert torch.any(first_conv_weight.grad != 0), "Gradient bị triệt tiêu bằng 0!"