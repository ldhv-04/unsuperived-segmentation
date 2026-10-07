import torch
import pytest
from tasdf.models.dual_heads import PixelHead, TopoHead, DualHeads


def test_pixel_head_shape():
    """Kiểm tra kích thước đầu ra của PixelHead."""
    head = PixelHead(in_channels=64, num_classes=5)
    f = torch.randn(2, 64, 128, 128)
    p = head(f)
    assert p.shape == (2, 5, 128, 128), f"PixelHead shape sai: {p.shape}"


def test_topo_head_shape_and_range():
    """Kiểm tra kích thước và miền giá trị [0, 1] của TopoHead."""
    head = TopoHead(in_channels=64, k_iters=5)
    f = torch.randn(2, 64, 128, 128)
    t = head(f)
    
    assert t.shape == (2, 1, 128, 128), f"TopoHead shape sai: {t.shape}"
    assert t.min() >= 0.0 and t.max() <= 1.0, "Giá trị khung xương vượt quá [0, 1]!"


def test_dual_heads_gradient_flow():
    """Kiểm tra cả 2 nhánh đều có thể nhận gradient truyền ngược về Feature Map."""
    dual_heads = DualHeads(in_channels=64, num_classes=5, k_iters=5)
    dual_heads.train()

    f = torch.randn(1, 64, 64, 64, requires_grad=True)
    p, t = dual_heads(f)

    # Giả lập hàm mất mát tổng hợp từ cả 2 nhánh
    loss = p.sum() + t.sum()
    loss.backward()

    # Kiểm tra gradient đã truyền tới các trọng số của cả 2 head
    assert dual_heads.pixel_head.conv.weight.grad is not None
    assert dual_heads.topo_head.proj.weight.grad is not None
    assert f.grad is not None, "Gradient không truyền ngược về được Feature Map F!"
    assert torch.any(f.grad != 0), "Gradient truyền về F bị triệt tiêu bằng 0!"