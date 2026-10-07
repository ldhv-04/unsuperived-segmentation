import torch
import pytest
from tasdf.models.final_model import TASDFSubNet, TASDFNet


def test_tasdf_subnet_shape():
    """Kiểm tra kích thước đầu ra của một nhánh mạng con TASDFSubNet."""
    subnet = TASDFSubNet(in_channels=1, base_channels=16, num_classes=5, k_iters=3)
    subnet.eval()
    
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        p, t = subnet(x)
        
    assert p.shape == (2, 5, 64, 64), f"Shape Logits sai: {p.shape}"
    assert t.shape == (2, 1, 64, 64), f"Shape Skeleton sai: {t.shape}"
    assert t.min() >= 0.0 and t.max() <= 1.0, "Giá trị Skeleton vượt ngưỡng [0, 1]!"


def test_tasdf_non_shared_weights():
    """Đảm bảo net1 và net2 là hai thực thể độc lập trọng số (Non-Shared Weights)."""
    model = TASDFNet(in_channels=1, base_channels=16, num_classes=5, k_iters=3)
    
    # Kiểm tra con trỏ vùng nhớ (memory pointer) không được trùng nhau
    weight1 = model.net1.backbone.inc.double_conv[0].weight
    weight2 = model.net2.backbone.inc.double_conv[0].weight
    
    assert weight1.data_ptr() != weight2.data_ptr(), "LỖI: Trọng số của 2 mạng bị chia sẻ (Shared Memory)!"


def test_tasdf_forward_modes():
    """Kiểm tra hoạt động của cả 4 chế độ forward: both, net1, net2, ensemble."""
    model = TASDFNet(in_channels=1, base_channels=16, num_classes=5, k_iters=3)
    model.eval()
    x = torch.randn(1, 1, 64, 64)

    with torch.no_grad():
        # Mode 1: both
        (p1, t1), (p2, t2) = model(x, branch="both")
        assert p1.shape == (1, 5, 64, 64) and p2.shape == (1, 5, 64, 64)

        # Mode 2: net1
        p_n1, t_n1 = model(x, branch="net1")
        assert p_n1.shape == (1, 5, 64, 64)

        # Mode 3: net2
        p_n2, t_n2 = model(x, branch="net2")
        assert p_n2.shape == (1, 5, 64, 64)

        # Mode 4: ensemble
        p_ens, t_ens = model(x, branch="ensemble")
        assert p_ens.shape == (1, 5, 64, 64)


def test_tasdf_end_to_end_gradients():
    """Kiểm tra truyền ngược gradient toàn diện cho cả 2 mạng song sinh."""
    model = TASDFNet(in_channels=1, base_channels=16, num_classes=5, k_iters=3)
    model.train()

    x = torch.randn(1, 1, 64, 64)
    (p1, t1), (p2, t2) = model(x, branch="both")

    # Giả lập hàm mất mát tổng hợp từ cả 2 mạng
    loss = p1.sum() + t1.sum() + p2.sum() + t2.sum()
    loss.backward()

    # Kiểm tra gradient đã truyền đến tận tầng đầu của net1 và net2
    grad1 = model.net1.backbone.inc.double_conv[0].weight.grad
    grad2 = model.net2.backbone.inc.double_conv[0].weight.grad

    assert grad1 is not None and torch.any(grad1 != 0), "Gradient không truyền về net1!"
    assert grad2 is not None and torch.any(grad2 != 0), "Gradient không truyền về net2!"