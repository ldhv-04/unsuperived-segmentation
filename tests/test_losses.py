import torch
import pytest
from tasdf.losses.losses import sigmoid_rampup, MainLoss




def test_sigmoid_rampup():
    """Test increase ability of Sigmoid Rampup"""

    w_0 = sigmoid_rampup(current_epoch=0, max_epochs=100, alpha=1.0)
    w_50 = sigmoid_rampup(current_epoch=50, max_epochs= 100, alpha=1.0)
    w_100 = sigmoid_rampup(current_epoch=100, max_epochs= 100, alpha= 1.0)

    assert w_0 < 0.01, f"w(0) too large: {w_0}"
    assert w_0 < w_50 < w_100
    assert pytest.approx(w_100, rel=1e-3) == 1.0

def test_loss_values_and_gradient_flow():
    """Test gradient backprop througt 3 loss component"""

    criterion = MainLoss(lambda_topo=0.10, gamma= 0.5, alpha=1.0, max_epochs=100)

    # Labeled Data
    p1_l = torch.randn(2,5,32,32, requires_grad = True)
    p2_l = torch.randn(2,5,32,32, requires_grad = True)
    targets_l = torch.randint(0,5, (2,32,32), dtype=torch.long)

    p1_u = torch.randn(2,5,32,32, requires_grad = True)
    p2_u = torch.randn(2,5,32,32, requires_grad = True)
    t1_u = torch.rand(2,1,32,32, requires_grad = True)
    t2_u = torch.rand(2,1,32,32, requires_grad = True)

    #Calculate at epoch 50
    total_loss, metrics = criterion(
        p1_l, p2_l, targets_l, 
        p1_u, p2_u, t1_u, t2_u,
        current_epoch = 50,
    )

    assert total_loss.item() > 0, "loss must be positive value"
    assert "loss_sup" in metrics and "loss_pc" in metrics and "loss_tc" in metrics

    #Gradient backprop

    total_loss.backward()

    for tensor, name in [
        (p1_l, "p1_l"), (p2_l,"p2_l"),
        (p1_u, "p1_u"), (p2_u, "p2_u"),
        (t1_u, "t1_u"), (t2_u, "t2_u"),
    ]:
        assert tensor.grad is not None, f"Grad backprop not return to {name}"
        assert torch.any(tensor.grad != 0), f"The gradient at {name} vanishes to zero."
    