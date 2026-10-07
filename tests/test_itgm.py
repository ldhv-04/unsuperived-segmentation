import pytest
import torch

from tasdf.models.itgm import ITGM


def test_itgm_shape_and_range():

    model = ITGM(iterations = 10, kernel_size = 3)

    x = torch.rand(2,1,64,64)

    output = model(x)

    assert output.shape == (2,1,64,64), f"Shape false! Output{output.shape}"

    assert output.min() >= 0.0, f"Negative soft skeleton values"
    assert output.max() <= 1.0 + 1e-6, f"Value over 1.0"

def test_itgm_gradient_flow():
    """Check gradient can backprop through minpool/maxpool"""

    model = ITGM(iterations = 1, kernel_size = 3)

    x = torch.tensor(
        [[[
            [1.,2.,3.,4.],
            [5.,6.,7.,8.],
            [9.,0.,1.,2.],
            [4.,5.,6.,7.]

        ]]], 
        requires_grad=True)

    output = model(x)

    loss = output.sum()
    loss.backward()


    assert x.grad is not None, f"Gradient does not backprop"
    assert not torch.isnan(x.grad).any(), f"Gradient get NaN"
    assert x.grad.abs().sum() > 0, (
        "All values are 0"
    )
