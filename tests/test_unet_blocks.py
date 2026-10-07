import pytest
from tasdf.models.unet_blocks import DoubleConv, Down, Up
import torch


def test_double_conv_shape():

    block = DoubleConv(in_channels=1, out_channels=64)
    #x: batch 2, channels 1, H, W
    x = torch.rand(2, 1, 256, 256)
    out = block(x)


    assert out.shape == (2,64,256,256), f"Shape Error {out.shape}"

def test_douwn_block_shape_and_skip_connection():
    up_block = Up(in_channels=256, out_channels=128)

    x1 = torch.rand(2,256,64,64)
    x2 = torch.rand(2,128,128,128)

    out = up_block(x1,x2)

    assert out.shape == (2,128,128,128), f"Shape error, {out.shape}"