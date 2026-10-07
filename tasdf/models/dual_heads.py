import torch
import torch.nn as nn
from .itgm import ITGM



class PixelHead(nn.Module):
    """
    Pixel segmentation branch
    """

    def __init__(self, in_channels: int = 64, num_classes: int = 5):
        super(PixelHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size = 1)

    def forward(self, f:torch.Tensor) -> torch.Tensor:
        """
        f: [B, 64, H, W]
        Return P: [B, num_classes, H, W]
        """

        return self.conv(f)


class TopoHead(nn.Module):
    """
    Project F map to continous map channel and extract soft-skeleton through ITGM module
    """

    def __init__(self, in_channels:int = 64, k_iters:int = 10):
        super(TopoHead, self).__init__()
        self.proj = nn.Conv2d(in_channels, 1, kernel_size = 1)
        self.itgm = ITGM(iterations=k_iters)
    def forward(self, f:torch.Tensor) -> torch.Tensor:
        """
        f: [B, 64, H, W]
        Return soft-skeleton T: [B,1,W,G] with values [0,1]
        """

        struct_map = torch.sigmoid(self.proj(f))
        skeleton = self.itgm(struct_map)

        return skeleton

class DualHeads(nn.Module):
    """
    Combination PixelHead and TopoHead
    Gain F from UNetBackbone, return (P, T) Tuple
    """

    def __init__(self, in_channels: int = 64, num_classes: int = 5, k_iters: int = 10):
        super(DualHeads, self).__init__()
        self.pixel_head = PixelHead(in_channels, num_classes)
        self.topo_head = TopoHead(in_channels, k_iters)

    def forward(self, f:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p_logits = self.pixel_head(f)
        t_skeleton = self.topo_head(f)
        return p_logits, t_skeleton