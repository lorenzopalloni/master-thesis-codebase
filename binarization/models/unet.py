"""UNet model to generate super-resolution images."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from binarization.models.modeltools import generate_unet_block_sequence


class UNet(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """U-Net for super-resolution.

    Some references:
        Implementation based on:
            - https://github.com/usuyama/pytorch-unet
            - https://github.com/fede-vaccaro/fast-sr-unet

        PixelShuffle: https://arxiv.org/abs/1609.05158.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_filters: int = 64,
        use_batch_norm: bool = False,
        scale_factor: int = 4,
    ):
        """U-Net.

        Args:
            in_channels (int, optional): channel dimension of the input.
                Defaults to 3.
            out_channels (int, optional): channel dimension of the output.
                Defaults to 3.
            num_filters (int, optional): number of filters in the first hidden
                layer. Each of the following layers gets twice the number of
                filters of its previous layer during encoding phase, and half
                the number of filters of its previous layer during decoding
                phase. Defaults to 64.
            use_batch_norm (bool): flag for batch normalization. Defaults to
                False.
            scale_factor (int): scaling factor. Defaults to 4.
        """
        assert scale_factor == int(scale_factor) and scale_factor > 1

        super().__init__()

        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.down1 = generate_unet_block_sequence(
            in_channels=in_channels,
            out_channels=num_filters,
            use_batch_norm=False,
        )
        self.down2 = generate_unet_block_sequence(
            in_channels=num_filters,
            out_channels=num_filters * 2,
            use_batch_norm=use_batch_norm,
        )
        self.down3 = generate_unet_block_sequence(
            in_channels=num_filters * 2,
            out_channels=num_filters * 4,
            use_batch_norm=use_batch_norm,
        )
        self.down4 = generate_unet_block_sequence(
            in_channels=num_filters * 4,
            out_channels=num_filters * 8,
            use_batch_norm=use_batch_norm,
        )

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2.0, mode='bilinear')

        self.up4 = generate_unet_block_sequence(
            in_channels=num_filters * 8 + num_filters * 4,
            out_channels=num_filters * 4,
            use_batch_norm=use_batch_norm,
        )
        self.up3 = generate_unet_block_sequence(
            in_channels=num_filters * 4 + num_filters * 2,
            out_channels=num_filters * 2,
            use_batch_norm=use_batch_norm,
        )
        self.up2 = generate_unet_block_sequence(
            in_channels=num_filters * 2 + num_filters,
            out_channels=num_filters,
            use_batch_norm=False,
        )
        self.up1 = torch.nn.Conv2d(
            in_channels=num_filters,
            out_channels=(self.scale_factor**2) * out_channels,
            kernel_size=1,
            padding=0,
        )
        self.pixel_shuffle = torch.nn.PixelShuffle(self.scale_factor)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """UNet forward method."""
        out = batch

        out_down1 = self.down1(out)
        out = self.maxpool(out_down1)

        out_down2 = self.down2(out)
        out = self.maxpool(out_down2)

        out_down3 = self.down3(out)
        out = self.maxpool(out_down3)

        out = self.down4(out)

        out = self.upsample(out)
        out = torch.cat([out, out_down3], dim=1)

        out = self.up4(out)
        out = self.upsample(out)
        out = torch.cat([out, out_down2], dim=1)

        out = self.up3(out)
        out = self.upsample(out)
        out = torch.cat([out, out_down1], dim=1)

        out = self.up2(out)

        out = self.up1(out)

        out = self.pixel_shuffle(out)

        out += F.interpolate(
            batch,
            scale_factor=float(self.scale_factor),
            mode='bilinear',
        )

        return torch.clamp(out, min=0, max=1)
