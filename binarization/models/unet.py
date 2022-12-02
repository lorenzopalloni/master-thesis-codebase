"""Implementation of a UNet model to generate super-resolution images."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from binarization.models.common import layer_generator


class UNet(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """U-Net implementation.

    Source: https://github.com/usuyama/pytorch-unet.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_filters: int = 64,
        use_residual: bool = True,
        use_batch_norm: bool = False,
        scale_factor: int = 4,
    ):
        """U-Net implementation.

        Source: https://github.com/usuyama/pytorch-unet.

        Args:
            in_channels (int, optional): Channel dimension of the input.
                Defaults to 3.
            out_channels (int, optional): Channel dimension of the output.
                Defaults to 3.
            num_filters (int, optional): Number of filters in the first hidden
                layer. Each of the following layers gets twice the number of
                filters of its previous layer during encoding phase, and half
                the number of filters of its previous layer during decoding
                phase. Defaults to 64.
            use_residual (bool): Flag for residual scheme, that concatenates
                the input to the final output. Defaults to True.
            use_batch_norm (bool): Flag for batch_normalization. Defaults to
                False.
            scale_factor (int): Scaling factor. Defaults to 4.
        """
        assert scale_factor == int(scale_factor) and scale_factor > 1

        super().__init__()

        self.use_residual = use_residual
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.dconv_down1 = layer_generator(
            in_channels, num_filters, use_batch_norm=False
        )
        self.dconv_down2 = layer_generator(
            num_filters,
            num_filters * 2,
            use_batch_norm=use_batch_norm,
        )
        self.dconv_down3 = layer_generator(
            num_filters * 2,
            num_filters * 4,
            use_batch_norm=use_batch_norm,
        )
        self.dconv_down4 = layer_generator(
            num_filters * 4,
            num_filters * 8,
            use_batch_norm=use_batch_norm,
        )

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2.0, mode='bilinear')

        self.dconv_up3 = layer_generator(
            num_filters * 8 + num_filters * 4,
            num_filters * 4,
            use_batch_norm=use_batch_norm,
        )
        self.dconv_up2 = layer_generator(
            num_filters * 4 + num_filters * 2,
            num_filters * 2,
            use_batch_norm=use_batch_norm,
        )
        self.dconv_up1 = layer_generator(
            num_filters * 2 + num_filters,
            num_filters,
            use_batch_norm=False,
        )

        self.conv_last = torch.nn.Conv2d(
            in_channels=num_filters,
            out_channels=(self.scale_factor**2) * out_channels,
            kernel_size=1,
            padding=0,
        )
        self.pixel_shuffle = torch.nn.PixelShuffle(self.scale_factor)

        self.layers = [
            self.dconv_down1,
            self.dconv_down2,
            self.dconv_down3,
            self.dconv_down4,
            self.dconv_up3,
            self.dconv_up2,
            self.dconv_up1,
        ]

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """UNet forward method."""
        x = batch

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)

        x = self.pixel_shuffle(x)

        if self.use_residual:
            x += F.interpolate(
                batch[:, -self.out_channels :, :, :],  # RGB -> BGR
                scale_factor=float(self.scale_factor),
                mode='bicubic',
            )

        return torch.clamp(x, min=-1, max=1)

    def reparametrize(self):
        for layer in self.layers:
            for block in layer:
                if hasattr(block, 'conv_adapter'):
                    block.reparametrize_convs()
