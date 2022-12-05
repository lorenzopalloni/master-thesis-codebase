"""Implementation of a SRUNet model to generate super-resolution images."""

from __future__ import annotations

import torch

from binarization.models.common import generate_unet_block_sequence


class SRUNet(torch.nn.Module):
    """SRUNet (Super-Resolution UNet)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_filters=64,
        downsample=None,
        batchnorm=False,
        scale_factor=2,
        layer_multiplier=1,
    ):
        """SRUNet (Super-Resolution UNet).

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
                downsample (None or float, optional)
                    can be used for downscaling the output. e.g., if you use downsample=0.5 the output resolution will be halved
                scale_factor (int):
                    upscale factor. if you want a rational upscale (e.g. 720p to 1080p, which is 1.5) combine it
                    with the downsample parameter
                layer_multiplier (int or float):
                    compress or extend the network depth in terms of total layers. configured as a multiplier to the number of the
                    basic blocks which composes the layers
                batchnorm (bool, default=False):
                    whether use batchnorm or not. If True should decrease quality and performances.
            """

        super().__init__()

        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.down1 = generate_unet_block_sequence(
            in_channels=in_channels,
            out_channels=num_filters // 2,
            use_batch_norm=False,
            num_blocks=2 * layer_multiplier,
        )
        self.down2 = generate_unet_block_sequence(
            in_channels=num_filters // 2,
            out_channels=num_filters,
            use_batch_norm=batchnorm,
            num_blocks=3 * layer_multiplier,
        )
        self.down3 = generate_unet_block_sequence(
            in_channels=num_filters,
            out_channels=num_filters,
            use_batch_norm=batchnorm,
            num_blocks=3 * layer_multiplier,
        )
        self.down4 = generate_unet_block_sequence(
            in_channels=num_filters,
            out_channels=num_filters,
            use_batch_norm=batchnorm,
            num_blocks=3 * layer_multiplier,
        )

        self.maxpool = torch.nn.MaxPool2d(2)
        if downsample is not None and downsample != 1.0:
            self.downsample = torch.nn.Upsample(
                scale_factor=downsample, mode='bicubic', align_corners=True
            )
        else:
            self.downsample = torch.nn.Identity()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        self.up4 = generate_unet_block_sequence(
            in_channels=num_filters + num_filters,
            out_channels=num_filters,
            use_batch_norm=batchnorm,
            num_blocks=3 * layer_multiplier,
        )
        self.up3 = generate_unet_block_sequence(
            in_channels=num_filters + num_filters,
            out_channels=num_filters,
            use_batch_norm=batchnorm,
            num_blocks=3 * layer_multiplier,
        )
        self.up2 = generate_unet_block_sequence(
            in_channels=num_filters + num_filters // 2,
            out_channels=num_filters // 2,
            use_batch_norm=False,
            num_blocks=3 * layer_multiplier,
        )
        self.up1 = torch.nn.Conv2d(
            in_channels=num_filters // 2,
            out_channels=(self.scale_factor ** 2) * out_channels,
            kernel_size=1,
            padding=0,
        )
        self.pixel_shuffle = torch.nn.PixelShuffle(self.scale_factor)

    def forward(self, batch):
        out = batch

        conv1 = self.down1(out)
        out = self.maxpool(conv1)

        conv2 = self.down2(out)
        out = self.maxpool(conv2)

        conv3 = self.down3(out)
        out = self.maxpool(conv3)

        out = self.down4(out)

        out = self.upsample(out)
        out = torch.cat([out, conv3], dim=1)

        out = self.up4(out)
        out = self.upsample(out)
        out = torch.cat([out, conv2], dim=1)

        out = self.up3(out)
        out = self.upsample(out)
        out = torch.cat([out, conv1], dim=1)

        out = self.up2(out)

        out = self.conv_last(out)

        sf = self.scale_factor

        out = self.pixel_shuffle(out)

        out += torch.nn.functional.interpolate(
            batch[:, -self.out_channels :, :, :],
            scale_factor=sf,
            mode='bicubic',
        )

        return torch.clamp(self.downsample(out), min=-1, max=1)
