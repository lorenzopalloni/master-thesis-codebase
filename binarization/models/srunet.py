"""Implementation of a SRUNet model to generate super-resolution images."""

from __future__ import annotations

import torch

from binarization.models.common import layer_generator


class SRUNet(torch.nn.Module):
    """SRUNet (Super-Resolution UNet)"""

    def __init__(
        self,
        in_dim=3,
        num_class=3,
        downsample=None,
        residual=False,
        batchnorm=False,
        scale_factor=2,
        num_filters=64,
        layer_multiplier=1,
    ):
        """
        Args:
            in_dim (float, optional):
                channel dimension of the input
            num_class (str):
                channel dimension of the output
            num_filters (int, optional):
                maximum number of filters. the layers start with num_filters / 2,  after each layer this number gets multiplied by 2
                 during the encoding stage and until it reaches num_filters. During the decoding stage the number follows the reverse
                 scheme. Default is 64
            downsample (None or float, optional)
                can be used for downscaling the output. e.g., if you use downsample=0.5 the output resolution will be halved
            residual (bool):
                if using the residual scheme and adding the input to the final output
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

        self.residual = residual
        self.num_class = num_class
        self.scale_factor = scale_factor

        self.dconv_down1 = layer_generator(
            in_dim,
            num_filters // 2,
            use_batch_norm=False,
            num_blocks=2 * layer_multiplier,
        )
        self.dconv_down2 = layer_generator(
            num_filters // 2,
            num_filters,
            use_batch_norm=batchnorm,
            num_blocks=3 * layer_multiplier,
        )
        self.dconv_down3 = layer_generator(
            num_filters,
            num_filters,
            use_batch_norm=batchnorm,
            num_blocks=3 * layer_multiplier,
        )
        self.dconv_down4 = layer_generator(
            num_filters,
            num_filters,
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

        self.dconv_up3 = layer_generator(
            num_filters + num_filters,
            num_filters,
            use_batch_norm=batchnorm,
            num_blocks=3 * layer_multiplier,
        )
        self.dconv_up2 = layer_generator(
            num_filters + num_filters,
            num_filters,
            use_batch_norm=batchnorm,
            num_blocks=3 * layer_multiplier,
        )
        self.dconv_up1 = layer_generator(
            num_filters + num_filters // 2,
            num_filters // 2,
            use_batch_norm=False,
            num_blocks=3 * layer_multiplier,
        )

        self.layers = [
            self.dconv_down1,
            self.dconv_down2,
            self.dconv_down3,
            self.dconv_down4,
            self.dconv_up3,
            self.dconv_up2,
            self.dconv_up1,
        ]

        sf = self.scale_factor

        self.to_rgb = torch.nn.Conv2d(num_filters // 2, 3, kernel_size=1)
        if sf > 1:
            self.conv_last = torch.nn.Conv2d(
                num_filters // 2,
                (sf**2) * num_class,
                kernel_size=1,
                padding=0,
            )
            self.pixel_shuffle = torch.nn.PixelShuffle(sf)
        else:
            self.conv_last = torch.nn.Conv2d(
                num_filters // 2, 3, kernel_size=1
            )

    def forward(self, input):
        x = input

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

        sf = self.scale_factor

        if sf > 1:
            x = self.pixel_shuffle(x)

        # x = self.to_rgb(x)
        if self.residual:
            sf = (
                self.scale_factor
            )  # (self.scale_factor // (2 if self.use_s2d and self.scale_factor > 1 else 1))
            x += torch.nn.functional.interpolate(
                input[:, -self.num_class :, :, :],
                scale_factor=sf,
                mode='bicubic',
            )
            x = torch.clamp(x, min=-1, max=1)

        return torch.clamp(
            self.downsample(x), min=-1, max=1
        )  # self.downsample(x)

    def reparametrize(self):
        for layer in self.layers:
            for block in layer:
                if hasattr(block, 'conv_adapter'):
                    block.reparametrize_convs()
