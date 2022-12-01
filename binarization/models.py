"""Super-resolution model implementations"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Discriminator(torch.nn.Module):
    """Discriminator model for a Super-Resolution GAN framework.

    References:
    - https://arxiv.org/pdf/1609.04802.pdf
    - https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
    """

    def __init__(
        self,
        kernel_size: int = 3,
        num_channels: int = 32,
        num_blocks: int = 8,
        fc_size: int = 1024,
        input_num_channels: int = 3,
    ):
        """A series of convolutional blocks.

        The first, third, fifth (and so on) convolutional blocks increase
        the number of channels but retain image size.
        The second, fourth, sixth (and so on) convolutional blocks retain the
        same number of output channels, but halve image size.
        The first convolutional block is unique because it does not use batch
        normalization.

        Args:
            kernel_size (int, optional): Kernel size (same in all
                convolutional blocks). Defaults to 3.
            num_channels (int, optional): Number of output channels in the
                first convolutional block, then the number of output channels
                is doubled every other 2 convolutional blocks thereafter.
                Defaults to 32.
            num_blocks (int, optional): Number of convolutional blocks.
                Defaults to 8.
            fc_size (int, optional): Number of output neurons for the first
                fully connected layer. Defaults to 1024.
            input_num_channels (int, optional): Number of channels in input
                images.
        """
        super().__init__()

        conv_blocks = []
        all_out_channels = [input_num_channels]
        for i in range(num_blocks):
            in_channels = all_out_channels[-1]
            out_channels = self.out_channels_helper(
                i=i, default=all_out_channels[-1], init=num_channels
            )
            all_out_channels.append(out_channels)

            conv_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1 if i % 2 == 0 else 2,
                    use_batch_norm=bool(i),
                    activation=torch.nn.LeakyReLU(0.2),
                )
            )

        self.sequential = torch.nn.Sequential(
            *conv_blocks,
            torch.nn.AdaptiveAvgPool2d((6, 6)),
            torch.nn.Flatten(),
            torch.nn.Linear(all_out_channels[-1] * 6 * 6, fc_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(fc_size, 1),
            torch.nn.Sigmoid(),
        )

    @staticmethod
    def out_channels_helper(i: int, default: int, init: int) -> int:
        """Computes num of output channels for each ConvBlock."""
        if i == 0:
            return init
        if i % 2 == 0:
            return default * 2
        return default

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs (torch.Tensor): High-resolution or super-resolution
                images. A tensor of size (
                    num_of_images,
                    input_num_channels,
                    w * scaling_factor,
                    h * scaling_factor
                )

        Returns:
            torch.Tensor: Expected probability for each given image to
            be a high-resolution image. A tensor of size (num_of_images, 1).
        """
        return self.sequential(inputs)


class ConvBlock(torch.nn.Module):
    """A convolutional block, comprising convolutional, batch normalization,
    and activation layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        use_batch_norm: bool = False,
        activation: torch.nn.Module | None = None,
        dilation: int = 1,
        groups: int = 1,
    ):
        """Convolutional block initializer.

        Args:
            in_channels (int): num of input channels.
            out_channels (int): num of output channels.
            kernel_size (int): kernel size for each convolutional block.
            stride (int, optional): stride. Defaults to 1.
            use_batch_norm (bool, optional): flag for batch normalization.
                Defaults to False.
            activation (str | None, optional): activation function name.
                Defaults to None.
            dilation (int, optional): defaults to 1.
            groups (int, optional): defaults to 1.
        """
        super().__init__()

        layers: list[torch.nn.Module] = []

        layers.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                dilation=dilation,
            )
        )

        if use_batch_norm:
            layers.append(torch.nn.BatchNorm2d(num_features=out_channels))

        if activation:
            layers.append(activation)

        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            batch (torch.Tensor): Input images.
                A tensor or size (batch_size, in_channels, w, h).

        Returns:
            torch.Tensor: Output images.
                A tensor of size (batch_size, out_channels, w, h)
        """
        return self.sequential(batch)


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        stride: int = 1,
        kernel_size: int = 3,
        use_residual: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = use_residual
        self.is_reparametrized = False

        self.conv_adapter = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.batch_norm = (
            torch.nn.BatchNorm2d(num_features=out_channels)
            if use_batch_norm
            else torch.nn.Identity()
        )
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """UNet forward method.

        Args:
            batch (torch.Tensor): a batch of tensor images.

        Returns:
            torch.Tensor: a batch of tensor images.
        """
        out = self.conv1(batch)
        condition1 = self.use_residual and not self.is_reparametrized
        condition2 = self.in_channels == self.out_channels
        if condition1 and condition2:
            out += batch + self.conv_adapter(batch)
        out = self.batch_norm(out)
        out = self.activation(out)
        return out

    def reparametrize_convs(self):
        identity_conv = torch.nn.init.dirac_(
            torch.empty_like(self.conv1.weight)
        )
        padded_conv_adapter = F.pad(
            input=self.conv_adapter.weight,
            pad=(1, 1, 1, 1),
            mode="constant",
            value=0,
        )
        if self.in_channels == self.out_channels:
            new_conv_weights = (
                self.conv1.weight + padded_conv_adapter + identity_conv
            )
            new_conv_bias = self.conv1.bias + self.conv_adapter.bias

            self.conv1.weight.data = new_conv_weights
            self.conv1.bias.data = new_conv_bias

        self.is_reparametrized = True


def layer_generator(
    in_channels: int,
    out_channels: int,
    use_batch_norm: bool = False,
    use_residual: bool = True,
    num_blocks: int = 2,
):
    return torch.nn.Sequential(
        *(
            UNetBlock(
                in_channels=in_channels if block_id == 0 else out_channels,
                out_channels=out_channels,
                use_batch_norm=use_batch_norm,
                use_residual=use_residual,
            )
            for block_id in range(int(num_blocks))
        )
    )


class UNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_filters: int = 64,
        use_residual: bool = True,
        use_batch_norm: bool = False,
        scale_factor: int = 2,
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
            scale_factor (int): Scaling factor. Defaults to 2.
        """
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

        self.to_rgb = torch.nn.Conv2d(num_filters, 3, kernel_size=1)

        if self.scale_factor is not None:
            self.conv_last = torch.nn.Conv2d(
                in_channels=num_filters,
                out_channels=(self.scale_factor**2) * out_channels,
                kernel_size=1,
                padding=0,
            )
            self.pixel_shuffle = torch.nn.PixelShuffle(self.scale_factor)

        else:
            self.conv_last = torch.nn.Conv2d(
                num_filters, out_channels, kernel_size=1
            )
            # self.downsample = torch.nn.Upsample(
            #     scale_factor=1 / self.scale_factor,
            #     mode='bicubic',
            #     align_corners=True
            # )
        # self.downsample = torch.nn.Identity()

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

        if self.scale_factor > 1:
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


class SRUNet(torch.nn.Module):
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


# def sr_espcn(num_filters, scale_factor=2, out_channels=3, kernel_size=1):
#     return torch.nn.Sequential(
#         *(
#             torch.nn.Conv2d(
#                 kernel_size=kernel_size,
#                 in_channels=num_filters,
#                 out_channels=(scale_factor**2) * out_channels,
#                 padding=kernel_size // 2,
#             ),
#             torch.nn.PixelShuffle(scale_factor),
#         )
#     )


# class SimpleResNet(torch.nn.Module):
#     def __init__(self, num_filters, num_blocks):
#         super().__init__()
#         self.conv1 = UNetBlock(
#             in_channels=3,
#             out_channels=num_filters,
#             use_residual=True,
#             use_batch_norm=False,
#         )
#         convblock = [
#             UNetBlock(
#                 in_channels=num_filters,
#                 out_channels=num_filters,
#                 use_residual=True,
#                 use_batch_norm=False,
#             )
#             for _ in range(num_blocks - 1)
#         ]
#         self.convblocks = torch.nn.Sequential(*convblock)
#         self.sr = sr_espcn(num_filters, scale_factor=2, out_channels=3)
#         self.upscale = torch.nn.Upsample(
#             scale_factor=2, mode='bicubic', align_corners=True
#         )
#         self.clip = torch.nn.Hardtanh()

#     def forward(self, input):
#         x = self.conv1(input)
#         x = self.convblocks(x)
#         x = self.sr(x)

#         return self.clip(x + self.upscale(input))

#     def reparametrize(self):
#         for block in self.convblocks:
#             if hasattr(block, 'conv_adapter'):
#                 block.reparametrize_convs()
