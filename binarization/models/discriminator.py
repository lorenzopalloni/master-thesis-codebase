"""Implementation of a discriminator model that should discriminate if an
image is original (high-resolution) or generated (super-resolution).
"""

from __future__ import annotations

import torch


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

        The first, third, fifth (and so on) convolutional blocks increase the
        number of channels but retain image size.
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
            out_channels = self.compute_out_channels(
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
    def compute_out_channels(i: int, default: int, init: int) -> int:
        """Computes num of output channels for each ConvBlock."""
        if i == 0:
            return init
        if i % 2 == 0:
            return default * 2
        return default

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Discriminator forward method.

        Args:
            batch (torch.Tensor): One or more high-resolution or
                super-resolution images. A tensor of size
                (
                    num_images,
                    input_num_channels,
                    width * scale_factor,
                    height * scale_factor
                ).

        Returns:
            torch.Tensor: Expected probability for each given image to
            be a high-resolution image. A tensor of size (num_images, 1).
        """
        return self.sequential(batch)


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
