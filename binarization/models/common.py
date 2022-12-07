"""Common pieces required by super-resolution model implementations."""

from __future__ import annotations

import torch


class UNetBlock(torch.nn.Module):
    """UNet block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        stride: int = 1,
        kernel_size: int = 3,
    ):
        """UNet block.

        Args:
            in_channels (int): channel dimension of the input.
            out_channels (int): channel dimension of the output.
            use_batch_norm (bool): flag for batch normalization. Defaults to False.
            stride (int, optional): convolution stride. Defaults to 1.
            kernel_size (int, optional): convolution kernel size. Defaults to 3.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = torch.nn.Conv2d(
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
        out = self.conv(batch)
        out = self.batch_norm(out)
        out = self.activation(out)
        return out


def generate_unet_block_sequence(
    in_channels: int,
    out_channels: int,
    use_batch_norm: bool = False,
    num_blocks: int = 2,
) -> torch.nn.Module:
    """Generates a sequence of UNet blocks.

    Args:
        in_channels (int): channel dimension of the input.
        out_channels (int): channel dimension of the output.
        use_batch_norm (bool): flag for batch normalization. Defaults to False.
        num_blocks (int, optional): num of UNet blocks. Defaults to 2.

    Returns:
        torch.nn.Module: a sequence of UNet blocks.
    """
    return torch.nn.Sequential(
        *(
            UNetBlock(
                in_channels=in_channels if block_id == 0 else out_channels,
                out_channels=out_channels,
                use_batch_norm=use_batch_norm,
            )
            for block_id in range(int(num_blocks))
        )
    )
