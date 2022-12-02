"""Common pieces required by super-resolution model implementations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
