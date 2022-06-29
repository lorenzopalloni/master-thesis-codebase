import torch
from typing import Optional

"""
in_dim (float, optional):
    channel dimension of the input
num_classes (str).
    channel dimension of the output
num_filters (int, optional)
    number of filters of the first channel. after layer it gets multiplied
    by 2 during the encoding stage, and divided during the decoding
downsample (None or float, optional):
    can be used for downscaling the output. e.g., if you use downsample=0.5
    the output resolution will be halved
residual (bool):
    if using the residual scheme and adding the input to the final output
scale_factor (int):
    basic upscale factor. if you want a rational upscale (e.g. 720p to 1080p,
    which is 1.5) combine it with the downsample parameter
"""


def layer_generator(
    in_channels: int,
    out_channels: int,
    use_batch_norm: bool = False,
    use_residual: bool = True,
    num_blocks: int = 2,
):
    return torch.nn.Sequential(*(
        UNetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
        ) for _ in range(int(num_blocks))
    ))


class UNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        num_filters: int = 32,
        downsample: Optional[float] = None,
        use_residual: bool = True,
        use_batch_norm: bool = False,
        scale_factor: float = 2.0,
    ):
        super().__init__()

        self.use_residual = use_residual
        self.num_classes = num_classes
        self.scale_factor = scale_factor

        self.dconv_down1 = layer_generator(
            in_channels,
            num_filters,
            use_batch_norm=False
        )
        self.dconv_down2 = layer_generator(
            num_filters,
            num_filters * 2,
            use_batch_norm=use_batch_norm,
            n_blocks=2
        )
        self.dconv_down3 = layer_generator(
            num_filters * 2,
            num_filters * 4,
            use_batch_norm=use_batch_norm,
            n_blocks=2
        )
        self.dconv_down4 = layer_generator(
            num_filters * 4,
            num_filters * 8,
            use_batch_norm=use_batch_norm,
            n_blocks=2
        )

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        self.dconv_up3 = layer_generator(
            num_filters * 8 + num_filters * 4,
            num_filters * 4,
            use_batch_norm=use_batch_norm,
            n_blocks=2,
        )
        self.dconv_up2 = layer_generator(
            num_filters * 4 + num_filters * 2,
            num_filters * 2,
            use_batch_norm=use_batch_norm,
            n_blocks=2,
        )
        self.dconv_up1 = layer_generator(
            num_filters * 2 + num_filters,
            num_filters,
            use_batch_norm=False,
            n_blocks=2,
        )

        sf = self.scale_factor * (2 if self.use_s2d else 1)

        self.to_rgb = torch.nn.Conv2d(num_filters, 3, kernel_size=1)
        if sf > 1:
            self.conv_last = torch.nn.Conv2d(
                num_filters, (sf**2) * num_classes, kernel_size=1, padding=0
            )
            self.pixel_shuffle = torch.nn.PixelShuffle(sf)
        else:
            self.conv_last = torch.nn.Conv2d(num_filters, 3, kernel_size=1)

        if downsample is not None and downsample != 1.0:
            self.downsample = torch.nn.Upsample(
                scale_factor=downsample, mode='bicubic', align_corners=True
            )
        else:
            self.downsample = torch.nn.Identity()
        self.layers = [
            self.dconv_down1,
            self.dconv_down2,
            self.dconv_down3,
            self.dconv_down4,
            self.dconv_up3,
            self.dconv_up2,
            self.dconv_up1,
        ]

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

        sf = self.scale_factor * (2 if self.use_s2d else 1)

        if sf > 1:
            x = self.pixel_shuffle(x)
        if self.residual:
            sf = (
                self.scale_factor
            )  # (self.scale_factor // (2 if self.use_s2d and self.scale_factor > 1 else 1))
            x += torch.nn.functional.interpolate(
                input[:, -self.num_classes :, :, :],
                scale_factor=sf,
                mode='bicubic',
            )
            x = torch.clamp(x, min=-1, max=1)

        return torch.clamp(self.downsample(x), min=-1, max=1)

    def reparametrize(self):
        for layer in self.layers:
            for block in layer:
                if hasattr(block, 'conv_adapter'):
                    block.reparametrize_convs()


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
        super(UNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = use_residual
        self.is_reparametrized = False

        self.conv_adapter = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, stride=1, padding=0
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
        )
        self.bn = (
            torch.nn.use_batch_norm2d(num_features=out_channels)
            if use_batch_norm else torch.nn.Identity()
        )
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        condition1 = self.use_residual and not self.is_reparametrized
        condition2 = self.in_channels == self.out_channels
        if condition1 and condition2:
            x += input + self.conv_adapter(input)
        x = self.bn(x)
        x = self.act(x)
        return x

    def reparametrize_convs(self):
        identity_conv = torch.nn.init.dirac_(
            torch.empty_like(self.conv1.weight)
        )
        padded_conv_adapter = torch.nn.functional.pad(
            input=self.conv_adapter.weight,
            pad=(1, 1, 1, 1),
            mode="constant",
            value=0
        )
        if self.in_channels == self.out_channels:
            new_conv_weights = (
                self.conv1.weight + padded_conv_adapter + identity_conv
            )
            new_conv_bias = self.conv1.bias + self.conv_adapter.bias

            self.conv1.weight.data = new_conv_weights
            self.conv1.bias.data = new_conv_bias

        self.is_reparametrized = True
