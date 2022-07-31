import torch
import torch.nn.functional as F
from torch import nn

from typing import Optional, Callable, List


class Discriminator(nn.Module):
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

        conv_blocks = list()
        all_out_channels = [input_num_channels]
        for i in range(num_blocks):
            in_channels = all_out_channels[-1]
            out_channels = self._out_channels_helper(
                i=i, default=all_out_channels[-1], init=num_channels
            )
            all_out_channels.append(out_channels)

            conv_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1 if i % 2 == 0 else 2,
                    use_batch_norm=False if i == 0 else True,
                    activation='LeakyReLu',
                    use_spectral_norm=False,
                )
            )

        self.sequential = nn.Sequential(
            *conv_blocks,
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(all_out_channels[-1] * 6 * 6, fc_size),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_size, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _out_channels_helper(i: int, default: int, init: int) -> int:
        """Compute the number of output channels for each Conv block."""
        if i == 0:
            return init
        elif i % 2 == 0:
            return default * 2
        else:
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


class ConvBlock(nn.Module):
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
        activation: Optional[str] = None,
        dilation: int = 1,
        groups: int = 1,
        use_spectral_norm: bool = False,
    ):
        """Convolutional block initializer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for each convolutional block.
            stride (int, optional): Stride. Defaults to 1.
            use_batch_norm (bool, optional): Flag for batch normalization.
                Defaults to False.
            activation (Optional[str], optional): Activation function name.
                Defaults to None.
            dilation (int, optional): Defaults to 1.
            groups (int, optional): Defaults to 1.
            use_spectral_norm (bool, optional):
                Flag for spectral normalization. Defaults to False.
        """
        super().__init__()

        layers: List[torch.nn.Module] = list()

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(
                    kernel_size // 2 + dilation // 2
                    if use_spectral_norm
                    else kernel_size // 2
                ),
                groups=groups,
                dilation=dilation,
            )
        )

        if use_spectral_norm:
            layers.append(nn.utils.spectral_norm)

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        activation_op = self._activation_layer_helper(activation)
        if activation_op is not None:
            layers.append(activation_op)

        self.sequential = nn.Sequential(*layers)

    @staticmethod
    def _activation_layer_helper(
        activation_name: Optional[str] = None,
    ) -> Optional[Callable]:
        if activation_name is not None:
            available_activation_names = {'prelu', 'leakyrelu', 'tanh'}
            activation_name = activation_name.lower()
            assert activation_name in available_activation_names

        if activation_name == 'prelu':
            return nn.PReLU()
        elif activation_name == 'leakyrelu':
            return nn.LeakyReLU(0.2)
        elif activation_name == 'tanh':
            return nn.Tanh()

        return None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input (torch.Tensor): Input images.
                A tensor or size (num_of_images, in_channels, w, h).

        Returns:
            torch.Tensor: Output images.
                A tensor of size (num_of_images, out_channels, w, h)
        """
        return self.sequential(input)


# reference: https://github.com/usuyama/pytorch-unet


class UNetBlock(nn.Module):
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

        self.conv_adapter = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, stride=1, padding=0
        )
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
        )
        self.bn = (
            nn.BatchNorm2d(num_features=out_channels)
            if use_batch_norm else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

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
        identity_conv = nn.init.dirac_(
            torch.empty_like(self.conv1.weight)
        )
        padded_conv_adapter = F.pad(
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


def layer_generator(
    in_channels: int,
    out_channels: int,
    use_batch_norm: bool = False,
    use_residual: bool = True,
    num_blocks: int = 2,
):
    return nn.Sequential(*(
        UNetBlock(
            in_channels=in_channels if block_id == 0 else out_channels,
            out_channels=out_channels,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
        ) for block_id in range(int(num_blocks))
    ))


def sr_espcn(n_filters, scale_factor=2, out_channels=3, kernel_size=1):
    return nn.Sequential(
        *(
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=n_filters,
                out_channels=(scale_factor**2) * out_channels,
                padding=kernel_size // 2,
            ),
            nn.PixelShuffle(scale_factor),
        )
    )


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_filters: int = 64,
        use_residual: bool = True,
        use_batch_norm: bool = False,
        scale_factor: Optional[int] = 2
    ):
        """U-Net initializer.

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
            scale_factor (Optional[int]): Scaling factor. Defaults to 2.
        """
        super().__init__()

        self.use_residual = use_residual
        self.out_channels = out_channels
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

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2.0, mode='bilinear')

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

        self.to_rgb = nn.Conv2d(num_filters, 3, kernel_size=1)

        if self.scale_factor is not None:
            self.conv_last = nn.Conv2d(
                in_channels=num_filters,
                out_channels=(self.scale_factor ** 2) * out_channels,
                kernel_size=1,
                padding=0,
            )
            self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

        else:
            self.conv_last = nn.Conv2d(
                num_filters, out_channels, kernel_size=1
            )
            # self.downsample = nn.Upsample(
            #     scale_factor=1 / self.scale_factor,
            #     mode='bicubic',
            #     align_corners=True
            # )
        # self.downsample = nn.Identity()

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

        if self.scale_factor is not None:
            x = self.pixel_shuffle(x)

        if self.use_residual:
            x += F.interpolate(
                input[:, -self.out_channels:, :, :],  # RGB -> BGR
                scale_factor=float(self.scale_factor),
                mode='bicubic',
            )

        return torch.clamp(x, min=-1, max=1)

    def reparametrize(self):
        for layer in self.layers:
            for block in layer:
                if hasattr(block, 'conv_adapter'):
                    block.reparametrize_convs()


class SRUnet(nn.Module):
    def __init__(
        self,
        in_dim=3,
        n_class=3,
        downsample=None,
        residual=False,
        batchnorm=False,
        scale_factor=2,
        n_filters=64,
        layer_multiplier=1,
    ):
        """
        Args:
            in_dim (float, optional):
                channel dimension of the input
            n_class (str):
                channel dimension of the output
            n_filters (int, optional):
                maximum number of filters. the layers start with n_filters / 2,  after each layer this number gets multiplied by 2
                 during the encoding stage and until it reaches n_filters. During the decoding stage the number follows the reverse
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
        self.n_class = n_class
        self.scale_factor = scale_factor

        self.dconv_down1 = layer_generator(
            in_dim,
            n_filters // 2,
            use_batch_norm=False,
            n_blocks=2 * layer_multiplier,
        )
        self.dconv_down2 = layer_generator(
            n_filters // 2,
            n_filters,
            use_batch_norm=batchnorm,
            n_blocks=3 * layer_multiplier,
        )
        self.dconv_down3 = layer_generator(
            n_filters,
            n_filters,
            use_batch_norm=batchnorm,
            n_blocks=3 * layer_multiplier,
        )
        self.dconv_down4 = layer_generator(
            n_filters,
            n_filters,
            use_batch_norm=batchnorm,
            n_blocks=3 * layer_multiplier,
        )

        self.maxpool = nn.MaxPool2d(2)
        if downsample is not None and downsample != 1.0:
            self.downsample = nn.Upsample(
                scale_factor=downsample, mode='bicubic', align_corners=True
            )
        else:
            self.downsample = nn.Identity()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.dconv_up3 = layer_generator(
            n_filters + n_filters,
            n_filters,
            use_batch_norm=batchnorm,
            n_blocks=3 * layer_multiplier,
        )
        self.dconv_up2 = layer_generator(
            n_filters + n_filters,
            n_filters,
            use_batch_norm=batchnorm,
            n_blocks=3 * layer_multiplier,
        )
        self.dconv_up1 = layer_generator(
            n_filters + n_filters // 2,
            n_filters // 2,
            use_batch_norm=False,
            n_blocks=3 * layer_multiplier,
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

        self.to_rgb = nn.Conv2d(n_filters // 2, 3, kernel_size=1)
        if sf > 1:
            self.conv_last = nn.Conv2d(
                n_filters // 2, (sf**2) * n_class, kernel_size=1, padding=0
            )
            self.pixel_shuffle = nn.PixelShuffle(sf)
        else:
            self.conv_last = nn.Conv2d(n_filters // 2, 3, kernel_size=1)

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
            x += nn.functional.interpolate(
                input[:, -self.n_class :, :, :],
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


class SimpleResNet(nn.Module):
    def __init__(self, n_filters, n_blocks):
        super(SimpleResNet, self).__init__()
        self.conv1 = UNetBlock(
            in_channels=3,
            out_channels=n_filters,
            use_residual=True,
            use_batch_norm=False,
        )
        convblock = [
            UNetBlock(
                in_channels=n_filters,
                out_channels=n_filters,
                use_residual=True,
                use_batch_norm=False,
            )
            for _ in range(n_blocks - 1)
        ]
        self.convblocks = nn.Sequential(*convblock)
        self.sr = sr_espcn(n_filters, scale_factor=2, out_channels=3)
        self.upscale = nn.Upsample(
            scale_factor=2, mode='bicubic', align_corners=True
        )
        self.clip = nn.Hardtanh()

    def forward(self, input):
        x = self.conv1(input)
        x = self.convblocks(x)
        x = self.sr(x)

        return self.clip(x + self.upscale(input))

    def reparametrize(self):
        for block in self.convblocks:
            if hasattr(block, 'conv_adapter'):
                block.reparametrize_convs()
