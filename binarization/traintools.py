"""Collection of train-related utilities"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from lpips import lpips

from binarization.config import Gifnoc
from binarization.models import SRUNet, UNet


def prepare_cuda_device(device_id: int = 0, verbose: bool = False) -> str:
    """Tries to set device id to `1` with 3 GPUs and to `0` with only one."""
    if not torch.cuda.is_available():
        raise ValueError("pytorch was not able to detect any GPU.")
    curr_device_name = torch.cuda.get_device_name()
    curr_device_id = torch.cuda.current_device()
    curr_device_count = torch.cuda.device_count()
    if verbose:
        print(f'Current device name: {curr_device_name}')
        print(f'Current device id: {curr_device_id}')
        print('Trying to change device id...')
    condition1 = curr_device_count == 3  # likely solaris workstation
    condition2 = curr_device_id != device_id
    if condition1 and condition2:
        torch.cuda.set_device(f'cuda:{device_id}')
        if verbose:
            print(
                f'Device has been changed from cuda:{curr_device_id} to cuda:{device_id}'
            )
            print('Current device name:', torch.cuda.get_device_name())
    elif verbose:
        print('Nothing changed.')
    return f'cuda:{torch.cuda.current_device()}'


def prepare_checkpoints_dir(artifacts_dir: Path) -> Path:
    """Sets up unique-time-related dirs for model checkpoints and runs"""
    str_now = datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
    checkpoints_dir = Path(artifacts_dir, 'checkpoints', str_now)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir


def prepare_generator(
    cfg: Gifnoc, device: str | torch.device
) -> torch.nn.Module:
    """Inits a generator resuming model weights if provided."""

    generators = {'unet': UNet, 'srunet': SRUNet}
    if cfg.model.name not in generators:
        raise ValueError(f"`{cfg.model.name=}`, choose in {'unet', 'srunet'}.")

    generator: torch.nn.Module = generators[cfg.model.name](
        num_filters=cfg.model.num_filters,
        use_batch_norm=cfg.model.use_batch_norm,
        scale_factor=cfg.params.scale_factor,
    )

    generator.to(device)

    if cfg.model.ckpt_path_to_resume:
        print(f'>>> Resuming from {cfg.model.ckpt_path_to_resume}')
        generator.load_state_dict(
            torch.load(
                cfg.model.ckpt_path_to_resume.as_posix(), map_location=device
            )
        )
    return generator


class CustomLPIPS(torch.nn.Module):
    """Custom LPIPS."""

    def __init__(self, net: str = 'vgg'):
        """PyTorch custom module for LPIPS.

        Args:
        net (str): ['alex','vgg','squeeze'] are the base/trunk networks
            available. Defaults to 'vgg'.
        """
        super().__init__()
        self.lpips = lpips.LPIPS(net=net, version='0.1', verbose=False)
        self.magic_mean = 0.4

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Normalizes in [-1, 1], and computes LPIPS."""
        normalized_y_pred = y_pred - self.magic_mean
        normalized_y_pred = torch.clamp(normalized_y_pred, -1.0, 1.0)
        normalized_y_true = y_true - self.magic_mean
        normalized_y_true = torch.clamp(normalized_y_true, -1.0, 1.0)
        lpips_op = self.lpips(normalized_y_pred, normalized_y_true)
        return lpips_op
