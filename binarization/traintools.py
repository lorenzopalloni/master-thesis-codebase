"""Collection of train-related utilities"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch

from binarization.config import Gifnoc
from binarization import models


def set_cuda_device(device_id: int = 1, verbose: bool = False) -> str:
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
    if (
        curr_device_count == 3  # solaris workstation
        and curr_device_id != device_id
    ):
        torch.cuda.set_device(f'cuda:{device_id}')
        if verbose:
            print(f'Device has been changed from cuda:{curr_device_id} to cuda:{device_id}')
            print('Current device name:', torch.cuda.get_device_name())
    else:
        if verbose:
            print('Nothing changed.')
    return f'cuda:{torch.cuda.current_device()}'

def set_up_checkpoints_dir(artifacts_dir: Path) -> Path:
    """Sets up unique-time-related dirs for model checkpoints and runs"""
    str_now = datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
    checkpoints_dir = Path(artifacts_dir, 'checkpoints', str_now)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir


def set_up_unet(cfg: Gifnoc) -> models.UNet:
    """Instantiates a UNet, resuming model weights if provided"""
    generator = models.UNet(
        num_filters=cfg.params.unet.num_filters,
        use_residual=cfg.params.unet.use_residual,
        use_batch_norm=cfg.params.unet.use_batch_norm,
        scale_factor=cfg.params.unet.scale_factor
    )
    if cfg.params.unet.ckpt_path_to_resume:
        print(f'>>> resume from {cfg.params.unet.ckpt_path_to_resume}')
        generator.load_state_dict(torch.load(cfg.params.unet.ckpt_path_to_resume.as_posix()))
    return generator

# def set_up_srunet(cfg: Gifnoc) -> models.SRUNet:
#     """Instantiates a UNet, resuming model weights if provided"""
#     generator = models.SRUNet(
#         num_filters=cfg.params.unet.num_filters,
#         use_residual=cfg.params.unet.use_residual,
#         use_batch_norm=cfg.params.unet.use_batch_norm,
#         scale_factor=cfg.params.unet.scale_factor
#     )
#     if cfg.params.srunet.ckpt_path_to_resume:
#         print(f'>>> resume from {cfg.params.srunet.ckpt_path_to_resume}')
#         generator.load_state_dict(torch.load(cfg.params.srunet.ckpt_path_to_resume.as_posix()))
#     return generator
