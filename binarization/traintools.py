"""Collection of train-related utilities"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch

from binarization.config import Gifnoc
from binarization.models import SRUNet, UNet


def set_up_cuda_device(device_id: int = 1, verbose: bool = False) -> str:
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
            print(
                f'Device has been changed from cuda:{curr_device_id} to cuda:{device_id}'
            )
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


def set_up_generator(cfg: Gifnoc, device: str | torch.device) -> UNet | SRUNet:
    """Instantiates a UNet or a SRUNet, resuming model weights if provided"""
    if cfg.model.name == 'unet':
        generator = UNet(
            num_filters=cfg.model.num_filters,
            use_residual=cfg.model.use_residual,
            use_batch_norm=cfg.model.use_batch_norm,
            scale_factor=cfg.params.scale_factor,
        )
    elif cfg.model.name == 'srunet':
        raise NotImplementedError("[WIP] SRUNet not implementated.")
        # generator = SRUNet(
        #     num_filters=cfg.model.num_filters,
        #     use_residual=cfg.model.use_residual,
        #     use_batch_norm=cfg.model.use_batch_norm,
        #     scale_factor=cfg.params.scale_factor
        # )
    else:
        raise ValueError(f"`{cfg.model.name=}`, choose in {'unet', 'srunet'}.")

    generator.to(device)
    if cfg.model.ckpt_path_to_resume:
        print(f'>>> Resuming from {cfg.model.ckpt_path_to_resume}')
        generator.load_state_dict(
            torch.load(
                cfg.model.ckpt_path_to_resume.as_posix(), map_location=device
            )
        )
    return generator


# def set_up_srunet(cfg: Gifnoc) -> SRUNet:
#     """Instantiates a UNet, resuming model weights if provided"""
#     generator = SRUNet(
#         num_filters=cfg.model.num_filters,
#         use_residual=cfg.model.use_residual,
#         use_batch_norm=cfg.model.use_batch_norm,
#         scale_factor=cfg.params.scale_factor
#     )
#     if cfg.params.srunet.ckpt_path_to_resume:
#         print(f'>>> resume from {cfg.params.srunet.ckpt_path_to_resume}')
#         generator.load_state_dict(torch.load(cfg.params.srunet.ckpt_path_to_resume.as_posix()))
#     return generator
