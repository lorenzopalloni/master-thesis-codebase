"""Script to map old UNet weight names to new ones."""
from pathlib import Path

import torch

from binarization.config import get_default_config

KEY_MAP = {
    'down1.0.conv.weight': 'dconv_down1.0.conv1.weight',
    'down1.0.conv.bias': 'dconv_down1.0.conv1.bias',
    'down1.1.conv.weight': 'dconv_down1.1.conv1.weight',
    'down1.1.conv.bias': 'dconv_down1.1.conv1.bias',
    'down2.0.conv.weight': 'dconv_down2.0.conv1.weight',
    'down2.0.conv.bias': 'dconv_down2.0.conv1.bias',
    'down2.1.conv.weight': 'dconv_down2.1.conv1.weight',
    'down2.1.conv.bias': 'dconv_down2.1.conv1.bias',
    'down3.0.conv.weight': 'dconv_down3.0.conv1.weight',
    'down3.0.conv.bias': 'dconv_down3.0.conv1.bias',
    'down3.1.conv.weight': 'dconv_down3.1.conv1.weight',
    'down3.1.conv.bias': 'dconv_down3.1.conv1.bias',
    'down4.0.conv.weight': 'dconv_down4.0.conv1.weight',
    'down4.0.conv.bias': 'dconv_down4.0.conv1.bias',
    'down4.1.conv.weight': 'dconv_down4.1.conv1.weight',
    'down4.1.conv.bias': 'dconv_down4.1.conv1.bias',
    'up4.0.conv.weight': 'dconv_up3.0.conv1.weight',
    'up4.0.conv.bias': 'dconv_up3.0.conv1.bias',
    'up4.1.conv.weight': 'dconv_up3.1.conv1.weight',
    'up4.1.conv.bias': 'dconv_up3.1.conv1.bias',
    'up3.0.conv.weight': 'dconv_up2.0.conv1.weight',
    'up3.0.conv.bias': 'dconv_up2.0.conv1.bias',
    'up3.1.conv.weight': 'dconv_up2.1.conv1.weight',
    'up3.1.conv.bias': 'dconv_up2.1.conv1.bias',
    'up2.0.conv.weight': 'dconv_up1.0.conv1.weight',
    'up2.0.conv.bias': 'dconv_up1.0.conv1.bias',
    'up2.1.conv.weight': 'dconv_up1.1.conv1.weight',
    'up2.1.conv.bias': 'dconv_up1.1.conv1.bias',
    'up1.weight': 'conv_last.weight',
    'up1.bias': 'conv_last.bias',
}

if __name__ == '__main__':
    default_cfg = get_default_config()

    state_dict_name: str = "2022_11_21_unet.pth"

    state_dict_path = Path(
        default_cfg.paths.artifacts_dir,
        "best_checkpoints",
        state_dict_name,
    )
    state_dict = torch.load(state_dict_path, map_location='cpu')

    new_state_dict = {}
    for new_k, old_k in KEY_MAP.items():
        new_state_dict[new_k] = state_dict[old_k]

    torch.save(
        new_state_dict,
        Path(state_dict_path.parent, f'new_{state_dict_name}').as_posix(),
    )
