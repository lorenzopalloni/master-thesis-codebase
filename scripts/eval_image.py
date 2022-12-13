"""Script to evaluate an image with a super-resolution model"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from binarization.config import Gifnoc, get_default_config
from binarization.dataset import get_test_batches
from binarization.datatools import (
    draw_validation_fig,
    make_4times_downscalable,
    postprocess,
)
from binarization.traintools import set_up_cuda_device, set_up_generator


def eval_images(cfg: Gifnoc, n_evaluations: int | None = None):
    """Upscales a bunch of images with a super-resolution model.

    Args:
        cfg (Gifnoc): configuration settings.
        n_evaluations (Union[int, None], optional): num of images to evaluate.
            Defaults to None (that means all the available frames).
    """
    ckpt_path = cfg.model.ckpt_path_to_resume
    save_dir = cfg.paths.outputs_dir / ckpt_path.stem

    save_dir.mkdir(exist_ok=True, parents=True)

    device = set_up_cuda_device(0)

    gen = set_up_generator(cfg, device=device)
    gen.to(device)

    test_batches = get_test_batches(cfg)
    progress_bar = tqdm(test_batches)

    counter = 0
    for step_id, (original, compressed) in enumerate(progress_bar):
        if n_evaluations and step_id > n_evaluations - 1:
            break

        original = original.to(device)
        compressed = compressed.to(device)
        compressed = make_4times_downscalable(compressed)

        gen.eval()
        with torch.no_grad():
            generated = gen(compressed)

        original = original.cpu()
        compressed = compressed.cpu()
        generated = generated.cpu()
        generated = postprocess(original=original, generated=generated)

        for i in range(original.shape[0]):
            fig = draw_validation_fig(
                original_image=original[i],
                compressed_image=compressed[i],
                generated_image=generated[i],
            )
            counter += 1
            fig.savefig(save_dir / f'validation_fig_{counter}.jpg')
            plt.close(fig)  # close the current fig to prevent OOM issues


if __name__ == "__main__":
    default_cfg = get_default_config()
    default_cfg.params.buffer_size = 1

    unet_ckpt_path = Path(
        default_cfg.paths.artifacts_dir,
        "best_checkpoints",
        "2022_12_13_unet_0_39999.pth",
    )

    srunet_ckpt_path = Path(
        default_cfg.paths.artifacts_dir,
        "best_checkpoints",
        "2022_12_13_srunet_0_39999.pth",
    )
    unet_cfg = default_cfg.copy()
    srunet_cfg = default_cfg.copy()

    unet_cfg.model.ckpt_path_to_resume = unet_ckpt_path
    unet_cfg.model.name = 'unet'

    srunet_cfg.model.ckpt_path_to_resume = srunet_ckpt_path
    srunet_cfg.model.name = 'srunet'

    assert unet_cfg.model.name == 'unet'

    max_n_frames = 128
    eval_images(unet_cfg, n_evaluations=max_n_frames)
    eval_images(srunet_cfg, n_evaluations=max_n_frames)
