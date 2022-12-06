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
from binarization.traintools import set_up_generator


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

    device = 'cpu'  # set_up_cuda_device()

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
            save_path = save_dir / f'validation_fig_{counter}.jpg'
            counter += 1
            fig.savefig(save_path)
            plt.close(fig)  # close the current fig to prevent OOM issues


if __name__ == "__main__":
    default_cfg = get_default_config()

    # unet_ckpt_path = Path(
    #     default_cfg.paths.artifacts_dir,
    #     "checkpoints",
    #     "2022_11_15_07_43_30/unet_2_191268.pth",
    # )

    srunet_ckpt_path = Path(
        default_cfg.paths.artifacts_dir,
        "best_checkpoints",
        "2022_12_06_srunet.pth",
    )

    default_cfg.model.ckpt_path_to_resume = srunet_ckpt_path
    default_cfg.params.buffer_size = 1
    default_cfg.model.name = 'srunet'

    eval_images(default_cfg, n_evaluations=128)
