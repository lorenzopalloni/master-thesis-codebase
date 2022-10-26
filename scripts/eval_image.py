# pylint: disable=missing-function-docstring
"""Script to evaluate a super-resolution model"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from binarization.config import Gifnoc, get_default_config
from binarization.dataset import get_test_batches
from binarization.datatools import (
    adjust_image_for_unet,
    draw_validation_fig,
    process_raw_generated,
)
from binarization.traintools import set_cuda_device, set_up_unet


def main(cfg: Gifnoc, n_evaluations: int):
    save_dir = cfg.paths.outputs_dir / cfg.params.ckpt_path_to_resume.stem
    save_dir.mkdir(exist_ok=True, parents=True)

    device = set_cuda_device()
    gen = set_up_unet(cfg)
    gen.to(device)

    test_batches = get_test_batches(cfg)
    progress_bar = tqdm(test_batches)

    counter = 0
    for step_id, (original, compressed) in enumerate(progress_bar):
        if step_id > n_evaluations - 1:
            break

        original = original.to(device)
        compressed = compressed.to(device)
        compressed = adjust_image_for_unet(compressed)

        gen.eval()
        with torch.no_grad():
            generated = gen(compressed)

        original = original.cpu()
        compressed = compressed.cpu()
        generated = generated.cpu()
        generated = process_raw_generated(
            original=original, generated=generated
        )

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
    default_cfg.params.ckpt_path_to_resume = Path(
        default_cfg.paths.artifacts_dir,
        "/checkpoints/2022_10_21_07_54_20/unet_0_59999.pth",
    )
    default_cfg.params.buffer_size = 1
    main(default_cfg, n_evaluations=128)
