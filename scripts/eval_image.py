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
from binarization.traintools import prepare_cuda_device, prepare_generator


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

    device = prepare_cuda_device(0)

    # gen = prepare_generator(cfg, device=device)
    import torch_tensorrt

    gen = torch.jit.load("trt_ts_module.ts").to(device).eval()

    test_batches = get_test_batches(cfg)
    progress_bar = tqdm(test_batches, total=n_evaluations)

    for step_id, (original, compressed) in enumerate(progress_bar):
        if n_evaluations and step_id > n_evaluations - 1:
            break

        compressed = compressed.to(device)
        compressed = make_4times_downscalable(compressed)

        gen.eval()
        with torch.no_grad():
            generated = gen(compressed)

        compressed = compressed.cpu()
        generated = generated.cpu()
        generated = postprocess(original=original, generated=generated)

        for i in range(original.shape[0]):
            fig = draw_validation_fig(
                original_image=original[i],
                compressed_image=compressed[i],
                generated_image=generated[i],
            )
            fig.savefig(save_dir / f'f{step_id:05d}_validation_fig.jpg')
            plt.close(fig)  # close the current fig to prevent OOM issues


if __name__ == "__main__":
    default_cfg = get_default_config()
    default_cfg.params.buffer_size = 1

    unet_ckpt_path = Path(
        default_cfg.paths.artifacts_dir,
        "best_checkpoints",
        "2022_12_19_unet_4_318780.pth",
    )

    srunet_ckpt_path = Path(
        default_cfg.paths.artifacts_dir,
        "best_checkpoints",
        "2022_12_19_srunet_4_318780.pth",
    )
    unet_cfg = default_cfg.copy()
    srunet_cfg = default_cfg.copy()

    unet_cfg.model.ckpt_path_to_resume = unet_ckpt_path
    unet_cfg.model.name = 'unet'

    srunet_cfg.model.ckpt_path_to_resume = srunet_ckpt_path
    srunet_cfg.model.name = 'srunet'

    assert unet_cfg.model.name == 'unet'

    max_n_frames = 10
    # eval_images(unet_cfg, n_evaluations=max_n_frames)
    # eval_images(srunet_cfg, n_evaluations=max_n_frames)

    import time

    import numpy as np
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    def benchmark(
        model,
        input_shape=(1, 3, 288, 480),
        dtype='fp32',
        nwarmup=10,
        nruns=300,
    ):
        input_data = torch.randn(input_shape)
        input_data = input_data.to("cuda")
        if dtype == 'fp16':
            input_data = input_data.half()

        print("Warm up ...")
        with torch.no_grad():
            for _ in range(nwarmup):
                _ = model(input_data)
        torch.cuda.synchronize()

        print("Start timing ...")
        timings = []
        with torch.no_grad():
            for i in range(1, nruns + 1):
                start_time = time.perf_counter_ns()
                output = model(input_data)
                torch.cuda.synchronize()
                end_time = time.perf_counter_ns()
                timings.append(end_time - start_time)
                if i % (nruns // 10) == 0:
                    print(
                        f"# {i}/{nruns}, avg batch time: "
                        f"{np.mean(timings) / 1e+9:.6f} [s]."
                    )

        print("Input shape:", input_data.size())
        print("Output shape:", output.shape)
        print(f"Average batch time: {np.mean(timings) / 1e+9:.6f} [s]")

    old_unet = prepare_generator(srunet_cfg, device="cuda").eval()
    benchmark(old_unet)
    import torch_tensorrt

    quant_unet = torch.jit.load("trt_ts_module.ts").eval()
    benchmark(quant_unet)
