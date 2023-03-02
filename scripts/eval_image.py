"""Script to evaluate an image with a super-resolution model"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_tensorrt  # mandatory for inference even without calling it
from tqdm import tqdm

from binarization.config import Gifnoc, get_default_config
from binarization.dataset import get_test_batches
from binarization.datatools import draw_validation_fig, postprocess
from binarization.traintools import prepare_cuda_device, prepare_generator


def model_speedtest(
    model: torch.nn.Module,
    input_shape: tuple = (1, 3, 288, 480),
    dtype: str = 'fp32',
    nwarmup: int = 10,
    nruns: int = 300,
) -> list[float]:
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")

    if dtype == 'fp16':
        input_data = input_data.half()
    elif dtype not in {'fp32', 'int8'}:
        raise ValueError(
            f"Unknown dtype: {dtype}. Choose in {'fp32', 'fp16', 'int8'}."
        )

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
    return timings


def eval_images(
    gen: torch.nn.Module,
    save_dir: Path,
    cfg: Gifnoc = None,
    n_evaluations: int | None = None,
    dtype: str = "fp32",
    cuda_or_cpu: str = "cuda",
):
    """Upscales a bunch of images given a super-resolution model.

    Args:
        gen (torch.nn.Module): a PyTorch generator model.
        save_dir (Path): path to directory where to save evaluation figures.
        cfg (Gifnoc, optional): configuration settings. The only useful
            option to be modified here is `cfg.params.buffer_size`.
            Defaults to None.
        n_evaluations (Union[int, None], optional): num of images to evaluate.
            Defaults to None (that means all the available frames).
        cuda_or_cpu (str, optional): {"cuda", "cpu"}. Defaults to "cuda".
    """
    if cfg is None:
        cfg = get_default_config()
    if cuda_or_cpu.startswith("cuda"):
        cuda_or_cpu = prepare_cuda_device(0)

    test_batches = get_test_batches(cfg)
    progress_bar = tqdm(test_batches, total=n_evaluations)

    for step_id, (original, compressed) in enumerate(progress_bar):
        if n_evaluations and step_id > n_evaluations - 1:
            break

        compressed = compressed.to(cuda_or_cpu)
        if dtype == 'fp16':
            compressed = compressed.half()
        elif dtype not in {'fp32', 'int8'}:
            raise ValueError(
                f"Unknown dtype: {dtype}. Choose in {'fp32', 'fp16', 'int8'}."
            )

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
            fig.savefig(save_dir / f'{step_id:05d}_validation_fig.jpg')
            plt.close(fig)  # close the current fig to prevent OOM issues


def eval_trt_models(
    model_name: str = "unet",
    n_evaluations: int = 10,
    cuda_or_cpu: str = "cuda",
    cfg: Gifnoc = None,
) -> dict[str, list[float]]:
    if cfg is None:
        cfg = get_default_config()

    available_dtypes = ("int8", "fp16", "fp32")
    timings_dict = {}
    for dtype in available_dtypes:
        quant_save_dir = cfg.paths.outputs_dir / f"{model_name}_{dtype}"
        quant_save_dir.mkdir(exist_ok=True, parents=True)

        quant_path = cfg.paths.trt_dir / f"{model_name}_{dtype}.ts"
        quant_gen = torch.jit.load(quant_path).to(cuda_or_cpu).eval()

        eval_images(
            gen=quant_gen,
            save_dir=quant_save_dir,
            n_evaluations=n_evaluations,
            dtype=dtype,
        )
        timings_dict[f"{model_name}_{dtype}"] = model_speedtest(
            quant_gen, dtype=dtype
        )
    return timings_dict


def eval_model(
    model_name: str = "unet",
    n_evaluations: int = 10,
    cfg: Gifnoc | None = None,
    cuda_or_cpu: str = "cuda",
) -> dict[str, list[float]]:
    if cfg is None:
        cfg = get_default_config()

    ckpt_path = Path(
        cfg.paths.artifacts_dir,
        "best_checkpoints",
        f"2022_12_19_{model_name}_4_318780.pth",
    )
    cfg.model.ckpt_path_to_resume = ckpt_path
    cfg.model.name = model_name

    save_dir = cfg.paths.outputs_dir / ckpt_path.stem
    save_dir.mkdir(exist_ok=True, parents=True)

    gen = prepare_generator(cfg, device=cuda_or_cpu).eval()
    eval_images(gen=gen, save_dir=save_dir, n_evaluations=n_evaluations)
    return {model_name: model_speedtest(gen)}


if __name__ == "__main__":
    import json
    from datetime import datetime

    n_evaluations = 10

    cfg = get_default_config()
    today_str = datetime.now().strftime(r"%Y_%m_%d")

    model_name = "unet"
    timings_dict = eval_model(
        model_name=model_name, n_evaluations=n_evaluations
    )
    trt_timings_dict = eval_trt_models(
        model_name=model_name, n_evaluations=n_evaluations
    )
    timings_dict.update(trt_timings_dict)

    save_path = (
        cfg.paths.outputs_dir / f"{today_str}_timings_{model_name}.json"
    )

    with open(save_path, "w") as out_file:
        json.dump(timings_dict, out_file)

    model_name = "srunet"
    timings_dict = eval_model(
        model_name=model_name, n_evaluations=n_evaluations
    )
    trt_timings_dict = eval_trt_models(
        model_name=model_name, n_evaluations=n_evaluations
    )
    timings_dict.update(trt_timings_dict)
    save_path = (
        cfg.paths.outputs_dir / f"{today_str}_timings_{model_name}.json"
    )

    with open(save_path, "w") as out_file:
        json.dump(timings_dict, out_file)

    # quant_save_dir = user_cfg.paths.outputs_dir / f"quant_{model_name}"
    # quant_save_dir.mkdir(exist_ok=True, parents=True)

    # quant_path = user_cfg.paths.trt_dir / f"{model_name}.ts"
    # quant_gen = torch.jit.load(quant_path).to(cuda_or_cpu).eval()

    # eval_images(
    #     gen=quant_gen, save_dir=quant_save_dir, n_evaluations=n_evaluations
    # )
    # model_speedtest(quant_gen)
