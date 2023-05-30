# pylint: disable=redefined-outer-name
"""Script to evaluate a video with a super-resolution model"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch_tensorrt  # pylint: disable=unused-import
import torchvision.transforms.functional as TF
from gifnoc import Gifnoc

from binarization.config import get_default_config
from binarization.datatools import postprocess, preprocess
from binarization.traintools import prepare_cuda_device, prepare_generator
from binarization.videotools import (
    compress_video,
    frames_to_video,
    video_to_frames,
)


def eval_images(
    gen: torch.nn.Module,
    compressed_path_list: list[Path],
    generated_dir: Path,
    cfg: Gifnoc = None,
    dtype: str = "fp32",
    cuda_or_cpu: str = "cuda",
    width_original: int = 1920,
    height_original: int = 1080,
):
    if cfg is None:
        cfg = get_default_config()
    if cuda_or_cpu.startswith("cuda"):
        cuda_or_cpu = prepare_cuda_device(0)

    id_counter = 0
    for compressed_path in compressed_path_list:
        compressed = preprocess(
            compressed_path=compressed_path,
            dtype=dtype,
            cuda_or_cpu=cuda_or_cpu,
        )
        generated = gen(compressed)
        generated = postprocess(
            generated=generated,
            width_original=width_original,
            height_original=height_original,
        )
        for _ in range(generated.shape[0]):
            generated_pil = TF.to_pil_image(generated.squeeze(0))
            generated_path = Path(
                generated_dir, f"generated_{id_counter:04d}.png"
            )
            generated_pil.save(generated_path)
            id_counter += 1


def compute_vmaf(
    original_video_path: Path,
    generated_video_path: Path,
    fps: int = 30,
) -> tuple[float, float]:
    """Computes the VMAF quality score between two videos."""

    model_path = Path(
        Path.home(), "ffmpeg_sources/vmaf-2.1.1/model/vmaf_v0.6.1.json"
    )

    cmd = (
        f'ffmpeg -nostats -loglevel 0'
        f' -r {fps} -i {original_video_path}'
        f' -r {fps} -i {generated_video_path}'
        # f' -ss 00:{from_minute}:{from_second} -to 00:{to_minute}:{to_second}'
        f' -lavfi "[0:v]setpts=PTS-STARTPTS[reference];'
        f' [1:v]scale=-1:1080:flags=bicubic,setpts=PTS-STARTPTS[distorted];'
        f' [distorted][reference]libvmaf='
        f'log_fmt=json:'
        f'log_path=/dev/stdout:'
        f'model_path={model_path}:'
        f'n_threads=4"'
        f' -f null - | grep "mean"'
    )
    res = os.popen(cmd).read()
    mean, hmean = [
        float(res.split('"')[idx].strip("\n :,")) for idx in (-3, -1)
    ]
    return mean, hmean


def generate_video(
    gen,
    original_video_path: Path,
    output_dir: Path,
    cfg: Gifnoc = None,
    dtype: str = "fp32",
    cuda_or_cpu: str = "cuda",
    width_original: int = 1920,
    height_original: int = 1080,
):
    compressed_video_path = output_dir / "compressed.mp4"
    generated_video_path = output_dir / "generated.mp4"
    another_original_video_path = output_dir / "original.mp4"

    original_frames_dir = output_dir / "original_frames"
    compressed_frames_dir = output_dir / "compressed_frames"
    generated_frames_dir = output_dir / "generated_frames"
    output_dir.mkdir(exist_ok=True, parents=False)
    original_frames_dir.mkdir(exist_ok=True, parents=False)
    compressed_frames_dir.mkdir(exist_ok=True, parents=False)
    generated_frames_dir.mkdir(exist_ok=True, parents=False)

    compress_video(
        original_video_path=original_video_path,
        compressed_video_path=compressed_video_path,
    )
    video_to_frames(
        video_path=original_video_path,
        frames_dir=original_frames_dir,
        ext=".png",
    )
    frames_to_video(
        frames_dir=original_frames_dir, video_path=another_original_video_path
    )

    video_to_frames(
        video_path=compressed_video_path,
        frames_dir=compressed_frames_dir,
        ext=".jpg",
    )
    eval_images(
        gen=gen,
        compressed_path_list=sorted(compressed_frames_dir.iterdir()),
        generated_dir=generated_frames_dir,
        cfg=cfg,
        dtype=dtype,
        cuda_or_cpu=cuda_or_cpu,
        width_original=width_original,
        height_original=height_original,
    )
    frames_to_video(
        frames_dir=generated_frames_dir, video_path=generated_video_path
    )


def generate_video_and_compute_vmaf(
    original_video_path: Path,
    model_name: str = "unet",
    cfg: Gifnoc | None = None,
    cuda_or_cpu: str = "cuda",
) -> dict[str, tuple[float, float]]:

    if cfg is None:
        cfg = get_default_config()

    ckpt_path = Path(
        cfg.paths.artifacts_dir,
        "best_checkpoints",
        # f"2022_12_19_{model_name}_4_318780.pth",
        f"2023_03_24_{model_name}_2_191268.pth",
    )
    cfg.model.ckpt_path_to_resume = ckpt_path
    cfg.model.name = model_name
    gen = prepare_generator(cfg, device=cuda_or_cpu).eval()

    vmaf_dir = cfg.paths.artifacts_dir / "vmaf"
    output_dir = vmaf_dir / ckpt_path.stem
    output_dir.mkdir(exist_ok=True, parents=True)

    generate_video(
        gen,
        original_video_path,
        output_dir=output_dir,
        cfg=cfg,
    )
    mean, hmean = compute_vmaf(
        original_video_path=output_dir / "original.mp4",
        generated_video_path=output_dir / "generated.mp4",
    )

    return {model_name: (mean, hmean)}


def generate_video_and_compute_vmaf_for_trt_models(
    original_video_path: Path,
    model_name: str = "unet",
    cuda_or_cpu: str = "cuda",
    cfg: Gifnoc = None,
) -> dict[str, tuple[float, float]]:
    if cfg is None:
        cfg = get_default_config()

    available_dtypes = ("int8", "fp16", "fp32")
    res_dict = {}
    for dtype in available_dtypes:

        vmaf_dir = cfg.paths.artifacts_dir / "vmaf"
        output_dir = vmaf_dir / f"{model_name}_{dtype}"
        output_dir.mkdir(exist_ok=True, parents=True)

        quant_path = cfg.paths.trt_dir / f"{model_name}_{dtype}.ts"
        quant_gen = torch.jit.load(quant_path).to(cuda_or_cpu).eval()

        generate_video(
            quant_gen,
            original_video_path,
            output_dir=output_dir,
            cfg=cfg,
            dtype=dtype,
            cuda_or_cpu=cuda_or_cpu,
        )
        res_dict[f"{model_name}_{dtype}"] = compute_vmaf(
            original_video_path=output_dir / "original.mp4",
            generated_video_path=output_dir / "generated.mp4",
        )
    return res_dict


if __name__ == "__main__":
    cfg = get_default_config()
    vmaf_dir: Path = cfg.paths.artifacts_dir / "vmaf"
    original_video_path = vmaf_dir / "original.y4m"

    res = {}
    res.update(
        generate_video_and_compute_vmaf(
            original_video_path=original_video_path,
            model_name="unet",
        )
    )
    res.update(
        generate_video_and_compute_vmaf(
            original_video_path=original_video_path,
            model_name="srunet",
        )
    )
    res.update(
        generate_video_and_compute_vmaf_for_trt_models(
            original_video_path=original_video_path,
            model_name="unet",
        )
    )
    res.update(
        generate_video_and_compute_vmaf_for_trt_models(
            original_video_path=original_video_path,
            model_name="srunet",
        )
    )
    with open(vmaf_dir / "vmaf_res.json", "w", encoding="utf-8") as out_file:
        json.dump(res, out_file)
