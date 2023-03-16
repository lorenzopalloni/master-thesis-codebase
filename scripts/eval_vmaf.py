#pylint: disable=redefined-outer-name
"""Script to evaluate a video with a super-resolution model"""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from gifnoc import Gifnoc
from PIL import Image

from binarization.config import get_default_config
from binarization.traintools import prepare_cuda_device, prepare_generator


def min_max_scaler(
    tensor: torch.Tensor, tensor_min: float = 0.0, tensor_max: float = 255.0
) -> torch.Tensor:
    """Scales any value of a tensor between two given values."""
    return (tensor - tensor_min) / (tensor_max - tensor_min)


def inv_min_max_scaler(
    tensor: torch.Tensor, tensor_min: float = 0.0, tensor_max: float = 255.0
) -> torch.Tensor:
    """Inverts min_max_scaler function."""
    return (tensor * (tensor_max - tensor_min) + tensor_min).int()


def make_4times_downscalable(image: torch.Tensor) -> torch.Tensor:
    """Pads until img_h and img_w are both divisible by 2 at least 4 times."""

    def make_4times_divisible(an_integer: int) -> int:
        """Given an integer `an_integer`, returns another integer that:
        - is greater than `an_integer`
        - is divisible at least four times by 2
        - is the closest to `an_integer`

        Adapts image sizes to feed a UNet-like architecture.

        Args:
            an_integer (int): an integer greater than 0.

        Returns:
            int: an integer with the properties described above.
        """
        assert (
            an_integer > 0
        ), f"Input should be > 0, but `{an_integer}` was provided."
        if an_integer % 2 != 0:  # make it even
            an_integer += 1
        while an_integer / 2**4 % 2 != 0:  # assure divisibility by 16
            an_integer += 2  # jump from one even number to the next one
        return an_integer

    height, width = image.shape[-2], image.shape[-1]
    adjusted_height = make_4times_divisible(height)
    adjusted_width = make_4times_divisible(width)
    return TF.pad(
        image,
        padding=[
            (adjusted_width - width) // 2,  # left/right
            (adjusted_height - height) // 2,  # top/bottom
        ],
    )


def inv_make_4times_downscalable(
    generated: torch.Tensor,
    width_original: int,
    height_original: int,
) -> torch.Tensor:
    """Crops as much as needed to invert `make_4times_downscalable`."""
    height_generated, width_generated = (
        generated.shape[-2],
        generated.shape[-1],
    )
    height_offset = (height_generated - height_original) // 2
    width_offset = (width_generated - width_original) // 2
    return TF.crop(
        generated, height_offset, width_offset, height_original, width_original
    )


def postprocess(
    generated: torch.Tensor,
    width_original: int,
    height_original: int,
) -> torch.Tensor:
    """Postprocesses a super-resolution generator output."""
    generated = generated.cpu()
    generated = inv_make_4times_downscalable(
        generated=generated,
        width_original=width_original,
        height_original=height_original,
    )
    generated = inv_min_max_scaler(generated)
    generated = generated.clip(0, 255)
    generated = generated / 255.0
    return generated


def compress_video(
    original_video_path: Path,
    compressed_video_path: Path,
    crf: int = 23,
    scale_factor: int = 4,
):
    """Compresses and downscale a video.

    Note: do not worry about the following warning (source: google it):
    "deprecated pixel format used, make sure you did set range correctly"

    Args:
        original_video_path (Path): Filename of the input video.
        compressed_video_path (Path): Filename of the compressed video.
        crf (int, optional): Constant Rate Factor. Defaults to 23.
        scale_factor (int): Scale factor. Defaults to 4.
    """
    scaled_w = f"'iw/{scale_factor}-mod(iw/{scale_factor},2)'"
    scaled_h = f"'ih/{scale_factor}-mod(ih/{scale_factor},2)'"
    cmd = [
        'ffmpeg',
        '-i',
        f'{original_video_path}',
        '-c:v',  # codec video
        'libx265',  # H.265/HEVC
        '-crf',  # constant rate factor
        f'{crf}',  # defaults to 23
        '-preset',  # faster -> less quality, slower -> better quality
        'medium',  # defaults to medium
        '-c:a',  # codec audio
        'aac',  # AAC audio format
        '-b:a',  # bitrate audio
        '128k',  # AAC audio at 128 kBit/s
        '-movflags',  # weird option
        'faststart',
        '-vf',  # video filters
        (
            f'scale=w={scaled_w}:h={scaled_h}'  # downscale
            ',format=yuv420p'  # output format, defaults to yuv420p
        ),
        f'{compressed_video_path}',
    ]
    subprocess.run(cmd, check=True)


def video_to_frames(
    video_path: Path,
    frames_dir: Path,
    ext: str = ".jpg",
):
    """Splits a video into frames.

    Args:
        video_path (Path): Filename of the input video.
        frames_dir (Path): Output directory where all the frames
            will be stored.
        ext (str): Image file extension, {'.png', '.jpg'}.
    """
    assert ext in {'.png', '.jpg'}
    video_name = Path(video_path).stem
    cmd = [
        'ffmpeg',
        '-i',
        f'{Path(video_path).as_posix()}',
        '-vf',  # video filters
        r'select=not(mod(n\,1))',  # select all frames, ~same as 'select=1'
        '-vsync',
        'vfr',  # original option in fede-vaccaro/fast-sr-unet
        # '-vsync', '0',  # should avoid drops or duplications
        '-q:v',
        '1',
        f'{ (Path(frames_dir) / video_name).as_posix() }_%4d{ext}',
    ]
    subprocess.run(cmd, check=True)

def preprocess(
    compressed_path: Path,
    dtype: str = "fp32",
    cuda_or_cpu = "cuda",
):
    compressed = Image.open(compressed_path)
    compressed = TF.pil_to_tensor(compressed)
    compressed = min_max_scaler(compressed)
    compressed = make_4times_downscalable(compressed)
    if len(compressed.shape) == 3:
        compressed = compressed.unsqueeze(0)

    if dtype == 'fp16':
        compressed = compressed.half()
    elif dtype not in {'fp32', 'int8'}:
        raise ValueError(
            f"Unknown dtype: {dtype}. Choose in {'fp32', 'fp16', 'int8'}."
        )

    return compressed.to(cuda_or_cpu)

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
            cuda_or_cpu=cuda_or_cpu
        )
        generated = gen(compressed)
        generated = postprocess(
            generated=generated,
            width_original=width_original,
            height_original=height_original
        )
        for _ in range(generated.shape[0]):
            generated_pil = TF.to_pil_image(generated.squeeze(0))
            generated_path = Path(
                generated_dir, f"generated_{id_counter:04d}.png"
            )
            generated_pil.save(generated_path)
            id_counter += 1


def frames_to_video(
    frames_dir: Path, video_path: Path, fps: int = 30
) -> None:
    """Convert a bunch of frames to a video.

    Frame names should follow a pattern like:
        frames_dir/
            - frame_0001.png
            - frame_0002.png
            - frame_0003.png
            - ...
    """
    video_name = '_'.join(Path(frames_dir).stem.split('_')[:-1])
    cmd = (
        f"ffmpeg -r {fps}"
        f" -i {(Path(frames_dir) / video_name).as_posix()}_%04d.png"
        f" -c:v libx264 -preset medium -crf 23"
        f" {video_path.as_posix()}"
    )
    subprocess.run(cmd.split(" "), check=True)


def vmaf(
    original_video_path: Path,
    generated_video_path: Path,
    fps: int = 30,
    from_minute="00",
    from_second="00",
    to_minute="00",
    to_second="03",
):
    """Computes the VMAF quality score between two videos."""

    cmd = (
        f"ffmpeg -nostats -loglevel 0"
        f" -r {fps} -i {original_video_path}"
        f" -r {fps} -i {generated_video_path}"
        f" -ss 00:{from_minute}:{from_second} -to 00:{to_minute}:{to_second}"
        f" -lavfi '[0:v]setpts=PTS-STARTPTS[reference];"
        f" [1:v]scale=-1:1080:flags=bicubic,setpts=PTS-STARTPTS[distorted];"
        f" [distorted][reference]libvmaf=log_fmt=xml:log_path=/dev/stdout'"
        f" -f null - | grep -i 'aggregateVMAF'"
    )
    return subprocess.run(
        cmd.split(" "), check=True, capture_output=True, text=True
    )


if __name__ == "__main__":
    cfg = get_default_config()
    vmaf_dir: Path = cfg.paths.artifacts_dir / "vmaf"

    original_video_path = vmaf_dir / "original.y4m"
    compressed_video_path = vmaf_dir / "compressed.mp4"
    generated_video_path = vmaf_dir / "generated.mp4"
    another_original_video_path = vmaf_dir / "original.mp4"

    original_frames_dir = vmaf_dir / "original_frames"
    compressed_frames_dir = vmaf_dir / "compressed_frames"
    generated_frames_dir = vmaf_dir / "generated_frames"
    vmaf_dir.mkdir(exist_ok=True, parents=False)
    original_frames_dir.mkdir(exist_ok=True, parents=False)
    compressed_frames_dir.mkdir(exist_ok=True, parents=False)
    generated_frames_dir.mkdir(exist_ok=True, parents=False)

    # load generator
    cuda_or_cpu = "cuda"
    model_name = "srunet"
    ckpt_path = Path(
        cfg.paths.artifacts_dir,
        "best_checkpoints",
        f"2022_12_19_{model_name}_4_318780.pth",
    )
    cfg.model.ckpt_path_to_resume = ckpt_path
    cfg.model.name = model_name
    gen = prepare_generator(cfg, device=cuda_or_cpu).eval()

    is_done = 0
    if is_done:
        compress_video(
            original_video_path=original_video_path,
            compressed_video_path=compressed_video_path
        )
        video_to_frames(
            video_path=original_video_path,
            frames_dir=original_frames_dir,
            ext=".png",
        )
        frames_to_video(
            frames_dir=original_frames_dir,
            video_path=another_original_video_path
        )

        video_to_frames(
            video_path=compressed_video_path,
            frames_dir=compressed_frames_dir,
            ext=".jpg",
        )
        eval_images(
            gen=gen,
            compressed_path_list=sorted(compressed_frames_dir.iterdir()),
            generated_dir=generated_frames_dir
        )
        frames_to_video(
            frames_dir=generated_frames_dir,
            video_path=generated_video_path
        )

    vmaf(
        original_video_path=another_original_video_path,
        generated_video_path=generated_video_path
    )



# f"ffmpeg -r {fps} -i img%03d.png -c:v libx264 -preset medium -crf 23 output.mp4"


# "ffmpeg -r 30 -i img%03d.png -c:v libx264 -preset medium -crf 23 output.mp4"

# vmaf_command = (
#     f"./ffmpeg -nostats -loglevel 0"
#     f" -r {fps} -i {dest_dir / (video_prefix + '.mp4')}"
#     f" -r {fps} -i {dest_dir / 'output_testing.mp4'}"
#     f" -ss 00:{from_minute}:{from_second_} -to 00:{to_minute}:{to_second_}"
#     f" -lavfi '[0:v]setpts=PTS-STARTPTS[reference];"
#     f" [1:v]scale=-1:{resolution_hq}:flags=bicubic,setpts=PTS-STARTPTS[distorted];"
#     f" [distorted][reference]libvmaf=log_fmt=xml:log_path=/dev/stdout'"
#     f" -f null - | grep -i 'aggregateVMAF'"
# )
