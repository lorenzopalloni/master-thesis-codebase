"""Script to evaluate a video with a super-resolution model"""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from gifnoc import Gifnoc
from PIL import Image
from tqdm import tqdm

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
    generated = inv_min_max_scaler(generated)
    generated = generated.clip(0, 255)
    generated = generated / 255.0
    return inv_make_4times_downscalable(
        generated=generated,
        width_original=width_original,
        height_original=height_original,
    )


def compress(
    input_fn: Path | str,
    output_fn: Path | str,
    crf: int = 23,
    scale_factor: int = 4,
):
    """Compresses a video.

    Note: do not worry about the following warning (source: google it):
    "deprecated pixel format used, make sure you did set range correctly"

    Args:
        input_fn (Union[Path, str]): Filename of the input video.
        output_fn (Union[Path, str]): Filename of the compressed video.
        crf (int, optional): Constant Rate Factor. Defaults to 23.
        scale_factor (int): Scale factor. Defaults to 4.
    """
    scaled_w = f"'iw/{scale_factor}-mod(iw/{scale_factor},2)'"
    scaled_h = f"'ih/{scale_factor}-mod(ih/{scale_factor},2)'"
    cmd = [
        'ffmpeg',
        '-i',
        f'{input_fn}',
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
        f'{output_fn}',
    ]
    subprocess.run(cmd, check=True)


def video_to_frames(
    input_fn: Path | str,
    output_dir: Path | str,
    ext: str = ".jpg",
):
    """Splits a video into frames.

    Args:
        input_fn (Union[Path, str]): Filename of the input video.
        output_dir (Union[Path, str]): Output directory where all the frames
        will be stored.
        ext (str): Image file extension, {'.png', '.jpg'}.
    """
    assert ext in {'.png', '.jpg'}
    cmd = [
        'ffmpeg',
        '-i',
        f'{input_fn}',
        '-vf',  # video filters
        r'select=not(mod(n\,1))',  # select all frames, ~same as 'select=1'
        '-vsync',
        'vfr',  # original option in fede-vaccaro/fast-sr-unet
        # '-vsync', '0',  # should avoid drops or duplications
        '-q:v',
        '1',
        f'{ (Path(output_dir) / Path(input_fn).stem).as_posix() }_%4d{ext}',
    ]
    subprocess.run(cmd, check=True)


def eval_images(
    gen: torch.nn.Module,
    compressed_path_list: list[Path],
    generated_dir: Path,
    cfg: Gifnoc = None,
    dtype: str = "fp32",
    cuda_or_cpu: str = "cuda",
):
    if cfg is None:
        cfg = get_default_config()
    if cuda_or_cpu.startswith("cuda"):
        cuda_or_cpu = prepare_cuda_device(0)

    for i, compressed_path in enumerate(compressed_path_list):
        compressed = Image.open(compressed_path)
        compressed = min_max_scaler(compressed)
        compressed = make_4times_downscalable(compressed).unsqueeze(0)

        if dtype == 'fp16':
            compressed = compressed.half()
        elif dtype not in {'fp32', 'int8'}:
            raise ValueError(
                f"Unknown dtype: {dtype}. Choose in {'fp32', 'fp16', 'int8'}."
            )

        generated = gen(compressed.to(cuda_or_cpu)).clip(0, 1).cpu()
        generated = inv_min_max_scaler(generated)
        generated = generated.clip(0, 255)
        generated = generated / 255.0
        generated_pil = TF.to_pil_image(generated)
        generated_path = Path(generated_dir, f"generated_{i:04d}.png")
        generated_pil.save(generated_path)
    # return generated
    # return inv_make_4times_downscalable(generated=generated)


def frames_to_video(
    generated_dir: Path, output_video_path: Path, fps: int = 30
) -> None:
    """Convert a bunch of frames to a video.

    Frame paths should follow a pattern like frame%03d.png
    """
    cmd = (
        f"ffmpeg -r {fps}"
        f" -i {Path(generated_dir, 'generated').as_posix()}_%04d.png"
        f" -c:v libx264 -preset medium -crf 23"
        f" {output_video_path.as_posix()}"
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

    cmd = (
        f"./ffmpeg -nostats -loglevel 0"
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
    input_video_path = ""
    compressed_video_path = ""
    vmaf_dir = cfg.paths.artifacts_dir / "vmaf"
    original_frames_dir = vmaf_dir / "original_frames"
    compressed_frames_dir = vmaf_dir / "compressed_frames"
    generated_frames_dir = vmaf_dir / "generated_frames"

    compress(input_fn=input_video_path, output_fn=compressed_video_path)
    video_to_frames(
        input_fn=input_video_path,
        output_dir=vmaf_dir / "original_frames",
        ext=".png",
    )
    video_to_frames(
        input_fn=compressed_video_path,
        output_dir=vmaf_dir / "compressed_frames",
        ext=".jpg",
    )
    eval_images()
    frames_to_video()
    vmaf()


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


f"ffmpeg -r {fps} -i img%03d.png -c:v libx264 -preset medium -crf 23 output.mp4"


"ffmpeg -r 30 -i img%03d.png -c:v libx264 -preset medium -crf 23 output.mp4"

vmaf_command = (
    f"./ffmpeg -nostats -loglevel 0"
    f" -r {fps} -i {dest_dir / (video_prefix + '.mp4')}"
    f" -r {fps} -i {dest_dir / 'output_testing.mp4'}"
    f" -ss 00:{from_minute}:{from_second_} -to 00:{to_minute}:{to_second_}"
    f" -lavfi '[0:v]setpts=PTS-STARTPTS[reference];"
    f" [1:v]scale=-1:{resolution_hq}:flags=bicubic,setpts=PTS-STARTPTS[distorted];"
    f" [distorted][reference]libvmaf=log_fmt=xml:log_path=/dev/stdout'"
    f" -f null - | grep -i 'aggregateVMAF'"
)
