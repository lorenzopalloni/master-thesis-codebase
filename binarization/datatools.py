"""Collection of data-related utilities"""

from __future__ import annotations

import functools
import itertools
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def compose(*functions):
    """Composes several functions together."""
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


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


def make_4times_downscalable(image: torch.Tensor) -> torch.Tensor:
    """Pads until img_h and img_w are both divisible by 2 at least 4 times."""
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


def preprocess(
    compressed_path: Path,
    dtype: str = "fp32",
    cuda_or_cpu="cuda",
):
    """Preprocesses a compressed frame for evaluation."""
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


def get_starting_random_position(
    initial_position: int, patch_size: int
) -> int:
    """Chooses a random starting position on a given axis for a patch."""
    random_position = 0
    if initial_position > patch_size:
        initial_position -= patch_size
        random_position = int(np.random.randint(initial_position))
    return random_position


def random_crop_images(
    original_image: PIL.Image.Image,
    compressed_image: PIL.Image.Image,
    patch_size: int = 96,
    scale_factor: int = 4,
) -> tuple[PIL.Image.Image, PIL.Image.Image]:
    """Randomly crops two images.

    Crops at a random position `compressed_image`, taking a square
    (`patch_size`, `patch_size`). Then it crops `original_image`
    taking a square of dimensions:
    (`patch_size` * `scale_factor`, `patch_size` * `scale_factor`).
    """
    width, height = compressed_image.size
    random_width = get_starting_random_position(width, patch_size)
    random_height = get_starting_random_position(height, patch_size)

    compressed_image_positions = [
        random_width,
        random_height,
        random_width + patch_size,
        random_height + patch_size,
    ]
    # scale positions for the original image
    original_image_positions = [
        compressed_position * scale_factor
        for compressed_position in compressed_image_positions
    ]
    original_patch = original_image.crop(original_image_positions)
    compressed_patch = compressed_image.crop(compressed_image_positions)
    return original_patch, compressed_patch


def draw_validation_fig(
    original_image: torch.Tensor,
    compressed_image: torch.Tensor,
    generated_image: torch.Tensor,
    figsize: tuple[int, int] = (36, 15),
    save_path: Path | None = None,
) -> plt.Figure:
    """Draws three images in a row with matplotlib."""
    original_image_pil = TF.to_pil_image(original_image)
    compressed_image_pil = TF.to_pil_image(compressed_image)
    generated_image_pil = TF.to_pil_image(generated_image)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1.imshow(original_image_pil)
    ax1.set_title('high quality')
    ax1.axis('off')
    ax2.imshow(generated_image_pil)
    ax2.set_title('reconstructed')
    ax2.axis('off')
    ax3.imshow(compressed_image_pil)
    ax3.set_title('low quality')
    ax3.axis('off')
    fig.subplots_adjust(
        top=1.0, bottom=0.0, right=1.0, left=0.0, hspace=0.0, wspace=0.0
    )
    if save_path is not None:
        fig.savefig(save_path)
    return fig


def lists_have_same_elements(a_list: list, another_list: list) -> bool:
    """Assures that two given lists have the same elements."""
    a_set = set(a_list)
    another_set = set(another_list)
    if len(a_set) != len(another_set):
        return False
    return len(a_set.difference(another_set)) == 0


def list_files(
    path: Path, extensions: str | list[str] = '', sort_ascending: bool = True
) -> list[Path]:
    """Lists files in a given directory.

    By default, the result is provided in lexicographic order.
    """
    res = []
    for child_path in path.iterdir():
        if child_path.is_dir():
            continue
        if extensions and child_path.suffix.lower() not in extensions:
            warnings.warn(
                f'{child_path} has no valid extension ({extensions}).',
                UserWarning,
            )
            continue
        res.append(child_path)
    if sort_ascending:
        return sorted(res)
    return res


def list_directories(path: Path, sort_ascending: bool = True) -> list[Path]:
    """Lists all the directories in a given path.

    By default, the result is provided in lexicographic order.
    """
    res = [x for x in path.iterdir() if x.is_dir()]
    if sort_ascending:
        return sorted(res)
    return res


def list_subdir_files(
    path: Path, extensions: str | list[str] = '', sort_ascending: bool = True
) -> list[Path]:
    """lists all files in the second level directories of the given path.

    By default, the result is provided in lexicographic order.
    """
    res = itertools.chain.from_iterable(
        (
            list_files(i_dir, extensions, sort_ascending=False)
            for i_dir in list_directories(path)
        )
    )
    if sort_ascending:
        return sorted(res)
    return list(res)


def estimate_n_batches_per_buffer(
    factor: float = 3.0,
    buffer_size: int = 16,
    compressed_image_width: int = 944,
    compressed_image_height: int = 544,
    batch_size: int = 14,
    patch_size: int = 96,
) -> int:
    """Roughly estimates a good number of batches per buffer."""
    average_patches_per_image = round(
        (compressed_image_width * compressed_image_height) / (patch_size**2)
    )  # 56
    average_available_patches = buffer_size * average_patches_per_image
    n_batches_per_buffer = round(
        (average_available_patches / factor) / batch_size
    )
    return n_batches_per_buffer


def tensor_to_numpy(tensor_img: torch.Tensor) -> npt.NDArray[np.uint8]:
    """Casts an array from torch to numpy."""
    tensor_img = inv_min_max_scaler(tensor_img.squeeze(0))
    tensor_img = tensor_img.permute(1, 2, 0)
    numpy_img = tensor_img.byte().cpu().numpy()
    numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
    return numpy_img


def concatenate_images(
    img1: torch.Tensor,
    img2: torch.Tensor,
    crop: bool = False,
) -> torch.Tensor:
    """Concatenates two images.

    Args:
        img1 (torch.Tensor): an image.
        img2 (torch.Tensor): another image.
        crop (bool, optional): flag to crop 1/4 of each image before
            contatenating them. Defaults to False.

    Returns:
        torch.Tensor: an image that is the concatenation of the two
            given as inputs.
    """
    assert (
        img1.shape[-1] == img2.shape[-1]
    ), f"{img1.shape=} not equal to {img2.shape=}."
    if crop:
        width = img1.shape[-1]
        w_4 = width // 4
        img1 = img1[:, :, :, w_4 : w_4 * 3]
        img2 = img2[:, :, :, w_4 : w_4 * 3]
    return torch.cat([img1, img2], dim=3)


def tensor_image_interpolation(
    tensor_img: torch.Tensor, scale_factor: int = 4, mode: str = "bilinear"
) -> torch.Tensor:
    """Upscales by interpolation a given image."""
    return torch.clip(
        F.interpolate(
            tensor_img,
            scale_factor=scale_factor,
            mode=mode,
        ),
        min=0,
        max=1,
    )
