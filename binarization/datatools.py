"""Collection of data-related utilities"""

from __future__ import annotations

import functools
import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision.transforms.functional as F


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


def get_starting_random_position(
    initial_position: int, patch_size: int
) -> int:
    """Chooses a random starting position on a given axis for a patch."""
    random_position = 0
    if initial_position > patch_size:
        initial_position -= patch_size
        random_position = np.random.randint(initial_position)
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

    compressed_image_positions = (
        random_width,
        random_height,
        random_width + patch_size,
        random_height + patch_size,
    )
    # scale positions
    original_image_positions = tuple(
        map(lambda x: x * scale_factor, compressed_image_positions)
    )
    original_patch = original_image.crop(original_image_positions)
    compressed_patch = compressed_image.crop(compressed_image_positions)
    return original_patch, compressed_patch


def compute_adjusted_dimension(an_integer: int) -> int:
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


def adjust_image_for_unet(image: torch.Tensor) -> torch.Tensor:
    """Pads until img_h and img_w are both divisible by 2 at least 4 times."""
    height, width = image.shape[-2], image.shape[-1]
    adjusted_height = compute_adjusted_dimension(height)
    adjusted_width = compute_adjusted_dimension(width)
    return F.pad(
        image,
        padding=[
            (adjusted_width - width) // 2,  # left/right
            (adjusted_height - height) // 2,  # top/bottom
        ],
    )

def inv_adjust_image_for_unet(
    original: torch.Tensor, generated: torch.Tensor
) -> torch.Tensor:
    """Crops as much as needed to invert `adjust_image_for_unet`."""
    height_original, width_original = original.shape[-2], original.shape[-1]
    height_generated, width_generated = (
        generated.shape[-2],
        generated.shape[-1],
    )
    height_offset = (height_generated - height_original) // 2
    width_offset = (width_generated - width_original) // 2
    return F.crop(
        generated, height_offset, width_offset, height_original, width_original
    )


def process_raw_generated(
    original: torch.Tensor, generated: torch.Tensor
) -> torch.Tensor:
    """Postprocesses outputs from super-resolution generator models"""
    generated = inv_min_max_scaler(generated)
    generated = generated.clip(0, 255)
    generated = generated / 255.0
    return inv_adjust_image_for_unet(original=original, generated=generated)


def draw_validation_fig(
    original_image: torch.Tensor,
    compressed_image: torch.Tensor,
    generated_image: torch.Tensor,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Draws three images in a row with matplotlib."""
    original_image_pil = F.to_pil_image(original_image)
    compressed_image_pil = F.to_pil_image(compressed_image)
    generated_image_pil = F.to_pil_image(generated_image)
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
    """Lists files in a given directory with the same extension.

    By default, the result is provided in lexicographic order.
    """
    res = []
    for child_path in path.iterdir():
        if child_path.is_dir(): continue
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
    w, h = compressed_image_width, compressed_image_height
    average_patches_per_image = round((w * h) / (patch_size ** 2))  # 56
    average_available_patches = buffer_size * average_patches_per_image
    n_batches_per_buffer = round((average_available_patches / factor) / batch_size)
    return n_batches_per_buffer

if __name__ == '__main__':
    print(estimate_n_batches_per_buffer())
