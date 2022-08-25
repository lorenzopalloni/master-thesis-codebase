# TODO: refactor CustomPyTorchDataset using function composition

import functools
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from gifnoc import Gifnoc
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import Image


def compute_adjusted_dimension(an_integer: int) -> int:
    """Given an integer `an_integer`, return another integer that:
    - is greater than `an_integer`
    - is divisible at least four times by 2
    - is the closest to `an_integer`

    Useful for adapting the size of an image to feed a UNet-like architecture.

    Args:
        an_integer (int): an integer greater than 0.

    Returns:
        int: an integer with the properties described above.
    """
    assert (
        an_integer > 0
    ), f"Input should be > 0, but `{an_integer}` was provided."
    if an_integer % 2 != 0:
        an_integer += 1
    while an_integer / 2**4 % 2 != 0:
        an_integer += 2
    return an_integer


def adjust_image_for_unet(image: torch.Tensor) -> torch.Tensor:
    """Pad until image height and width are divisible by 2 at least 4 times"""
    _, height, width = image.shape
    adjusted_height = compute_adjusted_dimension(height)
    adjusted_width = compute_adjusted_dimension(width)
    return F.pad(
        image,
        padding=[
            (adjusted_width - width) // 2,  # left/right
            (adjusted_height - height) // 2,  # top/bottom
        ],
    )


def draw_validation_fig(
    original_image: torch.Tensor,
    compressed_image: torch.Tensor,
    generated_image: torch.Tensor,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Draw a matplotlib figure useful for validating the training process
    of a representation model.
    """
    original_image_pil = F.to_pil_image(original_image)
    compressed_image_pil = F.to_pil_image(compressed_image)
    generated_image_pil = F.to_pil_image(generated_image)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1.imshow(original_image_pil)
    ax1.set_title('high quality')
    ax2.imshow(generated_image_pil)
    ax2.set_title('reconstructed')
    ax3.imshow(compressed_image_pil)
    ax3.set_title('low quality')
    return fig


def compose(*functions):
    """Compose several functions together."""
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


def get_starting_random_position(x: int, patch_size: int) -> int:
    """Random starting position on a given axis for a patch."""
    res = 0
    if x > patch_size:
        x -= patch_size
        res = np.random.randint(x)
    return res


def lists_have_same_elements(a_list: List, another_list: List) -> bool:
    """Assure that two given lists have the same elements."""
    a_set = set(a_list)
    another_set = set(another_list)
    if len(a_set) != len(another_set):
        return False
    return len(a_set.difference(another_set)) == 0


def list_files(
    path: Path, extension: str = '.jpg', sort_result: bool = True
) -> List[Path]:
    """List files in a given directory with the same extension.

    By default, the result is provided in lexicographic order.
    """
    res = [
        x
        for x in path.iterdir()
        if not x.is_dir() and x.name.endswith(extension)
    ]
    if sort_result:
        return sorted(res)
    return res


def list_directories(path: Path, sort_result: bool = True) -> List[Path]:
    """List all the directories in a given path.

    By default, the result is provided in lexicographic order.
    """
    res = [x for x in path.iterdir() if x.is_dir()]
    if sort_result:
        return sorted(res)
    return res


def list_all_files_in_all_second_level_directories(
    path: Path, extension: str = '.jpg', sort_result: bool = True
) -> List[Path]:
    """List all files in the second level directories of the given path.

    By default, the result is provided in lexicographic order.
    """
    res = itertools.chain.from_iterable(
        [
            list_files(i_dir, extension, sort_result=False)
            for i_dir in list_directories(path)
        ]
    )
    if sort_result:
        return sorted(res)
    return list(res)


def min_max_scaler(
    x: torch.Tensor, x_min: float = 0.0, x_max: float = 255.0
) -> torch.Tensor:
    return (x - x_min) / (x_max - x_min)


def inv_min_max_scaler(
    x: torch.Tensor, x_min: float = 0.0, x_max: float = 255.0
) -> torch.Tensor:
    return (x * (x_max - x_min) + x_min).int()


class CustomPyTorchDataset(Dataset):
    def __init__(
        self,
        original_frames_dir: Union[Path, str],
        encoded_frames_dir: Union[Path, str],
        patch_size: int = 96,  # not sure about this initial value
        training: bool = True,
        scale_factor: int = 4,
    ):
        """Custom PyTorch Dataset loader for the training phase. Yield a pair
        (x, y), where x is the encoded version of the original image y.

        Args:
            original_frames_dir (Union[Path, str]): Original frames directory.
            encoded_frames_dir (Union[Path, str]): Encoded frames directory.
            patch_size (int): Width/height of a training patch.
                A training patch will be choosen at random from a given frame.
            training (bool, optional): Flag for training vs evaluation phase.
                Defaults to False.
            scale_factor (Optional[int]): Scale factor between original and
                encoded frames. Defaults to 4, this means that original frames
                have a 4:1 resolution ratio compared to encoded frames.
        """
        self.patch_size = patch_size
        self.training = training
        self.original_filenames = (
            list_all_files_in_all_second_level_directories(
                Path(original_frames_dir)
            )
        )
        self.encoded_filenames = (
            list_all_files_in_all_second_level_directories(
                Path(encoded_frames_dir)
            )
        )
        self.num_examples = len(self.original_filenames)
        self.scale_factor = scale_factor

    def __len__(self):
        return self.num_examples

    def __getitem__(self, i):

        compressed_image = Image.open(self.encoded_filenames[i])
        original_image = Image.open(self.original_filenames[i])

        if not self.training:
            return (
                min_max_scaler(F.pil_to_tensor(compressed_image)),
                min_max_scaler(F.pil_to_tensor(original_image)),
            )

        # crop
        ######################################################################
        w, h = compressed_image.size
        a = get_starting_random_position(w, self.patch_size)
        b = get_starting_random_position(h, self.patch_size)
        compressed_image_positions = (
            a,
            b,
            a + self.patch_size,
            b + self.patch_size,
        )
        compressed_patch = compressed_image.crop(compressed_image_positions)
        original_image_positions = tuple(
            map(lambda x: x * self.scale_factor, compressed_image_positions)
        )
        original_patch = original_image.crop(original_image_positions)
        ######################################################################

        # original_image = original_image.resize((
        #     int(upscaling_factor * self.patch_size),
        #     int(upscaling_factor * self.patch_size)
        # ))

        if np.random.random() < 0.5:
            compressed_patch = F.hflip(compressed_patch)
            original_patch = F.hflip(original_patch)

        # custom_transform = torchvision.transforms.Compose([
        #     torchvision.transforms.ToTensor(),
        #     lambda x: (x - 0.5) * 2.0,
        # ])

        # compressed_image, original_image = custom_transform(compressed_image), custom_transform(original_image)

        return (
            min_max_scaler(F.pil_to_tensor(compressed_patch)),
            min_max_scaler(F.pil_to_tensor(original_patch)),
        )


def make_train_dataloader(
    original_frames_dir: Path,
    encoded_frames_dir: Path,
    patch_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = CustomPyTorchDataset(
        original_frames_dir=original_frames_dir,
        encoded_frames_dir=encoded_frames_dir,
        patch_size=patch_size,
        training=True,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )


def make_val_dataloader(
    original_frames_dir: Path,
    encoded_frames_dir: Path,
    patch_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = CustomPyTorchDataset(
        original_frames_dir=original_frames_dir,
        encoded_frames_dir=encoded_frames_dir,
        patch_size=patch_size,
        training=False,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )


def make_test_dataloader(
    original_frames_dir: Path,
    encoded_frames_dir: Path,
    patch_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = CustomPyTorchDataset(
        original_frames_dir=original_frames_dir,
        encoded_frames_dir=encoded_frames_dir,
        patch_size=patch_size,
        training=False,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )


def make_dataloaders(
    cfg: Gifnoc,
    patch_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        make_train_dataloader(
            original_frames_dir=cfg.paths.train_original_frames_dir,
            encoded_frames_dir=cfg.paths.train_encoded_frames_dir,
            patch_size=patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        make_val_dataloader(
            original_frames_dir=cfg.paths.val_original_frames_dir,
            encoded_frames_dir=cfg.paths.val_encoded_frames_dir,
            patch_size=patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        make_test_dataloader(
            original_frames_dir=cfg.paths.test_original_frames_dir,
            encoded_frames_dir=cfg.paths.test_encoded_frames_dir,
            patch_size=patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    )


if __name__ == "__main__":
    ...
