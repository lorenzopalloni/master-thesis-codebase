"""Data module"""

import functools
import itertools
import warnings
import json
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision.transforms.functional as F
from gifnoc import Gifnoc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import Image


class Stage(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


def compute_adjusted_dimension(an_integer: int) -> int:
    """Given an integer `an_integer`, return another integer that:
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


def draw_validation_fig(
    original_image: torch.Tensor,
    compressed_image: torch.Tensor,
    generated_image: torch.Tensor,
    figsize: Tuple[int, int] = (12, 5),
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


def compose(*functions):
    """Compose several functions together."""
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


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
    scale_factor: int = 2,
) -> Tuple[PIL.Image.Image, PIL.Image.Image]:
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


def lists_have_same_elements(a_list: List, another_list: List) -> bool:
    """Assure that two given lists have the same elements."""
    a_set = set(a_list)
    another_set = set(another_list)
    if len(a_set) != len(another_set):
        return False
    return len(a_set.difference(another_set)) == 0


def list_files(
    path: Path, extension: str, sort_ascending: bool = True
) -> List[Path]:
    """List files in a given directory with the same extension.

    By default, the result is provided in lexicographic order.
    """
    res = []
    for x in path.iterdir():
        if not x.is_dir():
            if x.suffix.lstrip('.') == extension.lstrip('.'):
                res.append(x)
            else:
                warnings.warn(
                    f'{x} has not been included since it has '
                    f'a different extension than {extension}.',
                    UserWarning,
                )
    if sort_ascending:
        return sorted(res)
    return res


def list_directories(path: Path, sort_ascending: bool = True) -> List[Path]:
    """List all the directories in a given path.

    By default, the result is provided in lexicographic order.
    """
    res = [x for x in path.iterdir() if x.is_dir()]
    if sort_ascending:
        return sorted(res)
    return res


def list_subdir_files(
    path: Path, extension: str, sort_ascending: bool = True
) -> List[Path]:
    """List all files in the second level directories of the given path.

    By default, the result is provided in lexicographic order.
    """
    res = itertools.chain.from_iterable(
        (
            list_files(i_dir, extension, sort_ascending=False)
            for i_dir in list_directories(path)
        )
    )
    if sort_ascending:
        return sorted(res)
    return list(res)


def min_max_scaler(
    tensor: torch.Tensor, tensor_min: float = 0.0, tensor_max: float = 255.0
) -> torch.Tensor:
    """Scales any value of a tensor between two given values."""
    return (tensor - tensor_min) / (tensor_max - tensor_min)


def inv_min_max_scaler(
    tensor: torch.Tensor, tensor_min: float = 0.0, tensor_max: float = 255.0
) -> torch.Tensor:
    """Inverse of the min_max_scaler function."""
    return (tensor * (tensor_max - tensor_min) + tensor_min).int()


def default_test_pipe(
    original_filepath: Path,
    compressed_filepath: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_image = Image.open(original_filepath)
    compressed_image = Image.open(compressed_filepath)

    return (
        min_max_scaler(F.pil_to_tensor(original_image)),
        min_max_scaler(F.pil_to_tensor(compressed_image)),
    )


def default_val_pipe(
    original_filepath: Path,
    compressed_filepath: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_image = Image.open(original_filepath)
    compressed_image = Image.open(compressed_filepath)

    original_patch, compressed_patch = random_crop_images(
        original_image=original_image,
        compressed_image=compressed_image,
    )

    return (
        min_max_scaler(F.pil_to_tensor(original_patch)),
        min_max_scaler(F.pil_to_tensor(compressed_patch)),
    )


def default_train_pipe(
    original_filepath: Path, compressed_filepath: Path
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_image = Image.open(original_filepath)
    compressed_image = Image.open(compressed_filepath)

    original_patch, compressed_patch = random_crop_images(
        original_image=original_image,
        compressed_image=compressed_image,
    )

    if np.random.random() < 0.5:
        original_patch = F.hflip(original_patch)
        compressed_patch = F.hflip(compressed_patch)

    return (
        min_max_scaler(F.pil_to_tensor(original_patch)),
        min_max_scaler(F.pil_to_tensor(compressed_patch)),
    )


def identity_pipe(original_filepath: Path, compressed_filepath: Path):
    return original_filepath, compressed_filepath


class ImageFilepathDataset(Dataset):
    def __init__(
        self,
        original_filepaths: List[Path],
        compressed_filepaths: List[Path],
        pipe: Callable = identity_pipe,
    ):
        self.original_filepaths = original_filepaths
        self.compressed_filepaths = compressed_filepaths
        self.pipe = pipe

    def __len__(self) -> int:
        _len = len(self.original_filepaths)
        assert len(self.compressed_filepaths) == _len
        return _len

    def __getitem__(self, i):
        original_fn, compressed_fn = (
            self.original_filepaths[i],
            self.compressed_filepaths[i],
        )
        return self.pipe(original_fn, compressed_fn)


def get_train_val_test_indexes(
    n: int,
    val_ratio: float = 0.025,
    test_ratio: float = 0.025,
    random_state: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Returns a partition of three arrays of random indexes.

    >>> a, b, c = get_train_val_test_indexes(800)
    >>> len(a), len(b), len(c)
    (760, 20, 20)

    Args:
        n (int): Number of indexes that will rane from `0` to `n - 1`.
        val_ratio (float, optional): Percentage of integers to
            allocate for the validation set, between `0.0` and `1.0`.
            Defaults to 0.025.
        test_ratio (float, optional): Percentage of integers to
            allocate for the test set, between `0.0` and `1.0`.
            Defaults to 0.025.
        random_state (int, optional): Random seed for replicability.
            Defaults to 42.

    Returns:
        Tuple[List[int], List[int], List[int]]: Partition of three list of
            integers choosen at random that range from `0` to `n - 1`,
            without replacement.
    """
    random_state = 42
    train_indexes, val_test_indexes = train_test_split(
        range(n), test_size=val_ratio + test_ratio, random_state=random_state
    )
    val_indexes, test_indexes = train_test_split(
        val_test_indexes,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state,
    )
    return train_indexes, val_indexes, test_indexes


def make_splits(
    original_frames_dir: Path,
    compressed_frames_dir: Path,
    save_path: Optional[Path] = None,
    val_ratio: float = 0.025,
    test_ratio: float = 0.025,
    random_state: int = 42,
) -> Dict[str, List[str]]:
    """Makes train, val, test splits.

    It needs two directory paths, one of original frames and
    another one for the compressed versions. It checks first
    that the two folders have the same folder names as children.

    Args:
        original_frames_dir (Path): Directory containing folders
            for all original videos that contain their frame each.
        compressed_frames_dir (Path): Directory containing folders
            for all compressed videos that contain their frame each.
        save_path (Optional[Path], optional): Filepath with ext .json
            to save the resulting dictionary. Defaults to None.

    Returns:
        Dict[str, List[str]]: Partition in lists of all the filepaths
            found in `original_frames_dir`, of the form
            {'train': [], 'val': [], 'test': []}.
    """
    o_subdirs = list_directories(original_frames_dir, sort_ascending=True)
    c_subdirs = list_directories(compressed_frames_dir, sort_ascending=True)
    assert len(o_subdirs) == len(c_subdirs) and all(
        x.name == y.name for x, y in zip(o_subdirs, c_subdirs)
    )
    n_videos = len(o_subdirs)
    train_indexes, val_indexes, test_indexes = get_train_val_test_indexes(
        n=n_videos,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    splits = {
        'train': [o_subdirs[idx].name for idx in train_indexes],
        'val': [o_subdirs[idx].name for idx in val_indexes],
        'test': [o_subdirs[idx].name for idx in test_indexes],
    }
    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as out_file:
            json.dump(splits, out_file)
    return splits


def get_splits(cfg: Gifnoc) -> Dict[str, List[str]]:
    """Fetches `data_dir/splits.json`.

    If it does not exist, it creates it, then returns it.

    Args:
        cfg (Gifnoc): Configuration object.

    Returns:
        Dict[str, List[str]]: Partition in lists of all the filepaths
            found in `original_frames_dir`, of the form
            {'train': [], 'val': [], 'test': []}.
    """
    splits_fp = Path(cfg.paths.data_dir, 'splits.json')
    if splits_fp.exists():
        with open(splits_fp, 'r', encoding='utf-8') as in_file:
            splits = json.load(in_file)
    else:
        splits = make_splits(
            original_frames_dir=cfg.paths.original_frames_dir,
            compressed_frames_dir=cfg.paths.compressed_frames_dir,
            save_path=splits_fp,
        )
    return splits


def make_dataset(
    cfg: Gifnoc,
    stage: Stage,
    pipe: Callable[[Path, Path], Any] = identity_pipe,
) -> ImageFilepathDataset:
    """Makes an ImageFilepathDataset for a specific stage.

    An ImageFilepathDataset is a custom pytorch dataset that
    is able to handle image filepath and a custom function
    called `pipe` to feed a model train/val/test pipeline.

    Available stages: {`train`, `val`, 'test'}

    Args:
        cfg (Gifnoc): Configuration object
        stage (str): A string representing the stage,
            the possible choices are `train`, `val`,
            and `test`.
        pipe: (Callable): Custom function to transform
            image filepaths to something else.

    Returns:
        ImageFilepathDataset: An ImageFilepathDataset for a specific stage.
    """
    splits = get_splits(cfg)
    original_filepaths = list(
        itertools.chain.from_iterable(
            list_files(
                Path(cfg.paths.original_frames_dir, path), extension='.png'
            )
            for path in splits[stage.value]
        )
    )
    compressed_filepaths = list(
        itertools.chain.from_iterable(
            list_files(
                Path(cfg.paths.compressed_frames_dir, path), extension='.jpg'
            )
            for path in splits[stage.value]
        )
    )
    return ImageFilepathDataset(
        original_filepaths=original_filepaths,
        compressed_filepaths=compressed_filepaths,
        pipe=pipe,
    )


def make_train_dataloader(cfg) -> DataLoader:
    """Makes a training dataloader."""
    dataset = make_dataset(cfg, Stage.TRAIN, default_train_pipe)
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.params.batch_size,
        num_workers=cfg.params.num_workers,
        shuffle=True,
        pin_memory=True,
    )


def make_val_dataloader(cfg) -> DataLoader:
    """Makes a validation dataloader."""
    dataset = make_dataset(cfg, Stage.VAL, default_val_pipe)
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.params.batch_size,
        num_workers=cfg.params.num_workers,
        shuffle=False,
        pin_memory=True,
    )


def make_test_dataloader(cfg) -> DataLoader:
    """Makes a testing dataloader."""
    dataset = make_dataset(cfg, Stage.TEST, default_test_pipe)
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.params.batch_size,
        num_workers=cfg.params.num_workers,
        shuffle=False,
        pin_memory=True,
    )


def make_dataloaders(
    cfg: Gifnoc,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Makes train/val/test dataloaders."""
    return (
        make_train_dataloader(cfg),
        make_val_dataloader(cfg),
        make_test_dataloader(cfg),
    )


if __name__ == "__main__":
    ...
