# TODO: refactor CustomPyTorchDataset using function composition

import itertools
import functools
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np


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


def index_handler(
    i: int, train_size: int, val_size: int, training: bool
) -> Union[int, ValueError]:
    if train_size == 0:
        raise ValueError('Empty training set.')
    if val_size == 0:
        raise ValueError('Empty validation set.')
    if training:
        return i % train_size
    return train_size + i % val_size


def get_train_and_val_sizes(n: int, train_pct: float = 0.8) -> Tuple[int, int]:
    train_size = int(round(n * train_pct, 0))
    val_size = n - train_size
    return train_size, val_size


def lists_have_same_elements(a: List, b: List) -> bool:
    set_a = set(a)
    set_b = set(b)
    if len(set_a) != len(set_b):
        return False
    return len(set_a.difference(set_b)) == 0


def list_files(
    path: Union[Path, str], extension: str = '.jpg', sort_result: bool = True
) -> List[Path]:
    """List files in a given directory with the same extension.

    By default the result is provided in lexicographic order.
    """
    res = [
        x
        for x in path.iterdir()
        if not x.is_dir() and x.name.endswith(extension)
    ]
    if sort_result:
        return sorted(res)
    return res


def list_directories(
    path: Union[Path, str], sort_result: bool = True
) -> List[Path]:
    """List all the directories in a given path.

    By default the result is provided in lexicographic order.
    """
    res = [x for x in path.iterdir() if x.is_dir()]
    if sort_result:
        return sorted(res)
    return res


def list_all_files_in_all_second_level_directories(
    path: Union[Path, str], extension: str = '.jpg', sort_result: bool = True
) -> List[Path]:
    """List all files in the second level directories of the given path.

    path. By default the result is provided in lexicographic order.
    """
    res = itertools.chain.from_iterable(
        [
            list_files(i_dir, extension, sort_result=False)
            for i_dir in list_directories(path)
        ]
    )
    if sort_result:
        return sorted(res)
    return res


def min_max_scaler(
    x: torch.Tensor, x_min: float = 0.0, x_max: float = 255.0
) -> torch.Tensor:
    return (x - x_min) / (x_max - x_min)


def inv_min_max_scaler(
    x: torch.Tensor, x_min: float = 0.0, x_max: float = 255.0
) -> torch.Tensor:
    return (x * (x_max - x_min) + x_min).int()


class CustomPyTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        original_frames_dir: Union[Path, str],
        encoded_frames_dir: Union[Path, str],
        patch_size: int = 96,  # not sure about this initial value
        training: bool = True,
        train_pct: float = 0.8,
        scale_factor: float = 2.0,
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
            train_pct (float, optional): Percentage of training data.
                Defaults to 0.8. Random portion of the whole data used
                for training. The remaining (1 - `train_pct`) of the data
                will be used as validation.
            scale_factor (float, optional):
                Scale factor between original and encoded frames.
                Defaults to 2.0, this means that original frames have a 2:1
                resolution ratio compared to encoded frames.
        """
        self.patch_size = patch_size
        self.training = training
        self.train_pct = train_pct

        self.original_filenames = (
            list_all_files_in_all_second_level_directories(
                original_frames_dir
            )
        )
        self.encoded_filenames = (
            list_all_files_in_all_second_level_directories(
                encoded_frames_dir
            )
        )
        self.num_examples = len(self.original_filenames)
        self.train_size, self.val_size = get_train_and_val_sizes(
            self.num_examples, self.train_pct
        )
        self.val_size = self.num_examples - self.train_size
        self.scale_factor = scale_factor

    def __len__(self):
        return self.train_size if self.training else self.val_size

    def __getitem__(self, i):
        i = index_handler(i, self.train_size, self.val_size, self.training)

        lq = torchvision.utils.Image.open(self.encoded_filenames[i])
        hq = torchvision.utils.Image.open(self.original_filenames[i])

        # crop
        ######################################################################
        w, h = lq.size
        a = get_starting_random_position(w, self.patch_size)
        b = get_starting_random_position(h, self.patch_size)
        lq_positions = (a, b, a + self.patch_size, b + self.patch_size)
        lq = lq.crop(lq_positions)
        hq_positions = tuple(
            map(lambda x: x * self.scale_factor, lq_positions)
        )
        hq = hq.crop(hq_positions)
        ######################################################################

        # hq = hq.resize((
        #     int(upscaling_factor * self.patch_size),
        #     int(upscaling_factor * self.patch_size)
        # ))

        if np.random.random() < 0.5:
            lq = F.hflip(lq)
            hq = F.hflip(hq)

        # custom_transform = torchvision.transforms.Compose([
        #     torchvision.transforms.ToTensor(),
        #     lambda x: (x - 0.5) * 2.0,
        # ])

        # lq, hq = custom_transform(lq), custom_transform(hq)

        return (
            min_max_scaler(F.pil_to_tensor(lq)),
            min_max_scaler(F.pil_to_tensor(hq))
        )
