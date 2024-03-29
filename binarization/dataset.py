# pylint: disable=invalid-name,missing-function-docstring,too-many-arguments,missing-class-docstring
"""Custom dataloaders"""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
import torchvision.transforms.functional as F
from gifnoc import Gifnoc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import Image

from binarization.config import get_default_config
from binarization.datatools import (
    compose,
    list_directories,
    list_files,
    make_4times_downscalable,
    min_max_scaler,
    random_crop_images,
)


def get_train_val_test_indexes(
    n: int,
    val_ratio: float = 0.025,
    test_ratio: float = 0.025,
    random_state: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Returns a partition of three arrays of random indexes.

    >>> a, b, c = get_train_val_test_indexes(800)
    >>> len(a), len(b), len(c)
    (760, 20, 20)

    Args:
        n (int): Number of indexes that will range from `0` to `n - 1`.
        val_ratio (float, optional): Percentage of integers to
            allocate for the validation set, between `0.0` and `1.0`.
            Defaults to 0.025.
        test_ratio (float, optional): Percentage of integers to
            allocate for the test set, between `0.0` and `1.0`.
            Defaults to 0.025.
        random_state (int, optional): Random seed for replicability.
            Defaults to 42.

    Returns:
        tuple[list[int], list[int], list[int]]: Partition of three list of
            integers choosen at random that range from `0` to `n - 1`,
            without replacement.
    """
    trainVal_indexes, test_indexes = train_test_split(
        range(n), test_size=test_ratio, random_state=random_state
    )
    train_indexes, val_indexes = train_test_split(
        trainVal_indexes,
        test_size=val_ratio / (1 - test_ratio),
        random_state=random_state,
    )
    return train_indexes, val_indexes, test_indexes


def make_splits(
    original_frames_dir: Path,
    compressed_frames_dir: Path,
    save_path: Path | None = None,
    val_ratio: float = 0.025,
    test_ratio: float = 0.025,
    random_state: int = 42,
) -> dict[str, list[str]]:
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
        val_ratio (float): percentage of data assigned to the validation
            set. Defaults to 0.025.
        test_ratio (float): percentage of data assigned to the test
            set. Defaults to 0.025.
        random_state (int): random seed. Defaults to 42.

    Returns:
        dict[str, list[str]]: Partition in lists of all the filepaths
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


def get_splits(
    cfg: Gifnoc, exist_ok: bool = True, random_state: int = 42
) -> dict[str, list[str]]:
    """Fetches `<data_dir>/splits.json`.

    Returns `<data_dir>/splits.json` if it does exist.
    Otherwise, this function creates it, then returns it.

    Args:
        cfg (Gifnoc): Configuration object.
        exist_ok (bool): Load existing split .json file if already exists.
            Defaults to True.
        random_state (int): Random seed. Defaults to 42.

    Returns:
        dict[str, list[str]]: Partition in lists of all the filepaths
            found in `original_frames_dir`, of the form
            {'train': [], 'val': [], 'test': []}.
    """
    splits_fp = Path(cfg.paths.data_dir, 'splits.json')
    if splits_fp.exists() and exist_ok:
        with open(splits_fp, 'r', encoding='utf-8') as in_file:
            splits = json.load(in_file)
    else:
        splits = make_splits(
            original_frames_dir=cfg.paths.original_frames_dir,
            compressed_frames_dir=cfg.paths.compressed_frames_dir,
            save_path=splits_fp,
            random_state=random_state,
        )
    return splits


class Stage(Enum):
    """Model stages: {train, val, test}."""

    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


def identity_pipe(x, *args):
    return (x,) + args if args else x


class BufferGenerator:
    def __init__(
        self,
        init_list: list[Any],
        buffer_size: int,
        shuffle: bool,
        pipe: Callable[[Any], Any] = identity_pipe,
        n_iterations: int = 1,
    ):
        """Serves buffered chunk of items from a list of items.

        Args:
            init_list (list[Any]): A list of items
            buffer_size (int): Amount of element per buffer.
            shuffle (bool): If True, items are randomly permuted once
                at the beginning of each iteration.
            pipe (Callable[[Any], Any], optional): Function applied to
                each single item before entering the buffer. Defaults
                to identity_pipe.
            n_iterations (int, optional): Number of iterations (> 0).
                For each iteration, `len(init_list) // buffer_size` number
                of buffers will be served. Defaults to 1.
        """
        self.items = init_list
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.pipe = pipe
        self.n_iterations = n_iterations
        self.iteration_counter = 0

        self.n_items = len(self.items)
        self.indexes = np.arange(self.n_items)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.start_idx = 0

    def __len__(self):
        return (self.n_items // self.buffer_size) * self.n_iterations

    def end_iteration_step(self):
        self.iteration_counter += 1
        if self.iteration_counter >= self.n_iterations:
            raise StopIteration
        self.start_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_idx > self.n_items - self.buffer_size:
            self.end_iteration_step()
        indexes = self.indexes[
            self.start_idx : self.start_idx + self.buffer_size
        ]
        buffer = [self.pipe(self.items[idx]) for idx in indexes]
        self.start_idx += self.buffer_size
        return buffer


def get_paired_paths(cfg: Gifnoc, stage: Stage) -> list[tuple[Path, Path]]:
    """Lists pairs of original/compressed paths for a specific stage.

    Args:
        cfg (Gifnoc): Configuration object.
        stage (Stage): Choose in {`Stage.TRAIN`, `Stage.VAL`, `Stage.TEST`}.

    Returns:
        list[tuple[Path, Path]]: Pairs of original/compressed frame paths,
            such as [
                (out_path_1, in_path_1),
                (out_path_2, in_path_2),
                ...
            ].
    """
    splits = get_splits(cfg)
    out_paths = []
    in_paths = []
    for video_name in splits[stage.value]:
        out_frames_dir = Path(cfg.paths.original_frames_dir, video_name)
        out_paths.extend(
            list_files(out_frames_dir, extensions=['.png', '.jpg'])
        )
        in_frames_dir = Path(cfg.paths.compressed_frames_dir, video_name)
        in_paths.extend(list_files(in_frames_dir, extensions=['.png', '.jpg']))

    assert len(out_paths) == len(in_paths)
    return list(zip(out_paths, in_paths))


def default_train_pipe(
    original_image: PIL.Image.Image,
    compressed_image: PIL.Image.Image,
    scale_factor: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_patch, compressed_patch = random_crop_images(
        original_image=original_image,
        compressed_image=compressed_image,
        scale_factor=scale_factor,
    )
    if np.random.random() < 0.5:
        original_patch = F.hflip(original_patch)
        compressed_patch = F.hflip(compressed_patch)
    return (
        min_max_scaler(F.pil_to_tensor(original_patch)),
        min_max_scaler(F.pil_to_tensor(compressed_patch)),
    )


def default_val_pipe(
    original_image: PIL.Image.Image,
    compressed_image: PIL.Image.Image,
    scale_factor: int = 4,
    random_seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    np.random.seed(random_seed)  # crop at random positions but always the same
    original_patch, compressed_patch = random_crop_images(
        original_image=original_image,
        compressed_image=compressed_image,
        scale_factor=scale_factor,
    )
    np.random.seed(None)
    return (
        min_max_scaler(F.pil_to_tensor(original_patch)),
        min_max_scaler(F.pil_to_tensor(compressed_patch)),
    )


def default_test_pipe(
    original_image: PIL.Image.Image,
    compressed_image: PIL.Image.Image,
    scale_factor: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    del scale_factor
    pipe = compose(F.pil_to_tensor, min_max_scaler, make_4times_downscalable)
    return F.pil_to_tensor(original_image), pipe(compressed_image)


class BatchGenerator:
    def __init__(
        self,
        cfg: Gifnoc,
        stage: Stage,
        buffer_size: int,
        n_batches_per_buffer: int,
        batch_size: int,
    ):
        """Generates batches of patches taken from chunk of prefetched frames.

        Frames come in pairs original/compressed.
        By default, `Stage.VAL` and `Stage.TEST` force:
            1. `n_batches_per_buffer` := `buffer_size`
            2. `batch_size` := 1

        Both image selection and cropping take place completely
        at random, and can lead to duplicate patches (this is an
        expected behaviour).

        Note that the random position for cropping is the same in
        each pair of images.

        Args:
            cfg (Gifnoc): Configuration object.
            stage (Stage): Choose in {`Stage.TRAIN`, `Stage.VAL`, `Stage.TEST`}.
            buffer_size (int): Amount of frames loaded per buffer.
            n_batches_per_buffer (int): Number of batches (each
                consisting of `batch_size` patches) that can be drawn
                from a frame buffer before refreshing it with new frames.
            batch_size (int): Batch size.

        Returns:
            Iterable[batch]: Batch data-loader.
        """
        paired_paths = get_paired_paths(cfg=cfg, stage=stage)

        self.shuffle = stage == Stage.TRAIN
        self.stage = stage
        self.buffer_size = buffer_size

        def path_to_frame_pipe(
            paired_paths: tuple[Path, Path]
        ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
            original_path, compressed_path = paired_paths
            return Image.open(original_path), Image.open(compressed_path)

        self.buffer_generator = BufferGenerator(
            init_list=paired_paths,
            buffer_size=self.buffer_size,
            shuffle=self.shuffle,
            pipe=path_to_frame_pipe,
        )
        self.buffer = next(self.buffer_generator)
        self.batch_idx = 0  # tracks batch index in the current buffer

        if self.shuffle:
            self.n_batches_per_buffer = n_batches_per_buffer
            self.batch_size = batch_size
        else:
            self.n_batches_per_buffer = buffer_size
            self.batch_size = 1

        self.scale_factor = cfg.params.scale_factor

    def __len__(self):
        return len(self.buffer_generator) * self.n_batches_per_buffer

    @property
    def frame_to_patch_pipes(self):
        return {
            Stage.TRAIN: default_train_pipe,
            Stage.VAL: default_val_pipe,
            Stage.TEST: default_test_pipe,
        }

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch_idx == self.n_batches_per_buffer:
            self.buffer = next(self.buffer_generator)
            self.batch_idx = 0

        original_batch = []
        compressed_batch = []
        for _ in range(self.batch_size):
            random_idx = (
                np.random.randint(0, self.buffer_size, dtype=np.uint)
                if self.shuffle
                else self.batch_idx
            )
            original_image, compressed_image = self.buffer[random_idx]
            original_patch, compressed_patch = self.frame_to_patch_pipes[
                self.stage
            ](original_image, compressed_image, scale_factor=self.scale_factor)
            original_batch.append(original_patch)
            compressed_batch.append(compressed_patch)

        self.batch_idx += 1
        return torch.stack(original_batch), torch.stack(compressed_batch)


def get_train_batches(cfg):
    return BatchGenerator(
        cfg=cfg,
        stage=Stage.TRAIN,
        buffer_size=cfg.params.buffer_size,
        n_batches_per_buffer=cfg.params.n_batches_per_buffer,
        batch_size=cfg.params.batch_size,
    )


def get_val_batches(cfg):
    return BatchGenerator(
        cfg=cfg,
        stage=Stage.VAL,
        buffer_size=cfg.params.buffer_size,
        n_batches_per_buffer=cfg.params.n_batches_per_buffer,
        batch_size=cfg.params.batch_size,
    )


def get_test_batches(cfg: Gifnoc = None):
    if cfg is None:
        cfg = get_default_config()
    return BatchGenerator(
        cfg=cfg,
        stage=Stage.TEST,
        buffer_size=cfg.params.buffer_size,
        n_batches_per_buffer=cfg.params.n_batches_per_buffer,
        batch_size=cfg.params.batch_size,
    )


def get_batches(cfg):
    return (
        get_train_batches(cfg),
        get_val_batches(cfg),
        get_test_batches(cfg),
    )


class CalibrationDataset(Dataset):
    """Custom dataset for PTQ calibration of Torch-TensorRT."""

    def __init__(self, cfg: Gifnoc):
        self.paired_paths = get_paired_paths(cfg=cfg, stage=Stage.VAL)

    def __len__(self) -> int:
        return len(self.paired_paths)

    def __getitem__(self, index: int):
        _, compressed_path = self.paired_paths[index]
        compressed_image = Image.open(compressed_path)
        pipe = compose(
            F.pil_to_tensor, min_max_scaler, make_4times_downscalable
        )
        return pipe(compressed_image)


def get_calibration_dataloader(
    cfg: Gifnoc,
    subset_size: int | None = 5,
    random_seed: int | None = 42,
) -> DataLoader:
    """Returns a dataloader for PTQ calibration of Torch-TensorRT."""

    calibration_dataset = CalibrationDataset(cfg)

    dataset_size = len(calibration_dataset)
    if subset_size is not None and subset_size < dataset_size:
        if random_seed is not None:
            np.random.seed(random_seed)
            indices = np.random.choice(np.arange(dataset_size), subset_size)
        else:
            indices = np.arange(subset_size)
        subset_calibration_dataset = Subset(
            dataset=calibration_dataset, indices=indices.tolist()
        )
        calibration_dataloader = DataLoader(
            dataset=subset_calibration_dataset,
            batch_size=1,
            shuffle=False,
        )
    else:
        calibration_dataloader = DataLoader(
            dataset=calibration_dataset,
            batch_size=1,
            shuffle=False,
        )

    return calibration_dataloader
