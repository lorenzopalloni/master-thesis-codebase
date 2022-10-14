# pylint: disable=invalid-name,missing-function-docstring,too-many-arguments
"""Custom dataloaders"""

from __future__ import annotations

import itertools
import json
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import PIL
import torch
import torchvision.transforms.functional as F
from gifnoc import Gifnoc
from sklearn.model_selection import train_test_split
from torchvision.utils import Image

from binarization.datatools import (
    min_max_scaler,
    random_crop_images,
    list_files,
    list_directories,
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
    cfg: Gifnoc,
    exist_ok: bool = True,
    random_state: int = 42
) -> dict[str, list[str]]:
    """Fetches `data_dir/splits.json`.

    If it does not exist, it creates it, then returns it.

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


###############################################################################

class Stage(Enum):
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
        n_iterations: int | None = 1,
    ):
        """Generator that serves buffered chunk of items from a list of items.

        Args:
            init_list (list[Any]): A list of items
            buffer_size (int): Amount of element per buffer.
            shuffle (bool): If True, items are randomly permuted once
                at the beginning of each iteration.
            pipe (Callable[[Any], Any], optional): Function applied to
                each single item before entering the buffer. Defaults
                to identity_pipe.
            n_iterations (int | None, optional): Number of iterations.
                For each iteration, `len(init_list) // buffer_size` number
                of buffers will be served. Defaults to 1.
        """
        self.items = init_list
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.pipe = pipe
        self.n_iterations = (
            n_iterations
            if n_iterations is not None
            and n_iterations > 0
            else None
        )
        self.iteration_counter = 0

        self.n_items = len(self.items)
        self.indexes = np.arange(self.n_items)
        if self.shuffle: np.random.shuffle(self.indexes)
        self.start_idx = 0
        
    def __len__(self):
        return self.n_items // self.buffer_size

    def end_iteration_step(self):
        self.iteration_counter += 1
        if (
            self.n_iterations is not None
            and self.iteration_counter >= self.n_iterations
        ):
            raise StopIteration
        else:
            self.start_idx = 0
            if self.shuffle: np.random.shuffle(self.indexes)
    
    def __getitem__(self, i: int):
        del i

        if self.start_idx > self.n_items - self.buffer_size:
            self.end_iteration_step()
        
        indexes = self.indexes[self.start_idx: self.start_idx + self.buffer_size]
        buffer = [self.pipe(self.items[idx]) for idx in indexes]
        self.start_idx += self.buffer_size

        return buffer
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    

def get_paired_paths(cfg: Gifnoc, stage: Stage) -> list[tuple[Path, Path]]:
    """Lists pairs of original/compressed paths for a specific stage.

    Available stages: {`train`, `val`, 'test'}.

    Args:
        cfg (Gifnoc): Configuration object.
        stage (Stage): Choose in {`Stage.TRAIN`, `Stage.VAL`, `Stage.TEST`}.

    Returns:
        list[tuple[Path, Path]]: Pairs of original/compressed frame paths.
    """
    splits = get_splits(cfg)
    original_paths = list(
        itertools.chain.from_iterable(
            list_files(
                Path(cfg.paths.original_frames_dir, path), extension='.png'
            )
            for path in splits[stage.value]
        )
    )
    compressed_paths = list(
        itertools.chain.from_iterable(
            list_files(
                Path(cfg.paths.compressed_frames_dir, path), extension='.jpg'
            )
            for path in splits[stage.value]
        )
    )
    assert len(original_paths) == len(compressed_paths)
    return list(zip(original_paths, compressed_paths))

def default_train_pipe(
    original_image: PIL.Image.Image, compressed_image: PIL.Image.Image
) -> tuple[torch.Tensor, torch.Tensor]:
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

def default_val_pipe(
    original_image: PIL.Image.Image, compressed_image: PIL.Image.Image
) -> tuple[torch.Tensor, torch.Tensor]:
    np.random.seed(42)  # crop at random positions but always the same
    original_patch, compressed_patch = random_crop_images(
        original_image=original_image,
        compressed_image=compressed_image,
    )
    np.random.seed(None)
    return (
        min_max_scaler(F.pil_to_tensor(original_patch)),
        min_max_scaler(F.pil_to_tensor(compressed_patch)),
    )

def default_test_pipe(
    original_image: PIL.Image.Image, compressed_image: PIL.Image.Image
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        min_max_scaler(F.pil_to_tensor(original_image)),
        min_max_scaler(F.pil_to_tensor(compressed_image)),
    )

def batch_generator(
    cfg: Gifnoc,
    frame_buffer_size: int,
    n_batches_per_frame_buffer: int,
    n_paired_patches_per_batch: int,
    stage: Stage,
) -> Iterator:
    """Generates batches of random patches prefetching some paired frames.

    Both image selection and cropping take place completely
    at random, and can lead to duplicate patches (this is an
    expected behaviour).

    Note that the random position for cropping is the same in
    each pair of image. 

    Args:
        paired_paths (list[tuple[Path, Path]]):
            List of paired paths [
                (original_path_1, compressed_path_1),
                (original_path_2, compressed_path_2),
                ...
            ].
        frame_buffer_size (int): Amount of frames that the buffer loads.
        n_batches_per_frame_buffer (int): Number of batches (each
            consisting of `batch_size` patches) that can be drawn
            from a frame buffer before refreshing it with new frames.
        n_paired_patches_per_batch (int): Batch size.
        stage (Stage): Choose in {`Stage.TRAIN`, `Stage.VAL`, `Stage.TEST`}.

    Yields:
        Iterator: Batch iterator.
    """
    shuffle = True if stage == Stage.TRAIN else False
    paired_paths = get_paired_paths(cfg=cfg, stage=stage)

    def path_to_frame_pipe(
        paired_paths: tuple[Path, Path]
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        original_path, compressed_path = paired_paths
        return Image.open(original_path), Image.open(compressed_path)

    paired_frame_buffer_generator = BufferGenerator(
        init_list=paired_paths,
        buffer_size=frame_buffer_size,
        shuffle=shuffle,
        pipe=path_to_frame_pipe,
    )

    frame_to_patch_pipes = {
        Stage.TRAIN: default_train_pipe,
        Stage.VAL: default_val_pipe,
        Stage.TEST: default_test_pipe,
    }

    for paired_frame_buffer in paired_frame_buffer_generator:
        if not shuffle:
            n_batches_per_frame_buffer = frame_buffer_size
            n_paired_patches_per_batch = 1
        for idx in range(n_batches_per_frame_buffer):
            original_batch = []
            compressed_batch = []
            for _ in range(n_paired_patches_per_batch):
                random_idx = (
                    np.random.randint(0, frame_buffer_size, dtype=np.uint8)
                    if shuffle else idx
                )
                original_image, compressed_image = paired_frame_buffer[random_idx]
                original_patch, compressed_patch = frame_to_patch_pipes[stage](
                    original_image, compressed_image
                )
                original_batch.append(original_patch)
                compressed_batch.append(compressed_patch)
            yield torch.stack(original_batch), torch.stack(compressed_batch)

# def make_dataloader(
#     cfg: Gifnoc,
#     stage: Stage,
#     frame_buffer_size: int,
#     patch_buffer_size: int,
#     batch_size: int,
# ):
#     paired_paths = get_paired_paths(cfg=cfg, stage=stage)
#     paired_paths_buffer_generator = BufferGenerator(
#         init_list=paired_paths,
#         buffer_size=frame_buffer_size,
#         shuffle=True if stage == Stage.TRAIN else False
#     )
#     for buffer in paired_paths_buffer_generator: 
#         paired_images = [
#             (Image.open(original_path), Image.open(compressed_path))
#             for original_path, compressed_path in buffer
#         ]
#     return patch_generator(
#         paired_images=paired_images,
#         pipe=PATCH_GENERATOR_PIPES[stage],
#         batch_size=batch_size,
#         buffer_size=patch_buffer_size
#     )

###############################################################################


# class FrameBuffer:
#     def __init__(
#         self,
#         original_paths: list[Path],
#         compressed_paths: list[Path],
#         buffer_size: int = 8,
#         shuffle: bool = False,
#     ):
#         self.original_paths = original_paths
#         self.compressed_paths = compressed_paths
#         self.buffer_size = buffer_size
#         self.shuffle = shuffle

#         self.n_frames = len(self.original_paths)
#         self.indexes = np.arange(self.n_frames)
#         if self.shuffle: np.random.shuffle(self.indexes)
#         self.start_idx = 0
        
#     def __len__(self):
#         return self.n_frames // self.buffer_size
    
#     def __getitem__(self, i: int):
#         del i

#         if self.start_idx > self.n_frames - self.buffer_size:
#             raise StopIteration
        
#         indexes = self.indexes[self.start_idx: self.start_idx + self.buffer_size]
#         paths = [(self.original_paths[idx], self.compressed_paths[idx]) for idx in indexes]
#         buffer = [(Image.open(o), Image.open(c)) for o, c in paths]
#         self.start_idx += self.buffer_size

#         return buffer


# def default_test_pipe(
#     original_path: Path,
#     compressed_path: Path,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     original_image = Image.open(original_path)
#     compressed_image = Image.open(compressed_path)

#     return (
#         min_max_scaler(F.pil_to_tensor(original_image)),
#         min_max_scaler(F.pil_to_tensor(compressed_image)),
#     )


# def default_val_pipe(
#     original_path: Path,
#     compressed_path: Path,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     original_image = Image.open(original_path)
#     compressed_image = Image.open(compressed_path)

#     original_patch, compressed_patch = random_crop_images(
#         original_image=original_image,
#         compressed_image=compressed_image,
#     )

#     return (
#         min_max_scaler(F.pil_to_tensor(original_patch)),
#         min_max_scaler(F.pil_to_tensor(compressed_patch)),
#     )


# def default_train_pipe(
#     original_path: Path, compressed_path: Path
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     original_image = Image.open(original_path)
#     compressed_image = Image.open(compressed_path)

#     original_patch, compressed_patch = random_crop_images(
#         original_image=original_image,
#         compressed_image=compressed_image,
#     )

#     if np.random.random() < 0.5:
#         original_patch = F.hflip(original_patch)
#         compressed_patch = F.hflip(compressed_patch)

#     return (
#         min_max_scaler(F.pil_to_tensor(original_patch)),
#         min_max_scaler(F.pil_to_tensor(compressed_patch)),
#     )


# def image_only_pipe(original_path: Path, compressed_path: Path):
#     """Pipe that loads images without doing anything else."""
#     original_image = Image.open(original_path)
#     compressed_image = Image.open(compressed_path)
#     return original_image, compressed_image


# def identity_pipe(original_path: Path, compressed_path: Path):
#     return original_path, compressed_path


# def custom_collate_fn(batch):
#     return batch


# class FrameBuffer:
#     def __init__(self, fps, size: int = 8, shuffle: bool = False):
#         self.n_fps = len(fps)
#         indexes = np.arange(self.n_fps)
#         if shuffle:
#             np.random.shuffle(indexes)

#         self.fps = fps
#         self.size = size
#         self.pointer = 0

#     def __len__(self):
#         return self.n_fps // self.size

#     def __getitem__(self, i: int):
#         buffer = self.fps[self.pointer : self.pointer + self.size]
#         self.pointer += self.size
#         return buffer


# class BufferedFrameDataset(Dataset):
#     def __init__(
#         self,
#         original_paths: list[Path],
#         compressed_paths: list[Path],
#         frame_buffer_size: int = 8,  # a good value should be around the same as the batch_size
#         patch_buffer_size: int = 16,  # a good value should be 5 times the frame_buffer_size
#         shuffle: bool = True,
#     ):
#         self.original_paths = original_paths
#         self.compressed_paths = compressed_paths
#         assert len(self.original_paths) == len(self.compressed_paths)

#         self.frame_buffer_size = frame_buffer_size
#         self.patch_buffer_size = patch_buffer_size

#         frame_dataset = ImageFilepathDataset(
#             self.original_paths,
#             self.compressed_paths,
#             pipe=image_only_pipe,
#         )
#         self.all_frame_buffers = iter(
#             DataLoader(
#                 frame_dataset,
#                 batch_size=frame_buffer_size,
#                 shuffle=shuffle,
#                 collate_fn=custom_collate_fn,
#             )
#         )
#         self.curr_frame_buffer = next(self.all_frame_buffers)
#         self.frame_buffer_counter = 0

#     def __len__(self):
#         return len(self.original_paths)

#     def __getitem__(self, i):
#         if self.frame_buffer_counter > self.patch_buffer_size - 1:
#             self.curr_frame_buffer = next(self.all_frame_buffers)
#             self.frame_buffer_counter = 0
#         self.frame_buffer_counter += 1
#         j = i % self.frame_buffer_size
#         return self.curr_frame_buffer[j]


# class ImageFilepathDataset(Dataset):
#     def __init__(
#         self,
#         original_paths: list[Path],
#         compressed_paths: list[Path],
#         pipe: Callable = identity_pipe,
#     ):
#         self.original_paths = original_paths
#         self.compressed_paths = compressed_paths
#         assert len(self.original_paths) == len(self.compressed_paths)

#         self.pipe = pipe

#     def __len__(self) -> int:
#         return len(self.original_paths)

#     def __getitem__(self, i):
#         original_fn, compressed_fn = (
#             self.original_paths[i],
#             self.compressed_paths[i],
#         )
#         return self.pipe(original_fn, compressed_fn)

# ###############################################################################


# def make_dataset(
#     cfg: Gifnoc,
#     stage: Stage,
#     pipe: Callable[[Path, Path], Any] = identity_pipe,
# ) -> ImageFilepathDataset:
#     """Makes an ImageFilepathDataset for a specific stage.

#     An ImageFilepathDataset is a custom pytorch dataset that
#     is able to handle image filepath and a custom function
#     called `pipe` to feed a model train/val/test pipeline.

#     Available stages: {`train`, `val`, 'test'}

#     Args:
#         cfg (Gifnoc): Configuration object
#         stage (str): A string representing the stage,
#             the possible choices are `train`, `val`,
#             and `test`.
#         pipe: (Callable): Custom function to transform
#             image filepaths to something else.

#     Returns:
#         ImageFilepathDataset: An ImageFilepathDataset for a specific stage.
#     """
#     splits = get_splits(cfg)
#     original_paths = list(
#         itertools.chain.from_iterable(
#             list_files(
#                 Path(cfg.paths.original_frames_dir, path), extension='.png'
#             )
#             for path in splits[stage.value]
#         )
#     )
#     compressed_paths = list(
#         itertools.chain.from_iterable(
#             list_files(
#                 Path(cfg.paths.compressed_frames_dir, path), extension='.jpg'
#             )
#             for path in splits[stage.value]
#         )
#     )
#     return ImageFilepathDataset(
#         original_paths=original_paths,
#         compressed_paths=compressed_paths,
#         pipe=pipe,
#     )


# def make_train_dataloader(cfg) -> DataLoader:
#     """Makes a training dataloader."""
#     dataset = make_dataset(cfg, Stage.TRAIN, default_train_pipe)
#     return DataLoader(
#         dataset=dataset,
#         batch_size=cfg.params.batch_size,
#         num_workers=cfg.params.num_workers,
#         shuffle=True,
#         pin_memory=True,
#     )


# def make_val_dataloader(cfg) -> DataLoader:
#     """Makes a validation dataloader."""
#     dataset = make_dataset(cfg, Stage.VAL, default_val_pipe)
#     return DataLoader(
#         dataset=dataset,
#         batch_size=cfg.params.batch_size,
#         num_workers=cfg.params.num_workers,
#         shuffle=False,
#         pin_memory=True,
#     )


# def make_test_dataloader(cfg) -> DataLoader:
#     """Makes a testing dataloader."""
#     dataset = make_dataset(cfg, Stage.TEST, default_test_pipe)
#     return DataLoader(
#         dataset=dataset,
#         batch_size=cfg.params.batch_size,
#         num_workers=cfg.params.num_workers,
#         shuffle=False,
#         pin_memory=True,
#     )


# def make_dataloaders(
#     cfg: Gifnoc,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """Makes train/val/test dataloaders."""
#     return (
#         make_train_dataloader(cfg),
#         make_val_dataloader(cfg),
#         make_test_dataloader(cfg),
#     )


# if __name__ == "__main__":
#     import json
#     from pathlib import Path
#     from binarization import dataset, config, train
#     from binarization.dataset import (
#         Stage,
#         get_splits,
#         list_files,
#         BufferedFrameDataset,
#     )
#     import itertools

#     project_dir = Path().resolve().parent
#     data_dir = project_dir / 'data'
#     o_dir = data_dir / 'original_frames'
#     c_dir = data_dir / 'compressed_frames'
#     split_json = data_dir / 'splits.json'
#     device = train.set_cuda_device(verbose=True)
#     cfg = config.get_default_config()

#     from binarization.dataset import ImageFilepathDataset

#     splits = get_splits(cfg)
#     original_paths = list(
#         itertools.chain.from_iterable(
#             list_files(
#                 Path(cfg.paths.original_frames_dir, path), extension='.png'
#             )
#             for path in splits[Stage.VAL.value]
#         )
#     )
#     compressed_paths = list(
#         itertools.chain.from_iterable(
#             list_files(
#                 Path(cfg.paths.compressed_frames_dir, path), extension='.jpg'
#             )
#             for path in splits[Stage.VAL.value]
#         )
#     )
#     buffered_frame_dataset = BufferedFrameDataset(
#         original_paths=original_paths,
#         compressed_paths=compressed_paths,
#     )
#     buffered_frame_dl = DataLoader(
#         buffered_frame_dataset, batch_size=2, shuffle=True
#     )
#     buffered_frame_iterator = iter(buffered_frame_dl)
#     iterator = next(buffered_frame_iterator)
