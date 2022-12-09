"""Script to evaluate a video with a super-resolution model"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import torch
import torchvision
from gifnoc import Gifnoc
from tqdm import tqdm

from binarization.config import get_default_config
from binarization.datatools import (
    bicubic_interpolation,
    concatenate_images,
    make_4times_divisible,
    make_4times_downscalable,
    min_max_scaler,
    tensor_to_numpy,
)
from binarization.traintools import set_up_cuda_device, set_up_generator

torch.backends.cudnn.benchmark = False  # Defaults to True


class WriteToVideo:
    """Helper to write text on video frames."""

    def __init__(
        self,
        width: int,
        height: int,
        scale_factor: int = 4,
        enable_show_compressed: bool = True,
        enable_crop: bool = True,
    ):
        adjusted_width = make_4times_divisible(width)
        adjusted_height = make_4times_divisible(height)

        self.frame_width = int(
            adjusted_width
            * scale_factor
            * (2 ** int(enable_show_compressed))
            * (1 / 2 ** int(enable_crop))
        )
        self.frame_height = adjusted_height * scale_factor

        self.writer = cv2.VideoWriter(
            filename='rendered.mp4',
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=30,
            frameSize=(self.frame_width, self.frame_height),
        )

    def write(self, tensor_img: torch.Tensor, model_name: str) -> None:
        """Overlays text on a given frame.

        Args:
            tensor_img (torch.Tensor): a frame.
            model_name (str): name of the generator used.
        """
        numpy_img = tensor_to_numpy(tensor_img)
        h, w, _ = numpy_img.shape
        offset = int(min(w, h) * 0.05)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2_put_text = partial(
            cv2.putText,
            img=numpy_img,
            fontFace=font,
            fontScale=1,
            color=(10, 10, 10),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2_put_text(text='bicubic interpolation', org=(offset, h - offset))
        cv2_put_text(
            text=f'{model_name} (ours)', org=(w // 2 + offset, h - offset)
        )

        self.writer.write(numpy_img)

    def release(self) -> None:
        """Releases the modified video."""
        self.writer.release()


def get_video_size(video_path: Path) -> tuple[int, int]:
    """Retrieves video width/height."""
    temp_capture = cv2.VideoCapture(video_path.as_posix())
    width = int(temp_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(temp_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    del temp_capture
    return width, height


def eval_video(
    cfg: Gifnoc,
    video_path: Path,
    enable_show_compressed: bool = True,
    enable_crop: bool = True,
    enable_write_to_video: bool = False,
):
    """Upscales a compressed video with super-resolution.

    Args:
        cfg (Gifnoc): a valid configuration object.
        video_path (Path): path to the compressed video to be evaluated.
        enable_show_compressed (bool, optional): flag to show compressed
            video along the upscaled one. Defaults to True.
        enable_crop (bool, optional): flag to enable cropping video
            frames to enhance comparison. Defaults to True.
        enable_write_to_video (bool, optional): flag to enable little
            captions. Defaults to False.
    """
    scale_factor = cfg.params.scale_factor
    device = set_up_cuda_device(0)
    model = set_up_generator(cfg, device=device)
    reader = torchvision.io.VideoReader(video_path.as_posix(), 'video')

    metadata = reader.get_metadata()
    fps = metadata['video']['fps'][0]
    duration = metadata['video']['duration'][0]
    n_frames = int(fps * duration)

    if enable_write_to_video:
        width, height = get_video_size(video_path)
        writer = WriteToVideo(
            width=width,
            height=height,
            scale_factor=scale_factor,
            enable_show_compressed=enable_show_compressed,
        )

    queue0: Queue = Queue(1)
    queue1: Queue = Queue(1)

    reader.seek(0)

    def read_pic(
        reader: torchvision.io.VideoReader, queue: Queue, scale_factor: int = 4
    ) -> None:
        for frame in reader:
            img = frame['data']
            img = min_max_scaler(img)
            img = make_4times_downscalable(img).unsqueeze(0)
            interpolated_img = bicubic_interpolation(img, scale_factor)
            queue.put((img, interpolated_img))
            queue.task_done()

    def show_pic(queue):
        while True:
            tensor_img = queue.get()
            img = tensor_to_numpy(tensor_img)
            img = cv2.resize(img, (1000, 500))  # NOTE: remove this /!\
            cv2.imshow('rendering', img)
            cv2.waitKey(1)
            queue.task_done()

    thread0 = Thread(
        target=read_pic, args=(reader, queue0, scale_factor), daemon=True
    )
    thread1 = Thread(target=show_pic, args=(queue1,), daemon=True)
    thread0.start()
    thread1.start()

    model = model.eval()
    with torch.no_grad():
        tqdm_ = tqdm(range(n_frames), disable=False)
        for frame_idx in tqdm_:
            del frame_idx

            try:
                tic = time.perf_counter()

                compressed, resized_compressed = queue0.get()
                generated = model(compressed.to(device)).clip(0, 1).cpu()

                if enable_show_compressed:
                    generated = concatenate_images(
                        resized_compressed, generated, crop=enable_crop
                    )

                queue1.put(generated)

                toc = time.perf_counter()
                elapsed = toc - tic

                expected_time_between_frames = fps / 1000
                if elapsed < expected_time_between_frames:
                    time.sleep(expected_time_between_frames - elapsed)

                if enable_write_to_video:
                    writer.write(generated, cfg.model.name)

            except KeyboardInterrupt:
                break

        if enable_write_to_video:
            writer.release()

    queue0.join()
    queue1.join()


if __name__ == '__main__':
    default_cfg = get_default_config()

    best_checkpoints_dir = Path(
        default_cfg.paths.artifacts_dir, "best_checkpoints"
    )

    unet_ckpt_path = Path(
        best_checkpoints_dir,
        "2022_11_21_unet.pth",
    )
    del unet_ckpt_path

    srunet_ckpt_path = Path(
        best_checkpoints_dir,
        "2022_12_09_srunet.pth",
    )

    default_cfg.model.ckpt_path_to_resume = srunet_ckpt_path
    default_cfg.params.buffer_size = 1
    default_cfg.model.name = 'srunet'

    test_video_path = Path(
        default_cfg.paths.data_dir,
        'compressed_videos',
        'old_town_cross_1080p50.mp4',
    )

    homer_video_path = Path(
        default_cfg.paths.project_dir,
        "tests/assets/compressed_videos/homer_arch_512x372_120K.mp4",
    )
    del homer_video_path

    eval_video(
        cfg=default_cfg,
        video_path=test_video_path,
        enable_show_compressed=True,
        enable_write_to_video=True,
    )
