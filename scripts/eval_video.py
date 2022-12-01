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
from binarization.traintools import set_up_generator

torch.backends.cudnn.benchmark = False  # Defaults to True


def write_to_video(tensor_img: torch.Tensor, writer: cv2.VideoWriter) -> None:
    numpy_img = tensor_to_numpy(tensor_img)
    h, w, _ = numpy_img.shape
    offset = int(min(w, h) * 0.05)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2_put_text = partial(
        cv2.putText,
        img=numpy_img,
        fontFace=font,
        fontScale=0.5,
        color=(10, 10, 10),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    cv2_put_text(text='bicubic interpolation', org=(offset, h - offset))
    cv2_put_text(text='Unet (ours)', org=(w // 2 + offset, h - offset))

    writer.write(numpy_img)


def eval_video(
    cfg: Gifnoc,
    video_path: Path,
    enable_show_compressed: bool = True,
    enable_write_to_video: bool = False,
):
    scale_factor = cfg.params.scale_factor
    device = 'cpu'  # set_up_cuda_device()
    model = set_up_generator(cfg, device=device)
    reader = torchvision.io.VideoReader(video_path.as_posix(), 'video')

    metadata = reader.get_metadata()
    fps = metadata['video']['fps'][0]
    duration = metadata['video']['duration'][0]
    n_frames = int(fps * duration)

    # use cv2.VideoCapture to get w/h of the video
    temp_capture = cv2.VideoCapture(video_path.as_posix())
    width = int(temp_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(temp_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    adjusted_width = make_4times_divisible(width)
    adjusted_height = make_4times_divisible(height)
    del temp_capture

    if enable_write_to_video:
        frame_width = (
            adjusted_width * scale_factor * (2 ** int(enable_show_compressed))
        )
        frame_height = adjusted_height * scale_factor
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            filename='rendered.mp4',
            fourcc=fourcc,
            fps=30,
            frameSize=(frame_width, frame_height),
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
            # img = cv2.resize(img, (1000, 500))  # NOTE: remove this /!\
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
        tqdm_ = tqdm(range(n_frames), disable=True)
        for i in tqdm_:

            tic = time.perf_counter()

            compressed, resized_compressed = queue0.get()
            generated = model(compressed).clip(0, 1)

            if enable_show_compressed:
                generated = concatenate_images(resized_compressed, generated)

            queue1.put(generated)

            toc = time.perf_counter()
            elapsed = toc - tic

            expected_time_between_frames = fps / 1000
            if elapsed < expected_time_between_frames:
                time.sleep(expected_time_between_frames - elapsed)

            if enable_write_to_video:
                write_to_video(generated, writer)
                if i == n_frames - 1:
                    writer.release()
    queue0.join()
    queue1.join()


if __name__ == '__main__':
    default_cfg = get_default_config()
    default_cfg.model.ckpt_path_to_resume = Path(
        default_cfg.paths.artifacts_dir,
        "best_checkpoints",
        "2022_11_21_unet.pth",
    )

    default_cfg.params.buffer_size = 1
    default_cfg.model.name = 'unet'

    # video_path = Path(
    #     default_cfg.paths.data_dir,
    #     'compressed_videos',
    #     'DFireS18Mitch_480x272_24fps_10bit_420.mp4'
    # )
    homer_video_path = Path(
        default_cfg.paths.project_dir,
        "tests/assets/compressed_videos/homer_arch_512x372_120K.mp4",
    )

    eval_video(
        cfg=default_cfg,
        video_path=homer_video_path,
        enable_show_compressed=True,
        enable_write_to_video=True,
    )
