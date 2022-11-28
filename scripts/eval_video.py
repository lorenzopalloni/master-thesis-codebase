from __future__ import annotations

import time
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torchvision
from gifnoc import Gifnoc
from tqdm import tqdm

from binarization.config import get_default_config
from binarization.datatools import (  # inv_adjust_image_for_unet,
    adjust_image_for_unet,
    inv_min_max_scaler,
    min_max_scaler,
)
from binarization.traintools import set_up_generator

torch.backends.cudnn.benchmark = False  # Defaults to True


def save_with_cv2(tensor_img: torch.Tensor, path: str) -> None:
    tensor_img = inv_min_max_scaler(tensor_img.squeeze(0))
    numpy_img = np.transpose(tensor_img.cpu().numpy(), (1, 2, 0)) * 255
    numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, numpy_img)


def cv2_to_torch(numpy_img: npt.NDArray[np.uint8]) -> torch.Tensor:
    scaled_numpy_img: npt.NDArray[np.float64] = numpy_img / 255
    tensor_img = torch.Tensor(scaled_numpy_img).cuda()
    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0)
    tensor_img = min_max_scaler(tensor_img)
    return tensor_img


def tensor_to_numpy(tensor_img: torch.Tensor) -> npt.NDArray[np.uint8]:
    tensor_img = inv_min_max_scaler(tensor_img.squeeze(0))
    tensor_img = tensor_img.permute(1, 2, 0)
    numpy_img = tensor_img.byte().cpu().numpy()
    numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
    return numpy_img


def blend_images(
    img1: torch.Tensor,
    img2: torch.Tensor,
    shrink: bool = False,
) -> torch.Tensor:
    assert (
        img1.shape[-1] == img2.shape[-1]
    ), f"{img1.shape=} not equal to {img2.shape=}."
    if shrink:
        width = img1.shape[-1]
        w_4 = width // 4
        img1 = img1[:, :, :, w_4 : w_4 * 3]
        img2 = img2[:, :, :, w_4 : w_4 * 3]
    return torch.cat([img1, img2], dim=3)


def resize_tensor_img(
    tensor_img: torch.Tensor, scale_factor: int = 4
) -> torch.Tensor:
    return torch.clip(
        F.interpolate(
            tensor_img,
            scale_factor=scale_factor,
            mode='bicubic',
        ),
        min=0,
        max=1,
    )


def write_to_video(tensor_img: torch.Tensor, writer: cv2.VideoWriter) -> None:
    numpy_img = tensor_to_numpy(tensor_img)
    h, w, _ = numpy_img.shape
    offset = int(min(w, h) * 0.05)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        img=numpy_img,
        text='bicubic interpolation',
        # org=(50, 1030),
        org=(offset, h - offset),
        fontFace=font,
        fontScale=0.5,
        color=(10, 10, 10),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img=numpy_img,
        text='Unet (ours)',
        # org=(1920 // 2 + 50, 1020),
        org=(w // 2 + offset, h - offset),
        fontFace=font,
        fontScale=0.5,
        color=(10, 10, 10),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

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

    if enable_write_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        hr_video_writer = cv2.VideoWriter(
            filename='rendered.mp4',
            fourcc=fourcc,
            fps=30,
            # frameSize=(1920, 1080),
            frameSize=(1024, 384),
        )

    queue0: Queue = Queue(1)
    queue1: Queue = Queue(1)

    reader.seek(0)

    def read_pic(
        reader: torchvision.io.VideoReader, queue: Queue, scale_factor: int = 4
    ) -> None:
        while True:
            img = next(reader)['data']
            img = min_max_scaler(img)
            img = adjust_image_for_unet(img).unsqueeze(0)
            resized_img = resize_tensor_img(img, scale_factor)
            queue.put((img, resized_img))

    def show_pic(queue):
        while True:
            tensor_img = queue.get()
            img = tensor_to_numpy(tensor_img)
            # img = cv2.resize(img, (64, 64))  # NOTE: remove this /!\
            cv2.imshow('rendering', img)
            cv2.waitKey(1)

    thread0 = Thread(target=read_pic, args=(reader, queue0, scale_factor))
    thread1 = Thread(target=show_pic, args=(queue1,))
    thread0.start()
    thread1.start()

    model = model.eval()
    with torch.no_grad():
        tqdm_ = tqdm(range(n_frames))
        for i in tqdm_:

            tic = time.perf_counter()

            compressed, resized_compressed = queue0.get()
            generated = model(compressed).clip(0, 1)

            if enable_show_compressed:
                generated = blend_images(
                    resized_compressed, generated, shrink=False
                )

            queue1.put(generated)

            toc = time.perf_counter()
            elapsed = toc - tic

            if elapsed < fps * 1e-3:
                time.sleep(fps * 1e-3 - elapsed)

            if enable_write_to_video:
                write_to_video(generated, hr_video_writer)
                # if i == 30 * 10:
                if i == n_frames - 1:
                    hr_video_writer.release()
                    print("Releasing video")


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
