from __future__ import annotations

import time
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import matplotlib.pyplot as plt
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


def write_to_video(pic, writer):
    pic = inv_min_max_scaler(pic.squeeze(0))
    npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0)) * 255
    npimg = npimg.astype('uint8')
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        npimg,
        '540p CRF 23 + bicubic',
        (50, 1030),
        font,
        1,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        npimg,
        'SR-Unet (ours)',
        (1920 // 2 + 50, 1020),
        font,
        1,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )

    writer.write(npimg)


def cv2_to_torch(numpy_img: npt.NDArray[np.uint8]) -> torch.Tensor:
    numpy_img = numpy_img / 255
    torch_img = torch.Tensor(numpy_img).cuda()
    torch_img = torch.img.permute(2, 0, 1).unsqueeze(0)
    torch_img = min_max_scaler(torch_img)
    return torch_img


def torch_to_cv2(tensor_img: torch.Tensor) -> npt.NDArray[np.uint8]:
    tensor_img = inv_min_max_scaler(tensor_img.squeeze(0))
    tensor_img = tensor_img.permute(1, 2, 0)
    numpy_img = tensor_img.byte().cpu().numpy()
    numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
    return numpy_img


def blend_images(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    assert (
        img1.shape[-1] == img2.shape[-1]
    ), f"{img1.shape=} not equal to {img2.shape=}."
    width = img1.shape[-1]
    w_4 = width // 4
    cropped_i1 = img1[:, :, :, w_4 : w_4 * 3]
    cropped_i2 = img2[:, :, :, w_4 : w_4 * 3]
    return torch.cat([cropped_i1, cropped_i2], dim=3)


def resize_torch_img(
    torch_img: torch.Tensor, scale_factor: int = 4
) -> torch.Tensor:
    return torch.clip(
        F.interpolate(
            torch_img,
            scale_factor=scale_factor,
            mode='bicubic',
        ),
        min=0,
        max=1,
    )


def read_pic(cap, q: Queue, scale_factor: int = 4):
    while True:
        img = next(cap)['data']
        img = min_max_scaler(img)
        img = adjust_image_for_unet(img).unsqueeze(0)
        resized_img = resize_torch_img(img, scale_factor)
        q.put((img, resized_img))


def show_pic(q):
    while True:
        torch_img = q.get()
        img = torch_to_cv2(torch_img)
        img = cv2.resize(img, (64, 64))  # NOTE: remove this /!\
        cv2.imshow('rendering', img)
        cv2.waitKey(1)


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

    if enable_write_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        hr_video_writer = cv2.VideoWriter(
            'rendered.mp4', fourcc, 30, (1920, 1080)
        )

    metadata = reader.get_metadata()
    fps = metadata['video']['fps'][0]
    duration = metadata['video']['duration'][0]
    n_frames = int(fps * duration)

    q0: Queue = Queue(1)
    q1: Queue = Queue(1)

    reader.seek(0)

    t0 = Thread(target=read_pic, args=(reader, q0, scale_factor))
    t1 = Thread(target=show_pic, args=(q1,))
    t0.start()
    t1.start()

    model = model.eval()
    with torch.no_grad():
        tqdm_ = tqdm(range(n_frames))
        for i in tqdm_:

            tic = time.perf_counter()

            compressed, resized_compressed = q0.get()
            generated = model(compressed)

            if enable_show_compressed:
                generated = blend_images(resized_compressed, generated)

            q1.put(generated)

            toc = time.perf_counter()
            elapsed = toc - tic

            # if elapsed < target_frame_time * 1e-3:
            #     time.sleep(target_frame_time * 1e-3 - elapsed)

            if enable_write_to_video:
                write_to_video(generated, hr_video_writer)
                if i == 30 * 10:
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
    video_path = Path(
        default_cfg.paths.project_dir,
        "tests/assets/compressed_videos/homer_arch_512x372_120K.mp4",
    )
    print(video_path)

    eval_video(
        cfg=default_cfg,
        video_path=video_path,
        enable_show_compressed=True,
        enable_write_to_video=False,
    )
