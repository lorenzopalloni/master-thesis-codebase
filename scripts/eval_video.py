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
from binarization.datatools import (
    adjust_image_for_unet,
    inv_adjust_image_for_unet,
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


def cv2_to_torch(im):
    im = im / 255
    im = torch.Tensor(im).cuda()
    im = im.permute(2, 0, 1).unsqueeze(0)
    im = min_max_scaler(im)
    return im


def torch_to_cv2(tensor_img: torch.Tensor) -> npt.NDArray[np.uint8]:
    tensor_img = inv_min_max_scaler(tensor_img.squeeze(0))
    tensor_img = tensor_img.permute(1, 2, 0)
    numpy_img = tensor_img.byte().cpu().numpy()
    numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
    return numpy_img


def blend_images(i1, i2):
    w = i1.shape[-1]
    w_4 = w // 4
    i1 = i1[:, :, :, w_4 : w_4 * 3]
    i2 = i2[:, :, :, w_4 : w_4 * 3]
    out = torch.cat([i1, i2], dim=3)
    return out


def eval_video(
    cfg: Gifnoc,
    video_path: Path,
    enable_show_compressed: bool = True,
    enable_write_to_video: bool = False,
):
    scale_factor = cfg.params.scale_factor
    device = 'cpu'  # set_up_cuda_device()
    model = set_up_generator(cfg, device=device)
    cap = cv2.VideoCapture(video_path.as_posix())
    reader = torchvision.io.VideoReader(video_path.as_posix(), 'video')

    if enable_write_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        hr_video_writer = cv2.VideoWriter(
            'rendered.mp4', fourcc, 30, (1920, 1080)
        )

    # metadata = reader.get_metadata()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_queue: Queue = Queue(1)
    out_queue: Queue = Queue(1)

    reader.seek(0)

    def read_pic(cap, q: Queue, scale_factor: int = 4):
        while True:
            x = next(cap)['data']
            x = min_max_scaler(x)
            x = adjust_image_for_unet(x).unsqueeze(0)

            x_bicubic = torch.clip(
                F.interpolate(
                    x,
                    scale_factor=scale_factor,
                    mode='bicubic',
                ),
                min=0,
                max=1,
            )

            q.put((x, x_bicubic))

    def show_pic(cap, q):
        while True:
            out = q.get()
            cv2_out = torch_to_cv2(out)
            cv2.imshow('rendering', cv2_out)
            cv2.waitKey(1)

    Thread(target=read_pic, args=(reader, frame_queue, scale_factor)).start()
    Thread(target=show_pic, args=(cap, out_queue)).start()
    target_fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame_time = 1000 / target_fps

    model = model.eval()
    with torch.no_grad():
        tqdm_ = tqdm(range(frame_count))
        for i in tqdm_:

            frame_time_start = time.perf_counter()

            x, x_bicubic = frame_queue.get()
            out = model(x)
            # out = _out[:, :, : int(height) * scale_factor, : int(width) * scale_factor]

            out_true = i // (target_fps * 3) % 2 == 0

            if enable_show_compressed:
                out = blend_images(x_bicubic, out)

            out_queue.put(out)

            frame_time = time.perf_counter() - frame_time_start

            if frame_time < target_frame_time * 1e-3:
                time.sleep(target_frame_time * 1e-3 - frame_time)

            # if enable_write_to_video:
            #     write_to_video(out, hr_video_writer)
            #     if i == 30 * 10:
            #         hr_video_writer.release()
            #         print("Releasing video")

            # tqdm_.set_description((
            #     f"frame time: {frame_time * 1e3}; "
            #     f"fps: {1000 / frame_time}; {out_true}"
            # ))


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
