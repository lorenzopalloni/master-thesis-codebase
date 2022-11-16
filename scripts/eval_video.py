import time
from pathlib import Path
from threading import Thread  # not sure about this

import cv2
import torch
import torchvision
from tqdm import tqdm

torch.backends.cudnn.benchmark = True  # not sure about this
from queue import Queue  # not sure about this

from gifnoc import Gifnoc
from binarization.traintools import set_up_generator
from binarization.config import get_default_config


def save_with_cv(pic, imname):
    pic = dl.de_normalize(pic.squeeze(0))
    npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0)) * 255
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)

    cv2.imwrite(imname, npimg)


def write_to_video(pic, writer):
    pic = dl.de_normalize(pic.squeeze(0))
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


def get_padded_dim(H_x, W_x, border=0, mod=16):
    modH, modW = H_x % (mod + border), W_x % (mod + border)
    padW = ((mod + border) - modW) % (mod + border)
    padH = ((mod + border) - modH) % (mod + border)

    new_H = H_x + padH
    new_W = W_x + padW

    return new_H, new_W, padH, padW


def pad_input(x, padH, padW):
    x = F.pad(x, [0, padW, 0, padH])
    return x


def cv2toTorch(im):
    im = im / 255
    im = torch.Tensor(im).cuda()
    im = im.permute(2, 0, 1).unsqueeze(0)
    im = dl.normalize_img(im)
    return im


def torchToCv2(pic, rescale_factor=1.0):
    if rescale_factor != 1.0:
        pic = F.interpolate(
            pic,
            scale_factor=rescale_factor,
            align_corners=True,
            mode='bicubic',
        )
    pic = dl.de_normalize(pic.squeeze(0))
    pic = pic.permute(1, 2, 0) * 255
    npimg = pic.byte().cpu().numpy()
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
    return npimg


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
    enable_show_only_hq: bool = True,
    enable_write_to_video: bool = False
): 
    device = 'cpu'  # set_up_cuda_device()
    model = set_up_generator(cfg, device=device)
    cap = cv2.VideoCapture(video_path.as_posix())
    reader = torchvision.io.VideoReader(video_path.as_posix(), 'video')

    if write_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        hr_video_writer = cv2.VideoWriter(
            'rendered.mp4', fourcc, 30, (1920, 1080)
        )

    metadata = reader.get_metadata()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height_fix, width_fix, padH, padW = get_padded_dim(height, width)

    frame_queue: Queue = Queue(1)
    out_queue: Queue = Queue(1)

    reader.seek(0)

    def read_pic(cap, q):
        count = 0
        start = time.time()
        while True:
            cv2_im = next(cap)['data']
            cv2_im = cv2_im.cpu().float()

            x = dl.normalize_img(cv2_im / 255.0).unsqueeze(0)

            x_bicubic = torch.clip(
                F.interpolate(
                    x,
                    scale_factor=cfg.params.scale_factor,
                    mode='bicubic',
                ),
                min=-1,
                max=1,
            )

            x = F.pad(x, [0, padW, 0, padH])
            count += 1
            q.put((x, x_bicubic))

    def show_pic(cap, q):
        while True:
            out = q.get()
            scale = 1
            cv2_out = torchToCv2(out, rescale_factor=scale)
            cv2.imshow('rendering', cv2_out)
            cv2.waitKey(1)

    Thread(target=read_pic, args=(reader, frame_queue)).start()
    Thread(target=show_pic, args=(cap, out_queue)).start()
    target_fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame_time = 1000 / target_fps

    model = model.eval()
    with torch.no_grad():
        tqdm_ = tqdm(range(frame_count))
        for i in tqdm_:

            frame_time_start = time.perf_counter()

            x, x_bicubic = frame_queue.get()
            out = model(x)[:, :, : int(height) * 2, : int(width) * 2]

            out_true = i // (target_fps * 3) % 2 == 0

            if not enable_show_only_hq:
                out = blend_images(x_bicubic, out)
            out_queue.put(out)

            frame_time = time.perf_counter() - frame_time_start

            if frame_time < target_frame_time * 1e-3:
                time.sleep(target_frame_time * 1e-3 - frame_time)

            if enable_write_to_video:
                write_to_video(out, hr_video_writer)
                if i == 30 * 10:
                    hr_video_writer.release()
                    print("Releasing video")

            tqdm_.set_description(
                "frame time: {}; fps: {}; {}".format(
                    frame_time * 1e3, 1000 / frame_time, out_true
                )
            )

if __name__ == '__main__':
    default_cfg = get_default_config()
    default_cfg.model.ckpt_path_to_resume = Path(
        default_cfg.paths.artifacts_dir,
        "checkpoints",
        "2022_11_15_07_43_30/unet_0_39999.pth"
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
        "tests/assets/compressed_videos/homer_arch_512x372_120K.mp4"
    )

    eval_video(
        cfg=default_cfg,
        video_path=video_path, 
        enable_show_only_hq = True,
        enable_write_to_video = False,
    )
