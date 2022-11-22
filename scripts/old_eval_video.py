"""Script to evaluate a trained super-resolution model"""

### binarization/vaccaro/render.py - START
import time
from pathlib import Path
from threading import Thread

import cv2
import data_loader as dl
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms.functional as F
from tqdm import tqdm

from binarization import dataset, train
from binarization.config import Gifnoc, get_default_config

torch.backends.cudnn.benchmark = True
from queue import Queue

import cv2
import numpy as np
import utils
from models import *
from pytorch_unet import SimpleResNet, SRUnet, UNet
from tqdm import tqdm

# from apex import amp


def save_with_cv2(pic, imname):
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


def cv2_to_torch(im):
    im = im / 255
    im = torch.Tensor(im).cuda()
    im = im.permute(2, 0, 1).unsqueeze(0)
    im = dl.normalize_img(im)
    return im


def torch_to_cv2(pic, rescale_factor=1.0):
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


if __name__ == '__main__':
    args = utils.ARArgs()
    enable_write_to_video = False
    arch_name = args.ARCHITECTURE
    dataset_upscale_factor = args.UPSCALE_FACTOR

    if arch_name == 'srunet':
        model = SRUnet(
            3,
            residual=True,
            scale_factor=dataset_upscale_factor,
            n_filters=args.N_FILTERS,
            downsample=args.DOWNSAMPLE,
            layer_multiplier=args.LAYER_MULTIPLIER,
        )
    elif arch_name == 'unet':
        model = UNet(
            3,
            residual=True,
            scale_factor=dataset_upscale_factor,
            n_filters=args.N_FILTERS,
        )
    elif arch_name == 'srgan':
        model = SRResNet()
    elif arch_name == 'espcn':
        model = SimpleResNet(n_filters=64, n_blocks=6)
    else:
        raise Exception(
            "Unknown architecture. Select one between:", args.archs
        )

    model_path = args.MODEL_NAME
    model.load_state_dict(torch.load(model_path))

    model = model.cuda()
    model.reparametrize()

    path = args.CLIPNAME
    cap = cv2.VideoCapture(path)
    reader = torchvision.io.VideoReader(path, "video")

    if enable_write_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        hr_video_writer = cv2.VideoWriter(
            'rendered.mp4', fourcc, 30, (1920, 1080)
        )

    metadata = reader.get_metadata()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_fix, width_fix, padH, padW = get_padded_dim(height, width)

    frame_queue = Queue(1)
    out_queue = Queue(1)

    reader.seek(0)

    def read_pic(cap, q):
        count = 0
        start = time.time()
        while True:
            cv2_im = next(cap)['data']  # .cuda().float()
            cv2_im = cv2_im.cuda().float()

            x = dl.normalize_img(cv2_im / 255.0).unsqueeze(0)

            x_bicubic = torch.clip(
                F.interpolate(
                    x,
                    scale_factor=args.UPSCALE_FACTOR * args.DOWNSAMPLE,
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
            cv2_out = torch_to_cv2(out, rescale_factor=scale)
            cv2.imshow('rendering', cv2_out)
            cv2.waitKey(1)

    t1 = Thread(target=read_pic, args=(reader, frame_queue)).start()
    t2 = Thread(target=show_pic, args=(cap, out_queue)).start()
    target_fps = cap.get(cv2.CAP_PROP_FPS)
    target_frametime = 1000 / target_fps

    model = model.eval()
    with torch.no_grad():
        tqdm_ = tqdm(range(frame_count))
        for i in tqdm_:
            t0 = time.time()

            x, x_bicubic = frame_queue.get()
            out = model(x)[:, :, : int(height) * 2, : int(width) * 2]

            out_true = i // (target_fps * 3) % 2 == 0

            if not args.SHOW_ONLY_HQ:
                out = blend_images(x_bicubic, out)
            out_queue.put(out)
            frametime = time.time() - t0
            if frametime < target_frametime * 1e-3:
                time.sleep(target_frametime * 1e-3 - frametime)

            if enable_write_to_video:
                write_to_video(out, hr_video_writer)
                if i == 30 * 10:
                    hr_video_writer.release()
                    print("Releasing video")

            tqdm_.set_description(
                "frame time: {}; fps: {}; {}".format(
                    frametime * 1e3, 1000 / frametime, out_true
                )
            )
### binarization/vaccaro/render.py - END


def inv_adjust_image_for_unet(
    generated: torch.Tensor, original: torch.Tensor
) -> torch.Tensor:
    height_generated, width_generated = (
        generated.shape[-2],
        generated.shape[-1],
    )
    height_original, width_original = original.shape[-2], original.shape[-1]
    height_offset = (height_generated - height_original) // 2
    width_offset = (width_generated - width_original) // 2
    return F.crop(
        generated, height_offset, width_offset, height_original, width_original
    )


def process_raw_generated(
    generated: torch.Tensor, original: torch.Tensor
) -> torch.Tensor:
    """Postprocesses outputs from super-resolution generator models"""
    out = generated
    out = dataset.inv_min_max_scaler(out)
    out = out.clip(0, 255)
    out = out / 255.0
    out = inv_adjust_image_for_unet(out, original)
    return out


def main(cfg: Gifnoc):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = train.set_up_generator(cfg)
    gen.to(device)
    video_fp = Path('hola.mp4')
    cap = cv2.VideoCapture(video_fp)
    reader = torchvision.io.VideoReader(video_fp, 'video')


def main(cfg: Gifnoc):
    save_dir = cfg.paths.outputs_dir / cfg.model.ckpt_path_to_resume.stem
    save_dir.mkdir(exist_ok=True, parents=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whole_images_dataset = dataset.WholeImagesDataset(
        original_frames_dir=cfg.paths.val_original_frames_dir,
        compressed_frames_dir=cfg.paths.val_compressed_frames_dir,
    )
    dl_val = dataset.DataLoader(
        dataset=whole_images_dataset,
        batch_size=cfg.params.batch_size,
        shuffle=None,
    )
    progress_bar_val = tqdm(dl_val)
    gen = train.set_up_generator(cfg)
    gen.to(device)
    counter = 0
    for step_id_val, (compressed_val, original_val) in enumerate(
        progress_bar_val
    ):

        compressed_val = compressed_val.to(device)
        original_val = original_val.to(device)

        compressed_val = dataset.adjust_image_for_unet(compressed_val)

        gen.eval()
        with torch.no_grad():
            generated_val = gen(compressed_val)

        original_val = original_val.cpu()
        generated_val = generated_val.cpu()
        compressed_val = compressed_val.cpu()
        generated_val = process_raw_generated(generated_val, original_val)

        for i in range(original_val.shape[0]):
            fig = dataset.draw_validation_fig(
                original_image=original_val[i],
                compressed_image=compressed_val[i],
                generated_image=generated_val[i],
            )
            save_path = save_dir / f'validation_fig_{counter}.jpg'
            counter += 1
            fig.savefig(save_path)
            plt.close(fig)  # close the current fig to prevent OOM issues


if __name__ == "__main__":
    cfg = get_default_config()
    # default_config.params.ckpt_path_to_resume = Path('/home/loopai/Projects/binarization/artifacts/best_checkpoints/2022_08_28_epoch_9.pth')
    cfg.model.ckpt_path_to_resume = Path(
        '/home/loopai/Projects/binarization/artifacts/best_checkpoints/2022_08_31_epoch_13.pth'
    )
    cfg.params.batch_size = 10
    main(cfg)
