"""Training module"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import lpips
import piq
import torch
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from binarization import dataset, models
from binarization.config import Gifnoc, default_config
from binarization.vaccaro import pytorch_ssim


def set_up_artifacts_dirs(artifacts_dir: Path) -> Tuple[Path, Path]:
    """Sets up unique-time-related dirs for model checkpoints and runs"""
    str_now = datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")

    checkpoints_dir = Path(artifacts_dir, 'checkpoints', str_now)
    runs_dir = Path(artifacts_dir, 'runs', str_now)

    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return checkpoints_dir, runs_dir


def get_unet(cfg: Gifnoc) -> models.UNet:
    """Instantiates a UNet, resuming model weights if provided"""
    generator = models.UNet(
        num_filters=cfg.params.unet.num_filters,
        use_residual=cfg.params.unet.use_residual,
        use_batch_norm=cfg.params.unet.use_batch_norm,
        scale_factor=cfg.params.unet.scale_factor
    )
    if cfg.params.ckpt_path_to_resume:
        print(f'>>> resume from {cfg.params.ckpt_path_to_resume}')
        generator.load_state_dict(torch.load(cfg.params.ckpt_path_to_resume.as_posix()))
    return generator


def process_raw_generated(raw_generated, original):
    """Postprocesses outputs from super-resolution generator models"""
    out = raw_generated.cpu()
    out = out.cpu()
    out = dataset.inv_min_max_scaler(out)
    out = out.clip(0, 255)
    out = out.squeeze()
    out = out / 255.0
    out = F.crop(out, 0, 0, original.shape[-2], original.shape[-1])
    return out


def main(cfg: Gifnoc):
    """Main for training a model for super-resolution"""
    checkpoints_dir, runs_dir = set_up_artifacts_dirs(cfg.paths.artifacts_dir)
    tensorboard_logger = SummaryWriter(log_dir=runs_dir)
    gen = get_unet(cfg=cfg)
    dis = models.Discriminator()

    gen_optim = torch.optim.Adam(lr=cfg.params.gen_lr, params=gen.parameters())
    dis_optim = torch.optim.Adam(lr=cfg.params.dis_lr, params=dis.parameters())

    ssim_op = pytorch_ssim.SSIM()  # piq.SSIMLoss()
    lpips_vgg_loss_op = lpips.LPIPS(net='vgg', version='0.1')
    lpips_alex_metric_op = lpips.LPIPS(net='alex', version='0.1')
    bce_loss_op = torch.nn.BCELoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen.to(device)
    dis.to(device)
    lpips_vgg_loss_op.to(device)
    lpips_alex_metric_op.to(device)
    print(f'INFO: current device: {device}')

    dl_train, dl_val, dl_test = dataset.make_dataloaders(cfg=cfg)

    global_step_id = 0
    for epoch_id in range(cfg.params.num_epochs):
        progress_bar_train = tqdm(dl_train, total=cfg.params.limit_train_batches)
        for step_id_train, (compressed_patches, original_patches) in enumerate(progress_bar_train):
            # training_step - START
            ##################################################################
            if (cfg.params.limit_train_batches is not None and step_id_train > cfg.params.limit_train_batches):
                break
            gen.train()
            dis.train()

            # Discriminator training step - START
            ##################################################################
            dis_optim.zero_grad()

            compressed_patches = compressed_patches.to(device)
            original_patches = original_patches.to(device)
            generated_patches = gen(compressed_patches)
            pred_original_patches = dis(original_patches)

            loss_true = bce_loss_op(pred_original_patches, torch.ones_like(pred_original_patches))
            pred_generated_patches = dis(generated_patches.detach())
            loss_fake = bce_loss_op(pred_generated_patches, torch.zeros_like(pred_generated_patches))

            loss_dis = (loss_true + loss_fake) * 0.5

            loss_dis.backward()
            dis_optim.step()
            ##################################################################
            # Discriminator training step - END

            # Generator training step - START
            ##################################################################
            gen_optim.zero_grad()

            loss_lpips = lpips_vgg_loss_op(generated_patches, original_patches).mean()

            # x_min = min(
            #     generated_patches.min(), original_patches.min()
            # )
            # x_max = max(
            #     generated_patches.max(), original_patches.max()
            # )
            # loss_ssim = 1.0 - ssim_op(
            #     dataset.min_max_scaler(
            #         generated_patches, x_min, x_max
            #     ),
            #     dataset.min_max_scaler(original_patches, x_min, x_max),
            # )

            loss_ssim = 1.0 - ssim_op(generated_patches, original_patches)

            pred_generated_patches = dis(generated_patches)
            loss_bce = bce_loss_op(pred_generated_patches, torch.ones_like(pred_generated_patches))
            loss_gen = (
                cfg.params.w0 * loss_lpips
                + cfg.params.w1 * loss_ssim
                + cfg.params.w2 * loss_bce
            )

            loss_gen.backward()
            gen_optim.step()
            ##################################################################
            # Generator training step - END

            # Log statistics on training set - START
            ##################################################################
            progress_bar_train.set_description(
                f'Epoch #{epoch_id} - '
                f'Loss dis: {float(loss_dis):.8f}; '
                f'Loss gen: {float(loss_gen):.4f} = '
                f'w0 * {float(loss_lpips):.4f}'
                f' + w1 * {float(loss_ssim):.4f}'
                f' + w2 * {float(loss_bce):.4f})'
            )
            tensorboard_logger.add_scalar('lossD', scalar_value=loss_dis, global_step=global_step_id)
            tensorboard_logger.add_scalar('lossG', scalar_value=loss_gen, global_step=global_step_id)
            # tensorboard_logger.add_image('output_example', img_tensor=<insert-image-here>, global_step=epoch_id * step_id)
            ##################################################################
            # Log statistics on training set - END

            global_step_id += 1
            ##################################################################
            # training_step - END

        progress_bar_val = tqdm(dl_val, total=cfg.params.limit_val_batches)
        for step_id_val, (compressed_patches_val, original_patches_val) in enumerate(progress_bar_val):
            # validation_step - START
            ##################################################################
            if (cfg.params.limit_val_batches is not None and step_id_val > cfg.params.limit_val_batches):
                break

            metrics: Dict[str, List[float]] = defaultdict(list)

            compressed_patches_val.to(device)
            original_patches_val.to(device)

            gen.eval()
            with torch.no_grad():
                generated_patches_val = gen(compressed_patches_val)
                metrics['lpips_alex'] = lpips_alex_metric_op(generated_patches_val, original_patches_val)
                metrics['ssim'] = ssim_op(generated_patches_val, original_patches_val)

            for metric in metrics:
                tensorboard_logger.add_scalar(f'{metric}_val', sum(metrics[metric]) / len(metrics[metric]), global_step=global_step_id)
            ##################################################################
            # validation_epoch_end - END

        # training_epoch_end - START
        ##################################################################
        current_ckpt_path = Path(checkpoints_dir, f"epoch_{epoch_id}.pth")
        torch.save(gen.state_dict(), current_ckpt_path)
        ##################################################################
        # training_epoch_end - END


if __name__ == "__main__":
    default_config.params.limit_train_batches = 4
    default_config.params.limit_val_batches = 4
    default_config.params.ckpt_path_to_resume = Path('/home/loopai/Projects/binarization/artifacts/best_checkpoints/2022_08_27_epoch_7.pth')

    main(default_config)
