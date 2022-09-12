"""Module to train a super-resolution model"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import lpips
import piq
import torch
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


def set_up_unet(cfg: Gifnoc) -> models.UNet:
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


def get_starting_epoch_id(ckpt_path_to_resume: Path) -> int:
    """Extract the epoch id from the ckpt resuming path, if possible"""
    try:
        return int(Path(ckpt_path_to_resume).stem.split('_')[-1]) + 1
    except (ValueError, TypeError):
        return 0


def main(cfg: Gifnoc):
    """Main for training a model for super-resolution"""
    checkpoints_dir, runs_dir = set_up_artifacts_dirs(cfg.paths.artifacts_dir)
    tensorboard_logger = SummaryWriter(log_dir=runs_dir)
    gen = set_up_unet(cfg=cfg)
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

    starting_epoch_id = get_starting_epoch_id(cfg.params.ckpt_path_to_resume)
    for epoch_id in range(starting_epoch_id, cfg.params.num_epochs):
        progress_bar_train = tqdm(dl_train, total=cfg.params.limit_train_batches)
        for step_id_train, (compressed, original) in enumerate(progress_bar_train):
            # training_step - START
            ##################################################################
            if (cfg.params.limit_train_batches is not None and step_id_train > cfg.params.limit_train_batches):
                break
            gen.train()
            dis.train()

            # Discriminator training step - START
            ##################################################################
            dis_optim.zero_grad()

            compressed = compressed.to(device)
            original = original.to(device)
            generated = gen(compressed)  # maybe clip it in [0, 1]
            pred_original = dis(original)

            loss_true = bce_loss_op(pred_original, torch.ones_like(pred_original))
            pred_generated = dis(generated.detach())
            loss_fake = bce_loss_op(pred_generated, torch.zeros_like(pred_generated))

            loss_dis = (loss_true + loss_fake) * 0.5

            loss_dis.backward()
            dis_optim.step()
            ##################################################################
            # Discriminator training step - END

            # Generator training step - START
            ##################################################################
            gen_optim.zero_grad()

            loss_lpips = lpips_vgg_loss_op(generated, original).mean()
            loss_ssim = 1.0 - ssim_op(generated, original)
            pred_generated = dis(generated)
            loss_bce = bce_loss_op(pred_generated, torch.ones_like(pred_generated))
            loss_gen = (
                cfg.params.w0 * loss_lpips
                + cfg.params.w1 * loss_ssim
                + cfg.params.w2 * loss_bce
            )

            loss_gen.backward()
            gen_optim.step()
            ##################################################################
            # Generator training step - END

            progress_bar_train.set_description(
                f'Epoch #{epoch_id} - '
                f'loss_dis: {loss_dis.item():.8f} - '
                f'loss_gen: {loss_gen.item():.4f}'
            )
            tensorboard_logger.add_scalar('loss_dis', scalar_value=loss_dis, global_step=global_step_id)
            tensorboard_logger.add_scalar('loss_gen', scalar_value=loss_gen, global_step=global_step_id)
            tensorboard_logger.add_scalar('loss_lpips', scalar_value=loss_lpips, global_step=global_step_id)
            tensorboard_logger.add_scalar('loss_ssim', scalar_value=loss_ssim, global_step=global_step_id)
            tensorboard_logger.add_scalar('loss_bce', scalar_value=loss_bce, global_step=global_step_id)

            global_step_id += 1
            ##################################################################
            # training_step - END

        progress_bar_val = tqdm(dl_val, total=cfg.params.limit_val_batches)
        for step_id_val, (compressed_val, original_val) in enumerate(progress_bar_val):
            # validation_step - START
            ##################################################################
            if (cfg.params.limit_val_batches is not None and step_id_val > cfg.params.limit_val_batches):
                break
            compressed_val = compressed_val.to(device)
            original_val = original_val.to(device)
            metrics: Dict[str, float] = {}
            gen.eval()
            with torch.no_grad():
                generated_val = gen(compressed_val).clip(0, 1)
                metrics['lpips_alex'] = lpips_alex_metric_op(generated_val, original_val).mean().item()
                metrics['ssim'] = ssim_op(generated_val, original_val).item()
                metrics['psnr'] = piq.psnr(generated_val, original_val).item()
                metrics['ms_ssim'] = piq.multi_scale_ssim(generated_val, original_val).item()
                metrics['brisque'] = piq.brisque(generated_val).item()
            for metric_name, metric_value in metrics.items():
                tensorboard_logger.add_scalar(f'{metric_name}_val', metric_value, global_step=epoch_id * progress_bar_val.total + step_id_val)
            ##################################################################
            # validation_epoch_end - END

        # training_epoch_end - START
        ##################################################################
        current_ckpt_path = Path(checkpoints_dir, f"epoch_{epoch_id}.pth")
        torch.save(gen.state_dict(), current_ckpt_path)
        ##################################################################
        # training_epoch_end - END


if __name__ == "__main__":
    default_config.params.limit_train_batches = None
    default_config.params.limit_val_batches = None
    default_config.params.ckpt_path_to_resume = Path('/home/loopai/Projects/binarization/artifacts/best_checkpoints/2022_08_31_epoch_13.pth')

    main(default_config)
