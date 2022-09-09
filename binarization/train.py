"""Module to train a super-resolution model"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Union

import lpips
import mlflow
import piq
import torch
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
    """Main for training a super-resolution model"""

    mlflow.log_params(cfg.params.stringify())
    checkpoints_dir, runs_dir = set_up_artifacts_dirs(cfg.paths.artifacts_dir)
    # mlflow.set_tracking_uri(runs_dir)

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

            metrics_train: Dict[str, Union[int, float]] = {}
            metrics_train['epoch_train'] = epoch_id
            metrics_train['loss_dis_train'] = loss_dis.item()
            metrics_train['loss_gen_train'] = loss_gen.item()
            metrics_train['loss_lpips_train'] = loss_lpips.item()
            metrics_train['loss_ssim_train'] = loss_ssim.item()
            metrics_train['loss_bce_train'] = loss_bce.item()
            mlflow.log_metrics(metrics_train, step=global_step_id)

            progress_bar_train.set_description(
                f'Epoch #{epoch_id} - '
                f'loss_dis: {metrics_train["loss_dis_train"]:.8f} - '
                f'loss_gen: {metrics_train["loss_gen_train"]:.4f}'
            )

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
            gen.eval()
            with torch.no_grad():
                generated_val = gen(compressed_val).clip(0, 1)
                metrics_val: Dict[str, Union[int, float]] = {}
                metrics_val['lpips_alex_val'] = lpips_alex_metric_op(generated_val, original_val).mean().item()
                metrics_val['ssim_val'] = ssim_op(generated_val, original_val).item()
                metrics_val['psnr_val'] = piq.psnr(generated_val, original_val).item()
                metrics_val['ms_ssim_val'] = piq.multi_scale_ssim(generated_val, original_val).item()
                metrics_val['brisque_val'] = piq.brisque(generated_val).item()
            mlflow.log_metrics(metrics_val, step=epoch_id * progress_bar_val.total + step_id_val)
            ##################################################################
            # validation_epoch_end - END

        # training_epoch_end - START
        ##################################################################
        current_ckpt_path = Path(checkpoints_dir, f"epoch_{epoch_id}.pth")
        torch.save(gen.state_dict(), current_ckpt_path)
        ##################################################################
        # training_epoch_end - END


if __name__ == "__main__":
    default_config.params.limit_train_batches = 2
    default_config.params.limit_val_batches = 2
    default_config.params.num_epochs = 16
    default_config.params.ckpt_path_to_resume = Path('/home/loopai/Projects/binarization/artifacts/best_checkpoints/2022_08_31_epoch_13.pth')
    main(default_config)
