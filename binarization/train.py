"""Training module for super-resolution models"""

from __future__ import annotations

from pathlib import Path

import lpips
import mlflow
import piq
import torch
from tqdm import tqdm

from binarization.config import Gifnoc, get_default_config
from binarization.dataset import get_train_batches, get_val_batches
from binarization.models import Discriminator
from binarization.traintools import (
    set_up_checkpoints_dir,
    set_up_cuda_device,
    set_up_generator,
)
from binarization.vaccaro import pytorch_ssim


def run_training(cfg: Gifnoc):
    """Launches a super-resolution model training."""

    checkpoints_dir = set_up_checkpoints_dir(cfg.paths.artifacts_dir)

    device = set_up_cuda_device(device_id=1, verbose=True)

    gen = set_up_generator(cfg=cfg, device=device)
    dis = Discriminator()
    dis.to(device)

    gen_optim = torch.optim.Adam(lr=cfg.params.gen_lr, params=gen.parameters())
    dis_optim = torch.optim.Adam(lr=cfg.params.dis_lr, params=dis.parameters())

    ssim_op = pytorch_ssim.SSIM()  # piq.SSIMLoss()
    lpips_vgg_loss_op = lpips.LPIPS(net='vgg', version='0.1')
    lpips_alex_metric_op = lpips.LPIPS(net='alex', version='0.1')
    bce_loss_op = torch.nn.BCELoss()

    lpips_vgg_loss_op.to(device)
    lpips_alex_metric_op.to(device)

    global_step_id = 0
    global_step_id_val = 0

    for epoch_id in range(cfg.params.num_epochs):
        train_batches = get_train_batches(cfg)
        progress_bar_train = tqdm(
            train_batches, total=cfg.params.limit_train_batches
        )
        for step_id_train, (original, compressed) in enumerate(
            progress_bar_train
        ):
            # training_step - START
            ##################################################################
            if (
                cfg.params.limit_train_batches is not None
                and step_id_train > cfg.params.limit_train_batches
            ):
                break
            gen.train()
            dis.train()

            # Discriminator training step - START
            ##################################################################
            dis_optim.zero_grad()

            original = original.to(device)
            compressed = compressed.to(device)
            generated = gen(compressed)  # maybe clip it in [0, 1]
            pred_original = dis(original)

            loss_true = bce_loss_op(
                pred_original, torch.ones_like(pred_original)
            )
            pred_generated = dis(generated.detach())
            loss_fake = bce_loss_op(
                pred_generated, torch.zeros_like(pred_generated)
            )

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
            loss_bce = bce_loss_op(
                pred_generated, torch.ones_like(pred_generated)
            )
            loss_gen = (
                cfg.params.w0 * loss_lpips
                + cfg.params.w1 * loss_ssim
                + cfg.params.w2 * loss_bce
            )

            loss_gen.backward()
            gen_optim.step()

            if (global_step_id + 1) % cfg.params.save_ckpt_every == 0:
                current_ckpt_path = Path(
                    checkpoints_dir,
                    f"{cfg.model.name}_{epoch_id}_{global_step_id}.pth",
                )
                torch.save(gen.state_dict(), current_ckpt_path)
            ##################################################################
            # Generator training step - END

            metrics_train: dict[str, int | float] = {}
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

        val_batches = get_val_batches(cfg)
        progress_bar_val = tqdm(
            val_batches, total=cfg.params.limit_val_batches
        )
        for step_id_val, (original_val, compressed_val) in enumerate(
            progress_bar_val
        ):
            # validation_step - START
            ##################################################################
            if (
                cfg.params.limit_val_batches is not None
                and step_id_val > cfg.params.limit_val_batches
            ):
                break
            original_val = original_val.to(device)
            compressed_val = compressed_val.to(device)
            gen.eval()
            with torch.no_grad():
                generated_val = gen(compressed_val).clip(0, 1)
                metrics_val: dict[str, int | float] = {}
                metrics_val['lpips_alex_val'] = (
                    lpips_alex_metric_op(generated_val, original_val)
                    .mean()
                    .item()
                )
                metrics_val['ssim_val'] = ssim_op(
                    generated_val, original_val
                ).item()
                metrics_val['psnr_val'] = piq.psnr(
                    generated_val, original_val
                ).item()
                metrics_val['ms_ssim_val'] = piq.multi_scale_ssim(
                    generated_val, original_val
                ).item()
                metrics_val['brisque_val'] = piq.brisque(generated_val).item()
            global_step_id_val += 1
            mlflow.log_metrics(metrics_val, step=global_step_id_val)
            ##################################################################
            # validation_epoch_end - END

        # training_epoch_end - START
        ##################################################################
        current_ckpt_path = Path(
            checkpoints_dir,
            f"{cfg.model.name}_{epoch_id}_{global_step_id}.pth",
        )
        torch.save(gen.state_dict(), current_ckpt_path)
        ##################################################################
        # training_epoch_end - END
    mlflow.log_artifacts(checkpoints_dir)

def run_experiment(
    cfg: Gifnoc,
    model_name: str,
    experiment_name: str,
    ckpt_path_to_resume: Path | None = None,
) -> None:
    """Launches an mlflow experiment.

    Args:
        cfg (Gifnoc): a valid configuration object.
        model_name (str): choose in ('unet', 'srunet').
        experiment_name (str): name of the mlflow experiment.
        ckpt_path_to_resume (Path | None, optional): path to model weights.
            Defaults to None.
    """
    cfg.model.name = model_name
    cfg.model.ckpt_path_to_resume = ckpt_path_to_resume

    mlflow.set_tracking_uri(cfg.paths.mlruns_dir.as_uri())
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        description=f"Running {experiment_name}",
    ):
        mlflow.log_params(cfg.params.stringify())
        mlflow.log_params(cfg.model.stringify())
        run_training(cfg)

if __name__ == "__main__":
    default_cfg = get_default_config()
    # default_cfg.params.limit_train_batches = 2
    # default_cfg.params.limit_val_batches = 2
    # default_cfg.params.num_epochs = 2

    unet_ckpt = Path(
        default_cfg.paths.artifacts_dir,
        '/checkpoints/2022_09_30_06_31_40/unet_34_106400.pth'
    )
    # srunet_ckpt = Path(cfg.paths.artifacts_dir, '/checkpoints/')

    # run_experiment(
    #     cfg=default_cfg,
    #     model_name='unet',
    #     experiment_name='UNet experiment',
    #     ckpt_path_to_resume=unet_ckpt,
    # )

    run_experiment(
        cfg=default_cfg,
        model_name='srunet',
        experiment_name='SRUNet experiment',
        # ckpt_path_to_resume=srunet_ckpt,
    )

