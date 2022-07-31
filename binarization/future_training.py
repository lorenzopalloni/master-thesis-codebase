from pathlib import Path
from datetime import datetime

import torch
import piq
import lpips
import tqdm
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
from config import MainConfig

from binarization import models
from binarization import dataset

cs = hydra.core.config_store.ConfigStore()
cs.store(name="config_schema", node=MainConfig)

@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg: DictConfig):

    str_now = datetime.now().strftime(r"%Y%m%d%H%M%S")
    artifacts_dir = Path(cfg.paths.artifacts_dir, str_now)
    checkpoints_dir = artifacts_dir / 'checkpoints'
    runs_dir = artifacts_dir / 'runs'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=runs_dir)

    gen = models.UNet(
        num_filters=cfg.params.unet.num_filters,
        use_residual=cfg.params.unet.use_residual,
        use_batch_norm=cfg.params.unet.use_batch_norm,
        scale_factor=cfg.params.unet.scale_factor,
    )

    dis = models.Discriminator()

    dis_optim = torch.optim.Adam(lr=cfg.params.dis_lr, params=dis.parameters())
    gen_optim = torch.optim.Adam(lr=cfg.params.gen_lr, params=gen.parameters())

    ssim_loss_op = piq.SSIMLoss()
    lpips_vgg_loss_op = lpips.LPIPS(net='vgg', version='0.1')
    lpips_alex_loss_op = lpips.LPIPS(net='alex', version='0.1')
    bce_loss_op = torch.nn.BCELoss()

    # move models and losses to cuda device if any is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen.to(device)
    dis.to(device)
    lpips_vgg_loss_op.to(device)
    lpips_alex_loss_op.to(device)
    print(f'current device: {device}')

    dl_train = dataset.make_train_dataloader(cfg)
    # dl_val = dataset.make_val_dataloader(cfg)
    # dl_test = dataset.make_test_dataloader(cfg)

    global_step_id = 0
    for epoch_id in range(cfg.params.num_epochs):
        progress_bar = tqdm.tqdm(dl_train)
        for step_id, (lq, hq) in enumerate(progress_bar):
            gen.train()
            dis.train()

            # discriminator training step
            ##################################################################
            dis_optim.zero_grad()

            lq = lq.to(device)
            hq = hq.to(device)
            generated_hq = gen(lq)

            pred_hq = dis(hq)

            loss_true = bce_loss_op(pred_hq, torch.ones_like(pred_hq))

            pred_generated_hq = dis(generated_hq.detach())
            loss_fake = bce_loss_op(
                pred_generated_hq, torch.zeros_like(pred_generated_hq)
            )

            loss_dis = loss_true + loss_fake
            loss_dis *= 0.5

            loss_dis.backward()
            dis_optim.step()
            ##################################################################

            # generator training step
            ##################################################################
            gen_optim.zero_grad()

            loss_lpips = lpips_vgg_loss_op(generated_hq, hq).mean()

            x_min = min(generated_hq.min(), hq.min())
            x_max = max(generated_hq.max(), hq.max())
            loss_ssim = 1.0 - ssim_loss_op(
                dataset.min_max_scaler(generated_hq, x_min, x_max),
                dataset.min_max_scaler(hq, x_min, x_max),
            )

            pred_generated_hq = dis(generated_hq)
            loss_bce = bce_loss_op(
                pred_generated_hq, torch.ones_like(pred_generated_hq)
            )
            loss_gen = (
                cfg.params.w0 * loss_lpips
                + cfg.params.w1 * loss_ssim
                + cfg.params.w2 * loss_bce
            )

            loss_gen.backward()
            gen_optim.step()
            ##################################################################

            # logging statistics on training set
            ##################################################################
            progress_bar.set_description(
                'Epoch #{} - '
                'Loss dis: {:.8f}; '
                'Loss gen: {:.4f} = '
                '({:.4f} + {:.4f} * {:.4f})'.format(
                    epoch_id,
                    float(loss_dis),
                    float(loss_gen),
                    float(
                        cfg.params.w0 * loss_lpips + cfg.params.w1 * loss_ssim
                    ),
                    float(cfg.params.w2),
                    float(loss_bce),
                )
            )
            ##################################################################
            writer.add_scalar(
                'lossD', scalar_value=loss_dis, global_step=global_step_id
            )
            writer.add_scalar(
                'lossG', scalar_value=loss_gen, global_step=global_step_id
            )
            # writer.add_image('output_example', img_tensor=<insert-image-here>, global_step=epoch_id * step_id)
            global_step_id += 1

        torch.save(
            gen.state_dict(),
            Path(
                checkpoints_dir,
                "epoch-{}-lossD-{:.2f}-lossG-{:.2f}.pth".format(
                    epoch_id, loss_dis, loss_gen
                ),
            ),
        )


if __name__ == "__main__":
    main()
